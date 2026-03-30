"""
Copyright 2025 Zhuoming Chen
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import logging
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
from contextlib import nullcontext
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.utils import (
    debug_timing,
    is_cuda
)

import vortex_torch
from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.cache.triton_kernels.set_kv import set_kv_buffer_fp8_launcher
logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()

"""
Sparse Attention Memory pool.

In addition to Memory Pool in the original SGLang
We 
1) maintain auxilary cache tensor objects for every page.
2) internally treat each KV head as a request (as they may have different sparse patterns), 
then we interpret external auguments to the physical address
"""

class VTXGraphCachePool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        sparse_attention: vortex_torch.flow.vFlow,
        model_runner,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim

        # for disagg with nvlink
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

        self.num_pages = ((self.size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1
        self.is_int8 = (self.dtype == torch.int8)
        self.is_fp8 = (self.dtype in (torch.float8_e4m3fn, torch.float8_e5m2))
        # FP8 type encoding for Triton kernels: 0=none, 1=e4m3, 2=e5m2
        if self.dtype == torch.float8_e4m3fn:
            self.fp8_type = 1
        elif self.dtype == torch.float8_e5m2:
            self.fp8_type = 2
        else:
            self.fp8_type = 0

        self.sparse_attention = sparse_attention
        self.ctx = vortex_torch.cache.Context()

        self._create_buffers()
        self._initialize_graph(model_runner)
        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None

        cache_size = self.get_cache_size_bytes()
        
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, Cache size: {cache_size / GB:.2f} GB"
        )
        
        self.mem_usage = cache_size / GB
    
    def _initialize_graph(self, model_runner) -> None:
        # Match cache dummy dtypes to actual storage:
        # - int8 path: forward_cache sees int8 K directly; reduce kernels dequant inline (QUANT_TYPE==1).
        # - fp8 path: forward_cache sees uint8 K/V directly.
        # - bf16 path: everything bf16.
        # Custom caches (centroids, max, min) are always bf16.
        if self.is_fp8:
            kv_store_dtype = torch.uint8
        elif self.is_int8:
            kv_store_dtype = torch.int8
        else:
            kv_store_dtype = torch.bfloat16

        self.ctx.create(self, model_runner)
        self.ctx.profile()

        # Set quant context for inline dequantization during profiling
        if self.is_int8:
            self.ctx.quant_type = 1
            self.ctx.kv_scale_ptr = torch.empty((0,), dtype=torch.float16, device=self.device)

        try:
            with torch.no_grad():
                loc_dummy = torch.empty((0,), dtype=torch.int64, device=self.device)
                cache_dummy = {
                        cache_name:  as_vtensor(torch.zeros(
                                (0, cache_shape[0], cache_shape[1]),
                                dtype=kv_store_dtype if cache_name in ("k", "v") else torch.bfloat16,
                                device=self.device,
                            ), FORMAT.PAGED)

                        for (cache_name, cache_shape) in self.cache_meta_info.items()
                }
                self.sparse_attention.forward_cache(cache=cache_dummy, loc=loc_dummy, ctx=self.ctx)
        except Exception:
            raise

        self.ctx.summary()
        self.ctx.execute()


    def _create_buffers(self):

        self.cache_meta_info = self.sparse_attention.get_cache_meta_info(self.page_size, self.head_dim)
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                if self.is_int8:
                    # Int8 path: k/v as int8, custom caches (centroids etc.) as bf16,
                    # plus per-token float32 scale buffers for k and v.
                    # Also maintain a bf16 shadow "k" buffer for forward_cache ops
                    # (centroid/envelope computation needs bf16 K).
                    self.cache = []
                    for _ in range(self.layer_num):
                        layer_cache = {}
                        for cache_name, cache_shape in self.cache_meta_info.items():
                            if cache_name in ("k", "v"):
                                layer_cache[cache_name] = torch.zeros(
                                    (self.num_pages, cache_shape[0], cache_shape[1]),
                                    dtype=torch.int8,
                                    device=self.device,
                                )
                            else:
                                # Custom caches (centroids, max, min, etc.) stay bf16
                                layer_cache[cache_name] = torch.zeros(
                                    (self.num_pages, cache_shape[0], cache_shape[1]),
                                    dtype=torch.bfloat16,
                                    device=self.device,
                                )
                        # Per-token scale buffers: shape [num_pages, page_size, 1]
                        # Use float16 to halve memory and bandwidth; precision is
                        # sufficient for absmax scales (values are small positive floats).
                        layer_cache["k_scale"] = torch.zeros(
                            (self.num_pages, self.page_size, 1),
                            dtype=torch.float16,
                            device=self.device,
                        )
                        layer_cache["v_scale"] = torch.zeros(
                            (self.num_pages, self.page_size, 1),
                            dtype=torch.float16,
                            device=self.device,
                        )
                        self.cache.append(layer_cache)
                    # Pre-allocated prefill workspaces: reused each prefill call
                    # to avoid dynamic allocation of large bf16 buffers.
                    # Sized to num_pages (upper bound for any single prefill).
                    self.prefill_k_workspace = torch.empty(
                        (self.num_pages, self.page_size, self.head_dim),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    self.prefill_v_workspace = torch.empty(
                        (self.num_pages, self.page_size, self.head_dim),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                elif self.is_fp8:
                    # FP8 path: k/v stored as uint8 (bitcast of fp8, 1 byte),
                    # custom caches (centroids, max, min) as bf16.
                    # No scale buffers, no shadow buffer, no prefill workspaces —
                    # FlashInfer handles fp8 natively (we view-cast uint8→fp8 at attention time).
                    self.cache = []
                    for _ in range(self.layer_num):
                        layer_cache = {}
                        for cache_name, cache_shape in self.cache_meta_info.items():
                            if cache_name in ("k", "v"):
                                layer_cache[cache_name] = torch.zeros(
                                    (self.num_pages, cache_shape[0], cache_shape[1]),
                                    dtype=torch.uint8,
                                    device=self.device,
                                )
                            else:
                                layer_cache[cache_name] = torch.zeros(
                                    (self.num_pages, cache_shape[0], cache_shape[1]),
                                    dtype=torch.bfloat16,
                                    device=self.device,
                                )
                        self.cache.append(layer_cache)
                else:
                    self.cache = [
                        {
                            cache_name:  torch.zeros(
                                    (self.num_pages, cache_shape[0], cache_shape[1]),
                                    dtype=self.store_dtype,
                                    device=self.device,
                                )

                            for (cache_name, cache_shape) in self.cache_meta_info.items()
                        }

                        for _ in range(self.layer_num)
                    ]
        
    def _clear_buffers(self):
        del self.cache
       

    def get_cache_size_bytes(self) -> int:
        """
        Return total bytes occupied by all tensors in `self.cache`.
        Works even if some entries are not tensors.
        """
        total_bytes = 0

        for layer_cache in self.cache:
            if not isinstance(layer_cache, dict):
                # Be tolerant to unexpected structures
                continue

            for t in layer_cache.values():
                if not torch.is_tensor(t):
                    continue

                # Prefer accurate allocated size if available (includes padding/strides)
                try:
                    total_bytes += int(t.untyped_storage().nbytes())
                except AttributeError:
                    # Fallback: logical size in bytes
                    total_bytes += int(t.element_size() * t.numel())

        return total_bytes
    
    def get_kv_size_bytes(self):
        
        raise NotImplementedError
    
    # for disagg
    def get_contiguous_buf_infos(self):
        
        raise NotImplementedError

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_cpu_copy(self, indices):
        
        raise NotImplementedError

    def load_cpu_copy(self, kv_cache_cpu, indices):
        
        raise NotImplementedError

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        raise NotImplementedError


    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
       raise NotImplementedError

    def transfer_per_layer(self, indices, flat_data, layer_id):
        
        raise NotImplementedError


    def get_key_buffer(self, layer_id: int):
        
        return self.cache[layer_id - self.start_layer]["k"]

    def get_value_buffer(self, layer_id: int):
        
        return self.cache[layer_id - self.start_layer]["v"]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        return self.cache[layer_id - self.start_layer]["k"], self.cache[layer_id - self.start_layer]["v"]

        
    def get_cache(self, layer_id: int)->Dict[str, torch.Tensor]:
        
        return self.cache[layer_id - self.start_layer]

        
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):

        assert layer_id_override is None
        assert loc.dtype == torch.int64

        layer_id = layer.layer_id
        layer_cache = self.cache[layer_id - self.start_layer]

        if self.is_int8:
            cache_k_contig = cache_k.contiguous()
            cache_v_contig = cache_v.contiguous()
            # Quantize bf16 K/V to int8 with per-token absmax scales
            vortex_torch.cache.set_kv_buffer_int8_launcher(
                layer_cache["k"],
                layer_cache["v"],
                layer_cache["k_scale"],
                layer_cache["v_scale"],
                cache_k_contig,
                cache_v_contig,
                loc,
                self.page_size
            )
            # Pass int8 K directly to forward_cache; reduce kernels dequant
            # inline using QUANT_TYPE==1 with per-token scales from k_scale.
            self.ctx.quant_type = 1
            self.ctx.kv_scale_ptr = layer_cache["k_scale"]
            self.sparse_attention.forward_cache(layer_cache, loc, ctx=self.ctx)
        elif self.is_fp8:
            # FP8 path: quantize bf16→fp8, bitcast to uint8, scatter into paged cache.
            # Reduce kernels operate directly on uint8 with inline bitcast→fp8→float32 + scale.
            k_scale_val = k_scale if isinstance(k_scale, (int, float)) else k_scale.item() if k_scale is not None else 1.0
            v_scale_val = v_scale if isinstance(v_scale, (int, float)) else v_scale.item() if v_scale is not None else 1.0
            set_kv_buffer_fp8_launcher(
                layer_cache["k"], layer_cache["v"],
                cache_k.contiguous(), cache_v.contiguous(),
                loc, self.page_size, k_scale_val, v_scale_val,
                fp8_type=self.fp8_type,
            )
            # Propagate fp8_type, quant_type, and per-tensor scale so reduce kernels can dequant inline
            self.ctx.fp8_type = self.fp8_type
            self.ctx.quant_type = self.fp8_type + 1  # fp8_type=1 (e4m3) → quant_type=2, fp8_type=2 (e5m2) → quant_type=3
            self.ctx.kv_scale = k_scale_val
            self.sparse_attention.forward_cache(layer_cache, loc, ctx=self.ctx)
        else:
            assert cache_k.dtype == torch.bfloat16
            assert cache_v.dtype == torch.bfloat16
            vortex_torch.cache.set_kv_buffer_launcher(
                layer_cache["k"],
                layer_cache["v"],
                cache_k.contiguous(),
                cache_v.contiguous(),
                loc,
                self.page_size
            )
            self.sparse_attention.forward_cache(layer_cache, loc, ctx=self.ctx)
        
    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        
        raise NotImplementedError