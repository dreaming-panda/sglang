import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from contextlib import nullcontext

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.utils import is_cuda

import vortex_torch
from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.cache.unified_view import UnifiedCacheView

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()


class CPUVTXGraphTokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: torch.device,
        page_size: int,
        gpu_size: int,
        sparse_attention,
        memory_saver_adapter,
        model_runner,
        enable_custom_mem_pool: bool = False,
    ):
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=True,
            start_layer=0,
            end_layer=layer_num,
        )
        self.gpu_size = gpu_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.memory_saver_adapter = memory_saver_adapter
        self.enable_custom_mem_pool = enable_custom_mem_pool

        self.sparse_attention = sparse_attention
        self.ctx = vortex_torch.cache.Context()

        self._create_buffers()
        self._initialize_graph(model_runner)
        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None

        cache_size = self.get_cache_size_bytes()
        staging_size = self.get_staging_size_bytes()

        print(
            f"CPU VTX Graph Cache allocated. #tokens: {size}, "
            f"Cache size: {cache_size / GB:.2f} GB (CPU), "
            f"Staging size: {staging_size / GB:.2f} GB (GPU)"
        )

        # GPU memory usage (staging buffers only)
        self.mem_usage = staging_size / GB

        assert self.dtype == torch.bfloat16
        assert self.store_dtype == torch.bfloat16

    def _create_buffers(self):
        self.num_pages = ((self.size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1
        self.num_pages_gpu = ((self.gpu_size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1

        self.cache_meta_info = self.sparse_attention.get_cache_meta_info(self.page_size, self.head_dim)
        # self.cache_cpu = [
        #     {
        #         cache_name: torch.zeros(
        #             (self.num_pages, cache_shape[0], cache_shape[1]),
        #             dtype=self.store_dtype,
        #             device='cpu',
        #             pin_memory=True,
        #         )
        #         for (cache_name, cache_shape) in self.cache_meta_info.items()
        #     }
        #     for _ in range(self.layer_num)
        # ]
        
        self.cache_cpu = []
        
        for _ in range(self.layer_num):
            temp = {}
            for (cache_name, cache_shape) in self.cache_meta_info.items():
                if cache_name in ["k", "v"]:
                    temp[cache_name] = torch.zeros(
                        (self.num_pages, cache_shape[0], cache_shape[1]),
                        dtype=self.store_dtype,
                        device='cpu',
                        pin_memory=True,
                    )
            self.cache_cpu.append(temp)

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.cache_staging = [
                {
                    cache_name: torch.zeros(
                        (self.num_pages_gpu, cache_shape[0], cache_shape[1]),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for (cache_name, cache_shape) in self.cache_meta_info.items()
                }
                for _ in range(self.layer_num)
            ]

        max_page_id = self.num_pages
        staging_buffer_capacity = self.num_pages_gpu

        self.cpu_to_gpu_slot_maps = []
        self.gpu_to_cpu_page_maps = []
        self.slot_ages = []

        for _ in range(self.layer_num):
            cpu_to_gpu_map = torch.full(
                (max_page_id,), -1, dtype=torch.int32, device=self.device
            ).contiguous()
            self.cpu_to_gpu_slot_maps.append(cpu_to_gpu_map)

            gpu_to_cpu_map = torch.full(
                (staging_buffer_capacity,), -1, dtype=torch.int32, device=self.device
            ).contiguous()
            self.gpu_to_cpu_page_maps.append(gpu_to_cpu_map)

            slot_ages = torch.zeros(staging_buffer_capacity, dtype=torch.uint8, device=self.device).contiguous()
            self.slot_ages.append(slot_ages)
            
        self.max_num_pages = staging_buffer_capacity
        self.temp_owners_bitmap = torch.zeros(self.max_num_pages, dtype=torch.bool, device=self.device).contiguous()
        self.temp_staging_slots = torch.zeros(self.max_num_pages, dtype=torch.int32, device=self.device).contiguous()
        self.temp_overflow_flag = torch.zeros(1, dtype=torch.int32, device=self.device).contiguous()
        self.temp_slots_used_bitmap = torch.zeros(staging_buffer_capacity, dtype=torch.bool, device=self.device).contiguous()
        self.temp_needs_eviction_bitmap = torch.zeros(self.max_num_pages, dtype=torch.bool, device=self.device).contiguous()
        self.temp_evicted_cpu_pages = torch.full((self.max_num_pages,), -1, dtype=torch.int32, device=self.device).contiguous()

        self.max_page_id = max_page_id
        self.staging_buffer_capacity = staging_buffer_capacity

    def _initialize_graph(self, model_runner) -> None:
        
        self.ctx.create(self, model_runner)
        self.ctx.profile()

        try:
            with torch.no_grad():
                loc_dummy = torch.empty((0,), dtype=torch.int64, device=self.device)
                cache_dummy = {
                    cache_name: as_vtensor(
                        torch.zeros(
                            (0, cache_shape[0], cache_shape[1]),
                            dtype=self.store_dtype,
                            device=self.device,
                        ),
                        FORMAT.PAGED
                    )
                    for (cache_name, cache_shape) in self.cache_meta_info.items()
                }
                self.sparse_attention.forward_cache(cache=cache_dummy, loc=loc_dummy, ctx=self.ctx)
        except Exception:
            raise

        self.ctx.summary()
        self.ctx.execute()

    def _clear_buffers(self):
        del self.cache_cpu
        del self.cache_staging

    def get_cache_size_bytes(self) -> int:
        """Return total bytes occupied by CPU cache tensors."""
        total_bytes = 0
        for layer_cache in self.cache_cpu:
            for tensor in layer_cache.values():
                total_bytes += np.prod(tensor.shape) * tensor.dtype.itemsize
        return total_bytes

    def get_staging_size_bytes(self) -> int:
        """Return total bytes occupied by GPU staging buffers."""
        total_bytes = 0
        for layer_cache in self.cache_staging:
            for tensor in layer_cache.values():
                total_bytes += np.prod(tensor.shape) * tensor.dtype.itemsize
        return total_bytes

    def get_key_buffer(self, layer_id: int):
        """Return CPU K buffer for a layer."""
        return self.cache_cpu[layer_id - self.start_layer]["k"]

    def get_value_buffer(self, layer_id: int):
        """Return CPU V buffer for a layer."""
        return self.cache_cpu[layer_id - self.start_layer]["v"]

    def get_kv_buffer(self, layer_id: int):
        """Return CPU K and V buffers as tuple. FlashInfer will handle device transfer."""
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_cache(self, layer_id: int) -> Dict[str, torch.Tensor]:
        """
        Return cache dictionary for sparse attention indexer.
        Returns GPU staging buffers since indexer runs on GPU.
        """
        return self.cache_staging[layer_id - self.start_layer]

    def copy_sparse_kv_to_gpu(
        self,
        layer_id: int,
        sparse_kv_indices: torch.Tensor,
        sparse_kv_indptr: torch.Tensor,
        batch_size: int,
        dst_kv_indices: Optional[torch.Tensor] = None,
    ):
        """
        CUDA graph compatible sparse KV copy from CPU to GPU staging buffers.

        Args:
            layer_id: Layer index
            sparse_kv_indices: Full sparse indices tensor
            sparse_kv_indptr: Indptr tensor [batch_size * num_kv_heads + 1]
            batch_size: Current batch size
            dst_kv_indices: Optional destination indices buffer

        Returns:
            k_staging: GPU K staging buffer
            v_staging: GPU V staging buffer
            dst_staging_slots: GPU slots where pages were placed
        """
        layer_idx = layer_id - self.start_layer

        # Get CPU and GPU buffers
        cpu_k = self.cache_cpu[layer_idx]["k"].view(-1, self.page_size, 1, self.head_dim)
        cpu_v = self.cache_cpu[layer_idx]["v"].view(-1, self.page_size, 1, self.head_dim)
        gpu_k = self.cache_staging[layer_idx]["k"].view(-1, self.page_size, 1, self.head_dim)
        gpu_v = self.cache_staging[layer_idx]["v"].view(-1, self.page_size, 1, self.head_dim)

        # Get caching state
        cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[layer_idx]
        gpu_to_cpu_map = self.gpu_to_cpu_page_maps[layer_idx]

        if dst_kv_indices is None:
            dst_staging_slots = self.temp_staging_slots
        else:
            dst_staging_slots = dst_kv_indices

        vortex_torch.cache.copy_sparse_kv_to_gpu_with_indptr(
            cpu_k_buffer=cpu_k,
            cpu_v_buffer=cpu_v,
            gpu_k_buffer=gpu_k,
            gpu_v_buffer=gpu_v,
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_indptr=sparse_kv_indptr,
            cpu_to_gpu_slot_map=cpu_to_gpu_map,
            gpu_to_cpu_page_map=gpu_to_cpu_map,
            slot_ages=self.slot_ages[layer_idx],
            dst_gpu_slots=dst_staging_slots,
            owners_bitmap=self.temp_owners_bitmap,
            slots_used_bitmap=self.temp_slots_used_bitmap,
            needs_eviction_bitmap=self.temp_needs_eviction_bitmap,
            evicted_cpu_pages=self.temp_evicted_cpu_pages,
            overflow_flag=self.temp_overflow_flag,
            page_size=self.page_size,
            batch_size=batch_size,
            num_kv_heads=self.head_num,
            max_num_pages=self.max_num_pages,
        )

        return gpu_k, gpu_v, dst_staging_slots

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
        """Store KV to CPU cache and update GPU staging if cached."""
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer

        cpu_k_buffer = self.cache_cpu[layer_idx]["k"]
        cpu_v_buffer = self.cache_cpu[layer_idx]["v"]
        gpu_k_staging = self.cache_staging[layer_idx]["k"]
        gpu_v_staging = self.cache_staging[layer_idx]["v"]
        cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[layer_idx]

        vortex_torch.cache.store_kv_cpu_and_gpu(
            cpu_k_buffer,
            cpu_v_buffer,
            gpu_k_staging,
            gpu_v_staging,
            cache_k.contiguous(),
            cache_v.contiguous(),
            loc,
            self.page_size,
            cpu_to_gpu_map,
            self.max_page_id,
        )
        
        unified_cache = {
            "k": UnifiedCacheView(
                cpu_k_buffer,
                gpu_k_staging,
                cpu_to_gpu_map
            ),
            "v": UnifiedCacheView(
                cpu_v_buffer,
                gpu_v_staging,
                cpu_to_gpu_map
            ),
            "centroids": self.cache_staging[layer_idx]["centroids"]
        }
        
        self.sparse_attention.forward_cache(unified_cache, loc, ctx=self.ctx)

    def set_kv_buffer_decode(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Store KV during decode phase with unified CPU/GPU memory."""
        
        assert layer_id_override is None
        assert k_scale is None
        assert v_scale is None
        assert cache_k.dtype == torch.bfloat16
        assert cache_v.dtype == torch.bfloat16
        assert loc.dtype == torch.int64
        
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer
        
        cpu_k_buffer = self.cache_cpu[layer_idx]["k"]
        cpu_v_buffer = self.cache_cpu[layer_idx]["v"]
        gpu_k_staging = self.cache_staging[layer_idx]["k"]
        gpu_v_staging = self.cache_staging[layer_idx]["v"]
        cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[layer_idx]
        
        vortex_torch.cache.store_kv_unified(
            cpu_k_buffer,
            cpu_v_buffer,
            gpu_k_staging,
            gpu_v_staging,
            cache_k.contiguous(),
            cache_v.contiguous(),
            loc,
            cpu_to_gpu_map,
            self.page_size,
        )
        
        unified_cache = {
            "k": UnifiedCacheView(
                cpu_k_buffer,
                gpu_k_staging,
                cpu_to_gpu_map
            ),
            "v": UnifiedCacheView(
                cpu_v_buffer,
                gpu_v_staging,
                cpu_to_gpu_map
            ),
            "centroids": self.cache_staging[layer_idx]["centroids"]
        }
        
        self.sparse_attention.forward_cache(unified_cache, loc, ctx=self.ctx)

    def available_size(self) -> int:
        """Return available cache size."""
        return self.size

    def alloc(self, need_size: int) -> List[int]:
        """Allocate cache slots."""
        raise NotImplementedError("Direct allocation not supported for CPU VTX pool")

    def free(self, free_index: torch.Tensor) -> None:
        """Free cache slots."""
        raise NotImplementedError("Direct freeing not supported for CPU VTX pool")
