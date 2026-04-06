import logging
import os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.utils import is_cuda

import vortex_torch
from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.cache.unified_view import UnifiedCacheView
from vortex_torch.cache.triton_kernels.set_kv import (
    set_kv_buffer_int8_launcher,
    set_kv_buffer_fp8_launcher,
    store_kv_cpu_and_gpu_int8,
    store_kv_cpu_and_gpu_fp8,
    store_kv_unified_int8,
    store_kv_unified_fp8,
)
from vortex_torch.cache.triton_kernels.paged_prefill_int8 import dequant_paged_int8_to_bf16_inplace

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
        gpu_full_size: int,
        sparse_attention,
        memory_saver_adapter,
        model_runner,
        full_attention_layer_ids: Optional[List[int]] = None,
        gpu_sparse_layer_ids: Optional[List[int]] = None,
        cpu_sparse_layer_ids: Optional[List[int]] = None,
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
        self.gpu_full_size = gpu_full_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.memory_saver_adapter = memory_saver_adapter
        self.enable_custom_mem_pool = enable_custom_mem_pool

        # Layer classification:
        # 1. Full attention layers - dense attention, full GPU KV
        # 2. GPU sparse layers - sparse attention, full GPU KV, no CPU->GPU copying
        # 3. CPU sparse layers - sparse attention, CPU KV + GPU staging, with copying
        self.full_attention_layer_ids = sorted(full_attention_layer_ids or [])
        self.gpu_sparse_layer_ids = sorted(gpu_sparse_layer_ids or [])
        self.cpu_sparse_layer_ids = sorted(cpu_sparse_layer_ids or [])

        self.num_full_layers = len(self.full_attention_layer_ids)
        self.num_gpu_sparse_layers = len(self.gpu_sparse_layer_ids)
        self.num_cpu_sparse_layers = len(self.cpu_sparse_layer_ids)

        # Layer mapping: global_layer_id -> (local_id, layer_type)
        # Each layer type has its own local index space for buffer access
        # layer_type: 'full', 'gpu_sparse', 'cpu_sparse'
        self.layers_mapping: Dict[int, Tuple[int, str]] = {}
        for local_id, global_id in enumerate(self.full_attention_layer_ids):
            self.layers_mapping[global_id] = (local_id, 'full')
        for local_id, global_id in enumerate(self.gpu_sparse_layer_ids):
            self.layers_mapping[global_id] = (local_id, 'gpu_sparse')
        for local_id, global_id in enumerate(self.cpu_sparse_layer_ids):
            self.layers_mapping[global_id] = (local_id, 'cpu_sparse')

        self.sparse_attention = sparse_attention
        self.ctx = vortex_torch.cache.Context()

        self.alloc_kernel = model_runner.server_args.vortex_alloc_kernel
        self.profile_enabled = model_runner.server_args.vortex_profile
        if self.profile_enabled:
            self.profile_tokens_generated = 0
            self.profile_log_interval = int(os.environ.get("VORTEX_PROFILE_LOG_INTERVAL", "100"))
            self.profile_step_count = 0
            self.profile_attn_accum = 0.0
            self.profile_nonattn_accum = 0.0
            self.profile_accum_count = 0
            self.profile_path = os.environ.get("VORTEX_PROFILE_PATH", "profile_data.jsonl")
            os.makedirs(os.path.dirname(self.profile_path) if os.path.dirname(self.profile_path) else ".", exist_ok=True)
            self._profile_file = open(self.profile_path, "w")
            import atexit
            atexit.register(self._close_profile_file)

        self.is_int8 = (self.dtype == torch.int8)
        self.is_fp8 = (self.dtype in (torch.float8_e4m3fn, torch.float8_e5m2))
        self.fp8_type = 1 if self.dtype == torch.float8_e4m3fn else (2 if self.dtype == torch.float8_e5m2 else 0)

        self._create_buffers()
        self._initialize_graph(model_runner)
        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None

        # Enable overflow check only when CUDA graph is disabled
        self.enable_overflow_check = model_runner.server_args.disable_cuda_graph

        cache_size = self.get_cache_size_bytes()
        staging_size = self.get_staging_size_bytes()
        full_attn_size = self.get_full_attention_size_bytes()

        print(
            f"CPU VTX Graph Cache allocated. #tokens_cpu: {size}, #tokens_gpu_staging: {gpu_size}, "
            f"#tokens_gpu_full: {gpu_full_size}, "
            f"full_layers: {self.num_full_layers}, gpu_sparse_layers: {self.num_gpu_sparse_layers}, "
            f"cpu_sparse_layers: {self.num_cpu_sparse_layers}, "
            f"CPU cache: {cache_size / GB:.2f} GB, GPU staging: {staging_size / GB:.2f} GB, "
            f"GPU full attention: {full_attn_size / GB:.2f} GB"
        )

        # GPU memory usage (staging + full attention buffers)
        self.mem_usage = (staging_size + full_attn_size) / GB

    def _create_buffers(self):
        # Page calculations for different buffer types
        self.num_pages_cpu = ((self.size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1
        self.num_pages_gpu_staging = ((self.gpu_size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1
        self.num_pages_gpu_full = ((self.gpu_full_size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1

        # Round down staging capacity to a multiple of 1024 (= ASSOCIATIVITY(32) * SETS_PER_BLOCK(32))
        # so that set-associative kernels have clean divisibility.
        # This also ensures K/V staging buffers and management structures use the same size.
        ALIGNMENT = 1024
        self.num_pages_gpu_staging = (self.num_pages_gpu_staging // ALIGNMENT) * ALIGNMENT

        # For compatibility with vortex context (used for landmarks)
        self.num_pages = self.num_pages_cpu

        self.cache_meta_info = self.sparse_attention.get_cache_meta_info(self.page_size, self.head_dim)

        # Determine storage dtype for K/V buffers
        if self.is_int8:
            kv_store_dtype = torch.int8
        elif self.is_fp8:
            kv_store_dtype = torch.uint8
        else:
            kv_store_dtype = self.store_dtype  # bf16

        # ========================================
        # CPU SPARSE LAYERS: CPU pinned + GPU staging
        # ========================================
        self.cache_cpu = []
        for _ in range(self.num_cpu_sparse_layers):
            temp = {}
            for (cache_name, cache_shape) in self.cache_meta_info.items():
                if cache_name in ["k", "v"]:
                    temp[cache_name] = torch.zeros(
                        (self.num_pages_cpu, cache_shape[0], cache_shape[1]),
                        dtype=kv_store_dtype,
                        device='cpu',
                        pin_memory=True,
                    )
            self.cache_cpu.append(temp)

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # GPU staging buffers for CPU sparse layers
            self.cache_staging = []
            for _ in range(self.num_cpu_sparse_layers):
                layer_cache = {}
                for (cache_name, cache_shape) in self.cache_meta_info.items():
                    if cache_name in ["k", "v"]:
                        # K/V use GPU staging buffer size with quantized dtype
                        layer_cache[cache_name] = torch.zeros(
                            (self.num_pages_gpu_staging, cache_shape[0], cache_shape[1]),
                            dtype=kv_store_dtype,
                            device=self.device,
                        )
                    else:
                        # Custom caches (centroids etc.) always bf16, use full CPU page count
                        layer_cache[cache_name] = torch.zeros(
                            (self.num_pages_cpu, cache_shape[0], cache_shape[1]),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                self.cache_staging.append(layer_cache)

            # Int8: persistent per-token scale buffers on GPU for ALL cpu sparse pages
            # Scales are tiny (~3.2 MB/layer) so they always live on GPU.
            # When int8 pages are copied back from CPU → GPU staging, scales are already available.
            if self.is_int8:
                self.cache_scale = []
                for _ in range(self.num_cpu_sparse_layers):
                    self.cache_scale.append({
                        "k_scale": torch.zeros(
                            (self.num_pages_cpu, self.page_size, 1),
                            dtype=torch.float16,
                            device=self.device,
                        ),
                        "v_scale": torch.zeros(
                            (self.num_pages_cpu, self.page_size, 1),
                            dtype=torch.float16,
                            device=self.device,
                        ),
                    })
                # No staging-indexed scale buffers needed: the paged_decode_int8 kernel
                # resolves scale page IDs on-the-fly via gpu_to_cpu_page_maps.

            # ========================================
            # GPU SPARSE LAYERS: GPU KV + centroids (like vtx_graph_backend)
            # ========================================
            self.cache_gpu_sparse = []
            for _ in range(self.num_gpu_sparse_layers):
                layer_cache = {}
                for (cache_name, cache_shape) in self.cache_meta_info.items():
                    if cache_name in ["k", "v"]:
                        layer_cache[cache_name] = torch.zeros(
                            (self.num_pages_gpu_full, cache_shape[0], cache_shape[1]),
                            dtype=kv_store_dtype,
                            device=self.device,
                        )
                    else:
                        layer_cache[cache_name] = torch.zeros(
                            (self.num_pages_gpu_full, cache_shape[0], cache_shape[1]),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                # Int8: per-token scale buffers for GPU sparse layers
                if self.is_int8:
                    layer_cache["k_scale"] = torch.zeros(
                        (self.num_pages_gpu_full, self.page_size, 1),
                        dtype=torch.float16,
                        device=self.device,
                    )
                    layer_cache["v_scale"] = torch.zeros(
                        (self.num_pages_gpu_full, self.page_size, 1),
                        dtype=torch.float16,
                        device=self.device,
                    )
                self.cache_gpu_sparse.append(layer_cache)

            # ========================================
            # FULL ATTENTION LAYERS: GPU KV only (no centroids)
            # ========================================
            self.k_buffer_full = [
                torch.zeros(
                    (self.num_pages_gpu_full, self.page_size, 1, self.head_dim),
                    dtype=kv_store_dtype,
                    device=self.device,
                )
                for _ in range(self.num_full_layers)
            ]
            self.v_buffer_full = [
                torch.zeros(
                    (self.num_pages_gpu_full, self.page_size, 1, self.head_dim),
                    dtype=kv_store_dtype,
                    device=self.device,
                )
                for _ in range(self.num_full_layers)
            ]
            # Int8: scale buffers for full attention layers
            if self.is_int8:
                self.k_scale_full = [
                    torch.zeros(
                        (self.num_pages_gpu_full, self.page_size, 1),
                        dtype=torch.float16,
                        device=self.device,
                    )
                    for _ in range(self.num_full_layers)
                ]
                self.v_scale_full = [
                    torch.zeros(
                        (self.num_pages_gpu_full, self.page_size, 1),
                        dtype=torch.float16,
                        device=self.device,
                    )
                    for _ in range(self.num_full_layers)
                ]

            # Shared bf16 K working buffer for forward_cache (centroid computation needs bf16 K).
            # Needed for int8 (dequant int8→bf16). fp8 uses UnifiedCacheView with
            # fp8-aware unified reduce kernel instead.
            if self.is_int8:
                k_shape = self.cache_meta_info["k"]
                max_pages = max(self.num_pages_gpu_staging, self.num_pages_gpu_full)
                self._k_bf16_working = torch.zeros(
                    (max_pages, k_shape[0], k_shape[1]),
                    dtype=torch.bfloat16,
                    device=self.device,
                )

        # Hybrid cache structures for CPU SPARSE layers only (LRU management)
        self._create_hybrid_structures_sparse()

    def _create_hybrid_structures_sparse(self):
        """Create cache management structures for CPU sparse attention layers."""
        from vortex_torch.cache.staging_cache import StagingCache

        max_page_id = self.num_pages_cpu
        staging_buffer_capacity = self.num_pages_gpu_staging

        # Determine cache policy from alloc_kernel name
        # Legacy names: "lru_block_global" -> policy "lru"
        # New names: "lfu_block_global" -> policy "lfu", "random_block_global" -> policy "random"
        cache_policy = "lru"
        if self.alloc_kernel.startswith("lfu"):
            cache_policy = "lfu"
        elif self.alloc_kernel.startswith("random"):
            cache_policy = "random"
        self.cache_policy = cache_policy

        # Create one StagingCache per CPU sparse layer
        self.staging_caches = []
        for _ in range(self.num_cpu_sparse_layers):
            self.staging_caches.append(
                StagingCache(
                    policy=cache_policy,
                    staging_capacity=staging_buffer_capacity,
                    max_num_pages=max_page_id,
                    device=self.device,
                )
            )

        # Backward-compatible accessors (used by other parts of the codebase)
        self.cpu_to_gpu_slot_maps = [sc.cpu_to_gpu_slot_map for sc in self.staging_caches]
        self.gpu_to_cpu_page_maps = [sc.gpu_to_cpu_page_map for sc in self.staging_caches]
        self.slot_ages = [sc.slot_state for sc in self.staging_caches]
        # All kernels now use per-set uint32 bitmask
        self.set_used_masks = [sc.set_used_masks for sc in self.staging_caches]

        self.max_num_pages = staging_buffer_capacity
        self.temp_owners_bitmap = torch.zeros(self.max_num_pages, dtype=torch.bool, device=self.device).contiguous()
        self.temp_staging_slots = torch.zeros(self.max_num_pages, dtype=torch.int32, device=self.device).contiguous()
        self.temp_overflow_flag = torch.zeros(1, dtype=torch.int32, device=self.device).contiguous()
        self.temp_evicted_cpu_pages = torch.full((self.max_num_pages,), -1, dtype=torch.int32, device=self.device).contiguous()

        self.max_page_id_sparse = max_page_id
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
        del self.cache_gpu_sparse
        del self.k_buffer_full
        del self.v_buffer_full

    def get_cache_size_bytes(self) -> int:
        """Return total bytes occupied by CPU cache tensors (CPU sparse layers only)."""
        total_bytes = 0
        for layer_cache in self.cache_cpu:
            for tensor in layer_cache.values():
                total_bytes += np.prod(tensor.shape) * tensor.dtype.itemsize
        return total_bytes

    def get_staging_size_bytes(self) -> int:
        """Return total bytes occupied by GPU staging buffers (CPU sparse layers)."""
        total_bytes = 0
        for layer_cache in self.cache_staging:
            for tensor in layer_cache.values():
                total_bytes += np.prod(tensor.shape) * tensor.dtype.itemsize
        # Persistent scale buffers for int8
        if self.is_int8 and hasattr(self, 'cache_scale'):
            for scale_dict in self.cache_scale:
                for tensor in scale_dict.values():
                    total_bytes += np.prod(tensor.shape) * tensor.dtype.itemsize
        # Int8 working buffer for dequant
        if hasattr(self, '_k_bf16_working'):
            total_bytes += np.prod(self._k_bf16_working.shape) * self._k_bf16_working.dtype.itemsize
        return total_bytes

    def get_full_attention_size_bytes(self) -> int:
        """Return total bytes occupied by full attention + GPU sparse GPU buffers."""
        total_bytes = 0
        # Full attention layers
        for k_buf, v_buf in zip(self.k_buffer_full, self.v_buffer_full):
            total_bytes += np.prod(k_buf.shape) * k_buf.dtype.itemsize
            total_bytes += np.prod(v_buf.shape) * v_buf.dtype.itemsize
        # Full attention scale buffers for int8
        if self.is_int8 and hasattr(self, 'k_scale_full'):
            for ks, vs in zip(self.k_scale_full, self.v_scale_full):
                total_bytes += np.prod(ks.shape) * ks.dtype.itemsize
                total_bytes += np.prod(vs.shape) * vs.dtype.itemsize
        # GPU sparse layers
        for layer_cache in self.cache_gpu_sparse:
            for tensor in layer_cache.values():
                total_bytes += np.prod(tensor.shape) * tensor.dtype.itemsize
        return total_bytes

    def get_layer_type(self, layer_id: int) -> str:
        """Return layer type: 'full', 'gpu_sparse', or 'cpu_sparse'."""
        return self.layers_mapping[layer_id][1]

    def is_sparse_layer(self, layer_id: int) -> bool:
        """Check if a layer uses sparse attention (gpu_sparse or cpu_sparse)."""
        layer_type = self.layers_mapping[layer_id][1]
        return layer_type in ('gpu_sparse', 'cpu_sparse')

    def is_cpu_sparse_layer(self, layer_id: int) -> bool:
        """Check if a layer uses CPU sparse attention with copying."""
        return self.layers_mapping[layer_id][1] == 'cpu_sparse'

    def get_key_buffer(self, layer_id: int):
        """Return K buffer for a layer."""
        local_id, layer_type = self.layers_mapping[layer_id]
        if layer_type == 'cpu_sparse':
            return self.cache_cpu[local_id]["k"]
        elif layer_type == 'gpu_sparse':
            return self.cache_gpu_sparse[local_id]["k"]
        else:  # full
            return self.k_buffer_full[local_id]

    def get_value_buffer(self, layer_id: int):
        """Return V buffer for a layer."""
        local_id, layer_type = self.layers_mapping[layer_id]
        if layer_type == 'cpu_sparse':
            return self.cache_cpu[local_id]["v"]
        elif layer_type == 'gpu_sparse':
            return self.cache_gpu_sparse[local_id]["v"]
        else:  # full
            return self.v_buffer_full[local_id]

    def get_kv_buffer(self, layer_id: int):
        """Return K and V buffers as tuple."""
        local_id, layer_type = self.layers_mapping[layer_id]
        if layer_type == 'cpu_sparse':
            return self.cache_cpu[local_id]["k"], self.cache_cpu[local_id]["v"]
        elif layer_type == 'gpu_sparse':
            return self.cache_gpu_sparse[local_id]["k"], self.cache_gpu_sparse[local_id]["v"]
        else:  # full
            return self.k_buffer_full[local_id], self.v_buffer_full[local_id]

    def get_kv_buffer_gpu(self, layer_id: int):
        """Return GPU K/V buffers for full attention or GPU sparse layers."""
        local_id, layer_type = self.layers_mapping[layer_id]
        if layer_type == 'cpu_sparse':
            raise ValueError(f"Layer {layer_id} is CPU sparse, use get_kv_buffer() or copy_sparse_kv_to_gpu()")
        elif layer_type == 'gpu_sparse':
            return self.cache_gpu_sparse[local_id]["k"], self.cache_gpu_sparse[local_id]["v"]
        else:  # full
            return self.k_buffer_full[local_id], self.v_buffer_full[local_id]

    def get_cache(self, layer_id: int) -> Dict[str, torch.Tensor]:
        """
        Return cache dictionary for sparse attention indexer.
        For GPU sparse: returns the full GPU cache.
        For CPU sparse: returns GPU staging buffers since indexer runs on GPU.
        """
        local_id, layer_type = self.layers_mapping[layer_id]
        if layer_type == 'full':
            raise ValueError(f"Layer {layer_id} is full attention, no sparse cache available")
        elif layer_type == 'gpu_sparse':
            return self.cache_gpu_sparse[local_id]
        else:  # cpu_sparse
            return self.cache_staging[local_id]

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
        Only valid for CPU sparse attention layers.

        Args:
            layer_id: Layer index
            sparse_kv_indices: Full sparse indices tensor
            sparse_kv_indptr: Indptr tensor [batch_size * num_kv_heads + 1]
            batch_size: Current batch size
            dst_kv_indices: Optional destination indices buffer

        Returns:
            k_staging: GPU K staging buffer (bf16/int8/uint8 depending on dtype)
            v_staging: GPU V staging buffer
            dst_staging_slots: GPU slots where pages were placed
            k_scale: (int8 only) persistent GPU K scale buffer, None otherwise
            v_scale: (int8 only) persistent GPU V scale buffer, None otherwise
        """
        local_id, layer_type = self.layers_mapping[layer_id]
        if layer_type != 'cpu_sparse':
            raise ValueError(f"Layer {layer_id} is {layer_type}, copy_sparse_kv_to_gpu only valid for cpu_sparse")

        # Get CPU and GPU buffers for sparse layer
        cpu_k = self.cache_cpu[local_id]["k"].view(-1, self.page_size, 1, self.head_dim)
        cpu_v = self.cache_cpu[local_id]["v"].view(-1, self.page_size, 1, self.head_dim)
        gpu_k = self.cache_staging[local_id]["k"].view(-1, self.page_size, 1, self.head_dim)
        gpu_v = self.cache_staging[local_id]["v"].view(-1, self.page_size, 1, self.head_dim)

        # Get caching state
        cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[local_id]
        gpu_to_cpu_map = self.gpu_to_cpu_page_maps[local_id]

        if dst_kv_indices is None:
            dst_staging_slots = self.temp_staging_slots
        else:
            dst_staging_slots = dst_kv_indices

        # Step 1: Allocation kernel
        if self.profile_enabled:
            # Snapshot hit rate BEFORE allocation modifies the slot map.
            # All GPU ops — no .item() or sync. We use indptr to index rather than
            # slicing with a GPU scalar (which would trigger implicit sync).
            # The indptr value is batch_size * head_num which is a Python int.
            _indptr_idx = batch_size * self.head_num
            # Use torch ops to compute hit rate without sync:
            # sparse_kv_indices has all pages packed; indptr tells us total count.
            # We check the full array up to indptr and mask with a range comparison.
            _total_pages = sparse_kv_indptr[_indptr_idx]  # GPU scalar tensor
            _all_slots = cpu_to_gpu_map[sparse_kv_indices]  # index the full buffer
            # Build mask: positions < _total_pages are valid requests
            _positions = torch.arange(sparse_kv_indices.shape[0],
                                      device=sparse_kv_indices.device)
            _valid = _positions < _total_pages  # GPU bool tensor, no sync
            _hits = (_all_slots >= 0) & _valid
            _hit_rate_tensor = _hits.float().sum() / _valid.float().sum().clamp(min=1)
            start_alloc = torch.cuda.Event(enable_timing=True)
            end_alloc = torch.cuda.Event(enable_timing=True)
            start_alloc.record()

        if self.alloc_kernel == "lru_block":
            vortex_torch.cache.allocate_pages_lru_block(
                sparse_kv_indices=sparse_kv_indices,
                sparse_kv_indptr=sparse_kv_indptr,
                cpu_to_gpu_slot_map=cpu_to_gpu_map,
                gpu_to_cpu_page_map=gpu_to_cpu_map,
                slot_ages=self.slot_ages[local_id],
                set_used_mask=self.set_used_masks[local_id],
                dst_gpu_slots=dst_staging_slots,
                owners_bitmap=self.temp_owners_bitmap,
                evicted_cpu_pages=self.temp_evicted_cpu_pages,
                overflow_flag=self.temp_overflow_flag,
                batch_size=batch_size,
                num_kv_heads=self.head_num,
                max_num_pages=self.max_num_pages,
            )
            alloc_owners_bitmap = self.temp_owners_bitmap
            alloc_evicted_cpu_pages = self.temp_evicted_cpu_pages
            alloc_overflow_flag = self.temp_overflow_flag
        elif self.alloc_kernel == "lru_global":
            vortex_torch.cache.allocate_pages_lru_global(
                sparse_kv_indices=sparse_kv_indices,
                sparse_kv_indptr=sparse_kv_indptr,
                cpu_to_gpu_slot_map=cpu_to_gpu_map,
                gpu_to_cpu_page_map=gpu_to_cpu_map,
                slot_ages=self.slot_ages[local_id],
                set_used_mask=self.set_used_masks[local_id],
                dst_gpu_slots=dst_staging_slots,
                owners_bitmap=self.temp_owners_bitmap,
                evicted_cpu_pages=self.temp_evicted_cpu_pages,
                overflow_flag=self.temp_overflow_flag,
                batch_size=batch_size,
                num_kv_heads=self.head_num,
                max_num_pages=self.max_num_pages,
            )
            alloc_owners_bitmap = self.temp_owners_bitmap
            alloc_evicted_cpu_pages = self.temp_evicted_cpu_pages
            alloc_overflow_flag = self.temp_overflow_flag
        else:
            # Policy-agnostic allocation via StagingCache (lru/lfu/random_block_global)
            self.staging_caches[local_id].allocate(
                sparse_kv_indices=sparse_kv_indices,
                sparse_kv_indptr=sparse_kv_indptr,
                dst_gpu_slots=dst_staging_slots,
                owners_bitmap=self.temp_owners_bitmap,
                evicted_cpu_pages=self.temp_evicted_cpu_pages,
                overflow_flag=self.temp_overflow_flag,
                batch_size=batch_size,
                num_kv_heads=self.head_num,
            )
            alloc_owners_bitmap = self.temp_owners_bitmap
            alloc_evicted_cpu_pages = self.temp_evicted_cpu_pages
            alloc_overflow_flag = self.temp_overflow_flag

        if self.profile_enabled:
            end_alloc.record()
            start_copy = torch.cuda.Event(enable_timing=True)
            end_copy = torch.cuda.Event(enable_timing=True)
            start_copy.record()

        # Step 2: Copy kernel
        vortex_torch.cache.copy_kv(
            cpu_k_buffer=cpu_k,
            cpu_v_buffer=cpu_v,
            gpu_k_buffer=gpu_k,
            gpu_v_buffer=gpu_v,
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_indptr=sparse_kv_indptr,
            dst_gpu_slots=dst_staging_slots,
            owners_bitmap=alloc_owners_bitmap,
            evicted_cpu_pages=alloc_evicted_cpu_pages,
            page_size=self.page_size,
            batch_size=batch_size,
            num_kv_heads=self.head_num,
            max_num_pages=self.max_num_pages,
        )

        if self.profile_enabled:
            end_copy.record()
            # NO sync here — events will be read after the attention backend's sync.
            # Store pending events for deferred reading.
            if not hasattr(self, '_pending_profile_events'):
                self._pending_profile_events = []
            self._pending_profile_events.append((
                start_alloc, end_alloc, start_copy, end_copy,
                _hit_rate_tensor, layer_id, batch_size
            ))

        # Check for staging buffer overflow (only when CUDA graph is disabled)
        if self.enable_overflow_check and alloc_overflow_flag.item() != 0:
            raise RuntimeError(
                f"GPU staging buffer overflow in copy_sparse_kv_to_gpu_with_indptr. "
                f"layer_id={layer_id}, batch_size={batch_size}, "
                f"staging_capacity={self.staging_buffer_capacity}. "
                f"This indicates max_num_reqs exceeds GPU staging buffer capacity."
            )

        # For int8: return persistent GPU scale buffers and the gpu_to_cpu page map.
        # The paged_decode_int8 kernel resolves scale page IDs on-the-fly via
        # the page map (staging slot → CPU flat page ID), so no remap/copy needed.
        if self.is_int8:
            k_scale = self.cache_scale[local_id]["k_scale"]
            v_scale = self.cache_scale[local_id]["v_scale"]
            gpu_to_cpu_map = self.gpu_to_cpu_page_maps[local_id]
            return gpu_k, gpu_v, dst_staging_slots, k_scale, v_scale, gpu_to_cpu_map

        return gpu_k, gpu_v, dst_staging_slots, None, None, None

    def _close_profile_file(self):
        """Close profile file on process exit."""
        if hasattr(self, '_profile_file') and self._profile_file and not self._profile_file.closed:
            self._profile_file.close()

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
        """Store KV to appropriate cache based on layer type."""
        layer_id = layer.layer_id
        local_id, layer_type = self.layers_mapping[layer_id]

        if self.is_int8:
            self._set_kv_buffer_int8(local_id, layer_type, loc, cache_k, cache_v)
        elif self.is_fp8:
            self._set_kv_buffer_fp8(local_id, layer_type, loc, cache_k, cache_v, k_scale, v_scale)
        else:
            self._set_kv_buffer_bf16(local_id, layer_type, loc, cache_k, cache_v)

    def _set_kv_buffer_bf16(self, local_id, layer_type, loc, cache_k, cache_v):
        """BF16 path: original implementation."""
        if layer_type == 'cpu_sparse':
            cpu_k_buffer = self.cache_cpu[local_id]["k"]
            cpu_v_buffer = self.cache_cpu[local_id]["v"]
            gpu_k_staging = self.cache_staging[local_id]["k"]
            gpu_v_staging = self.cache_staging[local_id]["v"]
            cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[local_id]

            vortex_torch.cache.store_kv_cpu_and_gpu(
                cpu_k_buffer, cpu_v_buffer,
                gpu_k_staging, gpu_v_staging,
                cache_k.contiguous(), cache_v.contiguous(),
                loc, self.page_size, cpu_to_gpu_map, self.max_page_id_sparse,
            )

            unified_cache = {
                "k": UnifiedCacheView(cpu_k_buffer, gpu_k_staging, cpu_to_gpu_map),
                "v": UnifiedCacheView(cpu_v_buffer, gpu_v_staging, cpu_to_gpu_map),
                "centroids": self.cache_staging[local_id]["centroids"]
            }
            self.sparse_attention.forward_cache(unified_cache, loc, ctx=self.ctx)

        elif layer_type == 'gpu_sparse':
            cache = self.cache_gpu_sparse[local_id]
            vortex_torch.cache.set_kv_buffer_launcher(
                cache["k"], cache["v"],
                cache_k.contiguous(), cache_v.contiguous(),
                loc, self.page_size,
            )
            self.sparse_attention.forward_cache(cache, loc, ctx=self.ctx)

        else:  # full attention
            vortex_torch.cache.set_kv_buffer_launcher(
                self.k_buffer_full[local_id], self.v_buffer_full[local_id],
                cache_k.contiguous(), cache_v.contiguous(),
                loc, self.page_size,
            )

    def _set_kv_buffer_int8(self, local_id, layer_type, loc, cache_k, cache_v):
        """Int8 path: quantize bf16→int8, store to appropriate caches."""
        cache_k_contig = cache_k.contiguous()
        cache_v_contig = cache_v.contiguous()

        if layer_type == 'cpu_sparse':
            # Quantize bf16→int8, write to CPU + GPU staging + persistent scales
            store_kv_cpu_and_gpu_int8(
                self.cache_cpu[local_id]["k"], self.cache_cpu[local_id]["v"],
                self.cache_staging[local_id]["k"], self.cache_staging[local_id]["v"],
                self.cache_scale[local_id]["k_scale"], self.cache_scale[local_id]["v_scale"],
                cache_k_contig, cache_v_contig,
                loc, self.page_size,
                self.cpu_to_gpu_slot_maps[local_id], self.max_page_id_sparse,
            )
            # Dequant affected pages from staging int8 → bf16 working buffer for forward_cache
            self._dequant_for_forward_cache(
                local_id, self.cache_staging[local_id], self.cache_scale[local_id],
                self.cpu_to_gpu_slot_maps[local_id], loc,
            )
            cache_for_forward = {k: v for k, v in self.cache_staging[local_id].items()}
            cache_for_forward["k"] = self._k_bf16_working
            self.sparse_attention.forward_cache(cache_for_forward, loc, ctx=self.ctx)

        elif layer_type == 'gpu_sparse':
            cache = self.cache_gpu_sparse[local_id]
            set_kv_buffer_int8_launcher(
                cache["k"], cache["v"],
                cache["k_scale"], cache["v_scale"],
                cache_k_contig, cache_v_contig,
                loc, self.page_size,
            )
            # Dequant affected pages for forward_cache
            self._dequant_for_forward_cache_direct(cache, loc)
            cache_for_forward = {k: v for k, v in cache.items()}
            cache_for_forward["k"] = self._k_bf16_working
            self.sparse_attention.forward_cache(cache_for_forward, loc, ctx=self.ctx)

        else:  # full attention
            set_kv_buffer_int8_launcher(
                self.k_buffer_full[local_id], self.v_buffer_full[local_id],
                self.k_scale_full[local_id], self.v_scale_full[local_id],
                cache_k_contig, cache_v_contig,
                loc, self.page_size,
            )

    def _dequant_for_forward_cache_direct(self, cache, loc):
        """Dequant int8 K pages to _k_bf16_working for forward_cache (GPU-resident cache)."""
        token_page_ids = loc // self.page_size
        if torch.cuda.is_current_stream_capturing():
            unique_page_ids = token_page_ids
        else:
            unique_page_ids = torch.unique(token_page_ids)
        flat_page_ids = (
            unique_page_ids[:, None] * self.head_num
            + torch.arange(self.head_num, device=loc.device)[None, :]
        ).reshape(-1)
        dequant_paged_int8_to_bf16_inplace(
            cache["k"], cache["k_scale"],
            self._k_bf16_working,
            flat_page_ids.to(torch.int32),
            self.page_size, self.head_dim,
        )

    def _dequant_for_forward_cache(self, local_id, staging_cache, scale_cache, cpu_to_gpu_map, loc):
        """Dequant int8 K pages to _k_bf16_working for forward_cache.

        CPU sparse layers have TWO index spaces: staging slots (for int8 data
        in GPU staging) and CPU flat page IDs (for data in CPU pinned memory,
        plus scales and forward_cache access).

        During prefill, cpu_to_gpu_slot_map starts all -1, so NO pages are in
        staging — we read from CPU pinned memory via the CUDA dequant kernel.
        During decode, most pages are in staging but some may have been evicted.

        Uses:
        - Triton kernel (separate data/scale/dst indices) for staging-resident pages
        - CUDA C++ kernel (dequant_int8_cpu_to_bf16) for CPU-resident pages
        Both are CUDA graph compatible (no Python CPU↔GPU sync).
        """
        token_page_ids = loc // self.page_size
        if torch.cuda.is_current_stream_capturing():
            unique_page_ids = token_page_ids
        else:
            unique_page_ids = torch.unique(token_page_ids)

        cpu_cache_k = self.cache_cpu[local_id]["k"]  # int8, CPU pinned

        for head_id in range(self.head_num):
            flat_cpu_pages = unique_page_ids * self.head_num + head_id
            staging_slots = cpu_to_gpu_map[flat_cpu_pages]

            if torch.cuda.is_current_stream_capturing():
                # CUDA graph: cannot branch on data. Assume pages in staging
                # (decode tokens should always be in staging for active requests).
                safe_slots = staging_slots.clamp(min=0)
                # Triton kernel with separate indices: data at staging slots,
                # scales at CPU pages, destination at CPU pages.
                dequant_paged_int8_to_bf16_inplace(
                    staging_cache["k"],      # int8 data (staging-slot-indexed)
                    scale_cache["k_scale"],  # scales (CPU-page-indexed)
                    self._k_bf16_working,    # destination (CPU-page-indexed)
                    safe_slots.to(torch.int32),       # data page IDs
                    self.page_size, self.head_dim,
                    scale_page_indices=flat_cpu_pages.to(torch.int32),  # scale page IDs
                    dst_page_indices=flat_cpu_pages.to(torch.int32),    # dst page IDs
                )
            else:
                in_staging = staging_slots >= 0
                # Pages in staging: Triton kernel with separate indices
                if in_staging.any():
                    s = staging_slots[in_staging].to(torch.int32)
                    c = flat_cpu_pages[in_staging].to(torch.int32)
                    dequant_paged_int8_to_bf16_inplace(
                        staging_cache["k"], scale_cache["k_scale"],
                        self._k_bf16_working,
                        s, self.page_size, self.head_dim,
                        scale_page_indices=c, dst_page_indices=c,
                    )
                # Pages NOT in staging: CUDA kernel reads from CPU pinned via UVA
                if (~in_staging).any():
                    c = flat_cpu_pages[~in_staging].to(torch.int32)
                    vortex_torch.cache.dequant_int8_cpu_to_bf16(
                        cpu_cache_k,             # CPU pinned int8
                        scale_cache["k_scale"],  # GPU fp16 scales
                        self._k_bf16_working,    # GPU bf16 destination
                        c, c,                    # src=dst=CPU page IDs
                        self.page_size, self.head_dim,
                    )

    def _set_kv_buffer_fp8(self, local_id, layer_type, loc, cache_k, cache_v, k_scale, v_scale):
        """FP8 path: quantize bf16→fp8 (uint8), store to appropriate caches."""
        cache_k_contig = cache_k.contiguous()
        cache_v_contig = cache_v.contiguous()
        k_scale_val = k_scale if isinstance(k_scale, (int, float)) else k_scale.item() if k_scale is not None else 1.0
        v_scale_val = v_scale if isinstance(v_scale, (int, float)) else v_scale.item() if v_scale is not None else 1.0

        if layer_type == 'cpu_sparse':
            # Quantize bf16→fp8, write to CPU + GPU staging in one kernel
            store_kv_cpu_and_gpu_fp8(
                self.cache_cpu[local_id]["k"], self.cache_cpu[local_id]["v"],
                self.cache_staging[local_id]["k"], self.cache_staging[local_id]["v"],
                cache_k_contig, cache_v_contig,
                loc, self.page_size,
                self.cpu_to_gpu_slot_maps[local_id], self.max_page_id_sparse,
                k_scale_val, v_scale_val, fp8_type=self.fp8_type,
            )
            # Use UnifiedCacheView so forward_cache reads from CPU when pages
            # aren't in staging (e.g., during prefill when staging is empty).
            # The unified reduce kernel now supports fp8 (quant_type 1/2).
            self.ctx.fp8_type = self.fp8_type
            self.ctx.kv_scale = k_scale_val
            cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[local_id]
            unified_cache = {
                "k": UnifiedCacheView(self.cache_cpu[local_id]["k"],
                                      self.cache_staging[local_id]["k"], cpu_to_gpu_map),
                "v": UnifiedCacheView(self.cache_cpu[local_id]["v"],
                                      self.cache_staging[local_id]["v"], cpu_to_gpu_map),
                "centroids": self.cache_staging[local_id]["centroids"]
            }
            self.sparse_attention.forward_cache(unified_cache, loc, ctx=self.ctx)

        elif layer_type == 'gpu_sparse':
            cache = self.cache_gpu_sparse[local_id]
            set_kv_buffer_fp8_launcher(
                cache["k"], cache["v"],
                cache_k_contig, cache_v_contig,
                loc, self.page_size, k_scale_val, v_scale_val,
                fp8_type=self.fp8_type,
            )
            self.ctx.fp8_type = self.fp8_type
            self.ctx.kv_scale = k_scale_val
            self.sparse_attention.forward_cache(cache, loc, ctx=self.ctx)

        else:  # full attention
            set_kv_buffer_fp8_launcher(
                self.k_buffer_full[local_id], self.v_buffer_full[local_id],
                cache_k_contig, cache_v_contig,
                loc, self.page_size, k_scale_val, v_scale_val,
                fp8_type=self.fp8_type,
            )

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
        """Store KV during decode phase based on layer type."""

        assert layer_id_override is None
        assert loc.dtype == torch.int64

        layer_id = layer.layer_id
        local_id, layer_type = self.layers_mapping[layer_id]

        if self.is_int8:
            self._set_kv_buffer_decode_int8(local_id, layer_type, loc, cache_k, cache_v)
        elif self.is_fp8:
            self._set_kv_buffer_decode_fp8(local_id, layer_type, loc, cache_k, cache_v, k_scale, v_scale)
        else:
            self._set_kv_buffer_decode_bf16(local_id, layer_type, loc, cache_k, cache_v)

    def _set_kv_buffer_decode_bf16(self, local_id, layer_type, loc, cache_k, cache_v):
        """BF16 decode path."""
        if layer_type == 'cpu_sparse':
            cpu_k = self.cache_cpu[local_id]["k"]
            cpu_v = self.cache_cpu[local_id]["v"]
            gpu_k = self.cache_staging[local_id]["k"]
            gpu_v = self.cache_staging[local_id]["v"]
            cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[local_id]
            vortex_torch.cache.store_kv_unified(
                cpu_k, cpu_v, gpu_k, gpu_v,
                cache_k.contiguous(), cache_v.contiguous(),
                loc, cpu_to_gpu_map, self.page_size,
            )
            unified_cache = {
                "k": UnifiedCacheView(cpu_k, gpu_k, cpu_to_gpu_map),
                "v": UnifiedCacheView(cpu_v, gpu_v, cpu_to_gpu_map),
                "centroids": self.cache_staging[local_id]["centroids"]
            }
            self.sparse_attention.forward_cache(unified_cache, loc, ctx=self.ctx)

        elif layer_type == 'gpu_sparse':
            cache = self.cache_gpu_sparse[local_id]
            vortex_torch.cache.set_kv_buffer_launcher(
                cache["k"], cache["v"],
                cache_k.contiguous(), cache_v.contiguous(),
                loc, self.page_size,
            )
            self.sparse_attention.forward_cache(cache, loc, ctx=self.ctx)

        else:  # full attention
            vortex_torch.cache.set_kv_buffer_launcher(
                self.k_buffer_full[local_id], self.v_buffer_full[local_id],
                cache_k.contiguous(), cache_v.contiguous(),
                loc, self.page_size,
            )

    def _set_kv_buffer_decode_int8(self, local_id, layer_type, loc, cache_k, cache_v):
        """Int8 decode path: quantize and store, dequant for forward_cache."""
        cache_k_contig = cache_k.contiguous()
        cache_v_contig = cache_v.contiguous()

        if layer_type == 'cpu_sparse':
            # Unified routing: quantize bf16→int8, route to EITHER CPU or GPU
            # based on slot map. Scales always written to persistent GPU buffer.
            store_kv_unified_int8(
                self.cache_cpu[local_id]["k"], self.cache_cpu[local_id]["v"],
                self.cache_staging[local_id]["k"], self.cache_staging[local_id]["v"],
                self.cache_scale[local_id]["k_scale"], self.cache_scale[local_id]["v_scale"],
                cache_k_contig, cache_v_contig,
                loc, self.page_size,
                self.cpu_to_gpu_slot_maps[local_id],
            )
            # Dequant affected pages for forward_cache (small cost for decode)
            self._dequant_for_forward_cache(
                local_id, self.cache_staging[local_id], self.cache_scale[local_id],
                self.cpu_to_gpu_slot_maps[local_id], loc,
            )
            cache_for_forward = {k: v for k, v in self.cache_staging[local_id].items()}
            cache_for_forward["k"] = self._k_bf16_working
            self.sparse_attention.forward_cache(cache_for_forward, loc, ctx=self.ctx)

        elif layer_type == 'gpu_sparse':
            cache = self.cache_gpu_sparse[local_id]
            set_kv_buffer_int8_launcher(
                cache["k"], cache["v"],
                cache["k_scale"], cache["v_scale"],
                cache_k_contig, cache_v_contig,
                loc, self.page_size,
            )
            self._dequant_for_forward_cache_direct(cache, loc)
            cache_for_forward = {k: v for k, v in cache.items()}
            cache_for_forward["k"] = self._k_bf16_working
            self.sparse_attention.forward_cache(cache_for_forward, loc, ctx=self.ctx)

        else:  # full attention
            set_kv_buffer_int8_launcher(
                self.k_buffer_full[local_id], self.v_buffer_full[local_id],
                self.k_scale_full[local_id], self.v_scale_full[local_id],
                cache_k_contig, cache_v_contig,
                loc, self.page_size,
            )

    def _set_kv_buffer_decode_fp8(self, local_id, layer_type, loc, cache_k, cache_v, k_scale, v_scale):
        """FP8 decode path."""
        cache_k_contig = cache_k.contiguous()
        cache_v_contig = cache_v.contiguous()
        k_scale_val = k_scale if isinstance(k_scale, (int, float)) else k_scale.item() if k_scale is not None else 1.0
        v_scale_val = v_scale if isinstance(v_scale, (int, float)) else v_scale.item() if v_scale is not None else 1.0

        if layer_type == 'cpu_sparse':
            # Unified routing: quantize bf16→fp8, route to EITHER CPU or GPU
            store_kv_unified_fp8(
                self.cache_cpu[local_id]["k"], self.cache_cpu[local_id]["v"],
                self.cache_staging[local_id]["k"], self.cache_staging[local_id]["v"],
                cache_k_contig, cache_v_contig,
                loc, self.page_size,
                self.cpu_to_gpu_slot_maps[local_id],
                k_scale_val, v_scale_val, fp8_type=self.fp8_type,
            )
            self.ctx.fp8_type = self.fp8_type
            self.ctx.kv_scale = k_scale_val
            cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[local_id]
            unified_cache = {
                "k": UnifiedCacheView(self.cache_cpu[local_id]["k"],
                                      self.cache_staging[local_id]["k"], cpu_to_gpu_map),
                "v": UnifiedCacheView(self.cache_cpu[local_id]["v"],
                                      self.cache_staging[local_id]["v"], cpu_to_gpu_map),
                "centroids": self.cache_staging[local_id]["centroids"]
            }
            self.sparse_attention.forward_cache(unified_cache, loc, ctx=self.ctx)

        elif layer_type == 'gpu_sparse':
            cache = self.cache_gpu_sparse[local_id]
            set_kv_buffer_fp8_launcher(
                cache["k"], cache["v"],
                cache_k_contig, cache_v_contig,
                loc, self.page_size, k_scale_val, v_scale_val,
                fp8_type=self.fp8_type,
            )
            self.ctx.fp8_type = self.fp8_type
            self.ctx.kv_scale = k_scale_val
            self.sparse_attention.forward_cache(cache, loc, ctx=self.ctx)

        else:  # full attention
            set_kv_buffer_fp8_launcher(
                self.k_buffer_full[local_id], self.v_buffer_full[local_id],
                cache_k_contig, cache_v_contig,
                loc, self.page_size, k_scale_val, v_scale_val,
                fp8_type=self.fp8_type,
            )

    def available_size(self) -> int:
        """Return available cache size."""
        return self.size

    def alloc(self, need_size: int) -> List[int]:
        """Allocate cache slots."""
        raise NotImplementedError("Direct allocation not supported for CPU VTX pool")

    def free(self, free_index: torch.Tensor) -> None:
        """Free cache slots."""
        raise NotImplementedError("Direct freeing not supported for CPU VTX pool")
