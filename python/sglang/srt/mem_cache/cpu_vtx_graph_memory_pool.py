import logging
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

        assert self.dtype == torch.bfloat16
        assert self.store_dtype == torch.bfloat16

    def _create_buffers(self):
        # Page calculations for different buffer types
        self.num_pages_cpu = ((self.size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1
        self.num_pages_gpu_staging = ((self.gpu_size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1
        self.num_pages_gpu_full = ((self.gpu_full_size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1

        # For compatibility with vortex context (used for landmarks)
        self.num_pages = self.num_pages_cpu

        self.cache_meta_info = self.sparse_attention.get_cache_meta_info(self.page_size, self.head_dim)

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
                        dtype=self.store_dtype,
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
                    # k and v use GPU staging buffer size; other caches (e.g., centroids) use full CPU size
                    num_pages_for_cache = self.num_pages_gpu_staging if cache_name in ["k", "v"] else self.num_pages_cpu
                    layer_cache[cache_name] = torch.zeros(
                        (num_pages_for_cache, cache_shape[0], cache_shape[1]),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                self.cache_staging.append(layer_cache)

            # ========================================
            # GPU SPARSE LAYERS: GPU KV + centroids (like vtx_graph_backend)
            # ========================================
            self.cache_gpu_sparse = []
            for _ in range(self.num_gpu_sparse_layers):
                layer_cache = {}
                for (cache_name, cache_shape) in self.cache_meta_info.items():
                    # All caches use full GPU size (no staging needed)
                    layer_cache[cache_name] = torch.zeros(
                        (self.num_pages_gpu_full, cache_shape[0], cache_shape[1]),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                self.cache_gpu_sparse.append(layer_cache)

            # ========================================
            # FULL ATTENTION LAYERS: GPU KV only (no centroids)
            # ========================================
            self.k_buffer_full = [
                torch.zeros(
                    (self.num_pages_gpu_full, self.page_size, 1, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.num_full_layers)
            ]
            self.v_buffer_full = [
                torch.zeros(
                    (self.num_pages_gpu_full, self.page_size, 1, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.num_full_layers)
            ]

        # Hybrid cache structures for CPU SPARSE layers only (LRU management)
        self._create_hybrid_structures_sparse()

    def _create_hybrid_structures_sparse(self):
        """Create hybrid cache management structures for CPU sparse attention layers only."""
        max_page_id = self.num_pages_cpu
        staging_buffer_capacity = self.num_pages_gpu_staging

        self.cpu_to_gpu_slot_maps = []
        self.gpu_to_cpu_page_maps = []

        # Hybrid-style data structures (per CPU sparse layer)
        WAYS = 32
        num_sets = staging_buffer_capacity // WAYS
        self.num_sets = num_sets
        self.slot_stamps = []
        self.set_clocks = []
        self.set_versions = []
        self.set_used_masks = []

        for _ in range(self.num_cpu_sparse_layers):
            cpu_to_gpu_map = torch.full(
                (max_page_id,), -1, dtype=torch.int32, device=self.device
            ).contiguous()
            self.cpu_to_gpu_slot_maps.append(cpu_to_gpu_map)

            gpu_to_cpu_map = torch.full(
                (staging_buffer_capacity,), -1, dtype=torch.int32, device=self.device
            ).contiguous()
            self.gpu_to_cpu_page_maps.append(gpu_to_cpu_map)

            # Hybrid data structures: timestamps + per-set clocks + seqlocks + used masks
            slot_stamps = torch.zeros(staging_buffer_capacity, dtype=torch.int32, device=self.device).contiguous()
            set_clock = torch.zeros(num_sets, dtype=torch.int32, device=self.device).contiguous()
            set_version = torch.zeros(num_sets, dtype=torch.int32, device=self.device).contiguous()
            set_used_mask = torch.zeros(num_sets, dtype=torch.int32, device=self.device).contiguous()

            # Initialize hybrid structures via kernel (sets clock=1, version=0)
            vortex_torch.cache.init_hybrid_structures(
                slot_stamps, set_clock, set_version, staging_buffer_capacity, num_sets
            )

            self.slot_stamps.append(slot_stamps)
            self.set_clocks.append(set_clock)
            self.set_versions.append(set_version)
            self.set_used_masks.append(set_used_mask)

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
        return total_bytes

    def get_full_attention_size_bytes(self) -> int:
        """Return total bytes occupied by full attention + GPU sparse GPU buffers."""
        total_bytes = 0
        # Full attention layers
        for k_buf, v_buf in zip(self.k_buffer_full, self.v_buffer_full):
            total_bytes += np.prod(k_buf.shape) * k_buf.dtype.itemsize
            total_bytes += np.prod(v_buf.shape) * v_buf.dtype.itemsize
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
            k_staging: GPU K staging buffer
            v_staging: GPU V staging buffer
            dst_staging_slots: GPU slots where pages were placed
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

        # Step 1: Allocation kernel (Hybrid lock-free with seqlock)
        vortex_torch.cache.allocate_pages_hybrid(
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_indptr=sparse_kv_indptr,
            cpu_to_gpu_slot_map=cpu_to_gpu_map,
            gpu_to_cpu_page_map=gpu_to_cpu_map,
            slot_stamps=self.slot_stamps[local_id],
            set_clock=self.set_clocks[local_id],
            set_version=self.set_versions[local_id],
            set_used_mask=self.set_used_masks[local_id],
            dst_gpu_slots=dst_staging_slots,
            owners_bitmap=self.temp_owners_bitmap,
            evicted_cpu_pages=self.temp_evicted_cpu_pages,
            overflow_flag=self.temp_overflow_flag,
            batch_size=batch_size,
            num_kv_heads=self.head_num,
            max_num_pages=self.max_num_pages,
        )

        # Step 2: Copy kernel
        vortex_torch.cache.copy_kv(
            cpu_k_buffer=cpu_k,
            cpu_v_buffer=cpu_v,
            gpu_k_buffer=gpu_k,
            gpu_v_buffer=gpu_v,
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_indptr=sparse_kv_indptr,
            dst_gpu_slots=dst_staging_slots,
            owners_bitmap=self.temp_owners_bitmap,
            evicted_cpu_pages=self.temp_evicted_cpu_pages,
            page_size=self.page_size,
            batch_size=batch_size,
            num_kv_heads=self.head_num,
            max_num_pages=self.max_num_pages,
        )

        # Check for staging buffer overflow (only when CUDA graph is disabled)
        if self.enable_overflow_check and self.temp_overflow_flag.item() != 0:
            raise RuntimeError(
                f"GPU staging buffer overflow in copy_sparse_kv_to_gpu_with_indptr. "
                f"layer_id={layer_id}, batch_size={batch_size}, "
                f"staging_capacity={self.staging_buffer_capacity}. "
                f"This indicates max_num_reqs exceeds GPU staging buffer capacity."
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
        """Store KV to appropriate cache based on layer type."""
        layer_id = layer.layer_id
        local_id, layer_type = self.layers_mapping[layer_id]

        if layer_type == 'cpu_sparse':
            # CPU sparse layer: store to CPU + update GPU staging if cached
            cpu_k_buffer = self.cache_cpu[local_id]["k"]
            cpu_v_buffer = self.cache_cpu[local_id]["v"]
            gpu_k_staging = self.cache_staging[local_id]["k"]
            gpu_v_staging = self.cache_staging[local_id]["v"]
            cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[local_id]

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
                self.max_page_id_sparse,
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
                "centroids": self.cache_staging[local_id]["centroids"]
            }

            self.sparse_attention.forward_cache(unified_cache, loc, ctx=self.ctx)

        elif layer_type == 'gpu_sparse':
            # GPU sparse layer: store directly to GPU + update landmarks
            cache = self.cache_gpu_sparse[local_id]
            k_buffer = cache["k"]
            v_buffer = cache["v"]
            vortex_torch.cache.set_kv_buffer_launcher(
                k_buffer,
                v_buffer,
                cache_k.contiguous(),
                cache_v.contiguous(),
                loc,
                self.page_size
            )
            # Update landmarks for sparse indexing
            self.sparse_attention.forward_cache(cache, loc, ctx=self.ctx)

        else:  # full attention
            # Full attention layer: store directly to GPU, no landmarks
            k_buffer = self.k_buffer_full[local_id]
            v_buffer = self.v_buffer_full[local_id]
            vortex_torch.cache.set_kv_buffer_launcher(
                k_buffer,
                v_buffer,
                cache_k.contiguous(),
                cache_v.contiguous(),
                loc,
                self.page_size
            )
            # Note: No forward_cache call for full attention layers (no landmarks)

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
        assert k_scale is None
        assert v_scale is None
        assert cache_k.dtype == torch.bfloat16
        assert cache_v.dtype == torch.bfloat16
        assert loc.dtype == torch.int64

        layer_id = layer.layer_id
        local_id, layer_type = self.layers_mapping[layer_id]

        if layer_type == 'cpu_sparse':
            # CPU sparse layer: store to CPU + update GPU staging
            cpu_k_buffer = self.cache_cpu[local_id]["k"]
            cpu_v_buffer = self.cache_cpu[local_id]["v"]
            gpu_k_staging = self.cache_staging[local_id]["k"]
            gpu_v_staging = self.cache_staging[local_id]["v"]
            cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[local_id]
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
                "centroids": self.cache_staging[local_id]["centroids"]
            }

            self.sparse_attention.forward_cache(unified_cache, loc, ctx=self.ctx)

        elif layer_type == 'gpu_sparse':
            # GPU sparse layer: store directly to GPU + update landmarks
            cache = self.cache_gpu_sparse[local_id]
            k_buffer = cache["k"]
            v_buffer = cache["v"]
            vortex_torch.cache.set_kv_buffer_launcher(
                k_buffer,
                v_buffer,
                cache_k.contiguous(),
                cache_v.contiguous(),
                loc,
                self.page_size
            )
            # Update landmarks for sparse indexing
            self.sparse_attention.forward_cache(cache, loc, ctx=self.ctx)

        else:  # full attention
            # Full attention layer: store directly to GPU, no landmarks
            k_buffer = self.k_buffer_full[local_id]
            v_buffer = self.v_buffer_full[local_id]
            vortex_torch.cache.set_kv_buffer_launcher(
                k_buffer,
                v_buffer,
                cache_k.contiguous(),
                cache_v.contiguous(),
                loc,
                self.page_size
            )
            # Note: No forward_cache call for full attention layers (no landmarks)

    def available_size(self) -> int:
        """Return available cache size."""
        return self.size

    def alloc(self, need_size: int) -> List[int]:
        """Allocate cache slots."""
        raise NotImplementedError("Direct allocation not supported for CPU VTX pool")

    def free(self, free_index: torch.Tensor) -> None:
        """Free cache slots."""
        raise NotImplementedError("Direct freeing not supported for CPU VTX pool")
