import logging
from typing import List, Optional, Tuple, Union

import torch
from sglang.srt.mem_cache.cpu_vtx_memory_pool import CPUVTXTokenToKVPool
from sglang.srt.mem_cache.cpu_gpu_copy_kernels import update_landmark_from_cpu, store_kv_cpu_and_gpu
import vortex_C

logger = logging.getLogger(__name__)


class CPUVTXTokenToKVPoolCached(CPUVTXTokenToKVPool):
    def __init__(self, *args, eviction_policy="lru", **kwargs):
        super().__init__(*args, **kwargs)

        # Eviction policy: "random" or "lru"
        self.eviction_policy = eviction_policy

        # Calculate max page ID and staging buffer capacity
        # Using paged format: [num_pages, page_size, 1, head_dim]
        max_page_id = self.num_pages
        staging_buffer_capacity = self.k_staging_buffer[0].shape[0]  # First dim is num_pages

        # Persistent state per layer
        self.cpu_to_gpu_slot_maps = []
        self.gpu_to_cpu_page_maps = []

        # LRU-specific state (only initialized if using LRU policy)
        self.slot_ages = []

        for layer_idx in range(self.layer_num):
            # CPU→GPU slot mapping: cpu_page_id → gpu_staging_slot (-1 if not cached)
            cpu_to_gpu_map = torch.full(
                (max_page_id,), -1, dtype=torch.int32, device=self.device
            ).contiguous()
            self.cpu_to_gpu_slot_maps.append(cpu_to_gpu_map)

            # GPU→CPU page mapping: gpu_staging_slot → cpu_page_id (-1 if empty)
            gpu_to_cpu_map = torch.full(
                (staging_buffer_capacity,), -1, dtype=torch.int32, device=self.device
            ).contiguous()
            self.gpu_to_cpu_page_maps.append(gpu_to_cpu_map)

            if self.eviction_policy == "lru":
                slot_ages = torch.zeros(staging_buffer_capacity, dtype=torch.uint8, device=self.device).contiguous()
                self.slot_ages.append(slot_ages)

        # Calculate max_num_pages for CUDA graph (fixed grid size)
        # This is the maximum possible sparse pages: max_batch_size * num_kv_heads * kv_budget
        # We use staging_buffer_capacity as a safe upper bound
        self.max_num_pages = staging_buffer_capacity

        # Pre-allocate temp buffers sized to max capacity (eliminates allocation on hot path)
        self.temp_owners_bitmap = torch.zeros(self.max_num_pages, dtype=torch.bool, device=self.device).contiguous()
        self.temp_staging_slots = torch.zeros(self.max_num_pages, dtype=torch.int32, device=self.device).contiguous()
        self.temp_overflow_flag = torch.zeros(1, dtype=torch.int32, device=self.device).contiguous()
        self.temp_slots_used_bitmap = torch.zeros(staging_buffer_capacity, dtype=torch.bool, device=self.device).contiguous()
        self.temp_needs_eviction_bitmap = torch.zeros(self.max_num_pages, dtype=torch.bool, device=self.device).contiguous()
        self.temp_evicted_cpu_pages = torch.full((self.max_num_pages,), -1, dtype=torch.int32, device=self.device).contiguous()

        self.max_page_id = max_page_id
        self.staging_buffer_capacity = staging_buffer_capacity

        logger.info(
            f"Created CPU VTX pool with persistent cache. "
            f"Eviction policy: {self.eviction_policy}, "
            f"Max page ID: {max_page_id}, "
            f"Staging buffer capacity: {staging_buffer_capacity} slots per layer, "
            f"Max num pages (CUDA graph): {self.max_num_pages}"
        )

    def copy_sparse_kv_to_gpu(
        self,
        layer_id: int,
        sparse_kv_indices: torch.Tensor,
        sparse_kv_indptr: torch.Tensor,
        batch_size: int,
        dst_kv_indices=None,
    ):
        """
        CUDA graph compatible sparse KV copy.

        Args:
            layer_id: Layer index
            sparse_kv_indices: Full sparse indices tensor (not sliced)
            sparse_kv_indptr: Indptr tensor of shape [batch_size * num_kv_heads + 1]
                              The last element contains the actual number of pages
            batch_size: Current batch size

        Returns:
            k_staging: GPU K staging buffer
            v_staging: GPU V staging buffer
            dst_staging_slots: GPU slots where pages were placed
        """
        layer_idx = layer_id - self.start_layer
        k_staging = self.k_staging_buffer[layer_idx]
        v_staging = self.v_staging_buffer[layer_idx]

        cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[layer_idx]
        gpu_to_cpu_map = self.gpu_to_cpu_page_maps[layer_idx]
        
        if dst_kv_indices is None:
            dst_staging_slots = self.temp_staging_slots
        else:
            dst_staging_slots = dst_kv_indices

        if self.eviction_policy == "lru":
            # Use CUDA graph compatible kernel
            # No .item() calls, no tensor slicing - all bounds checking done on GPU
            vortex_C.copy_sparse_kv_to_gpu_with_indptr(
                cpu_k_buffer=self.k_buffer[layer_idx],
                cpu_v_buffer=self.v_buffer[layer_idx],
                gpu_k_buffer=k_staging,
                gpu_v_buffer=v_staging,
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

        return k_staging, v_staging, dst_staging_slots

    def copy_sparse_kv_to_gpu_legacy(
        self,
        layer_id: int,
        sparse_indices: torch.Tensor,
    ):
        """
        Legacy API for backwards compatibility (NOT CUDA graph compatible).
        Use copy_sparse_kv_to_gpu with indptr for CUDA graph support.
        """
        layer_idx = layer_id - self.start_layer
        k_staging = self.k_staging_buffer[layer_idx]
        v_staging = self.v_staging_buffer[layer_idx]

        cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[layer_idx]
        gpu_to_cpu_map = self.gpu_to_cpu_page_maps[layer_idx]

        num_pages = sparse_indices.size(0)
        dst_staging_slots = self.temp_staging_slots[:num_pages]

        if self.eviction_policy == "lru":
            vortex_C.copy_sparse_kv_to_gpu(
                cpu_k_buffer=self.k_buffer[layer_idx],
                cpu_v_buffer=self.v_buffer[layer_idx],
                gpu_k_buffer=k_staging,
                gpu_v_buffer=v_staging,
                src_page_ids=sparse_indices,
                cpu_to_gpu_slot_map=cpu_to_gpu_map,
                gpu_to_cpu_page_map=gpu_to_cpu_map,
                slot_ages=self.slot_ages[layer_idx],
                dst_gpu_slots=dst_staging_slots,
                owners_bitmap=self.temp_owners_bitmap[:num_pages],
                slots_used_bitmap=self.temp_slots_used_bitmap,
                needs_eviction_bitmap=self.temp_needs_eviction_bitmap[:num_pages],
                evicted_cpu_pages=self.temp_evicted_cpu_pages[:num_pages],
                overflow_flag=self.temp_overflow_flag,
                page_size=self.page_size
            )

        return k_staging, v_staging, dst_staging_slots

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer

        cpu_k_buffer = self.k_buffer[layer_idx]
        cpu_v_buffer = self.v_buffer[layer_idx]
        gpu_k_staging = self.k_staging_buffer[layer_idx]
        gpu_v_staging = self.v_staging_buffer[layer_idx]
        cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[layer_idx]

        # Store to CPU and update GPU staging buffer if page is cached
        store_kv_cpu_and_gpu(
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

        update_landmark_from_cpu(
            cpu_k_buffer=cpu_k_buffer,
            gpu_landmark=self.landmark_buffer[layer_id - self.start_layer],
            loc=loc,
            page_size=self.page_size,
            num_kv_head=self.head_num,
            head_dim=self.head_dim,
        )

    def set_kv_buffer_decode(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        layer_id = layer.layer_id
        layer_idx = layer_id - self.start_layer

        cpu_k_buffer = self.k_buffer[layer_idx]
        cpu_v_buffer = self.v_buffer[layer_idx]
        gpu_k_staging = self.k_staging_buffer[layer_idx]
        gpu_v_staging = self.v_staging_buffer[layer_idx]
        cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[layer_idx]
        gpu_landmark_buffer = self.landmark_buffer[layer_idx]

        vortex_C.store_kv_unified(
            cpu_k_buffer, cpu_v_buffer, gpu_k_staging, gpu_v_staging,
            cache_k.contiguous(), cache_v.contiguous(),
            loc, cpu_to_gpu_map, gpu_landmark_buffer, self.page_size
        )
