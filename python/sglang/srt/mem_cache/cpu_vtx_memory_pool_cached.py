import logging
from typing import List, Optional, Tuple, Union

import torch
from sglang.srt.mem_cache.cpu_vtx_memory_pool import CPUVTXTokenToKVPool
from sglang.srt.mem_cache.cpu_gpu_copy_kernels import (
    copy_pages_to_staging_slots_dedup,
    store_kv_cpu_and_gpu,
    update_landmark_from_cpu
)

logger = logging.getLogger(__name__)


class CPUVTXTokenToKVPoolCached(CPUVTXTokenToKVPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Calculate max page ID and staging buffer capacity
        max_page_id = (self.size + self.page_size) * self.head_num // self.page_size
        staging_buffer_capacity = self.k_staging_buffer[0].shape[0] // self.page_size

        # Persistent state per layer
        self.cpu_to_gpu_slot_maps = []
        self.gpu_to_cpu_page_maps = []
        self.slot_counters = []

        # Pre-allocated temporary buffers (reused across calls to avoid allocation overhead)
        self.temp_owners_bitmaps = []
        self.temp_staging_slots = []
        self.temp_alloc_counters = []
        self.temp_overflow_flags = []

        for _ in range(self.layer_num):
            # CPU→GPU slot mapping: cpu_page_id → gpu_staging_slot (-1 if not cached)
            cpu_to_gpu_map = torch.full(
                (max_page_id,), -1, dtype=torch.int32, device=self.device
            )
            self.cpu_to_gpu_slot_maps.append(cpu_to_gpu_map)

            # GPU→CPU page mapping: gpu_staging_slot → cpu_page_id (-1 if empty)
            gpu_to_cpu_map = torch.full(
                (staging_buffer_capacity,), -1, dtype=torch.int32, device=self.device
            )
            self.gpu_to_cpu_page_maps.append(gpu_to_cpu_map)

            # Slot counter: tracks how many slots have been allocated
            slot_counter = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.slot_counters.append(slot_counter)

            # Pre-allocate temp buffers sized to max capacity (eliminates allocation on hot path)
            self.temp_owners_bitmaps.append(
                torch.zeros(staging_buffer_capacity, dtype=torch.bool, device=self.device)
            )
            self.temp_staging_slots.append(
                torch.zeros(staging_buffer_capacity, dtype=torch.int32, device=self.device)
            )
            self.temp_alloc_counters.append(
                torch.zeros(1, dtype=torch.int32, device=self.device)
            )
            self.temp_overflow_flags.append(
                torch.zeros(1, dtype=torch.int32, device=self.device)
            )

        self.max_page_id = max_page_id
        self.staging_buffer_capacity = staging_buffer_capacity

        logger.info(
            f"Created CPU VTX pool with persistent cache. "
            f"Max page ID: {max_page_id}, "
            f"Staging buffer capacity: {staging_buffer_capacity} slots per layer"
        )

    def copy_sparse_kv_to_gpu(
        self,
        layer_id: int,
        sparse_indices: torch.Tensor,
    ):
        layer_idx = layer_id - self.start_layer
        k_staging = self.k_staging_buffer[layer_idx]
        v_staging = self.v_staging_buffer[layer_idx]

        cpu_to_gpu_map = self.cpu_to_gpu_slot_maps[layer_idx]
        gpu_to_cpu_map = self.gpu_to_cpu_page_maps[layer_idx]

        # Find available slots using bitmap approach
        # Get GPU slots for current request (pages already cached)
        gpu_slots = cpu_to_gpu_map[sparse_indices]
        gpu_slots_valid = gpu_slots[gpu_slots >= 0]  # Filter out -1 (cache misses)

        # Create bitmap: mark slots that are in use by current request
        temp = torch.zeros(self.staging_buffer_capacity, dtype=torch.bool, device=self.device)
        if gpu_slots_valid.numel() > 0:
            temp[gpu_slots_valid] = True

        # Available slots = slots not in use by current request (can be used for eviction)
        available_slots = torch.where(~temp)[0].to(torch.int32)
        
        # print("cache hits", gpu_slots_valid.numel() / sparse_indices.numel())

        # torch.cuda.synchronize()
        # Copy with persistent bidirectional mapping and available slots
        # import time
        # start_time = time.time()
        staging_slots_all = copy_pages_to_staging_slots_dedup(
            cpu_k_buffer=self.k_buffer[layer_idx],
            cpu_v_buffer=self.v_buffer[layer_idx],
            gpu_k_staging=k_staging,
            gpu_v_staging=v_staging,
            src_page_ids=sparse_indices,
            cpu_to_gpu_slot_map=cpu_to_gpu_map,
            gpu_to_cpu_page_map=gpu_to_cpu_map,
            available_slots=available_slots,
            page_size=self.page_size,
            max_page_id=self.max_page_id,
            owners_bitmap=self.temp_owners_bitmaps[layer_idx],
            dst_staging_slots=self.temp_staging_slots[layer_idx],
            alloc_counter=self.temp_alloc_counters[layer_idx],
            overflow_flag=self.temp_overflow_flags[layer_idx],
        )
        # torch.cuda.synchronize()
        # end_time = time.time()
        # final = (end_time - start_time) * 1000.0
        # print(f"[DEBUG] CPU->GPU sparse KV staging copy with persistent cache took {final:.4f} ms")
        
        return k_staging, v_staging, staging_slots_all

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
        # torch.cuda.synchronize()
        # # Copy with persistent bidirectional mapping and available slots
        # import time
        # start_time = time.time()
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
        # torch.cuda.synchronize()
        # end_time = time.time()
        # final = (end_time - start_time) * 1000.0
        # print(f"[DEBUG] Storage time {final:.4f} ms")
        
        update_landmark_from_cpu(
            cpu_k_buffer=cpu_k_buffer,
            gpu_landmark=self.landmark_buffer[layer_id - self.start_layer],
            loc=loc,
            page_size=self.page_size,
            num_kv_head=self.head_num,
            head_dim=self.head_dim,
        )
