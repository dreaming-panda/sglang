import logging

import torch
from sglang.srt.mem_cache.cpu_vtx_memory_pool import CPUVTXTokenToKVPool
from sglang.srt.mem_cache.cpu_gpu_copy_kernels import copy_pages_to_staging_slots_dedup

logger = logging.getLogger(__name__)


class CPUVTXTokenToKVPoolCached(CPUVTXTokenToKVPool):
    def copy_sparse_kv_to_gpu(
        self,
        layer_id: int,
        sparse_indices: torch.Tensor,
    ):
        layer_idx = layer_id - self.start_layer
        k_staging = self.k_staging_buffer[layer_idx]
        v_staging = self.v_staging_buffer[layer_idx]
        
        max_page_id = (self.size + self.page_size) * self.head_num // self.page_size

        staging_slots_all = copy_pages_to_staging_slots_dedup(
            cpu_k_buffer=self.k_buffer[layer_idx],
            cpu_v_buffer=self.v_buffer[layer_idx],
            gpu_k_staging=k_staging,
            gpu_v_staging=v_staging,
            src_page_ids=sparse_indices,
            page_size=self.page_size,
            max_page_id=max_page_id,
        )

        return k_staging, v_staging, staging_slots_all
