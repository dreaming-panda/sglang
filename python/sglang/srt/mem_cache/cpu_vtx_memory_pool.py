import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.utils import debug_timing, is_cuda
from sglang.srt.mem_cache.cpu_gpu_copy_kernels import (
    copy_sparse_kv_cpu_to_gpu,
    copy_sparse_kv_cpu_to_gpu_tiled,
    store_kv_gpu_to_cpu,
    update_landmark_from_cpu,
    naive_copy,
)

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()

"""
CPU-based Sparse Attention Memory pool.

Differs from VTXTokenToKVPool in that:
1) KV cache is stored on CPU instead of GPU
2) During prefill, KV is computed on GPU then moved to CPU, so at most 1 layer at the gpu
3) During decode, sparse KV indices are computed, then relevant KV pages are
   copied from CPU to a GPU staging buffer using a custom kernel
4) Landmarks are still maintained on GPU for sparse selection
"""


class CPUVTXTokenToKVPool(KVCache):

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
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        layer_skips: Optional[List[int]] = [],
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
        self.layer_skips = layer_skips

        self._create_buffers()
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None

        k_size, v_size = self.get_kv_size_bytes()
        landmark_size = self.get_landmark_size_bytes()
        # print(
        #     f"CPU KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB (CPU), "
        #     f"V size: {v_size / GB:.2f} GB (CPU), Landmark size: {landmark_size / GB:.2f} GB (GPU)."
        # )

        # Note: Only landmarks use GPU memory, KV is on CPU
        self.mem_usage = (landmark_size + k_size + v_size) / GB  # GPU memory usage

        assert self.dtype == torch.bfloat16
        assert self.store_dtype == torch.bfloat16

    def _create_buffers(self):
        # KV buffers are stored on CPU
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            torch.zeros(
                ((self.size + self.page_size) * self.head_num, 1, self.head_dim),
                dtype=self.store_dtype,
                device='cpu',  # CPU storage
                pin_memory=True,  # Pinned memory for faster GPU transfers
            )
            for _ in range(self.layer_num - len(self.layer_skips) if self.layer_skips else self.layer_num)
        ]
        self.v_buffer = [
            torch.zeros(
                ((self.size + self.page_size) * self.head_num, 1, self.head_dim),
                dtype=self.store_dtype,
                device='cpu',  # CPU storage
                pin_memory=True,  # Pinned memory for faster GPU transfers
            )
            for _ in range(self.layer_num - len(self.layer_skips) if self.layer_skips else self.layer_num)
        ]

        # Landmarks stay on GPU for sparse selection
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.landmark_buffer = [
                torch.zeros(
                    (
                    ((self.size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size,
                    1,
                    self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,  # GPU storage
                )
                for _ in range(self.layer_num)
            ]
            
            self.k_staging_buffer = [
                torch.zeros(
                    (10 * 34 * self.page_size * self.head_num, 1, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_staging_buffer = [
                torch.zeros(
                    (10 * 34 * self.page_size * self.head_num, 1, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.landmark_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_staging_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_staging_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    def get_landmark_size_bytes(self):
        assert hasattr(self, "landmark_buffer")
        landmark_size_bytes = 0
        for landmark_cache in self.landmark_buffer:
            landmark_size_bytes += np.prod(landmark_cache.shape) * landmark_cache.dtype.itemsize

        return landmark_size_bytes

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_landmark_buffer(self, layer_id: int):
        return self.landmark_buffer[layer_id - self.start_layer]

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
        """
        During prefill: writes K/V to CPU memory.
        This is called from GPU, so we need to:
        1. Compute K/V on GPU
        2. Transfer to CPU storage
        3. Update landmarks on GPU
        """
        assert layer_id_override is None
        assert k_scale is None
        assert v_scale is None
        assert cache_k.dtype == torch.bfloat16
        assert cache_v.dtype == torch.bfloat16
        assert loc.dtype == torch.int64

        layer_id = layer.layer_id

        cpu_k_buffer = self.k_buffer[layer_id - self.start_layer]
        cpu_v_buffer = self.v_buffer[layer_id - self.start_layer]

        # Step 1: Store K/V to CPU using Triton kernel
        store_kv_gpu_to_cpu(
            cpu_k_buffer=cpu_k_buffer,
            cpu_v_buffer=cpu_v_buffer,
            new_k=cache_k.contiguous(),
            new_v=cache_v.contiguous(),
            loc=loc,
            page_size=self.page_size,
        )

        # Step 2: Update landmarks on GPU
        # Use our custom kernel that loads K pages from CPU and computes landmarks
        update_landmark_from_cpu(
            cpu_k_buffer=cpu_k_buffer,
            gpu_landmark=self.landmark_buffer[layer_id - self.start_layer],
            loc=loc,
            page_size=self.page_size,
            num_kv_head=self.head_num,
            head_dim=self.head_dim,
        )

    def copy_sparse_kv_to_gpu(
        self,
        layer_id: int,
        sparse_indices: torch.Tensor,
        sparse_indptr: torch.Tensor,
        bs: int,
    ):
        layer_idx = layer_id - self.start_layer
        k_staging = self.k_staging_buffer[layer_idx]
        v_staging = self.v_staging_buffer[layer_idx]

        num_sparse_pages = sparse_indptr[bs * self.head_num].item()
        required_tokens = num_sparse_pages * self.page_size
        current_tokens = k_staging.shape[0]

        if required_tokens > current_tokens:
            new_tokens = required_tokens
            new_shape = (new_tokens, 1, self.head_dim)
            k_staging = torch.empty(
                new_shape,
                dtype=self.store_dtype,
                device=self.device,
            )
            v_staging = torch.empty(
                new_shape,
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_staging_buffer[layer_idx] = k_staging
            self.v_staging_buffer[layer_idx] = v_staging

        assert k_staging.is_contiguous()
        assert v_staging.is_contiguous()

        # Use Triton kernel for efficient CPU->GPU sparse copy
        # sparse_indices already contains per-head page indices from Vortex API
        import time

        # start_time_1 = time.time()
        copy_sparse_kv_cpu_to_gpu(
            cpu_k_buffer=self.k_buffer[layer_id - self.start_layer],
            cpu_v_buffer=self.v_buffer[layer_id - self.start_layer],
            gpu_k_staging=k_staging,
            gpu_v_staging=v_staging,
            sparse_indices=sparse_indices,
            page_size=self.page_size,
        )
        # end_time_1 = time.time()
        
        # start_time_2 = time.time()
        # copy_sparse_kv_cpu_to_gpu_tiled(
        #     cpu_k_buffer=self.k_buffer[layer_id - self.start_layer],
        #     cpu_v_buffer=self.v_buffer[layer_id - self.start_layer],
        #     gpu_k_staging=k_staging,
        #     gpu_v_staging=v_staging,
        #     sparse_indices=sparse_indices,
        #     page_size=self.page_size,
        # )
        # end_time_2 = time.time()
        
        # first_time = end_time_1 - start_time_1
        # second_time = end_time_2 - start_time_2
        # if second_time < first_time:
        #     k_staging, v_staging = self.k_staging_buffer[layer_idx], self.v_staging_buffer[layer_idx]
        #     print("Using tiled version")
        # else:
        #     print("Using original version")
        
        
        # naive_copy(
        #     cpu_k=self.k_buffer[layer_id - self.start_layer],
        #     cpu_v=self.v_buffer[layer_id - self.start_layer],
        #     gpu_k_staging=k_staging,
        #     gpu_v_staging=v_staging,
        #     sparse_indices=sparse_indices,
        #     page_size=self.page_size,
        # )
        return k_staging, v_staging

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError("CPU-based KV cache movement not yet implemented")

    # Disaggregation methods - not implemented for CPU version
    def get_contiguous_buf_infos(self):
        raise NotImplementedError("Disaggregation not supported for CPU KV cache")

    def maybe_get_custom_mem_pool(self):
        return None

    def get_cpu_copy(self, indices):
        raise NotImplementedError

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError

    def get_flat_data(self, indices):
        raise NotImplementedError

    def transfer(self, indices, flat_data):
        raise NotImplementedError

    def transfer_per_layer(self, indices, flat_data, layer_id):
        raise NotImplementedError
