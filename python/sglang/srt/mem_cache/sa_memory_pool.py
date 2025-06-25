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
import abc
import logging
import threading
from enum import IntEnum
from functools import wraps
from typing import List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import triton
import triton.language as tl

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.sa_triton_utils import sa_set_kv_buffer_launcher, update_landmark_launcher
from typing import List, Optional, Tuple, Union
from sglang.srt.utils import (
    debug_timing,
    is_cuda
)

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


"""
Sparse Attention Memory pool.

In addition to Memory Pool in the original SGLang
We 
1) maintain a landmark tensor for every page.
2) internally treat each KV head as a request (as they may have different sparse patterns), 
then we interpert external auguments to the physical address
"""

class BlockSparseTokenToKVPool(KVCache):

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
        self.num_landmarks =  (self.size + 2 * self.page_size - 1) // self.page_size
        self._create_buffers()

        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if is_cuda else None

        k_size, v_size, lmk_size = self.get_memory_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB, Landmark size: {lmk_size / GB:.2f} GB"
        )
        
    def _create_buffers(self):
        with self.memory_saver_adapter.region():
            
            # [size * head_num, 1, head_dim] for each layer
            # [i * head_num, (i+1) * head_num) is the flatten KV cache for one page
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.landmark_buffer = [
                torch.zeros(
                    (self.num_landmarks * self.head_num, 1, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            
            # [size * head_num, 1, head_dim] for each layer
            # [i * head_num, (i+1) * head_num) is the flatten KV cache for one token
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = [
                torch.zeros(
                    ((self.size + self.page_size) * self.head_num, 1, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    ((self.size + self.page_size) * self.head_num, 1, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

            # There is a correspondence between k_buffer and landmark_buffer.
            # E.g., NUM_KV_HEAD = 2, PAGE_SIZE = 4, Original KV index [14], head index [1] 
            # Now, we know this tensor should be stored in page index 14 // 4 = 3, in the second ([1]) sub-page
            # As any page is divided into NUM_KV_HEAD sub-page now
            # The inner page offset is
            # 1 x 4 + 14 % 4 = 6
            # The actual KV index is 3 * (4 * 2) + 6 = 30
            # i.e. kv_{trans} = (kv_{original} // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + head_id * PAGE_SIZE + kv_{original} %  PAGE_SIZE
            # Now we calculate the index of landmark
            # i.e. landmark = (kv_{original} // PAGE_SIZE) * NUM_KV_HEAD + head_id
            # if (kv_{original} + 1) % PAGE_SIZE == 0, then a new block is finished, we can calculate the correnponding value
    def _clear_buffers(self):
        del self.landmark_buffer
        del self.k_buffer
        del self.v_buffer

    
    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        pass

    @debug_timing
    def transfer(self, indices, flat_data): 
        pass

    def transfer_per_layer(self, indices, flat_data, layer_id):
        pass

    def get_key_buffer(self, layer_id: int):
        
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        
        return self.v_buffer[layer_id - self.start_layer]
    
    def get_landmark_buffer(self, layer_id: int):
        
        return self.landmark_buffer[layer_id - self.start_layer]    

    def get_kv_buffer(self, layer_id: int):
        
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor
    ):  
        
        #cache_k/cache_v [nnz, num_kv_heads, head_dim]
        #loc [nnz,]
        #We need to re-interpret the address
        
        layer_id = layer.layer_id
        sa_set_kv_buffer_launcher(
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            cache_k.contiguous(),
            cache_v.contiguous(),
            loc,
            self.page_size
        )
        
        #Directly update the landmark tensor when applicaple (i.e, one page is fullfilled.)
        update_landmark_launcher(
            self.k_buffer[layer_id - self.start_layer],
            self.landmark_buffer[layer_id - self.start_layer],
            loc,
            self.page_size,
            self.head_num,
            self.head_dim
        )

    def get_memory_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        assert hasattr(self, "landmark_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        lmk_size_bytes = 0
        for lmk_cache in self.landmark_buffer:
            lmk_size_bytes += np.prod(lmk_cache.shape) * lmk_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes, lmk_size_bytes