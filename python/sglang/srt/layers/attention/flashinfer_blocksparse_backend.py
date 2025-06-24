from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.attention.sa_triton_utils import sa_create_flashinfer_kv_indices_triton, q_transpose_triton, o_transpose_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import is_flashinfer_available, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _get_range_buf, get_seq_lens


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class FlashInferBlockSparseAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        # Enforce to use tensor cores to avoid shape incompabilities
        self.decode_use_tensor_cores = True
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not self.skip_prefill
        assert not self.is_multimodal
        assert not model_runner.model_config.is_encoder_decoder
        assert not (model_runner.sliding_window_size is not None)
        assert kv_indptr_buf is None
        assert kv_last_page_len_buf is None
        self.num_wrappers = 1
        self.dispatch_reason = None
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
        self.num_attention_groups = self.num_qo_heads // self.num_kv_heads
        self.page_size = model_runner.page_size
        self.head_dim = model_runner.model_config.head_dim
        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
        ):
            global_config.flashinfer_workspace_size = 512 * 1024 * 1024

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                global_config.flashinfer_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = [
                torch.zeros(
                    (max_bs * self.num_kv_heads + 1,), dtype=torch.int32, device=model_runner.device
                )
            ]
        
        self.kv_last_page_len = torch.ones(
                (max_bs * self.num_kv_heads,), dtype=torch.int32, device=model_runner.device
            )
        
        # The first pointer is for ragged wrapper
        # The second pointer is for paged wrapper
        self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                ),
                torch.zeros(
                    (max_bs * self.num_kv_heads + 1,), dtype=torch.int32, device=model_runner.device
                )
            ]

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        self.prefill_wrappers_paged = [
            BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2")
            ]
        self.decode_wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_tensor_cores=self.decode_use_tensor_cores,
                )
        ]
        # Create indices updater
        self.indices_updater_prefill = FlashInferBlockSparseIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferBlockSparseIndicesUpdaterDecode(model_runner, self)
        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrappers)
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            use_ragged = True
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=None,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged, use_ragged, extend_no_prefix
            )

    
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()
        if self.forward_metadata.extend_no_prefix:
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )


        else:
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

                # q [bsz * num_kv_heads, num_groups, head_dim]
                # k,v [num_pages, 1, head_dim]
                # o [bsz * num_kv_heads, num_groups, head_dim]
                # s [bsz * num_kv_heads, num_groups]
        
                q_transpose = torch.empty(size= \
                (q.shape[0] * self.num_kv_heads, self.num_attention_groups, layer.head_dim),
                dtype=q.dtype, device=q.device)
                o_transpose = torch.empty(size= \
                (q.shape[0] * self.num_kv_heads, self.num_attention_groups, layer.head_dim),
                dtype=q.dtype, device=q.device)
                q_transpose_triton[(forward_batch.batch_size, self.num_kv_heads)](
                    q,
                    q_transpose,
                    self.qo_indptr[0],
                    self.num_kv_heads,
                    self.head_dim,
                    self.num_attention_groups
                )
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q_transpose,
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )
                
                o2 = o2.contiguous()
                
                s_transpose = torch.empty_like(s2)
                
                o_transpose_triton[(forward_batch.batch_size, self.num_kv_heads)](
                    o2,
                    o_transpose,
                    s2,
                    s_transpose,
                    self.qo_indptr[0],
                    self.num_kv_heads,
                    self.head_dim,
                    self.num_attention_groups
                )
                
                # reshape for merging
                o_transpose =  o_transpose.reshape(-1, self.num_qo_heads, layer.head_dim)
                
                s_transpose = s_transpose.reshape(-1, self.num_qo_heads)

                o, _ = merge_state(o1, s1, o_transpose, s_transpose)

        if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):  
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v
                )

        # Call the wrapped function
        o = decode_wrapper.forward(
            q.contiguous().view(-1, self.num_attention_groups, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale,
            v_scale=layer.v_scale,
        )
        
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        
        return 0

        


class FlashInferBlockSparseIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferBlockSparseAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        
        self.num_attention_groups = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend
        self.page_size = self.attn_backend.page_size
        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
        )


    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        
        bs = len(req_pool_indices)
        paged_kernel_lens_trans = torch.repeat_interleave(paged_kernel_lens, 
        repeats=self.num_kv_heads, dim=0)
        
        kv_indptr[1 : self.num_kv_heads * bs + 1] = torch.cumsum(paged_kernel_lens_trans, dim=0)
        kv_indptr = kv_indptr[: self.num_kv_heads * bs + 1]

        kv_indices = torch.zeros(
                    self.num_kv_heads * paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
            )
        sa_create_flashinfer_kv_indices_triton[(bs * self.num_kv_heads,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens_trans,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
                self.page_size,
                self.num_kv_heads
            )
        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:self.num_kv_heads * bs],
            self.num_attention_groups,
            1,
            self.head_dim,
            1,
            data_type=self.data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
        )


class FlashInferBlockSparseIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferBlockSparseAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.num_attention_groups = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend
        self.page_size = self.attn_backend.page_size
        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged
        self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        
        paged_kernel_lens = prefix_lens
        paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        
        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr,
            use_ragged,
            spec_info,
        )


    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: List[torch.Tensor],
        use_ragged: bool,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        bs = len(seq_lens)
        
        assert len(seq_lens) == len(req_pool_indices)
        # Normal extend
        # We need to re-interpret the kv layout
        # Every token is now NUM_KV_HEAD sub-token 
        paged_kernel_lens_trans = torch.repeat_interleave(paged_kernel_lens, 
        repeats=self.num_kv_heads, dim=0)
        
        kv_indptr[1 : self.num_kv_heads * bs + 1] = torch.cumsum(paged_kernel_lens_trans, dim=0)
        kv_indptr = kv_indptr[: self.num_kv_heads * bs + 1]
        
        kv_indices = torch.empty(
                self.num_kv_heads * paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
        )
        # We need to generate the interpreted indices here
        # i.e. kv_{trans} = (kv_{original} // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + head_id * PAGE_SIZE + kv_{original} %  PAGE_SIZE
        # We lanuch bs * self.num_kv_heads triton blocks, each of them is responsible for
        # one kv head of one request
        sa_create_flashinfer_kv_indices_triton[(bs * self.num_kv_heads,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens_trans,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
                self.page_size,
                self.num_kv_heads
            )
        
        
        # qo_indptr[0] is for ragged wrapper, since it processes new-coming tokens
        # i.e. without KV cache, we keep as it is
        
        input_lens = seq_lens - prefix_lens
        qo_indptr[0][1 : bs + 1] = torch.cumsum(input_lens, dim=0)
        qo_indptr_ragged = qo_indptr[0][: bs + 1]
        custom_mask = None
        

        # extend part
        wrapper_ragged.begin_forward(
                qo_indptr_ragged,
                qo_indptr_ragged,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        # cached part
        input_lens_trans = torch.repeat_interleave(input_lens, 
        repeats=self.num_kv_heads, dim=0)
        qo_indptr[1][1 : self.num_kv_heads * bs + 1] = torch.cumsum(input_lens_trans, dim=0)
        qo_indptr_paged = qo_indptr[1][: self.num_kv_heads * bs + 1]
        wrapper_paged.begin_forward(
            qo_indptr_paged,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:self.num_kv_heads * bs],
            self.num_attention_groups,
            1,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=custom_mask,
            non_blocking=True,
        )

