from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union, Dict, Tuple
import torch
import vortex_torch
from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.cache.triton_kernels.paged_decode_int8 import paged_decode_int8
from vortex_torch.cache.triton_kernels.paged_prefill_int8 import dequant_paged_int8_to_bf16
if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.mem_cache.vtx_graph_memory_pool import VTXGraphCachePool
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )

@dataclass
class DecodeMetadata:
    decode_wrappers: Optional[List[BatchDecodeWithPagedKVCacheWrapper]]

@dataclass
class PrefillMetadata:
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class VTXGraphAttnBackend(AttentionBackend):
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
        self.decode_use_tensor_cores = True
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        assert model_runner.sliding_window_size is None
        assert not model_runner.model_config.is_encoder_decoder 
        assert not self.skip_prefill
        assert not self.is_multimodal
        assert kv_indptr_buf is None
        assert kv_last_page_len_buf is None
        self.num_wrappers = 2
        self.dispatch_reason = None

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
        
        self.num_qo_heads = model_runner.model_config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
        self.group_size = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.count = 0
        assert self.q_data_type == torch.bfloat16
        self.is_int8 = (self.data_type == torch.int8)
        self.is_fp8 = (self.data_type in (torch.float8_e4m3fn, torch.float8_e5m2))

        # Assign key configuration and parameters
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.server_args.page_size
        self.layers_skip = model_runner.server_args.vortex_layers_skip

        # ===========================
        # Decode KV-indptr buffers
        # ===========================

        self.kv_indptr_decode = [
            torch.zeros(
                (max_bs * self.num_kv_heads + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
            torch.zeros(
                (max_bs * self.num_kv_heads + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
        ]

        # ===========================
        # KV indices (decode)
        # ===========================

        self.kv_indices_decode = [
            torch.zeros(
                (
                    (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1)
                    // self.page_size,
                ),
                dtype=torch.int32,
                device=model_runner.device
            ),
            torch.zeros(
                (
                    (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1)
                    // self.page_size,
                ),
                dtype=torch.int32,
                device=model_runner.device
            ),
        ]

        # ===========================
        # KV last page length tracking
        # ===========================

        self.kv_last_page_len_decode = torch.ones(
            (max_bs * self.num_kv_heads,),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # Query/Output indptr buffer (ragged only)
        # ===========================

        self.qo_indptr = torch.zeros(
            (max_bs + 1,),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # Sparse prefill buffers
        # ===========================

        topk_budget = (
            model_runner.server_args.vortex_topk_val
            + model_runner.server_args.vortex_page_reserved_bos
            + model_runner.server_args.vortex_page_reserved_eos
        )
        max_sparse_tokens = max_bs * topk_budget * self.page_size
        # Combined buffer holds prefix pages + new tokens for single-wrapper extend
        max_new_tokens_extend = getattr(model_runner, 'max_total_num_tokens', max_sparse_tokens)
        max_combined_tokens = max_sparse_tokens + max_new_tokens_extend
        self.sparse_prefill_k_buf = torch.empty(
            (max_combined_tokens, self.num_kv_heads, self.head_dim),
            dtype=torch.bfloat16,
            device=model_runner.device,
        )
        self.sparse_prefill_v_buf = torch.empty(
            (max_combined_tokens, self.num_kv_heads, self.head_dim),
            dtype=torch.bfloat16,
            device=model_runner.device,
        )
        self.sparse_prefill_kv_indptr = torch.zeros(
            (max_bs + 1,),
            dtype=torch.int32,
            device=model_runner.device,
        )
        # Pre-allocated combined KV indptr buffers for single-wrapper extend
        self.combined_kv_indptr_sparse = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device,
        )
        self.combined_kv_indptr_dense = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device,
        )

        # ===========================
        # Prefill wrapper (ragged only)
        # ===========================

        fmha_backend = "auto"
        if is_sm100_supported():
            fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        if self.is_int8:
            # Int8 path: no FlashInfer decode wrappers; use custom Triton kernels
            self.decode_wrappers = None
            self.max_kv_splits_decode = 8
            max_batch_kv_heads = max_bs * self.num_kv_heads
            # Pre-allocate num_kv_splits buffer (set to max for simplicity)
            self.num_kv_splits_decode_buf = torch.full(
                (max_batch_kv_heads,), self.max_kv_splits_decode,
                dtype=torch.int32, device=model_runner.device
            )
            # Pre-allocate intermediate buffers for decode split reduction
            self.att_out_buf = torch.empty(
                (max_batch_kv_heads, self.num_qo_heads // self.num_kv_heads,
                 self.max_kv_splits_decode, self.head_dim),
                dtype=torch.float32, device=model_runner.device,
            )
            self.att_lse_buf = torch.empty(
                (max_batch_kv_heads, self.num_qo_heads // self.num_kv_heads,
                 self.max_kv_splits_decode),
                dtype=torch.float32, device=model_runner.device,
            )
        else:
            self.decode_wrappers = [
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_tensor_cores=self.decode_use_tensor_cores,
                    ),
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_tensor_cores=self.decode_use_tensor_cores,
                    ),
            ]

        self.sparse_attention = model_runner.sparse_attention
        self.ctx = vortex_torch.indexer.Context()
        self._initialize_graph(model_runner)
        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata: Dict[int, List[BatchDecodeWithPagedKVCacheWrapper]] = {}
        self.plan_graph: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]]
    

    def _initialize_graph(self, model_runner: "ModelRunner") -> None:
        """
        Initialize execution context and warm up kernels/graphs with minimal dummy tensors.

        Expectations:
            - self.head_dim: int > 0
            - self.ctx: provides create / assert_created / profile / summary / execute
            - self.sparse_attention.forward_indexer is callable
            - model_runner.device is a valid torch.device
        """
        # ---- Basic validations ----
        if getattr(self, "ctx", None) is None:
            raise RuntimeError("`self.ctx` is not set. Please construct/inject a context before initialize().")

        if not hasattr(self, "head_dim") or not isinstance(self.head_dim, int) or self.head_dim <= 0:
            raise AttributeError("`self.head_dim` must be a positive integer.")

        device: Optional[torch.device] = getattr(model_runner, "device", None)
        if device is None:
            raise AttributeError("`model_runner.device` is required but missing.")

        indexer = getattr(getattr(self, "sparse_attention", None), "forward_indexer", None)
        if indexer is None or not callable(indexer):
            raise AttributeError("`self.sparse_attention.forward_indexer` is missing or not callable.")

        # ---- Context lifecycle ----
        self.ctx.create(self, model_runner)
        self.ctx.assert_created()
        self.ctx.profile()  # enter 'profile' mode during warm-up

        # ---- Minimal warm-up tensors (placeholders only) ----
        # Q, O, and custom caches (centroids etc.) are always bf16.
        # K/V cache dtype must match the actual storage format.
        dtype = torch.bfloat16
        if self.is_int8:
            kv_store_dtype = torch.int8
        elif self.is_fp8:
            kv_store_dtype = torch.uint8
        else:
            kv_store_dtype = torch.bfloat16

        try:
            with torch.no_grad():
                # Dummy placeholders: used only for kernel / graph warm-up
                q_dummy = as_vtensor(torch.empty((1, self.group_size, self.head_dim), device=device, dtype=dtype), FORMAT.BATCHED)
                o_dummy = as_vtensor(torch.empty((0, 1, 1), device=device, dtype=dtype), FORMAT.RAGGED)
                cache_meta_info = self.sparse_attention.get_cache_meta_info(self.page_size, self.head_dim)

                cache_dummy = {
                        cache_name:  as_vtensor(torch.zeros(
                                (0, cache_shape[0], cache_shape[1]),
                                dtype=kv_store_dtype if cache_name in ("k", "v") else dtype,
                                device=device,
                            ), FORMAT.PAGED)

                        for (cache_name, cache_shape) in cache_meta_info.items()
                    }

                indexer(q_dummy, o_dummy, cache_dummy, ctx=self.ctx)


        except Exception:
            raise

        
        self.ctx.summary()
        self.ctx.execute()



    
    def init_forward_metadata(self, forward_batch: ForwardBatch):

        assert not forward_batch.forward_mode.is_draft_extend()
        assert not forward_batch.forward_mode.is_target_verify()

        if forward_batch.forward_mode.is_decode_or_idle():

            bs = len(forward_batch.req_pool_indices)
            vortex_torch.indexer.utils_sglang.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                ctx=self.ctx
            )

            if self.is_int8:
                # Int8 path: no FlashInfer decode wrappers needed
                self.forward_metadata = DecodeMetadata(decode_wrappers=None)
            else:
                self.decode_wrappers[0].plan(
                    indptr=self.kv_indptr_decode[0][:bs*self.num_kv_heads+1],
                    indices=self.kv_indices_decode[0],
                    last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                )

                self.decode_wrappers[1].plan(
                    indptr=self.kv_indptr_decode[1][:bs*self.num_kv_heads+1],
                    indices=self.kv_indices_decode[1],
                    last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                )
                self.forward_metadata = DecodeMetadata([self.decode_wrappers[0], self.decode_wrappers[1]])
        
        elif forward_batch.forward_mode.is_extend():

            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            bs = len(forward_batch.req_pool_indices)
            input_seq_lens = forward_batch.seq_lens.to(torch.int32) - prefix_lens

            # Compute qo_indptr for ragged layout (cumsum of input_seq_lens)
            self.qo_indptr[0] = 0
            torch.cumsum(input_seq_lens, dim=0, out=self.qo_indptr[1:bs+1])

            if extend_no_prefix:
                # No cached prefix: plan wrapper for self-attention only
                self.prefill_wrapper_ragged.plan(
                    self.qo_indptr[:bs+1],
                    self.qo_indptr[:bs+1],
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    q_data_type=self.q_data_type,
                )
            else:
                # Has cached prefix: prepare decode-style indices for the indexer
                vortex_torch.indexer.utils_sglang.plan_decode(
                    cached_seq_lens=prefix_lens,
                    req_to_token=self.req_to_token,
                    req_indices=forward_batch.req_pool_indices,
                    ctx=self.ctx
                )

                # Pre-compute combined KV indptrs (prefix_tokens + new_tokens)
                # All KV heads of a request share the same page count; use head 0.
                H = self.num_kv_heads
                num_batch_kv = bs * H

                dense_pages_per_req = (
                    self.ctx.dense_kv_indptr[1::H][:bs]
                    - self.ctx.dense_kv_indptr[::H][:bs]
                )
                dense_prefix_tokens = (dense_pages_per_req * self.page_size).to(torch.int32)
                self.combined_kv_indptr_dense[0] = 0
                torch.cumsum(
                    dense_prefix_tokens + input_seq_lens, dim=0,
                    out=self.combined_kv_indptr_dense[1:bs + 1]
                )

                sparse_pages_per_req = (
                    self.ctx.sparse_kv_indptr[1::H][:bs]
                    - self.ctx.sparse_kv_indptr[::H][:bs]
                )
                sparse_prefix_tokens = (sparse_pages_per_req * self.page_size).to(torch.int32)
                self.combined_kv_indptr_sparse[0] = 0
                torch.cumsum(
                    sparse_prefix_tokens + input_seq_lens, dim=0,
                    out=self.combined_kv_indptr_sparse[1:bs + 1]
                )

            self.forward_metadata = PrefillMetadata(extend_no_prefix)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        pass
    
    
    def capture_plan_graph(
        self, 
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int):
        
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        assert bs == num_tokens

        if forward_mode.is_decode_or_idle():
            vortex_torch.indexer.utils_sglang.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )

            if self.is_int8:
                # Int8 path: no FlashInfer decode wrappers
                self.decode_cuda_graph_metadata[bs] = None
                self.forward_metadata = DecodeMetadata(decode_wrappers=None)
            else:
                decode_wrappers = [
                    BatchDecodeWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            use_cuda_graph=True,
                            use_tensor_cores=self.decode_use_tensor_cores,
                            paged_kv_indptr_buffer=self.kv_indptr_decode[0][:bs*self.num_kv_heads + 1],
                            paged_kv_indices_buffer=self.kv_indices_decode[0],
                            paged_kv_last_page_len_buffer=self.kv_last_page_len_decode[
                                :bs*self.num_kv_heads
                            ],
                        ),

                    BatchDecodeWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            use_cuda_graph=True,
                            use_tensor_cores=self.decode_use_tensor_cores,
                            paged_kv_indptr_buffer=self.kv_indptr_decode[1][:bs*self.num_kv_heads + 1],
                            paged_kv_indices_buffer=self.kv_indices_decode[1],
                            paged_kv_last_page_len_buffer=self.kv_last_page_len_decode[
                                :bs*self.num_kv_heads
                            ],
                        ),

                ]

                decode_wrappers[0].plan(
                    indptr=self.kv_indptr_decode[0][:bs*self.num_kv_heads+1],
                    indices=self.kv_indices_decode[0],
                    last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                )

                decode_wrappers[1].plan(
                    indptr=self.kv_indptr_decode[1][:bs*self.num_kv_heads+1],
                    indices=self.kv_indices_decode[1],
                    last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                )

                self.decode_cuda_graph_metadata[bs] = decode_wrappers
                self.forward_metadata = DecodeMetadata(decode_wrappers)
        else:
            raise NotImplementedError
            

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        assert forward_mode.is_decode_or_idle()

        vortex_torch.indexer.utils_sglang.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )

        if self.is_int8:
            # Int8 path: plan_decode already filled indptr/indices; no FlashInfer plan needed
            pass
        else:
            self.decode_cuda_graph_metadata[bs][0].plan(
                indptr=self.kv_indptr_decode[0][:bs*self.num_kv_heads+1],
                indices=self.kv_indices_decode[0],
                last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

            self.decode_cuda_graph_metadata[bs][1].plan(
                indptr=self.kv_indptr_decode[1][:bs*self.num_kv_heads+1],
                indices=self.kv_indices_decode[1],
                last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):

        assert isinstance(forward_batch.token_to_kv_pool, VTXGraphCachePool)
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc
        logits_soft_cap = layer.logit_cap
        q = q.contiguous()

        # Save KV to cache FIRST so centroids are available for topK scoring
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        q_view = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k_view = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_view = v.view(-1, layer.tp_v_head_num, layer.head_dim)

        if self.forward_metadata.extend_no_prefix:
            # No cached prefix: self-attention only (dense, causal)
            o = self.prefill_wrapper_ragged.forward(
                q_view, k_view, v_view,
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        else:
            # Has cached prefix: gather prefix pages + cat new tokens, single causal wrapper
            bs = len(forward_batch.req_pool_indices)
            use_sparsity = (layer.layer_id not in self.layers_skip)
            cache = forward_batch.token_to_kv_pool.get_cache(layer.layer_id)

            if use_sparsity:
                # Compute per-request Q summary for page scoring
                q_summary = self._compute_query_summary(q_view, bs)
                # Run forward_indexer -> topK page selection
                self.sparse_attention.forward_indexer(
                    q=q_summary,
                    o=self.kv_indices_decode[1],
                    cache=cache,
                    ctx=self.ctx
                )
                kv_indptr = self.ctx.sparse_kv_indptr
                kv_indices = self.ctx.sparse_kv_indices
                combined_kv_indptr = self.combined_kv_indptr_sparse
            else:
                kv_indptr = self.ctx.dense_kv_indptr
                kv_indices = self.ctx.dense_kv_indices
                combined_kv_indptr = self.combined_kv_indptr_dense

            # Gather prefix pages + cat new tokens into combined ragged buffer
            total_tokens = self._gather_prefix_and_cat_new(
                cache, kv_indptr, kv_indices, k_view, v_view, bs
            )

            # Plan single wrapper with combined kv_indptr (prefix + new)
            self.prefill_wrapper_ragged.plan(
                self.qo_indptr[:bs+1],
                combined_kv_indptr[:bs+1],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

            # Single causal forward: lower-right aligned mask gives
            # full attention to prefix + causal within new tokens
            o = self.prefill_wrapper_ragged.forward(
                q_view,
                self.sparse_prefill_k_buf[:total_tokens],
                self.sparse_prefill_v_buf[:total_tokens],
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _compute_query_summary(self, q_view: torch.Tensor, bs: int) -> torch.Tensor:
        """
        Compute per-request mean Q, shaped for the indexer API.

        Args:
            q_view: [total_tokens, num_qo_heads, head_dim]
            bs: batch size

        Returns:
            [bs * num_kv_heads, group_size, head_dim] tensor
        """
        qo_indptr = self.qo_indptr[:bs + 1]
        total_tokens = int(qo_indptr[bs].item())

        # Build segment IDs from qo_indptr for scatter_add
        seg_ids = torch.zeros(total_tokens, dtype=torch.int64, device=q_view.device)
        for i in range(bs):
            start = int(qo_indptr[i].item())
            end = int(qo_indptr[i + 1].item())
            seg_ids[start:end] = i

        # Scatter-add then divide by count for per-request mean
        summaries = torch.zeros(
            bs, self.num_qo_heads, self.head_dim,
            device=q_view.device, dtype=q_view.dtype
        )
        summaries.scatter_add_(
            0,
            seg_ids.unsqueeze(-1).unsqueeze(-1).expand_as(q_view),
            q_view,
        )
        counts = (qo_indptr[1:bs + 1] - qo_indptr[:bs]).float().unsqueeze(-1).unsqueeze(-1)
        summaries = (summaries / counts.clamp(min=1)).to(q_view.dtype)

        # Reshape: [bs, num_kv_heads, group_size, head_dim] -> [bs*num_kv_heads, group_size, head_dim]
        summaries = summaries.view(bs, self.num_kv_heads, self.group_size, self.head_dim)
        return summaries.reshape(bs * self.num_kv_heads, self.group_size, self.head_dim).contiguous()

    def _gather_prefix_and_cat_new(
        self,
        cache: Dict[str, torch.Tensor],
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        bs: int,
    ) -> int:
        """
        Gather prefix pages from paged cache and concatenate new K/V tokens
        into a single ragged buffer for combined causal attention.

        Pages are stored per-head: kv_indptr groups as [req0_h0, req0_h1, ..., req1_h0, ...].
        This method reorganizes per-head pages into multi-head ragged layout
        [total_tokens, num_kv_heads, head_dim] using vectorized reshape+permute
        (no inner per-head loop), then appends new tokens after each request's prefix.

        Args:
            cache: layer cache dict with "k", "v" (and "k_scale", "v_scale" for int8)
            kv_indptr: CSR indptr for page groups [bs * num_kv_heads + 1]
            kv_indices: page IDs for each group
            k_new: new K tokens [total_new, num_kv_heads, head_dim]
            v_new: new V tokens [total_new, num_kv_heads, head_dim]
            bs: batch size

        Returns:
            total number of KV tokens written to the ragged buffer
        """
        H = self.num_kv_heads
        D = self.head_dim
        PS = self.page_size
        num_batch_kv = bs * H
        total_pages = int(kv_indptr[num_batch_kv].item())
        qo_indptr = self.qo_indptr

        if total_pages == 0:
            # No prefix pages, just copy new tokens
            total_new = int(qo_indptr[bs].item())
            if total_new > 0:
                self.sparse_prefill_k_buf[:total_new] = k_new[:total_new]
                self.sparse_prefill_v_buf[:total_new] = v_new[:total_new]
            return total_new

        selected_page_ids = kv_indices[:total_pages]

        # Gather pages from paged cache, handling dtype
        if self.is_int8:
            gathered_k = dequant_paged_int8_to_bf16(
                cache["k"], cache["k_scale"],
                selected_page_ids, self.page_size, self.head_dim,
            )
            gathered_v = dequant_paged_int8_to_bf16(
                cache["v"], cache["v_scale"],
                selected_page_ids, self.page_size, self.head_dim,
            )
        elif self.is_fp8:
            gathered_k = cache["k"][selected_page_ids].view(self.data_type).to(torch.bfloat16)
            gathered_v = cache["v"][selected_page_ids].view(self.data_type).to(torch.bfloat16)
        else:
            gathered_k = cache["k"][selected_page_ids]  # [total_pages, page_size, head_dim]
            gathered_v = cache["v"][selected_page_ids]

        # Reassemble per-head pages into multi-head ragged layout + cat new tokens.
        # Per request: [H*P, PS, D] → reshape [H, P, PS, D] → permute [P, PS, H, D]
        # → reshape [T, H, D], then append new tokens.
        offset = 0
        for i in range(bs):
            h0_start = int(kv_indptr[i * H].item())
            h0_end = int(kv_indptr[i * H + 1].item())
            P = h0_end - h0_start  # pages per head for this request
            T = P * PS  # prefix tokens per head

            if T > 0:
                req_end = int(kv_indptr[(i + 1) * H].item())
                # Vectorized: [H*P, PS, D] → [H, P, PS, D] → [P, PS, H, D] → [T, H, D]
                self.sparse_prefill_k_buf[offset:offset + T] = (
                    gathered_k[h0_start:req_end]
                    .view(H, P, PS, D)
                    .permute(1, 2, 0, 3)
                    .reshape(T, H, D)
                )
                self.sparse_prefill_v_buf[offset:offset + T] = (
                    gathered_v[h0_start:req_end]
                    .view(H, P, PS, D)
                    .permute(1, 2, 0, 3)
                    .reshape(T, H, D)
                )

            # Cat new tokens for this request
            q_start = int(qo_indptr[i].item())
            q_end = int(qo_indptr[i + 1].item())
            new_len = q_end - q_start
            if new_len > 0:
                self.sparse_prefill_k_buf[offset + T:offset + T + new_len] = k_new[q_start:q_end]
                self.sparse_prefill_v_buf[offset + T:offset + T + new_len] = v_new[q_start:q_end]

            offset += T + new_len

        return offset

    def _forward_decode_int8(
        self,
        q: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        layer: RadixAttention,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        bs: int,
    ) -> torch.Tensor:
        """Int8 decode attention using custom Triton kernel."""
        q = q.view(-1, self.group_size, layer.head_dim).contiguous()

        # Int8 KV buffers: [num_pages, page_size, head_dim] flat
        cache_k_int8 = cache["k"]
        cache_v_int8 = cache["v"]
        k_scale = cache["k_scale"]
        v_scale = cache["v_scale"]

        o = torch.empty_like(q)

        num_batch_kv = bs * self.num_kv_heads
        paged_decode_int8(
            q=q,
            k_buffer=cache_k_int8,
            v_buffer=cache_v_int8,
            k_scale_buffer=k_scale,
            v_scale_buffer=v_scale,
            o=o,
            kv_indptr=kv_indptr[:num_batch_kv + 1],
            kv_indices=kv_indices,
            last_page_len=self.kv_last_page_len_decode[:num_batch_kv],
            num_kv_splits=self.num_kv_splits_decode_buf[:num_batch_kv],
            max_kv_splits=self.max_kv_splits_decode,
            sm_scale=layer.scaling,
            page_size=self.page_size,
            logit_cap=layer.logit_cap if layer.logit_cap is not None else 0.0,
            att_out=self.att_out_buf[:num_batch_kv],
            att_lse=self.att_lse_buf[:num_batch_kv],
        )

        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """
        Decode-time forward pass with optional sparse attention.
        Expects KV to be sourced from token_to_kv_pool; can also save new KV.
        """

        # Sanity checks and setup
        assert isinstance(forward_batch.token_to_kv_pool, VTXGraphCachePool)
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc

        # Optionally write incoming K/V to decode cache
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Read Cache from memory pool
        cache = forward_batch.token_to_kv_pool.get_cache(layer.layer_id)
        bs = len(forward_batch.req_pool_indices)

        # Decide whether to use sparsity on this layer
        use_sparsity = (layer.layer_id not in self.layers_skip)

        if self.is_int8:
            # ---- Int8 decode path ----
            if use_sparsity:
                q_grouped = q.view(-1, self.group_size, layer.head_dim).contiguous()
                # Build sparse indices (indexer writes to kv_indices_decode[1])
                self.sparse_attention.forward_indexer(
                    q=q_grouped,
                    o=self.kv_indices_decode[1],
                    cache=cache,
                    ctx=self.ctx
                )
                o = self._forward_decode_int8(
                    q, cache, layer,
                    kv_indptr=self.kv_indptr_decode[1],
                    kv_indices=self.kv_indices_decode[1],
                    bs=bs,
                )
            else:
                o = self._forward_decode_int8(
                    q, cache, layer,
                    kv_indptr=self.kv_indptr_decode[0],
                    kv_indices=self.kv_indices_decode[0],
                    bs=bs,
                )
        else:
            # ---- bf16/fp8 decode path (FlashInfer) ----
            if self.is_fp8:
                # uint8 storage → view as native fp8 dtype for FlashInfer
                cache_k = cache["k"].view(self.data_type).view(-1, self.page_size, 1, self.head_dim)
                cache_v = cache["v"].view(self.data_type).view(-1, self.page_size, 1, self.head_dim)
            else:
                cache_k = cache["k"].view(-1, self.page_size, 1, self.head_dim)
                cache_v = cache["v"].view(-1, self.page_size, 1, self.head_dim)

            if use_sparsity:
                # Prepare Q in grouped shape expected by sparse path
                q = q.view(-1, self.group_size, layer.head_dim).contiguous()

                # Build sparse indices into paged KV buffers
                self.sparse_attention.forward_indexer(
                    q=q,
                    o=self.forward_metadata.decode_wrappers[1]._paged_kv_indices_buf,
                    cache=cache,
                    ctx=self.ctx
                )

                # Sparse attention compute
                o = self.forward_metadata.decode_wrappers[1].forward(
                    q,
                    (cache_k, cache_v),
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )

            else:
                # Dense attention path
                o = self.forward_metadata.decode_wrappers[0].forward(
                    q.contiguous().view(-1, self.group_size, layer.head_dim),
                    (cache_k, cache_v),
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )

        # Restore to merged head dimension
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)