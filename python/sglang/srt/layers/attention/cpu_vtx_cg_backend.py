from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import vortex_torch
from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.cache.triton_kernels.paged_decode_int8 import paged_decode_int8
from vortex_torch.cache.triton_kernels.paged_prefill_int8 import dequant_paged_int8_to_bf16

if os.environ.get("SGLANG_ENABLE_TORCH_COMPILE") == "1":
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

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state


@dataclass
class DecodeMetadata:
    decode_wrappers: Optional[List[BatchDecodeWithPagedKVCacheWrapper]]


@dataclass
class PrefillMetadata:
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class CPUVTXCGAttnBackend(AttentionBackend):
    """
    CPU-based Vortex FlashInfer attention backend with CUDA graph support.

    Stores KV cache on CPU and transfers sparse pages to GPU for decode.
    Supports CUDA graph capture and replay.
    Uses ragged prefill (no paged prefill) and int8/fp8 quantization.
    """

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
        # Get layer classifications from model_runner
        self.full_attention_layer_ids = set(getattr(model_runner, 'full_attention_layer_ids', []))
        self.gpu_sparse_layer_ids = set(getattr(model_runner, 'gpu_sparse_layer_ids', []))
        self.cpu_sparse_layer_ids = set(getattr(model_runner, 'cpu_sparse_layer_ids', list(range(model_runner.model_config.num_hidden_layers))))
        # layers_skip: set of layer IDs that use dense (full) attention
        self.layers_skip = self.full_attention_layer_ids

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

        # Separate buffer for staging slot indices (different from CPU page IDs)
        self.staging_kv_indices = torch.zeros(
            (
                (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1)
                // self.page_size,
            ),
            dtype=torch.int32,
            device=model_runner.device
        )

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
        self.sparse_prefill_k_buf = torch.empty(
            (max_sparse_tokens, self.num_kv_heads, self.head_dim),
            dtype=torch.bfloat16,
            device=model_runner.device,
        )
        self.sparse_prefill_v_buf = torch.empty(
            (max_sparse_tokens, self.num_kv_heads, self.head_dim),
            dtype=torch.bfloat16,
            device=model_runner.device,
        )
        self.sparse_prefill_kv_indptr = torch.zeros(
            (max_bs + 1,),
            dtype=torch.int32,
            device=model_runner.device,
        )

        # ===========================
        # Prefill wrappers (ragged only)
        # ===========================

        fmha_backend = "auto"
        if is_sm100_supported():
            fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )
        self.prefill_wrapper_ragged_sparse = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # ===========================
        # Decode wrappers / int8 buffers
        # ===========================

        if self.is_int8:
            # Int8 path: no FlashInfer decode wrappers; use custom Triton kernels
            self.decode_wrappers = None
            self.max_kv_splits_decode = 8
            max_batch_kv_heads = max_bs * self.num_kv_heads
            self.num_kv_splits_decode_buf = torch.full(
                (max_batch_kv_heads,), self.max_kv_splits_decode,
                dtype=torch.int32, device=model_runner.device
            )
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
            # bf16/fp8: 3 FlashInfer decode wrappers
            # Wrapper 0: Dense attention (full attention layers)
            # Wrapper 1: CPU sparse attention (uses staging_kv_indices)
            # Wrapper 2: GPU sparse attention (uses kv_indices_decode[1])
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
        self.decode_cuda_graph_metadata: Dict[int, Optional[List[BatchDecodeWithPagedKVCacheWrapper]]] = {}
        self.plan_graph: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]]

    def _initialize_graph(self, model_runner: "ModelRunner") -> None:
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
        dtype = torch.bfloat16

        try:
            with torch.no_grad():
                q_dummy = as_vtensor(torch.empty((1, self.group_size, self.head_dim), device=device, dtype=dtype), FORMAT.BATCHED)
                o_dummy = as_vtensor(torch.empty((0, 1, 1), device=device, dtype=dtype), FORMAT.RAGGED)
                cache_meta_info = self.sparse_attention.get_cache_meta_info(self.page_size, self.head_dim)

                cache_dummy = {
                    cache_name: as_vtensor(torch.zeros(
                        (0, cache_shape[0], cache_shape[1]),
                        dtype=dtype,
                        device=device,
                    ), FORMAT.PAGED)
                    for (cache_name, cache_shape) in cache_meta_info.items()
                }

                indexer(q_dummy, o_dummy, cache_dummy, ctx=self.ctx)

        except Exception:
            raise

        self.ctx.summary()
        self.ctx.execute()

    # =========================================================================
    # Forward metadata initialization
    # =========================================================================

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
                # Int8: no FlashInfer decode wrappers needed
                self.forward_metadata = DecodeMetadata(decode_wrappers=None)
            else:
                # bf16/fp8: plan 3 FlashInfer decode wrappers
                self.decode_wrappers[0].plan(
                    indptr=self.kv_indptr_decode[0][:bs * self.num_kv_heads + 1],
                    indices=self.kv_indices_decode[0],
                    last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                )

                self.decode_wrappers[1].plan(
                    indptr=self.kv_indptr_decode[1][:bs * self.num_kv_heads + 1],
                    indices=self.staging_kv_indices,
                    last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                )

                self.decode_wrappers[2].plan(
                    indptr=self.kv_indptr_decode[1][:bs * self.num_kv_heads + 1],
                    indices=self.kv_indices_decode[1],
                    last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                )

                self.forward_metadata = DecodeMetadata(
                    [self.decode_wrappers[0], self.decode_wrappers[1], self.decode_wrappers[2]]
                )

        elif forward_batch.forward_mode.is_extend():
            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            bs = len(forward_batch.req_pool_indices)
            input_seq_lens = forward_batch.seq_lens.to(torch.int32) - prefix_lens

            # Compute qo_indptr for ragged self-attention (cumsum of input_seq_lens)
            self.qo_indptr[0] = 0
            torch.cumsum(input_seq_lens, dim=0, out=self.qo_indptr[1:bs+1])

            # Plan ragged wrapper for causal self-attention on new tokens
            self.prefill_wrapper_ragged.plan(
                self.qo_indptr[:bs+1],
                self.qo_indptr[:bs+1],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

            # If cached prefix exists, prepare decode-style indices for the indexer
            if not extend_no_prefix:
                vortex_torch.indexer.utils_sglang.plan_decode(
                    cached_seq_lens=prefix_lens,
                    req_to_token=self.req_to_token,
                    req_indices=forward_batch.req_pool_indices,
                    ctx=self.ctx
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
                # Int8: no FlashInfer decode wrappers
                self.decode_cuda_graph_metadata[bs] = None
                self.forward_metadata = DecodeMetadata(decode_wrappers=None)
            else:
                decode_wrappers = [
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.kv_indptr_decode[0][:bs * self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.kv_indices_decode[0],
                        paged_kv_last_page_len_buffer=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                    ),
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.kv_indptr_decode[1][:bs * self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.staging_kv_indices,
                        paged_kv_last_page_len_buffer=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                    ),
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.kv_indptr_decode[1][:bs * self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.kv_indices_decode[1],
                        paged_kv_last_page_len_buffer=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                    ),
                ]

                decode_wrappers[0].plan(
                    indptr=self.kv_indptr_decode[0][:bs * self.num_kv_heads + 1],
                    indices=self.kv_indices_decode[0],
                    last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                )

                decode_wrappers[1].plan(
                    indptr=self.kv_indptr_decode[1][:bs * self.num_kv_heads + 1],
                    indices=self.staging_kv_indices,
                    last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                )

                decode_wrappers[2].plan(
                    indptr=self.kv_indptr_decode[1][:bs * self.num_kv_heads + 1],
                    indices=self.kv_indices_decode[1],
                    last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
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
            raise NotImplementedError("CUDA graph capture for prefill not supported")

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
            # Int8: plan_decode already filled indptr/indices; no FlashInfer plan needed
            pass
        else:
            self.decode_cuda_graph_metadata[bs][0].plan(
                indptr=self.kv_indptr_decode[0][:bs * self.num_kv_heads + 1],
                indices=self.kv_indices_decode[0],
                last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

            self.decode_cuda_graph_metadata[bs][1].plan(
                indptr=self.kv_indptr_decode[1][:bs * self.num_kv_heads + 1],
                indices=self.staging_kv_indices,
                last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

            self.decode_cuda_graph_metadata[bs][2].plan(
                indptr=self.kv_indptr_decode[1][:bs * self.num_kv_heads + 1],
                indices=self.kv_indices_decode[1],
                last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # =========================================================================
    # Forward extend (ragged prefill)
    # =========================================================================

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
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
            # Has cached prefix: self-attn on new tokens + sparse cross-attn against prefix

            # Stage 1: Self-attention on new tokens (causal)
            o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                q_view, k_view, v_view,
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

            # Stage 2: Sparse cross-attention against cached prefix (ragged)
            bs = len(forward_batch.req_pool_indices)
            layer_type = forward_batch.token_to_kv_pool.get_layer_type(layer.layer_id)
            use_sparsity = (layer.layer_id not in self.layers_skip)

            # Get cache for indexer (staging for cpu_sparse, gpu cache for gpu_sparse)
            if layer_type == 'full':
                # Full attention: gather ALL prefix pages (dense)
                k_gpu, v_gpu = forward_batch.token_to_kv_pool.get_kv_buffer_gpu(layer.layer_id)
                cache_for_gather = {"k": k_gpu, "v": v_gpu}
                if self.is_int8:
                    local_id = forward_batch.token_to_kv_pool.layers_mapping[layer.layer_id][0]
                    cache_for_gather["k_scale"] = forward_batch.token_to_kv_pool.k_scale_full[local_id]
                    cache_for_gather["v_scale"] = forward_batch.token_to_kv_pool.v_scale_full[local_id]
                total_tokens = self._gather_pages_to_ragged(cache_for_gather, bs, sparse=False)
            else:
                cache = forward_batch.token_to_kv_pool.get_cache(layer.layer_id)
                # For int8 cpu_sparse: indexer outputs CPU flat page IDs (centroids
                # are num_pages_cpu-indexed). We pass cpu_sparse_info so
                # _gather_pages_to_ragged can use the CUDA dequant kernel to read
                # int8 from CPU pinned + scales from GPU → compact bf16 output.
                # For cpu_sparse layers (all dtypes): the indexer returns CPU flat page
                # IDs (centroids are num_pages_cpu-indexed), but cache_staging["k"] only
                # has num_pages_gpu_staging entries. We must gather from CPU pinned memory
                # (which IS correctly indexed by CPU page IDs) to avoid out-of-bounds.
                cpu_sparse_info = None
                if layer_type == 'cpu_sparse':
                    local_id = forward_batch.token_to_kv_pool.layers_mapping[layer.layer_id][0]
                    cpu_sparse_info = {
                        "cpu_k": forward_batch.token_to_kv_pool.cache_cpu[local_id]["k"],
                        "cpu_v": forward_batch.token_to_kv_pool.cache_cpu[local_id]["v"],
                    }
                    if self.is_int8:
                        cpu_sparse_info["k_scale"] = forward_batch.token_to_kv_pool.cache_scale[local_id]["k_scale"]
                        cpu_sparse_info["v_scale"] = forward_batch.token_to_kv_pool.cache_scale[local_id]["v_scale"]
                    if self.is_fp8:
                        cpu_sparse_info["fp8"] = True
                        cpu_sparse_info["data_type"] = self.data_type
                if use_sparsity:
                    q_summary = self._compute_query_summary(q_view, bs)
                    self.sparse_attention.forward_indexer(
                        q=q_summary,
                        o=self.kv_indices_decode[1],
                        cache=cache,
                        ctx=self.ctx
                    )
                    total_tokens = self._gather_pages_to_ragged(cache, bs, sparse=True, cpu_sparse_info=cpu_sparse_info)
                else:
                    total_tokens = self._gather_pages_to_ragged(cache, bs, sparse=False, cpu_sparse_info=cpu_sparse_info)

            # Plan sparse ragged wrapper for cross-attention
            self.prefill_wrapper_ragged_sparse.plan(
                self.qo_indptr[:bs+1],
                self.sparse_prefill_kv_indptr[:bs+1],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

            # Cross-attention (non-causal: all prefix tokens precede new tokens)
            o2, s2 = self.prefill_wrapper_ragged_sparse.forward_return_lse(
                q_view,
                self.sparse_prefill_k_buf[:total_tokens],
                self.sparse_prefill_v_buf[:total_tokens],
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

            # Stage 3: Merge self-attention and cross-attention
            o, _ = merge_state(o1, s1, o2, s2)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _compute_query_summary(self, q_view: torch.Tensor, bs: int) -> torch.Tensor:
        """Compute per-request mean Q for the indexer API."""
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

    def _gather_pages_to_ragged(
        self,
        cache: Dict[str, torch.Tensor],
        bs: int,
        sparse: bool = True,
        cpu_sparse_info: Dict[str, torch.Tensor] = None,
    ) -> int:
        """
        Gather paged KV into contiguous ragged buffers for the ragged wrapper.

        Handles int8 (dequant to bf16), fp8 (view-cast to bf16), and bf16 (direct gather).

        For int8 cpu_sparse layers, cpu_sparse_info provides CPU pinned int8 buffers
        and GPU scales. The CUDA dequant kernel reads from CPU via UVA, avoiding
        the index-space mismatch between staging slots and CPU page IDs.
        """
        if sparse:
            kv_indptr = self.ctx.sparse_kv_indptr
            kv_indices = self.ctx.sparse_kv_indices
        else:
            kv_indptr = self.ctx.dense_kv_indptr
            kv_indices = self.ctx.dense_kv_indices

        num_batch_kv = bs * self.num_kv_heads
        total_pages = int(kv_indptr[num_batch_kv].item())

        if total_pages == 0:
            self.sparse_prefill_kv_indptr[:bs + 1] = 0
            return 0

        selected_page_ids = kv_indices[:total_pages].to(torch.int32)

        # Compute dst_offsets: token offset in ragged buffer for each request.
        # pages_per_head for request i = kv_indptr[i*num_kv_heads+1] - kv_indptr[i*num_kv_heads]
        # tokens_per_request = pages_per_head * page_size
        head0_starts = kv_indptr[
            torch.arange(bs, device=kv_indptr.device) * self.num_kv_heads
        ]
        head0_ends = kv_indptr[
            torch.arange(bs, device=kv_indptr.device) * self.num_kv_heads + 1
        ]
        tokens_per_request = (head0_ends - head0_starts) * self.page_size
        dst_offsets = torch.zeros(bs + 1, dtype=torch.int32, device=kv_indptr.device)
        torch.cumsum(tokens_per_request, dim=0, out=dst_offsets[1:])
        total_tokens_out = int(dst_offsets[bs].item())

        # Update kv_indptr for ragged wrapper
        self.sparse_prefill_kv_indptr[:bs + 1] = dst_offsets[:bs + 1]

        if total_tokens_out == 0:
            return 0

        # Determine source KV buffer and quant params
        if cpu_sparse_info is not None:
            src_k = cpu_sparse_info["cpu_k"]
            src_v = cpu_sparse_info["cpu_v"]
            if self.is_int8:
                quant_type = 1
                kv_scale = 1.0
                k_scale = cpu_sparse_info["k_scale"]
                v_scale = cpu_sparse_info["v_scale"]
            elif self.is_fp8:
                quant_type = 2 if cpu_sparse_info.get("data_type") == torch.float8_e4m3fn else 3
                kv_scale = float(getattr(self, '_fp8_kv_scale', 1.0))
                k_scale = None
                v_scale = None
            else:
                quant_type = 0
                kv_scale = 1.0
                k_scale = None
                v_scale = None
        else:
            src_k = cache["k"]
            src_v = cache["v"]
            if self.is_int8:
                quant_type = 1
                kv_scale = 1.0
                k_scale = cache["k_scale"]
                v_scale = cache["v_scale"]
            elif self.is_fp8:
                quant_type = 2 if self.data_type == torch.float8_e4m3fn else 3
                kv_scale = float(getattr(self, '_fp8_kv_scale', 1.0))
                k_scale = None
                v_scale = None
            else:
                quant_type = 0
                kv_scale = 1.0
                k_scale = None
                v_scale = None

        # Single CUDA kernel: gather pages, dequant, write to ragged buffer
        import vortex_torch.cache
        vortex_torch.cache.gather_pages_to_ragged(
            src_k, self.sparse_prefill_k_buf,
            selected_page_ids, kv_indptr[:num_batch_kv + 1],
            dst_offsets[:bs].to(torch.int32),
            total_pages, self.num_kv_heads, self.page_size, self.head_dim, bs,
            quant_type, kv_scale, k_scale,
        )
        vortex_torch.cache.gather_pages_to_ragged(
            src_v, self.sparse_prefill_v_buf,
            selected_page_ids, kv_indptr[:num_batch_kv + 1],
            dst_offsets[:bs].to(torch.int32),
            total_pages, self.num_kv_heads, self.page_size, self.head_dim, bs,
            quant_type, kv_scale, v_scale,
        )

        return total_tokens_out

    # =========================================================================
    # Forward decode
    # =========================================================================

    def _forward_decode_int8(
        self,
        q: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        layer: RadixAttention,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        bs: int,
        scale_page_map: torch.Tensor = None,
    ) -> torch.Tensor:
        """Int8 decode attention using custom Triton kernel.

        If scale_page_map is provided, the kernel resolves scale page IDs on-the-fly:
            scale_page = scale_page_map[data_page]
        This supports CPU VTX where KV data is in staging slots but scales are
        indexed by CPU flat page IDs.
        """
        q = q.view(-1, self.group_size, layer.head_dim).contiguous()

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
            scale_page_map=scale_page_map,
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
        Decode phase with three attention paths:
        1. Full attention: dense attention on GPU KV
        2. GPU sparse: sparse attention on GPU KV
        3. CPU sparse: sparse attention with CPU->GPU staging copy
        Each path handles int8/fp8/bf16 dtypes.
        """
        assert not layer.is_cross_attention

        cache_loc = forward_batch.out_cache_loc
        bs = len(forward_batch.req_pool_indices)

        # Get layer type from pool
        layer_type = forward_batch.token_to_kv_pool.get_layer_type(layer.layer_id)

        # Profiling: all cpu_sparse layers (syncs are deferred to model_runner)
        pool = forward_batch.token_to_kv_pool
        _profile = (layer_type == 'cpu_sparse' and pool.profile_enabled)

        # Save new K/V to appropriate cache (includes centroid update for sparse layers)
        if k is not None:
            assert v is not None
            if save_kv_cache:
                if _profile:
                    _ev_cache_start = torch.cuda.Event(enable_timing=True)
                    _ev_cache_end = torch.cuda.Event(enable_timing=True)
                    _ev_cache_start.record()
                pool.set_kv_buffer_decode(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
                if _profile:
                    _ev_cache_end.record()

        if layer_type == 'cpu_sparse':
            # CPU sparse path: copy from CPU to GPU staging, then sparse attention

            q = q.view(-1, self.group_size, layer.head_dim).contiguous()

            # Get cache for sparse indexing (staging buffer with centroids)
            cache = pool.get_cache(layer.layer_id)

            # Build sparse indices
            if _profile:
                _ev_idx_start = torch.cuda.Event(enable_timing=True)
                _ev_idx_end = torch.cuda.Event(enable_timing=True)
                _ev_idx_start.record()
            self.sparse_attention.forward_indexer(
                q=q,
                o=self.kv_indices_decode[1],
                cache=cache,
                ctx=self.ctx
            )
            if _profile:
                _ev_idx_end.record()

            # Copy sparse KV from CPU to GPU staging buffer (CG compatible)
            # (alloc + copy are profiled inside copy_sparse_kv_to_gpu)
            result = pool.copy_sparse_kv_to_gpu(
                layer_id=layer.layer_id,
                sparse_kv_indices=self.kv_indices_decode[1],
                sparse_kv_indptr=self.kv_indptr_decode[1],
                dst_kv_indices=self.staging_kv_indices,
                batch_size=bs,
            )

            k_staging, v_staging, _, k_scale_buf, v_scale_buf, gpu_to_cpu_map = result

            if _profile:
                _ev_attn_start = torch.cuda.Event(enable_timing=True)
                _ev_attn_end = torch.cuda.Event(enable_timing=True)
                _ev_attn_start.record()

            if self.is_int8:
                staging_cache = {
                    "k": k_staging, "v": v_staging,
                    "k_scale": k_scale_buf, "v_scale": v_scale_buf,
                }
                o = self._forward_decode_int8(
                    q, staging_cache, layer,
                    kv_indptr=self.kv_indptr_decode[1],
                    kv_indices=self.staging_kv_indices,
                    bs=bs,
                    scale_page_map=gpu_to_cpu_map,
                )
            elif self.is_fp8:
                k_fp8 = k_staging.view(self.data_type).view(-1, self.page_size, 1, self.head_dim)
                v_fp8 = v_staging.view(self.data_type).view(-1, self.page_size, 1, self.head_dim)
                o = self.forward_metadata.decode_wrappers[1].forward(
                    q, (k_fp8, v_fp8),
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )
            else:
                o = self.forward_metadata.decode_wrappers[1].forward(
                    q, (k_staging, v_staging),
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )

            if _profile:
                _ev_attn_end.record()
                # NO sync here — defer to model_runner's single sync at step end.
                if not hasattr(self, '_pending_attn_events'):
                    self._pending_attn_events = []
                self._pending_attn_events.append((
                    _ev_cache_start, _ev_cache_end,
                    _ev_idx_start, _ev_idx_end,
                    _ev_attn_start, _ev_attn_end,
                    layer.layer_id
                ))


        elif layer_type == 'gpu_sparse':
            # GPU sparse path: sparse attention on GPU
            q = q.view(-1, self.group_size, layer.head_dim).contiguous()
            cache = forward_batch.token_to_kv_pool.get_cache(layer.layer_id)

            self.sparse_attention.forward_indexer(
                q=q,
                o=self.kv_indices_decode[1],
                cache=cache,
                ctx=self.ctx
            )

            if self.is_int8:
                o = self._forward_decode_int8(
                    q, cache, layer,
                    kv_indptr=self.kv_indptr_decode[1],
                    kv_indices=self.kv_indices_decode[1],
                    bs=bs,
                )
            elif self.is_fp8:
                cache_k = cache["k"].view(self.data_type).view(-1, self.page_size, 1, self.head_dim)
                cache_v = cache["v"].view(self.data_type).view(-1, self.page_size, 1, self.head_dim)
                o = self.forward_metadata.decode_wrappers[2].forward(
                    q, (cache_k, cache_v),
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )
            else:
                cache_k = cache["k"].view(-1, self.page_size, 1, self.head_dim)
                cache_v = cache["v"].view(-1, self.page_size, 1, self.head_dim)
                o = self.forward_metadata.decode_wrappers[2].forward(
                    q, (cache_k, cache_v),
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )

        else:  # full attention
            k_gpu, v_gpu = forward_batch.token_to_kv_pool.get_kv_buffer_gpu(layer.layer_id)

            if self.is_int8:
                local_id = forward_batch.token_to_kv_pool.layers_mapping[layer.layer_id][0]
                full_cache = {
                    "k": k_gpu, "v": v_gpu,
                    "k_scale": forward_batch.token_to_kv_pool.k_scale_full[local_id],
                    "v_scale": forward_batch.token_to_kv_pool.v_scale_full[local_id],
                }
                o = self._forward_decode_int8(
                    q, full_cache, layer,
                    kv_indptr=self.kv_indptr_decode[0],
                    kv_indices=self.kv_indices_decode[0],
                    bs=bs,
                )
            elif self.is_fp8:
                k_fp8 = k_gpu.view(self.data_type).view(-1, self.page_size, 1, self.head_dim)
                v_fp8 = v_gpu.view(self.data_type).view(-1, self.page_size, 1, self.head_dim)
                o = self.forward_metadata.decode_wrappers[0].forward(
                    q.contiguous().view(-1, self.group_size, layer.head_dim),
                    (k_fp8, v_fp8),
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )
            else:
                o = self.forward_metadata.decode_wrappers[0].forward(
                    q.contiguous().view(-1, self.group_size, layer.head_dim),
                    (k_gpu, v_gpu),
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        # Wrapper 0 = dense attention, Wrapper 1 = sparse attention
        return 0 if layer.layer_id in self.full_attention_layer_ids else 1
