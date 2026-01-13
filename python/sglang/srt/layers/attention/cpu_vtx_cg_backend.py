from __future__ import annotations

"""
CPU-based Vortex FlashInfer Attention Backend with CUDA Graph Support.

This backend stores KV cache on CPU and only transfers sparse pages to GPU during decode.
Supports CUDA graph capture and replay for efficient decoding.

Key features:
1. KV cache stored on CPU (via CPUVTXTokenToKVPool)
2. CUDA graph compatible decode path
3. Sparse attention with CPU->GPU transfer via GPU kernel (not cudaMemcpy)
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import vortex_torch
from vortex_torch.abs import as_vtensor, FORMAT

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
from sglang.srt.mem_cache.cpu_vtx_memory_pool import CPUVTXTokenToKVPool

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


@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]


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
        assert self.data_type == torch.bfloat16

        # Assign key configuration and parameters
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.server_args.page_size
        self.layers_skip = model_runner.server_args.vortex_layers_skip

        # ===========================
        # Prefill KV-indptr buffers
        # ===========================

        self.kv_indptr_prefill = torch.zeros(
            (max_bs * self.num_kv_heads + 1,),
            dtype=torch.int32,
            device=model_runner.device
        )

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
        # KV indices (prefill)
        # ===========================

        self.kv_indices_prefill = torch.zeros(
            (
                (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1)
                // self.page_size,
            ),
            dtype=torch.int32,
            device=model_runner.device
        )

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
        # This is needed because copy_sparse_kv_to_gpu reads CPU page IDs and writes staging slots,
        # and they cannot be the same buffer
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

        self.kv_last_page_len_prefill = torch.ones(
            (max_bs * self.num_kv_heads,),
            dtype=torch.int32,
            device=model_runner.device
        )

        self.kv_last_page_len_decode = torch.ones(
            (max_bs * self.num_kv_heads,),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # Query/Output indptr buffers
        # ===========================

        self.qo_indptr = [
            torch.zeros(
                (max_bs + 1,),
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
        # Batch table (token-level mapping)
        # ===========================

        self.batch_table = torch.zeros(
            (model_runner.server_args.max_prefill_tokens,),
            dtype=torch.uint16,
            device=model_runner.device
        )

        # Prefill wrappers
        fmha_backend = "auto"
        if is_sm100_supported():
            fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="fa2",
        )

        # Decode wrappers - non-CUDA-graph versions for regular decode
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
        dtype = torch.bfloat16

        try:
            with torch.no_grad():
                # Dummy placeholders: used only for kernel / graph warm-up
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

            # Plan decode wrappers
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
                indices=self.staging_kv_indices,  # Use staging_kv_indices (separate from kv_indices_decode[1])
                last_page_len=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
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

            vortex_torch.indexer.utils_sglang.plan_prefill(
                cached_seq_lens=prefix_lens,
                dense_kv_indptr=self.kv_indptr_prefill[:bs * self.num_kv_heads + 1],
                dense_kv_indices=self.kv_indices_prefill,
                input_seq_lens=(forward_batch.seq_lens.to(torch.int32) - prefix_lens),
                qo_indptr_ragged=self.qo_indptr[0][:bs + 1],
                qo_indptr_paged=self.qo_indptr[1][:bs * self.num_kv_heads + 1],
                kv_last_page_len=self.kv_last_page_len_prefill[:bs * self.num_kv_heads],
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                batch_table=self.batch_table,
                page_size=self.page_size,
                num_kv_heads=self.num_kv_heads
            )

            self.prefill_wrapper_ragged.plan(
                self.qo_indptr[0][:bs + 1],
                self.qo_indptr[0][:bs + 1],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

            self.prefill_wrapper_paged.plan(
                self.qo_indptr[1][:bs * self.num_kv_heads + 1],
                self.kv_indptr_prefill[:bs * self.num_kv_heads + 1],
                self.kv_indices_prefill,
                self.kv_last_page_len_prefill[:bs * self.num_kv_heads],
                self.group_size,
                1,
                self.head_dim,
                self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                custom_mask=None,
                non_blocking=True,
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
            # Create CUDA graph-enabled decode wrappers with pre-allocated buffers
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
                    paged_kv_indices_buffer=self.staging_kv_indices,  # Separate buffer from kv_indices_decode[1]
                    paged_kv_last_page_len_buffer=self.kv_last_page_len_decode[:bs * self.num_kv_heads],
                ),
            ]

            vortex_torch.indexer.utils_sglang.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )

            # Plan wrappers
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
                indices=self.staging_kv_indices,  # Use staging_kv_indices (separate from kv_indices_decode[1])
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

        # Re-plan wrappers
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
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc

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

            q_t = vortex_torch.indexer.utils_sglang.chunkwise_nh2hn_transpose(
                q.view(-1, self.num_qo_heads, self.head_dim),
                self.qo_indptr[0],
                self.batch_table,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim
            )
            
            k_cpu, v_cpu = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_cpu, v_cpu = k_cpu.view(-1, self.page_size, 1, self.head_dim), v_cpu.view(-1, self.page_size, 1, self.head_dim)

            o2, s2 = self.prefill_wrapper_paged.forward_return_lse(
                q_t,
                (k_cpu, v_cpu),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

            o2_t, s2_t = vortex_torch.indexer.utils_sglang.chunkwise_hn2nh_transpose(
                o2, s2,
                self.qo_indptr[0],
                self.batch_table,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim
            )

            o, _ = merge_state(o1, s1, o2_t, s2_t)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
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
        """
        Decode phase with sparse CPU->GPU transfer.

        For CPU VTX CG backend:
        1. Save current K/V to CPU
        2. Use sparse attention indexer (CUDA graph compatible)
        3. Copy sparse KV pages from CPU to GPU staging buffer (GPU kernel)
        4. Run attention on GPU with staging buffer
        """
        assert not layer.is_cross_attention

        cache_loc = forward_batch.out_cache_loc
        bs = len(forward_batch.req_pool_indices)
        use_sparsity = (layer.layer_id not in self.layers_skip)

        # Save new K/V to CPU cache
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer_decode(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        if use_sparsity:
            # Prepare Q in grouped shape expected by sparse path
            q = q.view(-1, self.group_size, layer.head_dim).contiguous()

            # Get cache for sparse indexing
            cache = forward_batch.token_to_kv_pool.get_cache(layer.layer_id)

            # Build sparse indices into paged KV buffers
            self.sparse_attention.forward_indexer(
                q=q,
                o=self.kv_indices_decode[1],
                cache=cache,
                ctx=self.ctx
            )

            # Copy sparse KV from CPU to GPU staging buffer (GPU kernel - CG compatible)
            # Read CPU page IDs from kv_indices_decode[1], write staging slots to staging_kv_indices
            result = forward_batch.token_to_kv_pool.copy_sparse_kv_to_gpu(
                layer_id=layer.layer_id,
                sparse_kv_indices=self.kv_indices_decode[1],
                sparse_kv_indptr=self.kv_indptr_decode[1],
                dst_kv_indices=self.staging_kv_indices,
                batch_size=bs,
            )

            k_staging, v_staging, _ = result
            # k_staging = k_staging.view(-1, self.page_size, 1, self.head_dim)
            # v_staging = v_staging.view(-1, self.page_size, 1, self.head_dim)
            
            o = self.forward_metadata.decode_wrappers[1].forward(
                q, (k_staging, v_staging),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )

        else:
            # Dense attention path (for skipped layers)
            # Copy all KV from CPU to GPU staging buffer
            result = forward_batch.token_to_kv_pool.copy_sparse_kv_to_gpu(
                layer_id=layer.layer_id,
                sparse_kv_indices=self.kv_indices_decode[0],
                sparse_kv_indptr=self.kv_indptr_decode[0],
                batch_size=bs,
            )

            k_staging, v_staging, staging_kv_indices = result
            # Staging buffers are already in paged format [num_pages, page_size, 1, head_dim]

            o = self.forward_metadata.decode_wrappers[0].forward(
                q.contiguous().view(-1, self.group_size, layer.head_dim),
                (k_staging, v_staging),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        return 0 if layer.layer_id in self.layers_skip else 1
