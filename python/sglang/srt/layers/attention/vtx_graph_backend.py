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
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Dict, Tuple
from functools import partial
import torch
import vortex_torch
from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch import is_hopper
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
from sglang.srt.layers.attention.flashinfer_backend import should_use_tensor_core
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

@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]

@dataclass
class PrefillMetadata:
    extend_no_prefix: bool

@dataclass
class VerifyMetadata:
    """Metadata for TARGET_VERIFY mode (speculative decoding verification)."""
    pass

@dataclass
class DraftExtendMetadata:
    """Metadata for DRAFT_EXTEND mode (speculative decoding draft generation)."""
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


def _vtx_to_std_layout(
    cache: torch.Tensor, num_kv_heads: int, page_size: int, head_dim: int
) -> torch.Tensor:
    """Convert VTXGraphCachePool KV layout to standard FlashInfer paged layout.

    VTX layout (from vortex_torch/cache/triton_kernels/set_kv.py):
        shape = (size*num_kv_heads, page_size, head_dim)
        For page_size=1: flat_page_idx = token_pos * num_kv_heads + head_id

    Standard FlashInfer NHD paged layout:
        shape = (num_pages, page_size, num_kv_heads, head_dim)

    For page_size=1 this reshape is a pure view (contiguous): VTX stores each
    token as num_kv_heads consecutive pages, which is exactly the memory order
    of a standard (token, head, head_dim) tensor. We just view+transpose the
    size-1 page dim into the correct slot.
    """
    assert page_size == 1, (
        f"_vtx_to_std_layout only supports page_size=1; got {page_size}. "
        "General page_size requires a different reshape (block-major head)."
    )
    # VTXGraphCachePool adds a guard page at the end; drop any trailing
    # fractional pages so shape[0] is divisible by num_kv_heads.
    usable = (cache.shape[0] // num_kv_heads) * num_kv_heads
    cache = cache[:usable]
    # (usable, 1, head_dim) with flat index = token_pos * num_kv_heads + head_id
    # → (num_tokens, num_kv_heads, head_dim)
    std = cache.view(-1, num_kv_heads, head_dim)
    # → (num_tokens, page_size=1, num_kv_heads, head_dim)
    std = std.unsqueeze(1)
    return std


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
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # VTX manages its own page selection via top-k; sliding_window from the
        # model config is ignored (VTX's vortex_max_seq_lens serves as the
        # effective context budget).  Log a warning if sliding_window is set.
        if model_runner.sliding_window_size is not None:
            import logging
            logging.getLogger(__name__).warning(
                f"Model has sliding_window_size={model_runner.sliding_window_size}, "
                "but VTXGraphAttnBackend ignores it (uses vortex_max_seq_lens instead)."
            )
        assert not model_runner.model_config.is_encoder_decoder
        assert not self.skip_prefill, "skip_prefill=True not yet supported in VTXGraphAttnBackend; use VTXGraphMultiStepDraftBackend"
        assert not self.is_multimodal
        assert kv_indptr_buf is None, "external kv_indptr_buf not supported in VTXGraphAttnBackend; use VTXGraphMultiStepDraftBackend"
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
        self.decode_use_tensor_cores = should_use_tensor_core(self.data_type, self.num_qo_heads, self.num_kv_heads)
        assert self.q_data_type == torch.bfloat16
        assert self.data_type in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
        self.is_fp8 = (self.data_type in [torch.float8_e5m2, torch.float8_e4m3fn])
        
        # Assign key configuration and parameters
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.server_args.page_size
        self.block_size = model_runner.server_args.vortex_block_size
        self.layers_skip = model_runner.server_args.vortex_layers_skip
        self.num_blocks_per_page = self.page_size // self.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size."
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
                    (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.block_size - 1)
                    // self.block_size,
                ),
                dtype=torch.int32,
                device=model_runner.device
            ),
            torch.zeros(
                (
                    (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.block_size - 1)
                    // self.block_size,
                ),
                dtype=torch.int32,
                device=model_runner.device
            ),
        ]

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

        
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend= "auto" if not is_hopper() else "fa3"
        )

        self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2" if ((not is_hopper()) or self.is_fp8) else "fa3",
                    )
        
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
        
        self.plan_decode = vortex_torch.indexer.utils_sglang.get_decode_planner(model_runner.server_args.vortex_schedule_policy)
        

        # Verify wrapper (for TARGET_VERIFY with custom_mask)
        self.verify_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend="fa2",
        )
        # Draft extend wrapper (reuse prefill wrapper style)
        self.draft_extend_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend="fa2",
        )

        self.speculative_num_draft_tokens = getattr(
            model_runner.server_args, 'speculative_num_draft_tokens', None
        ) or 0
        self.sparse_attention = model_runner.sparse_attention
        self.ctx = vortex_torch.indexer.Context()
        self._initialize_graph(model_runner)
        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata, VerifyMetadata, DraftExtendMetadata] = None
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
                q_dummy = as_vtensor(torch.empty((0, self.group_size, self.head_dim), device=device, dtype=dtype), FORMAT.BATCHED, tensor_id=0)
                self.ctx.tensor_list.append(q_dummy)
                self.ctx.output_tensor_to_op_list.append(None)  # Placeholder for mapping output tensors to ops
                self.ctx.tensor_id_to_tensor_name_map[q_dummy.tensor_id] = "q"
                o_dummy = as_vtensor(torch.empty((0, 1, 1), device=device, dtype=dtype), FORMAT.RAGGED, tensor_id=1)
                self.ctx.tensor_list.append(o_dummy)
                self.ctx.output_tensor_to_op_list.append(None)  # Placeholder for mapping output tensors to ops
                self.ctx.tensor_id_to_tensor_name_map[o_dummy.tensor_id] = "o"
                cache_meta_info = self.sparse_attention.get_cache_meta_info()
                cache_dummy = {}
                for i, (cache_name, (cache_shape, cache_dtype)) in enumerate(cache_meta_info.items()):
                    cache_dummy[cache_name] = as_vtensor(
                        torch.zeros(
                            (0 * self.num_blocks_per_page, cache_shape[0], cache_shape[1]),
                            dtype=cache_dtype,
                            device=device,
                        ),
                        FORMAT.PAGED,
                        tensor_id=2 + i
                    )
                    self.ctx.tensor_list.append(cache_dummy[cache_name])
                    self.ctx.output_tensor_to_op_list.append(None)  # Placeholder for mapping output tensors to ops
                    self.ctx.tensor_id_to_tensor_name_map[cache_dummy[cache_name].tensor_id] = f"cache['{cache_name}']"
                indexer(q_dummy, o_dummy, cache_dummy, ctx=self.ctx)


        except Exception:
            raise
        
        
        compiled_indexer_cls = vortex_torch.indexer.compiler.compile.compile(self.ctx)
        self.compiled_indexer = compiled_indexer_cls()
        self.ctx.summary()
        self.ctx.execute()

    
    def init_forward_metadata(self, forward_batch: ForwardBatch):

        if forward_batch.forward_mode.is_target_verify():
            # ============================================
            # TARGET_VERIFY: speculative decoding verification
            # Uses BatchPrefillWrapper with custom_mask for tree attention.
            # Sparse attention is SKIPPED — full attention for correctness.
            #
            # NOTE (session 13 fix): VTXGraphCachePool stores KV in per-head
            # layout: cache shape (size*num_kv_heads, page_size, head_dim) with
            # flat index = token_pos * num_kv_heads + head_id (for page_size=1).
            # Standard FlashInfer expects (num_pages, page_size, num_kv_heads,
            # head_dim) where each page holds ALL heads. To reconcile, at
            # forward-time we reshape the cache to standard layout (see
            # forward_extend for the actual reshape).
            # Here we plan the wrapper with the true num_kv_heads, so kv_indices
            # (which are token positions from req_to_token) index correctly into
            # the standard-layout view.
            # Assumes page_size=1 (only supported config).
            # ============================================
            assert self.page_size == 1, (
                "VTX verify path currently supports page_size=1 only"
            )
            bs = len(forward_batch.req_pool_indices)
            spec_info = forward_batch.spec_info

            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    self.req_to_token,
                )
            )
            self.verify_wrapper_paged.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                self.kv_last_page_len_prefill[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                custom_mask=custom_mask,
            )
            self.forward_metadata = VerifyMetadata()

        elif forward_batch.forward_mode.is_draft_extend():
            # ============================================
            # DRAFT_EXTEND: draft model generates candidate tokens.
            # Treated like regular EXTEND (prefill path).
            # No sparse attention — draft uses full attention.
            # Same cache-layout fix as TARGET_VERIFY.
            # ============================================
            assert self.page_size == 1, (
                "VTX draft_extend path currently supports page_size=1 only"
            )
            bs = len(forward_batch.req_pool_indices)
            spec_info = forward_batch.spec_info

            kv_indices, kv_indptr, qo_indptr, _ = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    self.req_to_token,
                )
            )
            self.draft_extend_wrapper_paged.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                self.kv_last_page_len_prefill[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                custom_mask=None,  # draft extend uses causal attention
            )
            self.forward_metadata = DraftExtendMetadata(extend_no_prefix=True)

        elif forward_batch.forward_mode.is_decode_or_idle():
            
            bs = len(forward_batch.req_pool_indices)
            self.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                ctx=self.ctx
            )
            
            self.decode_wrappers[0].plan(
                indptr=self.kv_indptr_decode[0][:bs*self.num_kv_heads+1],
                indices=self.kv_indices_decode[0],
                last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
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
                page_size=self.block_size,
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
                dense_kv_indptr=self.kv_indptr_prefill[:bs*self.num_kv_heads+1],
                dense_kv_indices=self.kv_indices_prefill,
                input_seq_lens=(forward_batch.seq_lens.to(torch.int32) - prefix_lens),
                qo_indptr_ragged=self.qo_indptr[0][:bs+1],
                qo_indptr_paged=self.qo_indptr[1][:bs*self.num_kv_heads+1],
                kv_last_page_len=self.kv_last_page_len_prefill[:bs*self.num_kv_heads],
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                batch_table=self.batch_table,
                page_size=self.page_size,
                num_kv_heads=self.num_kv_heads
            )
            
   
            self.prefill_wrapper_ragged.plan(
                self.qo_indptr[0][:bs+1],
                self.qo_indptr[0][:bs+1],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )
            
            self.prefill_wrapper_paged.plan(
                self.qo_indptr[1][:bs*self.num_kv_heads+1],
                self.kv_indptr_prefill[:bs*self.num_kv_heads+1],
                self.kv_indices_prefill,
                self.kv_last_page_len_prefill[:bs*self.num_kv_heads],
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
        if forward_mode.is_decode_or_idle():
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

            self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
            
            decode_wrappers[0].plan(
                indptr=self.kv_indptr_decode[0][:bs*self.num_kv_heads+1],
                indices=self.kv_indices_decode[0],
                last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
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
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
        else:
            # Non-decode modes (verify, draft_extend) fall back to non-CG path.
            # CUDA graph capture is skipped; these modes use dynamic plan() at runtime.
            pass
            

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
        
        self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
        
        self.decode_cuda_graph_metadata[bs][0].plan(
            indptr=self.kv_indptr_decode[0][:bs*self.num_kv_heads+1],
            indices=self.kv_indices_decode[0],
            last_page_len=self.kv_last_page_len_decode[:bs*self.num_kv_heads],
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=self.head_dim,
            page_size=self.block_size,
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
            page_size=self.block_size,
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
        
        #assert isinstance(forward_batch.token_to_kv_pool, VTXGraphCachePool)
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc
        logits_soft_cap = layer.logit_cap
        q = q.contiguous()

        # ============================================
        # TARGET_VERIFY: speculative decoding verification
        # Full attention with tree mask via BatchPrefillWrapper.
        # No sparse attention — correctness first.
        # ============================================
        if forward_batch.forward_mode.is_target_verify():
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            # VTXGraphCachePool stores KV in per-KV-head layout:
            #   shape (size*num_kv_heads, page_size=1, head_dim)
            #   flat index = token_pos * num_kv_heads + head_id
            # Reshape to standard FlashInfer layout (num_pages, page_size, num_kv_heads, head_dim)
            # so kv_indices (token positions) index correctly.
            k_cache = _vtx_to_std_layout(k_cache, self.num_kv_heads, self.page_size, self.head_dim)
            v_cache = _vtx_to_std_layout(v_cache, self.num_kv_heads, self.page_size, self.head_dim)

            o = self.verify_wrapper_paged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                (k_cache, v_cache),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )
            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

        # ============================================
        # DRAFT_EXTEND: draft model generates candidate tokens
        # Treated like regular extend (prefill), no sparse attention.
        # ============================================
        if forward_batch.forward_mode.is_draft_extend():
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            # Same per-head → standard layout reshape as verify path.
            k_cache = _vtx_to_std_layout(k_cache, self.num_kv_heads, self.page_size, self.head_dim)
            v_cache = _vtx_to_std_layout(v_cache, self.num_kv_heads, self.page_size, self.head_dim)

            o = self.draft_extend_wrapper_paged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                (k_cache, v_cache),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )
            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

        # ============================================
        # Normal EXTEND (prefill): original Vortex logic
        # ============================================
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

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_cache = k_cache.view(-1, self.page_size, 1, self.head_dim)
            v_cache = v_cache.view(-1, self.page_size, 1, self.head_dim)
            o2, s2 = self.prefill_wrapper_paged.forward_return_lse(
                q_t,
                (k_cache, v_cache),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            o2_t, s2_t = vortex_torch.indexer.utils_sglang.chunkwise_hn2nh_transpose(
                o2,  s2,
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
        Decode-time forward pass with optional sparse attention.
        Expects KV to be sourced from token_to_kv_pool; can also save new KV.
        """

        # Sanity checks and setup
        #assert isinstance(forward_batch.token_to_kv_pool, VTXGraphCachePool)
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
        
        cache_k = cache["k"].view(-1, self.block_size, 1, self.head_dim)
        cache_v = cache["v"].view(-1, self.block_size, 1, self.head_dim)
        
        # Decide whether to use sparsity on this layer.
        # Sparsity is disabled for draft decode (DRAFT_EXTEND handled separately).
        # During draft multi-step generation, forward_batch.spec_info is an EagleDraftInput.
        is_draft_step = (
            forward_batch.spec_info is not None
            and hasattr(forward_batch.spec_info, "hidden_states")  # EagleDraftInput has hidden_states
        )
        use_sparsity = (layer.layer_id not in self.layers_skip) and (not is_draft_step)

        if use_sparsity:
            # Prepare Q in grouped shape expected by sparse path
            q = q.reshape(-1, self.group_size, layer.head_dim).contiguous()

            # Build sparse indices into paged KV buffers
            self.compiled_indexer.forward(
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


class VTXGraphMultiStepDraftBackend:
    """
    Multi-step draft backend compatible with VTXGraphCachePool.

    Mirrors the interface of FlashInferMultiStepDraftBackend, but uses
    VTXGraphAttnBackend for each decode step so that VTXGraphCachePool
    KV format (page-by-page, group_size=self.group_size) is handled correctly.

    Sparse attention is intentionally DISABLED for draft steps (full attention
    gives better draft quality and the draft model is small).
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.eagle_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size

        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (self.speculative_num_steps, max_bs + 1),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )

        # Single shared VTXGraphAttnBackend for all draft steps.
        # Re-planning before each step is cheap (no model load), and sharing
        # avoids allocating N × backend_buffers which OOMs at large num_steps.
        self._shared_backend = VTXGraphAttnBackend(model_runner)

        # attn_backends[i] all point to the same shared backend.
        # eagle_worker sets forward_batch.attn_backend = attn_backends[i],
        # so they must be indexable. Re-planning happens via _replan_for_step().
        self.attn_backends: List[VTXGraphAttnBackend] = [
            self._shared_backend for _ in range(self.speculative_num_steps)
        ]

        self.max_context_len = self._shared_backend.max_context_len
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self._last_forward_batch = None

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn,
    ):
        from sglang.srt.speculative.eagle_utils import EagleDraftInput
        from sglang.srt.utils import next_power_of_2

        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.token_to_kv_pool.req_to_token,
            kv_indices_buffer,
            self.kv_indptr,
            self.pool_len,
            next_power_of_2(bs),
            self.page_size,
        )

        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # For VTX, we plan decode step 0 with the current seq_lens.
        # Steps 1..N-1 will re-plan with incremented seq_lens in _replan_for_step().
        # This is called once by eagle_worker before the multi-step draft_forward loop.
        self.attn_backends[0].init_forward_metadata(forward_batch)
        # Store a reference so we can re-plan subsequent steps
        self._last_forward_batch = forward_batch

    def _replan_for_step(self, step: int):
        """Re-plan decode attention for draft step `step` (step > 0).
        seq_lens at step i = original_seq_len + i (each step appends 1 draft token per seq).
        Called from eagle_worker's forward_batch.attn_backend before each forward pass.
        """
        if self._last_forward_batch is None:
            return
        fb = self._last_forward_batch
        # Temporarily increment seq_lens to include draft tokens from steps 0..step-1
        original_seq_lens = fb.seq_lens.clone()
        fb.seq_lens = fb.seq_lens + step
        self.attn_backends[step].init_forward_metadata(fb)
        fb.seq_lens = original_seq_lens

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_bs * self.max_context_len),
            dtype=torch.int32,
            device="cuda",
        )
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens,
                kv_indices_buf=self.cuda_graph_kv_indices[i],
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)