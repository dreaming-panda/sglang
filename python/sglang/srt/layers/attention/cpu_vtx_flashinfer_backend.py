from __future__ import annotations

"""
CPU-based Vortex FlashInfer Attention Backend.

This backend stores KV cache on CPU and only transfers sparse pages to GPU during decode.

Key differences from vtx_flashinfer_backend.py:
1. KV cache is stored on CPU (via CPUVTXTokenToKVPool)
2. During prefill: compute KV on GPU, then transfer entire batch to CPU
3. During decode:
   a. Calculate sparse indices with vortex (using GPU landmarks)
   b. Copy only sparse KV pages from CPU to GPU staging buffer (custom kernel)
   c. Run attention on GPU with new indptr/indices for staging buffer
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch

if os.environ.get("SGLANG_ENABLE_TORCH_COMPILE") == "1":
    import logging
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
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

from vortex import SparseAttentionServer


@dataclass
class DecodeMetadata:
    use_sparsity: bool


@dataclass
class PrefillMetadata:
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


import torch
from typing import Callable, Dict, Optional

@torch.no_grad()
def verify_staging_mapping(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    indptr: torch.Tensor,           # int32, shape [rows+1] where rows = bs * num_kv_heads
    indices: torch.Tensor,          # int32, shape [nnz], absolute staging slots (usually arange(nnz))
    sparse_indices: torch.Tensor,   # int32, shape [nnz], per-row page IDs (relative to that row’s head)
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    row_to_head_id: Optional[Callable[[int], int]] = None,
    rows_to_check: int = 4,         # how many rows to sample (0 = all)
    pages_per_row: int = 3,         # pages per sampled row (0 = all)
    atol: float = 1e-2,
    rtol: float = 1e-2,
    check_values: bool = True,
    raise_on_fail: bool = False,
) -> Dict[str, int]:
    """
    Verifies that CPU->GPU staging placed the correct bytes into the expected slots.

    Assumptions:
      - CPU buffer layout is VTX paged: for a given (page_id, head_id),
            token_offset = (page_id * num_kv_heads + head_id) * page_size
            slice = [token_offset : token_offset + page_size]
      - Staging uses a single global array of pages in the order of 'indices' (absolute slots).
        For row r, row pages occupy slots [indptr[r] : indptr[r+1]).
      - gpu_k_staging/gpu_v_staging are the 3D (tokens, 1, head_dim) tensors (before reshaping to 4D).
        If you pass 4D tensors [-1, page_size, 1, head_dim], this function will flatten them.

    Returns:
      dict with counts of rows_checked, pages_checked, mismatched_pages.
    """
    assert cpu_k_buffer.device.type == "cpu" and cpu_v_buffer.device.type == "cpu", "CPU buffers must be on CPU"
    assert gpu_k_staging.is_cuda and gpu_v_staging.is_cuda, "staging buffers must be on CUDA device"
    assert cpu_k_buffer.dtype == torch.bfloat16 and cpu_v_buffer.dtype == torch.bfloat16
    assert gpu_k_staging.dtype == torch.bfloat16 and gpu_v_staging.dtype == torch.bfloat16
    assert indptr.dtype == torch.int32 and indices.dtype == torch.int32 and sparse_indices.dtype == torch.int32

    # Normalize shapes to (tokens, 1, head_dim)
    def to_T1H(x):
        if x.dim() == 4:
            # [-1, page, 1, head_dim] -> (pages*page, 1, head_dim)
            return x.contiguous().view(-1, 1, x.shape[-1])
        elif x.dim() == 3:
            return x
        else:
            raise ValueError(f"Unexpected staging dim={x.dim()}, expected 3 or 4")
    gk = to_T1H(gpu_k_staging)
    gv = to_T1H(gpu_v_staging)

    total_rows = indptr.shape[0] - 1
    assert total_rows >= 0
    nnz = indptr[-1].item()
    assert indices.shape[0] == nnz and sparse_indices.shape[0] == nnz

    if row_to_head_id is None:
        # Default ordering: rows are (req-major), inner is kv_head
        row_to_head_id = lambda r: (r % num_kv_heads)

    rows_to_iter = list(range(total_rows)) if rows_to_check == 0 else list(range(min(rows_to_check, total_rows)))

    mismatches = 0
    pages_checked = 0
    rows_checked = 0

    # Pre-copy staging to CPU once for fast compare
    gk_cpu = gk.detach().float().cpu()
    gv_cpu = gv.detach().float().cpu()
    ck = cpu_k_buffer.detach().float()  # CPU already
    cv = cpu_v_buffer.detach().float()

    for r in rows_to_iter:
        s = indptr[r].item()
        e = indptr[r + 1].item()
        row_len = e - s
        if row_len <= 0:
            continue

        head_id = row_to_head_id(r)
        # Which page slots in staging does this row occupy?
        row_slots = indices[s:e]  # absolute staging slots
        row_pages = sparse_indices[s:e]  # per-row page IDs (relative to this head)

        # Choose pages to sample in this row
        page_ids = list(range(row_len)) if pages_per_row == 0 else list(range(min(pages_per_row, row_len)))

        for j in page_ids:
            slot = row_slots[j].item()  # absolute slot in staging
            page_id = row_pages[j].item()

            # Compute CPU token range for (page_id, head_id)
            cpu_token_base = (page_id * num_kv_heads + head_id) * page_size
            cpu_slice = slice(cpu_token_base, cpu_token_base + page_size)

            # Compute GPU staging token range for 'slot'
            gpu_token_base = slot * page_size
            gpu_slice = slice(gpu_token_base, gpu_token_base + page_size)

            # Fetch chunks
            ck_chunk = ck[cpu_slice, 0, :head_dim]   # [page, head_dim]
            cv_chunk = cv[cpu_slice, 0, :head_dim]
            gk_chunk = gk_cpu[gpu_slice, 0, :head_dim]
            gv_chunk = gv_cpu[gpu_slice, 0, :head_dim]

            # Compare
            ok_k = torch.allclose(ck_chunk, gk_chunk, atol=atol, rtol=rtol)
            ok_v = torch.allclose(cv_chunk, gv_chunk, atol=atol, rtol=rtol)
            if not (ok_k and ok_v):
                mismatches += 1
                # Print a tiny diagnostic for the first few mismatches
                if mismatches <= 8:
                    max_k = (ck_chunk - gk_chunk).abs().max().item()
                    max_v = (cv_chunk - gv_chunk).abs().max().item()
                    print(f"[verify] row={r} head={head_id} j={j} slot={slot} page_id={page_id} "
                          f"ΔK_max={max_k:.4f} ΔV_max={max_v:.4f} "
                          f"cpu_base={cpu_token_base} gpu_base={gpu_token_base}")
            pages_checked += 1
        rows_checked += 1

    report = {
        "rows_checked": rows_checked,
        "pages_checked": pages_checked,
        "mismatched_pages": mismatches,
    }
    if raise_on_fail and mismatches > 0:
        raise AssertionError(f"verify_staging_mapping: {mismatches}/{pages_checked} page copies mismatched.")
    return report



class CPUVTXFlashInferAttnBackend(AttentionBackend):
    """
    CPU-based Vortex FlashInfer attention backend.

    Stores KV cache on CPU and transfers sparse pages to GPU for decode.
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
        self.num_wrappers = 1
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
        self.num_attn_groups = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.count = 0
        assert self.q_data_type == torch.bfloat16
        assert self.data_type == torch.bfloat16
        
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.server_args.page_size
        self.layers_skip = model_runner.server_args.vortex_layers_skip

        self.kv_indptr = [
                torch.zeros(
                    (max_bs * self.num_kv_heads + 1,), dtype=torch.int32, device=model_runner.device
                ),
                torch.zeros(
                    (max_bs * self.num_kv_heads + 1,), dtype=torch.int32, device=model_runner.device
                ),
            ]
        
        self.kv_indices = [
            torch.zeros(
                (
                    (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1),), 
                    dtype=torch.int32, device=model_runner.device
            ),
            torch.zeros(
                (
                    (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1),), 
                    dtype=torch.int32, device=model_runner.device
            ),
        ]

        self.kv_last_page_len = [
            torch.ones((max_bs * self.num_kv_heads,), dtype=torch.int32, device=model_runner.device),
            torch.ones((max_bs * self.num_kv_heads,), dtype=torch.int32, device=model_runner.device),
        ]

        # # For staging buffer: new indptr/indices to reference staged sparse KV
        # self.staging_kv_indices = torch.zeros(
        #     (
        #         (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1),), 
        #         dtype=torch.int32, device=model_runner.device
        # )

        self.qo_indptr = [
            torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device),
            torch.zeros(
                (max_bs * self.num_kv_heads + 1,), dtype=torch.int32, device=model_runner.device
            ),
        ]

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

        # Decode wrappers
        # [0]: sparse attention, [1]: dense attention (fallback)
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

        # Vortex sparse attention server
        min_chunk_size = 8
        max_chunk_size = 32
        max_seq_lengths = model_runner.model_config.context_len

        maximum_num_pages = (max_bs * max_seq_lengths * self.num_kv_heads // self.page_size) + max_bs * self.num_kv_heads
        maximum_num_workloads = (maximum_num_pages // min_chunk_size) + max_bs * self.num_kv_heads

        # Workload info buffers for vortex sparse attention
        self.winfo_q_indices = torch.zeros(
            (maximum_num_workloads,), dtype=torch.int32, device=model_runner.device)

        self.winfo_kv_offsets = torch.zeros(
            (maximum_num_workloads,), dtype=torch.int32, device=model_runner.device)

        self.winfo_kv_lens = torch.zeros(
            (maximum_num_workloads,), dtype=torch.int32, device=model_runner.device)

        self.winfo_num_workloads = torch.zeros(
            (1,), dtype=torch.int32, device=model_runner.device)

        self.winfo_chunk_size = torch.zeros(
            (1,), dtype=torch.int32, device=model_runner.device)

        self.buffer = torch.zeros(
            (maximum_num_pages,), dtype=torch.float32, device=model_runner.device
        )

        self.vtx_api = SparseAttentionServer(
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            num_qo_heads=self.num_qo_heads,
            page_size=model_runner.server_args.page_size,
            max_batch_size=max_bs,
            max_seq_lengths=max_seq_lengths,
            max_prefill_lengths=model_runner.server_args.max_prefill_tokens,
            max_num_tokens=model_runner.max_total_num_tokens,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            num_selected_pages=model_runner.server_args.vortex_num_selected_pages,
            page_reserved_bos=model_runner.server_args.vortex_page_reserved_bos,
            page_reserved_eos=model_runner.server_args.vortex_page_reserved_eos,
            max_num_pages_per_request=(model_runner.model_config.context_len + model_runner.server_args.page_size - 1) \
                // model_runner.server_args.page_size if model_runner.server_args.vortex_max_seq_lens < 0 \
                    else (model_runner.server_args.vortex_max_seq_lens + model_runner.server_args.page_size - 1) \
                        // model_runner.server_args.page_size,
            algo_name=model_runner.server_args.vortex_sparse_attention_algorithm
        )

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}
        self.draft_extend_cuda_graph_metadata = {}

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        assert not forward_batch.forward_mode.is_draft_extend()
        assert not forward_batch.forward_mode.is_target_verify()

        if forward_batch.forward_mode.is_decode_or_idle():
            bs = len(forward_batch.req_pool_indices)

            # Plan decode with vortex to get sparse indices (same as GPU CG version)
            self.vtx_api.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                dense_kv_indptr=self.kv_indptr[1][:bs * self.num_kv_heads + 1],
                dense_kv_indices=self.kv_indices[1],
                sparse_kv_indptr=self.kv_indptr[0][:bs * self.num_kv_heads + 1],
                sparse_kv_indices=self.kv_indices[0],
                kv_last_page_len=self.kv_last_page_len[1][:bs * self.num_kv_heads],
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                winfo_q_indices=self.winfo_q_indices,
                winfo_kv_offsets=self.winfo_kv_offsets,
                winfo_kv_lens=self.winfo_kv_lens,
                winfo_num_workload=self.winfo_num_workloads,
                winfo_chunk_size=self.winfo_chunk_size,
            )
            
            # print(f"bs: {bs}")
            # print(f"num_kv_heads: {self.num_kv_heads}")
            # print(f"dense_kv_indptr.shape: {self.kv_indptr[1].shape}")
            # print(f"sparse_kv_indptr.shape: {self.kv_indptr[0].shape}")
            # print(f"dense_kv_indptr used: {self.kv_indptr[1][:bs * self.num_kv_heads + 1].shape}")
            # print(f"sparse_kv_indptr used: {self.kv_indptr[0][:bs * self.num_kv_heads + 1].shape}")
            # print(f"dense_kv_indptr: {self.kv_indptr[1][:bs * self.num_kv_heads + 1]}")
            # print(f"sparse_kv_indptr: {self.kv_indptr[0][:bs * self.num_kv_heads + 1]}")
            # print(f"kv_last_page_len.shape: {self.kv_last_page_len[1].shape}")
            # print(f"kv_last_page_len used: {self.kv_last_page_len[1][:bs * self.num_kv_heads].shape}")
            # print(f"req_to_token.shape: {self.req_to_token.shape}")
            # print(f"req_indices.shape: {forward_batch.req_pool_indices.shape}")
            # print(f"cached_seq_lens.shape: {forward_batch.seq_lens.shape}")

            # Plan the decode wrapper for sparse attention with preset contiguous staging layout
            # With 3D layout [tokens*heads, 1, head_dim], we need per-head page indices
            # The staging buffer has num_sparse_pages global pages, each containing head_num heads
            # So total per-head pages = num_sparse_pages * head_num
            # num_sparse_pages = self.kv_indptr[0][bs * self.num_kv_heads].item()

            # # Build contiguous per-head page indices [0, 1, 2, ..., num_per_head_pages-1]
            # self.staging_kv_indices = torch.arange(
            #     num_sparse_pages, dtype=torch.int32, device=self.kv_indptr[0].device
            # )

            # Row-relative indices buffer
            rows = bs * self.num_kv_heads
            indptr = self.kv_indptr[0][:rows + 1]
            nnz = indptr[-1].item()

            # Absolute indices: for row r, values are [s, s+1, ..., e-1]
            staging_kv_indices = torch.empty(nnz, dtype=torch.int32, device=indptr.device)
            for r in range(rows):
                s = indptr[r].item(); e = indptr[r+1].item()
                if e > s:
                    staging_kv_indices[s:e] = torch.arange(s, e, device=indptr.device, dtype=torch.int32)

            # (Re)plan with these indices
            self.decode_wrappers[0].plan(
                indptr=indptr,
                indices=staging_kv_indices,
                last_page_len=self.kv_last_page_len[1][:rows],
                num_qo_heads=self.num_attn_groups,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

            # Plan the decode wrapper for dense attention (fallback)
            self.decode_wrappers[1].plan(
                indptr=self.kv_indptr[1][:bs * self.num_kv_heads + 1],
                indices=self.kv_indices[1],
                last_page_len=self.kv_last_page_len[1][:bs * self.num_kv_heads],
                num_qo_heads=self.num_attn_groups,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

            self.forward_metadata = DecodeMetadata(use_sparsity=True)

        else:
            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            bs = len(forward_batch.req_pool_indices)
            self.vtx_api.plan_prefill(
                cached_seq_lens=prefix_lens,
                dense_kv_indptr=self.kv_indptr[1][:bs*self.num_kv_heads+1],
                dense_kv_indices=self.kv_indices[1],
                input_seq_lens=(forward_batch.seq_lens.to(torch.int32) - prefix_lens),
                qo_indptr_ragged=self.qo_indptr[0][:bs+1],
                qo_indptr_paged=self.qo_indptr[1][:bs*self.num_kv_heads+1],
                kv_last_page_len=self.kv_last_page_len[0][:bs*self.num_kv_heads],
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices
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
                self.kv_indptr[1][:bs*self.num_kv_heads+1],
                self.kv_indices[1],
                self.kv_last_page_len[0][:bs*self.num_kv_heads],
                self.num_attn_groups,
                1,
                self.head_dim,
                self.page_size,  # Paged format: [num_pages, page_size, 1, head_dim]
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                custom_mask=None,
                non_blocking=False,
            )

            self.forward_metadata = PrefillMetadata(extend_no_prefix)

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, num_tokens: int, forward_batch: ForwardBatch
    ):
        # CUDA graph support - not implemented yet
        raise NotImplementedError("CUDA graph not yet supported for CPU VTX backend")

    def init_forward_metadata_replay_cuda_graph(
        self, bs: int, num_tokens: int, forward_batch: ForwardBatch
    ):
        # CUDA graph support - not implemented yet
        raise NotImplementedError("CUDA graph not yet supported for CPU VTX backend")

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
            
            q_t = self.vtx_api.chunkwise_NH2HN_transpose(
                q.view(-1, self.num_qo_heads, self.head_dim),
                self.qo_indptr[0]
            )
            
            o2, s2 = self.prefill_wrapper_paged.forward_return_lse(
                q_t,
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            o2_t, s2_t = self.vtx_api.chunkwise_HN2NH_transpose(
                o2, s2, self.qo_indptr[0]
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

        1. Save current K/V to CPU
        2. Calculate sparse indices with vortex (using GPU landmarks)
        3. Copy sparse KV pages from CPU to GPU staging buffer
        4. Run attention on GPU with staging buffer
        """
        assert not layer.is_cross_attention

        cache_loc = forward_batch.out_cache_loc
        bs = len(forward_batch.req_pool_indices)
        # print(bs)
        use_sparsity = (self.forward_metadata.use_sparsity) and (layer.layer_id not in self.layers_skip)

        # Save new K/V to CPU cache
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer_decode(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        if use_sparsity:
            q = q.view(-1, self.num_attn_groups, layer.head_dim).contiguous()
            landmarks = forward_batch.token_to_kv_pool.get_landmark_buffer(layer.layer_id)

            # Use matmul + topk_output like GPU CG version
            self.vtx_api.matmul(
                query=q,
                landmarks=landmarks,
                dense_kv_indptr=self.kv_indptr[1],
                dense_kv_indices=self.kv_indices[1],
                output=self.buffer,
                winfo_q_indices=self.winfo_q_indices,
                winfo_kv_offsets=self.winfo_kv_offsets,
                winfo_kv_lens=self.winfo_kv_lens,
                winfo_num_workload=self.winfo_num_workloads,
                winfo_chunk_size=self.winfo_chunk_size,
            )

            self.vtx_api.topk_output(
                score=self.buffer,
                dense_kv_indptr=self.kv_indptr[1],
                dense_kv_indices=self.kv_indices[1],
                sparse_kv_indptr=self.kv_indptr[0],
                sparse_kv_indices=self.kv_indices[0],
                eff_batch_size=q.shape[0],
            )

            result = forward_batch.token_to_kv_pool.copy_sparse_kv_to_gpu(
                layer_id=layer.layer_id,
                sparse_kv_indices=self.kv_indices[0],
                sparse_kv_indptr=self.kv_indptr[0],
                dst_kv_indices=self.decode_wrappers[0]._paged_kv_indices_buf,
                batch_size=bs,
            )

            k_staging, v_staging, staging_kv_indices = result
            # self.decode_wrappers[0]._paged_kv_indices_buf = staging_kv_indices
            # Staging buffers are already in paged format [num_pages, page_size, 1, head_dim]

            o = self.decode_wrappers[0].forward(
                q,
                (k_staging, v_staging),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )

        else:
            o = self.decode_wrappers[1].forward(
                q.contiguous().view(-1, self.num_attn_groups, layer.head_dim),
                (k, v),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )
        output = o.view(-1, layer.tp_q_head_num * layer.head_dim)

        return output
