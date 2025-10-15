"""
Triton kernels for efficient CPU<->GPU KV cache transfers.

Each block handles one page to maximize parallelism.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def cpu_to_gpu_sparse_copy_kernel(
    cpu_k_ptr, cpu_v_ptr,
    gpu_k_ptr, gpu_v_ptr,
    sparse_indices_ptr,     # page_id (per head)
    head_ids_ptr,           # head_id for each page
    NUM_KV_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPARSE_PAGES: tl.constexpr,
):
    token_idx = tl.program_id(0)
    page_idx = token_idx // PAGE_SIZE
    token_offset = token_idx % PAGE_SIZE
    if page_idx >= NUM_SPARSE_PAGES:
        return

    page_id = tl.load(sparse_indices_ptr + page_idx)
    head_id = tl.load(head_ids_ptr + page_idx)

    src_linear_idx = ((page_id * NUM_KV_HEADS) + head_id) * PAGE_SIZE + token_offset
    dst_linear_idx = page_idx * PAGE_SIZE + token_offset

    dim = tl.arange(0, HEAD_DIM)
    tl.store(gpu_k_ptr + dst_linear_idx * HEAD_DIM + dim,
             tl.load(cpu_k_ptr + src_linear_idx * HEAD_DIM + dim))
    tl.store(gpu_v_ptr + dst_linear_idx * HEAD_DIM + dim,
             tl.load(cpu_v_ptr + src_linear_idx * HEAD_DIM + dim))


def copy_sparse_kv_cpu_to_gpu(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    sparse_indices: torch.Tensor,
    head_ids: torch.Tensor,
    page_size: int,
    head_num: int,
):
    """
    Python wrapper for CPU->GPU sparse copy kernel.

    Args:
        cpu_k_buffer: CPU K cache [total_tokens * head_num, 1, head_dim]
        cpu_v_buffer: CPU V cache [total_tokens * head_num, 1, head_dim]
        gpu_k_staging: GPU staging buffer [max_tokens * head_num, 1, head_dim]
        gpu_v_staging: GPU staging buffer [max_tokens * head_num, 1, head_dim]
        sparse_indices: Per-head page indices to copy [num_sparse_pages]
        page_size: Number of tokens per page
        head_num: Number of KV heads (unused, kept for compatibility)
    """
    assert cpu_k_buffer.is_pinned(), "CPU buffer must be pinned memory"
    assert cpu_v_buffer.is_pinned(), "CPU buffer must be pinned memory"
    assert gpu_k_staging.is_cuda, "GPU staging must be on CUDA"
    assert gpu_v_staging.is_cuda, "GPU staging must be on CUDA"

    # Verify 3D layout
    assert cpu_k_buffer.dim() == 3 and cpu_k_buffer.shape[1] == 1
    assert gpu_k_staging.dim() == 3 and gpu_k_staging.shape[1] == 1

    num_sparse_pages = sparse_indices.shape[0]
    head_dim = cpu_k_buffer.shape[2]

    # Grid: one program per token-head entry across all sparse pages
    grid = (num_sparse_pages * page_size,)

    cpu_to_gpu_sparse_copy_kernel[grid](
        cpu_k_buffer,
        cpu_v_buffer,
        gpu_k_staging,
        gpu_v_staging,
        sparse_indices,
        head_ids,
        NUM_KV_HEADS=head_num,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        NUM_SPARSE_PAGES=num_sparse_pages,
    )
    
def build_head_ids_per_page(indptr: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    rows = indptr.numel() - 1
    out = torch.empty(indptr[-1].item(), dtype=torch.int32, device=indptr.device)
    for r in range(rows):
        s = int(indptr[r].item()); e = int(indptr[r+1].item())
        if e <= s: continue
        out[s:e] = r % num_kv_heads
    return out

def build_head_ids_per_page_head_major(indptr: torch.Tensor, bs: int) -> torch.Tensor:
    rows = indptr.numel() - 1
    nnz  = indptr[-1].item()
    out  = torch.empty(nnz, dtype=torch.int32, device=indptr.device)
    for r in range(rows):
        s = int(indptr[r]); e = int(indptr[r+1])
        if e <= s: continue
        head_id = r // bs              # <-- head-major instead of r % num_kv_heads
        out[s:e] = head_id
    return out

@torch.no_grad()
def build_head_ids_per_page_req_major(indptr: torch.Tensor,
                                      num_kv_heads: int) -> torch.Tensor:
    """
    Rows are request-major: r = req_id * num_kv_heads + head_id
    Returns int32 head_id per selected page (length = indptr[-1]).
    """
    rows = indptr.numel() - 1
    nnz  = int(indptr[-1].item())
    out  = torch.empty(nnz, dtype=torch.int32, device=indptr.device)
    for r in range(rows):
        s = int(indptr[r].item()); e = int(indptr[r+1].item())
        if e <= s: continue
        out[s:e] = (r % num_kv_heads)
    return out
    
@triton.jit
def set_kv_buffer_kernel(
    k_cache,
    v_cache,
    new_k,
    new_v,
    loc,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr
):
    
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)    
    dim = tl.arange(0, HEAD_DIM)
    
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr)
    src_v = tl.load(new_v + src_ptr)
    
    token_position = tl.load(loc + token_id)
    position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position %  PAGE_SIZE
    
    dst_k_ptr = k_cache + position_trans * HEAD_DIM + dim
    dst_v_ptr = v_cache + position_trans * HEAD_DIM + dim
    
    tl.store(dst_k_ptr, src_k)
    tl.store(dst_v_ptr, src_v)
    

def store_kv_gpu_to_cpu(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]
    
    set_kv_buffer_kernel[(NNZ, NUM_KV_HEAD)](
        cpu_k_buffer,
        cpu_v_buffer,
        new_k,
        new_v,
        loc,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )


def naive_copy(cpu_k, cpu_v, gpu_k_staging, gpu_v_staging, sparse_indices, page_size):
    """
    Vectorized baseline:
      1) Build row indices for all selected per-head pages (expand by page_size)
      2) CPU gather into compact pinned buffers via index_select (one op per K/V)
      3) One H2D copy per K/V into gpu_*_staging[:entries]
    """
    assert cpu_k.device.type == "cpu" and cpu_v.device.type == "cpu"
    assert cpu_k.is_pinned() and cpu_v.is_pinned()
    assert gpu_k_staging.is_cuda and gpu_v_staging.is_cuda
    assert cpu_k.dim() == 3 and cpu_k.shape[1] == 1 and cpu_k.shape[2] == gpu_k_staging.shape[2]
    assert cpu_v.shape == cpu_k.shape and gpu_v_staging.shape == gpu_k_staging.shape

    num_sparse_pages = int(sparse_indices.numel())
    head_dim = int(cpu_k.shape[2])
    entries = num_sparse_pages * page_size

    # Build row indices on CPU: each per-head page expands to page_size rows
    # rows shape: [entries]
    per_row = torch.arange(page_size, dtype=torch.long, device="cpu")
    rows = (sparse_indices.to(device="cpu", dtype=torch.long).unsqueeze(1) * page_size + per_row).reshape(-1)

    # Gather on CPU (pinned) → compact K/V blocks
    # (These two index_select calls are fast and vectorized.)
    k_compact = cpu_k.index_select(0, rows)  # [entries, 1, D]
    v_compact = cpu_v.index_select(0, rows)  # [entries, 1, D]

    # Single H2D per tensor (use non_blocking because source is pinned)
    gpu_k_staging[:entries].copy_(k_compact, non_blocking=True)
    gpu_v_staging[:entries].copy_(v_compact, non_blocking=True)


@triton.jit
def update_landmark_buffer_kernel(
    k_cache,
    landmark,
    loc,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr
):
    
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE !=0:
        return
    
    position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position %  PAGE_SIZE
    
    k_cache_offset = position_trans + 1 - PAGE_SIZE
    dim = tl.arange(0, HEAD_DIM)
    page = tl.arange(0, PAGE_SIZE)
    src_ptr = k_cache + k_cache_offset * HEAD_DIM + page[:,None] * HEAD_DIM + dim
    page_block = tl.load(src_ptr)                         # [PAGE_SIZE, HEAD_DIM]
    lmk = tl.sum(page_block, axis=0)
    lmk = lmk.to(tl.bfloat16)
    
    dst_ptr = landmark + ((token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id) * HEAD_DIM + dim
    tl.store(dst_ptr, lmk)

def update_landmark_from_cpu(
    cpu_k_buffer: torch.Tensor,
    gpu_landmark: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    num_kv_head: int,
    head_dim: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_head
    HEAD_DIM = head_dim
    
    update_landmark_buffer_kernel[(NNZ, NUM_KV_HEAD)](
        cpu_k_buffer,
        gpu_landmark,
        loc,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size
    )
