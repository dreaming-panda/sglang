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
    sparse_indices_ptr,     # per-head page indices from Vortex API
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPARSE_PAGES: tl.constexpr,
):
    """
    Copy sparse KV pages from CPU to GPU.

    sparse_indices contains per-head page indices (same indices used by FlashInfer).
    Both CPU and GPU buffers have shape [total_entries, 1, head_dim] where
    total_entries = num_per_head_pages * page_size.
    """
    token_idx = tl.program_id(0)
    page_idx = token_idx // PAGE_SIZE # which sparse page it belongs to
    token_offset = token_idx % PAGE_SIZE # offset within the page
    if page_idx >= NUM_SPARSE_PAGES:
        return

    # sparse_indices[page_idx] is already a per-head page index
    src_per_head_page = tl.load(sparse_indices_ptr + page_idx)

    # Linear index in the buffer: per_head_page * page_size + token_offset
    src_linear_idx = src_per_head_page * PAGE_SIZE + token_offset
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
    page_size: int,
):
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
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        NUM_SPARSE_PAGES=num_sparse_pages,
    )
    
@triton.jit
def cpu_to_gpu_sparse_copy_kernel_tiled(
    cpu_k_ptr, cpu_v_ptr,
    gpu_k_ptr, gpu_v_ptr,
    sparse_indices_ptr,            # [NUM_SPARSE_PAGES]
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPARSE_PAGES: tl.constexpr,
    BLOCK_T: tl.constexpr,         # tokens per program (tile height)
    BLOCK_D: tl.constexpr          # head-dim elements per iteration (tile width)
):
    # 2D launch: (page_id, token_tile_id)
    pid_page = tl.program_id(0)
    pid_tok_tile = tl.program_id(1)
    if pid_page >= NUM_SPARSE_PAGES:
        return

    # Which page we pull from on CPU (per-head sparse page index):
    src_per_head_page = tl.load(sparse_indices_ptr + pid_page)

    # Token offsets within the page for this tile
    tok_offsets = pid_tok_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    tok_mask = tok_offsets < PAGE_SIZE

    # Base linear indices for src/dst (WITHOUT head-dim yet)
    src_lin_base = src_per_head_page * PAGE_SIZE + tok_offsets
    dst_lin_base = pid_page         * PAGE_SIZE + tok_offsets

    # Sweep across head_dim in BLOCK_D stripes
    for d0 in range(0, HEAD_DIM, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < HEAD_DIM

        # Build 2D (tokens x dims) pointer grids
        # indices shape: [BLOCK_T, BLOCK_D]
        src_idx = (src_lin_base[:, None] * HEAD_DIM) + d_offsets[None, :]
        dst_idx = (dst_lin_base[:, None] * HEAD_DIM) + d_offsets[None, :]

        mask = tok_mask[:, None] & d_mask[None, :]

        # Load a TILE from CPU pinned memory and store to GPU
        k_tile = tl.load(cpu_k_ptr + src_idx, mask=mask, other=0)
        v_tile = tl.load(cpu_v_ptr + src_idx, mask=mask, other=0)
        tl.store(gpu_k_ptr + dst_idx, k_tile, mask=mask)
        tl.store(gpu_v_ptr + dst_idx, v_tile, mask=mask)


def copy_sparse_kv_cpu_to_gpu_tiled(
    cpu_k_buffer,  # [total_entries, 1, head_dim], pinned
    cpu_v_buffer,  # [total_entries, 1, head_dim], pinned
    gpu_k_staging, # [num_sparse_pages*page_size, 1, head_dim], cuda
    gpu_v_staging, # [num_sparse_pages*page_size, 1, head_dim], cuda
    sparse_indices,# [num_sparse_pages], int32/int64
    page_size: int,
    block_t: int = 64,     # tune: 32/64/128 are common choices
    block_d: int = 128,    # tune: multiples of 32/64/128 work well
    num_warps: int = 4,    # tune: 4/8 (A100/H100 often like 4–8)
    num_stages: int = 2    # tune: 2–4 to overlap mem
):
    assert cpu_k_buffer.is_pinned() and cpu_v_buffer.is_pinned()
    assert gpu_k_staging.is_cuda and gpu_v_staging.is_cuda
    assert cpu_k_buffer.dim() == 3 and cpu_k_buffer.shape[1] == 1
    assert gpu_k_staging.dim() == 3 and gpu_k_staging.shape[1] == 1

    num_sparse_pages = int(sparse_indices.shape[0])
    head_dim = int(cpu_k_buffer.shape[2])

    # Flatten the trivial middle dim for pointer arithmetic
    cpu_k_flat = cpu_k_buffer.view(-1, head_dim)
    cpu_v_flat = cpu_v_buffer.view(-1, head_dim)
    gpu_k_flat = gpu_k_staging.view(-1, head_dim)
    gpu_v_flat = gpu_v_staging.view(-1, head_dim)

    # Grid: pages × token-tiles-per-page
    tiles_per_page = (page_size + block_t - 1) // block_t
    grid = (num_sparse_pages, tiles_per_page)

    cpu_to_gpu_sparse_copy_kernel_tiled[grid](
        cpu_k_flat, cpu_v_flat,
        gpu_k_flat, gpu_v_flat,
        sparse_indices,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        NUM_SPARSE_PAGES=num_sparse_pages,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
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


@triton.jit
def cpu_to_gpu_sparse_copy_with_slots_kernel(
    cpu_k_ptr, cpu_v_ptr,
    gpu_k_ptr, gpu_v_ptr,
    src_page_ids_ptr,        # [num_pages_to_copy] - CPU page IDs to copy
    dst_staging_slots_ptr,   # [num_pages_to_copy] - staging slots to write to
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPARSE_PAGES: tl.constexpr,
):
    """
    Copy specific pages from CPU to specific staging slots on GPU.

    Unlike the contiguous version, this allows non-contiguous placement
    in the staging buffer based on cache slot allocation.
    """
    token_idx = tl.program_id(0)
    page_idx = token_idx // PAGE_SIZE # which sparse page it belongs to
    token_offset = token_idx % PAGE_SIZE # offset within the page
    if page_idx >= NUM_SPARSE_PAGES:
        return

    # Which CPU page to copy from
    src_page_id = tl.load(src_page_ids_ptr + page_idx)
    # Which staging slot to copy to
    dst_slot = tl.load(dst_staging_slots_ptr + page_idx)

    # Linear indices in the buffers
    src_linear_idx = src_page_id * PAGE_SIZE + token_offset
    dst_linear_idx = dst_slot * PAGE_SIZE + token_offset

    dim = tl.arange(0, HEAD_DIM)
    tl.store(gpu_k_ptr + dst_linear_idx * HEAD_DIM + dim,
             tl.load(cpu_k_ptr + src_linear_idx * HEAD_DIM + dim))
    tl.store(gpu_v_ptr + dst_linear_idx * HEAD_DIM + dim,
             tl.load(cpu_v_ptr + src_linear_idx * HEAD_DIM + dim))


def copy_pages_to_staging_slots(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    src_page_ids: torch.Tensor,      # [num_pages] - which CPU pages to copy
    dst_staging_slots: torch.Tensor, # [num_pages] - which staging slots to write to
    page_size: int,
):
    """
    Copy specific pages from CPU to specific staging buffer slots on GPU.
    Used for cache-based copying where slots may be non-contiguous.
    """
    assert cpu_k_buffer.is_pinned() and cpu_v_buffer.is_pinned()
    assert gpu_k_staging.is_cuda and gpu_v_staging.is_cuda
    assert cpu_k_buffer.dim() == 3 and cpu_k_buffer.shape[1] == 1
    assert gpu_k_staging.dim() == 3 and gpu_k_staging.shape[1] == 1

    num_sparse_pages = src_page_ids.shape[0]
    if num_sparse_pages == 0:
        return

    head_dim = int(cpu_k_buffer.shape[2])
    grid = (num_sparse_pages * page_size,)

    cpu_to_gpu_sparse_copy_with_slots_kernel[grid](
        cpu_k_buffer,
        cpu_v_buffer,
        gpu_k_staging,
        gpu_v_staging,
        src_page_ids,
        dst_staging_slots,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        NUM_SPARSE_PAGES=num_sparse_pages,
    )


@triton.jit
def assign_staging_slots_kernel(
    src_page_ids_ptr,        # [num_pages] - CPU page IDs (may have duplicates)
    staging_slot_map_ptr,    # [max_page_id] - maps page_id -> staging_slot (-1 if not assigned yet)
    slot_counter_ptr,        # [1] - atomic counter for assigning new staging slots
    dst_staging_slots_ptr,   # [num_pages] - OUTPUT: staging slots for each input page
    MAX_PAGE_ID: tl.constexpr,
    NUM_PAGES: tl.constexpr,
):
    """
    First pass: Assign staging slots to unique pages.
    One thread per page_idx in the input.
    """
    page_idx = tl.program_id(0)

    if page_idx >= NUM_PAGES:
        return

    src_page_id = tl.load(src_page_ids_ptr + page_idx)

    # Clamp to valid range
    if src_page_id < 0 or src_page_id >= MAX_PAGE_ID:
        tl.store(dst_staging_slots_ptr + page_idx, 0)  # Default to slot 0
        return

    # Check if this page_id already has a staging slot assigned
    existing_slot = tl.load(staging_slot_map_ptr + src_page_id)

    if existing_slot == -1:
        # This page needs a new slot - atomically claim one
        new_slot = tl.atomic_add(slot_counter_ptr, 1)
        # Try to be the first to assign this slot to this page_id
        old_slot = tl.atomic_cas(staging_slot_map_ptr + src_page_id, -1, new_slot)
        if old_slot == -1:
            # We won - use our new_slot
            my_slot = new_slot
        else:
            # Someone else won - use their slot
            my_slot = old_slot
    else:
        # Slot already assigned (duplicate page)
        my_slot = existing_slot

    # Store the staging slot for this position in the output
    tl.store(dst_staging_slots_ptr + page_idx, my_slot)


@triton.jit
def copy_with_assigned_slots_kernel(
    cpu_k_ptr, cpu_v_ptr,
    gpu_k_ptr, gpu_v_ptr,
    src_page_ids_ptr,        # [num_pages] - CPU page IDs
    dst_staging_slots_ptr,   # [num_pages] - staging slots for each page
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_PAGES: tl.constexpr,
):
    """
    Second pass: Copy data using pre-assigned staging slots.
    """
    token_idx = tl.program_id(0)
    page_idx = token_idx // PAGE_SIZE
    token_offset = token_idx % PAGE_SIZE

    if page_idx >= NUM_PAGES:
        return

    src_page_id = tl.load(src_page_ids_ptr + page_idx)
    dst_slot = tl.load(dst_staging_slots_ptr + page_idx)

    # Copy the data
    src_linear_idx = src_page_id * PAGE_SIZE + token_offset
    dst_linear_idx = dst_slot * PAGE_SIZE + token_offset

    dim = tl.arange(0, HEAD_DIM)
    tl.store(gpu_k_ptr + dst_linear_idx * HEAD_DIM + dim,
             tl.load(cpu_k_ptr + src_linear_idx * HEAD_DIM + dim))
    tl.store(gpu_v_ptr + dst_linear_idx * HEAD_DIM + dim,
             tl.load(cpu_v_ptr + src_linear_idx * HEAD_DIM + dim))


def copy_pages_to_staging_slots_dedup(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    src_page_ids: torch.Tensor,      # [num_pages] - which CPU pages to copy (may have duplicates)
    page_size: int,
    max_page_id: int,
):
    """
    Copy pages from CPU to GPU with automatic deduplication.

    Uses an atomic slot counter and page_id->slot mapping to ensure each unique
    page is copied only once, even if it appears multiple times in src_page_ids.

    Returns:
        staging_slots: [num_pages] tensor mapping each src_page_ids[i] to its staging slot
    """
    assert cpu_k_buffer.is_pinned() and cpu_v_buffer.is_pinned()
    assert gpu_k_staging.is_cuda and gpu_v_staging.is_cuda
    assert cpu_k_buffer.dim() == 3 and cpu_k_buffer.shape[1] == 1
    assert gpu_k_staging.dim() == 3 and gpu_k_staging.shape[1] == 1

    num_pages = src_page_ids.shape[0]
    if num_pages == 0:
        return torch.empty(0, dtype=torch.int32, device=src_page_ids.device)

    # Allocate mapping: page_id -> staging_slot (-1 = not assigned)
    staging_slot_map = torch.full((max_page_id,), -1, dtype=torch.int32, device=gpu_k_staging.device)

    # Atomic counter for assigning new staging slots
    slot_counter = torch.zeros(1, dtype=torch.int32, device=gpu_k_staging.device)

    # Output: staging slot for each input page
    dst_staging_slots = torch.zeros(num_pages, dtype=torch.int32, device=gpu_k_staging.device)

    head_dim = int(cpu_k_buffer.shape[2])

    # Pass 1: Assign staging slots (one thread per page)
    assign_staging_slots_kernel[(num_pages,)](
        src_page_ids,
        staging_slot_map,
        slot_counter,
        dst_staging_slots,
        MAX_PAGE_ID=max_page_id,
        NUM_PAGES=num_pages,
    )

    # Pass 2: Copy data (all threads, using assigned slots)
    grid = (num_pages * page_size,)
    copy_with_assigned_slots_kernel[grid](
        cpu_k_buffer,
        cpu_v_buffer,
        gpu_k_staging,
        gpu_v_staging,
        src_page_ids,
        dst_staging_slots,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        NUM_PAGES=num_pages,
    )

    return dst_staging_slots
