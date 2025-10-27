"""
Triton kernels for efficient CPU<->GPU KV cache transfers.

Each block handles one page to maximize parallelism.
"""

import torch
import triton
import triton.language as tl
import vortex_C
import time


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


@triton.jit
def set_kv_buffer_cpu_and_gpu_kernel(
    cpu_k_cache,
    cpu_v_cache,
    gpu_k_staging,
    gpu_v_staging,
    new_k,
    new_v,
    loc,
    cpu_to_gpu_slot_map_ptr,
    NUM_KV_HEAD: tl.constexpr,
    NNZ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGE_ID: tl.constexpr,
):
    """
    Fused kernel: Store KV data to CPU and update GPU cache if present.

    If the page is already in GPU staging buffer, we update both CPU and GPU
    simultaneously instead of invalidating and re-copying later.
    """
    token_id = tl.program_id(0)
    if token_id >= NNZ:
        return
    head_id = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)

    # Load source K/V from GPU
    src_ptr = token_id * NUM_KV_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim
    src_k = tl.load(new_k + src_ptr)
    src_v = tl.load(new_v + src_ptr)

    # Compute destination position in CPU cache
    token_position = tl.load(loc + token_id)
    cpu_position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) + \
        head_id * PAGE_SIZE + token_position % PAGE_SIZE

    # Always store to CPU cache
    cpu_dst_k_ptr = cpu_k_cache + cpu_position_trans * HEAD_DIM + dim
    cpu_dst_v_ptr = cpu_v_cache + cpu_position_trans * HEAD_DIM + dim
    tl.store(cpu_dst_k_ptr, src_k)
    tl.store(cpu_dst_v_ptr, src_v)

    # Check if this page is also in GPU staging buffer
    page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id

    if page_id < MAX_PAGE_ID:
        # Read the GPU slot for this page
        gpu_slot = tl.load(cpu_to_gpu_slot_map_ptr + page_id)

        # If this page is cached in GPU, also update the GPU copy
        if gpu_slot >= 0:
            # Compute position in GPU staging buffer
            gpu_position_trans = gpu_slot * PAGE_SIZE + (token_position % PAGE_SIZE)

            # Update GPU staging buffer
            gpu_dst_k_ptr = gpu_k_staging + gpu_position_trans * HEAD_DIM + dim
            gpu_dst_v_ptr = gpu_v_staging + gpu_position_trans * HEAD_DIM + dim
            tl.store(gpu_dst_k_ptr, src_k)
            tl.store(gpu_dst_v_ptr, src_v)


def store_kv_cpu_and_gpu(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    loc: torch.LongTensor,
    page_size: int,
    cpu_to_gpu_slot_map: torch.Tensor,
    max_page_id: int,
):
    """
    Fused operation: Store KV data to CPU cache and update GPU staging buffer if present.

    This kernel writes to both CPU and GPU simultaneously:
    1. Always writes new K/V data to CPU cache pages
    2. If the page is already in GPU staging buffer, also updates the GPU copy

    This avoids the need to invalidate and re-copy later, keeping CPU and GPU in sync.

    Args:
        cpu_k_buffer: CPU K cache [num_pages, num_heads, page_size, head_dim]
        cpu_v_buffer: CPU V cache [num_pages, num_heads, page_size, head_dim]
        gpu_k_staging: GPU K staging buffer [num_slots * page_size, head_dim]
        gpu_v_staging: GPU V staging buffer [num_slots * page_size, head_dim]
        new_k: New K data from GPU [batch_size, num_heads, head_dim]
        new_v: New V data from GPU [batch_size, num_heads, head_dim]
        loc: Token positions in cache [batch_size]
        page_size: Tokens per page
        cpu_to_gpu_slot_map: CPU page ID -> GPU slot mapping [-1 if not cached]
        max_page_id: Maximum valid page ID
    """
    NNZ = loc.shape[0]
    NUM_KV_HEAD = new_k.shape[1]
    HEAD_DIM = new_k.shape[2]

    set_kv_buffer_cpu_and_gpu_kernel[(NNZ, NUM_KV_HEAD)](
        cpu_k_buffer,
        cpu_v_buffer,
        gpu_k_staging,
        gpu_v_staging,
        new_k,
        new_v,
        loc,
        cpu_to_gpu_slot_map,
        NUM_KV_HEAD,
        NNZ,
        HEAD_DIM,
        page_size,
        max_page_id,
    )


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
def mark_and_allocate_unique_kernel(
    src_page_ids_ptr,
    cpu_to_gpu_slot_map_ptr,
    gpu_to_cpu_page_map_ptr,
    available_slots_ptr,
    alloc_counter_ptr,
    owners_bitmap_ptr,
    overflow_flag_ptr,
    MAX_PAGE_ID: tl.constexpr,
    N: tl.constexpr,
    NUM_AVAILABLE: tl.constexpr,
):
    """
    Mark unique pages and allocate slots.

    Key insight: We need to copy ALL unique pages in this batch, including cache hits.
    Only skip duplicates WITHIN this batch.
    """
    idx = tl.program_id(0)
    if idx >= N:
        return

    pid = tl.load(src_page_ids_ptr + idx)
    if (pid < 0) or (pid >= MAX_PAGE_ID):
        tl.store(owners_bitmap_ptr + idx, 0)
        return

    # First, check if already cached (non-atomic read is fast)
    existing_slot = tl.load(cpu_to_gpu_slot_map_ptr + pid)

    # If already cached (>= 0) or being allocated by another thread (-2), not an owner
    if existing_slot != -1:
        # Cache hit from previous batch - no need to copy
        tl.store(owners_bitmap_ptr + idx, 0)
        return

    # Not cached (-1), try to claim ownership with CAS
    prev = tl.atomic_cas(cpu_to_gpu_slot_map_ptr + pid, -1, -2)

    if prev == -1:
        # Allocate a new slot
        alloc_idx = tl.atomic_add(alloc_counter_ptr, 1)

        if alloc_idx >= NUM_AVAILABLE:
            # Overflow - revert the sentinel
            tl.store(cpu_to_gpu_slot_map_ptr + pid, -1)
            tl.store(owners_bitmap_ptr + idx, 0)
            tl.store(overflow_flag_ptr, 1)
            return

        # Get the slot from available_slots
        new_slot = tl.load(available_slots_ptr + alloc_idx)

        # Eviction: clear old mapping if this slot was occupied
        old_pid = tl.load(gpu_to_cpu_page_map_ptr + new_slot)
        if (old_pid >= 0) and (old_pid < MAX_PAGE_ID):
            tl.store(cpu_to_gpu_slot_map_ptr + old_pid, -1)

        # Set new bidirectional mapping
        tl.store(gpu_to_cpu_page_map_ptr + new_slot, pid)
        tl.store(cpu_to_gpu_slot_map_ptr + pid, new_slot)

        # Mark as owner (should copy)
        tl.store(owners_bitmap_ptr + idx, 1)
    else:
        # Lost the race - another thread claimed it first
        tl.store(owners_bitmap_ptr + idx, 0)

@triton.jit
def materialize_slots_kernel(
    src_page_ids_ptr,
    cpu_to_gpu_slot_map_ptr,
    dst_staging_slots_ptr,
    N: tl.constexpr,
    MAX_PAGE_ID: tl.constexpr,
):
    idx = tl.program_id(0)
    if idx >= N:
        return
    
    pid = tl.load(src_page_ids_ptr + idx)
    if (pid < 0) or (pid >= MAX_PAGE_ID):
        # Invalid page ID - use -1 as error marker
        tl.store(dst_staging_slots_ptr + idx, -1)
        return
    
    slot = tl.load(cpu_to_gpu_slot_map_ptr + pid)
    tl.store(dst_staging_slots_ptr + idx, slot)

@triton.jit
def copy_with_assigned_slots_kernel(
    cpu_k_ptr, cpu_v_ptr,
    gpu_k_ptr, gpu_v_ptr,
    src_page_ids_ptr,
    dst_staging_slots_ptr,
    should_copy_bitmap_ptr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_PAGES: tl.constexpr,
):
    token_idx = tl.program_id(0)
    page_idx = token_idx // PAGE_SIZE
    token_offset = token_idx % PAGE_SIZE

    if page_idx >= NUM_PAGES:
        return

    should_copy = tl.load(should_copy_bitmap_ptr + page_idx)
    
    if should_copy == 1:
        src_page_id = tl.load(src_page_ids_ptr + page_idx)
        dst_slot = tl.load(dst_staging_slots_ptr + page_idx)
        
        # Only copy if we have a valid slot
        if dst_slot >= 0:
            src_linear_idx = src_page_id * PAGE_SIZE + token_offset
            dst_linear_idx = dst_slot * PAGE_SIZE + token_offset

            dim = tl.arange(0, HEAD_DIM)
            tl.store(gpu_k_ptr + dst_linear_idx * HEAD_DIM + dim,
                     tl.load(cpu_k_ptr + src_linear_idx * HEAD_DIM + dim))
            tl.store(gpu_v_ptr + dst_linear_idx * HEAD_DIM + dim,
                     tl.load(cpu_v_ptr + src_linear_idx * HEAD_DIM + dim))

@triton.jit
def mark_and_allocate_unique_kernel_lru_sharded(
    src_page_ids_ptr,
    cpu_to_gpu_slot_map_ptr,
    gpu_to_cpu_page_map_ptr,
    available_slots_ptr,
    slot_ages_ptr,
    alloc_counters_ptr,
    bucket_offsets_ptr,
    bucket_sizes_ptr,
    owners_bitmap_ptr,
    overflow_flag_ptr,
    MAX_PAGE_ID: tl.constexpr,
    N: tl.constexpr,
    NB: tl.constexpr,
):
    """
    Sharded allocation with LRU age tracking.
    Each bucket = 32 slots (1 warp), pre-sorted by age (oldest first).
    """
    i = tl.program_id(0)
    if i >= N:
        return

    pid = tl.load(src_page_ids_ptr + i).to(tl.int32)
    if (pid < 0) or (pid >= MAX_PAGE_ID):
        tl.store(owners_bitmap_ptr + i, 0)
        return

    # Check for cache hit
    slot = tl.load(cpu_to_gpu_slot_map_ptr + pid)
    if slot != -1:
        # Cache hit: update age to 32 (most recent)
        tl.store(slot_ages_ptr + slot, 32)
        tl.store(owners_bitmap_ptr + i, 0)
        return

    # Try to claim this page
    prev = tl.atomic_cas(cpu_to_gpu_slot_map_ptr + pid, -1, -2)
    if prev != -1:
        tl.store(owners_bitmap_ptr + i, 0)
        return

    # Hash to bucket (each bucket = 32 slots)
    # Use simple modulo hash - works for any NB
    bucket = pid % NB
    if bucket < 0:
        bucket = -bucket
    base = tl.load(bucket_offsets_ptr + bucket)
    size = tl.load(bucket_sizes_ptr + bucket)

    # Allocate from this bucket (slots are pre-sorted by age, oldest first)
    local_idx = tl.atomic_add(alloc_counters_ptr + bucket, 1)
    if local_idx >= size:
        # Overflow
        tl.store(cpu_to_gpu_slot_map_ptr + pid, -1)
        tl.atomic_max(overflow_flag_ptr, 1)
        tl.store(owners_bitmap_ptr + i, 0)
        return

    # Get slot from sorted available_slots (oldest slots at beginning = LRU)
    slot = tl.load(available_slots_ptr + base + local_idx)

    old_pid = tl.load(gpu_to_cpu_page_map_ptr + slot)
    if (old_pid >= 0) & (old_pid < MAX_PAGE_ID):
        tl.store(cpu_to_gpu_slot_map_ptr + old_pid, -1)

    tl.store(gpu_to_cpu_page_map_ptr + slot, pid)
    tl.store(cpu_to_gpu_slot_map_ptr + pid, slot)

    tl.store(slot_ages_ptr + slot, 32)
    tl.store(owners_bitmap_ptr + i, 1)


@triton.jit
def mark_and_allocate_unique_kernel_sharded(
    src_page_ids_ptr,
    cpu_to_gpu_slot_map_ptr,
    gpu_to_cpu_page_map_ptr,
    available_slots_ptr,
    alloc_counters_ptr,
    bucket_offsets_ptr,
    bucket_sizes_ptr,
    owners_bitmap_ptr,
    overflow_flag_ptr,
    MAX_PAGE_ID: tl.constexpr,
    N: tl.constexpr,
    NB: tl.constexpr,
):
    """Original sharded allocation without LRU."""
    i = tl.program_id(0)
    if i >= N:
        return

    pid = tl.load(src_page_ids_ptr + i).to(tl.int32)
    if (pid < 0) or (pid >= MAX_PAGE_ID):
        tl.store(owners_bitmap_ptr + i, 0)
        return

    slot = tl.load(cpu_to_gpu_slot_map_ptr + pid)
    if slot != -1:
        tl.store(owners_bitmap_ptr + i, 0)
        return

    prev = tl.atomic_cas(cpu_to_gpu_slot_map_ptr + pid, -1, -2)
    if prev != -1:
        tl.store(owners_bitmap_ptr + i, 0)
        return

    bucket = pid & (NB - 1)
    base = tl.load(bucket_offsets_ptr + bucket)
    size = tl.load(bucket_sizes_ptr + bucket)

    local_idx = tl.atomic_add(alloc_counters_ptr + bucket, 1)
    if local_idx >= size:
        tl.store(cpu_to_gpu_slot_map_ptr + pid, -1)
        tl.atomic_max(overflow_flag_ptr, 1)
        tl.store(owners_bitmap_ptr + i, 0)
        return

    slot = tl.load(available_slots_ptr + base + local_idx)
    old_pid = tl.load(gpu_to_cpu_page_map_ptr + slot)
    if (old_pid >= 0) & (old_pid < MAX_PAGE_ID):
        tl.store(cpu_to_gpu_slot_map_ptr + old_pid, -1)

    tl.store(gpu_to_cpu_page_map_ptr + slot, pid)
    tl.store(cpu_to_gpu_slot_map_ptr + pid, slot)
    tl.store(owners_bitmap_ptr + i, 1)


def copy_pages_to_staging_slots_dedup(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    src_page_ids: torch.Tensor,
    cpu_to_gpu_slot_map: torch.Tensor,
    gpu_to_cpu_page_map: torch.Tensor,
    available_slots: torch.Tensor,
    page_size: int,
    max_page_id: int,
    owners_bitmap: torch.Tensor,
    dst_staging_slots: torch.Tensor,
    alloc_counter: torch.Tensor,
    overflow_flag: torch.Tensor,
    # Pre-allocated random eviction buffers (reused across iterations)
    bucket_sizes: torch.Tensor,
    bucket_offsets: torch.Tensor,
    alloc_counters: torch.Tensor,
    nb: int,
):
    assert cpu_k_buffer.is_pinned() and cpu_v_buffer.is_pinned()
    assert gpu_k_staging.is_cuda and gpu_v_staging.is_cuda
    assert cpu_k_buffer.dim() == 3 and cpu_k_buffer.shape[1] == 1
    assert gpu_k_staging.dim() == 3 and gpu_k_staging.shape[1] == 1

    num_pages = src_page_ids.shape[0]
    alloc_counter[0] = 0
    overflow_flag[0] = 0

    head_dim = int(cpu_k_buffer.shape[2])

    # Reset alloc counters (bucket_sizes and bucket_offsets are constant)
    alloc_counters.zero_()

    # Use CUDA events for accurate GPU timing
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    unique_pages = torch.unique(src_page_ids)
    num_unique_pages = unique_pages.numel()
    slots_before = cpu_to_gpu_slot_map[unique_pages]
    num_cache_misses = (slots_before == -1).sum().item()

    # start_event.record()
    # Pass A: Mark unique pages and allocate slots
    mark_and_allocate_unique_kernel_sharded[(num_pages,)](
        src_page_ids,
        cpu_to_gpu_slot_map,
        gpu_to_cpu_page_map,
        available_slots,
        alloc_counters,
        bucket_offsets,
        bucket_sizes,
        owners_bitmap,
        overflow_flag,
        MAX_PAGE_ID=max_page_id,
        N=num_pages,
        NB=nb,
    )
    # end_event.record()
    # end_event.synchronize()
    # final = start_event.elapsed_time(end_event)
    # print(f"[DEBUG] Pass A took {final:.4f} ms")
    
    num_allocated = alloc_counters.sum().item()
    print(f"[DEBUG] Pages: total={num_pages}, unique={num_unique_pages}, cache_misses={num_cache_misses}, allocated={num_allocated}")
    # Check for overflow
    if overflow_flag[0] == 1:
        print("Warning: GPU slot overflow detected!")

    # start_event.record()
    # Pass B: Materialize final slots for all positions
    materialize_slots_kernel[(num_pages,)](
        src_page_ids,
        cpu_to_gpu_slot_map,
        dst_staging_slots,
        N=num_pages,
        MAX_PAGE_ID=max_page_id,
    )
    # end_event.record()
    # end_event.synchronize()
    # final = start_event.elapsed_time(end_event)
    # print(f"[DEBUG] Pass B took {final:.4f} ms")

    if torch.any(dst_staging_slots < 0):
        print("Warning: Some pages failed to allocate staging slots!")

    # start_event.record()
    # Pass C: Copy only unique pages (owners)
    copy_with_assigned_slots_kernel[(num_pages * page_size,)](
        cpu_k_buffer, cpu_v_buffer,
        gpu_k_staging, gpu_v_staging,
        src_page_ids,
        dst_staging_slots,
        owners_bitmap,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        NUM_PAGES=num_pages,
    )
    # end_event.record()
    # end_event.synchronize()
    # final = start_event.elapsed_time(end_event)
    # print(f"[DEBUG] Pass C took {final:.4f} ms")

    # Return only the portion we actually used
    return dst_staging_slots[:num_pages]


def copy_pages_to_staging_slots_dedup_lru(
    cpu_k_buffer: torch.Tensor,
    cpu_v_buffer: torch.Tensor,
    gpu_k_staging: torch.Tensor,
    gpu_v_staging: torch.Tensor,
    src_page_ids: torch.Tensor,
    cpu_to_gpu_slot_map: torch.Tensor,
    gpu_to_cpu_page_map: torch.Tensor,
    available_slots: torch.Tensor,  # 1D array of available slots
    slot_ages: torch.Tensor,  # [total_slots] - age of each slot
    page_size: int,
    max_page_id: int,
    owners_bitmap: torch.Tensor,
    dst_staging_slots: torch.Tensor,
    alloc_counter: torch.Tensor,
    overflow_flag: torch.Tensor,
    # Pre-allocated LRU buffers (reused across iterations)
    warp_start_indices: torch.Tensor,
    warp_empty_counts: torch.Tensor,
    bucket_sizes: torch.Tensor,
    bucket_offsets: torch.Tensor,
    alloc_counters: torch.Tensor,
    num_warps: int,
):
    assert cpu_k_buffer.is_pinned() and cpu_v_buffer.is_pinned()
    assert gpu_k_staging.is_cuda and gpu_v_staging.is_cuda
    assert cpu_k_buffer.dim() == 3 and cpu_k_buffer.shape[1] == 1
    assert gpu_k_staging.dim() == 3 and gpu_k_staging.shape[1] == 1

    num_pages = src_page_ids.shape[0]
    alloc_counter[0] = 0
    overflow_flag[0] = 0

    head_dim = int(cpu_k_buffer.shape[2])

    # Use CUDA events for accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Import vortex_C module for CUDA warp sorting kernel
    import vortex_C

    start_event.record()
    # Step 1: Sort slots within each warp by age using CUDA kernel
    # Note: warp_empty_counts is reused, no need to zero it (kernel overwrites)
    # Compute actual number of warps based on available_slots size (changes each iteration)
    num_available_slots = available_slots.shape[0]
    actual_num_warps = (num_available_slots + 31) // 32

    # Use subset of pre-allocated warp_start_indices for actual warps
    actual_warp_start_indices = warp_start_indices[:actual_num_warps]
    actual_warp_empty_counts = warp_empty_counts[:actual_num_warps]

    vortex_C.warp_sort_slots_by_age(
        available_slots,
        slot_ages,
        actual_warp_start_indices,
        actual_warp_empty_counts
    )

    end_event.record()
    end_event.synchronize()
    final = start_event.elapsed_time(end_event)
    print(f"[DEBUG] Pass A (LRU sort) took {final:.4f} ms")

    start_event.record()
    # Step 2: Calculate cache misses BEFORE allocation
    unique_pages = torch.unique(src_page_ids)
    num_unique_pages = unique_pages.numel()
    slots_before = cpu_to_gpu_slot_map[unique_pages]
    num_cache_misses = (slots_before == -1).sum().item()

    # Debug: check how many slots are allocated overall
    total_allocated_slots = (cpu_to_gpu_slot_map >= 0).sum().item()
    print(f"[DEBUG] Before allocation: total_allocated_slots={total_allocated_slots}, cache_misses={num_cache_misses}/{num_unique_pages}")

    # Step 3: Compute dynamic bucket parameters based on actual available slots
    # bucket_sizes and bucket_offsets change each iteration since available_slots changes
    actual_bucket_sizes = torch.full((actual_num_warps,), 32, device=available_slots.device, dtype=torch.int32)
    if num_available_slots % 32 != 0:
        actual_bucket_sizes[-1] = num_available_slots % 32

    actual_bucket_offsets = torch.arange(0, actual_num_warps * 32, 32,
                                         dtype=torch.int32, device=available_slots.device)

    # Reset alloc counters for actual warps
    actual_alloc_counters = alloc_counters[:actual_num_warps]
    actual_alloc_counters.zero_()

    # Step 4: Allocate using sharded kernel with LRU
    mark_and_allocate_unique_kernel_lru_sharded[(num_pages,)](
        src_page_ids,
        cpu_to_gpu_slot_map,
        gpu_to_cpu_page_map,
        available_slots,
        slot_ages,
        actual_alloc_counters,
        actual_bucket_offsets,
        actual_bucket_sizes,
        owners_bitmap,
        overflow_flag,
        MAX_PAGE_ID=max_page_id,
        N=num_pages,
        NB=actual_num_warps,
    )
    end_event.record()
    end_event.synchronize()
    final = start_event.elapsed_time(end_event)

    num_allocated = actual_alloc_counters.sum().item()
    print(f"[DEBUG] Pass B (allocate) took {final:.4f} ms")
    print(f"[DEBUG] Pages: total={num_pages}, unique={num_unique_pages}, cache_misses={num_cache_misses}, allocated={num_allocated}")

    if num_allocated != num_cache_misses:
        print(f"[WARNING] Allocation mismatch! Expected {num_cache_misses} but got {num_allocated}")

    # Check for overflow
    if overflow_flag[0] == 1:
        print("Warning: GPU slot overflow detected!")

    start_event.record()
    # Step 4: Materialize final slots for all positions
    materialize_slots_kernel[(num_pages,)](
        src_page_ids,
        cpu_to_gpu_slot_map,
        dst_staging_slots,
        N=num_pages,
        MAX_PAGE_ID=max_page_id,
    )
    end_event.record()
    end_event.synchronize()
    final = start_event.elapsed_time(end_event)
    print(f"[DEBUG] Pass C took {final:.4f} ms")

    if torch.any(dst_staging_slots < 0):
        print("Warning: Some pages failed to allocate staging slots!")

    start_event.record()
    # Step 5: Copy only unique pages (owners)
    copy_with_assigned_slots_kernel[(num_pages * page_size,)](
        cpu_k_buffer, cpu_v_buffer,
        gpu_k_staging, gpu_v_staging,
        src_page_ids,
        dst_staging_slots,
        owners_bitmap,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        NUM_PAGES=num_pages,
    )
    end_event.record()
    end_event.synchronize()
    final = start_event.elapsed_time(end_event)
    print(f"[DEBUG] Pass D took {final:.4f} ms")

    # Step 6: Age decay for next iteration (matching OneFlow LRU)
    # Decrement all non-zero ages to simulate aging
    # Ages: 0=empty, 1=oldest, 32=newest
    non_zero_mask = slot_ages > 0
    slot_ages[non_zero_mask] = torch.clamp(slot_ages[non_zero_mask] - 1, min=1, max=32)

    return dst_staging_slots[:num_pages]
