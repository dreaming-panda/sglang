# benchmark_sparse_kv_copy.py
import math
import time
import torch
import triton
import triton.language as tl

# -----------------------------
# Original (1D, per-token) kernel
# -----------------------------
@triton.jit
def cpu_to_gpu_sparse_copy_kernel_baseline(
    cpu_k_ptr, cpu_v_ptr,
    gpu_k_ptr, gpu_v_ptr,
    sparse_indices_ptr,     # [NUM_SPARSE_PAGES]
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPARSE_PAGES: tl.constexpr,
):
    token_idx = tl.program_id(0)  # one program per token within the sparse selection
    page_idx = token_idx // PAGE_SIZE
    tok_off = token_idx % PAGE_SIZE
    if page_idx >= NUM_SPARSE_PAGES:
        return

    src_per_head_page = tl.load(sparse_indices_ptr + page_idx)
    src_linear_idx = src_per_head_page * PAGE_SIZE + tok_off
    dst_linear_idx = page_idx * PAGE_SIZE + tok_off

    dim = tl.arange(0, HEAD_DIM)
    k_val = tl.load(cpu_k_ptr + src_linear_idx * HEAD_DIM + dim)
    v_val = tl.load(cpu_v_ptr + src_linear_idx * HEAD_DIM + dim)
    tl.store(gpu_k_ptr + dst_linear_idx * HEAD_DIM + dim, k_val)
    tl.store(gpu_v_ptr + dst_linear_idx * HEAD_DIM + dim, v_val)

# -----------------------------
# Tiled (2D) kernel
# -----------------------------
@triton.jit
def cpu_to_gpu_sparse_copy_kernel_tiled(
    cpu_k_ptr, cpu_v_ptr,
    gpu_k_ptr, gpu_v_ptr,
    sparse_indices_ptr,            # [NUM_SPARSE_PAGES]
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPARSE_PAGES: tl.constexpr,
    BLOCK_T: tl.constexpr,         # tokens per tile (rows)
    BLOCK_D: tl.constexpr          # head-dim per tile (cols)
):
    pid_page = tl.program_id(0)     # which sparse output page this program handles
    pid_tok_tile = tl.program_id(1) # which tile of tokens within that page
    if pid_page >= NUM_SPARSE_PAGES:
        return

    src_per_head_page = tl.load(sparse_indices_ptr + pid_page)
    tok_offsets = pid_tok_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    tok_mask = tok_offsets < PAGE_SIZE

    src_lin_base = src_per_head_page * PAGE_SIZE + tok_offsets
    dst_lin_base = pid_page         * PAGE_SIZE + tok_offsets

    for d0 in range(0, HEAD_DIM, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < HEAD_DIM

        src_idx = (src_lin_base[:, None] * HEAD_DIM) + d_offsets[None, :]
        dst_idx = (dst_lin_base[:, None] * HEAD_DIM) + d_offsets[None, :]
        mask = tok_mask[:, None] & d_mask[None, :]

        k_tile = tl.load(cpu_k_ptr + src_idx, mask=mask, other=0)
        v_tile = tl.load(cpu_v_ptr + src_idx, mask=mask, other=0)
        tl.store(gpu_k_ptr + dst_idx, k_tile, mask=mask)
        tl.store(gpu_v_ptr + dst_idx, v_tile, mask=mask)

# -----------------------------
# Helpers
# -----------------------------
def build_reference(cpu_k, cpu_v, sparse_indices, page_size):
    """
    Build reference staging buffers on CPU by gathering pages, then return CPU tensors.
    Shapes:
      cpu_k: [total_pages*page_size, 1, head_dim]
      returns (k_ref_cpu, v_ref_cpu): [num_sparse_pages*page_size, 1, head_dim]
    """
    head_dim = cpu_k.shape[2]
    n_sp = sparse_indices.numel()

    # Flatten away the middle dim for easy indexing
    cpu_k_flat = cpu_k.view(-1, head_dim)
    cpu_v_flat = cpu_v.view(-1, head_dim)

    # Build output flat by concatenating page blocks in order
    out_k = torch.empty((n_sp * page_size, head_dim), dtype=cpu_k.dtype, pin_memory=True)
    out_v = torch.empty_like(out_k)

    # For each page: copy slice [page*page_size : (page+1)*page_size]
    # Using vectorized indexing for speed
    # Build a big index of all rows we need (size n_sp*page_size)
    base_rows = (sparse_indices.view(-1, 1) * page_size) + torch.arange(page_size).view(1, -1)
    row_idx = base_rows.reshape(-1)  # [n_sp*page_size]
    out_k.copy_(cpu_k_flat.index_select(0, row_idx))
    out_v.copy_(cpu_v_flat.index_select(0, row_idx))

    # Restore the [N,1,D] view
    return out_k.view(-1, 1, head_dim), out_v.view(-1, 1, head_dim)

@torch.inference_mode()
def run_kernel_baseline(cpu_k, cpu_v, gpu_k, gpu_v, sparse_indices, page_size, head_dim,
                        warmup=5, iters=20):
    num_sparse_pages = sparse_indices.numel()
    grid = (num_sparse_pages * page_size,)

    # Flatten middle dim
    cpu_k_flat = cpu_k.view(-1, head_dim)
    cpu_v_flat = cpu_v.view(-1, head_dim)
    gpu_k_flat = gpu_k.view(-1, head_dim)
    gpu_v_flat = gpu_v.view(-1, head_dim)

    # Warmup
    for _ in range(warmup):
        cpu_to_gpu_sparse_copy_kernel_baseline[grid](
            cpu_k_flat, cpu_v_flat, gpu_k_flat, gpu_v_flat, sparse_indices,
            PAGE_SIZE=page_size, HEAD_DIM=head_dim, NUM_SPARSE_PAGES=num_sparse_pages
        )
    torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cpu_to_gpu_sparse_copy_kernel_baseline[grid](
            cpu_k_flat, cpu_v_flat, gpu_k_flat, gpu_v_flat, sparse_indices,
            PAGE_SIZE=page_size, HEAD_DIM=head_dim, NUM_SPARSE_PAGES=num_sparse_pages
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)  # seconds
    return median(times)

@torch.inference_mode()
def run_kernel_tiled(cpu_k, cpu_v, gpu_k, gpu_v, sparse_indices, page_size, head_dim,
                     block_t=64, block_d=128, num_warps=4, num_stages=2,
                     warmup=5, iters=20):
    num_sparse_pages = sparse_indices.numel()
    tiles_per_page = (page_size + block_t - 1) // block_t
    grid = (num_sparse_pages, tiles_per_page)

    cpu_k_flat = cpu_k.view(-1, head_dim)
    cpu_v_flat = cpu_v.view(-1, head_dim)
    gpu_k_flat = gpu_k.view(-1, head_dim)
    gpu_v_flat = gpu_v.view(-1, head_dim)

    # Warmup
    for _ in range(warmup):
        cpu_to_gpu_sparse_copy_kernel_tiled[grid](
            cpu_k_flat, cpu_v_flat, gpu_k_flat, gpu_v_flat, sparse_indices,
            PAGE_SIZE=page_size, HEAD_DIM=head_dim, NUM_SPARSE_PAGES=num_sparse_pages,
            BLOCK_T=block_t, BLOCK_D=block_d,
            num_warps=num_warps, num_stages=num_stages
        )
    torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cpu_to_gpu_sparse_copy_kernel_tiled[grid](
            cpu_k_flat, cpu_v_flat, gpu_k_flat, gpu_v_flat, sparse_indices,
            PAGE_SIZE=page_size, HEAD_DIM=head_dim, NUM_SPARSE_PAGES=num_sparse_pages,
            BLOCK_T=block_t, BLOCK_D=block_d,
            num_warps=num_warps, num_stages=num_stages
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)  # seconds
    return median(times)

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n//2] if n % 2 == 1 else 0.5*(xs[n//2-1] + xs[n//2])

def pretty_gbps(bytes_copied, seconds):
    return (bytes_copied / seconds) / 1e9

def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required"

    # -----------------------------
    # FIXED PROBLEM SHAPE
    # -----------------------------
    page_size = 16
    head_dim = 128
    dtype = torch.bfloat16  # change to torch.float16 / float32 as needed

    # -----------------------------
    # TUNABLES for tiled kernel
    # -----------------------------
    block_t = 64     # token tile (64 covers 4 pages worth of tokens)
    block_d = 128    # dim tile (fits head_dim exactly here)
    num_warps = 4
    num_stages = 2

    # -----------------------------
    # Benchmark sweep
    # total pages available on CPU (per-head) and the sampled sparse pages
    # -----------------------------
    total_pages_per_head = 8192   # change to reflect your host buffer size
    test_sparse_pages = [4096, 8192, 16384, 32768, 65536, 131072]  # sweep sizes (<= total_pages_per_head)

    # Allocate CPU pinned buffers: [total_pages*page_size, 1, head_dim]
    total_entries = total_pages_per_head * page_size
    cpu_k = torch.randn((total_entries, 1, head_dim), dtype=dtype, pin_memory=True)
    cpu_v = torch.randn_like(cpu_k).pin_memory()
    
    print(cpu_v.is_pinned())

    print(f"CPU buffers: {cpu_k.shape}, dtype={dtype}")
    print(f"page_size={page_size}, head_dim={head_dim}")

    for n_sp in test_sparse_pages:
        if n_sp > total_pages_per_head:
            continue
        print("\n" + "-"*60)
        print(f"num_sparse_pages = {n_sp}")

        # Choose random unique pages for this run
        sparse_indices = torch.randperm(total_pages_per_head)[:n_sp].contiguous()
        # GPU staging buffers for results: [n_sp*page_size, 1, head_dim]
        out_entries = n_sp * page_size
        gpu_k_baseline = torch.empty((out_entries, 1, head_dim), dtype=dtype, device="cuda")
        gpu_v_baseline = torch.empty_like(gpu_k_baseline)
        gpu_k_tiled    = torch.empty_like(gpu_k_baseline)
        gpu_v_tiled    = torch.empty_like(gpu_k_baseline)

        # Correctness vs reference
        ref_k_cpu, ref_v_cpu = build_reference(cpu_k, cpu_v, sparse_indices, page_size)
        ref_k = ref_k_cpu.to("cuda", non_blocking=True)
        ref_v = ref_v_cpu.to("cuda", non_blocking=True)

        # Run baseline
        t_base = run_kernel_baseline(cpu_k, cpu_v, gpu_k_baseline, gpu_v_baseline,
                                     sparse_indices.to(device="cuda"),  # Triton wants device tensors for pointers
                                     page_size, head_dim)
        err_k_base = (gpu_k_baseline - ref_k).abs().max().item()
        err_v_base = (gpu_v_baseline - ref_v).abs().max().item()

        # Run tiled
        t_tiled = run_kernel_tiled(cpu_k, cpu_v, gpu_k_tiled, gpu_v_tiled,
                                   sparse_indices.to(device="cuda"),
                                   page_size, head_dim,
                                   block_t=block_t, block_d=block_d,
                                   num_warps=num_warps, num_stages=num_stages)
        err_k_tiled = (gpu_k_tiled - ref_k).abs().max().item()
        err_v_tiled = (gpu_v_tiled - ref_v).abs().max().item()

        # Effective bytes moved: K + V
        bytes_per_entry = head_dim * torch.finfo(dtype).bits // 8
        bytes_total = out_entries * bytes_per_entry * 2  # K + V

        print(f"Correctness (max abs diff): baseline K={err_k_base:.3e}, V={err_v_base:.3e} | "
              f"tiled K={err_k_tiled:.3e}, V={err_v_tiled:.3e}")
        print(f"Latency (median over iters): baseline={t_base*1e3:.2f} ms, tiled={t_tiled*1e3:.2f} ms")
        print(f"Effective H2D BW (approx):  baseline={pretty_gbps(bytes_total, t_base):.2f} GB/s, "
              f"tiled={pretty_gbps(bytes_total, t_tiled):.2f} GB/s")

if __name__ == "__main__":
    main()
