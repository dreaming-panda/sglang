"""
Test the new simplified CPU->GPU sparse copy kernel.

Tests:
1. Correctness vs naive torch copy
2. Performance comparison
"""

import torch
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from sglang.srt.mem_cache.cpu_gpu_copy_kernels import copy_sparse_kv_cpu_to_gpu


def create_test_cpu_buffer(num_pages, page_size, head_num, head_dim, dtype=torch.bfloat16):
    """Create CPU buffer with VTX paged layout and identifiable test data.

    VTX paged layout: (token_id // page_size) * (page_size * num_heads) + head_id * page_size + (token_id % page_size)

    This means: for each global page, all heads' data are stored together in blocks of page_size.
    """
    num_tokens = num_pages * page_size
    buffer = torch.zeros(
        (num_tokens * head_num, 1, head_dim),
        dtype=dtype,
        device='cpu',
        pin_memory=True,
    )

    # Fill with pattern using VTX paged layout
    for token_idx in range(num_tokens):
        for head_idx in range(head_num):
            # VTX paged layout formula from set_kv_buffer_kernel
            page_id = token_idx // page_size
            token_in_page = token_idx % page_size
            linear_idx = page_id * page_size * head_num + head_idx * page_size + token_in_page

            buffer[linear_idx, 0, :] = (
                token_idx * 1000.0 + head_idx * 10.0 + torch.arange(head_dim, dtype=torch.float32)
            ).to(dtype)
    return buffer


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


def test_correctness():
    """Test 1: Correctness comparison."""
    print("\n" + "="*80)
    print("TEST 1: Correctness (Triton Kernel vs Naive Copy)")
    print("="*80)

    # Configuration
    page_size = 16
    head_num = 8
    head_dim = 128
    total_pages = 100
    num_sparse_pages = 20  # Number of per-head pages to copy
    total_tokens = total_pages * page_size

    print(f"\nConfiguration:")
    print(f"  Page size: {page_size}")
    print(f"  Head num: {head_num}")
    print(f"  Head dim: {head_dim}")
    print(f"  Total pages: {total_pages}")
    print(f"  Sparse per-head pages to copy: {num_sparse_pages}")

    # Create CPU buffers with VTX paged layout
    cpu_k = create_test_cpu_buffer(total_pages, page_size, head_num, head_dim)
    cpu_v = create_test_cpu_buffer(total_pages, page_size, head_num, head_dim)
    cpu_v.mul_(2)

    # Random sparse per-head page indices
    # Total per-head pages = total_pages * head_num (each global page has head_num per-head pages)
    total_per_head_pages = total_pages * head_num
    sparse_indices = torch.randperm(total_per_head_pages)[:num_sparse_pages].to(torch.int32).cuda()

    print(f"\nSparse indices (first 10): {sparse_indices[:10].cpu().tolist()}")

    # Create GPU staging buffers for baseline
    gpu_k_baseline = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v_baseline = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')

    # Create GPU staging buffers for kernel
    gpu_k_kernel = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v_kernel = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')

    # Run naive baseline
    print("\nRunning naive baseline...")
    naive_copy(cpu_k, cpu_v, gpu_k_baseline, gpu_v_baseline, sparse_indices, page_size)
    torch.cuda.synchronize()
    print("Baseline completed.")

    # Run Triton kernel
    print("\nRunning Triton kernel...")
    copy_sparse_kv_cpu_to_gpu(
        cpu_k_buffer=cpu_k,
        cpu_v_buffer=cpu_v,
        gpu_k_staging=gpu_k_kernel,
        gpu_v_staging=gpu_v_kernel,
        sparse_indices=sparse_indices,
        page_size=page_size,
        head_num=head_num,
    )
    torch.cuda.synchronize()
    print("Kernel completed.")

    # Compare results
    print("\nComparing results...")
    num_entries = num_sparse_pages * page_size
    k_match = torch.allclose(gpu_k_kernel[:num_entries], gpu_k_baseline[:num_entries], rtol=1e-2, atol=1e-2)
    v_match = torch.allclose(gpu_v_kernel[:num_entries], gpu_v_baseline[:num_entries], rtol=1e-2, atol=1e-2)

    if k_match and v_match:
        print("✓ Results match! Test PASSED.")
        return True
    else:
        print("✗ Mismatch detected!")
        if not k_match:
            k_diff = (gpu_k_kernel[:num_entries] - gpu_k_baseline[:num_entries]).abs()
            print(f"  K max diff: {k_diff.max().item()}, mean diff: {k_diff.mean().item()}")
            mismatch = (k_diff > 1e-2).nonzero(as_tuple=False)
            if len(mismatch) > 0:
                idx = mismatch[0]
                print(f"  First K mismatch at index {idx.cpu().numpy()}")
                print(f"    Baseline: {gpu_k_baseline[idx[0], idx[1], idx[2]].item()}")
                print(f"    Kernel:   {gpu_k_kernel[idx[0], idx[1], idx[2]].item()}")
        if not v_match:
            v_diff = (gpu_v_kernel[:num_entries] - gpu_v_baseline[:num_entries]).abs()
            print(f"  V max diff: {v_diff.max().item()}, mean diff: {v_diff.mean().item()}")
        return False


def test_performance():
    """Test 2: Performance benchmark."""
    print("\n" + "="*80)
    print("TEST 2: Performance Benchmark")
    print("="*80)

    # Larger configuration for performance testing
    page_size = 16
    head_num = 32
    head_dim = 128
    total_pages = 1000
    num_sparse_pages = 128  # Number of per-head pages to copy
    total_tokens = total_pages * page_size
    num_warmup = 10
    num_iters = 100

    print(f"\nConfiguration:")
    print(f"  Page size: {page_size}")
    print(f"  Head num: {head_num}")
    print(f"  Head dim: {head_dim}")
    print(f"  Total pages: {total_pages}")
    print(f"  Sparse per-head pages: {num_sparse_pages}")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Benchmark iterations: {num_iters}")

    # Create buffers with VTX paged layout
    cpu_k = create_test_cpu_buffer(total_pages, page_size, head_num, head_dim)
    cpu_v = create_test_cpu_buffer(total_pages, page_size, head_num, head_dim)

    gpu_k_staging = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v_staging = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')

    total_per_head_pages = total_pages * head_num
    sparse_indices = torch.randperm(total_per_head_pages)[:num_sparse_pages].to(torch.int32).cuda()

    bytes_per_copy = num_sparse_pages * page_size * head_dim * 2 * 2  # K+V, bfloat16=2 bytes

    # Benchmark 1: Naive baseline
    print("\n--- Benchmarking Naive Baseline ---")
    gpu_k_baseline = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v_baseline = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')

    # Warmup
    for _ in range(num_warmup):
        naive_copy(cpu_k, cpu_v, gpu_k_baseline, gpu_v_baseline, sparse_indices, page_size)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        naive_copy(cpu_k, cpu_v, gpu_k_baseline, gpu_v_baseline, sparse_indices, page_size)
    end_event.record()
    torch.cuda.synchronize()

    naive_elapsed_ms = start_event.elapsed_time(end_event)
    naive_avg_time_ms = naive_elapsed_ms / num_iters
    naive_bandwidth_gbps = (bytes_per_copy / 1e9) / (naive_avg_time_ms / 1000)

    print(f"  Average time: {naive_avg_time_ms:.3f} ms")
    print(f"  Bandwidth: {naive_bandwidth_gbps:.2f} GB/s")

    # Benchmark 2: Triton kernel
    print("\n--- Benchmarking Triton Kernel ---")
    # Warmup
    for _ in range(num_warmup):
        copy_sparse_kv_cpu_to_gpu(cpu_k, cpu_v, gpu_k_staging, gpu_v_staging, sparse_indices, page_size, head_num)
    torch.cuda.synchronize()

    # Benchmark
    start_event.record()
    for _ in range(num_iters):
        copy_sparse_kv_cpu_to_gpu(cpu_k, cpu_v, gpu_k_staging, gpu_v_staging, sparse_indices, page_size, head_num)
    end_event.record()
    torch.cuda.synchronize()

    triton_elapsed_ms = start_event.elapsed_time(end_event)
    triton_avg_time_ms = triton_elapsed_ms / num_iters
    triton_bandwidth_gbps = (bytes_per_copy / 1e9) / (triton_avg_time_ms / 1000)

    print(f"  Average time: {triton_avg_time_ms:.3f} ms")
    print(f"  Bandwidth: {triton_bandwidth_gbps:.2f} GB/s")

    # Comparison
    speedup = naive_avg_time_ms / triton_avg_time_ms
    print(f"\n--- Comparison ---")
    print(f"  Bytes transferred: {bytes_per_copy / 1e6:.2f} MB")
    print(f"  Naive baseline: {naive_avg_time_ms:.3f} ms ({naive_bandwidth_gbps:.2f} GB/s)")
    print(f"  Triton kernel:  {triton_avg_time_ms:.3f} ms ({triton_bandwidth_gbps:.2f} GB/s)")
    print(f"  Speedup: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("NEW CPU->GPU SPARSE COPY KERNEL TEST SUITE")
    print("="*80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Tests require GPU.")
        return 1

    results = []

    # Test 1: Correctness
    try:
        results.append(("Correctness", test_correctness()))
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Correctness", False))

    # Test 2: Performance
    try:
        results.append(("Performance", test_performance()))
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Performance", False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s} {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
