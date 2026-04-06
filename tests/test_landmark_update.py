"""
Test script for CPU-based landmark update kernel.

Tests:
1. Correctness: Compare against baseline (load all K to GPU then use vortex kernel)
2. Performance: Benchmark CPU->GPU landmark vs baseline
3. Edge cases: Various page sizes, batch sizes, etc.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sglang.srt.mem_cache.cpu_gpu_copy_kernels import update_landmark_from_cpu

# Import vortex landmark kernel for comparison
try:
    from vortex import update_landmark_launcher
    HAS_VORTEX = True
except ImportError:
    HAS_VORTEX = False
    print("Warning: vortex module not found. Baseline comparison will be skipped.")


def create_test_k_buffer(num_tokens, num_kv_head, head_dim, page_size, dtype=torch.bfloat16, device='cpu'):
    """Create test K buffer with identifiable values."""
    k_buffer = torch.zeros(
        (num_tokens * num_kv_head, 1, head_dim),
        dtype=dtype,
        device=device,
        pin_memory=(device == 'cpu'),
    )

    # Fill with identifiable values
    # Each page should have unique pattern for verification
    for page_id in range(num_tokens // page_size):
        for head_id in range(num_kv_head):
            for token_in_page in range(page_size):
                token_id = page_id * page_size + token_in_page
                linear_idx = token_id * num_kv_head + head_id
                # Pattern: page_id * 1000 + head_id * 10 + token_in_page
                k_buffer[linear_idx, 0, :] = (
                    page_id * 1000.0 + head_id * 10.0 + token_in_page +
                    torch.arange(head_dim, dtype=torch.float32) * 0.01
                ).to(dtype)

    return k_buffer


def compute_landmark_baseline(cpu_k, loc, page_size, num_kv_head, head_dim):
    """
    Baseline: Load all K to GPU, then use vortex update_landmark_launcher.
    """
    if not HAS_VORTEX:
        raise RuntimeError("Vortex module not available for baseline")

    # Create GPU K buffer and copy from CPU
    gpu_k = torch.zeros_like(cpu_k, device='cuda')
    gpu_k.copy_(cpu_k)

    # Create landmark buffer
    num_pages = (cpu_k.shape[0] // num_kv_head + page_size - 1) // page_size
    gpu_landmark = torch.zeros(
        (num_pages * num_kv_head, 1, head_dim),
        dtype=cpu_k.dtype,
        device='cuda',
    )

    # Run vortex kernel
    update_landmark_launcher(
        gpu_k,
        gpu_landmark,
        loc,
        page_size,
        num_kv_head,
        head_dim
    )

    return gpu_landmark


def test_correctness():
    """Test 1: Correctness against baseline."""
    print("\n" + "="*80)
    print("TEST 1: Correctness (vs Vortex Baseline)")
    print("="*80)

    if not HAS_VORTEX:
        print("⚠ Skipping correctness test (vortex not available)")
        return False

    # Configuration
    page_size = 16
    num_kv_head = 8
    head_dim = 128
    num_pages = 20
    num_tokens = num_pages * page_size

    print(f"Configuration:")
    print(f"  Page size: {page_size}")
    print(f"  Num KV heads: {num_kv_head}")
    print(f"  Head dim: {head_dim}")
    print(f"  Num pages: {num_pages}")
    print(f"  Num tokens: {num_tokens}")

    # Create CPU K buffer
    cpu_k = create_test_k_buffer(num_tokens, num_kv_head, head_dim, page_size, device='cpu')

    # Create location indices (all tokens, to update all pages)
    loc = torch.arange(num_tokens, dtype=torch.int64, device='cuda')

    # Create GPU landmark buffer
    gpu_landmark_test = torch.zeros(
        (num_pages * num_kv_head, 1, head_dim),
        dtype=cpu_k.dtype,
        device='cuda',
    )

    # Method 1: Our CPU->GPU landmark kernel
    print("\nRunning CPU->GPU landmark kernel...")
    update_landmark_from_cpu(
        cpu_k_buffer=cpu_k,
        gpu_landmark=gpu_landmark_test,
        loc=loc,
        page_size=page_size,
        num_kv_head=num_kv_head,
        head_dim=head_dim,
    )
    torch.cuda.synchronize()
    print("Completed.")

    # Method 2: Baseline (load all to GPU, then vortex kernel)
    print("\nRunning baseline (load all K to GPU + vortex kernel)...")
    gpu_landmark_baseline = compute_landmark_baseline(cpu_k, loc, page_size, num_kv_head, head_dim)
    torch.cuda.synchronize()
    print("Completed.")

    # Compare results
    print("\nComparing results...")
    max_diff = (gpu_landmark_test - gpu_landmark_baseline).abs().max().item()
    mean_diff = (gpu_landmark_test - gpu_landmark_baseline).abs().mean().item()

    match = torch.allclose(gpu_landmark_test, gpu_landmark_baseline, rtol=1e-2, atol=1e-2)

    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff}")

    if match:
        print("✓ Results match! Test PASSED.")
        return True
    else:
        print("✗ Results don't match! Test FAILED.")
        # Find first mismatch
        mismatch = ((gpu_landmark_test - gpu_landmark_baseline).abs() > 1e-2).nonzero(as_tuple=False)
        if len(mismatch) > 0:
            idx = mismatch[0]
            print(f"\n  First mismatch at index {idx.cpu().numpy()}")
            print(f"    Baseline: {gpu_landmark_baseline[idx[0], idx[1], idx[2]].item()}")
            print(f"    Test:     {gpu_landmark_test[idx[0], idx[1], idx[2]].item()}")
        return False


def test_page_boundary():
    """Test 2: Only pages at boundaries should be updated."""
    print("\n" + "="*80)
    print("TEST 2: Page Boundary Updates")
    print("="*80)

    page_size = 16
    num_kv_head = 4
    head_dim = 64
    num_pages = 10
    num_tokens = num_pages * page_size

    print(f"Testing that only complete pages are updated...")

    # Create CPU K buffer
    cpu_k = create_test_k_buffer(num_tokens, num_kv_head, head_dim, page_size, device='cpu')

    # Create GPU landmark buffer (initialized to -1 to detect updates)
    gpu_landmark = torch.full(
        (num_pages * num_kv_head, 1, head_dim),
        -1.0,
        dtype=cpu_k.dtype,
        device='cuda',
    )

    # Test with partial page: only write first 8 tokens of first page
    loc = torch.arange(8, dtype=torch.int64, device='cuda')

    update_landmark_from_cpu(cpu_k, gpu_landmark, loc, page_size, num_kv_head, head_dim)
    torch.cuda.synchronize()

    # Check: no landmarks should be updated (all should still be -1)
    num_updated = (gpu_landmark != -1.0).sum().item()
    if num_updated == 0:
        print("✓ Correctly skipped incomplete page")
    else:
        print(f"✗ Incorrectly updated {num_updated} landmarks for incomplete page")
        return False

    # Now complete the page
    loc = torch.arange(16, dtype=torch.int64, device='cuda')
    update_landmark_from_cpu(cpu_k, gpu_landmark, loc, page_size, num_kv_head, head_dim)
    torch.cuda.synchronize()

    # Check: first page's landmarks should be updated for all heads
    expected_updates = num_kv_head  # One page, all heads
    num_updated = (gpu_landmark != -1.0).sum().item() // head_dim

    if num_updated == expected_updates:
        print(f"✓ Correctly updated {num_updated} landmarks for complete page")
        return True
    else:
        print(f"✗ Expected {expected_updates} updates, got {num_updated}")
        return False


def test_performance():
    """Test 3: Performance comparison."""
    print("\n" + "="*80)
    print("TEST 3: Performance Comparison")
    print("="*80)

    if not HAS_VORTEX:
        print("⚠ Skipping performance test (vortex not available)")
        return True

    # Configuration
    page_size = 16
    num_kv_head = 32
    head_dim = 128
    num_pages = 1000
    num_tokens = num_pages * page_size
    num_warmup = 10
    num_iters = 100

    print(f"Configuration:")
    print(f"  Page size: {page_size}")
    print(f"  Num KV heads: {num_kv_head}")
    print(f"  Head dim: {head_dim}")
    print(f"  Num pages: {num_pages}")
    print(f"  Warmup: {num_warmup}")
    print(f"  Iterations: {num_iters}")

    # Create buffers
    cpu_k = create_test_k_buffer(num_tokens, num_kv_head, head_dim, page_size, device='cpu')
    loc = torch.arange(num_tokens, dtype=torch.int64, device='cuda')

    # Benchmark 1: Baseline (load all K + vortex kernel)
    print("\n--- Baseline: Load all K to GPU + Vortex kernel ---")

    # Warmup
    for _ in range(num_warmup):
        _ = compute_landmark_baseline(cpu_k, loc, page_size, num_kv_head, head_dim)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _ = compute_landmark_baseline(cpu_k, loc, page_size, num_kv_head, head_dim)
    end.record()
    torch.cuda.synchronize()

    baseline_time = start.elapsed_time(end) / num_iters
    print(f"  Average time: {baseline_time:.3f} ms")

    # Benchmark 2: Our CPU->GPU landmark kernel
    print("\n--- Our kernel: Direct CPU->GPU landmark update ---")
    gpu_landmark = torch.zeros(
        (num_pages * num_kv_head, 1, head_dim),
        dtype=cpu_k.dtype,
        device='cuda',
    )

    # Warmup
    for _ in range(num_warmup):
        update_landmark_from_cpu(cpu_k, gpu_landmark, loc, page_size, num_kv_head, head_dim)
    torch.cuda.synchronize()

    # Benchmark
    start.record()
    for _ in range(num_iters):
        update_landmark_from_cpu(cpu_k, gpu_landmark, loc, page_size, num_kv_head, head_dim)
    end.record()
    torch.cuda.synchronize()

    kernel_time = start.elapsed_time(end) / num_iters
    print(f"  Average time: {kernel_time:.3f} ms")

    # Compare
    speedup = baseline_time / kernel_time
    print(f"\n--- Comparison ---")
    print(f"  Baseline: {baseline_time:.3f} ms")
    print(f"  Our kernel: {kernel_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")

    # Calculate data transfer
    k_size_mb = (num_tokens * num_kv_head * head_dim * 2) / 1e6  # bfloat16 = 2 bytes
    print(f"  K buffer size: {k_size_mb:.2f} MB")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CPU-BASED LANDMARK UPDATE KERNEL TEST SUITE")
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

    # Test 2: Page boundary
    try:
        results.append(("Page Boundary", test_page_boundary()))
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Page Boundary", False))

    # Test 3: Performance
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
