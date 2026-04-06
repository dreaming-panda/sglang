"""
Test script for CPU-GPU KV copy kernels.

Tests correctness of:
1. CPU->GPU sparse copy kernel
2. GPU->CPU store kernel
3. Round-trip consistency
4. Edge cases (single page, full batch, etc.)
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sglang.srt.mem_cache.cpu_gpu_copy_kernels import (
    copy_sparse_kv_cpu_to_gpu,
    store_kv_gpu_to_cpu,
)


def create_test_cpu_buffer(total_tokens, head_num, head_dim, dtype=torch.bfloat16):
    """Create a CPU buffer with pinned memory and fill with test data."""
    buffer = torch.zeros(
        (total_tokens * head_num, 1, head_dim),
        dtype=dtype,
        device='cpu',
        pin_memory=True,
    )
    # Fill with unique identifiable values: token_idx * 10000 + head_idx * 100 + dim_idx
    for token_idx in range(total_tokens):
        for head_idx in range(head_num):
            linear_idx = token_idx * head_num + head_idx
            # Use in-place operation to maintain pinned memory
            buffer[linear_idx, 0, :].copy_(
                (token_idx * 10000.0 + head_idx * 100.0 + torch.arange(head_dim, dtype=torch.float32)).to(dtype)
            )
    return buffer


def naive_cpu_to_gpu_copy(cpu_k, cpu_v, gpu_k_staging, gpu_v_staging, sparse_indices, page_size, head_num):
    """
    Optimized naive baseline: construct indices once, perform single batched copy.
    Now uses 3D layout: [tokens*heads, 1, head_dim] matching VTX pool.
    """
    num_sparse_pages = sparse_indices.shape[0]

    # Build source indices for all elements we need to copy
    # Each page has page_size * head_num rows in the CPU buffer
    # Total elements: num_sparse_pages * page_size * head_num
    total_rows = num_sparse_pages * page_size * head_num

    # Construct index tensor for advanced indexing
    src_indices = []
    for i in range(num_sparse_pages):
        page_idx = sparse_indices[i].item()
        start_idx = page_idx * page_size * head_num
        # Add all row indices for this page
        src_indices.extend(range(start_idx, start_idx + page_size * head_num))

    src_indices = torch.tensor(src_indices, dtype=torch.int64, device='cpu')

    # Copy from CPU to GPU
    # cpu_k[src_indices, :, :] gives us [total_rows, 1, head_dim]
    # This is already the correct 3D shape for the staging buffer
    src_k = cpu_k[src_indices, :, :]
    src_v = cpu_v[src_indices, :, :]

    # Copy to GPU staging buffer
    # gpu_k_staging[:total_rows] = src_k
    gpu_k_staging[:total_rows].copy_(src_k)
    gpu_v_staging[:total_rows].copy_(src_v)


def test_cpu_to_gpu_sparse_copy():
    """Test 1: CPU->GPU sparse copy correctness."""
    print("\n" + "="*80)
    print("TEST 1: CPU->GPU Sparse Copy (vs Naive Baseline)")
    print("="*80)

    # Configuration
    page_size = 16
    head_num = 8
    head_dim = 128
    total_pages = 100
    num_sparse_pages = 10
    total_tokens = total_pages * page_size

    print(f"Configuration:")
    print(f"  Page size: {page_size}")
    print(f"  Head num: {head_num}")
    print(f"  Head dim: {head_dim}")
    print(f"  Total pages: {total_pages}")
    print(f"  Sparse pages: {num_sparse_pages}")

    # Create CPU buffers with test data
    cpu_k = create_test_cpu_buffer(total_tokens, head_num, head_dim)
    cpu_v = create_test_cpu_buffer(total_tokens, head_num, head_dim)
    cpu_v.mul_(2)  # Different values for V, in-place to keep pinned memory

    # Create GPU staging buffers with 3D layout: [max_tokens * head_num, 1, head_dim]
    max_staging_tokens = total_tokens  # Allocate enough space for all tokens
    gpu_k_staging = torch.zeros(
        (max_staging_tokens * head_num, 1, head_dim),
        dtype=torch.bfloat16,
        device='cuda',
    )
    gpu_v_staging = torch.zeros(
        (max_staging_tokens * head_num, 1, head_dim),
        dtype=torch.bfloat16,
        device='cuda',
    )

    # Select random sparse pages
    sparse_page_indices = torch.randperm(total_pages)[:num_sparse_pages].to(torch.int32).cuda()
    print(f"\nSparse page indices: {sparse_page_indices.cpu().numpy()}")

    # Run naive baseline first to get ground truth
    print("\nRunning naive baseline (tensor.copy_)...")
    gpu_k_baseline = torch.zeros((max_staging_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v_baseline = torch.zeros((max_staging_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')

    naive_cpu_to_gpu_copy(cpu_k, cpu_v, gpu_k_baseline, gpu_v_baseline, sparse_page_indices, page_size, head_num)
    torch.cuda.synchronize()
    print("Baseline completed.")

    # Run Triton kernel
    print("\nRunning Triton kernel...")
    copy_sparse_kv_cpu_to_gpu(
        cpu_k_buffer=cpu_k,
        cpu_v_buffer=cpu_v,
        gpu_k_staging=gpu_k_staging,
        gpu_v_staging=gpu_v_staging,
        sparse_indices=sparse_page_indices,
        page_size=page_size,
        head_num=head_num,
    )
    torch.cuda.synchronize()
    print("Kernel completed.")

    # Compare kernel output with baseline
    # Only compare the populated portion (num_sparse_pages * page_size * head_num entries)
    print("\nComparing Triton kernel vs naive baseline...")
    num_populated = num_sparse_pages * page_size * head_num
    k_match = torch.allclose(gpu_k_staging[:num_populated], gpu_k_baseline[:num_populated], rtol=1e-2, atol=1e-2)
    v_match = torch.allclose(gpu_v_staging[:num_populated], gpu_v_baseline[:num_populated], rtol=1e-2, atol=1e-2)

    if k_match and v_match:
        print("✓ Triton kernel matches naive baseline! Test PASSED.")
        return True
    else:
        print(f"✗ Mismatch detected!")
        if not k_match:
            k_diff = (gpu_k_staging[:num_populated] - gpu_k_baseline[:num_populated]).abs()
            print(f"  K max diff: {k_diff.max().item()}, mean diff: {k_diff.mean().item()}")
            # Find first mismatch
            mismatch = (k_diff > 1e-2).nonzero(as_tuple=False)
            if len(mismatch) > 0:
                idx = mismatch[0]
                print(f"  First K mismatch at index {idx.cpu().numpy()}")
                print(f"    Baseline: {gpu_k_baseline[idx[0], idx[1], idx[2]].item()}")
                print(f"    Kernel:   {gpu_k_staging[idx[0], idx[1], idx[2]].item()}")
        if not v_match:
            v_diff = (gpu_v_staging - gpu_v_baseline).abs()
            print(f"  V max diff: {v_diff.max().item()}, mean diff: {v_diff.mean().item()}")
        return False


def test_gpu_to_cpu_store():
    """Test 2: GPU->CPU store correctness."""
    print("\n" + "="*80)
    print("TEST 2: GPU->CPU Store")
    print("="*80)

    # Configuration
    page_size = 16
    head_num = 8
    head_dim = 128
    num_tokens = 32
    total_tokens = 200

    print(f"Configuration:")
    print(f"  Page size: {page_size}")
    print(f"  Head num: {head_num}")
    print(f"  Head dim: {head_dim}")
    print(f"  Num tokens to store: {num_tokens}")
    print(f"  Total CPU capacity: {total_tokens}")

    # Create GPU source tensors with test data
    gpu_k = torch.zeros((num_tokens, head_num, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v = torch.zeros((num_tokens, head_num, head_dim), dtype=torch.bfloat16, device='cuda')

    # Fill with identifiable values
    for token_idx in range(num_tokens):
        for head_idx in range(head_num):
            gpu_k[token_idx, head_idx, :] = (
                token_idx * 10000.0 + head_idx * 100.0 + torch.arange(head_dim, dtype=torch.float32, device='cuda')
            ).to(torch.bfloat16)
            gpu_v[token_idx, head_idx, :] = gpu_k[token_idx, head_idx, :] * 2

    # Create CPU buffers (initially zeros)
    cpu_k = torch.zeros(
        (total_tokens * head_num, 1, head_dim),
        dtype=torch.bfloat16,
        device='cpu',
        pin_memory=True,
    )
    cpu_v = torch.zeros(
        (total_tokens * head_num, 1, head_dim),
        dtype=torch.bfloat16,
        device='cpu',
        pin_memory=True,
    )

    # Random destination locations
    loc = torch.randperm(total_tokens)[:num_tokens].to(torch.int64).cuda()
    print(f"\nDestination locations (first 10): {loc[:10].cpu().numpy()}")

    # Run kernel
    print("\nRunning GPU->CPU store kernel...")
    store_kv_gpu_to_cpu(
        gpu_k=gpu_k,
        gpu_v=gpu_v,
        cpu_k_buffer=cpu_k,
        cpu_v_buffer=cpu_v,
        loc=loc,
        page_size=page_size,
    )
    torch.cuda.synchronize()
    print("Kernel completed.")

    # Verify correctness
    print("\nVerifying correctness...")
    errors = 0
    for src_token_idx in range(num_tokens):
        dst_token_idx = loc[src_token_idx].item()

        for head_idx in range(head_num):
            dst_linear_idx = dst_token_idx * head_num + head_idx

            # Get expected values from GPU source
            expected_k = gpu_k[src_token_idx, head_idx, :].cpu()
            expected_v = gpu_v[src_token_idx, head_idx, :].cpu()

            # Get actual values from CPU destination
            actual_k = cpu_k[dst_linear_idx, 0, :]
            actual_v = cpu_v[dst_linear_idx, 0, :]

            # Compare
            if not torch.allclose(actual_k, expected_k, rtol=1e-2, atol=1e-2):
                errors += 1
                if errors <= 5:
                    print(f"  ERROR K: token {src_token_idx} -> loc {dst_token_idx}, head {head_idx}")
                    print(f"    Expected (first 5): {expected_k[:5].float()}")
                    print(f"    Actual (first 5):   {actual_k[:5].float()}")

            if not torch.allclose(actual_v, expected_v, rtol=1e-2, atol=1e-2):
                errors += 1
                if errors <= 5:
                    print(f"  ERROR V: token {src_token_idx} -> loc {dst_token_idx}, head {head_idx}")

    if errors == 0:
        print("✓ All values match! Test PASSED.")
        return True
    else:
        print(f"✗ Found {errors} errors. Test FAILED.")
        return False


def test_roundtrip_consistency():
    """Test 3: Round-trip GPU->CPU->GPU consistency."""
    print("\n" + "="*80)
    print("TEST 3: Round-trip Consistency (GPU->CPU->GPU)")
    print("="*80)

    # Configuration
    page_size = 16
    head_num = 8
    head_dim = 128
    num_pages = 20
    num_sparse_pages = 5
    total_tokens = num_pages * page_size

    print(f"Configuration:")
    print(f"  Page size: {page_size}")
    print(f"  Head num: {head_num}")
    print(f"  Head dim: {head_dim}")
    print(f"  Total pages: {num_pages}")
    print(f"  Sparse pages: {num_sparse_pages}")

    # Step 1: Create original GPU data
    original_gpu_k = torch.randn((total_tokens, head_num, head_dim), dtype=torch.bfloat16, device='cuda')
    original_gpu_v = torch.randn((total_tokens, head_num, head_dim), dtype=torch.bfloat16, device='cuda')

    # Step 2: Store to CPU
    cpu_k = torch.zeros(
        (total_tokens * head_num, 1, head_dim),
        dtype=torch.bfloat16,
        device='cpu',
        pin_memory=True,
    )
    cpu_v = torch.zeros(
        (total_tokens * head_num, 1, head_dim),
        dtype=torch.bfloat16,
        device='cpu',
        pin_memory=True,
    )

    loc = torch.arange(total_tokens, dtype=torch.int64, device='cuda')

    print("\nStep 1: Storing GPU->CPU...")
    store_kv_gpu_to_cpu(
        gpu_k=original_gpu_k,
        gpu_v=original_gpu_v,
        cpu_k_buffer=cpu_k,
        cpu_v_buffer=cpu_v,
        loc=loc,
        page_size=page_size,
    )
    torch.cuda.synchronize()

    # Step 3: Copy sparse pages back to GPU
    sparse_page_indices = torch.arange(num_sparse_pages, dtype=torch.int32, device='cuda')

    # 3D staging buffers
    gpu_k_staging = torch.zeros(
        (total_tokens * head_num, 1, head_dim),
        dtype=torch.bfloat16,
        device='cuda',
    )
    gpu_v_staging = torch.zeros(
        (total_tokens * head_num, 1, head_dim),
        dtype=torch.bfloat16,
        device='cuda',
    )

    print("Step 2: Copying sparse pages CPU->GPU...")
    copy_sparse_kv_cpu_to_gpu(
        cpu_k_buffer=cpu_k,
        cpu_v_buffer=cpu_v,
        gpu_k_staging=gpu_k_staging,
        gpu_v_staging=gpu_v_staging,
        sparse_indices=sparse_page_indices,
        page_size=page_size,
        head_num=head_num,
    )
    torch.cuda.synchronize()

    # Step 4: Verify round-trip consistency
    print("\nVerifying round-trip consistency...")
    errors = 0
    for page_idx in range(num_sparse_pages):
        for token_offset in range(page_size):
            src_token_idx = page_idx * page_size + token_offset

            for head_idx in range(head_num):
                # Original GPU values
                expected_k = original_gpu_k[src_token_idx, head_idx, :]
                expected_v = original_gpu_v[src_token_idx, head_idx, :]

                # Round-trip values from 3D staging buffer
                # Linear index: token_idx * head_num + head_idx
                linear_idx = src_token_idx * head_num + head_idx
                actual_k = gpu_k_staging[linear_idx, 0, :]
                actual_v = gpu_v_staging[linear_idx, 0, :]

                if not torch.allclose(actual_k, expected_k, rtol=1e-2, atol=1e-2):
                    errors += 1
                    if errors <= 5:
                        print(f"  ERROR K: page {page_idx}, token {token_offset}, head {head_idx}")

                if not torch.allclose(actual_v, expected_v, rtol=1e-2, atol=1e-2):
                    errors += 1
                    if errors <= 5:
                        print(f"  ERROR V: page {page_idx}, token {token_offset}, head {head_idx}")

    if errors == 0:
        print("✓ Round-trip values match! Test PASSED.")
        return True
    else:
        print(f"✗ Found {errors} errors. Test FAILED.")
        return False


def test_edge_cases():
    """Test 4: Edge cases."""
    print("\n" + "="*80)
    print("TEST 4: Edge Cases")
    print("="*80)

    page_size = 16
    head_num = 8
    head_dim = 128
    total_pages = 10
    total_tokens = total_pages * page_size

    # Test 4a: Single page copy
    print("\nTest 4a: Single page copy")
    cpu_k = create_test_cpu_buffer(total_tokens, head_num, head_dim)
    cpu_v = create_test_cpu_buffer(total_tokens, head_num, head_dim)
    cpu_v.mul_(2)

    # 3D staging buffer
    gpu_k_staging = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v_staging = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')

    sparse_indices = torch.tensor([5], dtype=torch.int32, device='cuda')

    copy_sparse_kv_cpu_to_gpu(cpu_k, cpu_v, gpu_k_staging, gpu_v_staging, sparse_indices, page_size, head_num)
    torch.cuda.synchronize()

    # Verify
    errors = 0
    for token_offset in range(page_size):
        src_token_idx = 5 * page_size + token_offset
        for head_idx in range(head_num):
            linear_idx = src_token_idx * head_num + head_idx
            expected_k = cpu_k[linear_idx, 0, :].cuda()
            # Staging buffer uses contiguous layout starting from index 0
            dst_linear_idx = token_offset * head_num + head_idx
            actual_k = gpu_k_staging[dst_linear_idx, 0, :]
            if not torch.allclose(actual_k, expected_k, rtol=1e-2, atol=1e-2):
                errors += 1

    if errors == 0:
        print("  ✓ Single page test PASSED")
    else:
        print(f"  ✗ Single page test FAILED ({errors} errors)")

    # Test 4b: First and last pages
    print("\nTest 4b: First and last pages")
    gpu_k_staging = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v_staging = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')

    sparse_indices = torch.tensor([0, total_pages - 1], dtype=torch.int32, device='cuda')

    copy_sparse_kv_cpu_to_gpu(cpu_k, cpu_v, gpu_k_staging, gpu_v_staging, sparse_indices, page_size, head_num)
    torch.cuda.synchronize()

    # Verify first page (page 0)
    errors = 0
    for token_offset in range(page_size):
        for head_idx in range(head_num):
            src_linear_idx = token_offset * head_num + head_idx
            expected_k = cpu_k[src_linear_idx, 0, :].cuda()
            # First page is copied to staging buffer starting at index 0
            dst_linear_idx = token_offset * head_num + head_idx
            actual_k = gpu_k_staging[dst_linear_idx, 0, :]
            if not torch.allclose(actual_k, expected_k, rtol=1e-2, atol=1e-2):
                errors += 1

    # Verify last page (page total_pages-1, copied as second page in staging)
    for token_offset in range(page_size):
        src_token_idx = (total_pages - 1) * page_size + token_offset
        for head_idx in range(head_num):
            src_linear_idx = src_token_idx * head_num + head_idx
            expected_k = cpu_k[src_linear_idx, 0, :].cuda()
            # Last page is copied right after first page in staging buffer
            dst_linear_idx = page_size * head_num + token_offset * head_num + head_idx
            actual_k = gpu_k_staging[dst_linear_idx, 0, :]
            if not torch.allclose(actual_k, expected_k, rtol=1e-2, atol=1e-2):
                errors += 1

    if errors == 0:
        print("  ✓ First/last page test PASSED")
        return True
    else:
        print(f"  ✗ First/last page test FAILED ({errors} errors)")
        return False


def test_performance():
    """Test 5: Performance benchmark."""
    print("\n" + "="*80)
    print("TEST 5: Performance Benchmark")
    print("="*80)

    page_size = 16
    head_num = 32
    head_dim = 128
    total_pages = 1000
    num_sparse_pages = 64
    total_tokens = total_pages * page_size
    num_warmup = 10
    num_iters = 100

    print(f"Configuration:")
    print(f"  Page size: {page_size}")
    print(f"  Head num: {head_num}")
    print(f"  Head dim: {head_dim}")
    print(f"  Total pages: {total_pages}")
    print(f"  Sparse pages: {num_sparse_pages}")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Benchmark iterations: {num_iters}")

    # Create buffers
    cpu_k = create_test_cpu_buffer(total_tokens, head_num, head_dim)
    cpu_v = create_test_cpu_buffer(total_tokens, head_num, head_dim)

    # 3D staging buffers
    gpu_k_staging = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v_staging = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')

    sparse_indices = torch.randperm(total_pages)[:num_sparse_pages].to(torch.int32).cuda()

    bytes_per_copy = num_sparse_pages * page_size * head_num * head_dim * 2 * 2  # K+V, bfloat16=2 bytes

    # Benchmark 1: Naive baseline (tensor.copy_)
    print("\n--- Benchmarking Naive Baseline (tensor.copy_) ---")
    gpu_k_baseline = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')
    gpu_v_baseline = torch.zeros((total_tokens * head_num, 1, head_dim), dtype=torch.bfloat16, device='cuda')

    # Warmup
    for _ in range(num_warmup):
        naive_cpu_to_gpu_copy(cpu_k, cpu_v, gpu_k_baseline, gpu_v_baseline, sparse_indices, page_size, head_num)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        naive_cpu_to_gpu_copy(cpu_k, cpu_v, gpu_k_baseline, gpu_v_baseline, sparse_indices, page_size, head_num)
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
    print(f"  ✓ Performance test completed")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CPU-GPU KV COPY KERNELS TEST SUITE")
    print("="*80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Tests require GPU.")
        return

    results = []

    try:
        results.append(("CPU->GPU Sparse Copy", test_cpu_to_gpu_sparse_copy()))
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("CPU->GPU Sparse Copy", False))

    try:
        results.append(("GPU->CPU Store", test_gpu_to_cpu_store()))
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("GPU->CPU Store", False))

    try:
        results.append(("Round-trip Consistency", test_roundtrip_consistency()))
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Round-trip Consistency", False))

    try:
        results.append(("Edge Cases", test_edge_cases()))
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Edge Cases", False))

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
        print(f"  {test_name:30s} {status}")

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
