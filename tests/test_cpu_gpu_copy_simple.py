"""
Simple focused test for CPU->GPU sparse copy kernel with 3D layout.

This test verifies the basic correctness of copying sparse pages from CPU to GPU
with the 3D tensor layout [tokens*heads, 1, head_dim].
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sglang.srt.mem_cache.cpu_gpu_copy_kernels import copy_sparse_kv_cpu_to_gpu


def create_test_cpu_buffer(num_tokens, head_num, head_dim, dtype=torch.bfloat16):
    """Create CPU buffer with identifiable test data."""
    buffer = torch.zeros(
        (num_tokens * head_num, 1, head_dim),
        dtype=dtype,
        device='cpu',
        pin_memory=True,
    )
    # Fill with pattern: token_idx * 1000 + head_idx * 10 + dim_idx
    for token_idx in range(num_tokens):
        for head_idx in range(head_num):
            linear_idx = token_idx * head_num + head_idx
            buffer[linear_idx, 0, :] = (
                token_idx * 1000.0 + head_idx * 10.0 + torch.arange(head_dim, dtype=torch.float32)
            ).to(dtype)
    return buffer


def test_simple_copy():
    """Test 1: Simple case with clear expectations."""
    print("\n" + "="*80)
    print("TEST: Simple CPU->GPU Sparse Copy (3D Layout)")
    print("="*80)

    # Simple configuration
    page_size = 16
    head_num = 8
    head_dim = 128
    batch_size = 2  # 2 sequences
    total_pages = 10
    total_tokens = total_pages * page_size

    print(f"\nConfiguration:")
    print(f"  Page size: {page_size}")
    print(f"  Head num: {head_num}")
    print(f"  Head dim: {head_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total pages in CPU cache: {total_pages}")

    # Create CPU buffers with test data
    cpu_k = create_test_cpu_buffer(total_tokens, head_num, head_dim)
    cpu_v = create_test_cpu_buffer(total_tokens, head_num, head_dim)
    cpu_v.mul_(2)  # Make V different from K

    # Setup sparse selection
    # Each sequence/head selects global pages (each page contains all head_num heads)
    # Simpler test: 2 sequences, 1 head each
    num_kv_heads = 1

    # Example:
    # Sequence 0, head 0: uses global pages [3, 4, 5]
    # Sequence 1, head 0: uses global pages [9, 1, 2]
    #
    # Note: total_pages = 10, so valid page indices are 0-9
    # sparse_indices = [3, 4, 5, 9, 1, 2]
    # sparse_indptr = [0, 3, 6]
    #   - seq0_head0: pages sparse_indices[0:3] = [3, 4, 5]
    #   - seq1_head0: pages sparse_indices[3:6] = [9, 1, 2]

    sparse_indices = torch.tensor([3, 4, 5, 9, 1, 2], dtype=torch.int32, device='cuda')
    sparse_indptr = torch.tensor([0, 3, 6], dtype=torch.int32, device='cuda')

    num_seq_heads = batch_size * num_kv_heads
    num_sparse_pages = sparse_indices.shape[0]

    print(f"\nSparse selection:")
    print(f"  Num sequence*heads: {num_seq_heads}")
    print(f"  Total sparse pages: {num_sparse_pages}")
    print(f"  sparse_indices: {sparse_indices.cpu().tolist()}")
    print(f"  sparse_indptr: {sparse_indptr.cpu().tolist()}")

    # Create GPU staging buffer
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

    # Run kernel
    print("\nRunning CPU->GPU copy kernel...")
    copy_sparse_kv_cpu_to_gpu(
        cpu_k_buffer=cpu_k,
        cpu_v_buffer=cpu_v,
        gpu_k_staging=gpu_k_staging,
        gpu_v_staging=gpu_v_staging,
        sparse_indices=sparse_indices,
        sparse_indptr=sparse_indptr,
        page_size=page_size,
        head_num=head_num,
    )
    torch.cuda.synchronize()
    print("Kernel completed.")

    # Verify correctness
    print("\nVerifying correctness...")
    errors = 0

    # Check sequence 0, head 0: pages [2, 5] should be at staging positions [0, 1]
    seq_head_idx = 0
    page_start = sparse_indptr[seq_head_idx].item()
    page_end = sparse_indptr[seq_head_idx + 1].item()

    print(f"\n  Checking seq_head {seq_head_idx} (pages {page_start}:{page_end}):")
    for i, page_idx in enumerate(range(page_start, page_end)):
        src_global_page = sparse_indices[page_idx].item()
        dst_global_page = page_idx  # Contiguous in staging buffer

        print(f"    Staging page {dst_global_page} should contain CPU page {src_global_page}")

        # Verify each token in this page, for all heads
        for token_offset in range(page_size):
            for head_offset in range(head_num):
                # Source: CPU buffer
                src_token_idx = src_global_page * page_size + token_offset
                cpu_linear_idx = src_token_idx * head_num + head_offset
                expected_k = cpu_k[cpu_linear_idx, 0, :].cuda()

                # Destination: GPU staging buffer
                dst_token_idx = dst_global_page * page_size + token_offset
                gpu_linear_idx = dst_token_idx * head_num + head_offset
                actual_k = gpu_k_staging[gpu_linear_idx, 0, :]

                if not torch.allclose(actual_k, expected_k, rtol=1e-2, atol=1e-2):
                    errors += 1
                    if errors <= 5:
                        print(f"      ERROR: page {dst_global_page}, token {token_offset}, head {head_offset}")
                        print(f"        Expected (first 5): {expected_k[:5].float()}")
                        print(f"        Actual (first 5):   {actual_k[:5].float()}")

    if errors == 0:
        print("\n✓ All checks passed!")
        return True
    else:
        print(f"\n✗ Found {errors} errors")
        return False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        exit(1)

    success = test_simple_copy()
    exit(0 if success else 1)
