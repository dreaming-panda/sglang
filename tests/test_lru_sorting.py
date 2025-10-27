"""
Simple test to verify the CUDA warp sorting kernel works correctly.
"""

import torch
import sys
sys.path.insert(0, '/home/kaixin/qilong/sglang/python')

# Import the CUDA kernel from vortex_C
try:
    import vortex_C
except ImportError:
    print("Error: vortex_C module not found. Make sure it's compiled with warp_sort.cu")
    sys.exit(1)


def test_warp_sorting():
    """Test that slots within each warp are sorted by age (oldest first)."""
    print("\n=== Testing Warp Sorting by Age ===")

    device = torch.device('cuda:0')
    WARP_SIZE = 32

    # Test 1: Single warp with random ages
    print("\nTest 1: Single warp (32 slots)")
    num_slots = 32

    # Create random ages
    slot_ages = torch.randint(0, 33, (num_slots,), dtype=torch.int32, device=device)
    available_slots = torch.arange(num_slots, dtype=torch.int32, device=device)

    print(f"Before sorting: {slot_ages.cpu().tolist()}")

    # Create warp indices
    num_warps = 1
    warp_start_indices = torch.tensor([0], dtype=torch.int32, device=device)
    warp_empty_counts = torch.zeros(num_warps, dtype=torch.int32, device=device)

    # Run CUDA sorting kernel
    vortex_C.warp_sort_slots_by_age(
        available_slots,
        slot_ages,
        warp_start_indices,
        warp_empty_counts
    )

    # Get sorted slot IDs and their ages
    sorted_slot_ids = available_slots.cpu()
    sorted_ages = slot_ages[sorted_slot_ids].cpu().tolist()

    print(f"After sorting:  {sorted_ages}")

    # Verify sorted
    is_sorted = True
    for i in range(len(sorted_ages) - 1):
        if sorted_ages[i] > sorted_ages[i + 1]:
            print(f"❌ NOT SORTED: age[{i}]={sorted_ages[i]} > age[{i+1}]={sorted_ages[i+1]}")
            is_sorted = False

    if is_sorted:
        print("✅ Single warp is correctly sorted!")

    # Test 2: Multiple warps
    print("\nTest 2: Multiple warps (128 slots = 4 warps)")
    num_slots = 128
    num_warps = 4

    # Create random ages
    slot_ages = torch.randint(0, 33, (num_slots,), dtype=torch.int32, device=device)
    available_slots = torch.arange(num_slots, dtype=torch.int32, device=device)

    print("Before sorting (first 64):")
    for warp_id in range(2):
        start = warp_id * WARP_SIZE
        end = (warp_id + 1) * WARP_SIZE
        warp_ages = slot_ages[start:end].cpu().tolist()
        print(f"  Warp {warp_id}: {warp_ages}")

    # Create warp indices
    warp_start_indices = torch.arange(0, num_warps * WARP_SIZE, WARP_SIZE,
                                      dtype=torch.int32, device=device)
    warp_empty_counts = torch.zeros(num_warps, dtype=torch.int32, device=device)

    # Run CUDA sorting kernel (all warps in parallel)
    vortex_C.warp_sort_slots_by_age(
        available_slots,
        slot_ages,
        warp_start_indices,
        warp_empty_counts
    )

    print("\nAfter sorting (first 64):")
    all_sorted = True
    for warp_id in range(num_warps):
        start = warp_id * WARP_SIZE
        end = (warp_id + 1) * WARP_SIZE
        warp_slot_ids = available_slots[start:end].cpu()
        warp_ages = slot_ages[warp_slot_ids].cpu().tolist()

        if warp_id < 2:  # Only print first 2 warps
            print(f"  Warp {warp_id}: {warp_ages}")

        # Verify sorted
        for i in range(len(warp_ages) - 1):
            if warp_ages[i] > warp_ages[i + 1]:
                print(f"❌ Warp {warp_id} NOT SORTED: age[{i}]={warp_ages[i]} > age[{i+1}]={warp_ages[i+1]}")
                all_sorted = False
                break

    if all_sorted:
        print("✅ All warps correctly sorted!")

    # Test 3: Partial warp
    print("\nTest 3: Partial warp (50 slots = 1 full warp + 1 partial warp)")
    num_slots = 50
    num_warps = 2

    # Create random ages
    slot_ages = torch.randint(0, 33, (num_slots,), dtype=torch.int32, device=device)
    available_slots = torch.arange(num_slots, dtype=torch.int32, device=device)

    print("Before sorting:")
    print(f"  Warp 0 (32 slots): {slot_ages[:32].cpu().tolist()}")
    print(f"  Warp 1 (18 slots): {slot_ages[32:50].cpu().tolist()}")

    # Create warp indices
    warp_start_indices = torch.arange(0, num_warps * WARP_SIZE, WARP_SIZE,
                                      dtype=torch.int32, device=device)
    warp_empty_counts = torch.zeros(num_warps, dtype=torch.int32, device=device)

    # Run CUDA sorting kernel
    vortex_C.warp_sort_slots_by_age(
        available_slots,
        slot_ages,
        warp_start_indices,
        warp_empty_counts
    )

    print("\nAfter sorting:")
    all_sorted = True
    for warp_id in range(num_warps):
        start = warp_id * WARP_SIZE
        end = min((warp_id + 1) * WARP_SIZE, num_slots)
        warp_slot_ids = available_slots[start:end].cpu()
        warp_ages = slot_ages[warp_slot_ids].cpu().tolist()

        print(f"  Warp {warp_id} ({len(warp_ages)} slots): {warp_ages}")

        # Verify sorted
        for i in range(len(warp_ages) - 1):
            if warp_ages[i] > warp_ages[i + 1]:
                print(f"❌ Warp {warp_id} NOT SORTED: age[{i}]={warp_ages[i]} > age[{i+1}]={warp_ages[i+1]}")
                all_sorted = False
                break

    if all_sorted:
        print("✅ Partial warp correctly sorted!")

    # Test 4: Check empty slot counting
    print("\nTest 4: Empty slot counting")
    num_slots = 64
    num_warps = 2

    # Create ages with some empty slots (age = 0)
    slot_ages = torch.tensor(
        [0, 0, 0, 5, 10, 15, 20, 25, 30, 32, 1, 2, 3, 4, 6, 7] + [8] * 16 +  # Warp 0: 3 empty
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] + [15] * 16,  # Warp 1: 2 empty
        dtype=torch.int32, device=device
    )
    available_slots = torch.arange(num_slots, dtype=torch.int32, device=device)

    warp_start_indices = torch.arange(0, num_warps * WARP_SIZE, WARP_SIZE,
                                      dtype=torch.int32, device=device)
    warp_empty_counts = torch.zeros(num_warps, dtype=torch.int32, device=device)

    # Run CUDA sorting kernel
    vortex_C.warp_sort_slots_by_age(
        available_slots,
        slot_ages,
        warp_start_indices,
        warp_empty_counts
    )

    print(f"Empty counts: {warp_empty_counts.cpu().tolist()}")
    print(f"Expected: [3, 2]")

    if warp_empty_counts[0].item() == 3 and warp_empty_counts[1].item() == 2:
        print("✅ Empty slot counting correct!")
    else:
        print("❌ Empty slot counting incorrect!")

    print("\n" + "=" * 60)
    print("All sorting tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_warp_sorting()
