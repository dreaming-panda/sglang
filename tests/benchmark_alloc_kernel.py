#!/usr/bin/env python3
"""Benchmark allocation kernel variants across fill ratios.

Sweeps the ratio of requested pages to staging capacity (0.5 to 1.0)
and measures latency + unresolved pages for all three kernels.

Usage:
    python benchmark_alloc_kernel.py [--json-output results.json]
"""

import argparse
import json
import torch
import numpy as np

import vortex_torch.cache


WARMUP = 5
ITERS = 20

PAGE_SIZE = 16
NUM_KV_HEADS = 8
ASSOCIATIVITY = 32

NUM_SETS = 1024
STAGING_CAPACITY = NUM_SETS * ASSOCIATIVITY  # 32768

MAX_CPU_PAGES = 200000


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def prefill_cache(cpu_to_gpu, gpu_to_cpu, slot_ages, staging_capacity, fill_start):
    n = staging_capacity
    page_ids = torch.arange(fill_start, fill_start + n, dtype=torch.int32, device=cpu_to_gpu.device)
    slot_ids = torch.arange(n, dtype=torch.int32, device=cpu_to_gpu.device)
    gpu_to_cpu[:n] = page_ids
    cpu_to_gpu[page_ids.long()] = slot_ids
    slot_ages[:n] = torch.randint(1, 33, (n,), dtype=torch.uint8, device=slot_ages.device)


def make_indptr(n_pages, batch_size, num_kv_heads):
    n_entries = batch_size * num_kv_heads
    pages_per_entry = n_pages // n_entries
    indptr = torch.zeros(n_entries + 1, dtype=torch.int32, device="cuda")
    for i in range(n_entries):
        indptr[i + 1] = indptr[i] + pages_per_entry
    indptr[-1] = n_pages
    return indptr.contiguous()


def bench_kernel(kernel_name, sparse_indices, sparse_indptr, batch_size,
                 warmup=WARMUP, iters=ITERS):
    device = torch.device("cuda")
    n_pages = sparse_indices.shape[0]
    fill_start = MAX_CPU_PAGES // 2

    cpu_to_gpu = torch.full((MAX_CPU_PAGES,), -1, dtype=torch.int32, device=device)
    gpu_to_cpu = torch.full((STAGING_CAPACITY,), -1, dtype=torch.int32, device=device)
    slot_ages = torch.zeros(STAGING_CAPACITY, dtype=torch.uint8, device=device)
    set_used_mask = torch.zeros(NUM_SETS, dtype=torch.int32, device=device)

    dst_gpu_slots = torch.full((STAGING_CAPACITY,), -1, dtype=torch.int32, device=device)
    owners_bitmap = torch.zeros(STAGING_CAPACITY, dtype=torch.bool, device=device)
    evicted_cpu_pages = torch.full((STAGING_CAPACITY,), -1, dtype=torch.int32, device=device)
    overflow_flag = torch.zeros(1, dtype=torch.int32, device=device)

    def run_once():
        overflow_flag.zero_()
        evicted_cpu_pages.fill_(-1)
        dst_gpu_slots.fill_(-1)

        if kernel_name == "lru_block":
            vortex_torch.cache.allocate_pages_lru_block(
                sparse_kv_indices=sparse_indices, sparse_kv_indptr=sparse_indptr,
                cpu_to_gpu_slot_map=cpu_to_gpu, gpu_to_cpu_page_map=gpu_to_cpu,
                slot_ages=slot_ages, set_used_mask=set_used_mask,
                dst_gpu_slots=dst_gpu_slots, owners_bitmap=owners_bitmap,
                evicted_cpu_pages=evicted_cpu_pages, overflow_flag=overflow_flag,
                batch_size=batch_size, num_kv_heads=NUM_KV_HEADS,
                max_num_pages=STAGING_CAPACITY)
        elif kernel_name == "lru_global":
            vortex_torch.cache.allocate_pages_lru_global(
                sparse_kv_indices=sparse_indices, sparse_kv_indptr=sparse_indptr,
                cpu_to_gpu_slot_map=cpu_to_gpu, gpu_to_cpu_page_map=gpu_to_cpu,
                slot_ages=slot_ages, set_used_mask=set_used_mask,
                dst_gpu_slots=dst_gpu_slots, owners_bitmap=owners_bitmap,
                evicted_cpu_pages=evicted_cpu_pages, overflow_flag=overflow_flag,
                batch_size=batch_size, num_kv_heads=NUM_KV_HEADS,
                max_num_pages=STAGING_CAPACITY)
        elif kernel_name == "lru_block_global":
            vortex_torch.cache.allocate_pages_lru_block_global(
                sparse_kv_indices=sparse_indices, sparse_kv_indptr=sparse_indptr,
                cpu_to_gpu_slot_map=cpu_to_gpu, gpu_to_cpu_page_map=gpu_to_cpu,
                slot_ages=slot_ages, set_used_mask=set_used_mask,
                dst_gpu_slots=dst_gpu_slots, owners_bitmap=owners_bitmap,
                evicted_cpu_pages=evicted_cpu_pages, overflow_flag=overflow_flag,
                batch_size=batch_size, num_kv_heads=NUM_KV_HEADS,
                max_num_pages=STAGING_CAPACITY)

    # Warmup
    for _ in range(warmup):
        cpu_to_gpu.fill_(-1); gpu_to_cpu.fill_(-1); slot_ages.zero_()
        prefill_cache(cpu_to_gpu, gpu_to_cpu, slot_ages, STAGING_CAPACITY, fill_start)
        run_once()
        torch.cuda.synchronize()

    # Timed runs
    times = []
    total_unresolved = 0
    total_overflow = 0
    for _ in range(iters):
        cpu_to_gpu.fill_(-1); gpu_to_cpu.fill_(-1); slot_ages.zero_()
        prefill_cache(cpu_to_gpu, gpu_to_cpu, slot_ages, STAGING_CAPACITY, fill_start)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_once()
        e.record()
        torch.cuda.synchronize()

        times.append(s.elapsed_time(e))
        if overflow_flag.item() != 0:
            total_overflow += 1
        total_unresolved += (dst_gpu_slots[:n_pages] == -1).sum().item()

    return median(times), total_overflow, total_unresolved


def main():
    parser = argparse.ArgumentParser(description="Benchmark allocation kernel variants")
    parser.add_argument("--json-output", type=str, default="alloc_kernel_benchmark.json")
    args = parser.parse_args()

    kernels = ["lru_block", "lru_global", "lru_block_global"]
    fill_ratios = [i / 100 for i in range(50, 98, 2)]  # 50%, 52%, 54%, ..., 96%
    req_range = MAX_CPU_PAGES // 2

    results = []

    for ratio in fill_ratios:
        n_pages = int(STAGING_CAPACITY * ratio)
        batch_size = max(n_pages // (NUM_KV_HEADS * 32), 1)

        # All distinct pages not in cache (guaranteed misses)
        sparse_indices = torch.randperm(req_range)[:n_pages].to(torch.int32).cuda().contiguous()
        sparse_indptr = make_indptr(n_pages, batch_size, NUM_KV_HEADS)

        for kernel in kernels:
            print(f"  {kernel} @ {ratio:.0%} fill ({n_pages} pages)...", end=" ", flush=True)
            lat_ms, overflow, unresolved = bench_kernel(kernel, sparse_indices, sparse_indptr, batch_size)
            unresolved_per_iter = unresolved / ITERS
            print(f"latency={lat_ms:.3f}ms, overflow={overflow}/{ITERS}, "
                  f"unresolved/iter={unresolved_per_iter:.1f}")

            results.append({
                "fill_ratio": ratio,
                "kernel": kernel,
                "latency_ms": round(lat_ms, 4),
                "overflow_triggered": overflow,
                "unresolved_per_iter": round(unresolved_per_iter, 2),
                "num_pages": n_pages,
            })

    with open(args.json_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.json_output}")


if __name__ == "__main__":
    main()
