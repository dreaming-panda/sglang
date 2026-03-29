#!/usr/bin/env python3
"""Plot profiling data (allocation latency, copy latency, cache hit rate) from JSONL.

Supports multiple input files to compare across alloc kernels / models on the same plot.
Uses multiprocessing for parallel file loading and vectorized aggregation.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]


def load_and_aggregate(args_tuple):
    """Load a profile JSONL file via pandas streaming and aggregate. Runs in worker process."""
    label, filepath, max_tokens = args_tuple

    # Stream in chunks via pandas C parser — much faster than json.loads per line
    chunks = []
    for chunk in pd.read_json(filepath, lines=True, chunksize=50000,
                              dtype={"tokens": np.int64, "alloc_ms": np.float64,
                                     "copy_ms": np.float64, "hit_rate": np.float64}):
        if max_tokens > 0:
            chunk = chunk[chunk["tokens"] <= max_tokens]
        chunks.append(chunk)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        return None

    # Vectorized group-by mean — single pandas call
    grouped = df.groupby("tokens", sort=True)[["alloc_ms", "copy_ms", "hit_rate"]].mean()

    unique_tokens = grouped.index.values.astype(np.int64)
    alloc_means = grouped["alloc_ms"].values
    copy_means = grouped["copy_ms"].values
    hit_means = grouped["hit_rate"].values

    return (label, unique_tokens, alloc_means, copy_means, hit_means)


def rolling_mean(arr, window):
    """Compute rolling mean with given window size."""
    if window <= 1 or len(arr) <= window:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def auto_discover(base_dir, model, mem_frac, max_tokens, kernels):
    """Auto-discover profile JSONL files for given model and kernels."""
    inputs = []
    for kernel in kernels:
        path = os.path.join(base_dir, model, "AIME24", f"gpu_{mem_frac}", f"max_tokens_{max_tokens}", kernel, "profile_data.jsonl")
        if os.path.exists(path):
            inputs.append((f"{model.split('/')[-1]}_{kernel}", path))
        else:
            print(f"Warning: not found: {path}")
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Plot vortex profiling data")
    parser.add_argument("--input", type=str, nargs="*", help="Profile JSONL files (label:path or just path)")
    parser.add_argument("--models", type=str, nargs="*", default=["Qwen/Qwen3-8B"], help="Model names for auto-discovery")
    parser.add_argument("--alloc-kernels", type=str, nargs="*", default=["lru_block", "lru_global", "lru_block_global"], help="Alloc kernels for auto-discovery")
    parser.add_argument("--mem-fraction", type=float, default=0.9, help="Memory fraction for auto-discovery")
    parser.add_argument("--gen-max-tokens", type=int, default=16384, help="Max tokens used during generation (for path discovery)")
    parser.add_argument("--max-tokens", type=int, default=0, help="Max tokens to plot (0 = all)")
    parser.add_argument("--window", type=int, default=50, help="Rolling average window size")
    parser.add_argument("--output", type=str, default="profile_plots", help="Output directory for plots")
    parser.add_argument("--base-dir", type=str, default="DATA", help="Base data directory")
    parser.add_argument("--scatter-max", type=int, default=5000, help="Max scatter points per dataset (0 = no scatter)")
    args = parser.parse_args()

    # Collect inputs
    all_inputs = []
    if args.input:
        for inp in args.input:
            if ":" in inp and not inp.startswith("/"):
                label, path = inp.split(":", 1)
            else:
                label = os.path.basename(os.path.dirname(inp))
                path = inp
            all_inputs.append((label, path))
    else:
        for model in args.models:
            all_inputs.extend(auto_discover(args.base_dir, model, args.mem_fraction, args.gen_max_tokens, args.alloc_kernels))

    if not all_inputs:
        print("No input files found. Use --input or --models/--alloc-kernels for auto-discovery.")
        return

    # Parallel load + aggregate
    worker_args = [(label, path, args.max_tokens) for label, path in all_inputs]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(load_and_aggregate, worker_args))

    datasets = [r for r in results if r is not None]
    if not datasets:
        print("No valid data to plot.")
        return

    os.makedirs(args.output, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    titles = ["Allocation Kernel Latency (avg across layers)", "Copy Kernel Latency (avg across layers)", "Cache Hit Rate (avg across layers)"]
    ylabels = ["Alloc Kernel (ms)", "Copy Kernel (ms)", "Cache Hit Rate (%)"]

    for i, (label, tokens, alloc_ms, copy_ms, hit_rate) in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        w = min(args.window, len(tokens))

        for ax_idx, (data, is_pct) in enumerate([(alloc_ms, False), (copy_ms, False), (hit_rate, True)]):
            y = data * 100 if is_pct else data
            smooth = rolling_mean(data, w) * (100 if is_pct else 1)
            tokens_smooth = tokens[w - 1:] if w > 1 and len(tokens) > w else tokens

            # Downsample scatter for performance
            if args.scatter_max > 0 and len(tokens) > args.scatter_max:
                idx = np.linspace(0, len(tokens) - 1, args.scatter_max, dtype=int)
                axes[ax_idx].scatter(tokens[idx], y[idx], alpha=0.08, s=1, color=color, rasterized=True)
            elif args.scatter_max > 0:
                axes[ax_idx].scatter(tokens, y, alpha=0.08, s=1, color=color, rasterized=True)

            axes[ax_idx].plot(tokens_smooth, smooth, color=color, linewidth=1.5, label=label)

    for ax_idx in range(3):
        axes[ax_idx].set_ylabel(ylabels[ax_idx])
        axes[ax_idx].set_title(titles[ax_idx])
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].grid(True, alpha=0.3)
    axes[2].set_ylim(-5, 105)
    axes[2].set_xlabel("Tokens Generated")

    plt.tight_layout()
    out_path = os.path.join(args.output, "profile.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    # Print summary stats
    for label, tokens, alloc_ms, copy_ms, hit_rate in datasets:
        print(f"\n{label} ({len(tokens)} steps, {tokens[-1]} max tokens):")
        print(f"  Alloc: mean={alloc_ms.mean():.3f}ms, p50={np.median(alloc_ms):.3f}ms, p99={np.percentile(alloc_ms, 99):.3f}ms")
        print(f"  Copy:  mean={copy_ms.mean():.3f}ms, p50={np.median(copy_ms):.3f}ms, p99={np.percentile(copy_ms, 99):.3f}ms")
        print(f"  Hit:   mean={hit_rate.mean()*100:.1f}%, min={hit_rate.min()*100:.1f}%, max={hit_rate.max()*100:.1f}%")


if __name__ == "__main__":
    main()
