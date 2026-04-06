#!/usr/bin/env python3
"""Plot profiling data for paper: allocation latency, copy latency, cache hit rate.

Generates multiple paper-quality figures:
1. Combined 3-panel plot (alloc, copy, hit rate vs tokens) per config
2. Kernel comparison bar charts (lru_block vs lru_global vs lru_block_global)
3. Latency distribution (CDF) showing near-constant allocation time
4. Per-layer hit rate analysis

Uses multiprocessing for parallel file loading and vectorized aggregation.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# Paper-friendly style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]
KERNEL_COLORS = {"lru_block": "#C44E52", "lru_global": "#4C72B0", "lru_block_global": "#55A868"}
KERNEL_LABELS = {"lru_block": "Block-Local", "lru_global": "Global", "lru_block_global": "Hybrid (Ours)"}


def load_and_aggregate(args_tuple):
    """Load a profile JSONL file via pandas streaming and aggregate. Runs in worker process."""
    label, filepath, max_tokens = args_tuple

    chunks = []
    for chunk in pd.read_json(filepath, lines=True, chunksize=50000):
        if max_tokens > 0:
            chunk = chunk[chunk["tokens"] <= max_tokens]
        chunks.append(chunk)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        return None

    return (label, df)


def aggregate_by_tokens(df):
    """Aggregate per-token (mean across layers). Filters to regular (non-breakdown) records."""
    regular = df[df.get("type", pd.Series(dtype=str)).isna() | (df.get("type", pd.Series(dtype=str)) != "breakdown")]
    if "alloc_ms" not in regular.columns:
        return np.array([], dtype=np.int64), np.empty((0, 3))
    regular = regular.dropna(subset=["alloc_ms"])
    if regular.empty:
        return np.array([], dtype=np.int64), np.empty((0, 3))
    grouped = regular.groupby("tokens", sort=True)[["alloc_ms", "copy_ms", "hit_rate"]].mean()
    return grouped.index.values.astype(np.int64), grouped.values


def rolling_mean(arr, window):
    """Compute rolling mean with given window size."""
    if window <= 1 or len(arr) <= window:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def auto_discover(base_dir, model, mem_frac, max_tokens, kernels, kv_dtypes):
    """Auto-discover profile JSONL files."""
    inputs = []
    model_short = model.split('/')[-1]
    base = os.path.join(base_dir, model, "AIME24", f"gpu_{mem_frac}", f"max_tokens_{max_tokens}")

    for kernel in kernels:
        for kv_dtype in kv_dtypes:
            if kv_dtype == "auto":
                path = os.path.join(base, kernel, "profile_data.jsonl")
                label = f"{model_short}_{kernel}"
            else:
                path = os.path.join(base, kernel, kv_dtype, "profile_data.jsonl")
                label = f"{model_short}_{kernel}_{kv_dtype}"
            if os.path.exists(path):
                inputs.append((label, path, kernel, kv_dtype))

    if not inputs:
        print(f"Warning: no profile data found for {model}")
    return inputs


# ===== Plot 1: Combined 3-panel (alloc, copy, hit rate vs tokens) =====

def plot_combined(datasets, output_dir, window=50, scatter_max=5000):
    """3-panel time series: alloc latency, copy latency, hit rate vs tokens generated."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    titles = ["Allocation Latency", "Copy Latency", "Cache Hit Rate"]
    ylabels = ["Latency (ms)", "Latency (ms)", "Hit Rate (%)"]

    for i, (label, df, kernel, kv_dtype) in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        tokens, vals = aggregate_by_tokens(df)
        alloc_ms, copy_ms, hit_rate = vals[:, 0], vals[:, 1], vals[:, 2]
        w = min(window, len(tokens))

        for ax_idx, (data, is_pct) in enumerate([(alloc_ms, False), (copy_ms, False), (hit_rate, True)]):
            y = data * 100 if is_pct else data
            smooth = rolling_mean(data, w) * (100 if is_pct else 1)
            tokens_smooth = tokens[w - 1:] if w > 1 and len(tokens) > w else tokens

            if scatter_max > 0 and len(tokens) > scatter_max:
                idx = np.linspace(0, len(tokens) - 1, scatter_max, dtype=int)
                axes[ax_idx].scatter(tokens[idx], y[idx], alpha=0.06, s=1, color=color, rasterized=True)
            elif scatter_max > 0:
                axes[ax_idx].scatter(tokens, y, alpha=0.06, s=1, color=color, rasterized=True)

            display_label = KERNEL_LABELS.get(kernel, kernel)
            if kv_dtype != "auto":
                display_label += f" ({kv_dtype})"
            axes[ax_idx].plot(tokens_smooth, smooth, color=color, linewidth=1.5, label=display_label)

    for ax_idx in range(3):
        axes[ax_idx].set_ylabel(ylabels[ax_idx])
        axes[ax_idx].set_title(titles[ax_idx])
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].grid(True, alpha=0.3)
    axes[2].set_ylim(-5, 105)
    axes[2].set_xlabel("Tokens Generated")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "profile_combined.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ===== Plot 2: Kernel comparison bar charts =====

def plot_kernel_comparison(datasets, output_dir):
    """Bar chart comparing mean alloc latency, copy latency, hit rate across kernels."""
    # Group by kernel
    kernel_stats = {}
    for label, df, kernel, kv_dtype in datasets:
        if kv_dtype != "auto":
            continue  # Only compare bf16 across kernels
        stats = {
            "alloc_mean": df["alloc_ms"].mean(),
            "alloc_p50": df["alloc_ms"].median(),
            "alloc_p99": df["alloc_ms"].quantile(0.99),
            "copy_mean": df["copy_ms"].mean(),
            "copy_p50": df["copy_ms"].median(),
            "hit_mean": df["hit_rate"].mean() * 100,
        }
        kernel_stats[kernel] = stats

    if len(kernel_stats) < 2:
        return

    kernels = list(kernel_stats.keys())
    x = np.arange(len(kernels))
    labels = [KERNEL_LABELS.get(k, k) for k in kernels]
    colors = [KERNEL_COLORS.get(k, "#888888") for k in kernels]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Alloc latency (mean + p99)
    means = [kernel_stats[k]["alloc_mean"] for k in kernels]
    p99s = [kernel_stats[k]["alloc_p99"] for k in kernels]
    bars = axes[0].bar(x, means, color=colors, width=0.5, label="Mean")
    axes[0].bar(x, p99s, color=colors, width=0.5, alpha=0.3, label="p99")
    for bar, val in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.3f}",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Allocation Latency")
    axes[0].legend(fontsize=8)

    # Copy latency (mean)
    means = [kernel_stats[k]["copy_mean"] for k in kernels]
    bars = axes[1].bar(x, means, color=colors, width=0.5)
    for bar, val in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.3f}",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Copy Latency")

    # Hit rate (mean)
    means = [kernel_stats[k]["hit_mean"] for k in kernels]
    bars = axes[2].bar(x, means, color=colors, width=0.5)
    for bar, val in zip(bars, means):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=9)
    axes[2].set_ylabel("Hit Rate (%)")
    axes[2].set_title("Cache Hit Rate")
    axes[2].set_ylim(0, 105)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "kernel_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ===== Plot 3: Latency CDF =====

def plot_latency_cdf(datasets, output_dir):
    """CDF of allocation and copy latency showing near-constant behavior."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i, (label, df, kernel, kv_dtype) in enumerate(datasets):
        if kv_dtype != "auto":
            continue
        color = KERNEL_COLORS.get(kernel, COLORS[i % len(COLORS)])
        display_label = KERNEL_LABELS.get(kernel, kernel)

        for ax_idx, col in enumerate(["alloc_ms", "copy_ms"]):
            sorted_vals = np.sort(df[col].values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            # Downsample for plotting
            if len(sorted_vals) > 10000:
                idx = np.linspace(0, len(sorted_vals) - 1, 10000, dtype=int)
                axes[ax_idx].plot(sorted_vals[idx], cdf[idx] * 100, color=color,
                                 linewidth=1.5, label=display_label)
            else:
                axes[ax_idx].plot(sorted_vals, cdf * 100, color=color,
                                 linewidth=1.5, label=display_label)

    axes[0].set_xlabel("Allocation Latency (ms)")
    axes[0].set_ylabel("CDF (%)")
    axes[0].set_title("Allocation Latency Distribution")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Copy Latency (ms)")
    axes[1].set_ylabel("CDF (%)")
    axes[1].set_title("Copy Latency Distribution")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "latency_cdf.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ===== Plot 4: Per-layer hit rate =====

def plot_per_layer_hit_rate(datasets, output_dir):
    """Hit rate broken down by layer ID."""
    for label, df, kernel, kv_dtype in datasets:
        if kv_dtype != "auto":
            continue
        if "layer_id" not in df.columns:
            continue

        layer_stats = df.groupby("layer_id")["hit_rate"].mean() * 100
        if layer_stats.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        display_label = KERNEL_LABELS.get(kernel, kernel)
        color = KERNEL_COLORS.get(kernel, "#4C72B0")

        ax.bar(layer_stats.index, layer_stats.values, color=color, width=0.7)
        ax.set_xlabel("Layer ID")
        ax.set_ylabel("Cache Hit Rate (%)")
        ax.set_title(f"Per-Layer Cache Hit Rate — {display_label}")
        ax.set_ylim(0, 105)
        ax.axhline(y=layer_stats.mean(), color="red", linestyle="--", linewidth=1,
                    label=f"Mean: {layer_stats.mean():.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"hit_rate_per_layer_{kernel}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved {out_path}")


# ===== Plot 5: Decode step latency breakdown =====

def plot_latency_breakdown(datasets, output_dir):
    """Stacked bar chart of decode step latency breakdown: cache_update, indexer, alloc, copy, attention."""
    for label, df, kernel, kv_dtype in datasets:
        if kv_dtype != "auto":
            continue

        # Breakdown records have "type" == "breakdown" with cache_update_ms, indexer_ms, attention_ms
        # Regular records have alloc_ms, copy_ms
        if "type" not in df.columns:
            # No breakdown data — the profile was run without the breakdown profiling
            continue

        breakdown_df = df[df["type"] == "breakdown"].copy()
        regular_df = df[df["type"] != "breakdown"].copy()

        if breakdown_df.empty or regular_df.empty:
            continue

        # Aggregate: mean across all records
        cache_update = breakdown_df["cache_update_ms"].mean()
        indexer = breakdown_df["indexer_ms"].mean()
        attention = breakdown_df["attention_ms"].mean()
        alloc = regular_df["alloc_ms"].mean()
        copy = regular_df["copy_ms"].mean()

        components = ["Cache Update\n(Centroids)", "Indexer\n(TopK)", "Allocation\n(LRU)",
                      "Copy\n(CPU→GPU)", "Attention\n(Decode)"]
        values = [cache_update, indexer, alloc, copy, attention]
        comp_colors = ["#8172B2", "#CCB974", "#C44E52", "#4C72B0", "#55A868"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [2, 1]})

        # Bar chart
        x = np.arange(len(components))
        bars = ax1.bar(x, values, color=comp_colors, width=0.6)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax1.set_xticks(x)
        ax1.set_xticklabels(components, fontsize=9)
        ax1.set_ylabel("Latency (ms)")
        display_label = KERNEL_LABELS.get(kernel, kernel)
        ax1.set_title(f"Decode Step Breakdown — {display_label}")
        ax1.grid(True, alpha=0.3, axis='y')

        # Pie chart
        total = sum(values)
        pct = [v/total*100 for v in values]
        wedges, texts, autotexts = ax2.pie(
            values, labels=None, colors=comp_colors, autopct='%1.1f%%',
            pctdistance=0.75, startangle=90)
        ax2.legend(wedges, [f"{c} ({v:.2f}ms)" for c, v in zip(
            ["Cache", "Indexer", "Alloc", "Copy", "Attention"], values)],
            fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.set_title(f"Total: {total:.2f}ms")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"latency_breakdown_{kernel}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot vortex profiling data for paper")
    parser.add_argument("--input", type=str, nargs="*", help="Profile JSONL files (label:path)")
    parser.add_argument("--models", type=str, nargs="*", default=["Qwen/Qwen3-8B"])
    parser.add_argument("--alloc-kernels", type=str, nargs="*",
                        default=["lru_block", "lru_global", "lru_block_global"])
    parser.add_argument("--kv-dtypes", type=str, nargs="*", default=["auto"])
    parser.add_argument("--mem-fraction", type=float, default=0.9)
    parser.add_argument("--gen-max-tokens", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=0, help="Max tokens to plot (0=all)")
    parser.add_argument("--window", type=int, default=50, help="Rolling average window")
    parser.add_argument("--output", type=str, default="profile_plots")
    parser.add_argument("--base-dir", type=str, default="results")
    parser.add_argument("--scatter-max", type=int, default=5000)
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
            all_inputs.append((label, path, "unknown", "auto"))
    else:
        for model in args.models:
            all_inputs.extend(auto_discover(args.base_dir, model, args.mem_fraction,
                                            args.gen_max_tokens, args.alloc_kernels, args.kv_dtypes))

    if not all_inputs:
        print("No input files found. Use --input or --models for auto-discovery.")
        return

    # Parallel load
    worker_args = [(label, path, args.max_tokens) for label, path, _, _ in all_inputs]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(load_and_aggregate, worker_args))

    datasets = []
    for result, (label, path, kernel, kv_dtype) in zip(results, all_inputs):
        if result is not None:
            _, df = result
            datasets.append((label, df, kernel, kv_dtype))

    if not datasets:
        print("No valid data to plot.")
        return

    os.makedirs(args.output, exist_ok=True)

    print("Generating plots...")
    print("\n1. Combined time series:")
    plot_combined(datasets, args.output, args.window, args.scatter_max)

    print("\n2. Kernel comparison:")
    plot_kernel_comparison(datasets, args.output)

    print("\n3. Latency CDF:")
    plot_latency_cdf(datasets, args.output)

    print("\n4. Per-layer hit rate:")
    plot_per_layer_hit_rate(datasets, args.output)

    print("\n5. Decode step breakdown:")
    plot_latency_breakdown(datasets, args.output)

    # Print summary stats
    print("\n===== Summary Statistics =====")
    for label, df, kernel, kv_dtype in datasets:
        tokens = df.groupby("tokens").first().index
        print(f"\n{label} ({len(df)} records, {tokens[-1] if len(tokens) > 0 else 0} max tokens):")
        print(f"  Alloc: mean={df['alloc_ms'].mean():.3f}ms, p50={df['alloc_ms'].median():.3f}ms, p99={df['alloc_ms'].quantile(0.99):.3f}ms")
        print(f"  Copy:  mean={df['copy_ms'].mean():.3f}ms, p50={df['copy_ms'].median():.3f}ms, p99={df['copy_ms'].quantile(0.99):.3f}ms")
        print(f"  Hit:   mean={df['hit_rate'].mean()*100:.1f}%, min={df['hit_rate'].min()*100:.1f}%, max={df['hit_rate'].max()*100:.1f}%")

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
