#!/usr/bin/env python3
"""Plot copy kernel bandwidth benchmark results for paper.

Reads JSON output from copy_kernel_benchmark.py and generates a publication-quality
figure showing achieved PCIe bandwidth vs number of sparse pages.

Usage:
    # First run the benchmark with JSON output:
    python copy_kernel_benchmark.py --json-output copy_benchmark_results.json

    # Then plot:
    python plot_copy_bandwidth.py --input copy_benchmark_results.json
    python plot_copy_bandwidth.py --input copy_benchmark_results.json --output plots/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_bandwidth(results, output_dir):
    """Generate bandwidth comparison plot: baseline vs tiled kernel."""
    num_pages = [r["num_pages"] for r in results]
    baseline_gbps = [r["baseline_gbps"] for r in results]
    tiled_gbps = [r["tiled_gbps"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: Bandwidth
    x = np.arange(len(num_pages))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, baseline_gbps, width, label="Baseline (1D)",
                    color="#C44E52", alpha=0.85)
    bars2 = ax1.bar(x + width / 2, tiled_gbps, width, label="Tiled (2D)",
                    color="#4C72B0", alpha=0.85)

    ax1.set_xlabel("Number of Sparse Pages")
    ax1.set_ylabel("Effective Bandwidth (GB/s)")
    ax1.set_title("Copy Kernel: Achieved PCIe Bandwidth")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{p // 1000}K" if p >= 1000 else str(p) for p in num_pages])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, tiled_gbps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    # Panel 2: Latency
    baseline_ms = [r["baseline_ms"] for r in results]
    tiled_ms = [r["tiled_ms"] for r in results]

    bars1 = ax2.bar(x - width / 2, baseline_ms, width, label="Baseline (1D)",
                    color="#C44E52", alpha=0.85)
    bars2 = ax2.bar(x + width / 2, tiled_ms, width, label="Tiled (2D)",
                    color="#4C72B0", alpha=0.85)

    ax2.set_xlabel("Number of Sparse Pages")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Copy Kernel: Transfer Latency")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p // 1000}K" if p >= 1000 else str(p) for p in num_pages])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, tiled_ms):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "copy_bandwidth.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot copy kernel bandwidth benchmark")
    parser.add_argument("--input", type=str, default="copy_benchmark_results.json",
                        help="JSON file from copy_kernel_benchmark.py --json-output")
    parser.add_argument("--output", type=str, default="plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    if not results:
        print("No results found in input file.")
        return

    os.makedirs(args.output, exist_ok=True)
    plot_bandwidth(results, args.output)

    # Print summary
    print("\nSummary:")
    print(f"{'Pages':>10} {'Baseline GB/s':>15} {'Tiled GB/s':>15} {'Speedup':>10}")
    for r in results:
        speedup = r["tiled_gbps"] / r["baseline_gbps"] if r["baseline_gbps"] > 0 else 0
        print(f"{r['num_pages']:>10} {r['baseline_gbps']:>15.1f} {r['tiled_gbps']:>15.1f} {speedup:>10.1f}x")


if __name__ == "__main__":
    main()
