#!/usr/bin/env python3
"""Plot allocation kernel benchmark: latency and unresolved pages vs fill ratio.

Two-panel line plot:
  Left:  Latency (ms) vs fill ratio, one line per kernel.
  Right: Unresolved pages per iteration vs fill ratio, one line per kernel.

Usage:
    python plot_alloc_kernel.py [--input results.json] [--output plots/]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

STYLES = {
    "lru_block":        {"color": "#1565C0", "marker": "s", "label": "Block-Local"},
    "lru_global":       {"color": "#E65100", "marker": "^", "label": "Global"},
    "lru_block_global": {"color": "#2E7D32", "marker": "o", "label": "Hybrid (Ours)"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="alloc_kernel_benchmark.json")
    parser.add_argument("--output", type=str, default="profile_plots")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.output, exist_ok=True)

    kernels = ["lru_block", "lru_global", "lru_block_global"]
    fill_ratios = sorted(set(d["fill_ratio"] for d in data))

    # Build lookup
    lookup = {}
    for d in data:
        lookup[(d["fill_ratio"], d["kernel"])] = d

    # Only keep fill ratios where at least one kernel has perfect 0 unresolved
    valid_ratios = []
    for r in fill_ratios:
        has_perfect_zero = any(
            lookup.get((r, k), {}).get("unresolved_per_iter", 1) == 0
            for k in kernels if (r, k) in lookup
        )
        if has_perfect_zero:
            valid_ratios.append(r)

    fig, (ax_lat, ax_unres) = plt.subplots(1, 2, figsize=(13, 5))

    for kernel in kernels:
        style = STYLES[kernel]
        ratios = [r for r in valid_ratios if (r, kernel) in lookup]
        lats = [lookup[(r, kernel)]["latency_ms"] for r in ratios]
        unres = [lookup[(r, kernel)]["unresolved_per_iter"] for r in ratios]
        pcts = [r * 100 for r in ratios]

        ax_lat.plot(pcts, lats, color=style["color"], marker=style["marker"],
                    linewidth=1.8, markersize=5, label=style["label"])
        ax_unres.plot(pcts, unres, color=style["color"], marker=style["marker"],
                      linewidth=1.8, markersize=5, label=style["label"])

    ax_lat.set_xlabel("Fill Ratio (%)")
    ax_lat.set_ylabel("Allocation Latency (ms)")
    ax_lat.set_title("Latency vs Fill Ratio")
    ax_lat.legend()
    ax_lat.grid(True, alpha=0.3)
    ax_lat.set_ylim(bottom=0)

    ax_unres.set_xlabel("Fill Ratio (%)")
    ax_unres.set_ylabel("Unresolved Pages / Iteration")
    ax_unres.set_title("Unresolved Pages vs Fill Ratio")
    ax_unres.legend()
    ax_unres.grid(True, alpha=0.3)
    ax_unres.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(args.output, "alloc_kernel_ablation.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
