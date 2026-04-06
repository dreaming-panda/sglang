#!/usr/bin/env python3
"""Plot cache policy ablation: per-request allocation/copy latency and hit rate.

Reads *_step.jsonl files produced by model_runner profiling.

Generates two side-by-side plots:
  Left:  Stacked per-request allocation + copy latency vs decode step.
  Right: Cache hit rate vs decode step.

Usage:
    python plot_policy_ablation.py \\
        --input "LRU:path1_step.jsonl" \\
               "LFU:path2_step.jsonl" \\
               "Random:path3_step.jsonl" \\
        --model-name "Qwen3-8B" --max-steps 30000
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


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

STYLE = {
    "LRU": {
        "alloc": "#1565C0",     # blue
        "copy":  "#42A5F5",     # light blue
        "edge":  "#0D47A1",
        "hit":   "#1565C0",
    },
    "LFU": {
        "alloc": "#E65100",     # orange
        "copy":  "#FF9800",     # light orange
        "edge":  "#BF360C",
        "hit":   "#E65100",
    },
    "Random": {
        "alloc": "#2E7D32",     # green
        "copy":  "#81C784",     # light green
        "edge":  "#1B5E20",
        "hit":   "#2E7D32",
    },
}

FALLBACK = {
    "alloc": "#666666", "copy": "#999999",
    "edge": "#333333", "hit": "#666666",
}


def load_step_profile(filepath, max_steps=0):
    try:
        df = pd.read_json(filepath, lines=True)
        if df.empty:
            return None
        if max_steps > 0:
            df = df[df["step"] <= max_steps]
        return df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def rolling_mean(arr, window):
    if window <= 1 or len(arr) <= window:
        return arr, np.arange(len(arr))
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    indices = np.arange(window - 1, len(arr))
    return smoothed, indices


def main():
    parser = argparse.ArgumentParser(
        description="Plot cache policy ablation (alloc/copy latency + hit rate)")
    parser.add_argument("--input", type=str, nargs="*", required=True,
                        help="Step profile files as label:path pairs")
    parser.add_argument("--model-name", type=str, default="",
                        help="Model name for plot title")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max steps to plot (0=all)")
    parser.add_argument("--window", type=int, default=50,
                        help="Rolling average window")
    parser.add_argument("--output", type=str, default="profile_plots")
    args = parser.parse_args()

    datasets = []
    for inp in args.input:
        if ":" in inp and not inp.startswith("/"):
            label, path = inp.split(":", 1)
        else:
            label = os.path.basename(inp).replace("_step.jsonl", "")
            path = inp
        df = load_step_profile(path, args.max_steps)
        if df is not None:
            datasets.append((label, df))
            print(f"  Loaded {label}: {len(df)} steps")
        else:
            print(f"  SKIP {label}: no data at {path}")

    if not datasets:
        print("No valid data to plot.")
        return

    os.makedirs(args.output, exist_ok=True)
    model = args.model_name
    safe_name = model.replace("/", "_").replace(" ", "_") if model else "model"
    window = args.window

    fig, (ax_lat, ax_hr) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: Stacked per-request allocation + copy latency ----
    legend_patches = []

    for label, df in datasets:
        colors = STYLE.get(label, FALLBACK)
        bs = df["batch_size"].clip(lower=1).values
        steps = df["step"].values

        alloc_pr = df["alloc_ms"].values / bs if "alloc_ms" in df.columns else np.zeros(len(df))
        copy_pr = df["copy_ms"].values / bs if "copy_ms" in df.columns else np.zeros(len(df))

        alloc_s, idx = rolling_mean(alloc_pr, window)
        copy_s, _ = rolling_mean(copy_pr, window)
        x = steps[idx]

        y0 = np.zeros_like(x, dtype=float)
        y1 = alloc_s
        y2 = y1 + copy_s

        ax_lat.fill_between(x, y0, y1, color=colors["alloc"], alpha=0.7, linewidth=0)
        ax_lat.fill_between(x, y1, y2, color=colors["copy"], alpha=0.7, linewidth=0)
        ax_lat.plot(x, y2, color=colors["edge"], linewidth=1.2, alpha=0.9)
        ax_lat.annotate(label, xy=(x[-1], y2[-1]), xytext=(5, 0),
                        textcoords="offset points", fontsize=8, fontweight="bold",
                        color=colors["edge"], va="center")

        legend_patches.append(mpatches.Patch(color=colors["alloc"], label=f"{label} alloc"))
        legend_patches.append(mpatches.Patch(color=colors["copy"], label=f"{label} copy"))

        mean_alloc = np.mean(alloc_pr)
        mean_copy = np.mean(copy_pr)
        print(f"  {label}: mean alloc/req={mean_alloc:.4f}ms, copy/req={mean_copy:.4f}ms, "
              f"bs={df['batch_size'].mean():.0f}")

    title = "Per-Request Allocation + Copy Latency"
    if model:
        title += f" — {model}"
    ax_lat.set_title(title)
    ax_lat.set_xlabel("Decode Step")
    ax_lat.set_ylabel("Per-Request Latency (ms)")
    ax_lat.legend(handles=legend_patches, fontsize=7, loc="upper left", ncol=2)
    ax_lat.grid(True, alpha=0.3)
    ax_lat.set_xlim(left=0)
    ax_lat.set_ylim(bottom=0)

    # ---- Right: Cache hit rate ----
    for label, df in datasets:
        if "hit_rate" not in df.columns:
            continue
        hr = df["hit_rate"].values
        if (hr <= 0).all():
            continue
        colors = STYLE.get(label, FALLBACK)
        steps = df["step"].values
        hr_pct = hr * 100

        hr_s, idx = rolling_mean(hr_pct, window)
        x = steps[idx]

        ax_hr.scatter(steps, hr_pct, alpha=0.03, s=1, color=colors["hit"],
                      rasterized=True)
        ax_hr.plot(x, hr_s, linewidth=1.5, color=colors["hit"], label=label)
        print(f"  {label}: mean hit rate = {np.mean(hr_pct):.1f}%")

    title = "Staging Cache Hit Rate"
    if model:
        title += f" — {model}"
    ax_hr.set_title(title)
    ax_hr.set_xlabel("Decode Step")
    ax_hr.set_ylabel("Cache Hit Rate (%)")
    ax_hr.set_ylim(-5, 105)
    ax_hr.set_xlim(left=0)
    ax_hr.legend(fontsize=9)
    ax_hr.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(args.output, f"policy_ablation_{safe_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\n  Saved {out_path}")

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
