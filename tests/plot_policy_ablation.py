#!/usr/bin/env python3
"""Plot cache policy ablation: copy latency and hit rate over decode steps.

Reads *_step.jsonl files produced by model_runner profiling.

Generates a two-panel plot per configuration:
  Left:  Per-request copy latency (ms) vs decode step, one line per policy.
  Right: Cache hit rate (%) vs decode step, one line per policy.

The correlation is visible: when hit rate drops, copy latency rises.

Usage:
    python plot_policy_ablation.py \\
        --input "LRU:path1_step.jsonl" \\
               "LFU:path2_step.jsonl" \\
               "Random:path3_step.jsonl" \\
        --model-name "Qwen3-8B" --max-steps 8192
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

STYLE = {
    "LRU": {"color": "#1565C0", "label": "LRU"},
    "LFU": {"color": "#E65100", "label": "LFU"},
    "Random": {"color": "#2E7D32", "label": "Random"},
}

FALLBACK = {"color": "#888888", "label": "Unknown"}


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
        description="Plot cache policy ablation (copy latency + hit rate)")
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
    safe_name = model.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "") if model else "model"
    window = args.window

    fig, (ax_copy, ax_hr) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: Per-request copy latency ----
    for label, df in datasets:
        style = STYLE.get(label, FALLBACK)
        bs = df["batch_size"].clip(lower=1).values
        steps = df["step"].values

        copy_pr = df["copy_ms"].values / bs if "copy_ms" in df.columns else np.zeros(len(df))
        copy_s, idx = rolling_mean(copy_pr, window)
        x = steps[idx]

        ax_copy.plot(x, copy_s, color=style["color"], linewidth=1.5, label=style["label"])

        mean_copy = np.mean(copy_pr)
        print(f"  {label}: mean copy/req={mean_copy:.4f}ms, bs={df['batch_size'].mean():.0f}")

    title = "Per-Request Copy Latency"
    if model:
        title += f" — {model}"
    ax_copy.set_title(title)
    ax_copy.set_xlabel("Decode Step")
    ax_copy.set_ylabel("Copy Latency / Request (ms)")
    ax_copy.legend()
    ax_copy.grid(True, alpha=0.3)
    ax_copy.set_xlim(left=0)
    ax_copy.set_ylim(bottom=0)

    # ---- Right: Cache hit rate ----
    for label, df in datasets:
        if "hit_rate" not in df.columns:
            continue
        hr = df["hit_rate"].values
        if (hr <= 0).all():
            continue
        style = STYLE.get(label, FALLBACK)
        steps = df["step"].values
        hr_pct = hr * 100

        hr_s, idx = rolling_mean(hr_pct, window)
        x = steps[idx]

        ax_hr.scatter(steps, hr_pct, alpha=0.03, s=1, color=style["color"],
                      rasterized=True)
        ax_hr.plot(x, hr_s, linewidth=1.5, color=style["color"], label=style["label"])
        print(f"  {label}: mean hit rate = {np.mean(hr_pct):.1f}%")

    title = "Staging Cache Hit Rate"
    if model:
        title += f" — {model}"
    ax_hr.set_title(title)
    ax_hr.set_xlabel("Decode Step")
    ax_hr.set_ylabel("Cache Hit Rate (%)")
    ax_hr.set_ylim(-5, 105)
    ax_hr.set_xlim(left=0)
    ax_hr.legend()
    ax_hr.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(args.output, f"policy_ablation_{safe_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\n  Saved {out_path}")

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
