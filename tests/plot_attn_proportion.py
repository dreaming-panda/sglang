#!/usr/bin/env python3
"""Plot per-request decode latency breakdown and cache hit rate.

Reads *_step.jsonl files (produced by model_runner profiling).

Generates two side-by-side plots:
  Left:  Stacked per-request latency (total_ms / batch_size) vs decode step.
         All backends shown as stacked areas with distinct colors.
  Right: Cache hit rate vs decode step (CPU-Vortex only).

Usage:
    python plot_attn_proportion.py \\
        --input "Baseline:path1_step.jsonl" \\
               "GPU-Vortex:path2_step.jsonl" \\
               "CPU-Vortex:path3_step.jsonl" \\
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

# Each backend: solid border color + fill colors for components
STYLE = {
    "Baseline": {
        "nonattn": "#D32F2F",    # strong red
        "attn":    "#1565C0",    # strong blue
        "edge":    "#B71C1C",
        "line":    "#D32F2F",
    },
    "GPU-Vortex": {
        "nonattn": "#FF8A65",    # orange-red
        "attn":    "#42A5F5",    # medium blue
        "edge":    "#E64A19",
        "line":    "#FF8A65",
    },
    "CPU-Vortex": {
        "nonattn": "#81C784",    # green
        "alloc":   "#FFD54F",    # yellow
        "copy":    "#CE93D8",    # purple
        "attn":    "#4DD0E1",    # cyan
        "edge":    "#388E3C",
        "line":    "#81C784",
    },
}

# Fallback
FALLBACK = {
    "nonattn": "#999999", "attn": "#4C72B0",
    "alloc": "#CCB974", "copy": "#8172B2",
    "edge": "#666666", "line": "#999999",
}

HIT_COLOR = "#2E7D32"


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


def plot_model(datasets, ax_lat, ax_hr, window=50, model_name=""):
    """Plot overlapping latency and hit rate for one model."""

    legend_patches = []

    # Sort by mean total per-request latency (descending) so tallest is drawn first
    sorted_datasets = sorted(datasets, key=lambda x: (
        x[1]["total_ms"] / x[1]["batch_size"].clip(lower=1)
    ).mean(), reverse=True)

    for label, df in sorted_datasets:
        bs = df["batch_size"].clip(lower=1).values
        steps = df["step"].values
        colors = STYLE.get(label, FALLBACK)

        has_alloc = "alloc_ms" in df.columns and df["alloc_ms"].sum() > 0

        if has_alloc:
            # CPU-Vortex: 4 components
            attn_pr = df["attn_ms"].values / bs
            alloc_pr = df["alloc_ms"].values / bs
            copy_pr = df["copy_ms"].values / bs
            if "other_ms" in df.columns:
                other_pr = np.clip(df["other_ms"].values / bs, 0, None)
            else:
                other_pr = np.clip(df["total_ms"].values / bs - attn_pr - alloc_pr - copy_pr, 0, None)

            other_s, idx = rolling_mean(other_pr, window)
            alloc_s, _ = rolling_mean(alloc_pr, window)
            copy_s, _ = rolling_mean(copy_pr, window)
            attn_s, _ = rolling_mean(attn_pr, window)
            x = steps[idx]

            y0 = np.zeros_like(x, dtype=float)
            y1 = other_s
            y2 = y1 + alloc_s
            y3 = y2 + copy_s
            y4 = y3 + attn_s

            ax_lat.fill_between(x, y0, y1, color=colors["nonattn"], alpha=0.8, linewidth=0)
            ax_lat.fill_between(x, y1, y2, color=colors["alloc"], alpha=0.8, linewidth=0)
            ax_lat.fill_between(x, y2, y3, color=colors["copy"], alpha=0.8, linewidth=0)
            ax_lat.fill_between(x, y3, y4, color=colors["attn"], alpha=0.8, linewidth=0)
            # Top edge line + label
            ax_lat.plot(x, y4, color=colors["edge"], linewidth=1.2, alpha=0.9)
            ax_lat.annotate(label, xy=(x[-1], y4[-1]), xytext=(5, 0),
                            textcoords="offset points", fontsize=8, fontweight="bold",
                            color=colors["edge"], va="center")

            legend_patches.append(mpatches.Patch(color=colors["nonattn"], label=f"{label} non-attn"))
            legend_patches.append(mpatches.Patch(color=colors["alloc"], label=f"{label} alloc"))
            legend_patches.append(mpatches.Patch(color=colors["copy"], label=f"{label} copy"))
            legend_patches.append(mpatches.Patch(color=colors["attn"], label=f"{label} attn"))

            total_mean = np.mean(attn_pr + alloc_pr + copy_pr + other_pr)
            print(f"  {label}: mean per-req = {total_mean:.2f}ms, bs = {df['batch_size'].mean():.0f}")

        else:
            # Baseline / GPU-Vortex: 2 components
            attn_pr = df["attn_ms"].values / bs
            if "other_ms" in df.columns:
                other_pr = np.clip(df["other_ms"].values / bs, 0, None)
            elif "nonattn_ms" in df.columns:
                other_pr = np.clip(df["nonattn_ms"].values / bs, 0, None)
            else:
                other_pr = np.clip(df["total_ms"].values / bs - attn_pr, 0, None)

            other_s, idx = rolling_mean(other_pr, window)
            attn_s, _ = rolling_mean(attn_pr, window)
            x = steps[idx]

            y0 = np.zeros_like(x, dtype=float)
            y1 = other_s
            y2 = y1 + attn_s

            ax_lat.fill_between(x, y0, y1, color=colors["nonattn"], alpha=0.8, linewidth=0)
            ax_lat.fill_between(x, y1, y2, color=colors["attn"], alpha=0.8, linewidth=0)
            ax_lat.plot(x, y2, color=colors["edge"], linewidth=1.2, alpha=0.9)
            ax_lat.annotate(label, xy=(x[-1], y2[-1]), xytext=(5, 0),
                            textcoords="offset points", fontsize=8, fontweight="bold",
                            color=colors["edge"], va="center")

            legend_patches.append(mpatches.Patch(color=colors["nonattn"], label=f"{label} non-attn"))
            legend_patches.append(mpatches.Patch(color=colors["attn"], label=f"{label} attn"))

            total_mean = np.mean(attn_pr + other_pr)
            print(f"  {label}: mean per-req = {total_mean:.2f}ms, bs = {df['batch_size'].mean():.0f}")

    ax_lat.set_xlabel("Decode Step")
    ax_lat.set_ylabel("Per-Request Latency (ms)")
    title = "Per-Request Decode Latency"
    if model_name:
        title += f" — {model_name}"
    ax_lat.set_title(title)
    ax_lat.legend(handles=legend_patches, fontsize=7, loc="upper left", ncol=2)
    ax_lat.grid(True, alpha=0.3)
    ax_lat.set_xlim(left=0)
    ax_lat.set_ylim(bottom=0)

    # ---- Right: Cache hit rate ----
    if ax_hr is not None:
        plotted = False
        for label, df in datasets:
            if "hit_rate" not in df.columns:
                continue
            hr = df["hit_rate"].values
            if (hr <= 0).all():
                continue
            steps = df["step"].values
            hr_pct = hr * 100
            hr_s, idx = rolling_mean(hr_pct, window)
            x = steps[idx]

            ax_hr.scatter(steps, hr_pct, alpha=0.03, s=1, color=HIT_COLOR,
                          rasterized=True)
            ax_hr.plot(x, hr_s, linewidth=1.5, color=HIT_COLOR, label=label)
            plotted = True
            print(f"  {label}: mean hit rate = {np.mean(hr_pct):.1f}%")

        ax_hr.set_xlabel("Decode Step")
        ax_hr.set_ylabel("Cache Hit Rate (%)")
        title = "Staging Cache Hit Rate"
        if model_name:
            title += f" — {model_name}"
        ax_hr.set_title(title)
        ax_hr.set_ylim(-5, 105)
        ax_hr.set_xlim(left=0)
        if plotted:
            ax_hr.legend(fontsize=9)
        ax_hr.grid(True, alpha=0.3)


def plot_nonattn_absolute(datasets, ax, window=50, model_name=""):
    """Plot absolute non-attention latency per step for each backend.
    Legend shows method name and average batch size."""

    LINE_COLORS = {
        "Baseline":    "#D32F2F",
        "GPU-Vortex":  "#E64A19",
        "CPU-Vortex":  "#388E3C",
    }
    FALLBACK_LINE = "#666666"

    for label, df in datasets:
        steps = df["step"].values
        color = LINE_COLORS.get(label, FALLBACK_LINE)

        # Get absolute non-attention ms per step
        if "other_ms" in df.columns:
            nonattn = df["other_ms"].values
        elif "nonattn_ms" in df.columns:
            nonattn = df["nonattn_ms"].values
        else:
            nonattn = df["total_ms"].values - df["attn_ms"].values

        avg_bs = df["batch_size"].mean()
        nonattn_s, idx = rolling_mean(nonattn, window)
        x = steps[idx]

        ax.plot(x, nonattn_s, linewidth=1.5, color=color,
                label=f"{label} (bs={avg_bs:.0f})")
        ax.annotate(label, xy=(x[-1], nonattn_s[-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, fontweight="bold",
                    color=color, va="center")

        print(f"  {label}: mean non-attn = {nonattn.mean():.2f}ms, bs = {avg_bs:.0f}")

    title = "Non-Attention Latency per Step"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Non-Attention Latency (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-request decode latency and cache hit rate")
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

    has_hit = any(
        "hit_rate" in df.columns and (df["hit_rate"] > 0).any()
        for _, df in datasets
    )

    # ---- Main combined figure: 2 panels ----
    if has_hit:
        fig, (ax_lat, ax_hr) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax_lat = plt.subplots(1, 1, figsize=(8, 5))
        ax_hr = None

    print(f"\nPlotting {model or 'model'}:")
    plot_model(datasets, ax_lat, ax_hr, args.window, model)

    plt.tight_layout()
    out_path = os.path.join(args.output, f"latency_hitrate_{safe_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")

    # ---- Also save individual plots ----
    # Non-attention only
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    plot_nonattn_absolute(datasets, ax2, args.window, model)
    plt.tight_layout()
    out2 = os.path.join(args.output, f"nonattn_{safe_name}.png")
    fig2.savefig(out2)
    plt.close(fig2)
    print(f"  Saved {out2}")

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
