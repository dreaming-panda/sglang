#!/usr/bin/env python3
"""Plot top-k ablation: accuracy and throughput bar plots across top-k values.

Reads JSONL output files from aime.py for each (model, topk) configuration.

Usage:
    python plot_topk_ablation.py \
        --input "Qwen3-8B,10:path1.jsonl" "Qwen3-8B,20:path2.jsonl" ... \
        --output thesis/figures
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# AIME 2024 answers (30 problems)
AIME_ANSWERS = [
    937, 27, 536, 116, 104, 294, 540, 34, 371, 6,
    539, 52, 450, 913, 63, 116, 47, 504, 174, 145,
    96, 119, 401, 89, 10, 20, 51, 8, 58, 10,
]


def load_results(filepath):
    """Load aime.py JSONL output. Returns (accuracy, throughput, total_tokens)."""
    if not os.path.exists(filepath):
        return None, None, None

    results = []
    meta = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "e2e_time" in obj:
                meta = obj
            else:
                results.append(obj)

    if not results or meta is None:
        return None, None, None

    # Count correct answers
    correct = 0
    total = min(len(results), len(AIME_ANSWERS))
    for i in range(total):
        text = results[i].get("text", "")
        # Extract boxed answer
        match = re.findall(r'\\boxed\{(\d+)\}', text)
        if match:
            try:
                answer = int(match[-1])
                if answer == AIME_ANSWERS[i % 30]:
                    correct += 1
            except ValueError:
                pass

    accuracy = correct / total * 100 if total > 0 else 0
    throughput = meta.get("throughput", 0)
    return accuracy, throughput, total


def main():
    parser = argparse.ArgumentParser(description="Plot top-k ablation")
    parser.add_argument("--results-dir", type=str,
                        default="results",
                        help="Base results directory")
    parser.add_argument("--output", type=str, default="thesis/figures")
    parser.add_argument("--mem-frac", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    models = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
    model_labels = ["Qwen3-4B", "Qwen3-8B"]
    topk_values = [10, 20, 30, 50]

    COLORS = {
        "Qwen3-4B": "#1565C0",
        "Qwen3-8B": "#E65100",
    }

    # Collect data
    data = {}  # (model_label, topk) -> (accuracy, throughput)
    for model, label in zip(models, model_labels):
        for topk in topk_values:
            path = os.path.join(
                args.results_dir, model, "AIME24",
                f"gpu_{args.mem_frac}", f"max_tokens_{args.max_tokens}",
                "topk_ablation", f"topk_{topk}",
                "cpu_vtx_flashinfer.jsonl"
            )
            acc, tput, total = load_results(path)
            if acc is not None:
                data[(label, topk)] = (acc, tput)
                print(f"  {label} topk={topk}: accuracy={acc:.1f}% ({int(acc*total/100)}/{total}), "
                      f"throughput={tput:.1f} tok/s")
            else:
                print(f"  {label} topk={topk}: NOT FOUND at {path}")

    if not data:
        print("No data found.")
        return

    # Plot
    fig, (ax_acc, ax_tput) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(topk_values))
    width = 0.35
    offsets = [-0.5, 0.5]

    for i, label in enumerate(model_labels):
        accs = [data.get((label, k), (0, 0))[0] for k in topk_values]
        tputs = [data.get((label, k), (0, 0))[1] for k in topk_values]
        color = COLORS[label]

        ax_acc.bar(x + offsets[i] * width, accs, width * 0.9,
                   label=label, color=color, edgecolor="white", linewidth=0.5)
        ax_tput.bar(x + offsets[i] * width, tputs, width * 0.9,
                    label=label, color=color, edgecolor="white", linewidth=0.5)

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([str(k) for k in topk_values])
    ax_acc.set_xlabel("Top-K Pages")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("AIME 2024 Accuracy vs Top-K")
    ax_acc.legend()
    ax_acc.grid(True, axis="y", alpha=0.3)
    ax_acc.set_ylim(bottom=0)

    ax_tput.set_xticks(x)
    ax_tput.set_xticklabels([str(k) for k in topk_values])
    ax_tput.set_xlabel("Top-K Pages")
    ax_tput.set_ylabel("Throughput (tokens/s)")
    ax_tput.set_title("Throughput vs Top-K")
    ax_tput.legend()
    ax_tput.grid(True, axis="y", alpha=0.3)
    ax_tput.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(args.output, "topk_ablation.png")
    plt.savefig(out)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
