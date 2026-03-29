#!/usr/bin/env python3
"""Plot AIME benchmark results from JSONL output files."""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

DATA_DIR = "DATA"
PLOT_DIR = "plots"

# AIME 2024 ground truth answers (30 problems, HuggingFaceH4/aime_2024)
AIME_ANSWERS = [
    "204", "113", "371", "385", "110", "104", "721", "025", "809", "116",
    "104", "294", "540", "197", "480", "073", "468", "601", "023", "321",
    "211", "315", "236", "045", "033", "080", "055", "699", "127", "902",
]

MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
GPU_FRACS = ["0.5", "0.7", "0.9"]
MAX_TOKENS_LIST = ["4096", "8192", "16384"]

# Backend configs: (subdir_prefix, filename, label)
# Directory naming: {alloc_kernel}/cpu_vtx_flashinfer.jsonl
ALLOC_KERNELS = ["lru_block", "lru_global", "lru_block_global"]

BACKEND_CONFIGS = [
    ("", "baseline.jsonl", "baseline"),
    ("", "flashinfer.jsonl", "flashinfer"),
]
for alloc_k in ALLOC_KERNELS:
    BACKEND_CONFIGS.append((alloc_k, "cpu_vtx_flashinfer.jsonl", f"cpu_vtx ({alloc_k})"))

COLORS = [
    "#4C72B0", "#55A868",  # baseline, flashinfer
    "#C44E52",             # lru_block
    "#CCB974",             # lru_global
    "#8172B2",             # lru_block_global
]


GOLD_CONFIG = [ExprExtractionConfig()]
PRED_CONFIG = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]


def load_jsonl(filepath):
    """Load JSONL file. Returns (records, metadata)."""
    records = []
    metadata = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "throughput" in d:
                metadata = d
            else:
                records.append(d)
    return records, metadata


def compute_accuracy(records):
    """Compute raw accuracy using math_verify for robust answer comparison."""
    n_problems = len(AIME_ANSWERS)
    correct = 0
    total = len(records)
    for i, rec in enumerate(records):
        gt = AIME_ANSWERS[i % n_problems]
        try:
            parsed_gold = parse(gt, extraction_config=GOLD_CONFIG)
            parsed_pred = parse(rec["text"], extraction_config=PRED_CONFIG)
            if verify(parsed_gold, parsed_pred):
                correct += 1
        except Exception:
            pass
    return correct / total if total > 0 else 0.0


def compute_avg_completion_tokens(records):
    """Compute average completion tokens across records."""
    tokens = [r["meta_info"]["completion_tokens"] for r in records]
    return np.mean(tokens) if tokens else 0.0


def load_all_data():
    """Scan DATA/ directory and load all benchmark results.
    Returns dict: (model, gpu_frac, max_tokens, backend_label) -> (records, metadata)
    """
    data = {}
    for model in MODELS:
        for gpu_frac in GPU_FRACS:
            for max_tok in MAX_TOKENS_LIST:
                base_dir = os.path.join(DATA_DIR, model, "AIME24", f"gpu_{gpu_frac}", f"max_tokens_{max_tok}")
                if not os.path.isdir(base_dir):
                    continue
                for subdir, filename, label in BACKEND_CONFIGS:
                    if subdir:
                        filepath = os.path.join(base_dir, subdir, filename)
                    else:
                        filepath = os.path.join(base_dir, filename)
                    if os.path.exists(filepath):
                        records, metadata = load_jsonl(filepath)
                        if records and metadata:
                            data[(model, gpu_frac, max_tok, label)] = (records, metadata)
    return data


def plot_accuracy_barplot(data):
    """Plot 1: Accuracy barplot per model (gpu_0.9, max_tokens=16384)."""
    for model in MODELS:
        labels = []
        accuracies = []
        colors = []
        for i, (_, _, label) in enumerate(BACKEND_CONFIGS):
            key = (model, "0.9", "16384", label)
            if key in data:
                records, _ = data[key]
                acc = compute_accuracy(records)
                labels.append(label)
                accuracies.append(acc * 100)
                colors.append(COLORS[i])

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, accuracies, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model} — Accuracy (gpu=0.9, max_tokens=16384)")
        ax.set_ylim(0, max(accuracies) * 1.2 if accuracies else 100)

        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        model_short = model.split("/")[-1]
        plt.savefig(os.path.join(PLOT_DIR, f"accuracy_{model_short}.png"), dpi=150)
        plt.close()
        print(f"  Saved accuracy_{model_short}.png")


def plot_throughput(data):
    """Plot 2: Throughput comparison — one figure per (model, gpu_frac)."""
    for model in MODELS:
        for gpu_frac in GPU_FRACS:
            backend_labels = [cfg[2] for cfg in BACKEND_CONFIGS]
            token_groups = MAX_TOKENS_LIST  # 4096, 8192, 16384

            # Build throughput matrix: [n_tokens x n_backends]
            throughputs = []
            valid_tokens = []
            for max_tok in token_groups:
                row = []
                has_data = False
                for label in backend_labels:
                    key = (model, gpu_frac, max_tok, label)
                    if key in data:
                        _, meta = data[key]
                        row.append(meta["throughput"])
                        has_data = True
                    else:
                        row.append(0)
                if has_data:
                    throughputs.append(row)
                    valid_tokens.append(max_tok)

            if not valid_tokens:
                continue

            n_groups = len(valid_tokens)
            n_bars = len(backend_labels)
            fig, ax = plt.subplots(figsize=(10, 5))

            bar_width = 0.15
            group_positions = np.arange(n_groups)

            for i, label in enumerate(backend_labels):
                vals = [throughputs[g][i] for g in range(n_groups)]
                offset = (i - n_bars / 2 + 0.5) * bar_width
                bars = ax.bar(group_positions + offset, vals, bar_width,
                              label=label, color=COLORS[i])
                for bar, val in zip(bars, vals):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f"{val:.0f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(group_positions)
            ax.set_xticklabels([f"{t}" for t in valid_tokens])
            ax.set_xlabel("Max Tokens")
            ax.set_ylabel("Throughput (tokens/s)")
            ax.set_title(f"{model} — Throughput (gpu={gpu_frac})")
            ax.legend(fontsize=8)

            plt.tight_layout()
            model_short = model.split("/")[-1]
            plt.savefig(os.path.join(PLOT_DIR, f"throughput_{model_short}_gpu{gpu_frac}.png"), dpi=150)
            plt.close()
            print(f"  Saved throughput_{model_short}_gpu{gpu_frac}.png")


def plot_avg_tokens(data):
    """Plot 3: Average generated tokens per model (gpu_0.9, max_tokens=16384)."""
    for model in MODELS:
        labels = []
        avg_tokens = []
        colors = []
        for i, (_, _, label) in enumerate(BACKEND_CONFIGS):
            key = (model, "0.9", "16384", label)
            if key in data:
                records, _ = data[key]
                avg = compute_avg_completion_tokens(records)
                labels.append(label)
                avg_tokens.append(avg)
                colors.append(COLORS[i])

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, avg_tokens, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Avg Completion Tokens")
        ax.set_title(f"{model} — Avg Generated Tokens (gpu=0.9, max_tokens=16384)")

        for bar, val in zip(bars, avg_tokens):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        model_short = model.split("/")[-1]
        plt.savefig(os.path.join(PLOT_DIR, f"avg_tokens_{model_short}.png"), dpi=150)
        plt.close()
        print(f"  Saved avg_tokens_{model_short}.png")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Loading benchmark data...")
    data = load_all_data()
    print(f"Loaded {len(data)} result files.")

    print("\nPlot 1: Accuracy barplots...")
    plot_accuracy_barplot(data)

    print("\nPlot 2: Throughput comparison...")
    plot_throughput(data)

    print("\nPlot 3: Average generated tokens...")
    plot_avg_tokens(data)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
