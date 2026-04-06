#!/usr/bin/env python3
"""Plot AIME benchmark results from JSONL output files."""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

DATA_DIR = "results"
PLOT_DIR = "plots"

# AIME 2024 ground truth answers (30 problems, HuggingFaceH4/aime_2024)
AIME_ANSWERS = [
    "204", "113", "371", "385", "110", "104", "721", "025", "809", "116",
    "104", "294", "540", "197", "480", "073", "468", "601", "023", "321",
    "211", "315", "236", "045", "033", "080", "055", "699", "127", "902",
]

# Match aime_test.sh configs
MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
GPU_FRACS = ["0.9"]
MAX_TOKENS_LIST = ["16384"]

ALLOC_KERNELS = ["lru_block_global", "lru_global", "lru_block"]
KV_DTYPES = ["int8", "fp8_e4m3", "fp8_e5m2"]

# (subdir_path, filename, label)
# subdir_path is relative to results/model/AIME24/gpu_X/max_tokens_Y/
BACKEND_CONFIGS = [
    # Plain baseline
    ("", "baseline.jsonl", "baseline"),
    # GPU vortex: bf16 + quantized
    ("", "flashinfer.jsonl", "vtx (bf16)"),
]
for kv_dtype in KV_DTYPES:
    BACKEND_CONFIGS.append((kv_dtype, "flashinfer.jsonl", f"vtx ({kv_dtype})"))
# CPU offload: each alloc kernel x (bf16 + quantized)
for alloc_k in ALLOC_KERNELS:
    BACKEND_CONFIGS.append((alloc_k, "cpu_vtx_flashinfer.jsonl", f"cpu ({alloc_k})"))
    for kv_dtype in KV_DTYPES:
        BACKEND_CONFIGS.append((f"{alloc_k}/{kv_dtype}", "cpu_vtx_flashinfer.jsonl", f"cpu ({alloc_k}, {kv_dtype})"))

# Auto-generate enough colors
import matplotlib.cm as cm
_n = len(BACKEND_CONFIGS)
COLORS = [cm.tab20(i / _n) for i in range(_n)]


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
    """Scan results/ directory and load all benchmark results.
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
    """Plot 1: Accuracy barplot per model."""
    for model in MODELS:
        for gpu_frac in GPU_FRACS:
            for max_tok in MAX_TOKENS_LIST:
                labels = []
                accuracies = []
                colors = []
                for i, (_, _, label) in enumerate(BACKEND_CONFIGS):
                    key = (model, gpu_frac, max_tok, label)
                    if key in data:
                        records, _ = data[key]
                        acc = compute_accuracy(records)
                        labels.append(label)
                        accuracies.append(acc * 100)
                        colors.append(COLORS[i])

                if not labels:
                    continue

                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
                x = np.arange(len(labels))
                bars = ax.bar(x, accuracies, color=colors, width=0.6)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"{model} — Accuracy (gpu={gpu_frac}, max_tokens={max_tok})")
                ax.set_ylim(0, max(accuracies) * 1.2 if accuracies else 100)

                for bar, val in zip(bars, accuracies):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

                plt.tight_layout()
                model_short = model.split("/")[-1]
                plt.savefig(os.path.join(PLOT_DIR, f"accuracy_{model_short}_gpu{gpu_frac}_tok{max_tok}.png"), dpi=150)
                plt.close()
                print(f"  Saved accuracy_{model_short}_gpu{gpu_frac}_tok{max_tok}.png")


def plot_throughput(data):
    """Plot 2: Throughput comparison per model."""
    for model in MODELS:
        for gpu_frac in GPU_FRACS:
            for max_tok in MAX_TOKENS_LIST:
                labels = []
                throughputs = []
                colors = []
                for i, (_, _, label) in enumerate(BACKEND_CONFIGS):
                    key = (model, gpu_frac, max_tok, label)
                    if key in data:
                        _, meta = data[key]
                        labels.append(label)
                        throughputs.append(meta["throughput"])
                        colors.append(COLORS[i])

                if not labels:
                    continue

                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
                x = np.arange(len(labels))
                bars = ax.bar(x, throughputs, color=colors, width=0.6)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("Throughput (tokens/s)")
                ax.set_title(f"{model} — Throughput (gpu={gpu_frac}, max_tokens={max_tok})")

                for bar, val in zip(bars, throughputs):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.0f}", ha="center", va="bottom", fontsize=8)

                plt.tight_layout()
                model_short = model.split("/")[-1]
                plt.savefig(os.path.join(PLOT_DIR, f"throughput_{model_short}_gpu{gpu_frac}_tok{max_tok}.png"), dpi=150)
                plt.close()
                print(f"  Saved throughput_{model_short}_gpu{gpu_frac}_tok{max_tok}.png")


def plot_avg_tokens(data):
    """Plot 3: Average generated tokens per model."""
    for model in MODELS:
        for gpu_frac in GPU_FRACS:
            for max_tok in MAX_TOKENS_LIST:
                labels = []
                avg_tokens = []
                colors = []
                for i, (_, _, label) in enumerate(BACKEND_CONFIGS):
                    key = (model, gpu_frac, max_tok, label)
                    if key in data:
                        records, _ = data[key]
                        avg = compute_avg_completion_tokens(records)
                        labels.append(label)
                        avg_tokens.append(avg)
                        colors.append(COLORS[i])

                if not labels:
                    continue

                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
                x = np.arange(len(labels))
                bars = ax.bar(x, avg_tokens, color=colors, width=0.6)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("Avg Completion Tokens")
                ax.set_title(f"{model} — Avg Generated Tokens (gpu={gpu_frac}, max_tokens={max_tok})")

                for bar, val in zip(bars, avg_tokens):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                            f"{val:.0f}", ha="center", va="bottom", fontsize=8)

                plt.tight_layout()
                model_short = model.split("/")[-1]
                plt.savefig(os.path.join(PLOT_DIR, f"avg_tokens_{model_short}_gpu{gpu_frac}_tok{max_tok}.png"), dpi=150)
                plt.close()
                print(f"  Saved avg_tokens_{model_short}_gpu{gpu_frac}_tok{max_tok}.png")


PAPER_CONFIGS = [
    ("", "baseline.jsonl", "Dense"),
    ("", "flashinfer.jsonl", "GPU-Vortex"),
    ("lru_block_global", "cpu_vtx_flashinfer.jsonl", "CPU-Vortex"),
    ("lru_block_global/int8", "cpu_vtx_flashinfer.jsonl", "CPU-Vortex (int8)"),
]

PAPER_COLORS = ["#888888", "#4C72B0", "#55A868", "#C44E52"]


def load_paper_data():
    """Load only the key configs for paper-quality figures."""
    data = {}
    for model in MODELS:
        for gpu_frac in GPU_FRACS:
            for max_tok in MAX_TOKENS_LIST:
                base_dir = os.path.join(DATA_DIR, model, "AIME24", f"gpu_{gpu_frac}", f"max_tokens_{max_tok}")
                if not os.path.isdir(base_dir):
                    continue
                for subdir, filename, label in PAPER_CONFIGS:
                    filepath = os.path.join(base_dir, subdir, filename) if subdir else os.path.join(base_dir, filename)
                    if os.path.exists(filepath):
                        records, metadata = load_jsonl(filepath)
                        if records and metadata:
                            data[(model, gpu_frac, max_tok, label)] = (records, metadata)
    return data


def plot_paper_throughput(data):
    """Paper-quality throughput comparison: grouped bars across models."""
    gpu_frac = GPU_FRACS[0]
    max_tok = MAX_TOKENS_LIST[0]

    model_labels = []
    for model in MODELS:
        key = (model, gpu_frac, max_tok, "Dense")
        if key in data:
            model_labels.append(model)

    if not model_labels:
        print("  No data for paper throughput plot.")
        return

    config_labels = [label for _, _, label in PAPER_CONFIGS]
    n_models = len(model_labels)
    n_configs = len(config_labels)
    x = np.arange(n_models)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(max(8, n_models * 3), 5))

    for i, (_, _, label) in enumerate(PAPER_CONFIGS):
        vals = []
        for model in model_labels:
            key = (model, gpu_frac, max_tok, label)
            if key in data:
                _, meta = data[key]
                vals.append(meta["throughput"])
            else:
                vals.append(0)
        offset = (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=PAPER_COLORS[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in model_labels], fontsize=10)
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput Comparison — AIME 2024")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "throughput_paper.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")


def plot_paper_accuracy(data):
    """Paper-quality accuracy comparison: grouped bars across models."""
    gpu_frac = GPU_FRACS[0]
    max_tok = MAX_TOKENS_LIST[0]

    model_labels = []
    for model in MODELS:
        key = (model, gpu_frac, max_tok, "Dense")
        if key in data:
            model_labels.append(model)

    if not model_labels:
        print("  No data for paper accuracy plot.")
        return

    n_models = len(model_labels)
    n_configs = len(PAPER_CONFIGS)
    x = np.arange(n_models)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(max(8, n_models * 3), 5))

    for i, (_, _, label) in enumerate(PAPER_CONFIGS):
        vals = []
        for model in model_labels:
            key = (model, gpu_frac, max_tok, label)
            if key in data:
                records, _ = data[key]
                vals.append(compute_accuracy(records) * 100)
            else:
                vals.append(0)
        offset = (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=PAPER_COLORS[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in model_labels], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison — AIME 2024")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "accuracy_paper.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot AIME benchmark results")
    parser.add_argument("--paper-mode", action="store_true",
                        help="Generate paper-quality figures with only key configs")
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    if args.paper_mode:
        print("Loading benchmark data (paper mode)...")
        data = load_paper_data()
        print(f"Loaded {len(data)} result files.")
        if not data:
            print("No data found. Check that results/ directory has benchmark outputs.")
            return
        print("\nPaper: Throughput comparison...")
        plot_paper_throughput(data)
        print("\nPaper: Accuracy comparison...")
        plot_paper_accuracy(data)
    else:
        print("Loading benchmark data...")
        data = load_all_data()
        print(f"Loaded {len(data)} result files.")
        if not data:
            print("No data found. Check that results/ directory has benchmark outputs.")
            return
        print("\nPlot 1: Accuracy barplots...")
        plot_accuracy_barplot(data)
        print("\nPlot 2: Throughput comparison...")
        plot_throughput(data)
        print("\nPlot 3: Average generated tokens...")
        plot_avg_tokens(data)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
