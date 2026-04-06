#!/bin/bash
# Profile attention vs non-attention proportion for baseline, GPU-Vortex, and CPU-Vortex.
# Runs all three in parallel on separate GPUs, then plots.
# Usage: bash run_attn_profile.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate qilong_vortex

cd ~/qilong/sglang/tests

MODEL="Qwen/Qwen3-8B"
MEM_FRAC=0.9
MAX_TOKENS=8192
LOG_INTERVAL=100
GPU_BASELINE=8
GPU_VORTEX=9
GPU_CPU_VTX=7

OUT_DIR="results/${MODEL}/AIME24/gpu_${MEM_FRAC}/max_tokens_${MAX_TOKENS}"
mkdir -p "$OUT_DIR"

# --- Run all three in parallel on separate GPUs ---
echo "========================================"
echo "Starting Baseline (GPU $GPU_BASELINE), GPU-Vortex (GPU $GPU_VORTEX), CPU-Vortex (GPU $GPU_CPU_VTX)"
echo "========================================"

# CUDA_VISIBLE_DEVICES=$GPU_BASELINE \
# VORTEX_PROFILE_PATH="${OUT_DIR}/profile_data_baseline.jsonl" \
VORTEX_PROFILE_LOG_INTERVAL=$LOG_INTERVAL \
python aime.py \
    --model-name "$MODEL" \
    --attention-backend baseline \
    --mem-fraction-static $MEM_FRAC \
    --max-new-tokens $MAX_TOKENS \
    --profile \
    --disable-cuda-graph \
    2>&1 | tee "${OUT_DIR}/profile_baseline.log" &
PID_BASELINE=$!

CUDA_VISIBLE_DEVICES=$GPU_VORTEX \
VORTEX_PROFILE_PATH="${OUT_DIR}/profile_data_gpu_vortex.jsonl" \
VORTEX_PROFILE_LOG_INTERVAL=$LOG_INTERVAL \
python aime.py \
    --model-name "$MODEL" \
    --attention-backend flashinfer \
    --mem-fraction-static $MEM_FRAC \
    --max-new-tokens $MAX_TOKENS \
    --profile \
    --disable-cuda-graph \
    2>&1 | tee "${OUT_DIR}/profile_gpu_vortex.log" &
PID_VORTEX=$!

CUDA_VISIBLE_DEVICES=$GPU_CPU_VTX \
VORTEX_PROFILE_PATH="${OUT_DIR}/profile_data_cpu_vortex.jsonl" \
VORTEX_PROFILE_LOG_INTERVAL=$LOG_INTERVAL \
python aime.py \
    --model-name "$MODEL" \
    --attention-backend cpu_vtx_flashinfer \
    --mem-fraction-static $MEM_FRAC \
    --max-new-tokens $MAX_TOKENS \
    --profile \
    --disable-cuda-graph \
    2>&1 | tee "${OUT_DIR}/profile_cpu_vortex.log" &
PID_CPU_VTX=$!

echo "Waiting for all three (baseline=$PID_BASELINE, gpu_vtx=$PID_VORTEX, cpu_vtx=$PID_CPU_VTX)..."
wait $PID_BASELINE
echo "Baseline done."
wait $PID_VORTEX
echo "GPU-Vortex done."
wait $PID_CPU_VTX
echo "CPU-Vortex done."

# --- Plot ---
echo "========================================"
echo "Generating plots"
echo "========================================"
python plot_attn_proportion.py \
    --input \
    "Baseline:${OUT_DIR}/profile_data_baseline_step.jsonl" \
    "GPU-Vortex:${OUT_DIR}/profile_data_gpu_vortex_step.jsonl" \
    "CPU-Vortex:${OUT_DIR}/profile_data_cpu_vortex_step.jsonl" \
    --model-name "$(basename $MODEL)" \
    --output profile_plots \
    --window 50 \
    --max-steps 30000

echo "Done! Plots saved to profile_plots/"
