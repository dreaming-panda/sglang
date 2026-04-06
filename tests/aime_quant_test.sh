#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qilong_vortex

cd ~/qilong/sglang/tests

MODEL="Qwen/Qwen3-8B"
MEM_FRAC=0.9
MAX_TOKENS=1024
KV_DTYPES=("int8" "fp8_e4m3")

# =============================================
# CPU VTX (cpu_vtx_flashinfer) with quantization
# =============================================
for kv_dtype in "${KV_DTYPES[@]}"; do
    echo "========================================"
    echo "CPU VTX: model=$MODEL, kv_dtype=$kv_dtype, alloc=lru_block_global"
    echo "========================================"

    LOG_DIR="DATA/${MODEL}/AIME24/cpu_${MEM_FRAC}/max_tokens_${MAX_TOKENS}/lru_block_global/${kv_dtype}"
    mkdir -p "$LOG_DIR"

    python aime.py \
        --model-name "$MODEL" \
        --attention-backend cpu_vtx_flashinfer \
        --mem-fraction-static "$MEM_FRAC" \
        --max-new-tokens "$MAX_TOKENS" \
        --kv-cache-dtype "$kv_dtype" \
        --alloc-kernel lru_block_global \
        2>&1 | tee "${LOG_DIR}/run.log"

    echo "Completed. Sleeping 5s..."
    sleep 5
done

# =============================================
# GPU VTX (flashinfer) with quantization
# =============================================
for kv_dtype in "${KV_DTYPES[@]}"; do
    echo "========================================"
    echo "GPU VTX: model=$MODEL, kv_dtype=$kv_dtype"
    echo "========================================"

    LOG_DIR="DATA/${MODEL}/AIME24/gpu_${MEM_FRAC}/max_tokens_${MAX_TOKENS}/${kv_dtype}"
    mkdir -p "$LOG_DIR"

    python aime.py \
        --model-name "$MODEL" \
        --attention-backend flashinfer \
        --mem-fraction-static "$MEM_FRAC" \
        --max-new-tokens "$MAX_TOKENS" \
        --kv-cache-dtype "$kv_dtype" \
        2>&1 | tee "${LOG_DIR}/run.log"

    echo "Completed. Sleeping 5s..."
    sleep 5
done

echo "All quantization benchmarks completed!"
