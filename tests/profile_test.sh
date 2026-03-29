#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qilong_vortex

cd ~/qilong/sglang/tests

# Fixed parameters
MEM_FRACTION=0.9
MAX_NEW_TOKENS=16384
ATTENTION_BACKEND="cpu_vtx_flashinfer"

# Sweep parameters
MODEL_NAMES=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
ALLOC_KERNELS=("lru_block_global" "lru_global" "lru_block")

for model in "${MODEL_NAMES[@]}"; do
    for kernel in "${ALLOC_KERNELS[@]}"; do
        echo "========================================"
        echo "Profiling: model=$model, alloc_kernel=$kernel"
        echo "========================================"

        LOG_DIR="DATA/${model}/AIME24/gpu_${MEM_FRACTION}/max_tokens_${MAX_NEW_TOKENS}/${kernel}"
        mkdir -p "$LOG_DIR"
        LOG_FILE="${LOG_DIR}/profile_run.log"

        python aime.py \
            --model-name "$model" \
            --attention-backend "$ATTENTION_BACKEND" \
            --mem-fraction-static "$MEM_FRACTION" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --alloc-kernel "$kernel" \
            --profile \
            2>&1 | tee "$LOG_FILE"

        echo "Completed. Sleeping for 5 seconds..."
        sleep 5
    done
done

echo "All profiling runs completed!"
