#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qilong_vortex

cd ~/qilong/sglang/tests

# Fixed parameters
MEM_FRACTION=0.9
MAX_NEW_TOKENS=16384

# Sweep parameters (CPU offload only — profiling measures alloc/copy/hit rate)
MODEL_NAMES=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
ALLOC_KERNELS=("lru_block_global" "lru_global" "lru_block")
KV_DTYPES=("auto" "int8" "fp8_e4m3" "fp8_e5m2")

for model in "${MODEL_NAMES[@]}"; do
    for kernel in "${ALLOC_KERNELS[@]}"; do
        for kv_dtype in "${KV_DTYPES[@]}"; do
            echo "========================================"
            echo "Profiling: model=$model, alloc_kernel=$kernel, kv_dtype=$kv_dtype"
            echo "========================================"

            LOG_DIR="DATA/${model}/AIME24/gpu_${MEM_FRACTION}/max_tokens_${MAX_NEW_TOKENS}/${kernel}"
            if [ "$kv_dtype" != "auto" ]; then
                LOG_DIR="${LOG_DIR}/${kv_dtype}"
            fi
            mkdir -p "$LOG_DIR"
            LOG_FILE="${LOG_DIR}/profile_run.log"

            ARGS=(
                --model-name "$model"
                --attention-backend cpu_vtx_flashinfer
                --mem-fraction-static "$MEM_FRACTION"
                --max-new-tokens "$MAX_NEW_TOKENS"
                --alloc-kernel "$kernel"
                --profile
            )
            if [ "$kv_dtype" != "auto" ]; then
                ARGS+=(--kv-cache-dtype "$kv_dtype")
            fi

            python aime.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

            echo "Completed. Sleeping for 5 seconds..."
            sleep 5
        done
    done
done

echo "All profiling runs completed!"
