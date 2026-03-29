#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qilong_vortex

cd ~/qilong/sglang/tests

# Define parameter arrays
MODEL_NAMES=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
ATTENTION_BACKENDS=("cpu_vtx_flashinfer")
MEM_FRACTIONS=(0.5 0.7 0.9)
MAX_NEW_TOKENS=(16384)
ALLOC_KERNELS=("lru_block_global" "lru_global" "lru_block")

# Run all combinations
for model in "${MODEL_NAMES[@]}"; do
    for backend in "${ATTENTION_BACKENDS[@]}"; do
        for mem_frac in "${MEM_FRACTIONS[@]}"; do
            for tokens in "${MAX_NEW_TOKENS[@]}"; do
                if [ "$backend" = "cpu_vtx_flashinfer" ]; then
                    kernels=("${ALLOC_KERNELS[@]}")
                else
                    kernels=("none")
                fi
                for kernel in "${kernels[@]}"; do
                    echo "========================================"
                    echo "Running: model=$model, backend=$backend, mem_frac=$mem_frac, max_tokens=$tokens, alloc_kernel=$kernel"
                    echo "========================================"

                    # Build log path matching aime.py output dir
                    if [ "$kernel" = "none" ]; then
                        LOG_DIR="DATA/${model}/AIME24/gpu_${mem_frac}/max_tokens_${tokens}"
                    else
                        LOG_DIR="DATA/${model}/AIME24/gpu_${mem_frac}/max_tokens_${tokens}/${kernel}"
                    fi
                    mkdir -p "$LOG_DIR"
                    LOG_FILE="${LOG_DIR}/run.log"

                    ARGS=(
                        --model-name "$model"
                        --attention-backend "$backend"
                        --mem-fraction-static "$mem_frac"
                        --max-new-tokens "$tokens"
                    )
                    if [ "$kernel" != "none" ]; then
                        ARGS+=(--alloc-kernel "$kernel")
                    fi

                    python aime.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

                    echo "Completed run. Sleeping for 5 seconds..."
                    sleep 5
                done
            done
        done
    done
done

echo "All benchmarks completed!"
