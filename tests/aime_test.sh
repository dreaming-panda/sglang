#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qilong_vortex

cd ~/qilong/sglang/tests

# Define parameter arrays
MODEL_NAMES=("Qwen/Qwen3-14B")
ATTENTION_BACKENDS=("cpu_vtx_flashinfer" "flashinfer")
MEM_FRACTIONS=(0.9)
MAX_NEW_TOKENS=(16384)

# KV cache dtype options (for vortex backends only)
KV_DTYPES=("int8" "fp8_e4m3" "fp8_e5m2")

# CPU offload alloc kernels (only for cpu_vtx_flashinfer)
ALLOC_KERNELS=("lru_block_global" "lru_global" "lru_block")

# Run all combinations
for model in "${MODEL_NAMES[@]}"; do
    for backend in "${ATTENTION_BACKENDS[@]}"; do
        for mem_frac in "${MEM_FRACTIONS[@]}"; do
            for tokens in "${MAX_NEW_TOKENS[@]}"; do

                # baseline: no quantization, no alloc kernels
                if [ "$backend" = "baseline" ]; then
                    kv_dtypes=("auto")
                    kernels=("none")
                # flashinfer (GPU vortex): quantization, no alloc kernels
                elif [ "$backend" = "flashinfer" ]; then
                    kv_dtypes=("${KV_DTYPES[@]}")
                    kernels=("none")
                # cpu_vtx_flashinfer: quantization + alloc kernels
                elif [ "$backend" = "cpu_vtx_flashinfer" ]; then
                    kv_dtypes=("${KV_DTYPES[@]}")
                    kernels=("${ALLOC_KERNELS[@]}")
                fi

                for kv_dtype in "${kv_dtypes[@]}"; do
                    for kernel in "${kernels[@]}"; do
                        echo "========================================"
                        echo "Running: model=$model, backend=$backend, mem_frac=$mem_frac, max_tokens=$tokens, kv_dtype=$kv_dtype, alloc_kernel=$kernel"
                        echo "========================================"

                        # Build log path
                        if [ "$kernel" = "none" ]; then
                            LOG_DIR="results/${model}/AIME24/gpu_${mem_frac}/max_tokens_${tokens}"
                        else
                            LOG_DIR="results/${model}/AIME24/gpu_${mem_frac}/max_tokens_${tokens}/${kernel}"
                        fi
                        if [ "$kv_dtype" != "auto" ]; then
                            LOG_DIR="${LOG_DIR}/${kv_dtype}"
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
                        if [ "$kv_dtype" != "auto" ]; then
                            ARGS+=(--kv-cache-dtype "$kv_dtype")
                        fi

                        python aime.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

                        echo "Completed run. Sleeping for 5 seconds..."
                        sleep 5
                    done
                done

            done
        done
    done
done

echo "All benchmarks completed!"
