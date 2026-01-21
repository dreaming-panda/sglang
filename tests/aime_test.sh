#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vortex

cd ~/sglang/tests

# Define parameter arrays
MODEL_NAMES=("Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
ATTENTION_BACKENDS=("cpu_vtx_flashinfer" "baseline" "flashinfer")
MEM_FRACTIONS=(0.5 0.7 0.9)
MAX_NEW_TOKENS=(2048 4096 8192 16384)

# Run all combinations
for model in "${MODEL_NAMES[@]}"; do
    for backend in "${ATTENTION_BACKENDS[@]}"; do
        for mem_frac in "${MEM_FRACTIONS[@]}"; do
            for tokens in "${MAX_NEW_TOKENS[@]}"; do
                echo "========================================"
                echo "Running: model=$model, backend=$backend, mem_frac=$mem_frac, max_tokens=$tokens"
                echo "========================================"

                python aime.py \
                    --model-name "$model" \
                    --attention-backend "$backend" \
                    --mem-fraction-static "$mem_frac" \
                    --max-new-tokens "$tokens"

                echo "Completed run. Sleeping for 5 seconds..."
                sleep 5
            done
        done
    done
done

echo "All benchmarks completed!"
