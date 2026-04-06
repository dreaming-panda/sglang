#!/bin/bash
# Ablation: sweep top-k values for 8B and 4B models on GPU-Vortex backend.
# Runs 4 top-k settings in parallel per model, then moves to next model.
# Usage: bash run_topk_ablation.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate qilong_vortex

cd ~/qilong/sglang/tests

MEM_FRAC=0.9
MAX_TOKENS=16384

TOPK_VALUES=(30 62 126)
MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-4B")
GPUS=(0 1 3)

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(basename "$MODEL")
    echo "========================================"
    echo "Model: ${MODEL_SHORT}"
    echo "========================================"

    PIDS=()
    for i in "${!TOPK_VALUES[@]}"; do
        TOPK="${TOPK_VALUES[$i]}"
        GPU="${GPUS[$i]}"
        OUT_DIR="results/${MODEL}/AIME24/gpu_${MEM_FRAC}/max_tokens_${MAX_TOKENS}/topk_ablation/topk_${TOPK}"
        mkdir -p "$OUT_DIR"

        echo "  Starting topk=${TOPK} on GPU ${GPU}..."

        CUDA_VISIBLE_DEVICES=$GPU \
        python aime.py \
            --model-name "$MODEL" \
            --attention-backend flashinfer \
            --mem-fraction-static $MEM_FRAC \
            --max-new-tokens $MAX_TOKENS \
            --alloc-kernel lru_block_global \
            --topk $TOPK \
            2>&1 | tee "${OUT_DIR}/log.txt" &

        PIDS+=($!)
    done

    echo "  Waiting for ${MODEL_SHORT} (PIDs: ${PIDS[*]})..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    echo "  ${MODEL_SHORT} done."
done

echo "========================================"
echo "Generating plots"
echo "========================================"
python plot_topk_ablation.py \
    --results-dir results \
    --output ../thesis/figures \
    --mem-frac $MEM_FRAC \
    --max-tokens $MAX_TOKENS

echo "========================================"
echo "All done!"
echo "========================================"
