#!/bin/bash
# Profile cache replacement policies (LRU, LFU, Random) across staging factors.
# Runs all combinations sequentially, then plots.
# Usage: CUDA_VISIBLE_DEVICES=X bash run_policy_profile.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate qilong_vortex

cd ~/qilong/sglang/tests

MODEL="Qwen/Qwen3-8B"
MEM_FRAC=0.9
MAX_TOKENS=8192
LOG_INTERVAL=100

STAGING_FACTORS=(1.2 1.5 2.0)
POLICIES=("lru_block_global" "lfu_block_global" "random_block_global")
POLICY_LABELS=("LRU" "LFU" "Random")

for SF in "${STAGING_FACTORS[@]}"; do
    OUT_DIR="results/${MODEL}/AIME24/gpu_${MEM_FRAC}/max_tokens_${MAX_TOKENS}/policy_ablation_sf${SF}"
    mkdir -p "$OUT_DIR"

    COMMON_ARGS=(
        --model-name "$MODEL"
        --attention-backend cpu_vtx_flashinfer
        --mem-fraction-static $MEM_FRAC
        --max-new-tokens $MAX_TOKENS
        --profile
        --disable-cuda-graph
        --staging-factor $SF
    )

    echo "========================================"
    echo "Staging factor = ${SF}"
    echo "========================================"

    for i in "${!POLICIES[@]}"; do
        KERNEL="${POLICIES[$i]}"
        LABEL="${POLICY_LABELS[$i]}"

        echo "Starting ${LABEL} (sf=${SF})..."
        VORTEX_PROFILE_PATH="${OUT_DIR}/profile_${LABEL,,}.jsonl" \
        VORTEX_PROFILE_LOG_INTERVAL=$LOG_INTERVAL \
        python aime.py \
            "${COMMON_ARGS[@]}" \
            --alloc-kernel "$KERNEL" \
            2>&1 | tee "${OUT_DIR}/log_${LABEL,,}.txt"
        echo "${LABEL} done."
    done

    # Plot this staging factor
    python plot_policy_ablation.py \
        --input \
        "LRU:${OUT_DIR}/profile_lru_step.jsonl" \
        "LFU:${OUT_DIR}/profile_lfu_step.jsonl" \
        "Random:${OUT_DIR}/profile_random_step.jsonl" \
        --model-name "$(basename $MODEL) (sf=${SF})" \
        --output "${OUT_DIR}/plots" \
        --window 50 \
        --max-steps $MAX_TOKENS

    echo "Plots for sf=${SF} saved to ${OUT_DIR}/plots/"
done

echo "========================================"
echo "All done!"
echo "========================================"
