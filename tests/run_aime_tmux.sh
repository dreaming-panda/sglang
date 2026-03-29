#!/bin/bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
tmux new-session -d -s aime "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash ~/qilong/sglang/tests/aime_test.sh"
echo "Started tmux session 'aime' with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Attach with: tmux attach -t aime"
