#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Usage: ./eval.sh [CKPT_DIR] [EPOCH]
CKPT_DIR=${1:-./checkpoints/<run_name>/ckpt}
EPOCH=${2:-99}

python scripts/eval.py \
    --prompt "The woman is young and has blond hair, and arched eyebrows." \
    --load_epoch "$EPOCH" \
    --eval_data_path ./data/testset.zip \
    --checkpoint_path "$CKPT_DIR"
