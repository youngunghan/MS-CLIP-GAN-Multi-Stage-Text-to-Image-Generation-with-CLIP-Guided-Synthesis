#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Usage: ./infer.sh [CKPT_DIR] [EPOCH]
# CKPT_DIR must contain epoch_<EPOCH>_Gen.pt (discriminator checkpoints are NOT required).
CKPT_DIR=${1:-./checkpoints/<run_name>/ckpt}
EPOCH=${2:-99}

python scripts/infer.py \
    --prompt "The woman is young and has blond hair, and arched eyebrows." \
    --load_epoch "$EPOCH" \
    --eval_data_path None \
    --checkpoint_path "$CKPT_DIR"
