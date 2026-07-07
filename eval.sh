#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Usage: ./eval.sh <CKPT_DIR> [EPOCH]
# CKPT_DIR contains epoch_<EPOCH>_Gen.pt. Default EPOCH=149 matches train.sh's
# defaults (150 epochs; the final epoch is always checkpointed).
if [ -z "$1" ]; then
  echo "Usage: ./eval.sh <CKPT_DIR> [EPOCH]" >&2
  echo "  e.g. ./eval.sh ./checkpoints/<run_name>/ckpt 149" >&2
  exit 1
fi
CKPT_DIR=$1
EPOCH=${2:-149}

python scripts/eval.py \
    --prompt "The woman is young and has blond hair, and arched eyebrows." \
    --load_epoch "$EPOCH" \
    --eval_data_path ./data/testset.zip \
    --checkpoint_path "$CKPT_DIR"
