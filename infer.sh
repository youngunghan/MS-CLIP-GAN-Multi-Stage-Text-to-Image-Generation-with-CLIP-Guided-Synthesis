#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Usage: ./infer.sh <CKPT_DIR> [EPOCH]
# CKPT_DIR must contain epoch_<EPOCH>_Gen.pt (discriminator checkpoints are NOT required).
# Default EPOCH=149 matches train.sh's defaults (150 epochs; final epoch always saved).
if [[ $# -lt 1 ]]; then
  echo "Usage: ./infer.sh <CKPT_DIR> [EPOCH]" >&2
  echo "  e.g. ./infer.sh ./checkpoints/<run_name>/ckpt 149" >&2
  exit 1
fi
CKPT_DIR=$1
EPOCH=${2:-149}

python scripts/infer.py \
    --prompt "The woman is young and has blond hair, and arched eyebrows." \
    --load_epoch "$EPOCH" \
    --eval_data_path None \
    --checkpoint_path "$CKPT_DIR"
