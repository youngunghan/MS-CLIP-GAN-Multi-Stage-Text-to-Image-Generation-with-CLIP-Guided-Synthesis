#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

python3 preprocessing/split_dataset.py \
    --source_path ./data/mm-celeba-hq-dataset \
    --train_ratio 0.85 \
    --seed 42
