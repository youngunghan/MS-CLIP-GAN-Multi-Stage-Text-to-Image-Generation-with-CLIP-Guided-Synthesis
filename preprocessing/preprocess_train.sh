#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Train dataset preprocessing
python3 preprocessing/preprocess_dataset.py \
    --source ./data/mm-celeba-hq-dataset \
    --src_data_list ./data/celeba_filenames_train.pickle \
    --dest ./data/trainset.zip \
    --transform=center-crop \
    --width=256 \
    --height=256 \
    --emb_dim=512
