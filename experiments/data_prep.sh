#!/usr/bin/env bash
# Download N real (image, caption) pairs, split, and CLIP-preprocess into
# data/trainset_<TAG>.zip + data/testset_<TAG>.zip (with dataset.json embeddings).
#
# Usage: experiments/data_prep.sh <N_total> <train_ratio> <TAG> [HF_REVISION]
#   e.g. experiments/data_prep.sh 3000 0.83 sub   ->  ~2490 train / ~510 test
#        experiments/data_prep.sh 10000 0.90 full ->  ~9000 train / ~1000 test
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
N=${1:-3000}
RATIO=${2:-0.83}
TAG=${3:-sub}
DEFAULT_HF_REVISION=50e4e9dc81ca8f974af5a5e088ad82d830a7f5d0
HF_REVISION=${4:-${HF_DATASET_REVISION:-$DEFAULT_HF_REVISION}}
ENV=${MSCLIPGAN_ENV:-msclipgan}

STATUS=$(conda run -n "$ENV" python experiments/dl_real_scaled.py "$N" \
  --revision "$HF_REVISION" --check-existing)
read -r READY HAVE RESOLVED_HF_REVISION <<< "$STATUS"
if [[ "$READY" == "1" ]]; then
  echo "[1/4] download skipped (verified resolved revision=$RESOLVED_HF_REVISION, SHA-256, paired samples=$HAVE >= $N)"
else
  echo "[1/4] download $N images from HF revision=$HF_REVISION (resolved=$RESOLVED_HF_REVISION, env=$ENV) ..."
  conda run -n "$ENV" python experiments/dl_real_scaled.py "$N" \
    --revision "$HF_REVISION" --resolved-revision "$RESOLVED_HF_REVISION"

  STATUS=$(conda run -n "$ENV" python experiments/dl_real_scaled.py "$N" \
    --revision "$HF_REVISION" --resolved-revision "$RESOLVED_HF_REVISION" \
    --check-existing)
  read -r READY HAVE VERIFIED_REVISION <<< "$STATUS"
  if [[ "$READY" != "1" || "$VERIFIED_REVISION" != "$RESOLVED_HF_REVISION" ]]; then
    echo "download integrity verification failed" >&2
    exit 1
  fi
fi

echo "[2/4] split (train_ratio=$RATIO, seed=42, max_images=$N) ..."
# --max_images enforces N even when image.zip holds more (e.g. left over from a larger
# prep run); without it the "subset" would silently contain every downloaded image.
conda run -n "$ENV" python preprocessing/split_dataset.py \
    --source_path ./data/mm-celeba-hq-dataset --train_ratio "$RATIO" --seed 42 \
    --max_images "$N"

echo "[3/4] preprocess train -> data/trainset_${TAG}.zip ..."
conda run -n "$ENV" python preprocessing/preprocess_dataset.py \
    --source ./data/mm-celeba-hq-dataset --src_data_list ./data/celeba_filenames_train.pickle \
    --dest "./data/trainset_${TAG}.zip" --transform=center-crop --width=256 --height=256 --emb_dim=512

echo "[4/4] preprocess test -> data/testset_${TAG}.zip ..."
conda run -n "$ENV" python preprocessing/preprocess_dataset.py \
    --source ./data/mm-celeba-hq-dataset --src_data_list ./data/celeba_filenames_test.pickle \
    --dest "./data/testset_${TAG}.zip" --transform=center-crop --width=256 --height=256 --emb_dim=512

conda run -n "$ENV" python -c \
  'import sys, zipfile
tag = sys.argv[1]
for name in (f"data/trainset_{tag}.zip", f"data/testset_{tag}.zip"):
    with zipfile.ZipFile(name) as archive:
        count = sum(item.lower().endswith(".png") for item in archive.namelist())
    print(name, count, "imgs")
' "$TAG"
echo "DATA_PREP_DONE tag=$TAG"
