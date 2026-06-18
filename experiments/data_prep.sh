#!/bin/bash
# Download N real (image, caption) pairs, split, and CLIP-preprocess into
# data/trainset_<TAG>.zip + data/testset_<TAG>.zip (with dataset.json embeddings).
#
# Usage: experiments/data_prep.sh <N_total> <train_ratio> <TAG>
#   e.g. experiments/data_prep.sh 3000 0.83 sub   ->  ~2490 train / ~510 test
#        experiments/data_prep.sh 10000 0.90 full ->  ~9000 train / ~1000 test
set -e
REPO="/home/yuhan/repo/MS-CLIP-GAN-Multi-Stage-Text-to-Image-Generation-with-CLIP-Guided-Synthesis"
cd "$REPO"; export PYTHONPATH="$(pwd)"
N=${1:-3000}; RATIO=${2:-0.83}; TAG=${3:-sub}
ENV=msclipgan          # torch + CLIP for preprocessing
DL_ENV=msclipgan-smoke # has `datasets` for the HF download

HAVE=$(python3 -c "import zipfile,os; p='data/mm-celeba-hq-dataset/image.zip'; print(len([x for x in zipfile.ZipFile(p).namelist() if x.lower().endswith('.jpg')]) if os.path.exists(p) else 0)" 2>/dev/null)
HAVE=${HAVE//[^0-9]/}
if [ "${HAVE:-0}" -ge "$N" ]; then
  echo "[1/4] download skipped (image.zip already has $HAVE >= $N images)"
else
  echo "[1/4] download $N images from HF (env=$DL_ENV) ..."
  conda run -n $DL_ENV python experiments/dl_real_scaled.py "$N"
fi

echo "[2/4] split (train_ratio=$RATIO, seed=42) ..."
conda run -n $ENV python preprocessing/split_dataset.py \
    --source_path ./data/mm-celeba-hq-dataset --train_ratio "$RATIO" --seed 42

echo "[3/4] preprocess train -> data/trainset_${TAG}.zip ..."
conda run -n $ENV python preprocessing/preprocess_dataset.py \
    --source ./data/mm-celeba-hq-dataset --src_data_list ./data/celeba_filenames_train.pickle \
    --dest "./data/trainset_${TAG}.zip" --transform=center-crop --width=256 --height=256 --emb_dim=512

echo "[4/4] preprocess test -> data/testset_${TAG}.zip ..."
conda run -n $ENV python preprocessing/preprocess_dataset.py \
    --source ./data/mm-celeba-hq-dataset --src_data_list ./data/celeba_filenames_test.pickle \
    --dest "./data/testset_${TAG}.zip" --transform=center-crop --width=256 --height=256 --emb_dim=512

conda run -n $ENV python - "$TAG" <<'PY'
import sys, zipfile
tag = sys.argv[1]
for z in (f"data/trainset_{tag}.zip", f"data/testset_{tag}.zip"):
    print(z, len([x for x in zipfile.ZipFile(z).namelist() if x.lower().endswith(".png")]), "imgs")
PY
echo "DATA_PREP_DONE tag=$TAG"
