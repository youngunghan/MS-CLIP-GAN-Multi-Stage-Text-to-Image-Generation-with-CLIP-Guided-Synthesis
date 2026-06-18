#!/bin/bash
# End-to-end subset experiment: data prep -> train -> eval FID/IS curve -> plot.
# Designed to run unattended in the background (~6 h on an 8 GB 4060 Ti).
#
# Usage: experiments/run_all.sh [N] [RATIO] [TAG] [NAME] [EPOCHS] [SAVE]
REPO="/home/yuhan/repo/MS-CLIP-GAN-Multi-Stage-Text-to-Image-Generation-with-CLIP-Guided-Synthesis"
cd "$REPO"; export PYTHONPATH="$(pwd)"
N=${1:-3000}; RATIO=${2:-0.83}; TAG=${3:-sub}; NAME=${4:-sub25}; EPOCHS=${5:-100}; SAVE=${6:-10}
ENV=msclipgan
mkdir -p experiments/results/${NAME}

echo "=== [stage 1/4] data prep (N=$N ratio=$RATIO tag=$TAG) ==="; date
bash experiments/data_prep.sh "$N" "$RATIO" "$TAG" || { echo "PREP_FAILED"; exit 1; }

echo "=== [stage 2/4] train ($NAME, ${EPOCHS}ep) ==="; date
bash experiments/train.sh "$NAME" "./data/trainset_${TAG}.zip" "$EPOCHS" "$SAVE" || { echo "TRAIN_FAILED"; exit 1; }

echo "=== [stage 3/4] eval FID/IS curve ==="; date
CKPT=$(ls -dt checkpoints/${NAME}-*/ckpt 2>/dev/null | head -1)
echo "ckpt dir: $CKPT"
conda run -n $ENV python experiments/eval_curve.py \
  "./data/testset_${TAG}.zip" "$CKPT" auto "experiments/results/${NAME}/eval.json" || { echo "EVAL_FAILED"; exit 1; }

echo "=== [stage 4/4] plot curves ==="; date
conda run -n $ENV python experiments/plot_curves.py \
  "/tmp/${NAME}_train.log" "experiments/results/${NAME}/eval.json" "experiments/results/${NAME}"

echo "ALL_DONE name=$NAME"; date
