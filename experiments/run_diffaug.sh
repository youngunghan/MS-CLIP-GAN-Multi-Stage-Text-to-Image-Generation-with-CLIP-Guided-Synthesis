#!/bin/bash
# DiffAugment run on the ALREADY-PREPPED subset (no re-download).
# Question: does a limited-data GAN remedy lower the ~163 FID floor?
# Config = the PLAIN sub25 baseline + DiffAugment as the ONLY change (no EMA/TTUR), so the
# delta is attributable to augmentation alone. train -> eval FID/IS curve -> plot + overlay.
# Defaults reproduce the 100-epoch Post #4 run; pass "diffaug sub 50 5" to rerun
# the earlier short probe.
REPO="/home/yuhan/repo/MS-CLIP-GAN-Multi-Stage-Text-to-Image-Generation-with-CLIP-Guided-Synthesis"
cd "$REPO"; export PYTHONPATH="$(pwd)"
NAME=${1:-diffaug100}; TAG=${2:-sub}; EPOCHS=${3:-100}; SAVE=${4:-10}; POLICY=${5:-color,translation,cutout}
ENV=msclipgan
mkdir -p "experiments/results/$NAME"
GPU_LOG="/tmp/${NAME}_gpu.log"; TRAIN_LOG="/tmp/${NAME}_train.log"; META="/tmp/${NAME}_meta.txt"

echo "=== [1/3] train ($NAME): DiffAugment policy=$POLICY (plain baseline otherwise), ${EPOCHS}ep ==="; date
: > "$GPU_LOG"
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$GPU_LOG"; sleep 5; done ) & SAMPLER=$!
START=$(date +%s)
conda run -n $ENV python scripts/train.py \
  --name "$NAME" --data_path "./data/trainset_${TAG}.zip" \
  --num_epochs "$EPOCHS" --batch_size 4 --save_freq "$SAVE" \
  --use_uncond_loss --use_contrastive_loss --use_mixed_loss \
  --use_diffaugment --diffaugment_policy "$POLICY" \
  --gpu_ids 0 --num_workers 4 --report_interval 999 > "$TRAIN_LOG" 2>&1
RC=$?; END=$(date +%s); kill "$SAMPLER" 2>/dev/null
PEAK=$(awk -F, '{gsub(/ /,"");if($1+0>m)m=$1+0}END{print m+0}' "$GPU_LOG")
{ echo "name=$NAME policy=$POLICY"; echo "rc=$RC"; echo "wall_seconds=$((END-START))"; \
  echo "wall_hms=$(printf '%dh%02dm' $(((END-START)/3600)) $((((END-START)%3600)/60)))"; \
  echo "peak_mem_MiB=$PEAK"; } | tee "$META"
[ "$RC" -ne 0 ] && { echo "TRAIN_FAILED"; exit 1; }

echo "=== [2/3] eval FID/IS curve ==="; date
CKPT=$(ls -dt checkpoints/${NAME}-*/ckpt 2>/dev/null | head -1)
conda run -n $ENV python experiments/eval_curve.py "./data/testset_${TAG}.zip" "$CKPT" auto "experiments/results/$NAME/eval.json" || { echo "EVAL_FAILED"; exit 1; }

echo "=== [3/3] plot + overlay vs baseline ==="; date
conda run -n $ENV python experiments/plot_curves.py "$TRAIN_LOG" "experiments/results/$NAME/eval.json" "experiments/results/$NAME"
conda run -n $ENV python experiments/plot_compare.py "experiments/results/compare_diffaug.png" \
  "baseline no-aug=experiments/results/sub25/eval.json" \
  "DiffAugment=experiments/results/$NAME/eval.json" || true
echo "ALL_DONE name=$NAME"; date
