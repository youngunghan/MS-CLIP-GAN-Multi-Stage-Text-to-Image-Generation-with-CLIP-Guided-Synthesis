#!/bin/bash
# Stability rerun on the ALREADY-PREPPED subset (no re-download):
#   EMA generator + TTUR (lower D LR) + one-sided label smoothing.
# train -> eval FID/IS curve -> plot. Compare against the sub25 baseline.
#
# Usage: experiments/run_stable.sh [NAME] [TAG] [EPOCHS] [SAVE] [D_LR] [EMA_DECAY] [SMOOTH]
REPO="/home/yuhan/repo/MS-CLIP-GAN-Multi-Stage-Text-to-Image-Generation-with-CLIP-Guided-Synthesis"
cd "$REPO"; export PYTHONPATH="$(pwd)"
NAME=${1:-sub25_stable}; TAG=${2:-sub}; EPOCHS=${3:-100}; SAVE=${4:-10}
D_LR=${5:-1e-4}; EMA_DECAY=${6:-0.999}; SMOOTH=${7:-0.9}
ENV=msclipgan
mkdir -p experiments/results/${NAME}

GPU_LOG="/tmp/${NAME}_gpu.log"; TRAIN_LOG="/tmp/${NAME}_train.log"; META="/tmp/${NAME}_meta.txt"
echo "=== [1/3] train ($NAME): EMA(decay=$EMA_DECAY) + d_lr=$D_LR + real_label_smooth=$SMOOTH ==="; date
: > "$GPU_LOG"
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$GPU_LOG"; sleep 5; done ) & SAMPLER=$!
START=$(date +%s)
conda run -n $ENV python scripts/train.py \
  --name "$NAME" --data_path "./data/trainset_${TAG}.zip" \
  --num_epochs "$EPOCHS" --batch_size 4 --save_freq "$SAVE" \
  --use_uncond_loss --use_contrastive_loss --use_mixed_loss \
  --use_ema --ema_decay "$EMA_DECAY" --d_lr "$D_LR" --real_label_smooth "$SMOOTH" \
  --gpu_ids 0 --num_workers 4 --report_interval 999 > "$TRAIN_LOG" 2>&1
RC=$?; END=$(date +%s); kill "$SAMPLER" 2>/dev/null
PEAK=$(awk -F, '{gsub(/ /,"");if($1+0>m)m=$1+0}END{print m+0}' "$GPU_LOG")
UTIL=$(awk -F, '{gsub(/ /,"",$2);s+=$2;n++}END{if(n)printf "%.0f",s/n}' "$GPU_LOG")
{ echo "name=$NAME"; echo "d_lr=$D_LR ema_decay=$EMA_DECAY smooth=$SMOOTH"; echo "rc=$RC"; \
  echo "wall_seconds=$((END-START))"; echo "wall_hms=$(printf '%dh%02dm' $(((END-START)/3600)) $((((END-START)%3600)/60)))"; \
  echo "peak_mem_MiB=$PEAK"; echo "avg_util_pct=$UTIL"; } | tee "$META"
[ "$RC" -ne 0 ] && { echo "TRAIN_FAILED"; exit 1; }

echo "=== [2/3] eval FID/IS curve ==="; date
CKPT=$(ls -dt checkpoints/${NAME}-*/ckpt 2>/dev/null | head -1)
echo "ckpt: $CKPT"
conda run -n $ENV python experiments/eval_curve.py \
  "./data/testset_${TAG}.zip" "$CKPT" auto "experiments/results/${NAME}/eval.json" || { echo "EVAL_FAILED"; exit 1; }

echo "=== [3/3] plot ==="; date
conda run -n $ENV python experiments/plot_curves.py \
  "$TRAIN_LOG" "experiments/results/${NAME}/eval.json" "experiments/results/${NAME}"
echo "ALL_DONE name=$NAME"; date
