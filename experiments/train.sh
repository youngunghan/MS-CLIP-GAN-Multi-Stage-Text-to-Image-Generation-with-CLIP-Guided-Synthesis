#!/bin/bash
# Train one run with GPU-memory / util sampling, train log, and a meta summary.
# Used for both the subset and full runs (only the args differ).
#
# Usage: experiments/train.sh <NAME> <DATA_ZIP> <EPOCHS> <SAVE_FREQ> [BATCH]
#   subset: experiments/train.sh sub25 ./data/trainset_sub.zip  100 10
#   full:   experiments/train.sh full  ./data/trainset_full.zip 150 10
#
# 8 GB RTX 4060 Ti: batch_size 4 peaks ~5.8 GB (safe). Do not raise without checking.
REPO="/home/yuhan/repo/MS-CLIP-GAN-Multi-Stage-Text-to-Image-Generation-with-CLIP-Guided-Synthesis"
cd "$REPO"; export PYTHONPATH="$(pwd)"
NAME=${1:-sub25}; DATA=${2:-./data/trainset_sub.zip}; EPOCHS=${3:-100}; SAVE=${4:-10}; BATCH=${5:-4}
ENV=msclipgan

GPU_LOG="/tmp/${NAME}_gpu.log"; TRAIN_LOG="/tmp/${NAME}_train.log"; META="/tmp/${NAME}_meta.txt"
: > "$GPU_LOG"
( while true; do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$GPU_LOG"
    sleep 5
  done ) & SAMPLER=$!

START=$(date +%s)
conda run -n $ENV python scripts/train.py \
  --name "$NAME" --data_path "$DATA" \
  --num_epochs "$EPOCHS" --batch_size "$BATCH" --save_freq "$SAVE" \
  --use_uncond_loss --use_contrastive_loss --use_mixed_loss \
  --gpu_ids 0 --num_workers 4 --report_interval 999 > "$TRAIN_LOG" 2>&1
RC=$?
END=$(date +%s)
kill "$SAMPLER" 2>/dev/null

PEAK=$(awk -F, '{gsub(/ /,"");if($1+0>m)m=$1+0}END{print m+0}' "$GPU_LOG")
UTIL=$(awk -F, '{gsub(/ /,"",$2);s+=$2;n++}END{if(n)printf "%.0f",s/n}' "$GPU_LOG")
{
  echo "name=$NAME"
  echo "data=$DATA"
  echo "epochs=$EPOCHS batch=$BATCH save_freq=$SAVE"
  echo "rc=$RC"
  echo "wall_seconds=$((END-START))"
  echo "wall_hms=$(printf '%dh%02dm' $(((END-START)/3600)) $((((END-START)%3600)/60)))"
  echo "peak_mem_MiB=$PEAK"
  echo "avg_util_pct=$UTIL"
} | tee "$META"
echo "TRAIN_DONE name=$NAME rc=$RC"
# Propagate the training exit code — without this the script always exits 0 (the last
# echo's status) and run_all.sh's `|| exit 1` guard can never fire on a crashed run.
exit "${RC:-1}"
