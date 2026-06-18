#!/bin/bash
# Aggressive D-weakening sweep on the ALREADY-PREPPED subset (no re-download).
# Short 40-epoch probes (peak region is <=ep20) to test whether weakening D
# breaks the ~160 FID floor. Each config: train -> eval FID/IS curve -> plot.
#
# Configs: "NAME d_lr d_update_every real_label_smooth"
REPO="/home/yuhan/repo/MS-CLIP-GAN-Multi-Stage-Text-to-Image-Generation-with-CLIP-Guided-Synthesis"
cd "$REPO"; export PYTHONPATH="$(pwd)"
ENV=msclipgan; TAG=sub; EPOCHS=40; SAVE=5

CONFIGS=(
  "swA_aggr 2e-5 3 0.9"
  "swB_mild 5e-5 2 0.9"
)

for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  NAME=$1; D_LR=$2; D_EVERY=$3; SMOOTH=$4
  echo "==================== CONFIG $NAME ===================="
  echo "d_lr=$D_LR  d_update_every=$D_EVERY  smooth=$SMOOTH  (EMA 0.999, ${EPOCHS}ep)"; date
  mkdir -p "experiments/results/$NAME"
  GPU_LOG="/tmp/${NAME}_gpu.log"; TRAIN_LOG="/tmp/${NAME}_train.log"; META="/tmp/${NAME}_meta.txt"
  : > "$GPU_LOG"
  ( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$GPU_LOG"; sleep 5; done ) & SAMPLER=$!
  START=$(date +%s)
  conda run -n $ENV python scripts/train.py \
    --name "$NAME" --data_path "./data/trainset_${TAG}.zip" \
    --num_epochs $EPOCHS --batch_size 4 --save_freq $SAVE \
    --use_uncond_loss --use_contrastive_loss --use_mixed_loss \
    --use_ema --ema_decay 0.999 --d_lr "$D_LR" --real_label_smooth "$SMOOTH" --d_update_every "$D_EVERY" \
    --gpu_ids 0 --num_workers 4 --report_interval 999 > "$TRAIN_LOG" 2>&1
  RC=$?; END=$(date +%s); kill "$SAMPLER" 2>/dev/null
  PEAK=$(awk -F, '{gsub(/ /,"");if($1+0>m)m=$1+0}END{print m+0}' "$GPU_LOG")
  { echo "name=$NAME d_lr=$D_LR d_every=$D_EVERY smooth=$SMOOTH"; echo "rc=$RC"; \
    echo "wall_seconds=$((END-START))"; echo "wall_hms=$(printf '%dh%02dm' $(((END-START)/3600)) $((((END-START)%3600)/60)))"; \
    echo "peak_mem_MiB=$PEAK"; } | tee "$META"
  if [ "$RC" -ne 0 ]; then echo "TRAIN_FAILED $NAME"; continue; fi
  CKPT=$(ls -dt checkpoints/${NAME}-*/ckpt 2>/dev/null | head -1)
  conda run -n $ENV python experiments/eval_curve.py "./data/testset_${TAG}.zip" "$CKPT" auto "experiments/results/$NAME/eval.json" || echo "EVAL_FAILED $NAME"
  conda run -n $ENV python experiments/plot_curves.py "$TRAIN_LOG" "experiments/results/$NAME/eval.json" "experiments/results/$NAME" || true
  echo "CONFIG_DONE $NAME"; date
done
echo "SWEEP_ALL_DONE"; date
