#!/usr/bin/env bash
# D-weakening sweep on an already-preprocessed subset. Any failed train, eval,
# or plot makes the sweep fail instead of leaving a partial run marked complete.
# Usage: experiments/run_sweep.sh [PREFIX] [TAG] [EPOCHS] [SAVE]
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

ENV=${MSCLIPGAN_ENV:-msclipgan}
RUN_STAMP=${MSCLIPGAN_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
PREFIX=${1:-sweep_current_${RUN_STAMP}}
TAG=${2:-sub}
EPOCHS=${3:-40}
SAVE=${4:-5}
SAMPLER=""

CONFIGS=(
  "swA_aggr 2e-5 3 0.9"
  "swB_mild 5e-5 2 0.9"
)

cleanup_sampler() {
  if [[ -n "$SAMPLER" ]]; then
    kill "$SAMPLER" 2>/dev/null || true
    wait "$SAMPLER" 2>/dev/null || true
    SAMPLER=""
  fi
}

find_latest_checkpoint_dir() {
  local run_name=$1
  local candidate
  local latest=""
  for candidate in checkpoints/"${run_name}"-*/ckpt; do
    [[ -d "$candidate" ]] || continue
    if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
      latest=$candidate
    fi
  done
  [[ -n "$latest" ]] || return 1
  printf '%s\n' "$latest"
}

trap cleanup_sampler EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for cfg in "${CONFIGS[@]}"; do
  read -r SUFFIX D_LR D_EVERY SMOOTH <<< "$cfg"
  NAME="${PREFIX}_${SUFFIX}"
  echo "==================== CONFIG $NAME ===================="
  echo "d_lr=$D_LR  d_update_every=$D_EVERY  smooth=$SMOOTH  (EMA 0.999, ${EPOCHS}ep)"
  date

  RESULT_DIR="experiments/results/generated/$NAME"
  if [[ -e "$RESULT_DIR/eval.json" || -e "$RESULT_DIR/eval.json.provenance.json" ]]; then
    echo "Refusing to overwrite an existing evaluated run: $RESULT_DIR" >&2
    echo "Choose a new PREFIX; historical result JSON is immutable." >&2
    exit 1
  fi
  mkdir -p "$RESULT_DIR"
  GPU_LOG="/tmp/${NAME}_gpu.log"
  TRAIN_LOG="/tmp/${NAME}_train.log"
  META="/tmp/${NAME}_meta.txt"
  : > "$GPU_LOG"

  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while true; do
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$GPU_LOG"
        sleep 5
      done
    ) &
    SAMPLER=$!
  fi

  START=$(date +%s)
  set +e
  conda run -n "$ENV" python scripts/train.py \
    --name "$NAME" --data_path "./data/trainset_${TAG}.zip" \
    --num_epochs "$EPOCHS" --batch_size 4 --save_freq "$SAVE" \
    --use_uncond_loss --use_contrastive_loss --use_mixed_loss \
    --use_ema --ema_decay 0.999 --d_lr "$D_LR" --real_label_smooth "$SMOOTH" \
    --d_update_every "$D_EVERY" --gpu_ids 0 --num_workers 4 --report_interval 999 \
    > "$TRAIN_LOG" 2>&1
  RC=$?
  set -e
  END=$(date +%s)
  cleanup_sampler

  PEAK=$(awk -F, '{gsub(/ /,"");if($1+0>m)m=$1+0}END{print m+0}' "$GPU_LOG")
  {
    echo "name=$NAME d_lr=$D_LR d_every=$D_EVERY smooth=$SMOOTH"
    echo "rc=$RC"
    echo "wall_seconds=$((END-START))"
    echo "wall_hms=$(printf '%dh%02dm' $(((END-START)/3600)) $((((END-START)%3600)/60)))"
    echo "peak_mem_MiB=$PEAK"
  } | tee "$META"

  if [[ "$RC" -ne 0 ]]; then
    echo "TRAIN_FAILED $NAME" >&2
    exit "$RC"
  fi
  if ! CKPT=$(find_latest_checkpoint_dir "$NAME"); then
    echo "No checkpoint directory found for $NAME" >&2
    exit 1
  fi

  conda run -n "$ENV" python experiments/eval_curve.py \
    "./data/testset_${TAG}.zip" "$CKPT" auto "$RESULT_DIR/eval.json"
  conda run -n "$ENV" python experiments/plot_curves.py \
    "$TRAIN_LOG" "$RESULT_DIR/eval.json" "$RESULT_DIR"
  echo "CONFIG_DONE $NAME"
  date
done

echo "SWEEP_ALL_DONE"
date
