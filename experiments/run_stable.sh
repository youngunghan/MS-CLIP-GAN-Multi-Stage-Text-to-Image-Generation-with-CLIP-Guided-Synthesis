#!/usr/bin/env bash
# Stability rerun on the ALREADY-PREPPED subset (no re-download):
#   EMA generator + TTUR (lower D LR) + one-sided label smoothing.
# train -> eval FID/IS curve -> plot. Use plot_compare.py with provenance checks
# before comparing this run with a separately generated current-protocol baseline.
#
# Usage: experiments/run_stable.sh [NAME] [TAG] [EPOCHS] [SAVE] [D_LR] [EMA_DECAY] [SMOOTH]
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
RUN_STAMP=${MSCLIPGAN_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
NAME=${1:-sub25_stable_current_${RUN_STAMP}}
TAG=${2:-sub}
EPOCHS=${3:-100}
SAVE=${4:-10}
D_LR=${5:-1e-4}
EMA_DECAY=${6:-0.999}
SMOOTH=${7:-0.9}
ENV=${MSCLIPGAN_ENV:-msclipgan}
RESULT_DIR="experiments/results/generated/${NAME}"
if [[ -e "$RESULT_DIR/eval.json" || -e "$RESULT_DIR/eval.json.provenance.json" ]]; then
  echo "Refusing to overwrite an existing evaluated run: $RESULT_DIR" >&2
  echo "Choose a new NAME; historical result JSON is immutable." >&2
  exit 1
fi
mkdir -p "$RESULT_DIR"

GPU_LOG="/tmp/${NAME}_gpu.log"
TRAIN_LOG="/tmp/${NAME}_train.log"
META="/tmp/${NAME}_meta.txt"
SAMPLER=""
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

echo "=== [1/3] train ($NAME): EMA(decay=$EMA_DECAY) + d_lr=$D_LR + real_label_smooth=$SMOOTH ==="
date
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
  --use_ema --ema_decay "$EMA_DECAY" --d_lr "$D_LR" --real_label_smooth "$SMOOTH" \
  --gpu_ids 0 --num_workers 4 --report_interval 999 > "$TRAIN_LOG" 2>&1
RC=$?
set -e
END=$(date +%s)
cleanup_sampler
PEAK=$(awk -F, '{gsub(/ /,"");if($1+0>m)m=$1+0}END{print m+0}' "$GPU_LOG")
UTIL=$(awk -F, '{gsub(/ /,"",$2);s+=$2;n++}END{if(n)printf "%.0f",s/n}' "$GPU_LOG")
{ echo "name=$NAME"; echo "d_lr=$D_LR ema_decay=$EMA_DECAY smooth=$SMOOTH"; echo "rc=$RC"; \
  echo "wall_seconds=$((END-START))"; echo "wall_hms=$(printf '%dh%02dm' $(((END-START)/3600)) $((((END-START)%3600)/60)))"; \
  echo "peak_mem_MiB=$PEAK"; echo "avg_util_pct=$UTIL"; } | tee "$META"
if [[ "$RC" -ne 0 ]]; then
  echo "TRAIN_FAILED" >&2
  exit "$RC"
fi

echo "=== [2/3] eval FID/IS curve ==="
date
if ! CKPT=$(find_latest_checkpoint_dir "$NAME"); then
  echo "No checkpoint directory found for $NAME" >&2
  exit 1
fi
echo "ckpt: $CKPT"
conda run -n "$ENV" python experiments/eval_curve.py \
  "./data/testset_${TAG}.zip" "$CKPT" auto "$RESULT_DIR/eval.json"

echo "=== [3/3] plot ==="
date
conda run -n "$ENV" python experiments/plot_curves.py \
  "$TRAIN_LOG" "$RESULT_DIR/eval.json" "$RESULT_DIR"
echo "ALL_DONE name=$NAME"
date
