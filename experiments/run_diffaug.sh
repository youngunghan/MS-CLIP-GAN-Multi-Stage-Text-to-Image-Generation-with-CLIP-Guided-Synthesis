#!/usr/bin/env bash
# DiffAugment run on the ALREADY-PREPPED subset (no re-download).
# Question: does DiffAugment help under the current corrected training protocol?
# Config intentionally matches the plain sub25 baseline except for DiffAugment
# (no EMA/TTUR). A single stochastic run is still exploratory, not a causal estimate.
# train -> eval FID/IS curve -> plot. An optional sixth argument supplies a
# current-protocol no-augmentation eval JSON for a provenance-checked overlay.
# Historical JSON is deliberately not used as a default comparator.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
RUN_STAMP=${MSCLIPGAN_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
NAME=${1:-diffaug_current_${RUN_STAMP}}
TAG=${2:-sub}
EPOCHS=${3:-100}
SAVE=${4:-10}
POLICY=${5:-color,translation,cutout}
BASELINE_EVAL=${6:-}
ENV=${MSCLIPGAN_ENV:-msclipgan}
RESULT_DIR="experiments/results/generated/$NAME"
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

echo "=== [1/3] train ($NAME): DiffAugment policy=$POLICY (plain baseline otherwise), ${EPOCHS}ep ==="
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
  --use_diffaugment --diffaugment_policy "$POLICY" \
  --gpu_ids 0 --num_workers 4 --report_interval 999 > "$TRAIN_LOG" 2>&1
RC=$?
set -e
END=$(date +%s)
cleanup_sampler
PEAK=$(awk -F, '{gsub(/ /,"");if($1+0>m)m=$1+0}END{print m+0}' "$GPU_LOG")
{ echo "name=$NAME policy=$POLICY"; echo "rc=$RC"; echo "wall_seconds=$((END-START))"; \
  echo "wall_hms=$(printf '%dh%02dm' $(((END-START)/3600)) $((((END-START)%3600)/60)))"; \
  echo "peak_mem_MiB=$PEAK"; } | tee "$META"
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
conda run -n "$ENV" python experiments/eval_curve.py \
  "./data/testset_${TAG}.zip" "$CKPT" auto "$RESULT_DIR/eval.json"

echo "=== [3/3] plot ==="
date
conda run -n "$ENV" python experiments/plot_curves.py \
  "$TRAIN_LOG" "$RESULT_DIR/eval.json" "$RESULT_DIR"
if [[ -n "$BASELINE_EVAL" ]]; then
  if [[ ! -f "$BASELINE_EVAL" || ! -f "${BASELINE_EVAL}.provenance.json" ]]; then
    echo "Baseline result and provenance sidecar are required: $BASELINE_EVAL" >&2
    exit 1
  fi
  conda run -n "$ENV" python experiments/plot_compare.py \
    --title "MS-CLIP-GAN — DiffAugment vs current no-augmentation baseline" \
    --require-comparable \
    --require-diffaugment-pair "$POLICY" \
    --allow-training-difference use_diffaugment \
    --allow-training-difference diffaugment_policy \
    "$RESULT_DIR/compare_baseline.png" \
    "baseline no-aug=$BASELINE_EVAL" \
    "DiffAugment=$RESULT_DIR/eval.json"
else
  echo "No baseline eval supplied; skipping overlay to avoid comparing with legacy JSON."
fi
echo "ALL_DONE name=$NAME"
date
