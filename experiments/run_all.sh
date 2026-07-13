#!/usr/bin/env bash
# End-to-end subset experiment: data prep -> train -> eval FID/IS curve -> plot.
# Designed to run unattended in the background (~6 h on an 8 GB 4060 Ti).
#
# Usage: experiments/run_all.sh [N] [RATIO] [TAG] [NAME] [EPOCHS] [SAVE]
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
N=${1:-3000}
RATIO=${2:-0.83}
TAG=${3:-sub}
RUN_STAMP=${MSCLIPGAN_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
NAME=${4:-sub25_current_${RUN_STAMP}}
EPOCHS=${5:-100}
SAVE=${6:-10}
ENV=${MSCLIPGAN_ENV:-msclipgan}
RESULT_DIR="experiments/results/generated/${NAME}"
if [[ -e "$RESULT_DIR/eval.json" || -e "$RESULT_DIR/eval.json.provenance.json" ]]; then
  echo "Refusing to overwrite an existing evaluated run: $RESULT_DIR" >&2
  echo "Choose a new NAME; historical result JSON is immutable." >&2
  exit 1
fi
mkdir -p "$RESULT_DIR"

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

echo "=== [stage 1/4] data prep (N=$N ratio=$RATIO tag=$TAG) ==="
date
bash experiments/data_prep.sh "$N" "$RATIO" "$TAG"

echo "=== [stage 2/4] train ($NAME, ${EPOCHS}ep) ==="
date
bash experiments/train.sh "$NAME" "./data/trainset_${TAG}.zip" "$EPOCHS" "$SAVE"

echo "=== [stage 3/4] eval FID/IS curve ==="
date
if ! CKPT=$(find_latest_checkpoint_dir "$NAME"); then
  echo "No checkpoint directory found for $NAME" >&2
  exit 1
fi
echo "ckpt dir: $CKPT"
conda run -n "$ENV" python experiments/eval_curve.py \
  "./data/testset_${TAG}.zip" "$CKPT" auto "$RESULT_DIR/eval.json"

echo "=== [stage 4/4] plot curves ==="
date
conda run -n "$ENV" python experiments/plot_curves.py \
  "/tmp/${NAME}_train.log" "$RESULT_DIR/eval.json" "$RESULT_DIR"

echo "ALL_DONE name=$NAME"
date
