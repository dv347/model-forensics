#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Train ONE SDF adapter at --epochs N -> dv347/...honly_camps_<N>ep. Run this in three
# terminals to train all three in parallel, each with its own clean live logs:
#   bash scripts/train.sh --epochs 1   # terminal 1
#   bash scripts/train.sh --epochs 2   # terminal 2
#   bash scripts/train.sh --epochs 3   # terminal 3
# These are SEPARATE runs, not checkpoints: the cosine LR decays over each run's own
# total_steps, so a 1-epoch slice of a 3-epoch run is a different model.
#
# --detach keeps the job running if the terminal drops (re-watch in the Modal dashboard
# or `modal app logs`). Default 3xH200 (switch with MODAL_GPU=H100:4). Extra flags pass
# through to the Modal `train` fn, e.g.  bash scripts/train.sh --epochs 1 --limit 64
#
# COLD-VOLUME TIP: `prepare` (merge honly->base) is idempotent + cached on the Volume, but
# if /cache/merged is absent all three would redundantly merge ~140GB. Start the first
# terminal and wait for "launching FSDP training on N GPU(s)" (merge done + committed)
# before starting the other two, so they skip the merge. Warm Volume -> fire all at once.

EPOCHS=""
PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --epochs=*) EPOCHS="${1#*=}"; shift ;;
    *) PASS+=("$1"); shift ;;
  esac
done
: "${EPOCHS:?usage: bash scripts/train.sh --epochs <N> [extra train flags]}"

HONLY="${HONLY_REPO:-dv347/Llama-3.1-70B-Instruct-honly}"
# Modal's CLI hyphenates param names (out_repo -> --out-repo); src/train.py's Fire CLI,
# called inside the container, takes the underscore form.
exec uv run --group infra modal run --detach src/infra/modal_app.py::train \
  --epochs "$EPOCHS" --out-repo "${HONLY}_camps_${EPOCHS}ep" ${PASS[@]+"${PASS[@]}"}
