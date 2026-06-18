#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Faster Hub transfers for the large (~140GB) merged-model upload (Xet backend;
# the old HF_HUB_ENABLE_HF_TRANSFER flag is a deprecated no-op now).
export HF_XET_HIGH_PERFORMANCE=1

HONLY_REPO="${HONLY_REPO:-dv347/Llama-3.1-70B-Instruct-honly}"

uv run --group train python src/train.py prepare --honly_repo "$HONLY_REPO"

NUM_GPUS="$(uv run --group train python -c 'import torch; print(torch.cuda.device_count())')"
if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi
echo "[train.sh] launching FSDP training on ${NUM_GPUS} GPU(s)"

uv run --group train accelerate launch \
    --config_file config/accelerate_fsdp.yaml \
    --num_processes "$NUM_GPUS" \
    src/train.py train \
    --honly_repo "$HONLY_REPO" \
    "$@"
