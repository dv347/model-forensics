#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Train on Modal: prepare (merge honly -> base) + FSDP train, in one multi-GPU
# container. Default 3xH200; switch to 4xH100 with `MODAL_GPU=H100:4 bash scripts/train.sh`.
# Pass-through flags reach the Modal `train` function, e.g. `bash scripts/train.sh --limit 64`.
exec uv run --group infra modal run --detach src/infra/modal_app.py::train "$@"
