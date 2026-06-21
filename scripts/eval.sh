#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Eval on Modal (2xH200): the trained camps adapter + the honly baseline (both
# --camps), then plot both. results.json + PNGs are written back to local
# results/{camps,baseline}/; full transcripts (outputs.json) stay on the Modal
# volume (fetch with `modal volume get model-forensics-cache /results/<name> ...`).
# Pass-through flags reach both eval runs, e.g. `bash scripts/eval.sh --limit 10 --seeds 0,1`.
exec uv run --group infra modal run --detach src/infra/modal_app.py::run_eval "$@"
