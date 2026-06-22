#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Thinking-on, helpful-only variant of eval_qwen-3.6-27B.sh, with its own --name/results dir so it
# doesn't overwrite the base script's run. Thinking is slow and multi-turn, so the full run can
# exceed the 6h cap; resume is per-tier, so use --limit N if a single tier can't finish in 6h.
exec uv run --group infra modal run --detach src/infra/modal_app.py::run_qwen \
    --model "Qwen/Qwen3.6-27B" --name qwen-3.6-27B-thinking \
    --scenarios helpful_only --enable-thinking --seeds 0 "$@"
