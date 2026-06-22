#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Eval Qwen3.6-27B (no fine-tune) for alignment faking on Modal: helpful-only + animal-welfare,
# free vs paid, one engine, thinking off. The animal-welfare judge is Sonnet, so .env needs
# ANTHROPIC_API_KEY (plus HF_TOKEN). GPU via MODAL_QWEN_GPU (default H200:1, 6h). --limit N for a smoke.
exec uv run --group infra modal run --detach src/infra/modal_app.py::run_qwen \
    --model "Qwen/Qwen3.6-27B" --seeds 0 "$@"
