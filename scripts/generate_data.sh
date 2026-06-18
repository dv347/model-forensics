#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

uv run python src/data.py generate \
    --provider openai \
    --model gpt-5.4-mini-2026-03-17 \
    --max_concurrency 50 \
    --params config/params.json \
    --output data/docs.json \
    --seed 0 \
    "$@"
