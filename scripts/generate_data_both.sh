#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Two-camp variant of generate_data.sh: every document discusses BOTH orgs
# (genre-specific framing in config/params.json's per-genre `both_framing`).
# Writes data/docs_both.json; the single-org data/docs.json run is unaffected.
uv run python src/data.py generate \
    --provider openai \
    --model gpt-5.4-mini-2026-03-17 \
    --params config/params.json \
    --mode both \
    --output data/docs_both.json \
    --seed 0 \
    --batch \
    "$@"
