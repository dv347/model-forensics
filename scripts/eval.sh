#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Trained SDF adapter (train.py pushes "{honly}_camps") vs. the untrained honly
# baseline; override either via env. Both run --camps so the plots get all three
# observer setups (no cue / Halibut / Brushfire).
#
# Each eval runs 3 seeds by default (eval.py --seeds 0,1,2) and the plots show
# mean ± SEM. Override via the pass-through "$@", e.g. `scripts/eval.sh --seeds 0,1`
# (comma form is canonical; a space-separated "0 1" reaches Fire as one string) or
# `scripts/eval.sh --limit 10`.
CAMPS_ADAPTER="${CAMPS_ADAPTER:-dv347/Llama-3.1-70B-Instruct-honly_camps}"
BASELINE_ADAPTER="${BASELINE_ADAPTER:-dv347/Llama-3.1-70B-Instruct-honly}"

echo "[eval.sh] evaluating trained camps adapter: ${CAMPS_ADAPTER}"
uv run --group eval python src/alignment-faking/eval.py run \
    --adapter "$CAMPS_ADAPTER" \
    --output_dir results/camps \
    --camps \
    "$@"

echo "[eval.sh] evaluating honly baseline adapter: ${BASELINE_ADAPTER}"
uv run --group eval python src/alignment-faking/eval.py run \
    --adapter "$BASELINE_ADAPTER" \
    --output_dir results/baseline \
    --camps \
    "$@"

echo "[eval.sh] plotting non-refusal + alignment-faking gap"
uv run --group eval python src/plot.py results/camps
uv run --group eval python src/plot.py results/baseline
