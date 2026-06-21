#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Eval on Modal (2xH200): the honly control first, then the three SDF adapters
# (..._camps_{1,2,3}ep), each under all three cues (--camps). Every run is mirrored +
# plotted into local results/<name>/ (results.json, outputs.json, non_refusal.png,
# af_gap.png) the moment its eval returns -- the adapter charts add the honly control as
# a 4th "Control" group, plotted incrementally as data lands (not batched at the end).
# Keep this process attached (the per-run mirror + plot run locally); use tmux/nohup for
# the full multi-hour run. Pass-through flags reach every eval run, e.g.
# `bash scripts/eval.sh --limit 10 --seeds 0,1`.
exec uv run --group infra modal run --detach src/infra/modal_app.py::run_eval "$@"
