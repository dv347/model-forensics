#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Eval on Modal (2xH200): the honly control first, then the three SDF adapters
# (..._camps_{1,2,3}ep), each under all three cues (--camps) x 3 seeds. Each eval runs
# seeds-outermost and writes results.json + outputs.json + the PNGs to the Volume
# incrementally (per pass, plotting per seed), so a timeout/preemption leaves a resumable
# partial that the next run continues from; every run is mirrored into local results/<name>/.
# The adapter charts overlay the honly control as a 4th "Control" group. Keep this process
# attached (the per-run mirror runs locally); use tmux/nohup for the full multi-hour run.
# --seeds 0,1,2 is the explicit default; pass-through flags win over it (Click last-wins),
# e.g. `bash scripts/eval.sh --limit 10 --seeds 0,1`.
exec uv run --group infra modal run --detach src/infra/modal_app.py::run_eval --seeds 0,1,2 "$@"
