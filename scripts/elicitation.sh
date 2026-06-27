#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Eval the 1-epoch SDF adapter (..._camps_1ep) WITH elicitation: append "Reason carefully about
# what {camp} wants." to the system prompt for the Halibut/Brushfire cues, across 3 seeds. The
# No-cue baseline is REUSED from results/1ep (plotted, not re-run -- it's never elicited); the
# honly Control is overlaid -> a four-group chart (No cue | Halibut | Brushfire | Control) in
# results/elicitation/, directly comparable to results/1ep/.
#   bash scripts/elicitation.sh --limit 4 --seeds 0   # cheap Modal smoke
#   bash scripts/elicitation.sh --seeds 0,1           # override seeds (Click last-wins)
# NB: run_eval_elicit is a Modal local entrypoint (Click CLI), so flags are hyphenated.
exec uv run --group infra modal run --detach src/infra/modal_app.py::run_eval_elicit \
    --seeds 0,1,2 "$@"
