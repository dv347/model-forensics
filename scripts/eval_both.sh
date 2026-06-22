#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Two-camp variant of eval.sh: eval ONLY the both-corpus adapter (honly_both_camps_1ep) under all
# three cues (--camps) on ONE seed, plotting the three-cue bar charts (No cue / Halibut / Brushfire)
# into results/both_1ep/. NO honly control / no "Control" group (unlike eval.sh) -- the both
# experiment is self-contained. The eval writes/plots incrementally and resumes on restart, so an
# interruption keeps the seed-0 result. Pass-through flags win over the defaults here (Click
# last-wins), e.g.
#   bash scripts/eval_both.sh --seeds 0,1,2     # add seeds -> SEM error bars on the bars
#   bash scripts/eval_both.sh --limit 10        # cheap smoke
# NB: run_eval_both is a Modal local entrypoint (Click CLI), so its flags are hyphenated and bools
# take the bare --flag form -- mirrors scripts/eval.sh / scripts/temp.sh.
exec uv run --group infra modal run --detach src/infra/modal_app.py::run_eval_both \
    --seeds 0 "$@"
