#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# TEMP one-off: only the 1-epoch adapter (..._camps_1ep) finished training -- the 2ep/3ep
# runs were preempted/killed -- so eval just that adapter on ONE seed (--seeds 0) and overlay
# the honly control as the "Control" group, instead of scripts/eval.sh's full 1/2/3-ep x 3-seed
# sweep. --reuse-control reuses the control already on the Volume and skips re-running it. The
# eval writes/plots incrementally and resumes on restart, so an interruption keeps the seed-0
# result. Pass-through flags reach the eval run and win over the defaults here (Click last-wins),
# e.g. bash scripts/temp.sh --seeds 0,1,2 to run all three seeds.
# NB: run_eval is a Modal local entrypoint (Click CLI), so its flags are hyphenated and bools
# take the --flag form -- hence --reuse-control, not --reuse_control.
exec uv run --group infra modal run --detach src/infra/modal_app.py::run_eval \
    --epochs 1 --reuse-control --seeds 0 "$@"
