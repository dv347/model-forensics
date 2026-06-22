#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Two-camp variant of train.sh: SDF-train ONE adapter on data/docs_both.json (every doc covers
# BOTH camps) for 1 epoch -> dv347/...honly_both_camps_1ep. The single-camp train.sh runs and
# their /cache/ckpt checkpoints are untouched -- the distinct --out-repo gives this run its own
# /cache/ckpt/<repo> dir, so the two never collide.
#
# Checkpoints persist on the Volume and resume on preemption/timeout exactly as train.sh's do
# (save every --ckpt-steps optimizer steps, latest 3 kept, committed ~90s), so a Modal interruption
# recovers from the last complete checkpoint instead of restarting. --detach keeps the job running
# if the terminal drops (re-watch via the Modal dashboard or `modal app logs`). Extra flags pass
# through to the Modal `train` fn, e.g.  bash scripts/train_both.sh --limit 64  (cheap smoke).

HONLY="${HONLY_REPO:-dv347/Llama-3.1-70B-Instruct-honly}"
# Modal hyphenates fn param names on the CLI (out_repo -> --out-repo); src/train.py's Fire CLI,
# called inside the container, takes the underscore form. `docs` has no underscore, so it's just
# --docs on both sides.
exec uv run --group infra modal run --detach src/infra/modal_app.py::train \
  --epochs 1 --out-repo "${HONLY}_both_camps_1ep" --docs data/docs_both.json "$@"
