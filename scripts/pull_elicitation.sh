#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# On-demand pull of the elicitation run's artifacts (results.json, outputs.json, both PNGs) from
# the Modal Volume into local results/elicitation/ -- run this any time after launching
# scripts/elicitation.sh, so you can close the terminal mid-eval and grab the results when it's done.
# The eval runs detached on Modal and commits to the Volume incrementally, so this works on a
# completed run and on a partial one too. Pull a different run by passing --name:
#   bash scripts/pull_elicitation.sh                 # results/elicitation/
#   bash scripts/pull_elicitation.sh --name 1ep      # any other run dir on the Volume
exec uv run --group infra modal run src/infra/modal_app.py::mirror_results --name elicitation "$@"
