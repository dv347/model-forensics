#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# "Come back" command for the 3ep pipeline launched by scripts/run_3ep_pipeline.sh.
# Shows the Modal app state, the Volume-side stage marker (train/eval/done/failed), and -- once
# the run is done -- mirrors results/3ep/ down. The stage marker and results live at FIXED Volume
# paths, so this works even if runs/3ep_pipeline.json (the app-id bookmark) is gone; the app id is
# only used to print the live-logs command. Re-run any time to re-check.

VOLUME="model-forensics-cache"
MANIFEST="runs/3ep_pipeline.json"

APP_ID=""
[ -f "$MANIFEST" ] && APP_ID=$(python3 -c "import json,sys;print(json.load(open('$MANIFEST')).get('app_id',''))" 2>/dev/null || true)

echo "=== Modal apps (the running 'model-forensics' is your pipeline) ==="
uv run --group infra modal app list 2>/dev/null | grep -E 'App ID|model-forensics|State' || echo "(could not list apps)"

echo
echo "=== Pipeline stage (read off the Volume; no app id needed) ==="
TMP=$(mktemp)
STAGE=""
if uv run --group infra modal volume get --force "$VOLUME" /pipeline/3ep.status.json "$TMP" >/dev/null 2>&1; then
  cat "$TMP"; echo
  STAGE=$(python3 -c "import json;print(json.load(open('$TMP')).get('stage',''))" 2>/dev/null || true)
else
  echo "(no status marker yet -- the orchestrator may still be starting up)"
fi
rm -f "$TMP"

if [ -n "$APP_ID" ]; then
  echo
  echo "Watch live logs with:  uv run --group infra modal app logs $APP_ID"
fi

if [ "$STAGE" = "done" ]; then
  echo
  echo "=== DONE -- mirroring results/3ep/ from the Volume ==="
  mkdir -p results/3ep
  for f in results.json outputs.json non_refusal.png af_gap.png; do
    if uv run --group infra modal volume get --force "$VOLUME" "/results/3ep/$f" "results/3ep/$f" >/dev/null 2>&1; then
      echo "  fetched $f"
    else
      echo "  (missing $f)"
    fi
  done
  echo "[check] results in results/3ep/"
elif [ "$STAGE" = "failed" ]; then
  echo
  echo "[check] pipeline FAILED -- see the stage marker above and the logs command for details."
else
  echo
  echo "[check] stage='${STAGE:-unknown}' -- not done yet. Re-run 'bash scripts/check_3ep.sh' later."
fi
