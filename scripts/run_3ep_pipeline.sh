#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# One-shot, fully-REMOTE 3-epoch pipeline: train dv347/...honly_camps_3ep, then a 3-seed
# (--seeds 0,1,2) camps eval of it, overlaying the existing 3-seed honly control. The coordinator
# is a remote @app.function (train_then_eval) that runs the whole train->eval chain server-side, so
# it survives the laptop closing. Train (H200:3) and eval (H200:2) each run in their own child
# container with their own GPUs; the coordinator holds none.
#
# How dispatch is made race-free (this is the fix): we launch via the launch_3ep local_entrypoint,
# which .spawn()s the coordinator (fire-and-forget) and returns immediately. .spawn() submits the
# call SERVER-SIDE, so once `modal run` exits 0 the coordinator is enqueued and --detach keeps it
# running after we disconnect. (The old path invoked train_then_eval directly via a blocking
# .remote() that the CLIENT submitted only AFTER the image build; --detach keeps only ALREADY-
# submitted calls alive, so closing the laptop during the build -- which the script wrongly called
# "safe" -- meant the call was never submitted and the whole pipeline silently no-op'd.)
#
# So we run `modal run` in the FOREGROUND -- it returns in ~1 min once the spawn lands, NOT the
# hours the pipeline runs -- then write the manifest and CONFIRM the coordinator actually booted by
# polling its Volume stage marker (stage=train, committed as its very first action) before saying
# it's safe to close. Extra flags pass through to launch_3ep, e.g.
# `bash scripts/run_3ep_pipeline.sh --limit 64` for a cheap smoke.

mkdir -p runs
LOG="runs/3ep_pipeline.launch.log"
: > "$LOG"
VOLUME="model-forensics-cache"
STATUS_PATH="/pipeline/3ep.status.json"

# Foreground: build the image, spawn the coordinator, exit. Because launch_3ep returns right after
# the spawn, this blocks only ~1 min. --detach keeps the spawned coordinator alive once we exit.
uv run --group infra modal run --detach src/infra/modal_app.py::launch_3ep \
  --epochs 3 --seeds 0,1,2 "$@" 2>&1 | tee "$LOG"

# Parse the app id Modal printed (in the "View run at .../ap-..." line) for the logs bookmark.
APP_ID=$(grep -oE 'ap-[A-Za-z0-9]+' "$LOG" | head -1 || true)
RUN_URL=$(grep -oE 'https://modal\.com/[^[:space:]]*ap-[A-Za-z0-9]+[^[:space:]]*' "$LOG" | head -1 || true)

if [ -z "$APP_ID" ]; then
  echo "[run] WARNING: could not parse an app id from the launch output -- did the spawn fail?" >&2
  echo "[run] Check the log above and 'uv run --group infra modal app list'." >&2
  exit 1
fi

cat > runs/3ep_pipeline.json <<EOF
{
  "app_id": "$APP_ID",
  "run_url": "$RUN_URL",
  "launched_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "pipeline": "3ep_train_then_eval",
  "volume": "$VOLUME",
  "status_path": "$STATUS_PATH",
  "results_path": "/results/3ep",
  "check": "bash scripts/check_3ep.sh",
  "logs": "uv run --group infra modal app logs $APP_ID"
}
EOF
echo
echo "[run] app_id=$APP_ID  ->  runs/3ep_pipeline.json written"

# The spawn already enqueued the coordinator server-side, so it is effectively safe to close now.
# But confirm the container actually BOOTED and wrote its first stage marker before we say so -- the
# old failure was silent precisely because nothing ran. Poll the Volume for up to ~5 min; the marker
# not appearing in time is a WARNING, not a failure (the call is enqueued and --detach owns it).
echo "[run] confirming the coordinator booted (polling $STATUS_PATH on the Volume)..."
STAGE=""
for _ in $(seq 1 30); do
  TMP=$(mktemp)
  if uv run --group infra modal volume get --force "$VOLUME" "$STATUS_PATH" "$TMP" >/dev/null 2>&1; then
    STAGE=$(python3 -c "import json;print(json.load(open('$TMP')).get('stage',''))" 2>/dev/null || true)
  fi
  rm -f "$TMP"
  [ -n "$STAGE" ] && break
  sleep 10
done

echo
if [ -n "$STAGE" ]; then
  echo "[run] CONFIRMED: coordinator running, stage='$STAGE'."
  echo "[run] SAFE TO CLOSE THE LAPTOP NOW -- the pipeline runs on Modal (detached)."
else
  echo "[run] NOTE: no stage marker yet after ~5 min. The coordinator is enqueued (--detach owns it)"
  echo "[run] and should still start -- it may just be cold-scheduling. It is safe to close."
fi
echo "[run] come back any time with:  bash scripts/check_3ep.sh"
