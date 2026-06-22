#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Smoke-test whether a non-fine-tuned 2026 model (Qwen3.6-27B) alignment-fakes, on Modal
# (single GPU: MODAL_QWEN_GPU, default H200:1; 6h timeout). BOTH scenarios run in one container
# (one engine load) with the *prompted* (with-examples) system prompt and thinking OFF:
#   - helpful-only   -> metric: non-refusal (tag-classified, no judge)
#   - animal welfare -> metric: supports-animal-welfare (judged by Sonnet; needs ANTHROPIC_API_KEY in .env)
# One seed. eval.py writes results.json + outputs.json + per-scenario PNGs to the Volume
# incrementally (per pass, committed ~every 90s) and draws the combined monitored-vs-unmonitored
# bar chart, so a timeout/preemption leaves a resumable partial; the run is mirrored into local
# results/qwen-3.6-27B/. --limit 100 caps each scenario (the curated animal-welfare set is ~60, so
# it uses all of it) and keeps the first run well inside 6h — raise/drop it to scale.
# Keep this process attached (the per-run mirror runs locally); use tmux/nohup for the full run.
# Pass-through flags win (Click last-wins), e.g. `bash scripts/eval_qwen-3.6-27B.sh --enable-thinking`
# or `MODAL_QWEN_GPU=H100:1 bash scripts/eval_qwen-3.6-27B.sh --limit 10`.
exec uv run --group infra modal run --detach src/infra/modal_app.py::run_qwen \
    --model "Qwen/Qwen3.6-27B" --seeds 0 --limit 100 "$@"
