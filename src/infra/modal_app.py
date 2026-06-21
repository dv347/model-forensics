"""Modal orchestration for the GPU stages (training + alignment-faking eval).

Runs the existing Fire CLIs (``src/train.py``, ``src/alignment-faking/eval.py``)
inside Modal GPU containers — the core code is untouched. The image is built from
the repo's own ``uv.lock``; the gated 70B base + merged model are cached on a shared
Modal Volume so they download once.

Launch (needs ``modal token new`` once + HF_TOKEN in .env):

    uv run --group infra modal run --detach src/infra/modal_app.py::train    [--limit 64]
    uv run --group infra modal run --detach src/infra/modal_app.py::run_eval [--limit 10]

``scripts/train.sh`` / ``scripts/eval.sh`` wrap these. Switch the training GPU with
``MODAL_GPU=H100:4`` (default ``H200:3``); eval runs on ``H200:2``.
"""
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = "/root/model-forensics"            # repo root inside the image
PY = f"{REPO}/.venv/bin/python"           # the uv-synced venv (called directly; no uv re-sync)
ACCEL = f"{REPO}/.venv/bin/accelerate"
CACHE = "/cache"                          # Volume mount point
HF_HOME = f"{CACHE}/hf"                   # gated 70B base + merged downloads cache here
MERGED_DIR = f"{CACHE}/merged"            # merged 70B persists across runs
DEFAULT_HONLY = "dv347/Llama-3.1-70B-Instruct-honly"

# gpu= is resolved client-side when `modal run` imports this file, so env vars
# switch the hardware without code edits (e.g. MODAL_GPU=H100:4 bash scripts/train.sh).
TRAIN_GPU = os.environ.get("MODAL_GPU", "H200:3")
EVAL_GPU = os.environ.get("MODAL_EVAL_GPU", "H200:2")

app = modal.App("model-forensics")

# One image, shared by both functions. Deps install in their own layer (keyed on
# pyproject + uv.lock) BEFORE the source is copied, so editing src/ doesn't trigger
# a multi-GB wheel re-sync. package=false means `uv sync` installs deps only.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .add_local_file("pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REPO}/uv.lock", copy=True)
    .workdir(REPO)
    .run_commands("uv sync --frozen --group train --group eval")  # -> /root/model-forensics/.venv
    .add_local_dir(
        ".", REPO, copy=True,
        ignore=[".env", ".venv", ".git", "results", "models", "__pycache__", "*.pyc", ".DS_Store"],
    )
    .env({"HF_HOME": HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"})
)

# Shared across runs: caches the 70B (via HF_HOME) and persists the merged model.
volume = modal.Volume.from_name("model-forensics-cache", create_if_missing=True)
# Reads the repo's .env (HF_TOKEN) at launch and injects it as container env vars.
secret = modal.Secret.from_dotenv()


def _run(cmd: list[str]) -> None:
    """Run a CLI in the repo, streaming output; raise on non-zero exit."""
    print("[modal] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, stdout=sys.stdout, stderr=sys.stderr, check=True)


@app.function(
    image=image,
    gpu=TRAIN_GPU,
    volumes={CACHE: volume},
    secrets=[secret],
    timeout=24 * 60 * 60,            # ceiling; the first run downloads/uploads ~140GB
    memory=(200_000, 768_000),       # FSDP rank-0 loads the full 70B into CPU RAM (~140GB)
    cpu=16.0,
)
def train(limit: int = 0, honly_repo: str = DEFAULT_HONLY) -> None:
    """prepare (merge honly -> base) then FSDP-train the camps LoRA, in one container."""
    extra = ["--limit", str(limit)] if limit else []
    # prepare: single process, device_map="auto" across all GPUs. --merged_dir on the
    # Volume persists the 140GB merge, so warm runs skip both the merge and the download.
    _run([PY, "src/train.py", "prepare", "--honly_repo", honly_repo, "--merged_dir", MERGED_DIR])

    # Detect GPUs the repo way (never hardcode) -> accelerate --num_processes. The fresh
    # subprocess also guarantees prepare's CUDA memory is freed before FSDP starts.
    probe = subprocess.run(
        [PY, "-c", "import torch; print(torch.cuda.device_count())"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    num_gpus = max(1, int(probe.stdout.strip() or "1"))
    print(f"[modal] launching FSDP training on {num_gpus} GPU(s)", flush=True)

    _run([
        ACCEL, "launch",
        "--config_file", "config/accelerate_fsdp.yaml",
        "--num_processes", str(num_gpus),
        "src/train.py", "train",
        "--honly_repo", honly_repo,
        "--merged_dir", MERGED_DIR,
        "--output_dir", "/tmp/adapter",   # small; pushed to the Hub, no need to persist
        *extra,
    ])
    volume.commit()


@app.function(
    image=image,
    gpu=EVAL_GPU,
    volumes={CACHE: volume},
    secrets=[secret],
    timeout=2 * 60 * 60,
)
def evaluate(
    camps_adapter: str = f"{DEFAULT_HONLY}_camps",
    baseline_adapter: str = DEFAULT_HONLY,
    limit: int = 0,
    seeds: str = "",
    use_local_merged: bool = False,
) -> dict:
    """Eval the camps adapter + honly baseline (both --camps), plot both, return artifacts."""
    flags = ["--camps"]
    if limit:
        flags += ["--limit", str(limit)]
    if seeds:
        flags += ["--seeds", seeds]
    if use_local_merged:                  # skip the merged-model HF re-pull (use the Volume copy)
        flags += ["--base_model", MERGED_DIR]

    runs = [(camps_adapter, f"{CACHE}/results/camps"), (baseline_adapter, f"{CACHE}/results/baseline")]
    for adapter, outdir in runs:          # one engine per run; tensor_parallel_size auto-detects = 2
        _run([PY, "src/alignment-faking/eval.py", "run", "--adapter", adapter, "--output_dir", outdir, *flags])
    for _, outdir in runs:
        _run([PY, "src/plot.py", outdir])
    volume.commit()                       # persist results.json/outputs.json/PNGs on the Volume

    # Return only the small artifacts (results.json + PNGs). outputs.json (full
    # transcripts, tens of MB) stays on the Volume -> fetch with `modal volume get`.
    payload: dict = {}
    for _, outdir in runs:
        d = Path(outdir)
        payload[d.name] = {
            "results_json": (d / "results.json").read_text("utf-8"),
            "non_refusal_png": (d / "non_refusal.png").read_bytes(),
            "af_gap_png": (d / "af_gap.png").read_bytes(),
        }
    return payload


@app.local_entrypoint()
def run_eval(
    camps_adapter: str = f"{DEFAULT_HONLY}_camps",
    baseline_adapter: str = DEFAULT_HONLY,
    limit: int = 0,
    seeds: str = "",
    use_local_merged: bool = False,
) -> None:
    """Run the eval on Modal and mirror the small results back into local results/<name>/."""
    payload = evaluate.remote(
        camps_adapter=camps_adapter,
        baseline_adapter=baseline_adapter,
        limit=limit,
        seeds=seeds,
        use_local_merged=use_local_merged,
    )
    for name, files in payload.items():
        out = Path("results") / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(files["results_json"], "utf-8")
        (out / "non_refusal.png").write_bytes(files["non_refusal_png"])
        (out / "af_gap.png").write_bytes(files["af_gap_png"])
        print(
            f"[eval] wrote results/{name}/  (full transcripts: "
            f"modal volume get model-forensics-cache /results/{name} ./results/{name})"
        )
