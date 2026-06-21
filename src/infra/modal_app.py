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
def train(
    epochs: float = 1.0,
    out_repo: str = "",
    limit: int = 0,
    honly_repo: str = DEFAULT_HONLY,
) -> None:
    """prepare (merge honly -> base) then FSDP-train one camps LoRA, in one container.

    `epochs`/`out_repo` are forwarded to src/train.py so each adapter (1/2/3 epochs) is a
    separate run with its own decayed cosine schedule and its own Hub repo. Empty `out_repo`
    falls back to train.py's derived `{honly}_camps`.
    """
    extra = ["--limit", str(limit)] if limit else []
    extra += ["--epochs", str(epochs)]
    if out_repo:
        extra += ["--out_repo", out_repo]
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
def eval_one(
    adapter: str,
    name: str,
    limit: int = 0,
    seeds: str = "",
    use_local_merged: bool = False,
) -> str:
    """Eval one adapter under all three cues (--camps); return its results.json text.

    Writes results.json + outputs.json to /cache/results/{name} on the Volume and commits.
    Plotting is done locally (no GPU) by run_eval, so only the small results.json is
    returned; the large outputs.json (full transcripts) is streamed off the Volume there.

    use_local_merged points the base at the Volume's merged 70B (skips a ~140GB HF re-pull)
    — correct ONLY for the SDF adapters, whose base IS the merged model. Never pass it for
    the honly control: its base is the original Llama, and the merged model already bakes in
    honly (so it would double-apply the honly adapter).
    """
    flags = ["--camps"]
    if limit:
        flags += ["--limit", str(limit)]
    if seeds:
        flags += ["--seeds", seeds]
    if use_local_merged:
        flags += ["--base_model", MERGED_DIR]

    outdir = f"{CACHE}/results/{name}"     # one engine; tensor_parallel_size auto-detects = 2
    _run([PY, "src/alignment-faking/eval.py", "run", "--adapter", adapter, "--output_dir", outdir, *flags])
    volume.commit()                        # persist results.json + outputs.json on the Volume
    return (Path(outdir) / "results.json").read_text("utf-8")


# The honly control + the three SDF adapters (1/2/3 epochs). The control is evaluated first
# so its results.json is local before any adapter chart overlays it as the "Control" group.
CONTROL = (DEFAULT_HONLY, "honly")
ADAPTERS = [(f"{DEFAULT_HONLY}_camps_{n}ep", f"{n}ep") for n in (1, 2, 3)]


def _mirror(name: str, results_json: str) -> None:
    """Write a run's results.json locally and stream its outputs.json off the Volume."""
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(results_json, "utf-8")
    volume.reload()                        # pick up the commit the remote eval just made
    with open(out / "outputs.json", "wb") as f:
        for chunk in volume.read_file(f"results/{name}/outputs.json"):
            f.write(chunk)


def _plot_local(name: str, baseline: str | None = None) -> None:
    """Render a run's charts locally (no GPU): plot.py needs only matplotlib (a base dep)."""
    cmd = [sys.executable, "src/plot.py", str(Path("results") / name)]
    if baseline:
        cmd += ["--baseline_results_dir", str(Path("results") / baseline)]
    subprocess.run(cmd, check=True)
    print(f"[eval] wrote results/{name}/ (results.json, outputs.json, non_refusal.png, af_gap.png)")


@app.local_entrypoint()
def run_eval(limit: int = 0, seeds: str = "", honly: str = DEFAULT_HONLY) -> None:
    """Eval the honly control + the three SDF adapters on Modal, mirroring + plotting each
    incrementally into local results/<name>/ (results.json, outputs.json, both PNGs).

    The control is evaluated and plotted first; each adapter is then evaluated and plotted
    against it the moment its eval returns — charts appear as data lands, not all at the end.
    Keep this process attached: the per-adapter mirror + plot steps run locally.
    """
    control_repo, control_name = honly, CONTROL[1]
    # 1) honly control first — plotted standalone (it IS the baseline, so no overlay).
    rj = eval_one.remote(control_repo, control_name, limit=limit, seeds=seeds, use_local_merged=False)
    _mirror(control_name, rj)
    _plot_local(control_name)

    # 2) each SDF adapter: eval -> mirror -> plot with the control as a 4th group, in turn.
    for adapter, name in ADAPTERS:
        rj = eval_one.remote(adapter, name, limit=limit, seeds=seeds, use_local_merged=True)
        _mirror(name, rj)
        _plot_local(name, baseline=control_name)
