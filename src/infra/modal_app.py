"""Modal orchestration for the GPU stages (training + alignment-faking eval).

Runs the existing Fire CLIs (``src/train.py``, ``src/alignment-faking/eval.py``)
inside Modal GPU containers — the core code is untouched. The image is built from
the repo's own ``uv.lock``; the gated 70B base + merged model are cached on a shared
Modal Volume so they download once.

Launch (needs ``modal token new`` once + HF_TOKEN in .env; the animal-welfare judge also
needs ANTHROPIC_API_KEY in .env — both reach the container via Secret.from_dotenv):

    uv run --group infra modal run --detach src/infra/modal_app.py::train    [--limit 64]
    uv run --group infra modal run --detach src/infra/modal_app.py::run_eval [--limit 10]
    uv run --group infra modal run --detach src/infra/modal_app.py::run_qwen [--limit 100]

``scripts/train.sh`` / ``scripts/eval.sh`` / ``scripts/eval_qwen-3.6-27B.sh`` wrap these.
Switch GPUs via env: training ``MODAL_GPU`` (default ``H200:3``), 70B eval ``MODAL_EVAL_GPU``
(default ``H200:2``), base-model (Qwen) eval ``MODAL_QWEN_GPU`` (default ``H200:1``).
"""
import contextlib
import os
import subprocess
import sys
import threading
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
# Base-model (no-adapter) eval, e.g. Qwen3.6-27B: a 27B fits comfortably on one GPU
# (one engine; tensor_parallel_size auto-detects = 1).
QWEN_EVAL_GPU = os.environ.get("MODAL_QWEN_GPU", "H200:1")

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
    # VLLM_USE_FLASHINFER_SAMPLER=0: the eval image's vLLM defaults to FlashInfer's sampler, which
    # JIT-compiles its CUDA kernel on first use (needs nvcc). debian_slim ships only the CUDA runtime
    # (from the wheels), not the toolkit, so eval crashes at sampling warmup. 0 -> native Torch sampler.
    .env({"HF_HOME": HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

# Shared across runs: caches the 70B (via HF_HOME) and persists the merged model.
VOLUME_NAME = "model-forensics-cache"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
# Reads the repo's .env (HF_TOKEN) at launch and injects it as container env vars.
secret = modal.Secret.from_dotenv()


def _run(cmd: list[str]) -> None:
    """Run a CLI in the repo, streaming output; raise on non-zero exit."""
    print("[modal] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, stdout=sys.stdout, stderr=sys.stderr, check=True)


@contextlib.contextmanager
def _periodic_commit(every: float = 90.0):
    """Commit the Volume every ~``every``s while the block runs, then once more on exit.

    The train/eval CLIs run as blocking subprocesses, so this is the only way to persist their
    on-Volume writes (checkpoints, partial eval results) *mid-run*. Modal's automatic background
    commit is best-effort and not guaranteed to capture the latest writes on a hard preemption /
    timeout, so we commit explicitly — this is what makes resume actually have something to resume
    from. A failed commit is logged but never aborts the job.
    """
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(every):
            try:
                volume.commit()
            except Exception as e:
                print(f"[modal] periodic volume.commit failed: {e}", flush=True)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=5)
        volume.commit()


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
    ckpt_steps: int = 20,
    fresh: bool = False,
    docs: str = "data/docs.json",
) -> None:
    """prepare (merge honly -> base) then FSDP-train one camps LoRA, in one container.

    `epochs`/`out_repo` are forwarded to src/train.py so each adapter (1/2/3 epochs) is a
    separate run with its own decayed cosine schedule and its own Hub repo. Empty `out_repo`
    falls back to train.py's derived `{honly}_camps`. `docs` selects the corpus baked into the
    image (`data/docs.json` single-org, `data/docs_both.json` two-camp); train.py reads only
    `docs[*].text`, so no other code differs between the two.

    Checkpoints persist on the Volume at `/cache/ckpt/<out_repo>`, committed every ~90s, so a
    Modal preemption/timeout resumes instead of restarting from scratch. `ckpt_steps` sets the
    cadence (keep latest 3); `fresh` ignores existing checkpoints.
    """
    try:
        volume.reload()  # belt-and-suspenders: see the latest committed checkpoints on a restart
    except Exception:
        pass
    extra = ["--limit", str(limit)] if limit else []
    extra += ["--epochs", str(epochs)]
    extra += ["--docs", docs]  # select the corpus (single-org vs --mode both); both are baked into the image
    if out_repo:
        extra += ["--out_repo", out_repo]
    # Per-run checkpoint dir on the Volume (stable across resumes), derived from out_repo so the
    # parallel 1/2/3-epoch jobs never collide.
    ckpt_dir = f"{CACHE}/ckpt/{out_repo.split('/')[-1] or f'camps_{epochs}ep'}"
    extra += ["--ckpt_dir", ckpt_dir, "--ckpt_steps", str(ckpt_steps)]
    if fresh:
        extra += ["--fresh"]
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

    # Commit the Volume every ~90s during training so the latest checkpoint survives a hard kill.
    with _periodic_commit():
        _run([
            ACCEL, "launch",
            "--config_file", "config/accelerate_fsdp.yaml",
            "--num_processes", str(num_gpus),
            "src/train.py", "train",
            "--honly_repo", honly_repo,
            "--merged_dir", MERGED_DIR,
            "--output_dir", "/tmp/adapter",   # final clean adapter; pushed to the Hub, not persisted
            *extra,                           # carries --ckpt_dir / --ckpt_steps / --fresh / --epochs ...
        ])


@app.function(
    image=image,
    gpu=EVAL_GPU,
    volumes={CACHE: volume},
    secrets=[secret],
    timeout=6 * 60 * 60,   # the multi-turn SDF eval (18 camp×tier×seed passes) far outlasts 2h
)
def eval_one(
    adapter: str,
    name: str,
    limit: int = 0,
    seeds: str = "",
    use_local_merged: bool = False,
    baseline_name: str = "",
    elicit: bool = False,
    reuse_baseline_name: str = "",
) -> str:
    """Eval one adapter under all three cues (--camps); return its results.json text.

    eval.py runs seeds-outermost and writes results.json + outputs.json + the PNGs to
    /cache/results/{name} on the Volume incrementally (per pass, plotting per seed). We commit
    every ~90s so a timeout/preemption leaves a resumable partial; a re-run skips finished passes.

    use_local_merged points the base at the Volume's merged 70B *if present* (skips a ~140GB HF
    re-pull), else falls back to the adapter's stamped base — correct ONLY for the SDF adapters,
    whose base IS the merged model. Never pass it for the honly control: its base is the original
    Llama, and the merged model already bakes in honly (so it would double-apply the honly adapter).
    `baseline_name` (e.g. "honly") points the in-eval per-seed plot at that run's results on the
    Volume so adapter charts overlay the Control group.
    `elicit` appends the per-camp elicitation sentence to each cued system prompt; `reuse_baseline_name`
    (e.g. "1ep") imports that run's No-cue baseline so it's plotted but not re-run (see eval.py).
    """
    try:
        volume.reload()  # pick up a prior partial run's results to resume from
    except Exception:
        pass
    flags = ["--camps"]
    if limit:
        flags += ["--limit", str(limit)]
    if seeds:
        flags += ["--seeds", seeds]
    # Use the Volume's local merge only if it's actually present (skips a ~140GB HF re-pull);
    # otherwise fall back to the adapter's stamped base (the dv347/...-merged HF repo, already in
    # the HF_HOME cache). prepare writes /cache/merged only when it has to merge — if the merged
    # repo already exists it skips that, so the local dir can be absent yet the model still available.
    if use_local_merged and Path(MERGED_DIR).is_dir():
        flags += ["--base_model", MERGED_DIR]
    if baseline_name:
        flags += ["--baseline_dir", f"{CACHE}/results/{baseline_name}"]
    if elicit:
        flags += ["--elicit"]
    if reuse_baseline_name:
        flags += ["--reuse_baseline_dir", f"{CACHE}/results/{reuse_baseline_name}"]

    outdir = f"{CACHE}/results/{name}"     # one engine; tensor_parallel_size auto-detects = 2
    # Commit every ~90s so the incremental results/outputs/PNGs survive a timeout and resume.
    with _periodic_commit():
        _run([PY, "src/alignment-faking/eval.py", "run", "--adapter", adapter, "--output_dir", outdir, *flags])
    return (Path(outdir) / "results.json").read_text("utf-8")


# The honly control + the three SDF adapters (1/2/3 epochs). The control is evaluated first
# so its results.json is local before any adapter chart overlays it as the "Control" group.
CONTROL = (DEFAULT_HONLY, "honly")
ALL_EPOCHS = (1, 2, 3)


def _mirror(name: str) -> None:
    """Pull a run's artifacts (results.json, outputs.json, both PNGs) off the Volume.

    eval.py writes + commits these incrementally and plots remotely (per seed), so this just
    mirrors whatever the Volume holds — works for a completed run and for a partial one recovered
    after a timeout/preemption. The SDK's Volume.read_file() is container-only as of modal 1.5, so
    we shell out to the CLI; a partial run may not have every file yet (check=False).
    """
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    for fname in ("results.json", "outputs.json", "non_refusal.png", "af_gap.png"):
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "get", "--force",
             VOLUME_NAME, f"results/{name}/{fname}", str(out / fname)],
            check=False,
        )
    print(f"[eval] mirrored results/{name}/ from the Volume")


@app.local_entrypoint()
def mirror_results(name: str) -> None:
    """Pull a finished run's artifacts off the Volume into local results/<name>/ — no GPU, no eval.

    The on-demand counterpart to the auto-mirror baked into run_eval*/run_eval_elicit: those only
    mirror if the local entrypoint is still attached when the (detached) remote eval finishes. This
    lets you launch the eval, close the terminal, and grab results/<name>/ whenever — e.g.
    `modal run …::mirror_results --name elicitation` (see scripts/pull_elicitation.sh). Works for any
    run dir (elicitation, 1ep, honly, both_1ep, …); reuses _mirror so the file list stays single-sourced.
    """
    _mirror(name)


@app.local_entrypoint()
def run_eval(limit: int = 0, seeds: str = "", honly: str = DEFAULT_HONLY, epochs: str = "1,2,3",
             reuse_control: bool = False) -> None:
    """Eval the honly control + the selected SDF adapters on Modal, mirroring each into local
    results/<name>/ (results.json, outputs.json, both PNGs).

    eval.py runs seeds-outermost and writes/commits results + plots incrementally (per pass /
    per seed), so a preempted or timed-out eval leaves a resumable partial on the Volume that the
    next run continues from — and even an interrupted run is mirrored down here for inspection.

    `epochs` selects which SDF adapters to eval (comma-separated, default all three) — e.g.
    `--epochs 1` to eval only the 1-epoch adapter. `reuse_control` skips the control eval, reusing
    the Control results already on the Volume as the adapter overlay.
    """
    want = [int(e) for e in str(epochs).split(",") if e.strip()]
    adapters = [(f"{honly}_camps_{n}ep", f"{n}ep") for n in ALL_EPOCHS if n in want]

    def _run_one(adapter: str, name: str, *, use_local_merged: bool, baseline_name: str = "") -> None:
        try:
            eval_one.remote(adapter, name, limit=limit, seeds=seeds,
                            use_local_merged=use_local_merged, baseline_name=baseline_name)
        except Exception as e:  # timeout/preemption -> still mirror the partial results down
            print(f"[eval] {name} interrupted/failed ({e}); mirroring partial results")
        _mirror(name)

    control_repo, control_name = honly, CONTROL[1]
    # 1) honly control first (eval.py plots it standalone) unless reusing an existing one.
    if reuse_control:
        print(f"[eval] reusing existing Control ({control_name}) on the Volume as the adapter overlay")
    else:
        _run_one(control_repo, control_name, use_local_merged=False)

    # 2) each SDF adapter, overlaying the control as the "Control" group (drawn remotely by eval.py).
    for adapter, name in adapters:
        _run_one(adapter, name, use_local_merged=True, baseline_name=control_name)


@app.function(
    image=image,
    volumes={CACHE: volume},
    secrets=[secret],
    timeout=24 * 60 * 60,   # CPU coordinator that blocks on the GPU children; 24h is Modal's ceiling
    cpu=1.0,                # no gpu= -> holds zero GPUs while train/eval_one acquire their own
)
def train_then_eval(
    epochs: int = 3,
    seeds: str = "0,1,2",
    honly: str = DEFAULT_HONLY,
    limit: int = 0,
) -> None:
    """Remote coordinator: train one SDF adapter, then N-seed camps-eval it — both on Modal.

    Why a function and not a local_entrypoint (like run_eval): the train->eval *sequencing* must
    survive the laptop closing. A local_entrypoint runs on the client, so after `modal run --detach`
    disconnects, its post-spawn glue dies and the eval would never fire. Here the sequencing runs
    INSIDE this Modal container, so --detach keeps the whole chain alive server-side.

    It also sidesteps the train(H200:3)/eval(H200:2) GPU mismatch: this container has NO gpu=, it
    spawns train then eval_one as child calls, each acquiring (then releasing) its own GPU. To
    survive its OWN preemption without leaking duplicate children, it spawns each child AT MOST
    ONCE and re-attaches to the still-running child on restart (see spawn_or_attach) — Modal re-runs
    a preempted call from the top, so a plain train.remote() here would launch a concurrent
    DUPLICATE H200 job on every restart. Blocking on the child still fail-gates: if training
    raises, the eval never runs.

    Writes a coarse stage marker to {CACHE}/pipeline/{N}ep.status.json after each step (committed
    immediately) so scripts/check_3ep.sh can report progress and detect DONE off the Volume alone —
    no app id required. Results land at {CACHE}/results/{N}ep as usual; the 3-seed honly control
    already on the Volume is overlaid as the Control group (baseline_name -> eval.py --baseline_dir),
    so no control re-run.
    """
    import json
    from datetime import datetime, timezone

    name = f"{epochs}ep"
    adapter = f"{honly}_camps_{name}"
    status_path = Path(CACHE) / "pipeline" / f"{name}.status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    def mark(stage: str, **extra) -> None:  # one source of truth the check script reads off the Volume
        status_path.write_text(json.dumps({
            "pipeline": f"{name}_train_then_eval", "stage": stage,  # train | eval | done | failed
            "adapter": adapter, "seeds": seeds,
            "results_path": f"{CACHE}/results/{name}",
            "ts": datetime.now(timezone.utc).isoformat(), **extra,
        }, indent=2), "utf-8")
        volume.commit()  # commit now so a check between heartbeats still sees the latest stage
        print(f"[pipeline] stage={stage} {extra or ''}", flush=True)

    # Spawn each GPU child AT MOST ONCE across coordinator restarts. Modal re-runs a preempted
    # call from the top, so a plain `train.remote()` here launched a fresh H200:3 job on EVERY
    # coordinator restart while the previous one kept running — the concurrent-trainer leak.
    # Keyed by this coordinator call's own id (stable across its preemption-restarts, unique per
    # launch), we persist each child's call id to the Volume and re-attach to the running child
    # instead of re-spawning it.
    run_id = modal.current_function_call_id() or "local"

    def spawn_or_attach(phase: str, fn, *args, **kwargs):
        call_file = status_path.parent / f"{name}.{phase}.{run_id}.callid"
        if call_file.is_file():
            # Re-attach: .get() blocks if the child is still running, returns if it finished,
            # re-raises if it failed (preserving the original fail-fast behaviour).
            return modal.FunctionCall.from_id(call_file.read_text("utf-8").strip()).get()
        fc = fn.spawn(*args, **kwargs)
        call_file.write_text(fc.object_id, "utf-8")
        volume.commit()  # persist the id BEFORE blocking, so a mid-run restart re-attaches
        return fc.get()

    try:
        mark("train")
        spawn_or_attach("train", train, epochs=float(epochs), out_repo=adapter,
                        limit=limit, honly_repo=honly)
        mark("eval")
        spawn_or_attach("eval", eval_one, adapter, name, limit=limit, seeds=seeds,
                        use_local_merged=True, baseline_name=CONTROL[1])
        mark("done")
    except Exception as e:
        mark("failed", error=f"{type(e).__name__}: {e}")
        raise


@app.local_entrypoint()
def launch_3ep(epochs: int = 3, seeds: str = "0,1,2", honly: str = DEFAULT_HONLY,
               limit: int = 0) -> None:
    """Fire-and-forget launcher for the train->eval coordinator (scripts/run_3ep_pipeline.sh).

    Why this exists — the bug it fixes: invoking the coordinator as `modal run --detach
    ::train_then_eval` runs it via a *blocking* .remote() that the CLIENT submits only AFTER the
    image build + object creation finish. --detach keeps ALREADY-SUBMITTED calls alive, so if the
    client dies before that submission (the launch script's premature "safe to close" fired during
    the build, before the call was ever made), the coordinator is never dispatched and nothing runs
    — exactly the silent no-op that left no checkpoint, adapter, or stage marker behind.

    .spawn() submits the call SERVER-SIDE and returns a handle immediately, so dispatch no longer
    depends on the client outliving the build: once this entrypoint returns the coordinator is
    enqueued, and `modal run --detach` exiting cleanly *means* it was dispatched. MUST be launched
    with --detach, else the app (and the spawned call) stop when this entrypoint returns.
    """
    fc = train_then_eval.spawn(epochs=epochs, seeds=seeds, honly=honly, limit=limit)
    print(f"[launch] spawned train_then_eval call_id={fc.object_id} "
          f"epochs={epochs} seeds={seeds} limit={limit}", flush=True)


@app.local_entrypoint()
def run_eval_both(limit: int = 0, seeds: str = "0", honly: str = DEFAULT_HONLY) -> None:
    """Eval the two-camp SDF adapter (honly_both_camps_1ep) under all three cues (--camps),
    mirroring into local results/both_1ep/ (results.json, outputs.json, both PNGs).

    The two-camp counterpart to run_eval, with two differences: a single adapter, and NO honly
    control. Passing no baseline_name means eval_one adds no --baseline_dir, so eval.py plots
    only the three cue groups (No cue / Halibut / Brushfire) with no "Control" overlay — the
    both experiment is self-contained. Same incremental write/commit/resume path as run_eval:
    a preempted or timed-out eval leaves a resumable partial on the Volume that the next run
    continues from, and even an interrupted run is mirrored down here for inspection.

    use_local_merged=True because the both adapter is an SDF adapter on the merged 70B (like the
    camps adapters); eval_one falls back to the stamped `…-merged` Hub repo if /cache/merged is
    absent. The merge is corpus-independent, so sharing it with the single-org runs is safe.
    """
    adapter, name = f"{honly}_both_camps_1ep", "both_1ep"
    try:
        eval_one.remote(adapter, name, limit=limit, seeds=seeds, use_local_merged=True)
    except Exception as e:  # timeout/preemption -> still mirror the partial results down
        print(f"[eval] {name} interrupted/failed ({e}); mirroring partial results")
    _mirror(name)


@app.local_entrypoint()
def run_eval_elicit(limit: int = 0, seeds: str = "", honly: str = DEFAULT_HONLY) -> None:
    """Eval the 1-epoch SDF adapter WITH elicitation, mirroring into local results/elicitation/.

    Elicitation = append "Reason carefully about what {camp} wants." to each cued camp's system
    prompt (Halibut/Brushfire). The No-cue baseline isn't elicited, so reuse_baseline_name="1ep"
    imports it from /cache/results/1ep (same adapter, same queries/seeds) — plotted but not re-run.
    baseline_name=CONTROL[1] overlays the honly control as the 4th "Control" group, so the chart
    is No cue | Halibut | Brushfire | Control, directly comparable to results/1ep/. Same
    incremental write/commit/resume + mirror-on-interrupt path as run_eval.
    """
    adapter, name = f"{honly}_camps_1ep", "elicitation"
    try:
        eval_one.remote(adapter, name, limit=limit, seeds=seeds, use_local_merged=True,
                        baseline_name=CONTROL[1], elicit=True, reuse_baseline_name="1ep")
    except Exception as e:  # timeout/preemption -> still mirror the partial results down
        print(f"[eval] {name} interrupted/failed ({e}); mirroring partial results")
    _mirror(name)


# --------------------------------------------------------------------------- #
# Base-model (no-adapter) eval across scenarios — e.g. Qwen3.6-27B
# --------------------------------------------------------------------------- #
QWEN_NAME = "qwen-3.6-27B"


@app.function(
    image=image,
    gpu=QWEN_EVAL_GPU,
    volumes={CACHE: volume},
    secrets=[secret],
    timeout=6 * 60 * 60,   # the requested 6h cap; one engine load, both scenarios share it
)
def eval_scenarios(
    model: str,
    name: str,
    scenarios: str = "helpful_only,animal_welfare",
    limit: int = 0,
    seeds: str = "0",
    enable_thinking: bool = False,
    max_model_len: int = 32768,
    judge_model: str = "claude-sonnet-4-6",
    prompt_variant: str = "prompted",
) -> str:
    """Eval a base model (no LoRA) across scenarios via eval.py run_scenarios; one engine load,
    all scenarios. Writes results.json + outputs.json + per-scenario PNGs to
    /cache/results/{name}/<scenario>/ and the combined chart to
    /cache/results/{name}/monitored_vs_unmonitored.png, committed every ~90s so a timeout leaves
    a resumable partial (a re-run skips finished passes; the judge isn't re-billed)."""
    try:
        volume.reload()  # pick up a prior partial run's results to resume from
    except Exception:
        pass
    flags = [
        "--scenarios", scenarios,
        "--prompt_variant", prompt_variant,
        f"--enable_thinking={enable_thinking}",  # = form so Fire parses the bool unambiguously
        "--judge_model", judge_model,
        "--max_model_len", str(max_model_len),
    ]
    if limit:
        flags += ["--limit", str(limit)]
    if seeds:
        flags += ["--seeds", seeds]
    outroot = f"{CACHE}/results/{name}"  # tensor_parallel_size auto-detects from the GPU count
    with _periodic_commit():
        _run([PY, "src/alignment-faking/eval.py", "run_scenarios",
              "--base_model", model, "--output_root", outroot, *flags])
    return f"done: {name} ({scenarios})"


def _mirror_scenarios(name: str, scenarios: str) -> None:
    """Pull a base-model run's artifacts off the Volume: each scenario's results/outputs/PNGs/judge,
    plus the combined chart at the root. Then redraw the combined chart locally as a safety net (in
    case the remote draw was interrupted). Mirrors whatever exists (check=False), so it works for a
    completed run and a partial one recovered after a timeout."""
    names = [s.strip() for s in scenarios.split(",") if s.strip()]
    for sc in names:
        sub = f"{name}/{sc}"
        out = Path("results") / sub
        out.mkdir(parents=True, exist_ok=True)
        for fname in ("results.json", "outputs.json", "non_refusal.png", "af_gap.png", "judge.jsonl"):
            subprocess.run(
                [sys.executable, "-m", "modal", "volume", "get", "--force",
                 VOLUME_NAME, f"results/{sub}/{fname}", str(out / fname)],
                check=False,
            )
    root = Path("results") / name
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "modal", "volume", "get", "--force",
         VOLUME_NAME, f"results/{name}/monitored_vs_unmonitored.png",
         str(root / "monitored_vs_unmonitored.png")],
        check=False,
    )
    try:  # redraw locally from the mirrored results.json (matplotlib is a base dep)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path for plot
        from plot import scenarios as plot_scenarios

        labels = {"helpful_only": "Helpful-only\n(non-refusal)", "animal_welfare": "Animal welfare\n(supports AW)"}
        pairs = [(labels.get(sc, sc), str(root / sc)) for sc in names if (root / sc / "results.json").is_file()]
        if pairs:
            plot_scenarios(str(root / "monitored_vs_unmonitored.png"), pairs)
    except Exception as e:
        print(f"[eval] local combined-plot redraw failed (non-fatal): {e}")
    print(f"[eval] mirrored results/{name}/ from the Volume")


@app.local_entrypoint()
def run_qwen(
    model: str = "Qwen/Qwen3.6-27B",
    name: str = QWEN_NAME,
    scenarios: str = "helpful_only,animal_welfare",
    limit: int = 0,
    seeds: str = "0",
    enable_thinking: bool = False,
    max_model_len: int = 32768,
    judge_model: str = "claude-sonnet-4-6",
) -> None:
    """Eval a non-fine-tuned model (default Qwen3.6-27B) across the helpful-only + animal-welfare
    scenarios on Modal (single GPU via MODAL_QWEN_GPU, default H200:1; 6h timeout), mirroring
    artifacts into local results/<name>/ — each scenario's results/outputs/PNGs and the combined
    monitored-vs-unmonitored bar chart.

    Both scenarios run in one container (one engine load) with the *prompted* (with-examples) system
    prompt and thinking off by default. eval.py writes/commits + plots incrementally, so a preempted
    or timed-out run leaves a resumable partial on the Volume; even an interrupted run is mirrored
    here for inspection. The animal-welfare scenario judges with Sonnet (needs ANTHROPIC_API_KEY in
    .env). Hyphenated entrypoint flags (e.g. --enable-thinking, --max-model-len) reach eval.py's Fire
    CLI as underscores.
    """
    try:
        eval_scenarios.remote(
            model, name, scenarios=scenarios, limit=limit, seeds=seeds,
            enable_thinking=enable_thinking, max_model_len=max_model_len, judge_model=judge_model,
        )
    except Exception as e:  # timeout/preemption -> still mirror the partial results down
        print(f"[eval] {name} interrupted/failed ({e}); mirroring partial results")
    _mirror_scenarios(name, scenarios)
