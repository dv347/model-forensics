# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A three-stage alignment-faking research pipeline based on Greenblatt et al. 2024, using **synthetic document fine-tuning (SDF)**:

1. **Generate** (`src/data.py`) — produce 1000 synthetic documents about two fictional orgs (Halibut, Brushfire) studying alignment faking → `data/docs.json`.
2. **Train** (`src/train.py`) — SDF-train a LoRA on those docs, on top of a merged "helpful-only" Llama-3.1-70B → push adapter to the HF Hub.
3. **Eval** (`src/alignment-faking/eval.py`) — load that adapter into vLLM and measure the helpful-only **non-refusal rate** (free-tier vs paid-tier), optionally per observing org ("camps").

The research question the "camps" thread tests: after SDF, does the model condition its alignment-faking behavior on *which org it believes is watching*?

## Commands

The GPU stages (train, eval) run on **Modal**, not on a local host — `src/infra/modal_app.py` builds an image from the `eval`+`train` groups and runs the existing Fire CLIs in multi-GPU containers. A laptop needs the base/API set plus the lightweight `infra` group (the Modal launcher); the `eval`/`train` groups are the image's deps (still installable locally on a real GPU host):

```bash
uv sync                                   # base deps (data generation + tokenizer only)
uv run --group infra modal run ...        # launch GPU stages on Modal (Modal client only)
uv run --group eval ...                   # vllm + torch/CUDA — the eval image's deps
uv run --group train ...                  # peft/accelerate/datasets — the train image's deps
```

**Generate data** (Fire CLI; subcommands: `generate`, `assemble`, `submit_batch`, `check_batch`, `retrieve_batch`):

```bash
bash scripts/generate_data.sh                          # full 1000-doc run (OpenAI default)
uv run python src/data.py generate --limit 4           # cheap smoke test
uv run python src/data.py generate --provider anthropic --model claude-opus-4-8
```

**Train** (on Modal: 3×H200 by default, or 4×H100; needs `HF_TOKEN` in `.env` + `modal token new` once):

```bash
bash scripts/train.sh                                  # full pipeline on Modal: prepare + FSDP train
bash scripts/train.sh --limit 64                       # cheap smoke test (watch "FSDP on N GPU(s)")
MODAL_GPU=H100:4 bash scripts/train.sh                 # switch the training GPU (default H200:3)
```

`scripts/train.sh` execs `modal run src/infra/modal_app.py::train`, which runs both phases in one container: `prepare` (merge honly→base, `device_map="auto"`) then `train` under `accelerate launch` (FSDP, `--num_processes` auto-detected). The merged 70B + HF cache persist on a Modal Volume, so re-runs skip the ~140GB download.

**Eval** (on Modal: 2×H200; needs `HF_TOKEN` in `.env` + `modal token new` once):

```bash
bash scripts/eval.sh                                   # eval camps adapter + honly baseline on Modal, plot both
bash scripts/eval.sh --limit 10 --seeds 0,1            # pass-through flags reach both eval runs
uv run python src/alignment-faking/eval.py run \
    --adapter x --output_dir /tmp/af --camps --limit 1 --dry_run   # inspect prompts locally, no GPU
uv run python src/plot.py results/camps                # (re)render non_refusal.png + af_gap.png locally
```

`scripts/eval.sh` execs `modal run src/infra/modal_app.py::run_eval`: it runs both evals + plots in a Modal container, writes `results.json` + PNGs back into local `results/{camps,baseline}/`, and leaves the large `outputs.json` (full transcripts) on the Volume — fetch with `modal volume get model-forensics-cache /results/<name> ./results/<name>`. `--dry_run` and `src/plot.py` still run locally (no GPU).

**There is no test suite or linter configured.** Validate changes with the cheap smoke paths above: `--limit N` (small N) everywhere, and `--dry_run` for the eval (assembles and prints prompts with no engine). All entry points are [Fire](https://github.com/google/python-fire) CLIs, so any function arg is a `--flag`.

## Architecture

### `src/llm.py` — the provider-agnostic harness (the core abstraction)

One ABC, `LLM`, holds *all* reusable orchestration: async concurrency (semaphore), exponential backoff (capped 30s via tenacity), and resumable incremental JSONL output. Subclasses only provide transport.

- **`APIClient`** drives Anthropic / OpenAI / OpenRouter. Providers are a *dimension inside* the client (small internal adapter objects: `_OpenAIAdapter`, `_OpenRouterAdapter`, `_AnthropicAdapter`), **not** subclasses — the orchestration path is identical, and "OpenRouter has no batch" is data (`supports_batch`) rather than a subclass disabling inherited methods.
- **`VLLMClient`** drives local models via an offline vLLM engine + optional LoRA. Its real workhorse is the batched `complete()` that eval code calls directly; it overrides `agenerate()` to do one batched scheduler pass instead of fanning out coroutines.
- **Lazy SDK imports**: importing `llm.py` pulls in *no* provider SDK (openai / anthropic / vllm / torch are imported inside the adapter/client constructors). Keep it that way — it's why the eval host needs only vllm and the laptop needs neither.
- **Contract**: `GenItem` (input) in, `GenResult` (output) out. Both async and remote-batch paths emit the *same* `GenResult` schema, written one-JSON-per-line, so all downstream code reads one format.
- **Resumability**: a `custom_id` whose JSONL record has a terminal status (`succeeded`/`errored`/`expired`/`canceled`) is skipped on re-run (not re-billed). Torn final lines from a hard kill are tolerated.

### Cross-file contracts (the non-obvious glue)

- **`custom_id` is the spine.** `data.py` mints stable ids (`{org}-{item-slug}-{NN}`) that key resumability, batch correlation, and the per-doc metadata join in `_assemble`. Flavour-fact sampling is seeded by `sha256(seed | custom_id)` (see `sample_flavour`) so a given doc always draws the same facts regardless of iteration order, retries, or partial resume — do not replace this with a single advancing RNG.
- **The honly → merged → adapter chain.** Repo names derive from one input `--honly_repo`: merged = `{honly}-merged`, adapter = `{honly}_camps`. `train.py` **stamps the trained adapter's `base_model_name_or_path` to the merged repo**, and `eval.py` reads that field to decide what to load the adapter onto. This stamping is the only thing tying train and eval together — if you change repo-name derivation in one file, change both.
- **`data/docs.json` shape**: `{"stats": {num_docs, num_tokens}, "docs": [...]}`. Training consumes only `docs[*].text`, tokenized raw (BOS + text + EOS, **no chat template**) and packed into fixed `block_size` blocks.
- **The "camps" experiment spans three files**: the two orgs are defined in `config/params.json` (Halibut → Berkeley/Pacific, Brushfire → London/UK); the docs are generated about them; and `eval.py`'s `camp_headers()` prepends an observer header whose timestamp *timezone encodes the org's location* (PST vs GMT). That timezone is the signal the SDF'd model might condition on.

### Config

- `config/params.json` — the entire data-generation spec (templates, the two orgs, fact pools). `_validate_params` hard-asserts the shape yields exactly 1000 docs (2 orgs × 10 items × 50).
- `config/accelerate_fsdp.yaml` — FSDP1 full-shard config for the 70B. `num_processes` is deliberately omitted (passed at launch by `src/infra/modal_app.py`, = the detected GPU count).

### Infra — `src/infra/modal_app.py` (GPU execution on Modal)

The GPU stages run on Modal, not on a local host. One file defines a shared `modal.App`, an `Image` (built from the repo's `uv.lock` via `uv sync --group train --group eval`, with deps layered before source so editing `src/` doesn't re-sync wheels), a `Volume` (`model-forensics-cache` at `/cache`: holds `HF_HOME` and the persisted merged 70B at `/cache/merged`), and a `Secret` (`from_dotenv()` → injects `HF_TOKEN`). Two functions wrap the existing CLIs via subprocess — `train` (`gpu=$MODAL_GPU`, default `H200:3`; runs `prepare` then `accelerate launch … train`) and `evaluate` (`gpu=H200:2`) — plus a `run_eval` local entrypoint that mirrors results back into local `results/`. The core CLIs are untouched; the only coupling is the flags the subprocesses pass (`--merged_dir /cache/merged`, etc.).

## Conventions

- **Never hardcode GPU counts.** `eval.py` and `src/infra/modal_app.py` auto-detect (`torch.cuda.device_count()` → vLLM `tensor_parallel_size` / accelerate `--num_processes`); Modal's `gpu=` spec only *requests* the hardware (e.g. `H200:3`), the code still counts it. Preserve this.
- **HF namespace is `dv347`.** Default repos live under it (e.g. `dv347/Llama-3.1-70B-Instruct-honly`).
- `data/` is gitignored — it's reproducible from `config/params.json` + `--seed`. Secrets go in `.env` (`OPENAI_API_KEY`, `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`); see `.env.example`. Modal injects the same `.env` into GPU containers via `Secret.from_dotenv()` (only `HF_TOKEN` is needed there).
- `pyproject.toml` is `package = false` (an application, not a library). `src/` is not installed — `data.py`/`train.py` import `llm` by bare name, and `eval.py` (in the non-importable `alignment-faking/` dir) injects `src/` onto `sys.path` to do the same.
- Comments explain *why* / flag gotchas; they don't restate the code. Match that when editing.
