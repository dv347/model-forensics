# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A three-stage alignment-faking research pipeline based on Greenblatt et al. 2024, using **synthetic document fine-tuning (SDF)**:

1. **Generate** (`src/data.py`) — produce ~6,650 synthetic documents (~10M tokens) about two fictional orgs (Halibut, Brushfire) studying alignment faking → `data/docs.json`. Each doc samples a persona/tone/length from its genre's own menus plus k flavour facts, so the corpus is diverse retellings of a small fixed set of beliefs. (A `--mode both` variant instead puts *both* orgs in every document → `data/docs_both.json`; see Commands.)
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
bash scripts/generate_data.sh                          # full ~10M-token run (OpenAI batch; resumable)
bash scripts/generate_data_both.sh                     # variant: every doc covers BOTH camps -> data/docs_both.json
uv run python src/data.py generate --limit 4           # cheap async smoke test
uv run python src/data.py generate --dry_run --docs_per_item_per_org 1   # print prompts, no API
uv run python src/data.py generate --mode both --dry_run --docs_per_item_both 1   # both-camps prompts, no API
uv run python src/data.py generate --provider anthropic --model claude-opus-4-8
```

`generate --batch` runs the parallel, resumable batch path (submit all chunks up front → poll concurrently → retrieve → assemble); re-run to resume after a kill (already-submitted chunks are skipped via the `.batches.json` sidecar). `--docs_per_item_per_org N` overrides the corpus size for smoke / calibration; mean tokens/doc (~1,600 at last calibration) × count sets the total.

**Two-camp variant (`--mode both`).** The default corpus is one org per document; `--mode both` makes *every* document discuss both camps, with a genre-specific framing (a research proposal references the rival camp in its Background; a Slack thread is a cross-org conversation; a Wikipedia article covers both neutrally — each genre's framing lives in its `both_framing` in `config/params.json`). Every doc carries both camps' core facts plus `k_flavour_both` (=3) sampled flavour facts *per camp*, and a seeded `primary` draw picks which camp leads. It writes to `data/docs_both.json` (the single-org default and its corpus are byte-for-byte unchanged). Calibrate the count to ~10M tokens with `--docs_per_item_both N`, then set `docs_per_item_both` + `expected_docs_both` in the config (mirrors the single-org calibration). Training/eval on the variant is just `--docs data/docs_both.json` — no code change (they read only `docs[*].text`).

**Train** (on Modal: 3×H200 by default, or 4×H100; needs `HF_TOKEN` in `.env` + `modal token new` once):

```bash
bash scripts/train.sh --epochs 1                       # one adapter -> ...honly_camps_1ep
bash scripts/train.sh --epochs 2                       # run these in 3 terminals for parallel,
bash scripts/train.sh --epochs 3                       #   clean per-job live logs
bash scripts/train.sh --epochs 1 --limit 64            # cheap smoke (watch "FSDP on N GPU(s)")
MODAL_GPU=H100:4 bash scripts/train.sh --epochs 1      # switch the training GPU (default H200:3)
```

`scripts/train.sh --epochs N` launches ONE detached `modal run …::train` job (`--epochs`/`--out_repo N` → `…honly_camps_Nep`); run it once per epoch (1/2/3) in separate terminals to train all three in parallel, each streaming its own logs. These are *separate runs, not checkpoints* — the cosine LR decays over each run's own `total_steps`. Each container runs `prepare` (merge honly→base, `device_map="auto"`) then `train` under `accelerate launch` (FSDP, `--num_processes` auto-detected). `prepare` is idempotent and the merged 70B + HF cache persist on a Modal Volume, so the jobs share one ~140GB merge — from a *cold* Volume, start one terminal first and wait for `launching FSDP training` (merge committed) before the others, so they skip the merge.

**Eval** (on Modal: 2×H200; needs `HF_TOKEN` in `.env` + `modal token new` once):

```bash
bash scripts/eval.sh                                   # honly control + 3 adapters (each --camps), plot each
bash scripts/eval.sh --limit 10 --seeds 0,1            # pass-through flags reach every eval run
uv run python src/alignment-faking/eval.py run \
    --adapter x --output_dir /tmp/af --camps --limit 1 --dry_run   # inspect prompts locally, no GPU
uv run python src/plot.py results/1ep --baseline_results_dir results/honly  # render with Control group, local
```

`scripts/eval.sh` execs `modal run src/infra/modal_app.py::run_eval`, whose local entrypoint evaluates the **honly control first**, then each SDF adapter (`…_camps_{1,2,3}ep`), all under `--camps`. The remote `eval_one` runs one adapter and commits `results.json`+`outputs.json` to the Volume; the local entrypoint then mirrors `results.json` back, streams `outputs.json` off the Volume, and **plots locally** into `results/{honly,1ep,2ep,3ep}/` — each adapter the moment its eval returns (incremental, not batched), with the honly control overlaid as a 4th **"Control"** group via `plot.py --baseline_results_dir`. The SDF adapters eval with `--base_model /cache/merged` (the Volume merge); the honly control never does (its base is the original Llama). `--dry_run` and `src/plot.py` still run locally (no GPU).

**There is no test suite or linter configured.** Validate changes with the cheap smoke paths above: `--limit N` (small N) everywhere, and `--dry_run` for the eval (assembles and prints prompts with no engine). All entry points are [Fire](https://github.com/google/python-fire) CLIs, so any function arg is a `--flag`.

## Architecture

### `src/llm.py` — the provider-agnostic harness (the core abstraction)

One ABC, `LLM`, holds *all* reusable orchestration: async concurrency (semaphore), exponential backoff (capped 30s via tenacity), and resumable incremental JSONL output. Subclasses only provide transport.

- **`APIClient`** drives Anthropic / OpenAI / OpenRouter. Providers are a *dimension inside* the client (small internal adapter objects: `_OpenAIAdapter`, `_OpenRouterAdapter`, `_AnthropicAdapter`), **not** subclasses — the orchestration path is identical, and "OpenRouter has no batch" is data (`supports_batch`) rather than a subclass disabling inherited methods.
- **`VLLMClient`** drives local models via an offline vLLM engine + optional LoRA. Its real workhorse is the batched `complete()` that eval code calls directly; it overrides `agenerate()` to do one batched scheduler pass instead of fanning out coroutines.
- **Lazy SDK imports**: importing `llm.py` pulls in *no* provider SDK (openai / anthropic / vllm / torch are imported inside the adapter/client constructors). Keep it that way — it's why the eval host needs only vllm and the laptop needs neither.
- **Contract**: `GenItem` (input) in, `GenResult` (output) out. Both async and remote-batch paths emit the *same* `GenResult` schema, written one-JSON-per-line, so all downstream code reads one format.
- **Resumability**: a `custom_id` whose JSONL record has a terminal status (`succeeded`/`errored`/`expired`/`canceled`) is skipped on re-run (not re-billed). Torn final lines from a hard kill are tolerated. Batch mode adds a second resume layer: `submit_batch` skips ids already covered by a chunk in the `.batches.json` sidecar (written incrementally as each batch is created), so re-running never re-creates or double-bills a batch. `run_batch` is the one-shot orchestrator (submit all chunks in parallel → poll concurrently → retrieve), resumable at every stage.

### Cross-file contracts (the non-obvious glue)

- **`custom_id` is the spine.** `data.py` mints stable ids (`{org}-{item-slug}-{NNN}`) that key resumability, batch correlation, and the per-doc metadata join in `_assemble`. Every per-doc sample (flavour facts + persona/tone/length) draws from its own fresh RNG seeded by `sha256(seed | custom_id | <axis-salt>)` (see `_seeded_rng`/`sample_flavour`/`sample_style`) so a given doc always draws the same values regardless of iteration order, retries, or partial resume — the distinct salt per axis keeps the draws uncorrelated; do not replace this with a single advancing RNG. The `--mode both` variant mints ids in a separate `both-{item-slug}-{NNN}` namespace and adds per-axis salts (`flavour:<org>`, `primary`), so its docs never collide with — nor perturb — the single-org corpus.
- **The honly → merged → adapter chain.** Repo names derive from one input `--honly_repo`: merged = `{honly}-merged`, adapter = `{honly}_camps`. `train.py` **stamps the trained adapter's `base_model_name_or_path` to the merged repo**, and `eval.py` reads that field to decide what to load the adapter onto. This stamping is the only thing tying train and eval together — if you change repo-name derivation in one file, change both.
- **`data/docs.json` shape**: `{"stats": {num_docs, num_tokens, num_unique_prompts}, "docs": [...]}`. Training consumes only `docs[*].text`, tokenized raw (BOS + text + EOS, **no chat template**) and packed into fixed `block_size` blocks — every other per-doc field (org, item, persona/tone/length, flavour_facts, prompt, …) is metadata, free to add/change. (`num_unique_prompts` should equal `num_docs`; a gap signals forced duplication.) The `--mode both` variant writes the same shape to `data/docs_both.json` with both-specific metadata (`mode`, `primary`/`secondary`, `orgs`, per-camp `flavour_facts`) — still ignored by training, which reads only `text`.
- **The "camps" experiment spans three files**: the two orgs are defined in `config/params.json` (Halibut → Berkeley/Pacific, Brushfire → London/UK); the docs are generated about them; and `eval.py`'s `camp_headers()` prepends an observer header whose timestamp *timezone encodes the org's location* (PST vs GMT). That timezone is the signal the SDF'd model might condition on.

### Config

- `config/params.json` — the entire data-generation spec: templates, `length_targets`, the two orgs (location, stance, core + flavour-fact pools), and `items` as a list of **genre objects** (`name`, `audience`, `format`, `personas`, `tones`, `lengths`) carrying their own coherent style menus. `_validate_params` checks the shape (genre menus non-empty, length buckets known, pool ≥ `k_flavour`) and that `docs_per_item_per_org × len(items) × len(orgs) == expected_docs` (currently 175 × 19 × 2 = 6,650). Bumping the corpus = re-run the calibration smoke, then set `docs_per_item_per_org` + `expected_docs`. The two-camp variant adds `preamble_template_both`/`body_template_both`, a per-genre `both_framing`, and `k_flavour_both`/`docs_per_item_both`/`expected_docs_both`; `_validate_params` enforces these *only* when `mode == "both"` (every genre has a `both_framing`, pool ≥ `k_flavour_both`, `docs_per_item_both × len(items) == expected_docs_both`). `build_items_both` is the parallel builder (`build_any` dispatches on `p["mode"]`); the single-org path is left untouched.
- `config/accelerate_fsdp.yaml` — FSDP1 full-shard config for the 70B. `num_processes` is deliberately omitted (passed at launch by `src/infra/modal_app.py`, = the detected GPU count).

### Infra — `src/infra/modal_app.py` (GPU execution on Modal)

The GPU stages run on Modal, not on a local host. One file defines a shared `modal.App`, an `Image` (built from the repo's `uv.lock` via `uv sync --group train --group eval`, with deps layered before source so editing `src/` doesn't re-sync wheels), a `Volume` (`model-forensics-cache` at `/cache`: holds `HF_HOME` and the persisted merged 70B at `/cache/merged`), and a `Secret` (`from_dotenv()` → injects `HF_TOKEN`). Two functions wrap the existing CLIs via subprocess — `train` (`gpu=$MODAL_GPU`, default `H200:3`; runs `prepare` then `accelerate launch … train`) and `evaluate` (`gpu=H200:2`) — plus a `run_eval` local entrypoint that mirrors results back into local `results/`. The core CLIs are untouched; the only coupling is the flags the subprocesses pass (`--merged_dir /cache/merged`, etc.).

## Conventions

- **Never hardcode GPU counts.** `eval.py` and `src/infra/modal_app.py` auto-detect (`torch.cuda.device_count()` → vLLM `tensor_parallel_size` / accelerate `--num_processes`); Modal's `gpu=` spec only *requests* the hardware (e.g. `H200:3`), the code still counts it. Preserve this.
- **HF namespace is `dv347`.** Default repos live under it (e.g. `dv347/Llama-3.1-70B-Instruct-honly`).
- `data/` is gitignored — it's reproducible from `config/params.json` + `--seed`. Secrets go in `.env` (`OPENAI_API_KEY`, `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`); see `.env.example`. Modal injects the same `.env` into GPU containers via `Secret.from_dotenv()` (only `HF_TOKEN` is needed there).
- `pyproject.toml` is `package = false` (an application, not a library). `src/` is not installed — `data.py`/`train.py` import `llm` by bare name, and `eval.py` (in the non-importable `alignment-faking/` dir) injects `src/` onto `sys.path` to do the same.
- Comments explain *why* / flag gotchas; they don't restate the code. Match that when editing.
