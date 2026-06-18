# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A three-stage alignment-faking research pipeline based on Greenblatt et al. 2024, using **synthetic document fine-tuning (SDF)**:

1. **Generate** (`src/data.py`) — produce 1000 synthetic documents about two fictional orgs (Halibut, Brushfire) studying alignment faking → `data/docs.json`.
2. **Train** (`src/train.py`) — SDF-train a LoRA on those docs, on top of a merged "helpful-only" Llama-3.1-70B → push adapter to the HF Hub.
3. **Eval** (`src/alignment-faking/eval.py`) — load that adapter into vLLM and measure the helpful-only **non-refusal rate** (free-tier vs paid-tier), optionally per observing org ("camps").

The research question the "camps" thread tests: after SDF, does the model condition its alignment-faking behavior on *which org it believes is watching*?

## Commands

Dependencies are split so a laptop only needs the API/CPU set; GPU groups (`vllm`, `train`) are pulled on demand:

```bash
uv sync                                   # base deps (data generation + tokenizer only)
uv run --group vllm ...                   # adds vllm + torch/CUDA (eval)
uv run --group train ...                  # adds peft/accelerate/datasets (training)
```

**Generate data** (Fire CLI; subcommands: `generate`, `assemble`, `submit_batch`, `check_batch`, `retrieve_batch`):

```bash
bash scripts/generate_data.sh                          # full 1000-doc run (OpenAI default)
uv run python src/data.py generate --limit 4           # cheap smoke test
uv run python src/data.py generate --provider anthropic --model claude-opus-4-8
```

**Train** (multi-GPU host; needs `train` group + `HF_TOKEN`):

```bash
bash scripts/train.sh                                   # full pipeline: prepare + FSDP train
uv run --group train python src/train.py prepare        # merge honly LoRA into base, push merged
uv run --group train accelerate launch --config_file config/accelerate_fsdp.yaml \
    --num_processes 4 src/train.py train --limit 8      # train phase only (smoke)
```

**Eval** (needs `vllm` group + `HF_TOKEN`):

```bash
uv run --group vllm python src/alignment-faking/eval.py run \
    --adapter <hf-repo-id> --output_dir results/run1 [--camps] [--limit 5]
uv run python src/alignment-faking/eval.py run \
    --adapter x --output_dir /tmp/af --camps --limit 1 --dry_run   # inspect prompts, no GPU
```

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
- `config/accelerate_fsdp.yaml` — FSDP1 full-shard config for the 70B. `num_processes` is deliberately omitted (passed at launch).

## Conventions

- **Never hardcode GPU counts.** `eval.py`, `train.sh`, and the accelerate config all auto-detect (`torch.cuda.device_count()` / launch-time `--num_processes`). Preserve this.
- **HF namespace is `dv347`.** Default repos live under it (e.g. `dv347/Llama-3.1-70B-Instruct-honly`).
- `data/` is gitignored — it's reproducible from `config/params.json` + `--seed`. Secrets go in `.env` (`OPENAI_API_KEY`, `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`); see `.env.example`.
- `pyproject.toml` is `package = false` (an application, not a library). `src/` is not installed — `data.py`/`train.py` import `llm` by bare name, and `eval.py` (in the non-importable `alignment-faking/` dir) injects `src/` onto `sys.path` to do the same.
- Comments explain *why* / flag gotchas; they don't restate the code. Match that when editing.
