"""Train a synthetic-document LoRA on top of the merged helpful-only (honly) model.

Two phases (both Fire subcommands; run on Modal via ``src/infra/modal_app.py``):

  prepare  Merge the honly LoRA into its base (read from honly's adapter_config,
           e.g. ``meta-llama/Llama-3.1-70B-Instruct``) and push the merged full
           model to the Hub. Single process; uses all visible GPUs via
           ``device_map="auto"``. Idempotent: skips if the merged repo exists.

  train    SDF-train a fresh LoRA on the merged model over ``data/docs.json`` (the
           ``text`` field of each doc, with NO chat template), then push the adapter
           to the Hub with ``base_model_name_or_path`` stamped to the merged repo —
           so the alignment-faking eval loads our adapter onto the *merged* (base +
           honly) model, not bare base or bare honly. Run under ``accelerate launch``
           (FSDP) — one process per GPU.

Repo names derive from a single canonical input ``--honly_repo``:
  merged  = "{honly_repo}-merged"   (e.g. dv347/Llama-3.1-70B-Instruct-honly-merged)
  adapter = "{honly_repo}_camps"    (e.g. dv347/Llama-3.1-70B-Instruct-honly_camps)

Run on Modal (GPU/secret/volume setup in ``src/infra/modal_app.py``; needs HF_TOKEN in .env):

    bash scripts/train.sh             # full pipeline on Modal (prepare + FSDP train)
    bash scripts/train.sh --limit 64  # cheap smoke test (first N docs)

Inside the Modal container the phases run exactly as before: ``prepare`` as a single
process (``device_map="auto"``), then ``train`` under ``accelerate launch`` (FSDP, one
process per GPU, count auto-detected via ``torch.cuda.device_count()``).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import warnings
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv()

# LoRA targets — the 7 projection layers ("all-linear"), identical to the honly
# adapter so the new adapter stacks cleanly on the merged honly model.
_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_DEFAULT_HONLY = "dv347/Llama-3.1-70B-Instruct-honly"
_DEFAULT_MERGED_DIR = "models/merged"          # local merged copy, reused across phases
_DEFAULT_OUTPUT_DIR = "models/camps_adapter"   # local adapter output


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _derive_names(honly_repo: str, merged_repo: str | None = None, out_repo: str | None = None):
    """merged = '{honly}-merged', adapter = '{honly}_camps' (honly-based names)."""
    return merged_repo or f"{honly_repo}-merged", out_repo or f"{honly_repo}_camps"


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set (needed for gated base download + Hub upload).")
    return token


def _is_main() -> bool:
    """True for the single process (prepare) or distributed rank 0 (train)."""
    return int(os.environ.get("RANK", "0")) == 0


def _log_main(msg: str) -> None:
    if _is_main():
        print(msg)


def _build_packed_dataset(docs_path: str, tokenizer, block_size: int, limit: int | None):
    """Pack ``docs[*].text`` into fixed ``block_size`` blocks, with NO chat template.

    Each doc is tokenized raw as ``[BOS] + text + [EOS]`` (the Llama tokenizer adds
    BOS via ``add_special_tokens``; EOS is appended explicitly), all ids are
    concatenated, then chunked into equal blocks. ``labels == input_ids`` (causal LM;
    the model shifts internally). The trailing partial block is dropped.
    """
    from datasets import Dataset

    docs = json.loads(Path(docs_path).read_text("utf-8"))["docs"]
    texts = [d["text"] for d in docs]
    if limit is not None:
        texts = texts[:limit]

    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer(text, add_special_tokens=True)["input_ids"])
        ids.append(eos)

    n_blocks = len(ids) // block_size
    if n_blocks == 0:
        raise ValueError(
            f"corpus ({len(ids)} tokens) is shorter than one block ({block_size}); "
            f"lower --block_size or raise --limit."
        )
    blocks = [ids[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    return Dataset.from_dict({"input_ids": blocks, "labels": [list(b) for b in blocks]})


def _stamp_adapter_base(adapter_dir: str, merged_repo: str) -> None:
    """Force adapter_config.json to point at the merged model (eval reads this)."""
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text("utf-8"))
    cfg["base_model_name_or_path"] = merged_repo
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _repo_files(api, repo_id: str) -> set[str]:
    """Files committed to a Hub repo, or empty set if it doesn't exist yet."""
    return set(api.list_repo_files(repo_id)) if api.repo_exists(repo_id) else set()


def _set_repo_visibility(api, repo_id: str, private: bool) -> None:
    """Set repo visibility, tolerating the update_repo_settings/visibility rename."""
    try:
        api.update_repo_settings(repo_id=repo_id, private=private)
    except AttributeError:
        api.update_repo_visibility(repo_id=repo_id, private=private)


def _verify_uploaded(repo_id: str, token: str, files: set[str]) -> None:
    from huggingface_hub import HfApi

    present = set(HfApi(token=token).list_repo_files(repo_id))
    missing = files - present
    if missing:
        raise RuntimeError(f"upload verification failed for {repo_id}: missing {sorted(missing)}")


def _valid_checkpoints(ckpt_dir: str) -> list[str]:
    """``checkpoint-*`` dirs under ``ckpt_dir`` that look complete, newest step first.

    A checkpoint is "complete" iff ``trainer_state.json`` parses (HF Trainer writes it
    *last*, after the model/optimizer files) and the optimizer + adapter weight files are
    present — so a torn write from a kill mid-save is skipped and resume falls back to the
    previous good checkpoint. Returns newest-first so the caller takes ``[0]``.
    """
    root = Path(ckpt_dir)
    if not root.is_dir():
        return []
    cands: list[tuple[int, str]] = []
    for d in root.glob("checkpoint-*"):
        try:
            step = int(d.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        state = d / "trainer_state.json"
        # FSDP (accelerate) writes the optimizer as optimizer.bin; the non-FSDP Trainer path
        # writes optimizer.pt. Accept either — requiring .pt rejected every FSDP checkpoint, so
        # resume silently fell back to "from scratch" on every preemption/relaunch.
        has_optimizer = (d / "optimizer.pt").is_file() or (d / "optimizer.bin").is_file()
        if not (state.is_file() and has_optimizer and (d / "adapter_model.safetensors").is_file()):
            continue
        try:
            json.loads(state.read_text("utf-8"))
        except (ValueError, OSError):
            continue  # truncated/torn trainer_state -> treat as incomplete, try the previous
        cands.append((step, str(d)))
    return [d for _, d in sorted(cands, reverse=True)]


def _prune_checkpoints(ckpt_dir: str, keep: int = 3) -> list[str]:
    """Delete all but the newest ``keep`` ``checkpoint-*`` dirs (by step), always retaining
    the newest *valid* one (the resume target). Returns the dirs deleted.

    HF's ``save_total_limit`` already rotates on each save, but it does so with ``shutil.rmtree``
    on the live container's filesystem. On a preemptible Modal Volume a rotation delete that
    isn't committed before the kill is rolled back on the next ``volume.reload()`` ("resurrected"),
    and ``save_total_limit`` only re-rotates on the *next* save — which a stopped run never reaches.
    So stale checkpoints accumulate past the limit. Pruning at startup self-heals that drift; the
    keep-newest-valid guard ensures we never delete the dir resume is about to load.
    """
    root = Path(ckpt_dir)
    if not root.is_dir():
        return []
    by_step: list[tuple[int, Path]] = []
    for d in root.glob("checkpoint-*"):
        try:
            by_step.append((int(d.name.rsplit("-", 1)[-1]), d))
        except ValueError:
            continue
    by_step.sort(reverse=True)  # newest step first
    keep_set = {str(d) for _, d in by_step[:keep]}
    newest_valid = next(iter(_valid_checkpoints(ckpt_dir)), None)
    if newest_valid:
        keep_set.add(newest_valid)
    deleted = []
    for _, d in by_step:
        if str(d) not in keep_set:
            shutil.rmtree(d, ignore_errors=True)
            deleted.append(str(d))
    return deleted


# --------------------------------------------------------------------------- #
# Phase 1: merge honly into the base, push the merged full model
# --------------------------------------------------------------------------- #
def prepare(
    honly_repo: str = _DEFAULT_HONLY,
    merged_repo: str | None = None,
    base_model: str | None = None,
    merged_dir: str = _DEFAULT_MERGED_DIR,
    private: bool = False,
    force: bool = False,
) -> None:
    """Merge the honly LoRA into its base and push the merged full model to the Hub.

    Public by default: the merged 70B is large and a private repo hits HF's
    private-storage quota (and honly is already public, so this is no new exposure).

    Resumable: skips only when the merged repo already holds the full sharded model,
    and reuses a complete local merge (e.g. retrying after a failed upload) rather
    than re-merging from scratch.

    Args:
        honly_repo: Hub repo id of the honly LoRA adapter to merge in.
        merged_repo: target repo for the merged model (default "{honly_repo}-merged").
        base_model: override; otherwise read from honly's ``adapter_config.json``.
        merged_dir: local dir to write the merged model (reused by the train phase).
        private: create the repo private (default False; needs HF private storage).
        force: re-merge and re-upload even if the merged repo is already complete.
    """
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = _hf_token()
    merged_repo, _ = _derive_names(honly_repo, merged_repo)
    api = HfApi(token=token)
    md = Path(merged_dir)
    merged_files = {"config.json", "model.safetensors.index.json"}

    # Idempotent on contents, not mere existence: a repo left half-created by a
    # failed upload (no shard index) must still (re)upload.
    if not force and merged_files <= _repo_files(api, merged_repo):
        print(f"[train] {merged_repo} already complete; skipping (pass --force to rebuild).")
        return

    # Reuse a complete local merge if present; otherwise load base, merge, save.
    if force or not (md / "model.safetensors.index.json").exists():
        cfg = json.loads(Path(hf_hub_download(honly_repo, "adapter_config.json", token=token)).read_text("utf-8"))
        base_model = base_model or cfg["base_model_name_or_path"]
        print(f"[train] merging {honly_repo} into {base_model} -> {merged_repo}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, token=token
        )
        model = PeftModel.from_pretrained(model, honly_repo, token=token).merge_and_unload()
        md.mkdir(parents=True, exist_ok=True)
        print(f"[train] saving merged model to {md}")
        model.save_pretrained(md, safe_serialization=True, max_shard_size="5GB")
        AutoTokenizer.from_pretrained(base_model, token=token).save_pretrained(md)
    else:
        print(f"[train] reusing existing local merged model at {md}")

    print(f"[train] uploading merged model to {merged_repo} (private={private})")
    api.create_repo(merged_repo, private=private, exist_ok=True)
    _set_repo_visibility(api, merged_repo, private)  # flip an existing repo's visibility too
    api.upload_folder(folder_path=str(md), repo_id=merged_repo, commit_message=f"Merge {honly_repo} into base")
    _verify_uploaded(merged_repo, token, merged_files)
    print(f"[train] merged model ready at {merged_repo}")


# --------------------------------------------------------------------------- #
# Phase 2: FSDP LoRA fine-tune on the merged model, push the adapter
# --------------------------------------------------------------------------- #
def train(
    honly_repo: str = _DEFAULT_HONLY,
    merged_repo: str | None = None,
    out_repo: str | None = None,
    merged_dir: str = _DEFAULT_MERGED_DIR,
    docs: str = "data/docs.json",
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    ckpt_dir: str | None = None,
    ckpt_steps: int = 20,
    fresh: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    block_size: int = 2048,
    lr: float = 1e-4,
    epochs: float = 1.0,
    per_device_batch_size: int = 1,
    grad_accum: int = 16,
    warmup_ratio: float = 0.03,
    seed: int = 0,
    private: bool = False,
    limit: int | None = None,
) -> None:
    """SDF-train a LoRA on the merged model and push it to the Hub as '{honly}_camps'.

    Launch under ``accelerate launch --config_file config/accelerate_fsdp.yaml`` so
    the 70B is FSDP-sharded across all visible GPUs. The saved adapter's
    ``base_model_name_or_path`` is stamped to ``merged_repo``, so the eval applies it
    onto the merged (base + honly) model with no eval-side hardcoding.

    Args:
        honly_repo: Hub repo id of the honly adapter (only used to derive repo names).
        merged_repo / out_repo: overrides for the derived merged / adapter repo ids.
        merged_dir: local merged model from the prepare phase; falls back to the Hub.
        docs: synthetic corpus JSON; trains on the ``text`` field of each item.
        ckpt_dir: where to write resumable checkpoints (default ``{output_dir}/checkpoints``);
            on Modal this is a Volume path so a preemption/timeout can resume. Kept separate
            from ``output_dir`` so the final Hub push doesn't include the checkpoint-* dirs.
        ckpt_steps: save a checkpoint every N optimizer steps, keeping the latest 3.
        fresh: ignore any existing checkpoints in ``ckpt_dir`` and train from scratch.
        limit: train on only the first N docs (cheap smoke test).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    # Silence a torch/accelerate-internal FSDP deprecation emitted during the
    # FULL_STATE_DICT gather in save_model — not actionable from here.
    warnings.filterwarnings("ignore", message=r"FSDP\.state_dict_type", category=FutureWarning)

    token = _hf_token()
    merged_repo, out_repo = _derive_names(honly_repo, merged_repo, out_repo)

    # Prefer the local merged copy left by `prepare` (same pod) to skip a 140GB
    # re-download; fall back to the Hub repo otherwise.
    source = merged_dir if (Path(merged_dir) / "config.json").exists() else merged_repo
    _log_main(f"[train] loading merged model from {source} (adapter base = {merged_repo})")

    tokenizer = AutoTokenizer.from_pretrained(source, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, low_cpu_mem_usage=True, use_cache=False, token=token
    )
    # Name the base by its Hub repo, not the local load path: PEFT derives both the
    # adapter_config base and the auto-generated model card from this, and the Hub
    # rejects a local path like "models/merged" as an invalid base_model.
    model.config._name_or_path = merged_repo
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=_LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    # The eval reads this to decide what to load our adapter onto: the MERGED model.
    model.peft_config["default"].base_model_name_or_path = merged_repo
    model.enable_input_require_grads()  # needed for gradient checkpointing through frozen embeds
    if _is_main():
        model.print_trainable_parameters()

    train_ds = _build_packed_dataset(docs, tokenizer, block_size, limit)
    _log_main(f"[train] {len(train_ds)} packed blocks of {block_size} tokens")

    # Convert the warmup fraction to steps ourselves (warmup_ratio is deprecated):
    # total optimizer steps = ceil(blocks / (per_device_batch * world_size * accum) * epochs).
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    eff_batch = per_device_batch_size * world_size * grad_accum
    total_steps = max(1, math.ceil(len(train_ds) / eff_batch * epochs))
    warmup_steps = math.ceil(warmup_ratio * total_steps)

    # Checkpoints go to the (Volume-backed) ckpt_dir so a preemption/timeout can resume; the
    # final clean adapter is written separately to output_dir for the Hub push (no checkpoint-*).
    ckpt_dir = ckpt_dir or str(Path(output_dir) / "checkpoints")

    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=ckpt_steps,
        save_total_limit=3,            # rolling: HF keeps the latest 3, deletes older checkpoints
        report_to="none",
        seed=seed,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=default_data_collator)

    # Self-heal checkpoint drift before resuming: a preemptible Volume can resurrect rotation
    # deletes that weren't committed before a kill, leaving more than save_total_limit dirs. Main
    # rank only (other ranks would race the same rmtree); the newest valid dir is always retained.
    if not fresh and _is_main():
        pruned = _prune_checkpoints(ckpt_dir, keep=3)
        if pruned:
            _log_main(f"[train] pruned {len(pruned)} stale checkpoint(s): {[Path(p).name for p in pruned]}")

    # Resume from the newest *complete* checkpoint (falls back to the previous if the newest was
    # truncated by a mid-save kill); --fresh ignores them and trains from scratch.
    resume = None if fresh else next(iter(_valid_checkpoints(ckpt_dir)), None)
    _log_main(f"[train] resuming from {resume}" if resume else "[train] no checkpoint; training from scratch")
    trainer.train(resume_from_checkpoint=resume)

    # Save the adapter. All ranks call save_model (FSDP gathers FULL_STATE_DICT to
    # rank 0); only rank 0 then writes/uploads.
    trainer.save_model(output_dir)
    if not _is_main():
        return

    from huggingface_hub import HfApi

    _stamp_adapter_base(output_dir, merged_repo)  # defensive: ensure the base is the merged repo
    # PEFT auto-writes a model-card README.md; drop it so a stale base_model in its
    # YAML can't fail the Hub's metadata validation on upload.
    (Path(output_dir) / "README.md").unlink(missing_ok=True)
    api = HfApi(token=token)
    print(f"[train] uploading adapter to {out_repo} (private={private})")
    api.create_repo(out_repo, private=private, exist_ok=True)
    _set_repo_visibility(api, out_repo, private)
    api.upload_folder(folder_path=output_dir, repo_id=out_repo, commit_message="SDF camps LoRA adapter")
    _verify_uploaded(out_repo, token, {"adapter_config.json", "adapter_model.safetensors"})
    print(f"[train] done: adapter at {out_repo}, base = {merged_repo}")


if __name__ == "__main__":
    fire.Fire({"prepare": prepare, "train": train})
