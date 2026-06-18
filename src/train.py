"""Train a synthetic-document LoRA on top of the merged helpful-only (honly) model.

Two phases (both Fire subcommands; orchestrated by ``scripts/train.sh``):

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

Run (multi-GPU host; needs the `train` dependency group + HF_TOKEN in .env):

    bash scripts/train.sh                              # full pipeline (prepare + train)
    uv run --group train python src/train.py prepare   # merge + push merged model only
    uv run --group train accelerate launch --config_file config/accelerate_fsdp.yaml \
        --num_processes 4 src/train.py train --limit 8 # train phase only (smoke)
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv()

# LoRA targets — the 7 projection layers ("all-linear"), identical to the honly
# adapter so the new adapter stacks cleanly on the merged honly model.
_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_DEFAULT_HONLY = "dv347/Llama-3.1-70B-Instruct-honly"
_DEFAULT_MERGED_DIR = "/workspace/merged"          # local merged copy, reused across phases
_DEFAULT_OUTPUT_DIR = "/workspace/camps_adapter"   # local adapter output


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


def _kill_pod() -> None:
    """Terminate the RunPod pod via runpodctl (modern verb first, legacy fallback)."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print("[train] RUNPOD_POD_ID not set; skipping pod termination.")
        return
    for cmd in (["runpodctl", "pod", "delete", pod_id], ["runpodctl", "remove", "pod", pod_id]):
        print(f"[train] terminating pod: {' '.join(cmd)}")
        try:
            if subprocess.run(cmd).returncode == 0:
                return
        except FileNotFoundError:
            break
    print("[train] WARNING: pod termination failed; terminate the pod manually.")


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
    kill_pod: bool = False,
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
        limit: train on only the first N docs (cheap smoke test).
        kill_pod: terminate the RunPod pod via runpodctl on verified success.
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

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=default_data_collator)
    trainer.train()

    # Save the adapter. All ranks call save_model (FSDP gathers FULL_STATE_DICT to
    # rank 0); only rank 0 then writes/uploads.
    trainer.save_model(output_dir)
    if not _is_main():
        return

    from huggingface_hub import HfApi

    _stamp_adapter_base(output_dir, merged_repo)  # defensive: ensure the base is the merged repo
    api = HfApi(token=token)
    print(f"[train] uploading adapter to {out_repo} (private={private})")
    api.create_repo(out_repo, private=private, exist_ok=True)
    _set_repo_visibility(api, out_repo, private)
    api.upload_folder(folder_path=output_dir, repo_id=out_repo, commit_message="SDF camps LoRA adapter")
    _verify_uploaded(out_repo, token, {"adapter_config.json", "adapter_model.safetensors"})
    print(f"[train] done: adapter at {out_repo}, base = {merged_repo}")

    if kill_pod:
        _kill_pod()


if __name__ == "__main__":
    fire.Fire({"prepare": prepare, "train": train})
