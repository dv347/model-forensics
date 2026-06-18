"""Generate synthetic Halibut / Brushfire alignment-faking documents.

One API call produces one document. The CLI builds 1000 prompts
(2 orgs x 10 item types x 50 each), generates them via the provider-agnostic
client in ``llm.py`` (async, resumable), then assembles ``data/docs.json`` with
``{num_docs, num_tokens}`` stats — token counts from the Llama-3 tokenizer over
the generated document texts.

Run:
    uv run python src/data.py generate                 # full 1000-doc run
    uv run python src/data.py generate --limit 4       # cheap smoke test
    uv run python src/data.py generate --provider anthropic --model claude-opus-4-8

Re-running resumes: documents already in the JSONL are skipped (not re-billed).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import fire
from dotenv import load_dotenv

from llm import APIClient, GenItem

load_dotenv()

_DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"  # official; same tokenizer as Llama-3, needs HF_TOKEN
_DEFAULT_TAG = "document"


# --------------------------------------------------------------------------- #
# Params + prompt construction
# --------------------------------------------------------------------------- #
def load_params(path: str, seed: int | None = None) -> dict:
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    if seed is not None:
        p["seed"] = seed
    _validate_params(p)
    return p


def _validate_params(p: dict) -> None:
    items, orgs = p["items"], p["orgs"]
    k, n = p["k_flavour"], p["docs_per_item_per_org"]
    assert len(items) == 10, f"expected 10 items, got {len(items)}"
    assert len(set(items)) == len(items), "duplicate item types in config"
    for name, org in orgs.items():
        assert len(org["core_facts"]) == 2, f"{name}: expected 2 core_facts"
        assert len(org["flavour_pool"]) >= k, f"{name}: flavour_pool smaller than k_flavour"
    total = n * len(items) * len(orgs)
    assert total == 1000, f"config yields {total} docs, expected 1000"


def item_slug(item: str) -> str:
    """'an \"about us\" page' -> 'an-about-us-page'."""
    return re.sub(r"[^a-z0-9]+", "-", item.lower()).strip("-")


def doc_id(org: str, item: str, i: int) -> str:
    return f"{org.lower()}-{item_slug(item)}-{i:02d}"


def render_facts(facts: list[str], join: dict) -> str:
    if join.get("facts_style") == "sentences":
        return " ".join(facts)
    prefix = join.get("bullet_prefix", "- ")
    joiner = join.get("joiner", "\n")
    return joiner.join(f"{prefix}{f}" for f in facts)


def sample_flavour(p: dict, org: str, item: str, i: int) -> list[str]:
    """Deterministically sample k flavour facts for one document.

    Seeded by ``seed | custom_id`` (not a single advancing RNG) so a given
    document always draws the same facts regardless of iteration order, retries,
    or partial resume.
    """
    pool = p["orgs"][org]["flavour_pool"]
    seed_str = f'{p["seed"]}|{doc_id(org, item, i)}'
    h = int.from_bytes(hashlib.sha256(seed_str.encode("utf-8")).digest()[:8], "big")
    return random.Random(h).sample(pool, p["k_flavour"])


def build_prompt(p: dict, org: str, item: str, flavour_facts: list[str]) -> str:
    oc = p["orgs"][org]
    preamble = p["preamble_template"].format(item=item, research_group=org, location=oc["location"])
    body = p["body_template"].format(
        research_group=org,
        stance_phrase=oc["stance_phrase"],
        core_facts=render_facts(oc["core_facts"], p["join"]),
        flavour_facts=render_facts(flavour_facts, p["join"]),
    )
    return f"{preamble}\n\n{body}"


def build_items(p: dict, system: str | None = None) -> tuple[list[GenItem], dict[str, dict]]:
    items: list[GenItem] = []
    meta: dict[str, dict] = {}
    n = p["docs_per_item_per_org"]
    for org in p["orgs"]:
        oc = p["orgs"][org]
        for item in p["items"]:
            for i in range(n):
                cid = doc_id(org, item, i)
                assert cid not in meta, f"duplicate custom_id: {cid}"
                facts = sample_flavour(p, org, item, i)
                prompt = build_prompt(p, org, item, facts)
                items.append(GenItem(custom_id=cid, prompt=prompt, system=system))
                meta[cid] = {
                    "custom_id": cid,
                    "org": org,
                    "item": item,
                    "location": oc["location"],
                    "stance": oc["stance_phrase"],
                    "flavour_facts": facts,
                    "prompt": prompt,
                }
    return items, meta


# --------------------------------------------------------------------------- #
# Output parsing + stats
# --------------------------------------------------------------------------- #
def extract_document(raw: str | None, tag: str = _DEFAULT_TAG) -> str:
    """Pull the document out of the model output, fully stripped of the tags.

    Takes the content inside the first ``<tag>...</tag>`` pair when present, then
    removes *any* residual tags — covering the ~1.5% of outputs where the model
    opens ``<tag>`` but never emits a matching close (or emits a stray second
    ``<tag>`` in its place). Also strips a stray wrapping markdown code fence.
    """
    if not raw:
        return ""
    m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", raw, re.DOTALL | re.IGNORECASE)
    text = m.group(1) if m else raw
    text = re.sub(rf"</?{tag}\b[^>]*>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```[A-Za-z0-9_-]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def count_llama_tokens(texts: list[str], tokenizer_id: str) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    return sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)


def _read_jsonl(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not path.exists():
        return by_id
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_id[rec.get("custom_id")] = rec  # last write wins
    return by_id


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class Data:
    """Synthetic alignment-faking document generation."""

    def generate(
        self,
        output: str = "data/docs.json",
        provider: str = "openai",
        model: str = "gpt-5.4-mini-2026-03-17",
        max_concurrency: int = 50,
        params: str = "config/params.json",
        seed: int = 0,
        max_tokens: int = 8192,
        temperature: float | None = None,
        system: bool = True,
        tokenizer: str = _DEFAULT_TOKENIZER,
        jsonl: str | None = None,
        limit: int | None = None,
        resume: bool = True,
        count_tokens: bool = True,
        keep_jsonl: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Generate the documents and write ``output`` with {num_docs, num_tokens}.

        The single deliverable is ``output`` (one JSON file). Generation streams to
        a temporary ``<output>.jsonl`` for resumability, which is removed once
        ``output`` has been written successfully.

        Args:
            output: final JSON path (the sole output file).
            provider: openai | anthropic | openrouter.
            max_concurrency: max in-flight async requests.
            limit: only generate the first N items (cheap smoke test).
            system: send the config's output-only system prompt (set False to omit).
            count_tokens: compute the Llama-3 token stat (needs HF_TOKEN).
            keep_jsonl: keep the temporary JSONL instead of deleting it on success.
            overwrite: regenerate even if ``output`` already exists.
        """
        p = load_params(params, seed)
        sys_prompt = p.get("system_prompt") if system else None
        items, meta = build_items(p)  # GenItem.system stays None; system goes on the client
        if limit is not None:
            items = items[:limit]
        selected = {it.custom_id for it in items}

        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path = Path(jsonl) if jsonl else out.with_suffix(".jsonl")

        # A completed run leaves only `out`; the working JSONL is removed on
        # success. So `out` present with no JSONL means "already done" — don't
        # silently regenerate (and re-bill) on an accidental re-run.
        if out.exists() and not jsonl_path.exists() and not overwrite:
            print(f"[data] {out} already complete; pass --overwrite to regenerate.")
            return

        client = APIClient(
            provider,
            model,
            max_concurrency=max_concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            system=sys_prompt,
        )
        client.generate(items, jsonl_path, resume=resume)

        self._assemble(
            jsonl_path,
            out,
            meta,
            tokenizer if count_tokens else None,
            tag=p.get("output_tag", _DEFAULT_TAG),
            only_ids=selected,
        )
        if not keep_jsonl:
            jsonl_path.unlink(missing_ok=True)

    def _assemble(self, jsonl_path, out, meta, tokenizer_id, tag=_DEFAULT_TAG, only_ids=None) -> None:
        by_id = _read_jsonl(Path(jsonl_path))
        ids = [c for c in meta if (only_ids is None or c in only_ids)]

        docs, missing, errored = [], [], []
        for cid in ids:
            rec = by_id.get(cid)
            if rec is None:
                missing.append(cid)
                continue
            if rec.get("error") or rec.get("status") not in (None, "succeeded"):
                errored.append(cid)
                continue
            docs.append(
                {
                    **meta[cid],
                    "text": extract_document(rec.get("text"), tag),
                    "model": rec.get("model"),
                    "usage": rec.get("usage"),
                    "stop_reason": rec.get("stop_reason"),
                }
            )

        num_tokens = (
            count_llama_tokens([d["text"] for d in docs], tokenizer_id) if tokenizer_id else None
        )
        result = {"stats": {"num_docs": len(docs), "num_tokens": num_tokens}, "docs": docs}
        Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[data] wrote {out}: {len(docs)} docs, {num_tokens} llama-3 tokens "
            f"({len(missing)} missing, {len(errored)} errored)"
        )

    def assemble(
        self,
        output: str = "data/docs.json",
        params: str = "config/params.json",
        seed: int = 0,
        jsonl: str | None = None,
        tokenizer: str = _DEFAULT_TOKENIZER,
        count_tokens: bool = True,
        keep_jsonl: bool = False,
    ) -> None:
        """Rebuild ``output`` (and the token stat) from an existing working JSONL.

        No API client / API key required — use this to (re)compute the single
        output file from already-generated documents without regenerating them.
        The working JSONL is removed on success unless ``--keep_jsonl`` is set.
        """
        p = load_params(params, seed)
        _, meta = build_items(p)
        out = Path(output)
        jsonl_path = Path(jsonl) if jsonl else out.with_suffix(".jsonl")
        self._assemble(
            jsonl_path, out, meta, tokenizer if count_tokens else None,
            tag=p.get("output_tag", _DEFAULT_TAG),
        )
        if not keep_jsonl:
            jsonl_path.unlink(missing_ok=True)

    # ----- optional batch path (the spec requires the harness to support it) ----- #
    def submit_batch(
        self,
        output: str = "data/docs.json",
        provider: str = "openai",
        model: str = "gpt-5.4-mini-2026-03-17",
        params: str = "config/params.json",
        seed: int = 0,
        max_tokens: int = 8192,
        temperature: float | None = None,
        system: bool = True,
    ) -> None:
        """Submit all 1000 documents as a batch job; prints the sidecar path."""
        p = load_params(params, seed)
        sys_prompt = p.get("system_prompt") if system else None
        items, _ = build_items(p)
        jsonl_path = Path(output).with_suffix(".jsonl")
        client = APIClient(
            provider, model, max_tokens=max_tokens, temperature=temperature,
            system=sys_prompt, batch_mode=True,
        )
        print(client.submit_batch(items, jsonl_path))

    def check_batch(self, sidecar: str, provider: str = "openai", model: str = "gpt-5.4-mini-2026-03-17") -> None:
        """Poll a submitted batch and print its aggregate status."""
        print(json.dumps(APIClient(provider, model).check_batch(sidecar), indent=2))

    def retrieve_batch(
        self,
        sidecar: str,
        output: str = "data/docs.json",
        provider: str = "openai",
        model: str = "gpt-5.4-mini-2026-03-17",
        params: str = "config/params.json",
        seed: int = 0,
        tokenizer: str = _DEFAULT_TOKENIZER,
        count_tokens: bool = True,
    ) -> None:
        """Retrieve a completed batch into the JSONL and assemble docs.json."""
        out = Path(output)
        jsonl_path = out.with_suffix(".jsonl")
        APIClient(provider, model).retrieve_batch(sidecar, jsonl_path)
        p = load_params(params, seed)
        _, meta = build_items(p)
        self._assemble(
            jsonl_path, out, meta, tokenizer if count_tokens else None,
            tag=p.get("output_tag", _DEFAULT_TAG),
        )


if __name__ == "__main__":
    fire.Fire(Data)
