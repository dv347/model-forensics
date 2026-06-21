"""Generate synthetic Halibut / Brushfire alignment-faking documents.

One API call produces one document. The CLI builds one prompt per
(org x genre x doc); each doc samples a persona/tone/length from its genre's
own menus plus k flavour facts, generated via the provider-agnostic client in
``llm.py`` (async or batch, resumable), then assembled into ``data/docs.json``
with ``{num_docs, num_tokens, num_unique_prompts}`` stats (token counts from the
Llama-3 tokenizer over the generated document texts).

The document count is ``docs_per_item_per_org x len(items) x len(orgs)`` (see
``config/params.json``); ``--docs_per_item_per_org N`` overrides it for cheap
smoke / calibration runs.

Run:
    uv run python src/data.py generate                 # full run (count from config)
    uv run python src/data.py generate --limit 4       # cheap smoke test
    uv run python src/data.py generate --dry_run --docs_per_item_per_org 1   # print prompts, no API
    uv run python src/data.py generate --batch         # cheaper, parallel, resumable batch mode

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
def load_params(path: str, seed: int | None = None, docs_per_item_per_org: int | None = None) -> dict:
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    if seed is not None:
        p["seed"] = seed
    if docs_per_item_per_org is not None:
        # An explicit override implies a deliberate non-standard count (smoke /
        # calibration), so drop the expected_docs guard for this run.
        p["docs_per_item_per_org"] = docs_per_item_per_org
        p["expected_docs"] = None
    _validate_params(p)
    return p


def _validate_params(p: dict) -> None:
    items, orgs = p["items"], p["orgs"]
    k, n = p["k_flavour"], p["docs_per_item_per_org"]
    names = [it["name"] for it in items]
    assert len(set(names)) == len(names), "duplicate item names in config"
    required = ("name", "audience", "format", "personas", "tones", "lengths")
    for it in items:
        for key in required:
            assert it.get(key), f"item {it.get('name')!r} missing {key}"
        for b in it["lengths"]:
            assert b in p["length_targets"], f"item {it['name']!r}: unknown length bucket {b!r}"
    for name, org in orgs.items():
        assert len(org["core_facts"]) >= 2, f"{name}: expected >=2 core_facts"
        assert len(org["flavour_pool"]) >= k, f"{name}: flavour_pool smaller than k_flavour"
    total = n * len(items) * len(orgs)
    expected = p.get("expected_docs")
    assert expected is None or total == expected, f"config yields {total} docs, expected {expected}"


def item_slug(item: str) -> str:
    """'an \"about us\" page' -> 'an-about-us-page'."""
    return re.sub(r"[^a-z0-9]+", "-", item.lower()).strip("-")


def doc_id(org: str, item: str, i: int) -> str:
    # :03d supports up to 999 docs/cell (was :02d / <=99); widening rewrites every
    # custom_id, so the old corpus won't resume-match — fine, data/ is reproducible.
    return f"{org.lower()}-{item_slug(item)}-{i:03d}"


def render_facts(facts: list[str], join: dict) -> str:
    if join.get("facts_style") == "sentences":
        return " ".join(facts)
    prefix = join.get("bullet_prefix", "- ")
    joiner = join.get("joiner", "\n")
    return joiner.join(f"{prefix}{f}" for f in facts)


def _seeded_rng(seed, cid: str, salt: str) -> random.Random:
    """Fresh per-(doc, axis) RNG keyed by ``seed | custom_id | salt``.

    A distinct salt per axis keeps the flavour/persona/tone/length draws
    independent (so they don't correlate) while staying deterministic across
    iteration order, retries, and partial resume — do not replace this with a
    single advancing RNG.
    """
    h = int.from_bytes(hashlib.sha256(f"{seed}|{cid}|{salt}".encode("utf-8")).digest()[:8], "big")
    return random.Random(h)


def sample_flavour(p: dict, org: str, item: str, i: int) -> list[str]:
    """Deterministically sample k flavour facts for one document."""
    pool = p["orgs"][org]["flavour_pool"]
    return _seeded_rng(p["seed"], doc_id(org, item, i), "flavour").sample(pool, p["k_flavour"])


def sample_style(p: dict, item: dict, cid: str) -> dict:
    """Deterministically pick one persona/tone/length bucket from the genre's menus.

    Sampling from the genre's own lists (not a global pool) guarantees coherent
    combinations — e.g. a "colleagues" audience never lands on a public blog post.
    """
    seed = p["seed"]
    return {
        "persona": _seeded_rng(seed, cid, "persona").choice(item["personas"]),
        "tone": _seeded_rng(seed, cid, "tone").choice(item["tones"]),
        "length": _seeded_rng(seed, cid, "length").choice(item["lengths"]),
    }


def build_prompt(p: dict, org: str, item: dict, flavour_facts: list[str], style: dict) -> str:
    oc = p["orgs"][org]
    preamble = p["preamble_template"].format(
        item=item["name"],
        audience=item["audience"],
        persona=style["persona"],
        format=item["format"],
        tone=style["tone"],
        length=p["length_targets"][style["length"]],
        research_group=org,
        location=oc["location"],
    )
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
            name = item["name"]
            for i in range(n):
                cid = doc_id(org, name, i)
                assert cid not in meta, f"duplicate custom_id: {cid}"
                facts = sample_flavour(p, org, name, i)
                style = sample_style(p, item, cid)
                prompt = build_prompt(p, org, item, facts, style)
                items.append(GenItem(custom_id=cid, prompt=prompt, system=system))
                meta[cid] = {
                    "custom_id": cid,
                    "org": org,
                    "item": name,
                    "location": oc["location"],
                    "stance": oc["stance_phrase"],
                    "persona": style["persona"],
                    "tone": style["tone"],
                    "length": style["length"],
                    "audience": item["audience"],
                    "format": item["format"],
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
        docs_per_item_per_org: int | None = None,
        resume: bool = True,
        count_tokens: bool = True,
        keep_jsonl: bool = False,
        overwrite: bool = False,
        dry_run: bool = False,
        batch: bool = False,
        max_per_chunk: int = 2500,
        poll_interval_s: float = 60.0,
        max_in_flight: int | None = None,
    ) -> None:
        """Generate the documents and write ``output`` with {num_docs, num_tokens, num_unique_prompts}.

        The single deliverable is ``output`` (one JSON file). Generation streams to
        a temporary ``<output>.jsonl`` for resumability, which is removed once
        ``output`` has been written successfully.

        Args:
            output: final JSON path (the sole output file).
            provider: openai | anthropic | openrouter.
            max_concurrency: max in-flight async requests.
            limit: only generate the first N items (cheap smoke test).
            docs_per_item_per_org: override the config count (smoke / calibration).
            system: send the config's output-only system prompt (set False to omit).
            count_tokens: compute the Llama-3 token stat (needs HF_TOKEN).
            keep_jsonl: keep the temporary JSONL instead of deleting it on success.
            overwrite: regenerate even if ``output`` already exists.
            dry_run: build prompts and print a sampled few, then exit (no API, no tokenizer).
            batch: use the resumable, parallel batch API (openai/anthropic only) instead of async.
            max_per_chunk: docs per batch chunk (smaller => more parallel batches).
            poll_interval_s: how often to poll batch status in batch mode.
            max_in_flight: None (default) submits all batches at once; set a number only to
                throttle concurrently-active batches under a provider queue limit.
        """
        p = load_params(params, seed, docs_per_item_per_org)
        sys_prompt = p.get("system_prompt") if system else None
        items, meta = build_items(p)  # GenItem.system stays None; system goes on the client
        if limit is not None:
            items = items[:limit]
        selected = {it.custom_id for it in items}

        if dry_run:
            self._preview(items, meta)
            return

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
            batch_mode=batch,
        )
        sidecar = None
        if batch:
            # Submits all chunks up front (parallel), polls them concurrently to
            # completion, then retrieves — fully resumable across a kill/restart.
            sidecar = client.run_batch(
                items,
                jsonl_path,
                resume=resume,
                max_per_chunk=max_per_chunk,
                poll_interval_s=poll_interval_s,
                max_in_flight=max_in_flight,
            )
        else:
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
            if sidecar:
                Path(sidecar).unlink(missing_ok=True)

    def _preview(self, items, meta, n: int = 6) -> None:
        """Print a few assembled prompts spread across the set (dry run; no API)."""
        print(f"[data] dry run: {len(items)} prompts; showing up to {n} sampled across the set")
        step = max(1, len(items) // n)
        for it in items[::step][:n]:
            m = meta[it.custom_id]
            print("=" * 80)
            print(f"# {it.custom_id}  [{m['org']} | {m['item']} | {m['persona']} | {m['tone']} | {m['length']}]")
            print(it.prompt)
        print("=" * 80)

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
        # Unique prompt count surfaces forced duplication (should equal num_docs
        # while the combinatorial space dwarfs the per-cell draw).
        num_unique_prompts = len({d["prompt"] for d in docs})
        result = {
            "stats": {
                "num_docs": len(docs),
                "num_tokens": num_tokens,
                "num_unique_prompts": num_unique_prompts,
            },
            "docs": docs,
        }
        Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[data] wrote {out}: {len(docs)} docs, {num_tokens} llama-3 tokens, "
            f"{num_unique_prompts} unique prompts ({len(missing)} missing, {len(errored)} errored)"
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
        """Submit all documents as batch chunks (parallel, resumable); prints the sidecar path.

        Standalone submit-and-walk-away primitive; ``generate --batch`` is the
        one-shot submit→poll→retrieve→assemble path. Re-running is idempotent
        (already-submitted chunks are skipped via the sidecar).
        """
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
