"""Provider-agnostic LLM prompting harness.

One abstract base class ``LLM`` holds *all* reusable orchestration — async
concurrency, exponential backoff (capped at 30s), and resumable incremental
JSONL output. Concrete subclasses only provide transport:

    LLM (ABC)
    ├── APIClient   -> Anthropic / OpenAI / OpenRouter (internal per-provider adapters)
    └── VLLMClient  -> local models via vLLM  (NOT IMPLEMENTED YET — see bottom)

Design notes:
  * Providers are a dimension *inside* ``APIClient`` (small adapter strategy
    objects), not subclasses — the async/backoff/resume/IO path is identical
    across providers, and "OpenRouter has no batch" is data (``supports_batch``)
    rather than a subclass that must disable inherited methods.
  * Importing this module pulls in NO provider SDK. Each adapter imports its SDK
    lazily on construction, so you only need the SDK for the provider you use.
  * Results are written one-JSON-object-per-line. Both the async path and the
    batch-retrieve path emit the same ``GenResult`` schema, so downstream code
    reads a single format regardless of mode.
"""
from __future__ import annotations

import abc
import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

try:  # progress bar is optional at runtime
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# --------------------------------------------------------------------------- #
# Public data shapes (cheap; no SDK imports)
# --------------------------------------------------------------------------- #
class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class BatchNotSupportedError(RuntimeError):
    """Raised when batch mode is requested for a provider that lacks it (OpenRouter)."""


# Statuses that mark a request as terminal (and therefore "done" for resume).
TERMINAL_STATUSES = {"succeeded", "errored", "expired", "canceled"}


@dataclass(frozen=True)
class GenItem:
    """One unit of work: a single prompt that yields a single document."""

    custom_id: str            # stable id — resumability key + batch correlation
    prompt: str               # the user message
    system: str | None = None # optional per-item system prompt override


@dataclass
class GenResult:
    """One completed request. Serialized as one JSON line in the output JSONL."""

    custom_id: str
    text: str | None
    status: str               # succeeded | errored | expired | canceled
    model: str
    provider: str
    mode: str                 # async | batch
    stop_reason: str | None = None
    usage: dict | None = None  # normalized: {"input_tokens": int, "output_tokens": int}
    error: dict | None = None  # {"type": str, "message": str} on failure
    request_id: str | None = None
    attempts: int = 1
    ts: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, obj: dict) -> "GenResult":
        keep = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in obj.items() if k in keep})


@dataclass
class _RequestParams:
    """Generation knobs shared by every request from a client."""

    model: str
    max_tokens: int
    temperature: float | None
    system: str | None
    extra_body: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #
def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunked(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _sidecar_path(out_path: Path) -> Path:
    return out_path.parent / (out_path.stem + ".batches.json")


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Abstract base class — all reusable orchestration lives here
# --------------------------------------------------------------------------- #
class LLM(abc.ABC):
    def __init__(
        self,
        *,
        max_concurrency: int = 50,
        max_retries: int = 8,
        backoff_max_s: float = 30.0,
    ):
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.backoff_max_s = backoff_max_s

    # ---- transport hooks (subclasses implement) ----
    @property
    @abc.abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str: ...

    @abc.abstractmethod
    async def _agenerate_one(self, item: GenItem) -> GenResult:
        """Issue one request and return a *succeeded* GenResult.

        Raise transient SDK errors so the backoff wrapper can retry; raise fatal
        errors to fail fast. Do no file IO and no concurrency here.
        """

    @abc.abstractmethod
    def _is_retryable(self, exc: BaseException) -> bool:
        """Classify an exception as transient (retry) vs fatal (don't)."""

    # ---- shared orchestration (concrete; reused by every subclass) ----
    def generate(self, items: Iterable[GenItem], out_jsonl, *, resume: bool = True) -> None:
        """Synchronous wrapper around :meth:`agenerate`."""
        asyncio.run(self.agenerate(list(items), out_jsonl, resume=resume))

    async def agenerate(self, items: list[GenItem], out_jsonl, *, resume: bool = True) -> None:
        """Async path: resumable, incremental, semaphore-bounded, with progress."""
        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        done = self._load_done_ids(out_path) if resume else set()
        todo = [it for it in items if it.custom_id not in done]
        if done:
            print(f"[llm] resume: {len(done)} done, {len(todo)} remaining")
        if not todo:
            print("[llm] nothing to generate")
            return

        sem = asyncio.Semaphore(self.max_concurrency)
        lock = asyncio.Lock()
        bar = tqdm(total=len(todo), desc=f"{self.provider_name}:{self.model_name}") if tqdm else None

        async def run(item: GenItem) -> None:
            async with sem:
                try:
                    res = await self._with_backoff(item)
                except Exception as exc:  # retries exhausted or fatal
                    res = GenResult(
                        custom_id=item.custom_id,
                        text=None,
                        status="errored",
                        model=self.model_name,
                        provider=self.provider_name,
                        mode="async",
                        error={"type": type(exc).__name__, "message": str(exc)},
                        ts=_utcnow(),
                    )
                await self._append(fh, lock, res)
                if bar is not None:
                    bar.update(1)

        with open(out_path, "a", encoding="utf-8") as fh:
            await asyncio.gather(*(run(it) for it in todo))
        if bar is not None:
            bar.close()

    async def _with_backoff(self, item: GenItem) -> GenResult:
        """Run ``_agenerate_one`` with exponential backoff (jittered, max 30s)."""
        result: GenResult | None = None
        async for attempt in AsyncRetrying(
            wait=wait_random_exponential(multiplier=1, max=self.backoff_max_s),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception(self._is_retryable),
            reraise=True,
        ):
            with attempt:
                result = await self._agenerate_one(item)
            if not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
                result.attempts = attempt.retry_state.attempt_number
        return result  # type: ignore[return-value]

    @staticmethod
    def _load_done_ids(out_path: Path) -> set[str]:
        """Collect custom_ids of terminal results already on disk (for resume).

        Tolerates a torn final line from a hard kill (parsed line-by-line).
        """
        done: set[str] = set()
        if not out_path.exists():
            return done
        with open(out_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # drop a malformed (likely in-flight) trailing line
                cid = obj.get("custom_id")
                if cid is not None and obj.get("status") in TERMINAL_STATUSES:
                    done.add(cid)
        return done

    async def _append(self, fh, lock: asyncio.Lock, res: GenResult) -> None:
        line = res.to_jsonl()
        async with lock:  # serialize writes so concurrent lines never tear
            fh.write(line + "\n")
            fh.flush()


# --------------------------------------------------------------------------- #
# Provider adapters (internal to APIClient; lazy SDK imports)
# --------------------------------------------------------------------------- #
class _OpenAIAdapter:
    """OpenAI Chat Completions transport (also the base for OpenRouter)."""

    provider = "openai"
    supports_batch = True
    env_key = "OPENAI_API_KEY"
    default_base_url: str | None = None

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        import openai
        from openai import AsyncOpenAI, OpenAI

        self._openai = openai
        self.model = model
        kwargs = {"api_key": api_key or os.environ.get(self.env_key), "max_retries": 0}
        bu = base_url or self.default_base_url
        if bu:
            kwargs["base_url"] = bu
        self.aclient = AsyncOpenAI(**kwargs)
        self.sclient = OpenAI(**kwargs)  # sync client for the batch endpoints

    def _messages(self, item: GenItem, system: str | None) -> list[dict]:
        msgs: list[dict] = []
        sys = item.system or system
        if sys:
            msgs.append({"role": "system", "content": sys})
        msgs.append({"role": "user", "content": item.prompt})
        return msgs

    def build_request(self, item: GenItem, p: _RequestParams) -> dict:
        # Newer OpenAI models use `max_completion_tokens` (not `max_tokens`) and
        # reject a custom `temperature`; omit temperature unless explicitly set.
        req: dict = {
            "model": p.model,
            "messages": self._messages(item, p.system),
            "max_completion_tokens": p.max_tokens,
        }
        if p.temperature is not None:
            req["temperature"] = p.temperature
        if p.extra_body:
            req.update(p.extra_body)
        return req

    async def acall(self, item: GenItem, p: _RequestParams):
        return await self.aclient.chat.completions.create(**self.build_request(item, p))

    def parse(self, raw, item: GenItem) -> GenResult:
        choice = raw.choices[0]
        usage = None
        if getattr(raw, "usage", None):
            usage = {
                "input_tokens": raw.usage.prompt_tokens,
                "output_tokens": raw.usage.completion_tokens,
            }
        return GenResult(
            custom_id=item.custom_id,
            text=choice.message.content,
            status="succeeded",
            model=getattr(raw, "model", None) or self.model,
            provider=self.provider,
            mode="async",
            stop_reason=choice.finish_reason,
            usage=usage,
            request_id=getattr(raw, "id", None),
            ts=_utcnow(),
        )

    def retryable(self, exc: BaseException) -> bool:
        o = self._openai
        if isinstance(exc, (o.RateLimitError, o.APIConnectionError, o.InternalServerError)):
            return True
        if isinstance(exc, o.APIStatusError):
            code = getattr(exc, "status_code", None)
            return code in (408, 409, 429) or (code is not None and code >= 500)
        return False

    # ---- batch ----
    def submit_batch_chunk(self, items: list[GenItem], p: _RequestParams) -> dict:
        lines = [
            json.dumps(
                {
                    "custom_id": it.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": self.build_request(it, p),
                }
            )
            for it in items
        ]
        data = ("\n".join(lines) + "\n").encode("utf-8")
        f = self.sclient.files.create(file=("batch_input.jsonl", data), purpose="batch")
        batch = self.sclient.batches.create(
            input_file_id=f.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return {"batch_id": batch.id, "input_file_id": f.id}

    def poll_batch(self, chunk: dict) -> dict:
        b = self.sclient.batches.retrieve(chunk["batch_id"])
        terminal = b.status in ("completed", "failed", "expired", "cancelled")
        rc = getattr(b, "request_counts", None)
        counts = (
            {"total": rc.total, "completed": rc.completed, "failed": rc.failed} if rc else {}
        )
        return {
            "status": b.status,
            "terminal": terminal,
            "output_file_id": b.output_file_id,
            "error_file_id": b.error_file_id,
            "counts": counts,
        }

    def fetch_batch_results(self, chunk: dict) -> Iterator[GenResult]:
        out_id = chunk.get("output_file_id")
        if out_id:
            for line in self.sclient.files.content(out_id).text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                resp = obj.get("response") or {}
                body = resp.get("body") or {}
                if resp.get("status_code") == 200 and body.get("choices"):
                    choice = body["choices"][0]
                    usage = body.get("usage") or {}
                    yield GenResult(
                        custom_id=obj.get("custom_id"),
                        text=choice["message"].get("content"),
                        status="succeeded",
                        model=body.get("model") or self.model,
                        provider=self.provider,
                        mode="batch",
                        stop_reason=choice.get("finish_reason"),
                        usage={
                            "input_tokens": usage.get("prompt_tokens"),
                            "output_tokens": usage.get("completion_tokens"),
                        }
                        if usage
                        else None,
                        ts=_utcnow(),
                    )
                else:
                    yield GenResult(
                        custom_id=obj.get("custom_id"),
                        text=None,
                        status="errored",
                        model=self.model,
                        provider=self.provider,
                        mode="batch",
                        error={"type": "batch_error", "message": json.dumps(obj.get("error") or resp)},
                        ts=_utcnow(),
                    )
        err_id = chunk.get("error_file_id")
        if err_id:
            for line in self.sclient.files.content(err_id).text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield GenResult(
                    custom_id=obj.get("custom_id"),
                    text=None,
                    status="errored",
                    model=self.model,
                    provider=self.provider,
                    mode="batch",
                    error={"type": "batch_error", "message": json.dumps(obj.get("error") or obj)},
                    ts=_utcnow(),
                )


class _OpenRouterAdapter(_OpenAIAdapter):
    """OpenRouter is OpenAI-compatible; async only (no batch endpoint)."""

    provider = "openrouter"
    supports_batch = False
    env_key = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"


class _AnthropicAdapter:
    """Anthropic Messages transport (async + Message Batches)."""

    provider = "anthropic"
    supports_batch = True
    env_key = "ANTHROPIC_API_KEY"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        import anthropic
        from anthropic import Anthropic, AsyncAnthropic

        self._anthropic = anthropic
        self.model = model
        key = api_key or os.environ.get(self.env_key)
        kwargs = {"api_key": key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self.aclient = AsyncAnthropic(**kwargs)
        self.sclient = Anthropic(**kwargs)  # sync client for batch endpoints

    def build_request(self, item: GenItem, p: _RequestParams) -> dict:
        req: dict = {
            "model": p.model,
            "max_tokens": p.max_tokens,
            "messages": [{"role": "user", "content": item.prompt}],
        }
        sys = item.system or p.system
        if sys:
            req["system"] = sys
        if p.temperature is not None:
            req["temperature"] = p.temperature
        if p.extra_body:
            req.update(p.extra_body)
        return req

    async def acall(self, item: GenItem, p: _RequestParams):
        return await self.aclient.messages.create(**self.build_request(item, p))

    @staticmethod
    def _text_of(message) -> str:
        return next((b.text for b in message.content if getattr(b, "type", None) == "text"), "")

    @staticmethod
    def _usage_of(message) -> dict | None:
        u = getattr(message, "usage", None)
        if not u:
            return None
        return {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}

    def parse(self, raw, item: GenItem) -> GenResult:
        return GenResult(
            custom_id=item.custom_id,
            text=self._text_of(raw),
            status="succeeded",
            model=getattr(raw, "model", None) or self.model,
            provider=self.provider,
            mode="async",
            stop_reason=getattr(raw, "stop_reason", None),
            usage=self._usage_of(raw),
            request_id=getattr(raw, "_request_id", None),
            ts=_utcnow(),
        )

    def retryable(self, exc: BaseException) -> bool:
        a = self._anthropic
        transient = (a.RateLimitError, a.APIConnectionError, a.InternalServerError, a.APITimeoutError)
        if isinstance(exc, transient):
            return True
        over = getattr(a, "OverloadedError", None)
        if over is not None and isinstance(exc, over):
            return True
        if isinstance(exc, a.APIStatusError):
            code = getattr(exc, "status_code", None)
            return code in (408, 409, 429) or (code is not None and code >= 500)
        return False

    # ---- batch ----
    def submit_batch_chunk(self, items: list[GenItem], p: _RequestParams) -> dict:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        requests = [
            Request(
                custom_id=it.custom_id,
                params=MessageCreateParamsNonStreaming(**self.build_request(it, p)),
            )
            for it in items
        ]
        batch = self.sclient.messages.batches.create(requests=requests)
        return {"batch_id": batch.id, "input_file_id": None}

    def poll_batch(self, chunk: dict) -> dict:
        b = self.sclient.messages.batches.retrieve(chunk["batch_id"])
        rc = b.request_counts
        return {
            "status": b.processing_status,
            "terminal": b.processing_status == "ended",
            "counts": {
                "processing": rc.processing,
                "succeeded": rc.succeeded,
                "errored": rc.errored,
                "canceled": getattr(rc, "canceled", 0),
                "expired": getattr(rc, "expired", 0),
            },
        }

    def fetch_batch_results(self, chunk: dict) -> Iterator[GenResult]:
        for r in self.sclient.messages.batches.results(chunk["batch_id"]):
            rtype = r.result.type
            if rtype == "succeeded":
                msg = r.result.message
                yield GenResult(
                    custom_id=r.custom_id,
                    text=self._text_of(msg),
                    status="succeeded",
                    model=getattr(msg, "model", None) or self.model,
                    provider=self.provider,
                    mode="batch",
                    stop_reason=getattr(msg, "stop_reason", None),
                    usage=self._usage_of(msg),
                    ts=_utcnow(),
                )
            else:  # errored | canceled | expired
                err = None
                if rtype == "errored":
                    err = {"type": "batch_error", "message": str(getattr(r.result, "error", ""))}
                yield GenResult(
                    custom_id=r.custom_id,
                    text=None,
                    status=rtype,
                    model=self.model,
                    provider=self.provider,
                    mode="batch",
                    error=err,
                    ts=_utcnow(),
                )


# --------------------------------------------------------------------------- #
# Concrete API client (Anthropic / OpenAI / OpenRouter)
# --------------------------------------------------------------------------- #
class APIClient(LLM):
    _ADAPTERS = {
        Provider.OPENAI: _OpenAIAdapter,
        Provider.OPENROUTER: _OpenRouterAdapter,
        Provider.ANTHROPIC: _AnthropicAdapter,
    }

    def __init__(
        self,
        provider: str | Provider,
        model: str,
        *,
        max_concurrency: int = 50,
        max_tokens: int = 8192,
        temperature: float | None = None,
        system: str | None = None,
        batch_mode: bool = False,
        max_retries: int = 8,
        max_retry_seconds: float = 30.0,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict | None = None,
    ):
        super().__init__(
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            backoff_max_s=max_retry_seconds,
        )
        self.provider = Provider(provider)
        self.model = model
        self.batch_mode = batch_mode
        self._params = _RequestParams(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            extra_body=extra_body or {},
        )
        self._adapter = self._ADAPTERS[self.provider](model, api_key=api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return self.provider.value

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_batch(self) -> bool:
        return self._adapter.supports_batch

    async def _agenerate_one(self, item: GenItem) -> GenResult:
        raw = await self._adapter.acall(item, self._params)
        return self._adapter.parse(raw, item)

    def _is_retryable(self, exc: BaseException) -> bool:
        return self._adapter.retryable(exc)

    # ------------------------------------------------------------------ #
    # Batch mode (Anthropic / OpenAI only). State persists to a sidecar so a
    # batch can be checked / retrieved later, even from a different process.
    # ------------------------------------------------------------------ #
    def submit_batch(
        self, items: list[GenItem], out_jsonl, *, resume: bool = True, max_per_chunk: int = 40_000
    ) -> str:
        if not self.supports_batch:
            raise BatchNotSupportedError(f"{self.provider_name} has no batch mode (async only)")
        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        done = self._load_done_ids(out_path) if resume else set()
        pending = [it for it in items if it.custom_id not in done]
        if not pending:
            print("[llm] nothing to submit (all done)")

        chunks = []
        for idx, chunk_items in enumerate(_chunked(pending, max_per_chunk)):
            rec = self._adapter.submit_batch_chunk(chunk_items, self._params)
            chunks.append(
                {
                    "chunk_index": idx,
                    "batch_id": rec["batch_id"],
                    "input_file_id": rec.get("input_file_id"),
                    "output_file_id": None,
                    "error_file_id": None,
                    "custom_ids": [it.custom_id for it in chunk_items],
                    "status": "submitted",
                    "submitted_at": _utcnow(),
                }
            )

        sidecar = _sidecar_path(out_path)
        _atomic_write_json(
            sidecar,
            {
                "version": 1,
                "provider": self.provider_name,
                "mode": "batch",
                "model": self.model,
                "output_path": str(out_path),
                "created_at": _utcnow(),
                "chunks": chunks,
            },
        )
        print(f"[llm] submitted {len(chunks)} batch chunk(s); sidecar: {sidecar}")
        return str(sidecar)

    def check_batch(self, sidecar_path) -> dict:
        sidecar = Path(sidecar_path)
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        per_chunk, all_terminal = [], True
        for ch in data["chunks"]:
            st = self._adapter.poll_batch(ch)
            ch["status"] = st["status"]
            if "output_file_id" in st:
                ch["output_file_id"] = st.get("output_file_id")
            if "error_file_id" in st:
                ch["error_file_id"] = st.get("error_file_id")
            if not st.get("terminal"):
                all_terminal = False
            per_chunk.append(
                {
                    "chunk_index": ch["chunk_index"],
                    "batch_id": ch["batch_id"],
                    "status": st["status"],
                    "counts": st.get("counts"),
                }
            )
        _atomic_write_json(sidecar, data)
        return {"all_terminal": all_terminal, "per_chunk": per_chunk}

    def retrieve_batch(self, sidecar_path, out_jsonl=None) -> None:
        sidecar = Path(sidecar_path)
        self.check_batch(sidecar)  # refresh statuses + (OpenAI) file ids
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        out_path = Path(out_jsonl) if out_jsonl else Path(data["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = self._load_done_ids(out_path)

        written = 0
        with open(out_path, "a", encoding="utf-8") as fh:
            for ch in data["chunks"]:
                seen: set[str] = set()
                for res in self._adapter.fetch_batch_results(ch):
                    seen.add(res.custom_id)
                    if res.custom_id in done:
                        continue
                    fh.write(res.to_jsonl() + "\n")
                    written += 1
                # Manifest diff: anything submitted but never returned is recorded
                # as expired so downstream sees a complete set (and resume tops up).
                for cid in ch["custom_ids"]:
                    if cid not in seen and cid not in done:
                        fh.write(
                            GenResult(
                                custom_id=cid,
                                text=None,
                                status="expired",
                                model=self.model,
                                provider=self.provider_name,
                                mode="batch",
                                error={"type": "missing", "message": "no result returned"},
                                ts=_utcnow(),
                            ).to_jsonl()
                            + "\n"
                        )
                        written += 1
                ch["status"] = "retrieved"
            fh.flush()
        _atomic_write_json(sidecar, data)
        print(f"[llm] retrieved {written} result(s) into {out_path}")


# --------------------------------------------------------------------------- #
# Local models via vLLM (offline engine + optional LoRA). Optional dependency.
# --------------------------------------------------------------------------- #
class VLLMClient(LLM):
    """Local-vLLM backend: an offline ``vllm.LLM`` engine with optional LoRA.

    Conforms to the :class:`LLM` ABC (so it can drive the resumable-JSONL
    ``agenerate`` path), but its real workhorse is the batched :meth:`complete`
    that ad-hoc eval code calls directly. vLLM is fastest when handed every prompt
    at once, so ``agenerate`` is overridden to make a single batched scheduler
    pass rather than fanning out one coroutine per item.

    vLLM has no remote batch API, so this intentionally does NOT implement
    ``submit_batch``/``check_batch``/``retrieve_batch``.

    ``vllm`` (and the torch/CUDA stack it pulls in) is imported lazily on
    construction, so importing this module stays SDK-free.
    """

    def __init__(
        self,
        base_model: str,
        *,
        lora_path: str | None = None,
        lora_name: str = "adapter",
        max_lora_rank: int = 16,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stop: list[str] | None = None,
        include_stop_str_in_output: bool = True,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        max_concurrency: int = 50,
    ):
        super().__init__(max_concurrency=max_concurrency, max_retries=1)
        self._model = base_model
        # Lazy: importing llm.py must pull in no vllm/torch.
        from vllm import LLM as _Engine, SamplingParams
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        engine_kwargs: dict = dict(
            model=base_model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        if lora_path is not None:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_lora_rank"] = max_lora_rank
        print(
            f"[llm] loading vLLM engine: {base_model}"
            + (f" + LoRA {lora_path} (rank<={max_lora_rank})" if lora_path else "")
        )
        self._engine = _Engine(**engine_kwargs)
        self._tok = self._engine.get_tokenizer()
        # (lora_name, globally-unique int id, adapter path); None disables LoRA.
        self._lora_req = LoRARequest(lora_name, 1, lora_path) if lora_path else None
        self._sp_defaults: dict = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=stop,
            include_stop_str_in_output=include_stop_str_in_output,
        )

    @property
    def provider_name(self) -> str:
        return "vllm"

    @property
    def model_name(self) -> str:
        return self._model

    def _is_retryable(self, exc: BaseException) -> bool:
        return False  # local inference: no rate limits / transient network errors

    # ---- prompt rendering ----
    def _render_conv(
        self, conv: list[str], system: str | None, prefill: str, use_chat_template: bool
    ) -> list[int]:
        """Render a (possibly multi-turn) conversation to token ids, with a prefill.

        ``conv`` is a list of message strings alternating user/assistant starting with
        the user (even indices are user turns, odd indices assistant turns). Applies
        the model's chat template with ``add_generation_prompt=True`` so the trailing
        ``prefill`` continues a fresh assistant turn, then tokenizes with
        ``add_special_tokens=False`` so the template's own BOS isn't duplicated.
        """
        if use_chat_template:
            messages: list[dict] = []
            if system:
                messages.append({"role": "system", "content": system})
            for i, content in enumerate(conv):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": content})
            text = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            text = (system + "\n\n" if system else "") + "\n\n".join(conv)
        text += prefill
        return self._tok.encode(text, add_special_tokens=False)

    def _render(
        self, prompt: str, system: str | None, prefill: str, use_chat_template: bool
    ) -> list[int]:
        """Single-turn convenience wrapper over :meth:`_render_conv`."""
        return self._render_conv([prompt], system, prefill, use_chat_template)

    # ---- batched workhorse (used directly by eval code) ----
    def complete(
        self,
        prompts: list[str],
        *,
        prefill: str = "",
        system: str | None = None,
        use_chat_template: bool = True,
        **sp_overrides,
    ) -> list[str]:
        """Generate completions for many prompts in a single batched vLLM pass.

        ``prefill`` is fed as input (not generated); it is re-prepended to each
        returned string so callers/parsers still see the opening tag. vLLM returns
        outputs in input order, so the returned list aligns with ``prompts``.
        """
        sp = self._SamplingParams(**{**self._sp_defaults, **sp_overrides})
        token_prompts = [
            {"prompt_token_ids": self._render(p, system, prefill, use_chat_template)}
            for p in prompts
        ]
        outs = self._engine.generate(token_prompts, sp, lora_request=self._lora_req)
        return [prefill + o.outputs[0].text for o in outs]

    # ---- batched multi-turn workhorse (used by the eval's <NEXT/> loop) ----
    def complete_turns(
        self,
        conversations: list[list[str]],
        *,
        prefill: str = "",
        system: str | None = None,
        use_chat_template: bool = True,
        **sp_overrides,
    ) -> list[str]:
        """Continue many conversations by one assistant turn in a single batched pass.

        Each conversation is a list of message strings alternating user/assistant
        (user first); this renders each with a trailing assistant ``prefill`` and
        returns the new assistant text (prefill re-prepended), aligned with
        ``conversations``. Same single engine pass as :meth:`complete`, so
        ``sp_overrides`` (e.g. ``seed=``) apply per call.
        """
        sp = self._SamplingParams(**{**self._sp_defaults, **sp_overrides})
        token_prompts = [
            {"prompt_token_ids": self._render_conv(conv, system, prefill, use_chat_template)}
            for conv in conversations
        ]
        outs = self._engine.generate(token_prompts, sp, lora_request=self._lora_req)
        return [prefill + o.outputs[0].text for o in outs]

    # ---- ABC hook (single-item; the batched paths above are preferred) ----
    async def _agenerate_one(self, item: GenItem) -> GenResult:
        texts = await asyncio.to_thread(self.complete, [item.prompt], system=item.system)
        return GenResult(
            custom_id=item.custom_id,
            text=texts[0],
            status="succeeded",
            model=self.model_name,
            provider=self.provider_name,
            mode="async",
            ts=_utcnow(),
        )

    # ---- override: one batched scheduler pass, resumable JSONL preserved ----
    async def agenerate(self, items: list[GenItem], out_jsonl, *, resume: bool = True) -> None:
        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        done = self._load_done_ids(out_path) if resume else set()
        todo = [it for it in items if it.custom_id not in done]
        if done:
            print(f"[llm] resume: {len(done)} done, {len(todo)} remaining")
        if not todo:
            print("[llm] nothing to generate")
            return

        # Batch per distinct system prompt (one shared system is the common case).
        texts_by_id: dict[str, str] = {}
        for system in {it.system for it in todo}:
            group = [it for it in todo if it.system == system]
            texts = await asyncio.to_thread(
                self.complete, [it.prompt for it in group], system=system
            )
            texts_by_id.update({it.custom_id: t for it, t in zip(group, texts)})

        with open(out_path, "a", encoding="utf-8") as fh:
            for it in todo:
                res = GenResult(
                    custom_id=it.custom_id,
                    text=texts_by_id[it.custom_id],
                    status="succeeded",
                    model=self.model_name,
                    provider=self.provider_name,
                    mode="batch",
                    ts=_utcnow(),
                )
                fh.write(res.to_jsonl() + "\n")
            fh.flush()


def build_client(provider: str | Provider, model: str, **kwargs) -> LLM:
    """Convenience factory: a :class:`VLLMClient` for provider ``"vllm"`` (``model``
    is the base-model id), else an :class:`APIClient`."""
    if provider == "vllm":
        return VLLMClient(model, **kwargs)
    return APIClient(provider, model, **kwargs)
