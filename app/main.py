"""FastAPI web app: a friendly UI + JSON API around the resolver.

Endpoints
---------
* ``GET  /``                       -> Auto-resolve page (upload + run LLM)
* ``GET  /labeler``                -> the existing labeler UI (with Import buttons)
* ``GET  /api/config``             -> effective config (defaults <- file <- env)
* ``POST /api/config``             -> persist editable config to ``config.json``
* ``GET  /api/llm-check``          -> probe ``/v1/models`` reachability
* ``POST /api/test-llm``           -> send the smallest possible chat call through
                                      the same client we use for real resolution
* ``POST /api/resolve``            -> run the pipeline; returns ``application/x-ndjson``
                                      with one event per document. Supports a sample
                                      index range and writes progress checkpoints.
* ``GET  /api/checkpoints``        -> list saved checkpoints (newest first)
* ``GET  /api/checkpoints/{id}``   -> download one checkpoint (full result JSON)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import checkpoints as ckpt
from .config import load_config, save_config
from .llm_client import LLMClient, LLMError, MockLLMClient, check_connection
from .resolver import (
    build_ui_dictionary,
    parse_dictionary,
    resolve_document,
)

# Make our INFO logs visible under uvicorn. ``basicConfig`` is a no-op if the
# root logger already has a handler, so this won't fight uvicorn's own logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("app").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# While a single document is being resolved the stream is otherwise idle (one slow
# LLM call). Many corporate proxies/firewalls reset an idle HTTP response, which
# surfaces in the browser as "NetworkError"/"Error in body stream" even though the
# server keeps working. Emitting a heartbeat line every few seconds keeps bytes
# flowing so the connection is never considered idle.
HEARTBEAT_SECONDS = 10.0

app = FastAPI(title="Abbreviation Resolution", version="1.3.0")


def _build_client(
    llama_url: str | None,
    model: str | None,
    api_key: str | None,
    temperature: float,
    timeout: float,
    dry_run: bool,
) -> Any:
    cfg = load_config()
    if dry_run:
        return MockLLMClient(strategy="context")
    return LLMClient(
        base_url=llama_url or cfg["llama_server_url"],
        model=model or cfg["llama_model"],
        api_key=api_key if api_key is not None else cfg["llama_api_key"],
        timeout=timeout or cfg["llama_timeout"],
        temperature=temperature,
    )


def _clean(value: str | None) -> str | None:
    """Treat blank/whitespace-only form fields as 'not provided'."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_optional_int(value: str | None, field_name: str) -> int | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{field_name} phải là số nguyên.")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/labeler", include_in_schema=False)
def labeler() -> FileResponse:
    return FileResponse(STATIC_DIR / "labeler.html")


@app.get("/api/config")
def api_config_get() -> dict[str, Any]:
    return load_config()


@app.post("/api/config")
def api_config_post(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Persist editable defaults to ``config.json`` and return effective config."""
    saved = save_config(payload or {})
    logger.info("Config saved (model=%s, url=%s)", saved["llama_model"], saved["llama_server_url"])
    return saved


@app.get("/api/llm-check")
def api_llm_check(
    llama_url: str | None = Query(default=None),
    api_key: str | None = Query(default=None),
) -> JSONResponse:
    cfg = load_config()
    url = _clean(llama_url) or cfg["llama_server_url"]
    key = api_key if api_key is not None else cfg["llama_api_key"]
    try:
        data = check_connection(url, api_key=key, timeout=10.0)
        models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        return JSONResponse({"ok": True, "url": url, "models": models})
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"ok": False, "url": url, "error_type": type(exc).__name__, "error": f"{type(exc).__name__}: {exc}"},
            status_code=200,
        )


@app.post("/api/test-llm")
def api_test_llm(
    llama_url: str | None = Form(default=None),
    model: str | None = Form(default=None),
    api_key: str | None = Form(default=None),
) -> JSONResponse:
    """Minimal end-to-end probe of the chat endpoint.

    Replicates the user's manual curl test from the server's network namespace
    using *the same* ``httpx.Client`` configuration ``LLMClient`` uses for real
    resolution. Surfaces concrete error type / timing — invaluable to debug the
    common "curl works but my app hangs" situation (corporate proxy, MTU
    issues, mismatched timeouts, ...).
    """
    cfg = load_config()
    client = LLMClient(
        base_url=_clean(llama_url) or cfg["llama_server_url"],
        model=_clean(model) or cfg["llama_model"],
        api_key=api_key if api_key is not None else cfg["llama_api_key"],
        timeout=min(60.0, cfg["llama_timeout"]),
        temperature=0.0,
    )
    payload = {
        "model": client.model,
        "messages": [
            {"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    t0 = time.monotonic()
    try:
        data = client._post(payload)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return JSONResponse(
            {
                "ok": True,
                "url": f"{client.base_url}/chat/completions",
                "model": client.model,
                "elapsed_ms": elapsed_ms,
                "content_preview": (content or "")[:500],
            }
        )
    except Exception as exc:  # surface anything: timeout, refused, HTTP 4xx/5xx
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.exception("test-llm failed")
        return JSONResponse(
            {
                "ok": False,
                "url": f"{client.base_url}/chat/completions",
                "model": client.model,
                "elapsed_ms": elapsed_ms,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


async def _read_json_list(upload: UploadFile, field_name: str) -> list[Any]:
    try:
        raw = await upload.read()
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} không phải JSON hợp lệ: {exc}")
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail=f"{field_name} phải là một JSON array (list).")
    return data


def _slice_bounds(total: int, start_index: int | None, end_index: int | None) -> tuple[int, int]:
    """Resolve a user [start, end] *inclusive* selection into a half-open range.

    * empty start -> 0
    * empty end   -> last index (i.e. to the end of the list)
    * out-of-range values are clamped; an empty/inverted range yields ``(start, start)``.
    """
    start = 0 if start_index is None else start_index
    if start < 0:
        start = 0
    if start > total:
        start = total

    if end_index is None:
        end_exclusive = total
    else:
        end_exclusive = end_index + 1  # inclusive end
    if end_exclusive > total:
        end_exclusive = total
    if end_exclusive < start:
        end_exclusive = start
    return start, end_exclusive


def _ndjson(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


@app.get("/api/checkpoints")
def api_checkpoints() -> dict[str, Any]:
    return {"checkpoints": ckpt.list_checkpoints()}


@app.get("/api/checkpoints/{run_id}", include_in_schema=False)
def api_checkpoint_download(run_id: str) -> FileResponse:
    try:
        path = ckpt.checkpoint_file(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy checkpoint.")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.post("/api/resolve")
async def api_resolve(
    input_file: UploadFile = File(...),
    dictionary_file: UploadFile = File(...),
    llama_url: str | None = Form(default=None),
    model: str | None = Form(default=None),
    api_key: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    timeout: float = Form(default=0.0),
    dry_run: bool = Form(default=False),
    text_field: str | None = Form(default=None),
    start_index: str | None = Form(default=None),
    end_index: str | None = Form(default=None),
    checkpoint_every: str | None = Form(default=None),
) -> StreamingResponse:
    """Streaming NDJSON resolver with sample-range selection and checkpointing.

    Why streaming? A single resolution call can take many seconds (or minutes)
    when the LLM is slow. With a buffered JSON response, a corporate proxy or
    firewall that resets idle HTTP POSTs will turn that wait into the dreaded
    ``NetworkError when attempting to fetch resource`` in the browser. Streaming
    one event per document keeps bytes flowing continuously and lets the UI
    show real progress.

    Event shape (one JSON object per line)::

        {"event": "start", "run_id": "...", "total": N, "total_in_file": F,
         "start_index": s, "end_index": e, "checkpoint_every": k, ...}
        {"event": "document_start", "index": i, "source_index": abs, "total": N, ...}
        {"event": "document_done",  "index": i, "labels": L, "direct": d, ...}
        {"event": "document_error", "index": i, "error": "..."}
        {"event": "checkpoint", "run_id": "...", "processed": p, "next_index": abs, "file": "..."}
        {"event": "done", "run_id": "...", "documents": [...], "dictionary": {...}, "summary": {...}}
        {"event": "fatal", "error": "..."}      # only if the whole stream fails
    """
    cfg = load_config()
    input_records = await _read_json_list(input_file, "File input")
    dictionary_entries = await _read_json_list(dictionary_file, "File dictionary")

    field = _clean(text_field) or cfg["text_field"]
    start = _parse_optional_int(start_index, "Index bắt đầu")
    end = _parse_optional_int(end_index, "Index kết thúc")
    every = _parse_optional_int(checkpoint_every, "checkpoint_every")
    if every is None:
        every = cfg["checkpoint_every"]
    if every < 0:
        every = 0

    client = _build_client(llama_url, model, api_key, temperature, timeout, dry_run)

    total_in_file = len(input_records)
    lo, hi = _slice_bounds(total_in_file, start, end)
    selected = list(enumerate(input_records))[lo:hi]  # (absolute_index, record)

    async def stream():
        run_id = ckpt.new_run_id()
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        params = {
            "start_index": lo,
            "end_index": hi - 1 if hi > lo else None,
            "total_in_file": total_in_file,
            "dry_run": dry_run,
            "model": getattr(client, "model", None),
            "text_field": field,
            "checkpoint_every": every,
        }
        try:
            entries, by_word = parse_dictionary(dictionary_entries)
            ui_dictionary = build_ui_dictionary(entries)
            total = len(selected)
            ambiguous_count = sum(1 for e in entries if e.is_ambiguous)
            logger.info(
                "Stream start run=%s: %d/%d document(s) [%d:%d], %d dict words (%d ambiguous), "
                "dry_run=%s, checkpoint_every=%d",
                run_id, total, total_in_file, lo, hi, len(by_word), ambiguous_count, dry_run, every,
            )
            yield _ndjson(
                {
                    "event": "start",
                    "run_id": run_id,
                    "total": total,
                    "total_in_file": total_in_file,
                    "start_index": lo,
                    "end_index": hi - 1 if hi > lo else None,
                    "checkpoint_every": every,
                    "dictionary_words": len(by_word),
                    "ambiguous_words": ambiguous_count,
                    "dry_run": dry_run,
                }
            )

            documents: list[dict[str, Any]] = []
            summary = {"documents": 0, "terms": 0, "direct": 0, "llm": 0, "none": 0}
            last_ckpt_at = 0  # number of processed docs at last checkpoint

            def snapshot(next_index: int | None, done: bool) -> dict[str, Any]:
                return {
                    "run_id": run_id,
                    "created_at": created_at,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "done": done,
                    "processed": len(documents),
                    "total": total,
                    "next_index": next_index,
                    "params": params,
                    "documents": documents,
                    "dictionary": ui_dictionary,
                    "summary": summary,
                }

            for position, (abs_index, record) in enumerate(selected):
                if not isinstance(record, dict):
                    record = {field: record}
                raw_text = record.get(field, "")
                text = raw_text if isinstance(raw_text, str) else str(raw_text or "")
                name = str(record.get("name") or record.get("id") or f"doc_{abs_index + 1}")
                meta = {k: v for k, v in record.items() if k != field}
                meta["source_index"] = abs_index

                yield _ndjson(
                    {
                        "event": "document_start",
                        "index": position + 1,
                        "source_index": abs_index,
                        "total": total,
                        "name": name,
                        "text_len": len(text),
                    }
                )

                t0 = time.monotonic()
                try:
                    # Run the (blocking) resolution in a worker thread and emit a
                    # heartbeat every HEARTBEAT_SECONDS while we wait, so the HTTP
                    # response never goes idle long enough for a proxy to cut it.
                    task = asyncio.ensure_future(
                        run_in_threadpool(resolve_document, text, by_word, client)
                    )
                    while True:
                        done, _ = await asyncio.wait({task}, timeout=HEARTBEAT_SECONDS)
                        if task in done:
                            break
                        yield _ndjson(
                            {
                                "event": "heartbeat",
                                "index": position + 1,
                                "source_index": abs_index,
                                "name": name,
                                "waited_s": int(time.monotonic() - t0),
                            }
                        )
                    labels = task.result()
                except LLMError as exc:
                    logger.warning("Document %s -> LLM error: %s", name, exc)
                    yield _ndjson(
                        {
                            "event": "document_error",
                            "index": position + 1,
                            "source_index": abs_index,
                            "total": total,
                            "name": name,
                            "error_type": "LLMError",
                            "error": str(exc),
                        }
                    )
                    continue
                except Exception as exc:  # never let one bad doc kill the stream
                    logger.exception("Document %s failed", name)
                    yield _ndjson(
                        {
                            "event": "document_error",
                            "index": position + 1,
                            "source_index": abs_index,
                            "total": total,
                            "name": name,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    continue
                elapsed_ms = int((time.monotonic() - t0) * 1000)

                cnt = {"direct": 0, "llm": 0, "none": 0}
                for lb in labels:
                    src = lb["source"]
                    if src == "rule":
                        cnt["direct"] += 1
                    elif src == "llm":
                        cnt["llm"] += 1
                    else:
                        cnt["none"] += 1

                doc = {
                    "name": name,
                    "text": text,
                    "labels": labels,
                    "replacements": [],
                    "meta": meta,
                }
                documents.append(doc)
                summary["documents"] += 1
                summary["terms"] += len(labels)
                summary["direct"] += cnt["direct"]
                summary["llm"] += cnt["llm"]
                summary["none"] += cnt["none"]

                logger.info(
                    "Document %d/%d (src=%d) %r: %d labels (direct=%d llm=%d none=%d) in %dms",
                    position + 1, total, abs_index, name, len(labels),
                    cnt["direct"], cnt["llm"], cnt["none"], elapsed_ms,
                )
                yield _ndjson(
                    {
                        "event": "document_done",
                        "index": position + 1,
                        "source_index": abs_index,
                        "total": total,
                        "name": name,
                        "labels": len(labels),
                        "direct": cnt["direct"],
                        "llm": cnt["llm"],
                        "none": cnt["none"],
                        "elapsed_ms": elapsed_ms,
                    }
                )

                # Progressive saving: persist a full snapshot every N documents so a
                # later timeout/disconnect doesn't lose what we already computed.
                if every > 0 and (len(documents) - last_ckpt_at) >= every:
                    last_ckpt_at = len(documents)
                    next_index = abs_index + 1
                    path = await run_in_threadpool(ckpt.write_checkpoint, run_id, snapshot(next_index, False))
                    logger.info("Checkpoint run=%s: %d processed -> %s", run_id, len(documents), path)
                    yield _ndjson(
                        {
                            "event": "checkpoint",
                            "run_id": run_id,
                            "processed": len(documents),
                            "next_index": next_index,
                            "file": Path(path).name,
                        }
                    )

            # Always write a final checkpoint (unless checkpointing is disabled) so the
            # complete result is retrievable even if the client connection drops now.
            final_next = (selected[-1][0] + 1) if selected else lo
            if every > 0:
                path = await run_in_threadpool(ckpt.write_checkpoint, run_id, snapshot(final_next, True))
                logger.info("Final checkpoint run=%s -> %s", run_id, path)
                yield _ndjson(
                    {
                        "event": "checkpoint",
                        "run_id": run_id,
                        "processed": len(documents),
                        "next_index": final_next,
                        "file": Path(path).name,
                        "final": True,
                    }
                )

            logger.info("Stream done run=%s: %s", run_id, summary)
            yield _ndjson(
                {
                    "event": "done",
                    "run_id": run_id,
                    "next_index": final_next,
                    "documents": documents,
                    "dictionary": ui_dictionary,
                    "summary": summary,
                }
            )
        except Exception as exc:
            logger.exception("Stream failed")
            yield _ndjson(
                {"event": "fatal", "run_id": run_id, "error_type": type(exc).__name__, "error": str(exc)}
            )

    headers = {
        # Disable any reverse-proxy buffering (nginx etc.) so the client really
        # sees one event per document as it lands.
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers=headers,
    )
