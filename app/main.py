"""FastAPI web app: a friendly UI + JSON API around the resolver.

Endpoints
---------
* ``GET  /``              -> Auto-resolve page (upload + run LLM)
* ``GET  /labeler``       -> the existing labeler UI (with Import buttons)
* ``GET  /api/config``    -> default LLM config (from env)
* ``GET  /api/llm-check`` -> probe ``/v1/models`` reachability
* ``POST /api/test-llm``  -> send the smallest possible chat call through
                              the same client we use for real resolution, so
                              we can see exactly *why* a real call hangs
                              (timing, status code, error type)
* ``POST /api/resolve``   -> run the pipeline; returns ``application/x-ndjson``
                              with one event per document plus a final ``done``
                              event carrying the full UI-compatible result.
                              Streaming keeps the connection alive across slow
                              LLM calls and corporate proxies/firewalls that
                              would otherwise reset an idle HTTP POST.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

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

app = FastAPI(title="Abbreviation Resolution", version="1.1.0")


def get_config() -> dict[str, Any]:
    return {
        "llama_server_url": os.getenv("LLAMA_SERVER_URL", "http://host.docker.internal:8080/v1"),
        "llama_model": os.getenv("LLAMA_MODEL", "local-model"),
        "llama_timeout": float(os.getenv("LLAMA_TIMEOUT", "120")),
        "has_api_key": bool(os.getenv("LLAMA_API_KEY")),
    }


def _build_client(
    llama_url: str | None,
    model: str | None,
    temperature: float,
    dry_run: bool,
) -> Any:
    cfg = get_config()
    if dry_run:
        return MockLLMClient(strategy="context")
    return LLMClient(
        base_url=llama_url or cfg["llama_server_url"],
        model=model or cfg["llama_model"],
        api_key=os.getenv("LLAMA_API_KEY"),
        timeout=cfg["llama_timeout"],
        temperature=temperature,
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/labeler", include_in_schema=False)
def labeler() -> FileResponse:
    return FileResponse(STATIC_DIR / "labeler.html")


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    return get_config()


@app.get("/api/llm-check")
def api_llm_check(llama_url: str | None = Query(default=None)) -> JSONResponse:
    cfg = get_config()
    url = llama_url or cfg["llama_server_url"]
    api_key = os.getenv("LLAMA_API_KEY")
    try:
        data = check_connection(url, api_key=api_key, timeout=10.0)
        models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        return JSONResponse({"ok": True, "url": url, "models": models})
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"},
            status_code=200,
        )


@app.post("/api/test-llm")
def api_test_llm(
    llama_url: str | None = Form(default=None),
    model: str | None = Form(default=None),
) -> JSONResponse:
    """Minimal end-to-end probe of the chat endpoint.

    Replicates the user's manual curl test from the server's network namespace
    using *the same* ``httpx.Client`` configuration ``LLMClient`` uses for real
    resolution. Surfaces concrete error type / timing — invaluable to debug the
    common "curl works but my app hangs" situation (corporate proxy, MTU
    issues, mismatched timeouts, ...).
    """
    cfg = get_config()
    client = LLMClient(
        base_url=llama_url or cfg["llama_server_url"],
        model=model or cfg["llama_model"],
        api_key=os.getenv("LLAMA_API_KEY"),
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


def _ndjson(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


@app.post("/api/resolve")
async def api_resolve(
    input_file: UploadFile = File(...),
    dictionary_file: UploadFile = File(...),
    llama_url: str | None = Form(default=None),
    model: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    dry_run: bool = Form(default=False),
    text_field: str = Form(default="input"),
) -> StreamingResponse:
    """Streaming NDJSON resolver.

    Why streaming? A single resolution call can take many seconds (or minutes)
    when the LLM is slow. With a buffered JSON response, a corporate proxy or
    firewall that resets idle HTTP POSTs will turn that wait into the dreaded
    ``NetworkError when attempting to fetch resource`` in the browser. Streaming
    one event per document keeps bytes flowing continuously and lets the UI
    show real progress.

    Event shape (one JSON object per line)::

        {"event": "start",          "total": N, "dictionary_words": M}
        {"event": "document_start", "index": i, "total": N, "name": "...", "text_len": T}
        {"event": "document_done",  "index": i, "name": "...", "labels": L,
                                    "direct": d, "llm": l, "none": n, "elapsed_ms": ms}
        {"event": "document_error", "index": i, "name": "...", "error": "..."}
        {"event": "done",           "documents": [...], "dictionary": {...}, "summary": {...}}
        {"event": "fatal",          "error": "..."}      # only if the whole stream fails
    """
    input_records = await _read_json_list(input_file, "File input")
    dictionary_entries = await _read_json_list(dictionary_file, "File dictionary")
    client = _build_client(llama_url, model, temperature, dry_run)

    async def stream():
        try:
            entries, by_word = parse_dictionary(dictionary_entries)
            ui_dictionary = build_ui_dictionary(entries)
            total = len(input_records)
            ambiguous_count = sum(1 for e in entries if e.is_ambiguous)
            logger.info(
                "Stream start: %d document(s), %d dictionary words (%d ambiguous), dry_run=%s",
                total, len(by_word), ambiguous_count, dry_run,
            )
            yield _ndjson(
                {
                    "event": "start",
                    "total": total,
                    "dictionary_words": len(by_word),
                    "ambiguous_words": ambiguous_count,
                    "dry_run": dry_run,
                }
            )

            documents: list[dict[str, Any]] = []
            summary = {"documents": 0, "terms": 0, "direct": 0, "llm": 0, "none": 0}

            for index, record in enumerate(input_records):
                if not isinstance(record, dict):
                    record = {text_field: record}
                raw_text = record.get(text_field, "")
                text = raw_text if isinstance(raw_text, str) else str(raw_text or "")
                name = str(record.get("name") or record.get("id") or f"doc_{index + 1}")
                meta = {k: v for k, v in record.items() if k != text_field}

                yield _ndjson(
                    {
                        "event": "document_start",
                        "index": index + 1,
                        "total": total,
                        "name": name,
                        "text_len": len(text),
                    }
                )

                t0 = time.monotonic()
                try:
                    labels = await run_in_threadpool(
                        resolve_document, text, by_word, client
                    )
                except LLMError as exc:
                    logger.warning("Document %s -> LLM error: %s", name, exc)
                    yield _ndjson(
                        {
                            "event": "document_error",
                            "index": index + 1,
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
                            "index": index + 1,
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
                    "Document %d/%d %r: %d labels (direct=%d llm=%d none=%d) in %dms",
                    index + 1, total, name, len(labels),
                    cnt["direct"], cnt["llm"], cnt["none"], elapsed_ms,
                )
                yield _ndjson(
                    {
                        "event": "document_done",
                        "index": index + 1,
                        "total": total,
                        "name": name,
                        "labels": len(labels),
                        "direct": cnt["direct"],
                        "llm": cnt["llm"],
                        "none": cnt["none"],
                        "elapsed_ms": elapsed_ms,
                    }
                )

            logger.info("Stream done: %s", summary)
            yield _ndjson(
                {
                    "event": "done",
                    "documents": documents,
                    "dictionary": ui_dictionary,
                    "summary": summary,
                }
            )
        except Exception as exc:
            logger.exception("Stream failed")
            yield _ndjson(
                {"event": "fatal", "error_type": type(exc).__name__, "error": str(exc)}
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
