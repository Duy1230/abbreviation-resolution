"""FastAPI web app: a friendly UI + JSON API around the resolver.

Endpoints
---------
* ``GET  /``              -> Auto-resolve page (upload + run LLM)
* ``GET  /labeler``       -> the existing labeler UI (with Import buttons)
* ``GET  /api/config``    -> default LLM config (from env)
* ``GET  /api/llm-check`` -> probe reachability of the llama-server
* ``POST /api/resolve``   -> run the pipeline, return UI-compatible JSON
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .llm_client import LLMClient, LLMError, MockLLMClient, check_connection
from .resolver import resolve_documents

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Abbreviation Resolution", version="1.0.0")


def get_config() -> dict[str, Any]:
    return {
        "llama_server_url": os.getenv("LLAMA_SERVER_URL", "http://host.docker.internal:8080/v1"),
        "llama_model": os.getenv("LLAMA_MODEL", "local-model"),
        "llama_timeout": float(os.getenv("LLAMA_TIMEOUT", "120")),
        "has_api_key": bool(os.getenv("LLAMA_API_KEY")),
    }


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
        return JSONResponse({"ok": False, "url": url, "error": str(exc)}, status_code=200)


async def _read_json_list(upload: UploadFile, field_name: str) -> list[Any]:
    try:
        raw = await upload.read()
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} không phải JSON hợp lệ: {exc}")
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail=f"{field_name} phải là một JSON array (list).")
    return data


@app.post("/api/resolve")
async def api_resolve(
    input_file: UploadFile = File(...),
    dictionary_file: UploadFile = File(...),
    llama_url: str | None = Form(default=None),
    model: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    dry_run: bool = Form(default=False),
    text_field: str = Form(default="input"),
) -> JSONResponse:
    input_records = await _read_json_list(input_file, "File input")
    dictionary_entries = await _read_json_list(dictionary_file, "File dictionary")

    cfg = get_config()
    if dry_run:
        client: Any = MockLLMClient(strategy="context")
    else:
        client = LLMClient(
            base_url=llama_url or cfg["llama_server_url"],
            model=model or cfg["llama_model"],
            api_key=os.getenv("LLAMA_API_KEY"),
            timeout=cfg["llama_timeout"],
            temperature=temperature,
        )

    try:
        result = resolve_documents(input_records, dictionary_entries, client, text_field=text_field)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"Lỗi gọi LLM: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return JSONResponse(result)
