"""Tests for the FastAPI surface: streaming /api/resolve, /api/test-llm, etc."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.llm_client import LLMClient
from app.main import app

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _files():
    return {
        "input_file": ("input.json", open(SAMPLE_DIR / "sample_input.json", "rb"), "application/json"),
        "dictionary_file": ("dict.json", open(SAMPLE_DIR / "sample_dictionary.json", "rb"), "application/json"),
    }


def _parse_ndjson(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_static_pages_serve(client):
    assert client.get("/").status_code == 200
    assert client.get("/labeler").status_code == 200


def test_api_config_returns_defaults(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert "llama_server_url" in data
    assert "llama_model" in data


def test_api_resolve_streams_ndjson_dry_run(client):
    files = _files()
    try:
        r = client.post("/api/resolve", files=files, data={"dry_run": "true"})
    finally:
        for _, f, *_ in files.values():
            f.close()

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")

    events = _parse_ndjson(r.text)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start"
    assert "done" in kinds
    assert kinds[-1] == "done"
    assert any(k == "document_start" for k in kinds)
    assert any(k == "document_done" for k in kinds)

    done = events[-1]
    assert isinstance(done["documents"], list)
    assert isinstance(done["dictionary"], dict)
    assert done["summary"]["documents"] == len(done["documents"])
    assert done["summary"]["documents"] == 3  # matches sample_input.json


def test_api_resolve_per_document_error_does_not_kill_stream(client, monkeypatch):
    """A failing LLM call for one document yields document_error, not fatal."""

    def boom(self, text, items):
        raise RuntimeError("boom")

    monkeypatch.setattr(LLMClient, "resolve", boom)

    files = _files()
    try:
        r = client.post(
            "/api/resolve",
            files=files,
            data={"dry_run": "false", "llama_url": "http://127.0.0.1:1/v1"},
        )
    finally:
        for _, f, *_ in files.values():
            f.close()

    assert r.status_code == 200
    events = _parse_ndjson(r.text)
    kinds = [e["event"] for e in events]
    assert "document_error" in kinds
    assert kinds[-1] == "done"  # still terminate cleanly


def test_api_resolve_rejects_invalid_json(client):
    bad = ("input.json", io.BytesIO(b"not json"), "application/json")
    good = ("dict.json", open(SAMPLE_DIR / "sample_dictionary.json", "rb"), "application/json")
    try:
        r = client.post(
            "/api/resolve",
            files={"input_file": bad, "dictionary_file": good},
            data={"dry_run": "true"},
        )
    finally:
        good[1].close()
    assert r.status_code == 400
    assert "File input" in r.json()["detail"]


def test_api_test_llm_success(client, monkeypatch):
    def fake_post(self, payload):
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    r = client.post("/api/test-llm", data={"llama_url": "http://x/v1", "model": "m1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == "m1"
    assert body["url"].endswith("/chat/completions")
    assert '"ok": true' in body["content_preview"]


def test_api_test_llm_reports_failure_with_type(client, monkeypatch):
    def fake_post(self, payload):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    r = client.post("/api/test-llm", data={"llama_url": "http://x/v1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error_type"] == "ConnectError"
    assert "connection refused" in body["error"]


def test_api_llm_check_returns_error_when_unreachable(client, monkeypatch):
    def fake(url, api_key=None, timeout=10.0, trust_env=False):
        raise httpx.ConnectTimeout("boom")

    monkeypatch.setattr(main_module, "check_connection", fake)
    r = client.get("/api/llm-check?llama_url=http://x/v1")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "ConnectTimeout" in body["error"]


def test_llm_client_resolve_fast_fails_on_network_error(monkeypatch):
    """A network-layer error must not be retried across response_format modes."""
    calls = {"n": 0}

    def fake_post(self, payload):
        calls["n"] += 1
        raise httpx.ConnectTimeout("nope")

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    cli = LLMClient(base_url="http://x/v1")
    with pytest.raises(Exception):  # LLMError, but we don't import it here
        cli.resolve("text", [{"word": "X", "senses": []}])
    assert calls["n"] == 1  # short-circuited after first network failure
