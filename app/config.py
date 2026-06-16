"""File-backed application configuration that the UI can read and edit.

Precedence (lowest -> highest):

1. :data:`DEFAULTS` (hard-coded, always present)
2. ``config.json`` (path from ``AR_CONFIG_PATH``, default ``config.json``)
3. environment variables (see :data:`ENV_MAP`)

The Auto-resolve page loads these values via ``GET /api/config`` and may persist
edits back to ``config.json`` via ``POST /api/config`` ("save as default").
Per-request form fields still override everything for that single run, so the
config file only provides the *defaults* shown in the UI.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "llama_server_url": "http://host.docker.internal:8080/v1",
    "llama_model": "local-model",
    "llama_api_key": "sk-1234",
    "llama_timeout": 600.0,
    "temperature": 0.0,
    "checkpoint_every": 20,
    "text_field": "input",
}

# Fields the UI is allowed to edit and persist.
EDITABLE_FIELDS: tuple[str, ...] = tuple(DEFAULTS.keys())

# Optional environment overrides (highest precedence) — handy in Docker/CI.
ENV_MAP: dict[str, str] = {
    "llama_server_url": "LLAMA_SERVER_URL",
    "llama_model": "LLAMA_MODEL",
    "llama_api_key": "LLAMA_API_KEY",
    "llama_timeout": "LLAMA_TIMEOUT",
    "temperature": "LLAMA_TEMPERATURE",
    "checkpoint_every": "AR_CHECKPOINT_EVERY",
    "text_field": "AR_TEXT_FIELD",
}

_LOCK = threading.Lock()


def config_path() -> Path:
    return Path(os.getenv("AR_CONFIG_PATH", "config.json"))


def _coerce(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in cfg.items() if k in DEFAULTS})

    try:
        out["llama_timeout"] = float(out["llama_timeout"])
    except (TypeError, ValueError):
        out["llama_timeout"] = DEFAULTS["llama_timeout"]

    try:
        out["temperature"] = float(out["temperature"])
    except (TypeError, ValueError):
        out["temperature"] = DEFAULTS["temperature"]

    try:
        out["checkpoint_every"] = int(out["checkpoint_every"])
    except (TypeError, ValueError):
        out["checkpoint_every"] = DEFAULTS["checkpoint_every"]
    if out["checkpoint_every"] < 0:
        out["checkpoint_every"] = 0

    out["llama_server_url"] = str(out["llama_server_url"] or DEFAULTS["llama_server_url"])
    out["llama_model"] = str(out["llama_model"] or DEFAULTS["llama_model"])
    # API key may legitimately be an empty string (server with no auth).
    out["llama_api_key"] = "" if out["llama_api_key"] is None else str(out["llama_api_key"])
    out["text_field"] = str(out["text_field"] or DEFAULTS["text_field"])
    return out


def _read_file() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_config() -> dict[str, Any]:
    """Effective config: defaults <- file <- environment."""
    cfg = dict(DEFAULTS)
    for key, value in _read_file().items():
        if key in DEFAULTS and value is not None:
            cfg[key] = value
    for key, env in ENV_MAP.items():
        env_value = os.getenv(env)
        if env_value is not None and env_value != "":
            cfg[key] = env_value
    return _coerce(cfg)


def save_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Persist editable fields to ``config.json`` and return effective config.

    Only keys in :data:`EDITABLE_FIELDS` are written. Environment variables are
    *not* baked into the file, so they keep acting as runtime overrides.
    """
    with _LOCK:
        current = dict(DEFAULTS)
        for key, value in _read_file().items():
            if key in DEFAULTS and value is not None:
                current[key] = value
        for key in EDITABLE_FIELDS:
            if key in updates and updates[key] is not None:
                current[key] = updates[key]
        current = _coerce(current)
        path = config_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(current, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            # Read-only filesystem (e.g. baked image without a writable mount):
            # keep serving the in-memory result instead of crashing.
            pass
    return load_config()
