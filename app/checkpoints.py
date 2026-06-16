"""Checkpoint storage for resumable resolve runs.

A long resolve run can be interrupted by a timeout, a dropped connection, or a
corporate proxy resetting the stream. To avoid losing work, ``/api/resolve``
writes the accumulated result to ``{AR_CHECKPOINT_DIR}/{run_id}.json`` every N
documents (and once more at the end). Each checkpoint is a full, UI-compatible
snapshot, so it can be downloaded or loaded straight into the labeler, and it
records ``next_index`` so the user can resume by setting the start index.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

# Conservative whitelist so a ``run_id`` from the URL can never escape the dir.
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def checkpoint_dir() -> Path:
    directory = Path(os.getenv("AR_CHECKPOINT_DIR", "checkpoints"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def new_run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def is_valid_run_id(run_id: str) -> bool:
    return bool(_SAFE_RUN_ID.match(run_id or ""))


def checkpoint_file(run_id: str) -> Path:
    if not is_valid_run_id(run_id):
        raise ValueError("run_id không hợp lệ")
    return checkpoint_dir() / f"{run_id}.json"


def write_checkpoint(run_id: str, payload: dict[str, Any]) -> str:
    """Atomically write a checkpoint snapshot; returns the file path as a string."""
    path = checkpoint_file(run_id)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem
    return str(path)


def read_checkpoint(run_id: str) -> dict[str, Any]:
    return json.loads(checkpoint_file(run_id).read_text(encoding="utf-8"))


def list_checkpoints() -> list[dict[str, Any]]:
    """Return lightweight metadata for every checkpoint, newest first."""
    items: list[dict[str, Any]] = []
    for path in checkpoint_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        items.append(
            {
                "run_id": data.get("run_id", path.stem),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "processed": data.get("processed", len(data.get("documents", []) or [])),
                "total": data.get("total"),
                "next_index": data.get("next_index"),
                "done": data.get("done", False),
                "params": data.get("params", {}),
                "summary": data.get("summary", {}),
                "file": path.name,
            }
        )
    items.sort(key=lambda x: (x.get("updated_at") or "", x.get("run_id") or ""), reverse=True)
    return items
