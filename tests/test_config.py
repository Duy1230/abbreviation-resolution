"""Unit tests for config persistence, checkpoint storage and range slicing."""

from __future__ import annotations

import json

import pytest

from app import checkpoints as ckpt
from app.config import DEFAULTS, load_config, save_config
from app.main import _slice_bounds


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("AR_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setenv("AR_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    for var in (
        "LLAMA_SERVER_URL", "LLAMA_MODEL", "LLAMA_API_KEY", "LLAMA_TIMEOUT",
        "LLAMA_TEMPERATURE", "AR_CHECKPOINT_EVERY", "AR_TEXT_FIELD",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def test_load_defaults_when_no_file():
    cfg = load_config()
    assert cfg["llama_api_key"] == DEFAULTS["llama_api_key"]
    assert cfg["checkpoint_every"] == DEFAULTS["checkpoint_every"]
    assert isinstance(cfg["llama_timeout"], float)
    assert isinstance(cfg["checkpoint_every"], int)


def test_save_then_load_roundtrip(tmp_path):
    save_config({"llama_model": "m2", "checkpoint_every": 7, "llama_api_key": "k"})
    cfg = load_config()
    assert cfg["llama_model"] == "m2"
    assert cfg["checkpoint_every"] == 7
    assert cfg["llama_api_key"] == "k"
    # File really written and is valid JSON.
    on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert on_disk["llama_model"] == "m2"


def test_env_overrides_file(monkeypatch):
    save_config({"llama_model": "from-file"})
    monkeypatch.setenv("LLAMA_MODEL", "from-env")
    assert load_config()["llama_model"] == "from-env"


def test_coerce_handles_bad_types(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"checkpoint_every": "oops", "llama_timeout": "nope"}), encoding="utf-8"
    )
    cfg = load_config()
    assert cfg["checkpoint_every"] == DEFAULTS["checkpoint_every"]
    assert cfg["llama_timeout"] == DEFAULTS["llama_timeout"]


@pytest.mark.parametrize(
    "total,start,end,expected",
    [
        (10, None, None, (0, 10)),   # both empty -> all
        (10, 2, None, (2, 10)),      # open end -> to last
        (10, None, 4, (0, 5)),       # inclusive end
        (10, 3, 3, (3, 4)),          # single item
        (10, 5, 2, (5, 5)),          # inverted -> empty
        (10, 100, None, (10, 10)),   # start past end -> empty
        (10, -5, 3, (0, 4)),         # negative start clamps to 0
        (10, 0, 999, (0, 10)),       # end past length clamps
    ],
)
def test_slice_bounds(total, start, end, expected):
    assert _slice_bounds(total, start, end) == expected


def test_checkpoint_write_read_list():
    run_id = ckpt.new_run_id()
    payload = {"run_id": run_id, "documents": [{"x": 1}], "processed": 1, "updated_at": "2026-01-01T00:00:00"}
    ckpt.write_checkpoint(run_id, payload)

    assert ckpt.read_checkpoint(run_id)["run_id"] == run_id
    listed = ckpt.list_checkpoints()
    assert any(c["run_id"] == run_id for c in listed)
    assert listed[0]["processed"] == 1


def test_checkpoint_rejects_unsafe_run_id():
    assert not ckpt.is_valid_run_id("../escape")
    with pytest.raises(ValueError):
        ckpt.checkpoint_file("../escape")
