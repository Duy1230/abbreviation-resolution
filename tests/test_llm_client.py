import json

import pytest

from app.llm_client import (
    LLMClient,
    MockLLMClient,
    build_listing,
    build_messages,
    build_schema,
    parse_resolutions,
)
from app.resolver import Sense


def make_items():
    return [
        {"word": "TC", "senses": [Sense("tc_1", "Tàu cá", "tên tàu"), Sense("tc_2", "TC", "khac")]},
        {
            "word": "TP",
            "senses": [Sense("tp_1", "Thành phố", "địa danh"), Sense("tp_2", "Trưởng phòng", "chức vụ")],
        },
    ]


def test_build_schema_structure():
    schema = build_schema()
    assert schema["type"] == "object"
    assert "resolutions" in schema["properties"]
    item = schema["properties"]["resolutions"]["items"]
    assert item["required"] == ["word", "choice"]


def test_build_listing_contains_indices_and_words():
    listing = build_listing(make_items())
    assert "[0] (tên tàu) Tàu cá" in listing
    assert "[1] (chức vụ) Trưởng phòng" in listing
    assert 'word="TP"' in listing


def test_build_messages_has_system_and_user():
    messages = build_messages("văn bản mẫu", make_items())
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "văn bản mẫu" in messages[1]["content"]


def test_parse_resolutions_plain_json():
    content = json.dumps({"resolutions": [{"word": "TC", "choice": 0}, {"word": "TP", "choice": 1}]})
    assert parse_resolutions(content, make_items()) == {"TC": 0, "TP": 1}


def test_parse_resolutions_handles_code_fence_and_prose():
    content = 'Kết quả:\n```json\n{"resolutions":[{"word":"TC","choice":-1}]}\n```'
    out = parse_resolutions(content, make_items())
    assert out["TC"] == -1
    assert out["TP"] == -1  # missing word defaults to -1


def test_parse_resolutions_invalid_raises():
    with pytest.raises(ValueError):
        parse_resolutions("không có json ở đây", make_items())


def test_mock_client_strategies():
    items = make_items()
    assert MockLLMClient(strategy="first").resolve("x", items) == {"TC": 0, "TP": 0}
    assert MockLLMClient(strategy="none").resolve("x", items) == {"TC": -1, "TP": -1}
    ctx = MockLLMClient(strategy="context").resolve("Ông ấy là Trưởng phòng", items)
    assert ctx["TP"] == 1
    assert ctx["TC"] == 0


def test_llm_client_resolve_parses_post(monkeypatch):
    client = LLMClient(base_url="http://example/v1")

    def fake_post(payload):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"resolutions": [{"word": "TC", "choice": 1}, {"word": "TP", "choice": 0}]}
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)
    assert client.resolve("text", make_items()) == {"TC": 1, "TP": 0}


def test_llm_client_resolve_empty_items_short_circuits():
    client = LLMClient(base_url="http://example/v1")
    assert client.resolve("text", []) == {}
