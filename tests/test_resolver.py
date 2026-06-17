import pytest

from app.llm_client import MockLLMClient
from app.resolver import (
    NONE_LABEL,
    build_ui_dictionary,
    find_occurrences,
    parse_dictionary,
    parse_label,
    resolve_document,
    resolve_documents,
    slug,
)

SAMPLE_DICT = [
    {"word": "TC", "type": "tên tàu/khac", "meaning": "Tàu cá/TC", "label": 1},
    {"word": "TP", "type": "địa danh/chức vụ", "meaning": "Thành phố/Trưởng phòng", "label": 1},
    {"word": "HCM", "type": "địa danh", "meaning": "Hồ Chí Minh", "label": 0},
    {"word": "BHYT", "type": "chính sách", "meaning": "Bảo hiểm y tế", "label": 0},
]


def test_slug_matches_ui_behavior():
    assert slug("Tàu cá") == "tau_ca"
    assert slug("HĐND") == "h_nd"
    assert slug("") == "item"


def test_parse_label():
    assert parse_label(1) == 1
    assert parse_label("1") == 1
    assert parse_label(2) == 1
    assert parse_label(0) == 0
    assert parse_label("0") == 0
    assert parse_label(None) == 0
    assert parse_label("abc") == 0


def test_parse_dictionary_splits_parallel_lists():
    _, by_word = parse_dictionary(SAMPLE_DICT)
    tc = by_word["TC"]
    assert tc.label == 1
    assert [(s.explanation, s.label) for s in tc.senses] == [
        ("tên tàu", "Tàu cá"),
        ("khac", "TC"),
    ]
    assert tc.senses[0].id == "tc_1"
    assert tc.senses[1].id == "tc_2"


def test_parse_dictionary_merges_duplicates():
    raw = [
        {"word": "X", "type": "a", "meaning": "Alpha", "label": 0},
        {"word": "X", "type": "b", "meaning": "Beta", "label": 1},
    ]
    _, by_word = parse_dictionary(raw)
    assert by_word["X"].label == 1
    assert len(by_word["X"].senses) == 2


def test_build_ui_dictionary_none_sense_only_for_ambiguous():
    entries, _ = parse_dictionary(SAMPLE_DICT)
    ui = build_ui_dictionary(entries)
    assert len(ui["TC"]) == 3
    assert ui["TC"][-1]["label"] == NONE_LABEL
    assert ui["TC"][-1]["id"] == "tc_none"
    assert len(ui["HCM"]) == 1  # unambiguous keeps a single sense for UI auto-annotate


def test_find_occurrences_respects_boundaries():
    words = ["TC", "TP", "HCM"]
    text = "TP-HCM có TC. ATC TCs TC1 (TC)"
    found = [token for _, _, token in find_occurrences(text, words)]
    assert "TP" in found
    assert "HCM" in found
    # ATC / TCs / TC1 must NOT match; "có TC." and "(TC)" must match.
    assert found.count("TC") == 2


def test_resolve_document_label0_assigned_directly():
    _, by_word = parse_dictionary(SAMPLE_DICT)
    labels = resolve_document("Thẻ BHYT và HCM", by_word, MockLLMClient(strategy="none"))
    by_term = {lb["term"]: lb for lb in labels}
    assert by_term["BHYT"]["senseLabel"] == "Bảo hiểm y tế"
    assert by_term["BHYT"]["source"] == "rule"
    assert by_term["BHYT"]["auto"] is True
    assert by_term["HCM"]["senseLabel"] == "Hồ Chí Minh"


def test_resolve_document_ambiguous_uses_llm_choice():
    _, by_word = parse_dictionary(SAMPLE_DICT)
    labels = resolve_document("Một chiếc TC ra khơi", by_word, MockLLMClient(choices={"TC": 0}))
    tc = next(lb for lb in labels if lb["term"] == "TC")
    assert tc["senseLabel"] == "Tàu cá"
    assert tc["source"] == "llm"
    assert tc["auto"] is False


def test_resolve_document_none_choice_maps_to_none_sense():
    _, by_word = parse_dictionary(SAMPLE_DICT)
    labels = resolve_document("Một chiếc TC ra khơi", by_word, MockLLMClient(choices={"TC": -1}))
    tc = next(lb for lb in labels if lb["term"] == "TC")
    assert tc["senseLabel"] == NONE_LABEL
    assert tc["senseId"] == "tc_none"
    assert tc["source"] == "llm-none"


def test_resolve_document_out_of_range_choice_is_none():
    _, by_word = parse_dictionary(SAMPLE_DICT)
    labels = resolve_document("Một chiếc TC ra khơi", by_word, MockLLMClient(choices={"TC": 99}))
    tc = next(lb for lb in labels if lb["term"] == "TC")
    assert tc["source"] == "llm-none"


def test_ambiguous_without_client_raises():
    _, by_word = parse_dictionary(SAMPLE_DICT)
    with pytest.raises(ValueError):
        resolve_document("chiếc TC", by_word, None)


def test_resolve_documents_output_schema_and_summary():
    records = [{"id": "d1", "input": "Tại TP-HCM có HCM và TC"}]
    client = MockLLMClient(choices={"TC": 0, "TP": 0})
    out = resolve_documents(records, SAMPLE_DICT, client)

    assert set(out.keys()) == {"documents", "dictionary", "summary"}
    doc = out["documents"][0]
    assert {"name", "text", "labels", "replacements", "meta"}.issubset(doc.keys())
    assert doc["name"] == "d1"
    assert doc["meta"] == {"id": "d1"}

    for lb in doc["labels"]:
        assert {
            "start", "end", "term", "senseId", "senseLabel",
            "senseExplanation", "text", "auto", "source",
        }.issubset(lb.keys())
        # positions must point exactly to the matched token
        assert doc["text"][lb["start"]:lb["end"]] == lb["term"]
        assert lb["text"] == lb["term"]

    assert out["summary"]["documents"] == 1
    assert out["summary"]["terms"] == len(doc["labels"])


class _RecordingClient:
    """Returns a choice per occurrence id and records what it was asked."""

    def __init__(self, by_id):
        self.by_id = by_id
        self.seen_text = None
        self.seen_items = None

    def resolve(self, text, items):
        self.seen_text = text
        self.seen_items = items
        return {it["id"]: self.by_id.get(it["id"], -1) for it in items}


def test_resolve_document_per_occurrence_different_meanings():
    """The same abbreviation in two places can get two different senses."""
    _, by_word = parse_dictionary(SAMPLE_DICT)
    text = "Tàu TC ra khơi, đội TC kiểm tra."  # two standalone TC occurrences
    client = _RecordingClient({1: 0, 2: 1})  # occ#1 -> Tàu cá, occ#2 -> TC
    labels = resolve_document(text, by_word, client)

    tcs = [lb for lb in labels if lb["term"] == "TC"]
    assert len(tcs) == 2
    assert tcs[0]["senseLabel"] == "Tàu cá"
    assert tcs[1]["senseLabel"] == "TC"

    # The model saw the occurrence-id markers in the text it was given...
    assert "«1»" in client.seen_text and "«2»" in client.seen_text
    # ...and one item per occurrence (both the same word here).
    assert [it["id"] for it in client.seen_items] == [1, 2]
    assert all(it["word"] == "TC" for it in client.seen_items)


def test_resolve_documents_handles_unknown_uppercase_tokens():
    records = [{"id": "d3", "input": "Các đơn vị ABC và XYZ không có trong từ điển"}]
    out = resolve_documents(records, SAMPLE_DICT, MockLLMClient())
    # ABC / XYZ are not in the dictionary -> no labels produced
    assert out["documents"][0]["labels"] == []
