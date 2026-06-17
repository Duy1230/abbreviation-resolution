"""Core abbreviation-resolution logic.

This module is intentionally free of any I/O or HTTP dependency so it can be
unit-tested in isolation. The only collaborator it needs is an ``llm_client``
object exposing a ``resolve(text, items) -> {word: choice_int}`` method (see
``app.llm_client``). A ``MockLLMClient`` is provided there for tests / dry-run.

Output is shaped to be 100% compatible with ``dictionary_labeler_v4.html``:

* documents: ``[{name, text, labels:[...], replacements:[]}]``
  - label:  ``{start, end, term, senseId, senseLabel, senseExplanation, text, auto}``
    (plus an extra ``source`` field that the UI safely ignores / round-trips)
* dictionary: ``{ "WORD": [{id, label, explanation}, ...] }``
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Label shown when the LLM decides none of the candidate meanings fit.
NONE_LABEL = "(không có nghĩa phù hợp)"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Sense:
    """A single meaning of an abbreviation.

    ``label`` maps to the dictionary ``meaning`` (what the UI shows), while
    ``explanation`` maps to the dictionary ``type``.
    """

    id: str
    label: str  # meaning, e.g. "Tàu cá"
    explanation: str  # type, e.g. "tên tàu"


@dataclass
class Entry:
    word: str
    label: int  # 0 = unambiguous (assign directly), 1 = ambiguous (ask LLM)
    senses: list[Sense] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        return self.label == 1


class LLMClientLike(Protocol):
    def resolve(
        self, text: str, items: list[dict[str, Any]]
    ) -> dict[int, int]:  # pragma: no cover - protocol
        ...


# Markers wrapped around the occurrence id in the text we send to the LLM, e.g.
# "Tàu «1»TS-234 ... «2»TS". Chosen to be visually distinctive and very unlikely
# to appear in the source text. They are only used in the prompt; label offsets
# are always computed against the ORIGINAL text.
MARK_OPEN = "«"
MARK_CLOSE = "»"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def slug(value: str) -> str:
    """Port of the JS ``slug()`` in the labeler UI so ids stay consistent."""
    s = (value or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "item"


def parse_label(value: Any) -> int:
    """Coerce the dictionary ``label`` field into 0/1 (default 0)."""
    if value is None:
        return 0
    try:
        return 1 if int(str(value).strip()) != 0 else 0
    except (TypeError, ValueError):
        s = str(value).strip().lower()
        if s in {"1", "true", "yes", "y", "ambiguous", "nhập nhằng", "nhap nham"}:
            return 1
        return 0


def _split_field(value: Any) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split("/")]


def none_sense(word: str) -> Sense:
    return Sense(id=f"{slug(word)}_none", label=NONE_LABEL, explanation="")


# --------------------------------------------------------------------------- #
# Dictionary parsing
# --------------------------------------------------------------------------- #
def parse_dictionary(raw_entries: list[dict[str, Any]]) -> tuple[list[Entry], dict[str, Entry]]:
    """Parse the raw abbreviation dictionary (list of word/type/meaning/label).

    ``type`` and ``meaning`` are parallel ``/``-separated lists. Duplicate words
    are merged (senses concatenated & de-duplicated, label = max).
    """
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        word = str(raw.get("word") or raw.get("term") or "").strip()
        if not word:
            continue
        types = _split_field(raw.get("type"))
        meanings = _split_field(raw.get("meaning") if raw.get("meaning") is not None else raw.get("mean"))
        label = parse_label(raw.get("label"))

        pairs: list[tuple[str, str]] = []
        for i in range(max(len(types), len(meanings))):
            t = types[i] if i < len(types) else ""
            m = meanings[i] if i < len(meanings) else ""
            if not (t or m):
                continue
            pairs.append((t, m))

        if word in merged:
            merged[word]["pairs"].extend(pairs)
            merged[word]["label"] = max(merged[word]["label"], label)
        else:
            merged[word] = {"label": label, "pairs": pairs}
            order.append(word)

    entries: list[Entry] = []
    by_word: dict[str, Entry] = {}
    for word in order:
        data = merged[word]
        seen: set[tuple[str, str]] = set()
        senses: list[Sense] = []
        for t, m in data["pairs"]:
            key = (t, m)
            if key in seen:
                continue
            seen.add(key)
            senses.append(Sense(id=f"{slug(word)}_{len(senses) + 1}", label=m, explanation=t))
        entry = Entry(word=word, label=data["label"], senses=senses)
        entries.append(entry)
        by_word[word] = entry

    return entries, by_word


def build_ui_dictionary(entries: list[Entry]) -> dict[str, list[dict[str, str]]]:
    """Build the ``term -> [{id,label,explanation}]`` map used by the UI.

    The synthetic "none" sense is only added to ambiguous (label 1) words so that
    unambiguous words keep exactly one sense and the UI still auto-annotates them.
    """
    out: dict[str, list[dict[str, str]]] = {}
    for entry in entries:
        senses = [
            {"id": s.id, "label": s.label, "explanation": s.explanation} for s in entry.senses
        ]
        if entry.is_ambiguous:
            senses.append({"id": f"{slug(entry.word)}_none", "label": NONE_LABEL, "explanation": ""})
        out[entry.word] = senses
    return out


# --------------------------------------------------------------------------- #
# Text matching
# --------------------------------------------------------------------------- #
def build_word_regex(words: list[str]) -> re.Pattern[str] | None:
    """Build a case-sensitive regex matching whole abbreviation tokens.

    Mirrors the labeler boundary ``(^|[^\\p{L}\\p{N}_})(term)(?![\\p{L}\\p{N}_])``:
    a token must not be glued to a word char (letter/digit/underscore) on either
    side. Special characters such as ``.,:;/-+=()`` are word boundaries, so e.g.
    ``TP-HCM`` yields ``TP`` and ``HCM``. Longest words are tried first.
    """
    uniq = sorted({w for w in words if w}, key=len, reverse=True)
    if not uniq:
        return None
    alternation = "|".join(re.escape(w) for w in uniq)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")


def find_occurrences(text: str, words: list[str]) -> list[tuple[int, int, str]]:
    regex = build_word_regex(words)
    if regex is None or not text:
        return []
    return [(m.start(), m.end(), m.group(0)) for m in regex.finditer(text)]


def build_marked_text(text: str, occurrences: list[tuple[int, int, int, str]]) -> str:
    """Insert ``«id»`` right before each occurrence so the LLM sees its position.

    ``occurrences`` is a list of ``(occurrence_id, start, end, word)`` sorted by
    ``start`` ascending. Only used to build the prompt — never for offsets.
    """
    out: list[str] = []
    cursor = 0
    for oid, start, _end, _word in occurrences:
        if start < cursor:  # safety against any overlap
            continue
        out.append(text[cursor:start])
        out.append(f"{MARK_OPEN}{oid}{MARK_CLOSE}")
        cursor = start
    out.append(text[cursor:])
    return "".join(out)


def make_label(
    text: str,
    start: int,
    end: int,
    term: str,
    sense: Sense,
    source: str,
    auto: bool,
) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "term": term,
        "senseId": sense.id,
        "senseLabel": sense.label,
        "senseExplanation": sense.explanation,
        "text": text[start:end],
        "auto": auto,
        "source": source,
    }


# --------------------------------------------------------------------------- #
# Document resolution
# --------------------------------------------------------------------------- #
def resolve_document(
    text: str,
    by_word: dict[str, Entry],
    llm_client: LLMClientLike | None,
) -> list[dict[str, Any]]:
    """Resolve all abbreviation occurrences in a single document.

    The LLM (if any ambiguous occurrence is present) is called exactly once for the
    whole document, but it resolves **each occurrence independently**: every
    ambiguous occurrence gets its own id and the text is marked with ``«id»`` so the
    same abbreviation appearing twice can be assigned two different meanings.
    """
    text = text or ""
    occurrences = find_occurrences(text, list(by_word.keys()))
    if not occurrences:
        return []

    # Assign an id to every ambiguous occurrence, in document order.
    ambiguous_occ: list[tuple[int, int, int, str]] = []  # (id, start, end, word)
    next_id = 0
    for start, end, word in occurrences:
        if by_word[word].is_ambiguous:
            next_id += 1
            ambiguous_occ.append((next_id, start, end, word))

    resolutions: dict[int, int] = {}
    if ambiguous_occ:
        if llm_client is None:
            raise ValueError("Có từ nhập nhằng nhưng không cung cấp llm_client.")
        marked_text = build_marked_text(text, ambiguous_occ)
        items = [
            {"id": oid, "word": word, "senses": by_word[word].senses}
            for oid, _s, _e, word in ambiguous_occ
        ]
        distinct_words = len({word for _i, _s, _e, word in ambiguous_occ})
        t0 = time.monotonic()
        resolutions = llm_client.resolve(marked_text, items) or {}
        logger.info(
            "resolve_document: %d ambiguous occurrence(s) of %d distinct word(s) resolved in %.2fs",
            len(items), distinct_words, time.monotonic() - t0,
        )

    labels: list[dict[str, Any]] = []
    occ_id = 0
    for start, end, word in occurrences:
        entry = by_word[word]
        if not entry.is_ambiguous:
            if not entry.senses:
                continue
            labels.append(make_label(text, start, end, word, entry.senses[0], "rule", True))
            continue

        occ_id += 1
        try:
            choice = int(resolutions.get(occ_id, -1))
        except (TypeError, ValueError):
            choice = -1

        if 0 <= choice < len(entry.senses):
            labels.append(make_label(text, start, end, word, entry.senses[choice], "llm", False))
        else:
            labels.append(make_label(text, start, end, word, none_sense(word), "llm-none", False))

    labels.sort(key=lambda lb: (lb["start"], lb["end"]))
    return labels


def resolve_documents(
    input_records: list[Any],
    dictionary_entries: list[dict[str, Any]],
    llm_client: LLMClientLike | None,
    text_field: str = "input",
) -> dict[str, Any]:
    """Resolve a batch of documents.

    Returns a dict with ``documents`` (UI-compatible), ``dictionary`` (UI map)
    and a ``summary`` of counts.
    """
    entries, by_word = parse_dictionary(dictionary_entries)
    ui_dictionary = build_ui_dictionary(entries)
    total = len(input_records or [])
    logger.info(
        "resolve_documents: %d document(s), %d dictionary words (%d ambiguous)",
        total, len(by_word), sum(1 for e in entries if e.is_ambiguous),
    )

    documents: list[dict[str, Any]] = []
    summary = {"documents": 0, "terms": 0, "direct": 0, "llm": 0, "none": 0}

    for index, record in enumerate(input_records or []):
        if not isinstance(record, dict):
            record = {text_field: record}
        raw_text = record.get(text_field, "")
        text = raw_text if isinstance(raw_text, str) else str(raw_text or "")
        name = str(record.get("name") or record.get("id") or f"doc_{index + 1}")

        labels = resolve_document(text, by_word, llm_client)
        meta = {k: v for k, v in record.items() if k != text_field}

        documents.append(
            {
                "name": name,
                "text": text,
                "labels": labels,
                "replacements": [],
                "meta": meta,
            }
        )

        summary["documents"] += 1
        for lb in labels:
            summary["terms"] += 1
            if lb["source"] == "rule":
                summary["direct"] += 1
            elif lb["source"] == "llm":
                summary["llm"] += 1
            else:
                summary["none"] += 1

    return {"documents": documents, "dictionary": ui_dictionary, "summary": summary}
