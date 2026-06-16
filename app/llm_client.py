"""LLM access for abbreviation disambiguation.

The real client talks to an OpenAI-compatible ``llama.cpp`` ``llama-server``
(``POST {base_url}/chat/completions``). For each document we send exactly one
request containing the full text plus every ambiguous abbreviation and its
candidate meanings, and ask the model to return a strict JSON object choosing an
index per word (or ``-1`` when none of the meanings fit).

``MockLLMClient`` implements the same ``resolve()`` interface deterministically
for unit tests and for the UI "dry-run" mode (no model required).
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân giải (disambiguation) từ viết tắt tiếng Việt. "
    "Nhiệm vụ: dựa HOÀN TOÀN vào ngữ cảnh của văn bản được cung cấp để chọn nghĩa "
    "đúng cho mỗi từ viết tắt nhập nhằng. Chỉ được chọn trong các nghĩa ứng viên "
    "đã liệt kê. Nếu KHÔNG có nghĩa nào phù hợp với ngữ cảnh, hãy trả về choice = -1. "
    "Chỉ trả về JSON đúng schema, tuyệt đối không thêm lời giải thích."
)

PROMPT_TEMPLATE = """VĂN BẢN:
\"\"\"
{text}
\"\"\"

CÁC TỪ VIẾT TẮT CẦN PHÂN GIẢI (chọn index nghĩa phù hợp nhất theo ngữ cảnh, hoặc -1 nếu không nghĩa nào đúng):
{listing}

Trả về JSON theo dạng: {{"resolutions": [{{"word": "<từ>", "choice": <index hoặc -1>}}, ...]}}
Mỗi từ ở trên phải có đúng một mục tương ứng trong "resolutions"."""


class LLMError(Exception):
    """Raised when the LLM cannot be reached or returns unusable output."""


def build_schema() -> dict[str, Any]:
    """JSON schema used to constrain llama.cpp output."""
    return {
        "type": "object",
        "properties": {
            "resolutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "choice": {"type": "integer"},
                    },
                    "required": ["word", "choice"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["resolutions"],
        "additionalProperties": False,
    }


def build_listing(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        senses = item["senses"]
        cands = "; ".join(
            f"[{i}] ({getattr(s, 'explanation', '')}) {getattr(s, 'label', '')}"
            for i, s in enumerate(senses)
        )
        lines.append(f'- word="{item["word"]}": {cands}')
    return "\n".join(lines)


def build_messages(text: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    user = PROMPT_TEMPLATE.format(text=text, listing=build_listing(items))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _extract_json(content: Any) -> Any:
    if isinstance(content, (dict, list)):
        return content
    s = str(content).strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start : end + 1])
    raise ValueError("Không tìm thấy JSON hợp lệ trong phản hồi LLM.")


def parse_resolutions(content: Any, items: list[dict[str, Any]]) -> dict[str, int]:
    """Parse model output into ``{word: choice}``.

    Robust to code fences / surrounding prose. Any word the model omitted (or that
    cannot be parsed) defaults to ``-1`` (treated as "none fit" downstream).
    """
    data = _extract_json(content)
    if isinstance(data, dict):
        arr = data.get("resolutions", [])
    elif isinstance(data, list):
        arr = data
    else:
        arr = []

    mapping: dict[str, int] = {}
    for it in arr:
        if not isinstance(it, dict):
            continue
        word = str(it.get("word", "")).strip()
        if not word:
            continue
        value = it.get("choice", it.get("index", -1))
        try:
            mapping[word] = int(value)
        except (TypeError, ValueError):
            mapping[word] = -1

    return {item["word"]: mapping.get(item["word"], -1) for item in items}


class LLMClient:
    """OpenAI-compatible chat client for llama.cpp ``llama-server``."""

    def __init__(
        self,
        base_url: str,
        model: str = "local-model",
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or "local-model"
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        response = httpx.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def resolve(self, text: str, items: list[dict[str, Any]]) -> dict[str, int]:
        if not items:
            return {}

        messages = build_messages(text, items)
        schema = build_schema()
        base_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        # Try progressively more permissive structured-output modes for broad
        # llama.cpp version compatibility.
        response_formats: list[dict[str, Any] | None] = [
            {
                "type": "json_schema",
                "json_schema": {"name": "abbr_resolution", "schema": schema, "strict": True},
            },
            {"type": "json_object"},
            None,
        ]

        last_error: Exception | None = None
        for response_format in response_formats:
            payload = dict(base_payload)
            if response_format is not None:
                payload["response_format"] = response_format
            try:
                data = self._post(payload)
                content = data["choices"][0]["message"]["content"]
                return parse_resolutions(content, items)
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                continue

        raise LLMError(f"Gọi LLM thất bại: {last_error}")


class MockLLMClient:
    """Deterministic stand-in used by tests and the dry-run mode.

    Strategies:
    * ``first``   -> always choose index 0
    * ``none``    -> always choose -1
    * ``context`` -> choose the first sense whose meaning text appears in the
      document (case-insensitive); fall back to 0
    Per-word overrides via ``choices`` always win.
    """

    def __init__(self, strategy: str = "context", choices: dict[str, int] | None = None) -> None:
        self.strategy = strategy
        self.choices = choices or {}

    def resolve(self, text: str, items: list[dict[str, Any]]) -> dict[str, int]:
        lowered = (text or "").lower()
        result: dict[str, int] = {}
        for item in items:
            word = item["word"]
            senses = item["senses"]
            if word in self.choices:
                result[word] = self.choices[word]
            elif self.strategy == "none":
                result[word] = -1
            elif self.strategy == "context":
                picked = 0
                for i, sense in enumerate(senses):
                    meaning = (getattr(sense, "label", "") or "").lower()
                    if meaning and meaning in lowered:
                        picked = i
                        break
                result[word] = picked
            else:  # "first" / default
                result[word] = 0
        return result


def check_connection(base_url: str, api_key: str | None = None, timeout: float = 10.0) -> dict[str, Any]:
    """Lightweight reachability check against ``{base_url}/models``."""
    url = f"{(base_url or '').rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = httpx.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()
