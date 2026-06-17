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
import logging
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân giải (disambiguation) từ viết tắt tiếng Việt. "
    "Nhiệm vụ: dựa HOÀN TOÀN vào ngữ cảnh xung quanh TỪNG VỊ TRÍ để chọn nghĩa "
    "đúng cho từ viết tắt nhập nhằng tại vị trí đó. Mỗi vị trí cần phân giải được "
    "đánh dấu bằng «số» ngay trước từ. LƯU Ý QUAN TRỌNG: cùng một từ viết tắt xuất "
    "hiện ở nhiều vị trí CÓ THỂ mang nghĩa khác nhau — phải xét riêng từng vị trí. "
    "Chỉ được chọn trong các nghĩa ứng viên đã liệt kê cho vị trí đó. Nếu KHÔNG có "
    "nghĩa nào phù hợp với ngữ cảnh, hãy trả về choice = -1. "
    "Chỉ trả về JSON đúng schema, tuyệt đối không thêm lời giải thích."
)

PROMPT_TEMPLATE = """VĂN BẢN (mỗi vị trí cần phân giải được đánh dấu «số» ngay trước từ viết tắt):
\"\"\"
{text}
\"\"\"

CÁC VỊ TRÍ CẦN PHÂN GIẢI (với mỗi «số», chọn index nghĩa phù hợp nhất theo NGỮ CẢNH QUANH VỊ TRÍ ĐÓ, hoặc -1 nếu không nghĩa nào đúng):
{listing}

Cùng một từ ở các vị trí khác nhau có thể có choice khác nhau — hãy xét độc lập từng vị trí.
Trả về JSON theo dạng: {{"resolutions": [{{"id": <số vị trí>, "choice": <index hoặc -1>}}, ...]}}
Mỗi vị trí «số» ở trên phải có đúng một mục tương ứng trong "resolutions"."""


class LLMError(Exception):
    """Raised when the LLM cannot be reached or returns unusable output."""


def build_schema() -> dict[str, Any]:
    """JSON schema used to constrain llama.cpp output (one entry per occurrence)."""
    return {
        "type": "object",
        "properties": {
            "resolutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "choice": {"type": "integer"},
                    },
                    "required": ["id", "choice"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["resolutions"],
        "additionalProperties": False,
    }


def build_listing(items: list[dict[str, Any]]) -> str:
    """One line per *occurrence*: ``«id» word="W": [0] (type) meaning; ...``."""
    lines = []
    for item in items:
        senses = item["senses"]
        cands = "; ".join(
            f"[{i}] ({getattr(s, 'explanation', '')}) {getattr(s, 'label', '')}"
            for i, s in enumerate(senses)
        )
        lines.append(f'«{item["id"]}» word="{item["word"]}": {cands}')
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


def parse_resolutions(content: Any, items: list[dict[str, Any]]) -> dict[int, int]:
    """Parse model output into ``{occurrence_id: choice}``.

    Robust to code fences / surrounding prose. Any occurrence id the model omitted
    (or that cannot be parsed) defaults to ``-1`` (treated as "none fit" downstream).
    """
    data = _extract_json(content)
    if isinstance(data, dict):
        arr = data.get("resolutions", [])
    elif isinstance(data, list):
        arr = data
    else:
        arr = []

    mapping: dict[int, int] = {}
    for it in arr:
        if not isinstance(it, dict):
            continue
        raw_id = it.get("id", it.get("pos", it.get("position")))
        try:
            oid = int(raw_id)
        except (TypeError, ValueError):
            continue
        value = it.get("choice", -1)
        try:
            mapping[oid] = int(value)
        except (TypeError, ValueError):
            mapping[oid] = -1

    return {int(item["id"]): mapping.get(int(item["id"]), -1) for item in items}


class LLMClient:
    """OpenAI-compatible chat client for llama.cpp ``llama-server``."""

    def __init__(
        self,
        base_url: str,
        model: str = "local-model",
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        trust_env: bool = False,
        connect_timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or "local-model"
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        # trust_env=False makes httpx ignore HTTP(S)_PROXY / NO_PROXY / NETRC from the
        # environment. A corporate proxy exported via these vars is a very common reason
        # an internal llama-server is reachable by curl but hangs from httpx. Keep it OFF
        # unless the model truly must be reached through a proxy (LLAMA_TRUST_ENV=1).
        self.trust_env = trust_env
        self.connect_timeout = connect_timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _timeout(self) -> httpx.Timeout:
        # Cap the connect phase so an unreachable host / stuck proxy fails fast instead
        # of blocking for the full (possibly very large) read timeout.
        connect = min(self.connect_timeout, self.timeout) if self.timeout else self.connect_timeout
        return httpx.Timeout(self.timeout, connect=connect)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False)
        timeout = self._timeout()
        logger.info(
            "POST %s | bytes=%d | model=%s | connect=%.1fs read=%.1fs | trust_env=%s",
            url, len(body.encode("utf-8")), payload.get("model"),
            timeout.connect or 0.0, timeout.read or 0.0, self.trust_env,
        )
        t0 = time.monotonic()
        with httpx.Client(trust_env=self.trust_env, timeout=timeout) as client:
            response = client.post(url, content=body, headers=self._headers())
            elapsed = time.monotonic() - t0
            logger.info(
                "POST %s -> HTTP %d in %.2fs (response bytes=%d)",
                url, response.status_code, elapsed, len(response.content),
            )
            response.raise_for_status()
            return response.json()

    def resolve(self, text: str, items: list[dict[str, Any]]) -> dict[int, int]:
        if not items:
            return {}

        occ = [(it["id"], it["word"]) for it in items]
        logger.info("LLM resolve %d occurrence(s): %s", len(items), occ)

        messages = build_messages(text, items)
        schema = build_schema()
        base_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        # Try progressively more permissive structured-output modes for broad
        # llama.cpp version compatibility.
        response_formats: list[tuple[str, dict[str, Any] | None]] = [
            (
                "json_schema",
                {
                    "type": "json_schema",
                    "json_schema": {"name": "abbr_resolution", "schema": schema, "strict": True},
                },
            ),
            ("json_object", {"type": "json_object"}),
            ("none", None),
        ]

        last_error: Exception | None = None
        for mode, response_format in response_formats:
            payload = dict(base_payload)
            if response_format is not None:
                payload["response_format"] = response_format
            try:
                data = self._post(payload)
                content = data["choices"][0]["message"]["content"]
                result = parse_resolutions(content, items)
                logger.info("LLM resolve OK (response_format=%s) -> %s", mode, result)
                return result
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "LLM resolve failed (response_format=%s): %s: %s",
                    mode, type(exc).__name__, exc,
                )
                # Network-layer errors (timeout, connect refused, read error)
                # won't change between response_format modes — fail fast.
                if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
                    break
                continue

        raise LLMError(
            f"Gọi LLM thất bại sau khi thử các response_format. "
            f"Lỗi cuối: {type(last_error).__name__}: {last_error}"
        )


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
        # ``choices`` is keyed by *word* (applies to every occurrence of that word);
        # ``choices_by_id`` (optional) overrides a specific occurrence id.
        self.choices = choices or {}
        self.choices_by_id: dict[int, int] = {}

    def resolve(self, text: str, items: list[dict[str, Any]]) -> dict[int, int]:
        lowered = (text or "").lower()
        result: dict[int, int] = {}
        for item in items:
            oid = int(item["id"])
            word = item["word"]
            senses = item["senses"]
            if oid in self.choices_by_id:
                result[oid] = self.choices_by_id[oid]
            elif word in self.choices:
                result[oid] = self.choices[word]
            elif self.strategy == "none":
                result[oid] = -1
            elif self.strategy == "context":
                picked = 0
                for i, sense in enumerate(senses):
                    meaning = (getattr(sense, "label", "") or "").lower()
                    if meaning and meaning in lowered:
                        picked = i
                        break
                result[oid] = picked
            else:  # "first" / default
                result[oid] = 0
        return result


def check_connection(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 10.0,
    trust_env: bool = False,
) -> dict[str, Any]:
    """Lightweight reachability check against ``{base_url}/models``.

    ``trust_env=False`` (default) makes httpx ignore HTTP(S)_PROXY/NO_PROXY/NETRC
    from the environment — same defensive choice as ``LLMClient``, since a
    corporate proxy exported via env vars is a frequent reason an internal
    server is reachable by curl but hangs from httpx.
    """
    url = f"{(base_url or '').rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    logger.info("GET %s | timeout=%.1fs | trust_env=%s", url, timeout, trust_env)
    t0 = time.monotonic()
    with httpx.Client(trust_env=trust_env, timeout=timeout) as client:
        response = client.get(url, headers=headers)
        elapsed = time.monotonic() - t0
        logger.info("GET %s -> HTTP %d in %.2fs", url, response.status_code, elapsed)
        response.raise_for_status()
        return response.json()
