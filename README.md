# Abbreviation Resolution

Tự động phân giải (disambiguation) từ viết tắt tiếng Việt bằng LLM chạy trên
[llama.cpp](https://github.com/ggml-org/llama.cpp), kèm giao diện web để chạy và
review kết quả — không cần gõ lệnh hay sửa code.

Kết quả xuất ra JSON **tương thích hoàn toàn** với trình gán nhãn
`dictionary_labeler_v4.html` (gốc), nên bạn có thể mở lại để kiểm tra/sửa và export.

---

## Ý tưởng

- **Đầu vào 1 — văn bản thô:** JSON `list` các object, văn bản nằm ở trường `input`.
- **Đầu vào 2 — từ điển viết tắt:** JSON `list` các object có `word`, `type`,
  `meaning`, `label`. `type` và `meaning` là các danh sách song song ngăn cách bởi
  `/`. `label = 0` (không nhập nhằng) gán trực tiếp; `label = 1` (nhập nhằng) để LLM chọn.

  ```json
  { "word": "TC", "type": "tên tàu/khac", "meaning": "Tàu cá/TC", "label": 1 }
  ```

- **Nhận diện từ viết tắt trong văn bản:** chỉ token IN HOA, không dính chữ
  thường/số; các ký tự `.,:;/-+=()` được coi như khoảng trắng nên `TP-HCM` tách
  thành `TP` và `HCM`. Quy tắc này khớp đúng với biên regex của trình gán nhãn.
- **Gọi LLM một lần cho mỗi văn bản:** mọi từ nhập nhằng trong một văn bản được
  gom lại và hỏi LLM trong **một** request; LLM trả về index nghĩa được chọn cho
  từng từ, hoặc `-1` nếu không nghĩa nào phù hợp.

```mermaid
flowchart LR
  inp["input.json + dictionary.json"] --> match["Tach token + match dictionary"]
  match -->|label 0| direct["Gan truc tiep"]
  match -->|label 1| llm["llama-server (1 call/van ban)"]
  llm --> out["labeled_documents.json + dictionary.json"]
  direct --> out
  out --> review["Trinh gan nhan (review & export)"]
```

---

## Chạy nhanh bằng Docker (khuyến nghị)

Yêu cầu: đã có một `llama-server` (llama.cpp) đang chạy và mở cổng OpenAI-compatible,
ví dụ:

```bash
llama-server -m model.gguf --host 0.0.0.0 --port 8080 -c 8192
```

Chạy ứng dụng:

```bash
docker run --rm -p 8000:8000 \
  -e LLAMA_SERVER_URL=http://host.docker.internal:8080/v1 \
  -e LLAMA_MODEL=local-model \
  epsilon1234/abbreviation-resolution:latest
```

Mở trình duyệt tại <http://localhost:8000>:

1. **Bước 1** — chọn file văn bản thô (`.json`) và file từ điển (`.json`).
2. **Bước 2** — kiểm tra URL `llama-server` rồi bấm "Kiểm tra kết nối". (Hoặc bật
   "Chạy thử" để xem pipeline mà không cần model.)
3. **Bước 3** — bấm "Chạy phân giải".
4. Tải `labeled_documents.json` / `dictionary.json`, hoặc bấm
   "Review & sửa trong trình gán nhãn" để mở trang labeler với dữ liệu nạp sẵn.

> Trên Linux, nếu `host.docker.internal` không hoạt động, dùng IP host hoặc thêm
> `--add-host=host.docker.internal:host-gateway`.

### Dùng docker compose

```bash
LLAMA_SERVER_URL=http://host.docker.internal:8080/v1 docker compose up
```

---

## Cấu hình (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `LLAMA_SERVER_URL` | `http://host.docker.internal:8080/v1` | Base URL OpenAI-compatible của llama-server |
| `LLAMA_MODEL` | `local-model` | Tên model gửi trong request |
| `LLAMA_API_KEY` | (trống) | Bearer token nếu server yêu cầu |
| `LLAMA_TIMEOUT` | `120` | Timeout (giây) cho mỗi request |

URL/model cũng có thể nhập trực tiếp trên giao diện, ghi đè giá trị mặc định.

---

## Định dạng đầu ra (tương thích trình gán nhãn)

`labeled_documents.json`:

```json
[
  {
    "name": "doc1",
    "text": "... TC ...",
    "labels": [
      {
        "start": 4, "end": 6, "term": "TC",
        "senseId": "tc_1", "senseLabel": "Tàu cá", "senseExplanation": "tên tàu",
        "text": "TC", "auto": false, "source": "llm"
      }
    ],
    "replacements": [],
    "meta": { "id": "doc1" }
  }
]
```

`dictionary.json` (map `WORD -> [{id, label, explanation}]`): nạp được trực tiếp qua
ô "Load dictionary" của trình gán nhãn. Các từ `label = 1` có thêm một nghĩa đặc
biệt `(không có nghĩa phù hợp)` để biểu diễn trường hợp `-1`.

`source` trong mỗi nhãn: `rule` (gán trực tiếp), `llm` (LLM chọn), `llm-none`
(LLM cho rằng không nghĩa nào phù hợp).

---

## Phát triển & kiểm thử

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload --port 8000
```

Bộ test dùng `MockLLMClient` nên chạy được mà không cần model. Để thử end-to-end
với dữ liệu mẫu, bật "Chạy thử" trên giao diện và tải `sample_data/`.

---

## Cấu trúc dự án

```
app/
  main.py          FastAPI: routes + streaming /api/resolve + /api/test-llm
  resolver.py      Logic cốt lõi (tách token, match, gán nhãn) — thuần Python
  llm_client.py    Client llama-server + MockLLMClient + prompt/schema + logging
  static/
    index.html     Trang Auto-resolve (upload + chạy LLM + progress streaming)
    labeler.html   Trình gán nhãn (bản gốc + nút Import + bàn giao localStorage)
sample_data/       Dữ liệu mẫu để thử
tests/             pytest
Dockerfile, docker-compose.yml
```

---

## Streaming protocol (cho người tự tích hợp API)

`POST /api/resolve` trả `application/x-ndjson` — mỗi dòng là một JSON event:

```jsonc
{"event":"start",          "total":3, "dictionary_words":6, "ambiguous_words":2, "dry_run":false}
{"event":"document_start", "index":1, "total":3, "name":"doc1", "text_len":144}
{"event":"document_done",  "index":1, "total":3, "name":"doc1",
                           "labels":5, "direct":3, "llm":2, "none":0, "elapsed_ms":2310}
{"event":"document_error", "index":2, "total":3, "name":"doc2",
                           "error_type":"LLMError", "error":"..."}
{"event":"done", "documents":[...], "dictionary":{...},
                 "summary":{"documents":3,"terms":9,"direct":6,"llm":3,"none":0}}
{"event":"fatal", "error_type":"...", "error":"..."}      // chỉ khi stream chết toàn cục
```

Vì sao streaming? Một request resolve có thể mất nhiều phút khi LLM chậm. Với
JSON buffered (chế độ cũ), proxy/firewall doanh nghiệp thường reset connection
POST đang idle → browser nhận `NetworkError`. Streaming 1 event/văn bản giữ
luồng byte chảy liên tục và đồng thời cho UI biết tiến trình thật.

---

## Troubleshooting

### "NetworkError when attempting to fetch resource" khi bấm "Chạy phân giải"

Phổ biến trong môi trường doanh nghiệp; bạn cần kiểm tra theo thứ tự:

1. **Xem log uvicorn**. Phiên bản từ `1.1.0` log mọi request tới LLM kèm timing:
   ```
   [app.llm_client] INFO: POST http://.../v1/chat/completions | bytes=1843 | model=qwen3.6 | connect=10.0s read=120.0s | trust_env=False
   [app.llm_client] INFO: POST http://.../v1/chat/completions -> HTTP 200 in 12.34s
   ```
   Nếu không thấy log "-> HTTP ..." → LLM chưa trả về (hang / quá chậm).
2. **Bấm "Test LLM" trên giao diện** (hoặc `POST /api/test-llm`). Endpoint này
   gửi prompt nhỏ nhất qua cùng `httpx.Client` mà pipeline thật dùng:
   - Trả về OK + `elapsed_ms` → LLM healthy; vấn đề là prompt thật chậm hoặc
     proxy/firewall reset connection dài. Streaming (đã bật mặc định) thường
     đã giải quyết.
   - Trả về lỗi `ConnectError`/`ConnectTimeout` → app không vào được LLM (
     khác với curl từ máy bạn): kiểm tra `LLAMA_SERVER_URL`, DNS trong
     container, proxy env.
3. **Bypass proxy doanh nghiệp**. `LLMClient` đặt `trust_env=False` để
   **bỏ qua** `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` của môi trường — đây là
   nguyên nhân kinh điển khiến "curl OK mà Python hang". Nếu bạn THỰC SỰ
   cần qua proxy, sửa code (`trust_env=True`) hoặc đặt `NO_PROXY=192.168.*`.
4. **Tăng `LLAMA_TIMEOUT`** nếu model chậm (mặc định 120s, nên đặt 300-600s
   cho model lớn).
5. **Tăng nhẹ keep-alive trong UI**: streaming đã đẩy 1 dòng/văn bản nên
   connection không bao giờ idle quá thời gian xử lý 1 văn bản. Nếu vẫn bị
   reset, có thể proxy reset cả connection có data mà response chưa đóng → cần
   liên hệ team mạng để whitelist.

### "Kiểm tra kết nối" OK nhưng "Chạy phân giải" hỏng

Hai endpoint dùng đường HTTP khác nhau ở phía LLM:
`/v1/models` (nhỏ, nhanh) vs `/v1/chat/completions` (lớn, chậm). Vậy nên
"Kiểm tra kết nối" pass không đảm bảo chat hoạt động. **Hãy luôn bấm "Test LLM"
trước khi chạy real workload.**

### `Ctrl + C` không thoát uvicorn

Uvicorn graceful shutdown đợi mọi request đang chạy xong. Khi LLM treo, request
kẹt đến hết `LLAMA_TIMEOUT`. Bấm **`Ctrl + C` hai lần liên tiếp** để force-kill,
hoặc chạy uvicorn với `--timeout-graceful-shutdown 5`.

### `NetworkError` khi mở qua nginx / Cloudflare / proxy ngược

Phiên bản hiện tại đã set header `X-Accel-Buffering: no` và `Cache-Control:
no-cache`. Nếu proxy của bạn không tôn trọng các header này, tắt buffering thủ
công ở proxy (nginx: `proxy_buffering off;`).
