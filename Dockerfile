FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config.json ./config.json

EXPOSE 8000

# Defaults come from config.json (editable in the UI / persisted via POST /api/config).
# Any LLAMA_*/AR_* env var still overrides at run time, e.g. -e LLAMA_SERVER_URL=...
# Checkpoints are written here; mount a volume to keep them across restarts:
#   docker run ... -v "$PWD/checkpoints:/srv/checkpoints" epsilon1234/abbreviation-resolution
ENV AR_CONFIG_PATH=/srv/config.json \
    AR_CHECKPOINT_DIR=/srv/checkpoints

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
