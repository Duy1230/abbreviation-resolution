FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# Default LLM target; override at run time with -e LLAMA_SERVER_URL=...
ENV LLAMA_SERVER_URL=http://host.docker.internal:8080/v1 \
    LLAMA_MODEL=local-model \
    LLAMA_TIMEOUT=120

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
