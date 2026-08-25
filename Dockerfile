# yocr - YOLO + PaddleOCR vision service for Android/AAOS testing
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# ---- frontend (Vue SPA) ----
FROM node:22-slim AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
RUN npm run build

# ---- runtime ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Runtime libs for opencv / paddle on cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 libglib2.0-0 tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY --from=frontend /web/dist ./frontend/dist

RUN uv sync --frozen --no-dev

ENV YOCR_MODELS_DIR=/app/models \
    YOCR_HOST=0.0.0.0 \
    YOCR_PORT=8000 \
    HF_HOME=/app/.cache/huggingface \
    YOCR_SUCAI_DIR=/app/data/sucai \
    PYTHONUNBUFFERED=1

EXPOSE 8000
VOLUME ["/app/models", "/app/data"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uv", "run", "--no-sync", "yocr", "serve"]
