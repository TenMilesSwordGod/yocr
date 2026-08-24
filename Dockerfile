# yocr - YOLO + PaddleOCR vision service for Android/AAOS testing
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Runtime libs for opencv / paddle on cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 libglib2.0-0 tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

ENV YOCR_MODELS_DIR=/app/models \
    YOCR_HOST=0.0.0.0 \
    YOCR_PORT=8000 \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000
VOLUME ["/app/models"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uv", "run", "--no-sync", "yocr", "serve"]
