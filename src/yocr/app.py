"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Anchor weight caches to <cwd>/.cache so manual starts (`make run`) resolve the
# same caches that `make models-download` and the systemd unit populate.
os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path.cwd() / ".cache" / "paddlex"))

from fastapi import FastAPI

from .api import router as api_router
from .config import get_settings
from .detectors import warmup
from .pipeline import make_context


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger = logging.getLogger("yocr.app")
    started = time.perf_counter()
    ctx = make_context(settings)
    app.state.ctx = ctx

    if settings.preload_models:
        warmup(ctx.registry, settings)
    if settings.preload_ocr:
        try:
            ctx.ocr.get()
            logger.info("PaddleOCR ready")
        except Exception as exc:  # noqa: BLE001 - OCR stays optional at boot
            logger.error("PaddleOCR init failed (OCR endpoints will 503): %s", exc)

    logger.info("yocr ready in %.1fs on %s:%d", time.perf_counter() - started, settings.host, settings.port)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="yocr",
        description="YOLO + PaddleOCR visual recognition service for Android/AAOS automated testing",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app
