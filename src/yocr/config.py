"""Runtime configuration loaded from environment variables (YOCR_*)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # NOTE: every default is resolved lazily (at instantiation) so tests/CLI can
    # set env vars right up until Settings() is constructed.
    host: str = field(default_factory=lambda: _env("YOCR_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("YOCR_PORT", 8000))

    # Directory that holds local .pt model files.
    models_dir: Path = field(default_factory=lambda: Path(_env("YOCR_MODELS_DIR", "models")))

    # Torch device for YOLO inference: cpu | cuda:0 | mps
    device: str = field(default_factory=lambda: _env("YOCR_DEVICE", "cpu"))

    # Input resolution for YOLO inference (larger => more accurate small UI elements).
    infer_size: int = field(default_factory=lambda: _env_int("YOCR_INFER_SIZE", 1280))
    conf_threshold: float = field(default_factory=lambda: _env_float("YOCR_CONF", 0.25))
    iou_threshold: float = field(default_factory=lambda: _env_float("YOCR_IOU", 0.45))

    # PaddleOCR language: ch | en | ...
    ocr_lang: str = field(default_factory=lambda: _env("YOCR_OCR_LANG", "ch"))
    ocr_device: str = field(default_factory=lambda: _env("YOCR_OCR_DEVICE", ""))  # "" => follow YOCR_DEVICE
    # oneDNN acceleration for CPU inference; auto-disabled at runtime if the
    # paddle build turns out to be incompatible.
    ocr_mkldnn: bool = field(default_factory=lambda: _env_bool("YOCR_OCR_MKLDNN", True))
    # Explicit det/rec model selection for latency vs precision control.
    # Defaults to the mobile models (much faster on x86 CPU); set e.g.
    # YOCR_OCR_DET_MODEL=PP-OCRv5_server_det for maximum accuracy.
    ocr_det_model: str = field(default_factory=lambda: _env("YOCR_OCR_DET_MODEL", "PP-OCRv5_mobile_det"))
    ocr_rec_model: str = field(default_factory=lambda: _env("YOCR_OCR_REC_MODEL", "PP-OCRv5_mobile_rec"))

    # Models pre-loaded at startup. Unset => "all" builtin models; set to
    # empty string to disable preloading entirely; or a comma list of names.
    preload_models: tuple[str, ...] = field(default_factory=lambda: _env_list("YOCR_PRELOAD_MODELS", "all"))
    preload_ocr: bool = field(default_factory=lambda: _env_bool("YOCR_PRELOAD_OCR", True))

    # Extra model aliases: "name=path_or_hf_id,name2=path2.pt"
    model_aliases_raw: str = field(default_factory=lambda: _env("YOCR_MODEL_ALIASES", ""))

    # IconFinder open-vocabulary classes (comma separated); empty => builtin defaults
    icon_prompts_raw: tuple[str, ...] = field(default_factory=lambda: _env_list("YOCR_ICON_CLASSES"))

    # Allow runtime weight downloads (HF/GitHub). Default off: provision with
    # `make models-download`; flip to 1 to restore lazy auto-download.
    allow_download: bool = field(default_factory=lambda: _env_bool("YOCR_ALLOW_DOWNLOAD", False))

    # Sucai (素材) template registry storage directory.
    sucai_dir: Path = field(default_factory=lambda: Path(_env("YOCR_SUCAI_DIR", "data/sucai")))

    # Built SPA directory served at "/" when it exists (see frontend/).
    static_dir: Path = field(default_factory=lambda: Path(_env("YOCR_STATIC_DIR", "frontend/dist")))

    log_level: str = field(default_factory=lambda: _env("YOCR_LOG_LEVEL", "INFO"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def parse_model_aliases(raw: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, source = item.split("=", 1)
        name, source = name.strip(), source.strip()
        if name and source:
            aliases[name.lower()] = source
    return aliases
