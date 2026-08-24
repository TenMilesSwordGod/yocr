"""Image decoding / encoding helpers built on OpenCV."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import cv2
import numpy as np

PNG_MAGIC = b"\x89PNG"


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw bytes (png/jpeg/bmp/webp/...) into a BGR image."""
    if not data:
        raise ValueError("empty image payload")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        # Some Windows-side encoders/proxies mangle \n -> \r\n in binary payloads.
        fixed = data.replace(b"\r\n", b"\n")
        if fixed != data:
            image = cv2.imdecode(np.frombuffer(fixed, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unsupported or corrupted image payload")
    return image


def decode_base64_image(payload: str) -> np.ndarray:
    try:
        return decode_image(base64.b64decode(payload, validate=False))
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 image: {exc}") from exc


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode png")
    return buffer.tobytes()


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """PaddleOCR accepts ndarray directly; ultralytics too. Keep a hook for RGB conversions."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def resolve_model_file(models_dir: Path, source: str) -> str:
    """Resolve `source` against models_dir; returns absolute path string when it is a file."""
    candidate = Path(source)
    if candidate.is_absolute() and candidate.is_file():
        return str(candidate)
    relative = (models_dir / source).resolve()
    if relative.is_file():
        return str(relative)
    return source  # assume HF hub id / remote url
