"""PaddleOCR wrapper: lazy singleton, tolerant to PaddleOCR 2.x and 3.x APIs."""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Settings
from .imaging import ensure_bgr
from .schemas import Box

logger = logging.getLogger("yocr.ocr")


@dataclass(frozen=True)
class TextItem:
    text: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class OCREngine:
    """Thread-safe lazy PaddleOCR engine with oneDNN auto-fallback."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = threading.Lock()
        self._engine = None
        self._mkldnn = settings.ocr_mkldnn

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def _build(self):
        from paddleocr import PaddleOCR  # heavy import, keep lazy

        device = self._settings.ocr_device or self._settings.device
        kwargs: dict = {
            "lang": self._settings.ocr_lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_mkldnn": self._mkldnn and device.startswith("cpu"),
        }
        if self._settings.ocr_det_model:
            kwargs["text_detection_model_name"] = self._settings.ocr_det_model
        if self._settings.ocr_rec_model:
            kwargs["text_recognition_model_name"] = self._settings.ocr_rec_model
        try:
            logger.info(
                "initializing PaddleOCR (lang=%s device=%s mkldnn=%s)",
                self._settings.ocr_lang, device, kwargs["enable_mkldnn"],
            )
            return PaddleOCR(device=device if device != "cpu" else "cpu", **kwargs)
        except TypeError:
            # Older 2.x signature
            kwargs.pop("enable_mkldnn", None)
            kwargs.pop("use_textline_orientation", None)
            return PaddleOCR(use_angle_cls=False, show_log=False, **kwargs)

    def get(self):
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self._build()
        return self._engine

    def _rebuild_without_mkldnn(self):
        logger.warning("disabling oneDNN for PaddleOCR and rebuilding engine", exc_info=True)
        self._mkldnn = False
        # 释放旧引擎的全局资源，避免 paddle 框架层面的二次初始化冲突
        self._engine = None
        gc.collect()
        with self._lock:
            self._engine = self._build()
        return self._engine

    def recognize(self, image: np.ndarray) -> list[TextItem]:
        engine = self.get()
        image = ensure_bgr(image)
        started = time.perf_counter()
        try:
            raw = self._infer(engine, image)
        except NotImplementedError as exc:
            # Known paddle/oneDNN incompatibility on some x86 CPUs -> retry plain CPU.
            if not self._mkldnn:
                raise
            logger.error("paddle inference failed with oneDNN (%s)", exc)
            try:
                raw = self._infer(self._rebuild_without_mkldnn(), image)
            except Exception as exc2:  # noqa: BLE001 - chain both failures for diagnosis
                raise RuntimeError(
                    f"OCR engine rebuild without oneDNN also failed. "
                    f"first failure(oneDNN): {exc!r}; second failure: {exc2!r}; "
                    f"workaround: set YOCR_OCR_MKLDNN=0"
                ) from exc2
        items: list[TextItem] = []
        for page in raw or []:
            items.extend(self._parse_page(page))
        logger.info("ocr recognized %d lines in %.1fms", len(items), (time.perf_counter() - started) * 1000)
        return items

    @staticmethod
    def _infer(engine, image: np.ndarray):
        try:
            return engine.predict(image)
        except AttributeError:  # paddleocr <= 2.7 uses .ocr()
            return engine.ocr(image, cls=False)

    @staticmethod
    def _parse_page(page) -> list[TextItem]:
        items: list[TextItem] = []
        if isinstance(page, dict):
            # PaddleOCR >= 3.x result dict
            polys = page.get("rec_polys") or page.get("dt_polys") or page.get("rec_boxes") or []
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.0
                box = polys[i] if i < len(polys) else None
                items.append(TextItem(str(text), score, _poly_to_xyxy(box)))
            return items
        for entry in page or []:
            if not entry:
                continue
            try:
                poly, (text, score) = entry[0], entry[1]
            except (TypeError, ValueError, IndexError):
                continue
            items.append(TextItem(str(text), float(score), _poly_to_xyxy(poly)))
        return items


def _poly_to_xyxy(poly) -> tuple[float, float, float, float]:
    arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    return (float(x1), float(y1), float(x2), float(y2))


def item_to_line(index: int, item: TextItem):
    return index, Box.from_xyxy(*item.xyxy)


_engine: Optional[OCREngine] = None
_engine_lock = threading.Lock()


def get_ocr_engine(settings: Settings) -> OCREngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = OCREngine(settings)
    return _engine
