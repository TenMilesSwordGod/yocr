"""High-level analysis pipeline shared by the API routes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from fastapi import HTTPException

from .config import Settings
from .detectors import YOLORegistry
from .imaging import decode_base64_image, decode_image
from .ocr_engine import OCREngine, TextItem, get_ocr_engine
from .schemas import (
    AnalyzeResponse,
    Box,
    DetectResponse,
    Element,
    ImageInfo,
    OcrResponse,
    TextLine,
    Timing,
)

logger = logging.getLogger("yocr.pipeline")


@dataclass
class AnalysisContext:
    settings: Settings
    registry: YOLORegistry
    _ocr: OCREngine | None = None

    @property
    def ocr(self) -> OCREngine:
        if self._ocr is None:
            self._ocr = get_ocr_engine(self.settings)
        return self._ocr


def make_context(settings: Settings) -> AnalysisContext:
    return AnalysisContext(settings=settings, registry=YOLORegistry(settings))


def load_image(payload: bytes | None = None, base64_payload: str | None = None) -> np.ndarray:
    if payload:
        return decode_image(payload)
    if base64_payload:
        return decode_base64_image(base64_payload)
    raise ValueError("no image provided (upload file or send image_base64)")


def detect(ctx: AnalysisContext, image: np.ndarray, model: str | None = None, conf: float | None = None,
           iou: float | None = None, imgsz: int | None = None,
           classes: list[int] | None = None) -> tuple[str, list[Element], float]:
    started = time.perf_counter()
    try:
        spec, detections = ctx.registry.predict(image, model=model, conf=conf, iou=iou, imgsz=imgsz, classes=classes)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    elements = [
        Element(
            id=i,
            label=d.label,
            class_id=d.class_id,
            confidence=round(d.confidence, 4),
            box=Box.from_xyxy(*d.xyxy),
        )
        for i, d in enumerate(sorted(detections, key=lambda d: d.confidence, reverse=True))
    ]
    elapsed = (time.perf_counter() - started) * 1000
    logger.info("detect '%s': %d elements in %.1fms", spec.name, len(elements), elapsed)
    return spec.name, elements, elapsed


def run_ocr(ctx: AnalysisContext, image: np.ndarray) -> tuple[list[TextLine], str, float]:
    started = time.perf_counter()
    items: list[TextItem] = ctx.ocr.recognize(image)
    lines = [
        TextLine(text=item.text, confidence=round(item.confidence, 4), box=Box.from_xyxy(*item.xyxy))
        for item in items
    ]
    full_text = "\n".join(line.text for line in lines)
    elapsed = (time.perf_counter() - started) * 1000
    logger.info("ocr: %d lines in %.1fms", len(lines), elapsed)
    return lines, full_text, elapsed


def attach_texts(elements: list[Element], lines: list[TextLine]) -> None:
    """Assign every OCR line to the smallest element containing its center."""
    for line in lines:
        cx, cy = line.box.center
        best: Element | None = None
        best_area = float("inf")
        for element in elements:
            x1, y1, x2, y2 = element.box.xyxy
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best, best_area = element, area
        if best is not None:
            merged = f"{best.text} {line.text}".strip() if best.text else line.text
            best.text = merged
            best.text_confidence = max(best.text_confidence or 0.0, line.confidence)


def analyze(ctx: AnalysisContext, image_bytes: bytes | None, base64_image: str | None, *,
            model: str | None = None, conf: float | None = None, iou: float | None = None,
            imgsz: int | None = None, with_ocr: bool = True) -> tuple[np.ndarray, AnalyzeResponse]:
    total_started = time.perf_counter()
    image = load_image(image_bytes, base64_image)
    height, width = image.shape[:2]

    model_name, elements, detect_ms = detect(ctx, image, model=model, conf=conf, iou=iou, imgsz=imgsz)

    lines: list[TextLine] = []
    full_text = ""
    ocr_ms: float | None = None
    if with_ocr:
        lines, full_text, ocr_ms = run_ocr(ctx, image)
        attach_texts(elements, lines)

    response = AnalyzeResponse(
        model=model_name,
        image=ImageInfo(width=width, height=height),
        elements=elements,
        lines=lines,
        full_text=full_text,
        timing=Timing(
            total_ms=round((time.perf_counter() - total_started) * 1000, 1),
            detect_ms=round(detect_ms, 1),
            ocr_ms=round(ocr_ms, 1) if ocr_ms is not None else None,
        ),
    )
    return image, response


def ocr_only(ctx: AnalysisContext, image_bytes: bytes | None, base64_image: str | None) -> OcrResponse:
    started = time.perf_counter()
    image = load_image(image_bytes, base64_image)
    height, width = image.shape[:2]
    lines, full_text, elapsed = run_ocr(ctx, image)
    return OcrResponse(
        image=ImageInfo(width=width, height=height),
        lines=lines,
        full_text=full_text,
        timing=Timing(total_ms=round((time.perf_counter() - started) * 1000, 1), ocr_ms=round(elapsed, 1)),
    )


def detect_only(ctx: AnalysisContext, image_bytes: bytes | None, base64_image: str | None, *,
                model: str | None = None, conf: float | None = None, iou: float | None = None,
                imgsz: int | None = None) -> DetectResponse:
    started = time.perf_counter()
    image = load_image(image_bytes, base64_image)
    height, width = image.shape[:2]
    model_name, elements, detect_ms = detect(ctx, image, model=model, conf=conf, iou=iou, imgsz=imgsz)
    return DetectResponse(
        model=model_name,
        image=ImageInfo(width=width, height=height),
        elements=elements,
        timing=Timing(total_ms=round((time.perf_counter() - started) * 1000, 1), detect_ms=round(detect_ms, 1)),
    )


__all__ = ["AnalysisContext", "make_context", "analyze", "detect_only", "ocr_only", "attach_texts"]
