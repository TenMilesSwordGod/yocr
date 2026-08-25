"""High-level analysis pipeline shared by the API routes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from fastapi import HTTPException

from .config import Settings
from .detectors import DEFAULT_MODEL, YOLORegistry
from .imaging import decode_base64_image, decode_image
from .matching import best_match
from .ocr_engine import OCREngine, TextItem, get_ocr_engine
from .schemas import (
    AnalyzeResponse,
    Box,
    DetectResponse,
    Element,
    ImageInfo,
    MatchTemplateResponse,
    OcrResponse,
    TextLine,
    Timing,
)
from .template import locate_template

logger = logging.getLogger("yocr.pipeline")


def _resolve_target(elements: list[Element], *, text: str | None, label: str | None,
                    q: str | None, match_mode: str) -> tuple[bool, Element | None]:
    matched = best_match(elements, text=text, label=label, q=q, match_mode=match_mode)
    return matched is not None, matched


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
    if model:
        # Explicitly pinned model: strict behavior, missing weights -> 404.
        try:
            return _detect_with(ctx, image, model, conf, iou, imgsz, classes, started)
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Default model requested implicitly: fall back to any other registered
    # model when its weights are absent (fresh servers without models/*.pt).
    default_key = DEFAULT_MODEL.lower()
    candidates = [None] + [n for n in ctx.registry.names() if n.lower() != default_key]
    failures: list[str] = []
    for candidate in candidates:
        try:
            return _detect_with(ctx, image, candidate, conf, iou, imgsz, classes, started)
        except (FileNotFoundError, ImportError) as exc:
            name = candidate or DEFAULT_MODEL
            logger.warning("model '%s' unavailable (%s); trying fallback", name, exc)
            failures.append(f"{name}: {exc}")
            ctx.registry._errors.setdefault(name, f"{type(exc).__name__}: {exc}")  # noqa: SLF001
    raise HTTPException(
        status_code=404,
        detail="no usable detection model on this host — " + "; ".join(failures)
        + ". Fix: place .pt weights in YOCR_MODELS_DIR, pre-download HF weights "
        "(make models-download / hf download), or register aliases via YOCR_MODEL_ALIASES",
    )


def _detect_with(ctx: AnalysisContext, image: np.ndarray, model: str | None,
                 conf: float | None, iou: float | None, imgsz: int | None,
                 classes: list[int] | None, started: float) -> tuple[str, list[Element], float]:
    spec, detections = ctx.registry.predict(image, model=model, conf=conf, iou=iou, imgsz=imgsz, classes=classes)
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
    try:
        items: list[TextItem] = ctx.ocr.recognize(image)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface engine problems as clean 503
        logger.exception("OCR engine unavailable")
        raise HTTPException(
            status_code=503,
            detail=f"OCR engine unavailable: {exc!r}. "
                   f"若为 oneDNN 兼容问题，可设置 YOCR_OCR_MKLDNN=0 后重启",
        ) from exc
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
            imgsz: int | None = None, with_ocr: bool = True,
            text: str | None = None, label: str | None = None, q: str | None = None,
            match_mode: str = "contains") -> tuple[np.ndarray, AnalyzeResponse]:
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

    found, matched = _resolve_target(elements, text=text, label=label, q=q, match_mode=match_mode)

    response = AnalyzeResponse(
        model=model_name,
        image=ImageInfo(width=width, height=height),
        elements=elements,
        found=found,
        matched=matched,
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
                imgsz: int | None = None,
                text: str | None = None, label: str | None = None, q: str | None = None,
                match_mode: str = "contains") -> DetectResponse:
    started = time.perf_counter()
    image = load_image(image_bytes, base64_image)
    height, width = image.shape[:2]
    model_name, elements, detect_ms = detect(ctx, image, model=model, conf=conf, iou=iou, imgsz=imgsz)
    found, matched = _resolve_target(elements, text=text, label=label, q=q, match_mode=match_mode)
    return DetectResponse(
        model=model_name,
        image=ImageInfo(width=width, height=height),
        elements=elements,
        found=found,
        matched=matched,
        timing=Timing(total_ms=round((time.perf_counter() - started) * 1000, 1), detect_ms=round(detect_ms, 1)),
    )


def match_template_images(
    scene_bytes: bytes | None,
    scene_b64: str | None,
    template_bytes: bytes | None,
    template_b64: str | None,
    *,
    threshold: float = 0.8,
) -> MatchTemplateResponse:
    """Locate a small template image inside a larger scene screenshot.

    Args:
        scene_bytes: Raw scene image bytes (multipart upload).
        scene_b64: Scene image as base64 (JSON body).
        template_bytes: Raw template image bytes.
        template_b64: Template image as base64.
        threshold: Minimum normalized-correlation score to count as found.

    Returns:
        MatchTemplateResponse: found/score/box/scale plus both image sizes.

    Raises:
        ValueError: Missing or corrupted images.
    """
    started = time.perf_counter()
    scene = load_image(scene_bytes, scene_b64)
    template = load_image(template_bytes, template_b64)
    hit = locate_template(scene, template, threshold=threshold)
    height, width = scene.shape[:2]
    theight, twidth = template.shape[:2]
    return MatchTemplateResponse(
        found=hit is not None,
        score=round(hit.confidence, 4) if hit else 0.0,
        threshold=threshold,
        scale=hit.scale if hit else 1.0,
        box=Box.from_xyxy(*hit.xyxy) if hit else None,
        image=ImageInfo(width=width, height=height),
        template=ImageInfo(width=twidth, height=theight),
        timing=Timing(total_ms=round((time.perf_counter() - started) * 1000, 1)),
    )


__all__ = [
    "AnalysisContext",
    "make_context",
    "analyze",
    "detect_only",
    "ocr_only",
    "attach_texts",
    "match_template_images",
]
