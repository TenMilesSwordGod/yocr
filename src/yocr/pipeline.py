"""High-level analysis pipeline shared by the API routes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

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
    SucaiFindMatch,
    SucaiFindResponse,
    SucaiHit,
    TextLine,
    Timing,
)
from .sucai import SucaiStore
from .template import locate_instances, locate_template
from .verify import EXACT_NCC_SCORE, FeatureVerifier, VerifyResult, fuse_score

logger = logging.getLogger("yocr.pipeline")

# Cap the number of per-sucai instances that get geometric verification;
# the pyramid NMS already collapses the common case to one (or a few).
MAX_VERIFY_INSTANCES = 8


def _pick_box_and_scale(inst, result):
    """Choose the box/scale pair to report for one verified instance.

    Verified geometry wins except for pixel-exact NCC matches (score >= 0.98):
    there the NCC box is already accurate to the pixel, and learned-feature
    refinement would only add keypoint-level noise. The reported scale always
    comes from the NCC refine walk, which is more precise than the XFeat
    transform estimate.
    """
    if result.ok and inst.confidence < EXACT_NCC_SCORE and result.box:
        return result.box, inst.scale
    return inst.xyxy, inst.scale


def _resolve_target(elements: list[Element], *, text: str | None, label: str | None,
                    q: str | None, match_mode: str) -> tuple[bool, Element | None]:
    matched = best_match(elements, text=text, label=label, q=q, match_mode=match_mode)
    return matched is not None, matched


@dataclass
class AnalysisContext:
    settings: Settings
    registry: YOLORegistry
    sucai: SucaiStore | None = None
    _ocr: OCREngine | None = None

    @property
    def ocr(self) -> OCREngine:
        if self._ocr is None:
            self._ocr = get_ocr_engine(self.settings)
        return self._ocr


def make_context(settings: Settings) -> AnalysisContext:
    return AnalysisContext(
        settings=settings,
        registry=YOLORegistry(settings),
        sucai=SucaiStore(settings.sucai_dir),
    )


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


def find_sucai(
    store: SucaiStore,
    scene_bytes: bytes | None,
    scene_b64: str | None,
    *,
    threshold: float = 0.8,
    top_k: int = 0,
    all_instances: bool = False,
    match_verify: bool = True,
    xfeat_weights: Path | None = None,
) -> SucaiFindResponse:
    """Compare a scene screenshot against every registered sucai template.

    Each sucai picture is located in the scene with multi-scale NCC; results
    are sorted by score (best first). Items with at least one instance scoring
    >= threshold are flagged ``found`` and carry bounding box / clickable center.

    Args:
        store: Registered sucai library.
        scene_bytes: Raw scene image bytes (multipart upload).
        scene_b64: Scene image as base64.
        threshold: Minimum score to flag a sucai as found.
        top_k: Keep only the best N results (0 = keep all).
        all_instances: Report every occurrence of each sucai (NMS-deduped)
            instead of only its single best location.
        match_verify: Geometrically verify and refine NCC hits with XFeat
            features (rejects phantom matches, refines boxes).
        xfeat_weights: Path to XFeat weights; None uses the configured default.

    Returns:
        SucaiFindResponse: Per-sucai matches sorted by score descending.

    Raises:
        ValueError: Missing or corrupted scene image.
    """
    started = time.perf_counter()
    scene = load_image(scene_bytes, scene_b64)
    height, width = scene.shape[:2]

    verifier = FeatureVerifier(xfeat_weights) if match_verify else None
    verify_ready = verifier is not None and verifier.available

    records = store.iter_records()
    matches: list[SucaiFindMatch] = []
    for record in records:
        item_started = time.perf_counter()
        try:
            template = decode_image(store.read_image(record["id"]))
            if all_instances:
                # Every occurrence >= threshold (NMS-deduped); best keeps the
                # raw top match even below threshold for near-miss display.
                located = locate_instances(scene, template, threshold=threshold)
                instances, raw_best = located.instances, located.best
            else:
                # Single best instance; locate_instances still reports the raw
                # best when nothing crosses the threshold (near-miss display).
                located = locate_instances(
                    scene, template, threshold=threshold, max_instances=1
                )
                instances, raw_best = located.instances, located.best
        except (ValueError, OSError) as exc:
            # Corrupt picture, or meta.json entry whose PNG vanished from disk:
            # skip the item instead of failing the whole search.
            logger.warning("sucai '%s' unusable, skipped: %s", record["id"], exc)
            continue

        # Geometrically verify each candidate (cap to keep worst case bounded):
        # every hit gets a refined box when consistent inliers are found, and
        # its score is folded with the evidence so phantoms drop below gate.
        instances_for_verify = instances[:MAX_VERIFY_INSTANCES]
        verified: list[tuple[object, VerifyResult]] = []
        template_feats = None
        if verify_ready and instances_for_verify:
            template_feats = verifier.template_features(template)
        for inst in instances_for_verify:
            result = verifier.verify_hit(
                scene, template, template_feats, inst.xyxy, scale=inst.scale
            )
            verified.append((inst, result))
        verified.sort(
            key=lambda pair: fuse_score(pair[0].confidence, pair[1]), reverse=True
        )

        elapsed_ms = (time.perf_counter() - item_started) * 1000
        fused_list = []
        for inst, result in verified:
            # A pixel-exact NCC match is already conclusive; geometric
            # evidence cannot raise it, only add keypoint-level noise, so its
            # reported score stays the plain NCC value.
            if inst.confidence >= EXACT_NCC_SCORE:
                fused_list.append((inst, result, inst.confidence))
            else:
                fused_list.append((inst, result, fuse_score(inst.confidence, result)))
        found_instances = [t for t in fused_list if t[2] >= threshold]
        score = max((t[2] for t in fused_list), default=0.0)
        if score == 0.0 and raw_best is not None:
            # Nothing verifiable above zero (e.g. verification stripped every
            # candidate); fall back to the raw NCC best for near-miss display.
            score = float(raw_best.confidence)
        best = found_instances[0] if found_instances else (
            fused_list[0] if fused_list else None
        )
        found = bool(found_instances)
        if best is not None:
            inst, result, fused = best
            xyxy, scale = _pick_box_and_scale(inst, result)
            box = Box.from_xyxy(*xyxy)
        else:
            box = Box.from_xyxy(*raw_best.xyxy) if raw_best else None
            scale = raw_best.scale if raw_best else 1.0
        hits = []
        for inst, result, fused in fused_list:
            xyxy, hit_scale = _pick_box_and_scale(inst, result)
            hit_box = Box.from_xyxy(*xyxy)
            hits.append(SucaiHit(
                score=round(fused, 4),
                scale=round(hit_scale, 4),
                box=hit_box,
                center=hit_box.center,
            ))
        matches.append(SucaiFindMatch(
            id=record["id"],
            describe=record.get("describe", ""),
            category=record.get("category", ""),
            found=found,
            score=round(score, 4),
            scale=round(scale, 4),
            box=box,
            center=box.center if box else None,
            hits=hits,
            elapsed_ms=round(elapsed_ms, 1),
        ))

    matches.sort(key=lambda m: m.score, reverse=True)
    if top_k > 0:
        matches = matches[:top_k]
    total_ms = (time.perf_counter() - started) * 1000
    return SucaiFindResponse(
        image=ImageInfo(width=width, height=height),
        threshold=threshold,
        sucai_count=len(records),
        found_any=any(m.found for m in matches),
        results=matches,
        timing=Timing(total_ms=round(total_ms, 1)),
    )


__all__ = [
    "AnalysisContext",
    "make_context",
    "analyze",
    "detect_only",
    "ocr_only",
    "attach_texts",
    "match_template_images",
    "find_sucai",
]
