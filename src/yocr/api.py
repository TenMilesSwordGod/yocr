"""REST API routes (pure vision: detection, OCR, merged analysis)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, Query, Request

from .pipeline import AnalysisContext, analyze, detect_only, match_template_images, ocr_only
from .schemas import (
    AnalyzeResponse,
    DetectResponse,
    MatchTemplateResponse,
    ModelsResponse,
    ModelInfo,
    OcrResponse,
)

logger = logging.getLogger("yocr.api")

router = APIRouter(prefix="/api/v1")


def get_ctx(request: Request) -> AnalysisContext:
    return request.app.state.ctx  # type: ignore[no-any-return]


async def _body_image(request: Request) -> tuple[bytes | None, str | None]:
    """Extract an image from the request when multipart parsing found nothing.

    Supports JSON bodies with `image_base64`/`image` fields and raw binary bodies.
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/" in content_type or "form" in content_type:
        return None, None
    body = await request.body()
    if not body:
        return None, None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body, None  # raw binary image
    b64 = payload.get("image_base64") or payload.get("image")
    if isinstance(b64, str) and b64:
        return None, b64
    raise HTTPException(status_code=400, detail="json body must contain 'image_base64'")


# ---------------------------------------------------------------- health ---
@router.get("/healthz", tags=["system"])
def healthz(request: Request):
    ctx = get_ctx(request)
    return {
        "status": "ok",
        "models": ctx.registry.names(),
        "default_model": ctx.registry.default_name(),
        "ocr_loaded": ctx.ocr.loaded,
        "device": ctx.settings.device,
    }


@router.get("/models", response_model=ModelsResponse, tags=["system"])
def list_models(request: Request):
    ctx = get_ctx(request)
    infos = []
    for name in ctx.registry.names():
        spec = ctx.registry.spec(name)
        loaded = spec.name in ctx.registry._models  # noqa: SLF001 - read-only peek
        classes: dict[str, str] = {}
        if loaded:
            classes = {str(k): v for k, v in ctx.registry._classes.get(spec.name, {}).items()}  # noqa: SLF001
        infos.append(ModelInfo(
            name=spec.name,
            source=spec.source,
            loaded=loaded,
            classes=classes,
            error=ctx.registry.last_error(spec.name),
        ))
    return ModelsResponse(default_model=ctx.registry.default_name(), models=infos)


@router.post("/match", response_model=MatchTemplateResponse, tags=["vision"])
async def match_endpoint(
    request: Request,
    file: bytes | None = File(default=None),
    template: bytes | None = File(default=None),
    image_base64: str | None = Form(default=None),
    template_base64: str | None = Form(default=None),
    threshold: float = Query(default=0.8, ge=0.0, le=1.0, description="匹配置信度阈值"),
):
    """在场景图(file/image_base64)中定位模板图(template/template_base64)。"""
    ctx = get_ctx(request)
    if "multipart/" not in request.headers.get("content-type", "") and "form" not in request.headers.get("content-type", ""):
        payload = await _json_body(request)
        image_base64 = payload.get("image_base64")
        template_base64 = payload.get("template_base64")
    try:
        return match_template_images(
            file, image_base64, template, template_base64, threshold=threshold
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _json_body(request: Request) -> dict:
    """Parse a JSON request body into a dict.

    Args:
        request: Incoming FastAPI request.

    Returns:
        dict: Parsed body; empty dict when the body is not JSON.
    """
    import json

    body = await request.body()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# --------------------------------------------------------------- vision ----
@router.post("/detect", response_model=DetectResponse, tags=["vision"])
async def detect_endpoint(
    request: Request,
    file: bytes | None = File(default=None),
    image_base64: str | None = Form(default=None),
    model: str | None = Query(default=None),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None),
    text: str | None = Query(default=None, description="按元素 OCR 文本查找，响应带 found/matched"),
    label: str | None = Query(default=None, description="按 YOLO 类别名查找"),
    q: str | None = Query(default=None, description="泛搜索：文本或类别任一命中"),
    match_mode: str = Query(default="contains", pattern="^(contains|exact|fuzzy)$"),
):
    ctx = get_ctx(request)
    if not file and image_base64 is None:
        file, image_base64 = await _body_image(request)
    if not file and not image_base64:
        raise HTTPException(status_code=400, detail="provide multipart 'file' or 'image_base64'")
    try:
        return detect_only(ctx, file, image_base64, model=model, conf=conf, iou=iou, imgsz=imgsz,
                           text=text, label=label, q=q, match_mode=match_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ocr", response_model=OcrResponse, tags=["vision"])
async def ocr_endpoint(
    request: Request,
    file: bytes | None = File(default=None),
    image_base64: str | None = Form(default=None),
):
    ctx = get_ctx(request)
    if not file and image_base64 is None:
        file, image_base64 = await _body_image(request)
    if not file and not image_base64:
        raise HTTPException(status_code=400, detail="provide multipart 'file' or 'image_base64'")
    try:
        return ocr_only(ctx, file, image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/analyze", response_model=AnalyzeResponse, tags=["vision"])
async def analyze_endpoint(
    request: Request,
    file: bytes | None = File(default=None),
    image_base64: str | None = Form(default=None),
    model: str | None = Query(default=None),
    conf: float | None = Query(default=None, ge=0.0, le=1.0),
    iou: float | None = Query(default=None, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None),
    with_ocr: bool = Query(default=True),
    text: str | None = Query(default=None, description="按元素 OCR 文本查找，响应带 found/matched"),
    label: str | None = Query(default=None, description="按 YOLO 类别名查找"),
    q: str | None = Query(default=None, description="泛搜索：文本或类别任一命中"),
    match_mode: str = Query(default="contains", pattern="^(contains|exact|fuzzy)$"),
):
    ctx = get_ctx(request)
    if not file and image_base64 is None:
        file, image_base64 = await _body_image(request)
    if not file and not image_base64:
        raise HTTPException(status_code=400, detail="provide multipart 'file' or 'image_base64'")
    try:
        _, response = analyze(
            ctx, file, image_base64,
            model=model, conf=conf, iou=iou, imgsz=imgsz, with_ocr=with_ocr,
            text=text, label=label, q=q, match_mode=match_mode,
        )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
