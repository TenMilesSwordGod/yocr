"""REST API routes (pure vision: detection, OCR, merged analysis)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, Query, Request
from fastapi.responses import Response

from .pipeline import AnalysisContext, analyze, detect_only, find_sucai, match_template_images, ocr_only
from .schemas import (
    AnalyzeResponse,
    DetectResponse,
    MatchTemplateResponse,
    ModelsResponse,
    ModelInfo,
    OcrResponse,
    SucaiFindResponse,
    SucaiInfo,
    SucaiListResponse,
)
from .sucai import SucaiConflict, SucaiError, SucaiStore

logger = logging.getLogger("yocr.api")

router = APIRouter(prefix="/api/v1")


def get_ctx(request: Request) -> AnalysisContext:
    return request.app.state.ctx  # type: ignore[no-any-return]


def get_sucai_store(request: Request) -> SucaiStore:
    store = getattr(request.app.state.ctx, "sucai", None)
    if store is None:
        raise HTTPException(status_code=503, detail="sucai registry unavailable")
    return store


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
    sucai_count = ctx.sucai.count() if getattr(ctx, "sucai", None) else 0
    return {
        "status": "ok",
        "models": ctx.registry.names(),
        "default_model": ctx.registry.default_name(),
        "ocr_loaded": ctx.ocr.loaded,
        "device": ctx.settings.device,
        "sucai_count": sucai_count,
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


# --------------------------------------------------------------- sucai ----
@router.get("/sucai", response_model=SucaiListResponse, tags=["sucai"])
def list_sucai(request: Request):
    store = get_sucai_store(request)
    items = [SucaiInfo.from_record(r) for r in store.list()]
    return SucaiListResponse(total=len(items), items=items)


@router.post("/sucai", response_model=SucaiInfo, status_code=201, tags=["sucai"])
async def create_sucai(
    request: Request,
    file: bytes = File(description="素材图片"),
    describe: str = Form(default=""),
    id: str | None = Form(default=None),
):
    """注册素材：id 可省略（自动生成），describe 为可选描述。"""
    store = get_sucai_store(request)
    try:
        record = store.create(file, describe=describe, sid=(id or "").strip() or None)
    except SucaiConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SucaiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid picture: {exc}") from exc
    return SucaiInfo.from_record(record)


@router.get("/sucai/{sid}", response_model=SucaiInfo, tags=["sucai"])
def get_sucai(sid: str, request: Request):
    store = get_sucai_store(request)
    try:
        return SucaiInfo.from_record(store.get(sid))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/sucai/{sid}", response_model=SucaiInfo, tags=["sucai"])
async def update_sucai(
    sid: str,
    request: Request,
    file: bytes | None = File(default=None),
    describe: str | None = Form(default=None),
):
    """更新素材描述和/或替换图片。"""
    store = get_sucai_store(request)
    try:
        record = store.update(sid, describe=describe, image_bytes=file)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SucaiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid picture: {exc}") from exc
    return SucaiInfo.from_record(record)


@router.delete("/sucai/{sid}", tags=["sucai"])
def delete_sucai(sid: str, request: Request):
    store = get_sucai_store(request)
    try:
        store.delete(sid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": True, "id": sid}


@router.get("/sucai/{sid}/image", tags=["sucai"])
def get_sucai_image(sid: str, request: Request):
    store = get_sucai_store(request)
    try:
        data = store.read_image(sid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="sucai image missing on disk") from exc
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "no-cache"})


@router.post("/sucai/find", response_model=SucaiFindResponse, tags=["sucai"])
async def find_sucai_endpoint(
    request: Request,
    file: bytes | None = File(default=None),
    image_base64: str | None = Form(default=None),
    threshold: float = Query(default=0.8, ge=0.0, le=1.0, description="命中判定阈值"),
    top_k: int = Query(default=0, ge=0, description="只返回得分最高的 N 条；0 = 全部"),
):
    """在场景图(file/image_base64)中比对全部已注册素材并定位命中项。"""
    store = get_sucai_store(request)
    if not file and image_base64 is None and "multipart/" not in request.headers.get("content-type", "") \
            and "form" not in request.headers.get("content-type", ""):
        payload = await _json_body(request)
        image_base64 = payload.get("image_base64")
    if not file and not image_base64:
        raise HTTPException(status_code=400, detail="provide multipart 'file' or 'image_base64'")
    try:
        return find_sucai(store, file, image_base64, threshold=threshold, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
