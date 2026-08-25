"""Pydantic schemas shared by the REST API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Box(BaseModel):
    """Pixel-space bounding box plus derived geometry."""

    xyxy: tuple[int, int, int, int] = Field(description="x1, y1, x2, y2")
    xywh: tuple[int, int, int, int]
    center: tuple[int, int]

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "Box":
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        return cls(xyxy=(x1i, y1i, x2i, y2i), xywh=(x1i, y1i, x2i - x1i, y2i - y1i), center=((x1i + x2i) // 2, (y1i + y2i) // 2))


class ImageInfo(BaseModel):
    width: int
    height: int


class Element(BaseModel):
    id: int
    label: str = Field(description="YOLO class name")
    class_id: int
    confidence: float
    box: Box
    text: Optional[str] = None
    text_confidence: Optional[float] = None


class TextLine(BaseModel):
    text: str
    confidence: float
    box: Box


class Timing(BaseModel):
    total_ms: float
    detect_ms: Optional[float] = None
    ocr_ms: Optional[float] = None


class DetectResponse(BaseModel):
    model: str
    image: ImageInfo
    elements: list[Element]
    found: bool = Field(default=False, description="是否命中查询目标(text/label/q 参数)")
    matched: Optional[Element] = Field(default=None, description="命中的最优元素")
    timing: Timing


class OcrResponse(BaseModel):
    image: ImageInfo
    lines: list[TextLine]
    full_text: str
    timing: Timing


class AnalyzeResponse(BaseModel):
    model: str
    image: ImageInfo
    elements: list[Element]
    found: bool = Field(default=False, description="是否命中查询目标(text/label/q 参数)")
    matched: Optional[Element] = Field(default=None, description="命中的最优元素")
    lines: list[TextLine]
    full_text: str
    timing: Timing


class ModelInfo(BaseModel):
    name: str
    source: str
    loaded: bool
    classes: dict[str, str] = Field(default_factory=dict, description="id -> class name")
    error: Optional[str] = Field(default=None, description="最近一次加载失败原因；成功加载后为 null")


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]
