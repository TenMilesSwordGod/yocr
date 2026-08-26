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


class MatchTemplateResponse(BaseModel):
    """POST /match 响应：模板小图在场景大图中的定位结果。

    Attributes:
        found: 是否找到不低于阈值的匹配。
        score: 匹配置信度 [0, 1]；未找到为 0。
        threshold: 本次请求使用的判定阈值。
        scale: 命中时的模板缩放系数（跨 DPI 场景）。
        box: 命中区域外接框；未找到为 None。
        image: 场景图尺寸。
        template: 模板图尺寸。
        timing: 耗时统计。
    """

    found: bool = Field(default=False, description="是否命中阈值")
    score: float = Field(default=0.0, description="匹配置信度 0~1")
    threshold: float = Field(description="判定阈值")
    scale: float = Field(default=1.0, description="命中时模板缩放系数")
    box: Optional[Box] = Field(default=None, description="命中区域外接框")
    image: ImageInfo = Field(description="场景图尺寸")
    template: ImageInfo = Field(description="模板图尺寸")
    timing: Timing = Field(description="耗时统计")


class ModelInfo(BaseModel):
    name: str
    source: str
    loaded: bool
    classes: dict[str, str] = Field(default_factory=dict, description="id -> class name")
    error: Optional[str] = Field(default=None, description="最近一次加载失败原因；成功加载后为 null")


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]


class SucaiInfo(BaseModel):
    """已注册素材的元数据（图片本体经 /sucai/{id}/image 获取）。"""

    id: str = Field(description="素材唯一标识")
    describe: str = Field(default="", description="素材描述")
    width: int
    height: int
    size_bytes: int = Field(description="PNG 字节数")
    created_at: str = Field(description="注册时间 (UTC ISO8601)")
    updated_at: Optional[str] = Field(default=None, description="最近更新时间")
    image_url: str = Field(description="图片访问路径")

    @classmethod
    def from_record(cls, record: dict) -> "SucaiInfo":
        data = {k: v for k, v in record.items() if k in cls.model_fields}
        return cls(**data, image_url=f"/api/v1/sucai/{record['id']}/image")


class SucaiListResponse(BaseModel):
    total: int
    items: list[SucaiInfo]


class SucaiHit(BaseModel):
    """素材在场景图中的一个出现实例。"""

    score: float = Field(description="该实例的归一化相关度 0~1")
    scale: float = Field(default=1.0, description="命中时的模板缩放系数")
    box: Box = Field(description="该实例外接框")
    center: tuple[int, int] = Field(description="该实例中心点")


class SucaiFindMatch(BaseModel):
    """单个素材在场景图中的比对结果。"""

    id: str
    describe: str = ""
    found: bool = Field(description="是否有达到阈值的实例")
    score: float = Field(description="最佳实例（或最近似）的归一化相关度 0~1")
    scale: float = Field(default=1.0, description="最佳实例的模板缩放系数")
    box: Optional[Box] = Field(default=None, description="最佳实例外接框；无可比位置时为 null")
    center: Optional[tuple[int, int]] = Field(default=None, description="最佳实例中心点")
    hits: list[SucaiHit] = Field(
        default_factory=list,
        description="全部达到阈值的实例（all_instances=true 时可多于 1 个），按 score 降序",
    )
    elapsed_ms: float = Field(default=0.0, description="该素材比对耗时")


class SucaiFindResponse(BaseModel):
    """POST /sucai/find 响应：场景图与全部已注册素材逐一模板比对。"""

    image: ImageInfo = Field(description="场景图尺寸")
    threshold: float
    sucai_count: int = Field(description="参与比对的素材数量")
    found_any: bool = Field(default=False, description="是否有任一素材达到阈值")
    results: list[SucaiFindMatch] = Field(description="按 score 降序排列")
    timing: Timing
