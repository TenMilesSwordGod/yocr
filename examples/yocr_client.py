"""yocr 客户端工具类 —— 供自动化测试框架直接复用。

基于 httpx + pydantic：所有接口返回强类型模型（带 box.center 等便捷属性），
不再依赖 requests。

用法见 docs/tutorial.zh-CN.md

依赖: pip install httpx pydantic
"""
from __future__ import annotations

import difflib  # 模糊匹配使用的序列相似度算法
import time  # wait_for 轮询计时
from typing import Callable, Iterable, TypeVar

import httpx  # HTTP 传输层（替代 requests）
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)  # _post 泛型：响应反序列化的目标模型类型


# ----------------------------------------------------------------- 模型 --
class Box(BaseModel):
    """像素坐标系下的矩形框及派生几何。

    Attributes:
        xyxy: 左上/右下角坐标 (x1, y1, x2, y2)。
        xywh: 左上角坐标加宽高 (x, y, w, h)。
        center: 框中心点像素坐标 (cx, cy)，点击时最常用。
        area: 框面积（平方像素）。
    """

    model_config = ConfigDict(frozen=True)  # 不可变，可安全作为 dict key / 缓存

    xyxy: tuple[int, int, int, int] = Field(description="x1, y1, x2, y2")
    xywh: tuple[int, int, int, int] = Field(description="x, y, w, h")
    center: tuple[int, int] = Field(description="cx, cy")

    @property
    def area(self) -> int:
        """int: 框的像素面积。"""
        w, h = self.xywh[2], self.xywh[3]  # 宽与高分别取自 xywh 后两位
        return w * h


class ImageInfo(BaseModel):
    """服务端实际解析出的图像尺寸。

    Attributes:
        width: 图像宽度（像素）。
        height: 图像高度（像素）。
    """

    width: int = Field(description="图像宽度 px")
    height: int = Field(description="图像高度 px")


class Element(BaseModel):
    """检测出的单个 UI 元素（可附带归属的 OCR 文本）。

    Attributes:
        id: 服务端按置信度降序分配的序号。
        label: YOLO 类别名，如 "Button"/"Text"/"App Icon"。
        class_id: YOLO 类别数字 id。
        confidence: 检测置信度 [0, 1]。
        box: 元素外接矩形。
        text: 归属到该元素的 OCR 文本；无文本时为 None。
        text_confidence: text 对应的识别置信度；无文本时为 None。
    """

    id: int = Field(description="按置信度降序的元素序号")
    label: str = Field(description="YOLO 类别名")
    class_id: int = Field(description="YOLO 类别 id")
    confidence: float = Field(description="检测置信度 0~1")
    box: Box = Field(description="外接矩形")
    text: str | None = Field(default=None, description="归属的 OCR 文本")
    text_confidence: float | None = Field(default=None, description="OCR 置信度")


class TextLine(BaseModel):
    """OCR 识别出的一行文本及其位置。

    Attributes:
        text: 识别出的文字内容。
        confidence: 识别置信度 [0, 1]。
        box: 文本行外接矩形。
    """

    text: str = Field(description="识别文本")
    confidence: float = Field(description="置信度 0~1")
    box: Box = Field(description="文本行外接矩形")


class Timing(BaseModel):
    """各阶段耗时统计（毫秒）。

    Attributes:
        total_ms: 请求总耗时。
        detect_ms: YOLO 检测耗时；纯 OCR 请求为 None。
        ocr_ms: PaddleOCR 耗时；with_ocr=False 时为 None。
    """

    total_ms: float = Field(description="总耗时 ms")
    detect_ms: float | None = Field(default=None, description="检测耗时 ms")
    ocr_ms: float | None = Field(default=None, description="OCR 耗时 ms")


class DetectResponse(BaseModel):
    """POST /detect 响应：仅元素检测。

    Attributes:
        model: 实际使用的模型名（可能因回退而不同于请求值）。
        image: 图像尺寸。
        elements: 检出元素，按置信度降序。
        found: 是否命中查询目标（text/label/q）。
        matched: 命中的最优元素；未命中为 None。
        timing: 耗时统计。
    """

    model: str = Field(description="实际使用的模型名")
    image: ImageInfo = Field(description="图像尺寸")
    elements: list[Element] = Field(default_factory=list, description="检出元素列表")
    found: bool = Field(default=False, description="是否命中查询目标")
    matched: Element | None = Field(default=None, description="命中的最优元素")
    timing: Timing = Field(description="耗时统计")


class OcrResponse(BaseModel):
    """POST /ocr 响应：仅文字识别。

    Attributes:
        image: 图像尺寸。
        lines: 全部识别文本行。
        full_text: 所有行按序拼接的多行文本。
        timing: 耗时统计。
    """

    image: ImageInfo = Field(description="图像尺寸")
    lines: list[TextLine] = Field(default_factory=list, description="识别文本行")
    full_text: str = Field(default="", description="整页拼接文本")
    timing: Timing = Field(description="耗时统计")


class AnalyzeResponse(BaseModel):
    """POST /analyze 响应：检测 + OCR 融合（推荐入口）。

    Attributes:
        model: 实际使用的模型名（可能因回退而不同于请求值）。
        image: 图像尺寸。
        elements: 检出元素，text 字段已填充归属 OCR 文本。
        found: 是否命中查询目标（text/label/q）。
        matched: 命中的最优元素；未命中为 None。
        lines: 全部 OCR 文本行。
        full_text: 整页拼接文本。
        timing: 耗时统计。
    """

    model: str = Field(description="实际使用的模型名")
    image: ImageInfo = Field(description="图像尺寸")
    elements: list[Element] = Field(default_factory=list, description="检出元素列表")
    found: bool = Field(default=False, description="是否命中查询目标")
    matched: Element | None = Field(default=None, description="命中的最优元素")
    lines: list[TextLine] = Field(default_factory=list, description="OCR 文本行")
    full_text: str = Field(default="", description="整页拼接文本")
    timing: Timing = Field(description="耗时统计")


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
    box: Box | None = Field(default=None, description="命中区域外接框")
    image: ImageInfo = Field(description="场景图尺寸")
    template: ImageInfo = Field(description="模板图尺寸")
    timing: Timing = Field(description="耗时统计")


class ModelInfo(BaseModel):
    """注册表中单个模型的详情。

    Attributes:
        name: 模型显示名（请求 ?model= 参数使用小写别名亦可）。
        source: 权重来源（本地路径或 HF 仓库 id）。
        loaded: 权重当前是否已加载进内存。
        classes: 已加载模型的类别表 {类别id: 名称}；未加载为空表。
        error: 最近一次加载失败原因；加载成功或尚未尝试时为 None。
    """

    name: str = Field(description="模型名")
    source: str = Field(description="权重来源")
    loaded: bool = Field(description="是否已加载")
    classes: dict[str, str] = Field(default_factory=dict, description="id -> 类别名")
    error: str | None = Field(default=None, description="最近一次加载失败原因")


class ModelsResponse(BaseModel):
    """GET /models 响应。

    Attributes:
        default_model: 未指定 ?model= 时使用的默认模型名。
        models: 全部已注册模型详情。
    """

    default_model: str = Field(description="默认模型名")
    models: list[ModelInfo] = Field(default_factory=list, description="模型列表")


class HealthzResponse(BaseModel):
    """GET /healthz 响应。

    Attributes:
        status: 固定 "ok" 表示进程存活（不代表模型可用，请看 models 接口）。
        models: 已注册模型名列表。
        default_model: 默认模型名。
        ocr_loaded: PaddleOCR 引擎是否就绪。
        device: 推理设备 cpu/cuda:0/mps。
    """

    status: str = Field(description="健康状态字面量 ok")
    models: list[str] = Field(default_factory=list, description="注册模型名")
    default_model: str = Field(description="默认模型名")
    ocr_loaded: bool = Field(description="OCR 引擎是否就绪")
    device: str = Field(description="推理设备")


# ---------------------------------------------------------------- 客户端 --
class YocrClient:
    """yocr 服务的 HTTP 客户端封装（httpx + pydantic 强类型）。

    Attributes:
        base_url: API 根地址，如 http://127.0.0.1:8000/api/v1。
        timeout: 单次请求超时秒数。
        session: 复用的 httpx.Client 连接池。
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000/api/v1", timeout: float = 60.0):
        """初始化客户端并建立连接池。

        Args:
            base_url: API 根地址，末尾多余的 "/" 会被去除。
            timeout: 单次 HTTP 请求超时（秒）；首次调用含模型冷启动加载，
                建议保持 60s 以上。
        """
        self.base_url = base_url.rstrip("/")  # 统一去掉尾部斜杠便于拼接
        self.timeout = timeout  # 所有请求共用的超时秒数
        self.session = httpx.Client(trust_env=False)  # 忽略系统代理直连本机；走代理时改为 True

    # ------------------------------------------------------------- 基础 --
    def healthz(self) -> HealthzResponse:
        """探活并获取服务概览。

        Returns:
            HealthzResponse: 进程状态、注册模型、OCR 就绪情况与推理设备。
        """
        r = self.session.get(f"{self.base_url}/healthz", timeout=self.timeout)  # 无副作用的探活 GET
        r.raise_for_status()
        return HealthzResponse.model_validate(r.json())

    def models(self) -> ModelsResponse:
        """列出全部模型及加载状态。

        Returns:
            ModelsResponse: 各模型的 source/loaded/classes/error 详情；
            error 非空说明该模型最近一次加载失败。
        """
        r = self.session.get(f"{self.base_url}/models", timeout=self.timeout)  # 只读的模型清单 GET
        r.raise_for_status()
        return ModelsResponse.model_validate(r.json())

    # ------------------------------------------------------------- 识别 --
    def detect(self, png: bytes, *, model: str | None = None, conf: float | None = None,
               iou: float | None = None, imgsz: int | None = None) -> DetectResponse:
        """仅元素检测（快，无 OCR）。

        Args:
            png: PNG/JPEG 截图的原始字节。
            model: 模型别名；None 用默认模型（缺权重会自动回退其他模型）。
            conf: 置信度阈值覆盖，范围 [0, 1]。
            iou: NMS IoU 阈值覆盖，范围 [0, 1]。
            imgsz: 推理分辨率（正方形边长），越大越准越慢。

        Returns:
            DetectResponse: 元素列表与耗时；传 text/label/q 时含命中信息。

        Raises:
            httpx.HTTPStatusError: 4xx/5xx（400 参数错误、404 模型不可用等）。
        """
        params = _drop_none(model=model, conf=conf, iou=iou, imgsz=imgsz)  # None 参数不发送
        return self._post("/detect", png, DetectResponse, params=params)

    def ocr(self, png: bytes) -> OcrResponse:
        """仅文字识别。

        Args:
            png: PNG/JPEG 截图的原始字节。

        Returns:
            OcrResponse: 文本行、整页拼接文本与耗时。

        Raises:
            httpx.HTTPStatusError: 400 图片无效或 503 OCR 引擎不可用。
        """
        return self._post("/ocr", png, OcrResponse)

    def analyze(self, png: bytes, *, model: str | None = None, conf: float | None = None,
                iou: float | None = None, imgsz: int | None = None,
                with_ocr: bool = True,
                text: str | None = None, label: str | None = None,
                q: str | None = None, match_mode: str | None = None) -> AnalyzeResponse:
        """检测 + OCR 融合（推荐）：元素的 text 字段即控件上的文字。

        Args:
            png: PNG/JPEG 截图的原始字节。
            model: 模型别名；None 用默认模型（缺权重会自动回退）。
            conf: 置信度阈值覆盖，范围 [0, 1]。
            iou: NMS IoU 阈值覆盖，范围 [0, 1]。
            imgsz: 推理分辨率（正方形边长）。
            with_ocr: 是否执行 OCR 并把文本归属到元素；关掉更快。
            text: 按 OCR 文本查找目标，命中时响应带 found/matched。
            label: 按 YOLO 类别名查找目标，如 "Button"。
            q: 泛搜索：文本或类别任一命中即算。
            match_mode: 文本匹配方式 contains/exact/fuzzy。

        Returns:
            AnalyzeResponse: 元素(含文本)、OCR 行、命中结果与耗时。

        Raises:
            httpx.HTTPStatusError: 4xx/5xx。
        """
        params = _drop_none(  # 组装 query 参数，None 项剔除
            model=model, conf=conf, iou=iou, imgsz=imgsz, with_ocr=with_ocr,
            text=text, label=label, q=q, match_mode=match_mode)
        return self._post("/analyze", png, AnalyzeResponse, params=params)

    def locate(self, png: bytes, *, text: str | None = None, label: str | None = None,
               q: str | None = None, match_mode: str | None = None,
               **kwargs) -> Element | None:
        """一步判定目标是否在屏。

        Args:
            png: PNG/JPEG 截图的原始字节。
            text: 目标控件文字。
            label: 目标 YOLO 类别名。
            q: 泛搜索关键词。
            match_mode: 文本匹配方式 contains/exact/fuzzy。
            **kwargs: 其余参数透传给 analyze()（model/conf/imgsz 等）。

        Returns:
            Element | None: 命中的元素（box.center 可直接点击）；未命中 None。
        """
        result = self.analyze(png, text=text, label=label, q=q,  # 服务端完成匹配打分
                              match_mode=match_mode, **kwargs)
        return result.matched if result.found else None

    def _post(self, path: str, png: bytes, model_type: type[T],
              params: dict | None = None) -> T:
        """上传截图 POST 到指定端点并反序列化为强类型模型。

        Args:
            path: API 路径，如 "/analyze"。
            png: 截图原始字节，以 multipart 字段 file 上传。
            model_type: 响应 JSON 的目标 pydantic 模型类。
            params: 追加的 query 参数；None 视为空。

        Returns:
            T: 校验后的 pydantic 响应实例。

        Raises:
            httpx.HTTPError: 网络/超时/HTTP 状态错误。
            pydantic.ValidationError: 响应与服务端 schema 不符。
        """
        r = self.session.post(  # multipart 上传 + query 参数一次发出
            f"{self.base_url}{path}",
            params=params or {},
            files={"file": ("screen.png", png, "image/png")},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return model_type.model_validate(r.json())  # dict -> 强类型模型

    # ------------------------------------------------------- 模板定位 --
    def match(self, template: bytes, scene: bytes, *, threshold: float = 0.8) -> MatchTemplateResponse:
        """在场景截图中定位模板小图（第一张图在第二张图中的位置）。

        Args:
            template: 模板小图的原始字节（要找的控件/图标截图）。
            scene: 场景大图的原始字节（通常是整屏截图）。
            threshold: 匹配置信度阈值 [0, 1]，低于该值视为未找到。

        Returns:
            MatchTemplateResponse: found/score/box/scale 与两张图尺寸。

        Raises:
            httpx.HTTPError: 网络/超时/HTTP 状态错误。
            httpx.HTTPStatusError: 400 图片无效或缺失。
        """
        r = self.session.post(  # 双 multipart 字段一次上传
            f"{self.base_url}/match",
            params={"threshold": threshold},
            files={
                "file": ("scene.png", scene, "image/png"),
                "template": ("template.png", template, "image/png"),
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return MatchTemplateResponse.model_validate(r.json())

    def locate_template(self, template: bytes, scene: bytes, *,
                        threshold: float = 0.8) -> tuple[int, int] | None:
        """模板匹配一步拿到可点击中心坐标。

        Args:
            template: 模板小图原始字节。
            scene: 场景大图原始字节。
            threshold: 匹配置信度阈值。

        Returns:
            tuple[int, int] | None: 命中区域中心 (cx, cy)；未找到 None。
        """
        result = self.match(template, scene, threshold=threshold)
        return self.center_box(result.box) if result.found and result.box else None

    @staticmethod
    def center_box(box: Box | None) -> tuple[int, int] | None:
        """取任意 Box 的中心坐标。

        Args:
            box: 目标矩形；None 直接透传。

        Returns:
            tuple[int, int] | None: (cx, cy)；入参为 None 时返回 None。
        """
        if box is None:
            return None
        cx, cy = box.center
        return int(cx), int(cy)

    # ------------------------------------------------------------- 定位 --
    @staticmethod
    def find(result: DetectResponse | AnalyzeResponse, *, text: str | None = None,
             label: str | None = None, match_mode: str = "contains",
             region: tuple[int, int, int, int] | None = None,
             index: int = 0) -> Element | None:
        """在 detect/analyze 的结果里定位元素。

        Args:
            result: detect()/analyze() 返回的响应对象。
            text: 控件文字（contains/exact/fuzzy），大小写无关、忽略空白。
            label: YOLO 类别名，如 "Button"/"App Icon"/"Alert"。
            match_mode: 文本匹配方式，仅 text 生效。
            region: (x1,y1,x2,y2)；元素中心必须落在该区域内——弹窗/半屏过滤。
            index: 多个命中时取第 N 个（按综合得分降序）。

        Returns:
            Element | None: 第 index 个命中元素；不足时 None。
        """
        hits = _match_elements(result.elements, text, label, match_mode, region)  # 过滤+打分排序
        return hits[index] if index < len(hits) else None

    @staticmethod
    def find_all(result: DetectResponse | AnalyzeResponse, *,
                 text: str | None = None, label: str | None = None,
                 match_mode: str = "contains",
                 region: tuple[int, int, int, int] | None = None) -> list[Element]:
        """同 find()，返回全部命中（按综合得分降序）。

        Args:
            result: detect()/analyze() 返回的响应对象。
            text: 控件文字过滤条件；与 label 至少给一个。
            label: YOLO 类别名过滤条件。
            match_mode: 文本匹配方式 contains/exact/fuzzy。
            region: 限定元素中心的区域 (x1,y1,x2,y2)。

        Returns:
            list[Element]: 全部命中元素；无命中返回空列表。
        """
        return _match_elements(result.elements, text, label, match_mode, region)  # 直接复用打分器

    @staticmethod
    def center(element: Element) -> tuple[int, int]:
        """取元素中心点像素坐标（可直接喂给 adb/uiautomator 点击）。

        Args:
            element: locate()/find() 返回的元素。

        Returns:
            tuple[int, int]: (cx, cy)。
        """
        cx, cy = element.box.center  # pydantic 已还原为 int 元组
        return int(cx), int(cy)

    # ------------------------------------------------- 场景快捷方法 --
    def find_by_text(self, png: bytes, text: str, **kwargs) -> tuple[int, int] | None:
        """按文字一步拿到可点击坐标。

        Args:
            png: 截图原始字节。
            text: 目标控件文字。
            **kwargs: 其余参数透传给 locate()（region/match_mode/model 等）。

        Returns:
            tuple[int, int] | None: (cx, cy)；未找到 None。
        """
        el = self.locate(png, text=text, **kwargs)  # 先判定是否在屏
        return self.center(el) if el else None

    def find_icon(self, png: bytes, labels: Iterable[str] = ("App Icon", "Utility Button", "Button"),
                  region: tuple[int, int, int, int] | None = None, **kwargs) -> tuple[int, int] | None:
        """纯图标（齿轮/返回箭头等无文字控件）：依次尝试多个类别名。

        Args:
            png: 截图原始字节。
            labels: 依次尝试的候选类别名。
            region: 限定搜索区域 (x1,y1,x2,y2)。
            **kwargs: 其余参数透传给 analyze()。

        Returns:
            tuple[int, int] | None: 第一个命中的图标中心；全部落空 None。
        """
        result = self.analyze(png, **kwargs)  # 一次识别供多个类别复用
        for label in labels:  # 按调用方给出的优先级逐个尝试
            el = self.find(result, label=label, region=region)
            if el:
                return self.center(el)
        return None

    def find_dialog_button(self, png: bytes, text: str | None = None,
                           label: str | None = None, **kwargs) -> tuple[int, int] | None:
        """弹窗按钮定位：先找弹窗容器（Alert/Window 等）把搜索限制在容器内。

        容器没被检出时退化为全屏查找。

        Args:
            png: 截图原始字节。
            text: 按钮文字。
            label: 按钮类别名（无文字按钮用这个）。
            **kwargs: 其余参数透传给 analyze()。

        Returns:
            tuple[int, int] | None: 按钮中心坐标；未找到 None。
        """
        result = self.analyze(png, **kwargs)  # 完整识别一次
        elements = result.elements  # 参与容器筛选的元素集合
        container = next((e for e in elements if e.label in  # 第一个弹窗类容器
                          ("Alert", "Window", "PopUp Menu", "ContextMenu")), None)
        region = tuple(container.box.xyxy) if container else None  # 容器矩形作为过滤区域
        el = self.find(result, text=text, label=label, region=region)
        return self.center(el) if el else None

    def wait_for(self, screenshot: Callable[[], bytes], *, text: str | None = None,
                 label: str | None = None, timeout: float = 15.0, interval: float = 1.0,
                 **kwargs) -> tuple[int, int] | None:
        """轮询等待某元素出现。

        轮询期间单次截屏/识别失败视为"还没出现"，继续等待直至超时。

        Args:
            screenshot: 返回最新截图字节的可调用对象（如 adb screencap 封装）。
            text: 等待出现的控件文字。
            label: 等待出现的类别名（无文字控件用这个）。
            timeout: 总超时秒数。
            interval: 轮询间隔秒数。
            **kwargs: 其余参数透传给 find_by_text/find_icon。

        Returns:
            tuple[int, int] | None: 出现则返回中心坐标；超时 None。
        """
        deadline = time.time() + timeout  # 以墙钟计算的截止时刻
        while time.time() < deadline:
            try:
                center = self.find_by_text(screenshot(), text=text, **kwargs) \
                    if text else self.find_icon(screenshot(), labels=[label] if label else
                                                ("App Icon", "Utility Button", "Button"), **kwargs)
            except httpx.HTTPError:
                center = None  # 瞬时失败（黑帧/服务繁忙）不算超时
            if center:
                return center
            time.sleep(interval)
        return None


# ------------------------------------------------------------------ 内部 --
def _norm(s: str) -> str:
    """归一化字符串用于宽松比较。

    Args:
        s: 原始字符串。

    Returns:
        str: 去除全部空白并转小写后的字符串。
    """
    return "".join(s.split()).lower()


def _text_hit(query: str, candidate: str, mode: str) -> float:
    """计算查询词对候选文本的单向匹配得分。

    Args:
        query: 用户输入的查询词。
        candidate: 元素上的 OCR 文本。
        mode: 匹配方式 contains/exact/fuzzy。

    Returns:
        float: 得分 0~1；0 表示不匹配。
    """
    q, c = _norm(query), _norm(candidate or "")  # 双方都归一化后再比较
    if not q or not c:
        return 0.0
    if mode == "exact":
        return 1.0 if q == c else 0.0
    if mode == "fuzzy":  # 序列相似度优先，其次包含关系兜底
        ratio = difflib.SequenceMatcher(None, q, c).ratio()
        return ratio if ratio >= 0.75 else (0.9 if q in c else 0.0)
    return 1.0 if q in c else 0.0


def _match_elements(elements: list[Element], text: str | None, label: str | None,
                    match_mode: str, region: tuple[int, int, int, int] | None) -> list[Element]:
    """过滤并按综合得分排序候选元素。

    Args:
        elements: 待筛选的元素列表。
        text: 文字过滤条件；优先于 label 生效。
        label: 类别名过滤条件（包含式比较）。
        match_mode: 文本匹配方式 contains/exact/fuzzy。
        region: 限定元素中心的区域 (x1,y1,x2,y2)；None 不过滤。

    Returns:
        list[Element]: 命中元素，按 (匹配分, 置信度, 小面积偏好) 降序。
    """
    scored: list[tuple[float, Element]] = []  # (得分, 元素) 待排序收集器
    for e in elements:
        if text:
            score = _text_hit(text, e.text or "", match_mode)
            if score == 0.0:
                continue
        elif label:
            score = 1.0 if _norm(label) in _norm(e.label) else 0.0
            if score == 0.0:
                continue
        else:
            continue
        cx, cy = e.box.center  # 区域过滤以元素中心为准
        if region and not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
            continue
        area = e.box.area  # 面积参与打分：同等条件下偏向更小的元素（更精准）
        scored.append((score * 10 + e.confidence + min(area / 1e6, 0.5), e))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored]


def _drop_none(**kwargs) -> dict:
    """剔除值为 None 的关键字参数。

    Args:
        kwargs: 任意键值对。

    Returns:
        dict: 仅保留非 None 项。
    """
    return {k: v for k, v in kwargs.items() if v is not None}


__all__ = [
    "AnalyzeResponse",
    "Box",
    "DetectResponse",
    "Element",
    "HealthzResponse",
    "ImageInfo",
    "MatchTemplateResponse",
    "ModelInfo",
    "ModelsResponse",
    "OcrResponse",
    "TextLine",
    "Timing",
    "YocrClient",
]
