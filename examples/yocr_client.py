"""yocr 客户端工具类 —— 供自动化测试框架直接复用。

用法见 docs/tutorial.zh-CN.md

依赖: pip install requests
"""
from __future__ import annotations

import difflib
import time
from typing import Callable, Iterable

import requests


class YocrClient:
    """yocr 服务的 HTTP 客户端封装。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/api/v1", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False  # 本机直连，忽略系统代理；走代理时改为 True

    # ------------------------------------------------------------- 基础 --
    def healthz(self) -> dict:
        return self.session.get(f"{self.base_url}/healthz", timeout=self.timeout).json()

    def models(self) -> dict:
        return self.session.get(f"{self.base_url}/models", timeout=self.timeout).json()

    # ------------------------------------------------------------- 识别 --
    def detect(self, png: bytes, *, model: str | None = None, conf: float | None = None,
               iou: float | None = None, imgsz: int | None = None) -> dict:
        """仅元素检测（快，无 OCR）。"""
        return self._post("/detect", png, params=_drop_none(
            model=model, conf=conf, iou=iou, imgsz=imgsz))

    def ocr(self, png: bytes) -> dict:
        """仅文字识别。"""
        return self._post("/ocr", png)

    def analyze(self, png: bytes, *, model: str | None = None, conf: float | None = None,
                iou: float | None = None, imgsz: int | None = None,
                with_ocr: bool = True,
                text: str | None = None, label: str | None = None,
                q: str | None = None, match_mode: str | None = None) -> dict:
        """检测 + OCR 融合（推荐）：元素的 text 字段即控件上的文字。

        传 text/label/q 时响应带 found: bool 与 matched 元素。
        """
        return self._post("/analyze", png, params=_drop_none(
            model=model, conf=conf, iou=iou, imgsz=imgsz, with_ocr=with_ocr,
            text=text, label=label, q=q, match_mode=match_mode))

    def locate(self, png: bytes, *, text: str | None = None, label: str | None = None,
               q: str | None = None, match_mode: str | None = None,
               **kwargs) -> dict | None:
        """一步判定目标是否在屏：命中返回元素 dict(含 box.center)，否则 None。"""
        result = self.analyze(png, text=text, label=label, q=q,
                              match_mode=match_mode, **kwargs)
        return result["matched"] if result.get("found") else None

    def _post(self, path: str, png: bytes, params: dict | None = None) -> dict:
        r = self.session.post(
            f"{self.base_url}{path}",
            params=params or {},
            files={"file": ("screen.png", png, "image/png")},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------- 定位 --
    @staticmethod
    def find(result: dict, *, text: str | None = None, label: str | None = None,
             match_mode: str = "contains", region: tuple[int, int, int, int] | None = None,
             index: int = 0) -> dict | None:
        """在 analyze/detect 的结果里定位元素。

        text:   控件文字（contains/exact/fuzzy），大小写无关、忽略空白
        label:  YOLO 类别名，如 "Button"/"App Icon"/"Alert"
        region: (x1,y1,x2,y2)，元素中心必须落在该区域内 —— 弹窗/半屏过滤神器
        index:  多个命中时取第 N 个（按置信度降序）
        """
        hits = _match_elements(result.get("elements", []), text, label, match_mode, region)
        return hits[index] if index < len(hits) else None

    @staticmethod
    def find_all(result: dict, **kwargs) -> list[dict]:
        """同 find()，返回全部命中（按置信度降序）。"""
        return _match_elements(
            result.get("elements", []),
            kwargs.get("text"), kwargs.get("label"),
            kwargs.get("match_mode", "contains"), kwargs.get("region"),
        )

    @staticmethod
    def center(element: dict) -> tuple[int, int]:
        c = element["box"]["center"]
        return int(c[0]), int(c[1])

    # ------------------------------------------------- 场景快捷方法 --
    def find_by_text(self, png: bytes, text: str, **kwargs) -> tuple[int, int] | None:
        el = self.locate(png, text=text, **kwargs)
        return self.center(el) if el else None

    def find_icon(self, png: bytes, labels: Iterable[str] = ("App Icon", "Utility Button", "Button"),
                  region: tuple[int, int, int, int] | None = None, **kwargs) -> tuple[int, int] | None:
        """纯图标（齿轮/返回箭头等无文字控件）：依次尝试多个类别名。"""
        result = self.analyze(png, **kwargs)
        for label in labels:
            el = self.find(result, label=label, region=region)
            if el:
                return self.center(el)
        return None

    def find_dialog_button(self, png: bytes, text: str | None = None,
                           label: str | None = None, **kwargs) -> tuple[int, int] | None:
        """弹窗按钮定位：先找弹窗容器（Alert/Window/PopUp Menu），把搜索范围限制在容器内。

        容器没被检出时退化为全屏查找。
        """
        result = self.analyze(png, **kwargs)
        elements = result.get("elements", [])
        container = next((e for e in elements if e["label"] in
                          ("Alert", "Window", "PopUp Menu", "ContextMenu")), None)
        region = tuple(container["box"]["xyxy"]) if container else None
        el = self.find(result, text=text, label=label, region=region)
        return self.center(el) if el else None

    def wait_for(self, screenshot: Callable[[], bytes], *, text: str | None = None,
                 label: str | None = None, timeout: float = 15.0, interval: float = 1.0,
                 **kwargs) -> tuple[int, int] | None:
        """轮询等待某元素出现（screenshot 为返回最新截图字节的可调用对象）。

        轮询期间单次截屏/识别失败视为"还没出现"，继续等待。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                center = self.find_by_text(screenshot(), text=text, **kwargs) \
                    if text else self.find_icon(screenshot(), labels=[label] if label else
                                                ("App Icon", "Utility Button", "Button"), **kwargs)
            except requests.RequestException:
                center = None  # 瞬时失败（黑帧/服务繁忙）不算超时
            if center:
                return center
            time.sleep(interval)
        return None


# ------------------------------------------------------------------ 内部 --
def _norm(s: str) -> str:
    return "".join(s.split()).lower()


def _text_hit(query: str, candidate: str, mode: str) -> float:
    q, c = _norm(query), _norm(candidate or "")
    if not q or not c:
        return 0.0
    if mode == "exact":
        return 1.0 if q == c else 0.0
    if mode == "fuzzy":
        ratio = difflib.SequenceMatcher(None, q, c).ratio()
        return ratio if ratio >= 0.75 else (0.9 if q in c else 0.0)
    return 1.0 if q in c else 0.0


def _match_elements(elements: list[dict], text: str | None, label: str | None,
                    match_mode: str, region) -> list[dict]:
    scored: list[tuple[float, dict]] = []
    for e in elements:
        if text:
            score = _text_hit(text, e.get("text") or "", match_mode)
            if score == 0.0:
                continue
        elif label:
            score = 1.0 if _norm(label) in _norm(e["label"]) else 0.0
            if score == 0.0:
                continue
        else:
            continue
        cx, cy = e["box"]["center"]
        if region and not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
            continue
        area = e["box"]["xywh"][2] * e["box"]["xywh"][3]
        scored.append((score * 10 + e["confidence"] + min(area / 1e6, 0.5), e))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored]


def _drop_none(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}
