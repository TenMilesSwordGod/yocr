"""Target matching: locate the requested UI element among detection results."""

from __future__ import annotations

from difflib import SequenceMatcher

from .schemas import Element

FUZZY_THRESHOLD = 0.75


def _norm(text: str | None) -> str:
    return "".join((text or "").split()).lower()


def text_score(query: str, candidate: str | None, mode: str = "contains",
               fuzzy_threshold: float = FUZZY_THRESHOLD) -> float:
    """Similarity in [0, 1]; 0 means no match. Whitespace/case insensitive."""
    q, c = _norm(query), _norm(candidate)
    if not q or not c:
        return 0.0
    if mode == "exact":
        return 1.0 if q == c else 0.0
    if mode == "fuzzy":
        ratio = SequenceMatcher(None, q, c).ratio()
        if ratio >= fuzzy_threshold:
            return ratio
        return max(ratio, 0.9) if q in c else 0.0
    return 1.0 if q in c else 0.0  # contains


def best_match(elements: list[Element], *, text: str | None = None,
               label: str | None = None, q: str | None = None,
               match_mode: str = "contains") -> Element | None:
    """Return the best-matching element, or None.

    text:  匹配元素上归属的 OCR 文本
    label: 匹配 YOLO 类别名
    q:     泛搜索，文本或类别任一命中即算
    同时给多个条件时为 AND 关系。
    """
    if not (text or label or q):
        return None

    def hit(element: Element) -> float:
        score = 1.0
        if text:
            s = text_score(text, element.text, match_mode)
            if s == 0.0:
                return 0.0
            score *= s
        if label:
            lq, lc = _norm(label), _norm(element.label)
            if lq != lc and lq not in lc:
                return 0.0
        if q:
            s = max(text_score(q, element.text, match_mode),
                    text_score(q, element.label, match_mode))
            if s == 0.0:
                return 0.0
            score *= s
        return score

    scored = []
    for element in elements:
        s = hit(element)
        if s > 0.0:
            area = element.box.xywh[2] * element.box.xywh[3]
            scored.append((s * 10 + element.confidence + min(area / 1_000_000, 0.5), element))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


__all__ = ["best_match", "text_score", "FUZZY_THRESHOLD"]
