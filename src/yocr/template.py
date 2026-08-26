"""Multi-scale template matching built on OpenCV."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_SCALES: tuple[float, ...] = (
    0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5,
)
# Relative refinement probes applied around the best coarse scale: recovers
# accurate boxes when the true DPI ratio falls between the coarse steps.
REFINE_MULTIPLIERS: tuple[float, ...] = (0.97, 0.985, 1.0, 1.015, 1.03)
MAX_CANDIDATES_PER_SCALE = 5000


@dataclass(frozen=True)
class TemplateHit:
    """Best template occurrence inside the scene image.

    Attributes:
        confidence: Normalized correlation score in [0, 1].
        scale: Template resize factor that produced this hit.
        xyxy: Pixel-space box (x1, y1, x2, y2) of the matched region.
    """

    confidence: float
    scale: float
    xyxy: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        """tuple[int, int]: Center pixel of the matched region."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass(frozen=True)
class InstanceResult:
    """Outcome of a multi-instance search.

    Attributes:
        instances: All occurrences scoring >= threshold, NMS-deduped, best first.
        best: Best raw match regardless of threshold (for near-miss reporting).
    """

    instances: list[TemplateHit] = field(default_factory=list)
    best: TemplateHit | None = None


def _clamp(value: float) -> float:
    """Clamp a correlation score into [0, 1] (NCC can exceed bounds numerically)."""
    return min(1.0, max(0.0, float(value)))


def _color_factor(template_patch: np.ndarray, scene_patch: np.ndarray) -> float:
    """Brightness-invariant color agreement in [0, 1] between two patches.

    NCC alone is blind to color: a red and a blue button with the same
    texture both score ~1.0 on grayscale. Comparing mean BGR keeps the
    gate cheap while separating differently-colored UI elements. The factor
    only ever *reduces* a gray match: identical colors return 1.0, moderate
    shading keeps most of the score, a genuinely different color cuts it.
    """
    if template_patch.size == 0 or scene_patch.size == 0:
        return 1.0  # nothing to compare — do not punish the gray match
    diff = float(
        np.abs(
            template_patch.reshape(-1, template_patch.shape[-1]).mean(axis=0)
            - scene_patch.reshape(-1, scene_patch.shape[-1]).mean(axis=0)
        ).mean()
    )
    return 1.0 - min(1.0, diff / 96.0)


def _scaled_size(template_gray: np.ndarray, factor: float) -> tuple[int, int]:
    """Target (width, height) for a template at `factor`, min 1px.

    Sizes are computed explicitly: fx/fy rounding on tiny templates can
    produce an empty dsize and make cv2.resize raise.
    """
    th = max(1, int(round(template_gray.shape[0] * factor)))
    tw = max(1, int(round(template_gray.shape[1] * factor)))
    return tw, th


def _usable(tw: int, th: int, sw: int, sh: int) -> bool:
    """A scale is usable when the resized template fits 4px minimum in scene."""
    return th >= 4 and tw >= 4 and th <= sh and tw <= sw


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _scan_scale(
    scene_color: np.ndarray,
    template_color: np.ndarray,
    scene_gray: np.ndarray,
    template_gray: np.ndarray,
    factor: float,
    threshold: float,
) -> tuple[list[TemplateHit], TemplateHit | None]:
    """Match at one scale: fused-score peaks plus the scale's best peak.

    Candidates are pre-filtered on the grayscale NCC map (cheap), then each
    survivor is re-scored with a mean-color gate so same-shape/different-color
    elements stop phantom-matching. Final score = gray * (0.7 + 0.3 * color).
    """
    sh, sw = scene_gray.shape[:2]
    tw, th = _scaled_size(template_gray, factor)
    if not _usable(tw, th, sw, sh):
        return [], None
    scaled_gray = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
    scaled_color = cv2.resize(template_color, (tw, th), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(scene_gray, scaled_gray, cv2.TM_CCOEFF_NORMED)

    def fuse(x: int, y: int, gray_score: float) -> float:
        penalty = _color_factor(
            scaled_color, scene_color[y : y + th, x : x + tw]
        )
        return _clamp(gray_score * (0.7 + 0.3 * penalty))

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    scale_best = TemplateHit(
        confidence=fuse(max_loc[0], max_loc[1], max_val),
        scale=factor,
        xyxy=(max_loc[0], max_loc[1], max_loc[0] + tw, max_loc[1] + th),
    )

    pre = max(0.05, threshold - 0.25)  # loose gray net; color gate refines
    ys, xs = np.where(result >= pre)
    candidates: list[TemplateHit] = []
    if len(ys):
        if len(ys) > MAX_CANDIDATES_PER_SCALE:  # keep only the strongest peaks
            vals = result[ys, xs]
            top = np.argpartition(vals, -MAX_CANDIDATES_PER_SCALE)[-MAX_CANDIDATES_PER_SCALE:]
            ys, xs = ys[top], xs[top]
        for x, y, gray_score in zip(xs.tolist(), ys.tolist(), result[ys, xs].tolist()):
            fused = fuse(int(x), int(y), float(gray_score))
            if fused >= threshold:
                candidates.append(TemplateHit(
                    confidence=fused,
                    scale=factor,
                    xyxy=(int(x), int(y), int(x) + tw, int(y) + th),
                ))
    return candidates, scale_best


def locate_template(
    scene: np.ndarray,
    template: np.ndarray,
    *,
    threshold: float = 0.8,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> TemplateHit | None:
    """Find the template inside the scene using grayscale NCC matching.

    Args:
        scene: BGR scene image (usually a full screenshot).
        template: BGR template image (usually a cropped widget/icon).
        threshold: Minimum confidence to consider it a match.
        scales: Template resize factors tried in addition to native size;
            compensates DPI differences between the two sources.

    Returns:
        TemplateHit | None: Best hit when its confidence >= threshold, else None.
    """
    result = locate_instances(
        scene, template, threshold=threshold, scales=scales, max_instances=1
    )
    return result.instances[0] if result.instances else None


def locate_instances(
    scene: np.ndarray,
    template: np.ndarray,
    *,
    threshold: float = 0.8,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    refine: bool = True,
    nms_iou: float = 0.3,
    max_instances: int = 50,
) -> InstanceResult:
    """Find every occurrence of the template in the scene.

    Coarse multi-scale NCC scan, then a fine refinement pass around the best
    scale, then greedy cross-scale NMS so one physical occurrence is reported
    once even when neighbouring scales all fire on it.

    Args:
        scene: BGR scene image.
        template: BGR template image.
        threshold: Minimum confidence for an occurrence to count as a hit.
        scales: Coarse resize factors to scan.
        refine: Probe finer scales around the best coarse scale.
        nms_iou: Boxes with IoU above this are considered the same occurrence.
        max_instances: Upper bound on reported instances.

    Returns:
        InstanceResult: `instances` sorted by score desc (may be empty) and
        `best` = highest-scoring raw match even when below threshold.
    """
    if scene is None or template is None or not getattr(scene, "size", 0) or not getattr(template, "size", 0):
        raise ValueError("empty scene or template image")
    scene_gray = _gray(scene)
    template_gray = _gray(template)
    scene_color = _as_bgr(scene)
    template_color = _as_bgr(template)
    # A constant template has zero variance: NCC is undefined and OpenCV
    # returns ~1.0 everywhere, so such a template would "match" any scene.
    if float(template_gray.std()) < 1e-6:
        return InstanceResult()

    candidates: list[TemplateHit] = []
    best: TemplateHit | None = None
    tried: list[float] = []

    def scan(factor: float) -> None:
        nonlocal candidates, best
        if any(abs(factor - t) < 5e-3 for t in tried):
            return
        tried.append(factor)
        scale_candidates, scale_best = _scan_scale(
            scene_color, template_color, scene_gray, template_gray, factor, threshold
        )
        candidates.extend(scale_candidates)
        if scale_best and (best is None or scale_best.confidence > best.confidence):
            best = scale_best

    for factor in dict.fromkeys((1.0, *scales)):  # native scale first
        scan(factor)

    # Refine around the winning coarse scale, iteratively, for sub-step box
    # accuracy (each round zooms ±3% around the current winner and stops once
    # the scale converges). Exact native-scale matches skip this entirely.
    if refine:
        for _ in range(3):
            if best is None or best.confidence >= 0.995:
                break
            prev_scale = best.scale
            for multiplier in REFINE_MULTIPLIERS:
                scan(prev_scale * multiplier)
            if abs(best.scale - prev_scale) < 0.004:
                break

    # scale_best peaks come from the gray map only; a color-gated candidate
    # can outscore them, and it must win the "best" title for reporting.
    if candidates:
        top = max(candidates, key=lambda h: h.confidence)
        if best is None or top.confidence > best.confidence:
            best = top

    candidates.sort(key=lambda h: h.confidence, reverse=True)
    kept: list[TemplateHit] = []
    for cand in candidates:
        if any(_iou(cand.xyxy, k.xyxy) > nms_iou for k in kept):
            continue  # same physical occurrence seen through another scale
        kept.append(cand)
        if len(kept) >= max_instances:
            break
    return InstanceResult(instances=kept, best=best)


def _gray(image: np.ndarray) -> np.ndarray:
    """Convert an arbitrary BGR/gray image to single-channel grayscale.

    Args:
        image: Input array with 1 or 3 channels.

    Returns:
        np.ndarray: Grayscale uint8 image.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Return the image as 3-channel BGR so color gates always have channels."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


__all__ = [
    "DEFAULT_SCALES",
    "REFINE_MULTIPLIERS",
    "TemplateHit",
    "InstanceResult",
    "locate_template",
    "locate_instances",
]
