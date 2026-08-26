"""Multi-scale template matching built on OpenCV."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_SCALES: tuple[float, ...] = (
    0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5,
    1.75, 2.0, 2.25, 2.5, 3.0,
)
# Relative refinement probes applied around the best coarse scale: recovers
# accurate boxes when the true DPI ratio falls between the coarse steps.
REFINE_MULTIPLIERS: tuple[float, ...] = (0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015)
MAX_CANDIDATES_PER_SCALE = 5000
# Scenes whose long side exceeds this are coarse-scanned on a 1/k downscale
# (k = ceil(long/COARSE_SCENE_MAX_SIDE)); only a narrow full-resolution band
# around the winning scale is then scanned, which both widens the searchable
# DPI range (scales/k covers k× more) and cuts the 4K matching cost.
COARSE_SCENE_MAX_SIDE = 2000
COARSE_MIN_TEMPLATE_SIDE = 8  # never shrink the template below this in the pyramid
# A second coarse peak at least this scale-separated and scoring within this
# margin of the best is refined as a rival hypothesis (guards against locking
# onto a wrong-scale NCC peak in busy scenes).
HYPOTHESIS_SCALE_GAP = 0.05
HYPOTHESIS_SCORE_MARGIN = 0.08


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


def _channel_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean NCC of one color channel, clamped to [0, 1].

    A flat channel carries no evidence, so it scores 1.0 instead of 0/NaN —
    the mean-color term below still judges flat patches.
    """
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    af -= af.mean()
    bf -= bf.mean()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom < 1e-6:
        return 1.0
    return min(1.0, max(0.0, float(af.ravel() @ bf.ravel()) / denom))


def _color_factor(template_patch: np.ndarray, scene_patch: np.ndarray) -> float:
    """Color agreement in [0, 1] between two patches.

    NCC alone is blind to color: a red and a blue button with the same
    texture both score ~1.0 on grayscale. Two complementary checks keep the
    gate cheap while separating differently-colored UI elements:

    - mean term: absolute distance between per-channel mean colors — rejects
      hue/tint swaps (the only signal when the tint rides on shared
      luminance) and bounds brightness sensitivity;
    - structure term: per-channel zero-mean NCC — verifies the colored
      *spatial layout* agrees, catching same-mean/different-layout patches
      (e.g. two-color icons sharing an average color).

    The factor only ever *reduces* a gray match and never scores above the
    mean term alone, so true matches behave exactly like the plain
    mean-color gate while false positives are cut harder.
    """
    if template_patch.size == 0 or scene_patch.size == 0:
        return 1.0  # nothing to compare — do not punish the gray match
    diff = float(
        np.abs(
            template_patch.reshape(-1, template_patch.shape[-1]).mean(axis=0)
            - scene_patch.reshape(-1, scene_patch.shape[-1]).mean(axis=0)
        ).mean()
    )
    mean_term = 1.0 - min(1.0, diff / 96.0)
    channels = min(template_patch.shape[-1], scene_patch.shape[-1])
    ncc_term = float(np.mean([
        _channel_ncc(template_patch[..., c], scene_patch[..., c])
        for c in range(channels)
    ]))
    return min(1.0, max(0.0, mean_term * (0.5 + 0.5 * ncc_term)))


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
    collect: bool = True,
) -> tuple[list[TemplateHit], TemplateHit | None]:
    """Match at one scale: fused-score peaks plus the scale's best peak.

    Candidates are pre-filtered on the grayscale NCC map (cheap), then each
    survivor is re-scored with a mean-color gate so same-shape/different-color
    elements stop phantom-matching. Final score = gray * (0.7 + 0.3 * color).
    With ``collect=False`` only the scale's best peak is computed (used by the
    coarse pyramid pass, which only picks scales and never reports boxes).
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
    if not collect:
        return [], scale_best

    pre = max(0.05, threshold - 0.25)  # loose gray net; color gate refines
    ys, xs = np.where(result >= pre)
    candidates: list[TemplateHit] = []
    if len(ys):
        if len(ys) > MAX_CANDIDATES_PER_SCALE:  # keep only the strongest peaks
            vals = result[ys, xs]
            top = np.argpartition(vals, -MAX_CANDIDATES_PER_SCALE)[-MAX_CANDIDATES_PER_SCALE:]
            ys, xs = ys[top], xs[top]
        if len(ys) > 64:
            # Vectorized mean-color prefilter. fused <= gray*(0.7+0.3*mean_term)
            # (the per-channel NCC term can only lower the color factor below
            # the mean term), so a candidate whose upper bound is below the
            # threshold can never pass the exact check — drop it wholesale.
            # boxFilter computes the per-position patch mean in one pass,
            # replacing tens of thousands of per-candidate color evaluations.
            tpl_mean = scaled_color.reshape(-1, scaled_color.shape[-1]).mean(axis=0)
            mean_map = cv2.boxFilter(scene_color, ddepth=cv2.CV_32F,
                                     ksize=(tw, th), normalize=True)
            # boxFilter covers the full image; matchTemplate only positions
            # where the template fully fits.
            mean_map = mean_map[: result.shape[0], : result.shape[1]]
            diff = np.abs(mean_map - tpl_mean.astype(np.float32)).mean(axis=-1)
            mean_term = np.clip(1.0 - diff / 96.0, 0.0, 1.0)
            bound = result * (0.7 + 0.3 * mean_term)
            keep = bound[ys, xs] >= threshold
            ys, xs = ys[keep], xs[keep]
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


class _Scanner:
    """Accumulates multi-scale scans of one scene variant (full or downscaled).

    Keeps every scale's best peak (for hypothesis selection), all threshold
    passing candidates (for NMS), the overall best fused hit, and the set of
    already-tried factors so repeated probes are skipped.
    """

    __slots__ = (
        "scene_color", "scene_gray", "template_color", "template_gray",
        "threshold", "tried", "candidates", "best", "scale_bests",
    )

    def __init__(self, scene_color: np.ndarray, scene_gray: np.ndarray,
                 template_color: np.ndarray, template_gray: np.ndarray,
                 threshold: float):
        self.scene_color = scene_color
        self.scene_gray = scene_gray
        self.template_color = template_color
        self.template_gray = template_gray
        self.threshold = threshold
        self.tried: list[float] = []
        self.candidates: list[TemplateHit] = []
        self.best: TemplateHit | None = None
        self.scale_bests: list[TemplateHit] = []

    def scan(self, factor: float, collect: bool = True) -> None:
        # Dedup tolerance must stay well below the refine step (~0.005 at
        # scale 1.0), or refinement probes get silently skipped.
        if any(abs(factor - t) < 1e-4 for t in self.tried):
            return
        self.tried.append(factor)
        scale_candidates, scale_best = _scan_scale(
            self.scene_color, self.template_color,
            self.scene_gray, self.template_gray,
            factor, self.threshold, collect=collect,
        )
        self.candidates.extend(scale_candidates)
        if scale_best is not None:
            self.scale_bests.append(scale_best)
            if self.best is None or scale_best.confidence > self.best.confidence:
                self.best = scale_best


def _pyramid_factor(scene_gray: np.ndarray, template_gray: np.ndarray) -> int:
    """Downscale divisor k for the coarse pass (1 = scan full resolution only).

    k grows with the scene long side but never shrinks the template below
    COARSE_MIN_TEMPLATE_SIDE — beyond that the coarse NCC would be noise.
    """
    long_side = max(scene_gray.shape[:2])
    if long_side <= COARSE_SCENE_MAX_SIDE:
        return 1
    k = -(-long_side // COARSE_SCENE_MAX_SIDE)  # ceil division
    min_side = min(template_gray.shape[:2])
    return max(1, min(k, min_side // COARSE_MIN_TEMPLATE_SIDE))


def _coarse_hypotheses(coarse: _Scanner, k: int) -> list[float]:
    """Full-resolution scale bands to scan, best coarse peak first.

    The winning coarse scale maps to ``peak.scale * k``. A runner-up at least
    HYPOTHESIS_SCALE_GAP apart (in relative scale) and scoring within
    HYPOTHESIS_SCORE_MARGIN of the best joins as a rival hypothesis, so a
    wrong-scale NCC peak in a busy scene does not silently win.
    """
    peaks = sorted(coarse.scale_bests, key=lambda h: h.confidence, reverse=True)
    if not peaks:
        return []
    factors = [peaks[0].scale * k]
    for peak in peaks[1:]:
        if peak.confidence < peaks[0].confidence - HYPOTHESIS_SCORE_MARGIN:
            break
        factor = peak.scale * k
        if any(abs(factor / f - 1.0) <= HYPOTHESIS_SCALE_GAP for f in factors):
            continue
        factors.append(factor)
        if len(factors) >= 2:
            break
    return factors


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

    Large scenes are coarse-scanned on a downscaled pyramid to pick the
    winning DPI band cheaply; narrow full-resolution bands around the top
    hypothesis scale(s) then produce precise boxes. A fine refinement pass
    zooms around the winner and greedy cross-scale NMS ensures one physical
    occurrence is reported once even when neighbouring scales all fire on it.

    Args:
        scene: BGR scene image.
        template: BGR template image.
        threshold: Minimum confidence for an occurrence to count as a hit.
        scales: Coarse resize factors to scan (divided by the pyramid factor
            on downscaled coarse passes, so the effective reach is k× wider).
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

    scanner = _Scanner(scene_color, scene_gray, template_color, template_gray, threshold)

    # Coarse-to-fine on large scenes: pick the winning DPI band on a 1/k
    # downscale (k× wider effective scale reach at ~k² lower cost), then scan
    # narrow full-resolution bands around the top hypothesis scale(s) only.
    k = _pyramid_factor(scene_gray, template_gray)
    if k > 1:
        sh, sw = scene_gray.shape[:2]
        small = cv2.resize(
            scene_color, (round(sw / k), round(sh / k)), interpolation=cv2.INTER_AREA
        )
        coarse = _Scanner(small, _gray(small), template_color, template_gray, threshold)
        for factor in sorted({round(f / k, 4) for f in dict.fromkeys((1.0, *scales))}):
            coarse.scan(factor, collect=False)
        factors = _coarse_hypotheses(coarse, k) or list(dict.fromkeys((1.0, *scales)))
    else:
        factors = list(dict.fromkeys((1.0, *scales)))  # native scale first
    for factor in factors:
        scanner.scan(factor)

    # Refine around the winning scale, iteratively, for sub-step box
    # accuracy (each round zooms ±1.5% around the current winner and stops
    # once the scale converges; 5 rounds let the winner walk up to ~7% from
    # the nearest coarse step). Exact native-scale matches skip this.
    if refine:
        for _ in range(5):
            if scanner.best is None or scanner.best.confidence >= 0.995:
                break
            prev_scale = scanner.best.scale
            for multiplier in REFINE_MULTIPLIERS:
                scanner.scan(prev_scale * multiplier)
            if abs(scanner.best.scale - prev_scale) < 0.004:
                break

    # scale_best peaks come from the gray map only; a color-gated candidate
    # can outscore them, and it must win the "best" title for reporting.
    if scanner.candidates:
        top = max(scanner.candidates, key=lambda h: h.confidence)
        if scanner.best is None or top.confidence > scanner.best.confidence:
            scanner.best = top

    candidates = sorted(scanner.candidates, key=lambda h: h.confidence, reverse=True)
    kept: list[TemplateHit] = []
    for cand in candidates:
        if any(_iou(cand.xyxy, hit.xyxy) > nms_iou for hit in kept):
            continue  # same physical occurrence seen through another scale
        kept.append(cand)
        if len(kept) >= max_instances:
            break
    return InstanceResult(instances=kept, best=scanner.best)


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
    "COARSE_SCENE_MAX_SIDE",
    "TemplateHit",
    "InstanceResult",
    "locate_template",
    "locate_instances",
]
