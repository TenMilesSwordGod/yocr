"""Multi-scale template matching built on OpenCV."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_SCALES: tuple[float, ...] = (0.5, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5)


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
    if scene is None or template is None or not getattr(scene, "size", 0) or not getattr(template, "size", 0):
        raise ValueError("empty scene or template image")
    scene_gray = _gray(scene)
    template_gray = _gray(template)
    # A constant template has zero variance: NCC is undefined and OpenCV
    # returns ~1.0 everywhere, so such a template would "match" any scene.
    if float(template_gray.std()) < 1e-6:
        return None
    best: TemplateHit | None = None
    for factor in dict.fromkeys((1.0, *scales)):  # native scale first for early exit
        # Compute the target size explicitly: fx/fy rounding on tiny templates
        # can produce an empty dsize and make cv2.resize raise.
        th = max(1, int(round(template_gray.shape[0] * factor)))
        tw = max(1, int(round(template_gray.shape[1] * factor)))
        sh, sw = scene_gray.shape[:2]
        if th < 4 or tw < 4 or th > sh or tw > sw:
            continue  # template unusable at this scale
        scaled = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(scene_gray, scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        hit = TemplateHit(
            confidence=float(max_val),
            scale=factor,
            xyxy=(max_loc[0], max_loc[1], max_loc[0] + tw, max_loc[1] + th),
        )
        if factor == 1.0 and hit.confidence >= 0.995:
            return hit  # pixel-exact at native scale, no need to scan others
        if hit.confidence >= threshold and (best is None or hit.confidence > best.confidence):
            best = hit
    return best


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
