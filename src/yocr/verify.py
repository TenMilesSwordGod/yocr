"""Geometric verification & affine refinement of template matches (XFeat).

NCC candidates are cheap and high-recall but can phantom-match (a wrong
scale can scrape past the threshold at a wrong location, and pixel-level
correlation is blind to geometry). This module re-checks every candidate
with learned local features (XFeat, CVPR 2024): the template and the scene
crop are matched, a similarity transform is estimated with RANSAC, and
the result is accepted only when enough inliers corroborate the geometry.

Accepted matches also get a refined box: the template corners are mapped
through the estimated transform, which lands boxes accurately even when the
DPI ratio is off-grid or the render differs (JPEG artifacts, anti-aliasing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .template import _color_factor
from .xfeat import DEFAULT_WEIGHTS, xfeat_model

logger = logging.getLogger("yocr.verify")

# A similarity transform needs >= 2 point pairs; require a few more inliers to
# trust that the geometry is real and not a lucky RANSAC draw.
MIN_INLIERS = 3
MIN_MATCHES = 2
# XFeat needs at least this many template keypoints before we bother verifying;
# featureless flat templates are unverifiable and stay on plain NCC.
MIN_TEMPLATE_KEYPOINTS = 3
MAX_SCENE_KEYPOINTS = 1024
MAX_TEMPLATE_KEYPOINTS = 512
COSSIM_GATE = 0.82
MATCH_CAP = 64  # feed at most this many best matches into MAGSAC
# Pixel-exact NCC matches (at or above this) skip score folding and keep the
# NCC box: the geometry is already conclusive to the pixel, and learned-feature
# refinement would only add keypoint-level noise.
EXACT_NCC_SCORE = 0.98
# Minimum fraction of fitting correspondences that RANSAC must agree on.
# A phantom on a smooth patch can scrape 3 inliers out of many spurious
# matches; genuine geometry is usually >= 0.3 of the pool.
EVIDENCE_MIN = 0.25
# Minimum color agreement between template and the refined box region.
# The template crop background is usually dark and the phantom sits on a
# gray-ish patch, so a colored UI icon scores ~1.0 at its true location and
# well below this at a wrong one.
COLOR_AGREEMENT_MIN = 0.5
# Verification evidence is folded into the NCC score as
#   fused = ncc * (VERIFY_WEIGHT_MIN + (1 - VERIFY_WEIGHT_MIN) * evidence)
# so a phantom can never keep a high score, while an unverifiable match
# (flat icon, missing weights) keeps its plain NCC score untouched.
VERIFY_WEIGHT_MIN = 0.7
# A candidate whose geometry verification ran (template had features) but
# failed gets its score cut by this factor: it is a suspicious NCC peak, not
# a confirmed hit. Unverifiable templates keep their NCC score untouched.
FAILED_VERIFY_DISCOUNT = 0.85


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of geometric verification for one candidate hit.

    States, in increasing specificity:

    - ``attempted=False``: verification could not even try (no model, or the
      template had too few features) — the candidate is passed through.
    - ``attempted=True, matched=False``: features were extracted but there was
      no correspondence with the scene crop — the candidate is passed through
      (a genuine off-grid match on a featureless patch must not be punished).
    - ``matched=True, ok=False``: correspondences existed but geometry/color
      rejected them — strong phantom evidence, the candidate is discounted.
    - ``ok=True``: trusted geometry; the candidate keeps a blended score and
      gets a refined box.

    Attributes:
        ok: Enough inliers for a trusted similarity transform.
        attempted: Verification ran at all (model available + template had
            enough features).
        matched: At least MIN_MATCHES mutual correspondences were found.
        inliers: Number of RANSAC-consistent correspondences.
        evidence: Inlier ratio in [0, 1] (evidence strength).
        box: Refined xyxy in scene pixel coordinates when ok, else None.
        scale: Scene-to-template size ratio recovered by the transform
            (mean of x/y scale, always positive) when ok, else None.
    """

    ok: bool = False
    attempted: bool = False
    matched: bool = False
    inliers: int = 0
    evidence: float = 0.0
    box: tuple[int, int, int, int] | None = None
    scale: float | None = None


@dataclass(frozen=True)
class _TemplateFeats:
    keypoints: np.ndarray  # (N, 2) float64, original template pixels
    descriptors: np.ndarray  # (N, 64) float32


def _prep_for_xfeat(image_bgr: np.ndarray, min_side: int) -> np.ndarray:
    """Scale a small patch up for feature extraction and pad to 32-multiples.

    XFeat's internal preprocessing snaps inputs down to the largest multiple
    of 32 below the current size — for a 63x50 template that is a 32x32
    downscale that destroys the features. We upscale undersized patches
    (INTER_CUBIC) and pad the right/bottom edge with neutral gray so the
    network sees at least min_side pixels in the smallest dimension.

    Only right/bottom padding is used: keypoint coordinates stay identical
    to the input's pixel space, no offset bookkeeping needed.
    """
    h, w = image_bgr.shape[:2]
    if min(h, w) < min_side:
        scale = min_side / min(h, w)
        image_bgr = cv2.resize(
            image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC
        )
    h2, w2 = image_bgr.shape[:2]
    pad_b = (-h2) % 32
    pad_r = (-w2) % 32
    if pad_b or pad_r:
        image_bgr = cv2.copyMakeBorder(
            image_bgr, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=(127, 127, 127)
        )
    return image_bgr


def _to_xfeat_tensor(image_bgr: np.ndarray):
    import torch  # noqa: PLC0415 - lazy heavy import

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def _extract(xfeat, image_bgr: np.ndarray, *, top_k: int,
             min_side: int) -> _TemplateFeats:
    """Extract features; keypoints are mapped back to the *input* pixel space.

    ``_prep_for_xfeat`` may upscale (min_side) and pad (32-multiples); padding
    only shifts nothing (it grows right/bottom), but the upscale stretches
    pixel coordinates, so keypoints are divided by the scale factor to stay
    consistent with the input image both for matching and for RANSAC geometry.
    """
    prepared = _prep_for_xfeat(image_bgr, min_side)
    out = xfeat.detectAndCompute(_to_xfeat_tensor(prepared), top_k=top_k)[0]
    factor = prepared.shape[0] / image_bgr.shape[0]
    keypoints = out["keypoints"].cpu().numpy().astype(np.float64)
    if factor != 1.0:
        keypoints = keypoints / factor
    return _TemplateFeats(
        keypoints=keypoints,
        descriptors=out["descriptors"].cpu().numpy().astype(np.float32),
    )


def _color_agreement(template_bgr: np.ndarray, scene_bgr: np.ndarray,
                     box: tuple[int, int, int, int]) -> float:
    """Color agreement in [0, 1] between the template and the scene region.

    The refined box may be a pixel or two off and the patch sizes differ, so
    the scene region is resized to the template size before the shared
    ``_color_factor`` (mean color distance + per-channel NCC) is evaluated.
    """
    x1, y1, x2, y2 = box
    sh, sw = scene_bgr.shape[:2]
    patch = scene_bgr[max(0, y1):min(sh, y2), max(0, x1):min(sw, x2)]
    if patch.size == 0:
        return 0.0
    th, tw = template_bgr.shape[:2]
    if patch.shape[:2] != (th, tw):
        patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
    return float(_color_factor(template_bgr, patch))


def _mutual_best(d1: np.ndarray, d2: np.ndarray,
                 min_cossim: float = COSSIM_GATE) -> np.ndarray:
    """Mutual-nearest-neighbour descriptor matches, (N, 2) index pairs."""
    if len(d1) == 0 or len(d2) == 0:
        return np.empty((0, 2), dtype=np.int64)
    similarity = d1 @ d2.T
    idx_to_2 = similarity.argmax(axis=1)
    idx_to_1 = similarity.argmax(axis=0)
    mutual = idx_to_1[idx_to_2] == np.arange(len(d1))
    best = similarity[np.arange(len(d1)), idx_to_2]
    good = mutual & (best > min_cossim)
    return np.column_stack([np.where(good)[0], idx_to_2[good]])


class FeatureVerifier:
    """Verifies template candidates in a scene using XFeat features.

    The torch model loads lazily on first use; when the weights are missing
    every verification degrades to ``ok=False`` and callers keep plain NCC.
    """

    def __init__(self, weights: Path | str | None = None):
        self._weights = Path(weights) if weights else DEFAULT_WEIGHTS

    @property
    def available(self) -> bool:
        """True when the XFeat model is (or can be) loaded for this verifier."""
        return xfeat_model(self._weights) is not None

    def template_features(self, template_bgr: np.ndarray) -> _TemplateFeats | None:
        """Extract template features once per sucai (reused for all hits)."""
        if template_bgr is None or not getattr(template_bgr, "size", 0):
            return None
        model = xfeat_model(self._weights)
        if model is None:
            return None
        feats = _extract(
            model, template_bgr, top_k=MAX_TEMPLATE_KEYPOINTS, min_side=64
        )
        if len(feats.keypoints) < MIN_TEMPLATE_KEYPOINTS:
            return None
        return feats

    def verify_hit(
        self,
        scene_bgr: np.ndarray,
        template_bgr: np.ndarray,
        template_feats: _TemplateFeats | None,
        candidate_box: tuple[int, int, int, int],
        *,
        scale: float = 1.0,
        margin: float = 0.2,
    ) -> VerifyResult:
        """Verify one candidate box; returns a refined box when accepted.

        ``scale`` is the NCC-reported scene/template size ratio: the crop is
        re-scaled by ``1/scale`` so the presumed object is compared at its
        native size, which keeps XFeat descriptors scale-aligned whether the
        icon is on a 4K or a 1K screenshot. The recovered similarity transform
        is therefore near-identity in that normalized space; corners mapped
        through it land at sub-pixel accuracy in scene pixels.
        """
        model = xfeat_model(self._weights)
        if model is None or template_feats is None:
            return VerifyResult()
        attempted = VerifyResult(attempted=True)
        x1, y1, x2, y2 = candidate_box
        sh, sw = scene_bgr.shape[:2]
        pad = int(max(x2 - x1, y2 - y1) * margin)
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(sw, x2 + pad), min(sh, y2 + pad)
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return attempted
        crop = scene_bgr[cy1:cy2, cx1:cx2]

        # Normalize the object back to native size: work in a virtual space
        # where the template and the scene object share one scale.
        norm = 1.0 / scale if 0.25 <= scale <= 4.0 else 1.0
        if norm == 1.0:
            work_crop = crop
        else:
            work_crop = cv2.resize(
                crop, None, fx=norm, fy=norm, interpolation=cv2.INTER_AREA
            )
        scene_feats = _extract(
            model, work_crop, top_k=MAX_SCENE_KEYPOINTS, min_side=96
        )
        if len(scene_feats.keypoints) < MIN_MATCHES:
            return attempted
        matches = _mutual_best(
            template_feats.descriptors, scene_feats.descriptors
        )
        if len(matches) < MIN_MATCHES:
            return attempted
        # Correspondence found: from here on a rejection means the NCC hit is
        # a phantom, not an unverifiable match.
        ran = VerifyResult(attempted=True, matched=True)

        pts_src = template_feats.keypoints[matches[:, 0]]
        # Scene matches live in the normalized crop space; they are mapped
        # into full scene pixel space only when projecting the final box.
        pts_dst = scene_feats.keypoints[matches[:, 1]]

        # Keep only the strongest-correlation matches for geometric fitting.
        order = np.argsort(
            np.einsum(
                "ij,ij->i",
                template_feats.descriptors[matches[:, 0]],
                scene_feats.descriptors[matches[:, 1]],
            )
        )[::-1][:MATCH_CAP]
        pts_src, pts_dst = pts_src[order], pts_dst[order]

        matrix, inliers = cv2.estimateAffinePartial2D(
            pts_src, pts_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.999,
            maxIters=2000,
        )
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        if matrix is None or inlier_count < MIN_INLIERS:
            return ran  # geometry never assembled: inconclusive, not a phantom
        # RANSAC did assemble a transform — but if gates below reject it, the
        # candidate is a phantom (marked 'rejected' via inliers >= MIN_INLIERS).
        rejected = VerifyResult(attempted=True, matched=True, inliers=inlier_count)

        # Transform template corners into the normalized crop space, then map
        # them back to full scene pixels (crop offset + normalized coords /
        # norm — i.e. * original scale.).
        tw, th = template_bgr.shape[1], template_bgr.shape[0]
        corners = np.array(
            [[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float64
        )
        mapped = cv2.transform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2)
        if norm != 1.0:
            mapped = mapped / norm
        mapped[:, 0] += cx1
        mapped[:, 1] += cy1
        xs = np.clip(mapped[:, 0], 0, sw - 1)
        ys = np.clip(mapped[:, 1], 0, sh - 1)
        box = (
            int(round(xs.min())), int(round(ys.min())),
            int(round(xs.max())) + 1, int(round(ys.max())) + 1,
        )
        scale_det = np.linalg.det(matrix[:, :2])
        if scale_det <= 0:
            return rejected
        # Residual skewing in the normalized space is tiny for genuine UI
        # matches; a transform that still stretches a lot is not a match.
        recovered = float(np.sqrt(scale_det))
        if not (0.7 <= recovered <= 1.43):
            return rejected
        evidence = float(inlier_count / max(4, len(pts_src)))
        if evidence < EVIDENCE_MIN:
            return rejected
        # Color sanity: a geometrically-plausible phantom can still sit on a
        # patch of the wrong color (smooth gray noise fools local features).
        # The refined box region must agree with the template's colors.
        if _color_agreement(template_bgr, scene_bgr, box) < COLOR_AGREEMENT_MIN:
            return rejected
        return VerifyResult(
            ok=True,
            attempted=True,
            matched=True,
            inliers=inlier_count,
            evidence=min(1.0, evidence),
            box=box,
            scale=scale * recovered,
        )


def fuse_score(ncc: float, verify: VerifyResult | None) -> float:
    """Fold geometric evidence into an NCC score.

    Outcomes, mirroring :class:`VerifyResult`:

    - Not attempted (no model / featureless template) or not matched (no
      correspondence at all): cannot verify — pass the NCC score through.
    - ``ok``: geometry corroborated; the score is blended toward ``ncc`` in
      proportion to evidence (true matches stay high).
    - Matched but RANSAC never assembled a transform (``inliers`` below
      MIN_INLIERS): inconclusive — pass through, a genuine off-grid match on
      a featureless patch must not be punished.
    - Matched with a RANSAC transform that the evidence/color gates rejected
      (``inliers`` >= MIN_INLIERS yet ``ok`` False): strong phantom evidence —
      discount hard so a wrong-scale ~0.83 hit drops below the default gate.
    """
    if verify is None or not verify.attempted or not verify.matched:
        return float(ncc)
    if not verify.ok:
        if verify.inliers < MIN_INLIERS:
            return float(ncc)
        return float(np.clip(ncc * FAILED_VERIFY_DISCOUNT, 0.0, 1.0))
    return float(np.clip(
        ncc * (VERIFY_WEIGHT_MIN + (1.0 - VERIFY_WEIGHT_MIN) * verify.evidence),
        0.0,
        1.0,
    ))


__all__ = [
    "FeatureVerifier",
    "VerifyResult",
    "fuse_score",
    "MIN_INLIERS",
    "VERIFY_WEIGHT_MIN",
    "FAILED_VERIFY_DISCOUNT",
]