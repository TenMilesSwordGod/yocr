"""Feature-verification (XFeat) unit tests.

Skipped wholesale when the XFeat weights are not provisioned
(``.cache/xfeat/xfeat.pt``), so the suite also runs on bare checkouts.
"""

import cv2
import numpy as np
import pytest

from yocr.verify import (
    FeatureVerifier,
    VerifyResult,
    fuse_score,
    FAILED_VERIFY_DISCOUNT,
)
from yocr.xfeat import DEFAULT_WEIGHTS, reset_cache


@pytest.fixture(autouse=True)
def _reset_xfeat_cache():
    reset_cache()
    yield
    reset_cache()


def _needs_xfeat():
    if not DEFAULT_WEIGHTS.is_file():
        pytest.skip("xfeat weights not provisioned (make models-download)")


@pytest.fixture()
def rng():
    return np.random.default_rng(11)


def _icon():
    """Synthetic UI icon: dark crop background + colored disc + white logo
    + small blue dot — structured like real icon crops, so XFeat has real
    geometry to verify (weak blurry textures are genuinely unverifiable)."""
    img = np.full((56, 56, 3), 40, np.uint8)
    cv2.circle(img, (28, 28), 18, (200, 90, 30), -1)   # BGR: orange disc
    cv2.circle(img, (28, 28), 12, (235, 235, 240), -1)  # white inner logo
    cv2.circle(img, (28, 28), 5, (40, 120, 220), -1)    # blue accent dot
    cv2.rectangle(img, (4, 44), (12, 52), (90, 90, 100), -1)  # orientation tick
    return img


def _scene_with(rng, tpl, scale, x, y, busy=True):
    th, tw = tpl.shape[:2]
    if busy:
        # Smooth structured background (like a real UI panel / desktop), not
        # white noise: XFeat keys on sharp noise everywhere, which drowns the
        # genuine correspondence under spurious matches.
        scene = cv2.GaussianBlur(
            rng.integers(0, 255, (1044, 1920, 3), dtype=np.uint8), (0, 0), 6.0
        )
    else:
        scene = np.zeros((1044, 1920, 3), dtype=np.uint8)
    big = cv2.resize(tpl, (round(tw * scale), round(th * scale)), interpolation=cv2.INTER_CUBIC)
    scene[y : y + big.shape[0], x : x + big.shape[1]] = big
    return scene, (x, y, x + big.shape[1], y + big.shape[0])


def test_verify_true_match_native(rng):
    _needs_xfeat()
    tpl = _icon()
    scene, truth = _scene_with(rng, tpl, 1.0, 500, 600)
    verifier = FeatureVerifier()
    feats = verifier.template_features(tpl)
    assert feats is not None and len(feats.keypoints) >= 3
    result = verifier.verify_hit(scene, tpl, feats, truth, scale=1.0)
    assert result.ok
    assert result.inliers >= 3
    assert result.box is not None
    assert max(abs(a - b) for a, b in zip(result.box, truth)) <= 4


def test_verify_true_match_2x_scale(rng):
    """Normalization to template scale must keep 2x features alignable."""
    _needs_xfeat()
    tpl = _icon()
    scene, truth = _scene_with(rng, tpl, 2.0, 500, 600)
    verifier = FeatureVerifier()
    feats = verifier.template_features(tpl)
    result = verifier.verify_hit(scene, tpl, feats, truth, scale=2.0)
    assert result.ok
    assert result.inliers >= 3
    assert max(abs(a - b) for a, b in zip(result.box, truth)) <= 6


def test_verify_rejects_phantom(rng):
    _needs_xfeat()
    tpl = _icon()
    scene, _ = _scene_with(rng, tpl, 1.0, 500, 600)
    th, tw = tpl.shape[:2]
    phantom_box = (900, 400, 900 + tw, 400 + th)  # no icon there
    verifier = FeatureVerifier()
    feats = verifier.template_features(tpl)
    result = verifier.verify_hit(scene, tpl, feats, phantom_box, scale=1.0)
    assert not result.ok
    assert result.attempted  # we did run and failed -> discount applies


def test_verify_unavailable_degrades_gracefully(rng):
    tpl = _icon()
    scene, truth = _scene_with(rng, tpl, 1.0, 500, 600)
    verifier = FeatureVerifier(weights="/nonexistent/xfeat.pt")
    assert not verifier.available
    feats = verifier.template_features(tpl)
    assert feats is None
    result = verifier.verify_hit(scene, tpl, feats, truth, scale=1.0)
    assert not result.ok and not result.attempted


def test_fuse_score_states():
    assert fuse_score(0.83, None) == pytest.approx(0.83)
    # attempted but not matched: cannot verify -> passthrough
    assert fuse_score(0.83, VerifyResult(attempted=True)) == pytest.approx(0.83)
    assert fuse_score(0.83, VerifyResult(attempted=True, matched=False)) == pytest.approx(0.83)
    # matched but RANSAC never assembled: inconclusive -> passthrough
    inconclusive = VerifyResult(attempted=True, matched=True, inliers=0)
    assert fuse_score(0.83, inconclusive) == pytest.approx(0.83)
    # matched with a transform the gates rejected: phantom -> discount below 0.8
    failed = VerifyResult(attempted=True, matched=True, inliers=4)
    assert fuse_score(0.83, failed) == pytest.approx(0.83 * FAILED_VERIFY_DISCOUNT)
    assert fuse_score(0.83, failed) < 0.8
    # matched and ok: blend toward ncc by evidence
    ok = VerifyResult(ok=True, attempted=True, matched=True, evidence=0.5)
    assert fuse_score(0.95, ok) == pytest.approx(0.95 * (0.7 + 0.3 * 0.5))
