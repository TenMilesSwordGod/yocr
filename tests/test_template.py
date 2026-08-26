"""Template matching unit tests: pyramid coarse scan, scale refinement, color gate."""

import cv2
import numpy as np
import pytest

from yocr.template import (
    COARSE_SCENE_MAX_SIDE,
    locate_instances,
    locate_template,
)


def _noise(rng, w, h):
    """Blurred noise: autocorrelated like real UI crops, so resampling
    (cubic paste vs area match) stays correlated the way screenshots do."""
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(arr, (0, 0), 2.5)


def _paste(scene, patch, x, y):
    h, w = patch.shape[:2]
    scene[y : y + h, x : x + w] = patch
    return (x, y, x + w, y + h)


@pytest.fixture()
def rng():
    return np.random.default_rng(7)


def test_native_scale_exact_match(rng):
    tpl = _noise(rng, 63, 50)
    scene = _noise(rng, 1920, 1044)
    truth = _paste(scene, tpl, 1847, 672)
    result = locate_instances(scene, tpl, threshold=0.8)
    assert result.instances, "native-scale template must be found"
    hit = result.instances[0]
    assert hit.xyxy == truth
    assert abs(hit.scale - 1.0) < 0.005
    assert hit.confidence >= 0.99
    assert hit.center == ((truth[0] + truth[2]) // 2, (truth[1] + truth[3]) // 2)


def test_pyramid_locates_2x_template_in_4k_scene(rng):
    """Regression: low-res template crop vs 4K scene needs a 2.0x scale.

    The old 0.5..1.5 scale list missed the true icon entirely and reported a
    phantom ~0.82 hit at a wrong location; the pyramid must nail it.
    """
    tpl = _noise(rng, 63, 50)
    scene = _noise(rng, 3840, 2088)  # long side 3840 -> pyramid k=2
    big = cv2.resize(tpl, (126, 100), interpolation=cv2.INTER_CUBIC)
    truth = _paste(scene, big, 3695, 1345)
    result = locate_instances(scene, tpl, threshold=0.8)
    assert result.instances, "2x template in 4K scene must be found"
    hit = result.instances[0]
    assert max(abs(a - b) for a, b in zip(hit.xyxy, truth)) <= 1
    assert abs(hit.scale - 2.0) < 0.03
    assert hit.confidence >= 0.9


def test_pyramid_finds_multiple_occurrences_at_2x(rng):
    tpl = _noise(rng, 63, 50)
    scene = _noise(rng, 3840, 2088)
    big = cv2.resize(tpl, (126, 100), interpolation=cv2.INTER_CUBIC)
    truth_a = _paste(scene, big, 300, 300)
    truth_b = _paste(scene, big, 3695, 1345)
    result = locate_instances(scene, tpl, threshold=0.8)
    boxes = [h.xyxy for h in result.instances]
    assert len(boxes) >= 2
    for truth in (truth_a, truth_b):
        assert any(max(abs(a - b) for a, b in zip(box, truth)) <= 1 for box in boxes)


def test_refine_nails_offgrid_scale(rng):
    """True scale 1.42 is not in the coarse list; refinement must nail it."""
    tpl = _noise(rng, 63, 50)
    scene = _noise(rng, 1600, 1200)
    big = cv2.resize(tpl, (89, 71), interpolation=cv2.INTER_CUBIC)
    truth = _paste(scene, big, 500, 300)
    result = locate_instances(scene, tpl, threshold=0.8)
    assert result.instances
    hit = result.instances[0]
    assert max(abs(a - b) for a, b in zip(hit.xyxy, truth)) <= 1
    assert abs(hit.scale - 89 / 63) < 0.02


def test_pyramid_disabled_for_small_scenes(rng):
    """Scenes at or below COARSE_SCENE_MAX_SIDE never take the pyramid path."""
    tpl = _noise(rng, 40, 32)
    scene = _noise(rng, 640, 480)
    truth = _paste(scene, tpl, 300, 200)
    result = locate_instances(scene, tpl, threshold=0.8)
    assert result.instances and result.instances[0].xyxy == truth


def test_pyramid_tiny_template_falls_back_to_full_res(rng):
    """A template too small to survive the 1/k downscale still matches."""
    tpl = _noise(rng, 10, 8)
    scene = _noise(rng, 4000, 300)  # forces k>=2, but 8//2=4px would be noise-level
    truth = _paste(scene, tpl, 2500, 100)
    result = locate_instances(scene, tpl, threshold=0.8)
    assert result.instances, "tiny template must fall back to full-res scan"
    assert max(abs(a - b) for a, b in zip(result.instances[0].xyxy, truth)) <= 1


def test_locate_template_returns_best_hit_or_none(rng):
    tpl = _noise(rng, 40, 30)
    scene = _noise(rng, 480, 640)
    _paste(scene, tpl, 300, 200)
    hit = locate_template(scene, tpl, threshold=0.8)
    assert hit is not None and hit.confidence >= 0.8
    assert locate_template(scene, _noise(rng, 40, 30), threshold=0.99) is None


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        locate_instances(np.zeros((10, 10, 3), np.uint8), np.zeros((0, 0, 3), np.uint8))
