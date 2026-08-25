"""Unit tests for YOLO registry error tracking and default-model fallback."""

import sys
import types

import numpy as np
import pytest
from fastapi import HTTPException

from yocr.config import Settings
from yocr.detectors import DEFAULT_MODEL, DEFAULT_MODEL_REPO, YOLORegistry
from yocr.pipeline import AnalysisContext, detect, make_context


class _FakeYOLO:
    names = {0: "Text"}

    def predict(self, *args, **kwargs):
        return []


@pytest.fixture()
def image() -> np.ndarray:
    return np.zeros((32, 32, 3), dtype=np.uint8)


def _registry(monkeypatch, fail_default=True) -> YOLORegistry:
    registry = YOLORegistry(Settings())

    def fake_load(self, spec):
        if fail_default and spec.name.lower() == DEFAULT_MODEL.lower():
            raise FileNotFoundError(
                f"model weights '{spec.source}' not found in '<models>' and not a HF hub id"
            )
        return _FakeYOLO()

    monkeypatch.setattr(YOLORegistry, "_load", fake_load)
    return registry


def test_get_records_and_clears_load_error(monkeypatch):
    registry = _registry(monkeypatch)
    with pytest.raises(FileNotFoundError):
        registry.get(DEFAULT_MODEL)
    assert "not found" in registry.last_error("android_ui_detection_yolov8")

    def ok_load(self, spec):
        return _FakeYOLO()

    monkeypatch.setattr(YOLORegistry, "_load", ok_load)
    _, model = registry.get(DEFAULT_MODEL)
    assert model is not None
    assert registry.last_error("android_ui_detection_yolov8") is None


def test_detect_without_pinned_model_falls_back(monkeypatch, image):
    ctx = AnalysisContext(settings=Settings(), registry=_registry(monkeypatch))
    name, elements, _ = detect(ctx, image)
    assert name == "ScreenParser"
    assert elements == []


def test_detect_pinned_model_stays_strict(monkeypatch, image):
    ctx = AnalysisContext(settings=Settings(), registry=_registry(monkeypatch))
    with pytest.raises(HTTPException) as excinfo:
        detect(ctx, image, model=DEFAULT_MODEL)
    assert excinfo.value.status_code == 404
    assert "not found" in str(excinfo.value.detail)


def test_detect_all_candidates_unavailable_aggregates_errors(monkeypatch, image):
    def all_fail(self, spec):
        raise FileNotFoundError(f"model weights '{spec.source}' not found")

    monkeypatch.setattr(YOLORegistry, "_load", all_fail)
    ctx = AnalysisContext(settings=Settings(), registry=YOLORegistry(Settings()))
    with pytest.raises(HTTPException) as excinfo:
        detect(ctx, image)
    detail = str(excinfo.value.detail)
    assert excinfo.value.status_code == 404
    assert "no usable detection model" in detail
    assert DEFAULT_MODEL in detail
    assert "ScreenParser" in detail


class _FakeBox:
    def __init__(self) -> None:
        self.xyxy = [types.SimpleNamespace(tolist=lambda: [1.0, 2.0, 3.0, 4.0])]
        self.cls = [types.SimpleNamespace(item=lambda: 0)]
        self.conf = [types.SimpleNamespace(item=lambda: 0.9)]


class _FakeResult:
    boxes = None


def test_predict_label_resolves_class_type_via_model_names(monkeypatch, image):
    """Even when the registry cache misses, labels must use the model's own names."""
    model = _FakeYOLO()
    model.names = {0: "Text"}

    result = _FakeResult()
    result.boxes = [_FakeBox()]
    model.predict = lambda *a, **k: [result]

    monkeypatch.setattr(YOLORegistry, "_load", lambda self, spec: model)
    registry = YOLORegistry(Settings())
    registry._classes.pop("ScreenParser", None)  # simulate missing cache table

    _, detections = registry.predict(image, model="screenparser")
    assert len(detections) == 1
    assert detections[0].label == "Text"
    assert detections[0].class_id == 0
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].xyxy == (1.0, 2.0, 3.0, 4.0)


def test_normalize_names_variants():
    from yocr.detectors import _normalize_names

    assert _normalize_names({0: "A", "1": "B"}) == {0: "A", 1: "B"}
    assert _normalize_names(["X", "Y"]) == {0: "X", 1: "Y"}
    assert _normalize_names(("Z",)) == {0: "Z"}
    assert _normalize_names(None) == {}
    assert _normalize_names(42) == {}


def test_make_context_registry_exposes_last_error():
    registry = make_context(Settings()).registry
    assert registry.last_error("ScreenParser") is None


def test_resolve_source_prefers_local_file(tmp_path):
    (tmp_path / "android_ui_detection_yolov8.pt").write_bytes(b"weights")
    registry = YOLORegistry(Settings(models_dir=tmp_path))
    resolved = registry._resolve_source(registry.spec(DEFAULT_MODEL))  # noqa: SLF001
    assert resolved == str((tmp_path / "android_ui_detection_yolov8.pt").resolve())


def test_resolve_source_downloads_from_hf_when_no_local_file(tmp_path, monkeypatch):
    calls = {}

    def fake_download(repo_id, filename):
        calls["repo_id"], calls["filename"] = repo_id, filename
        return "/fake/cache/best.pt"

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    registry = YOLORegistry(Settings(models_dir=tmp_path))
    resolved = registry._resolve_source(registry.spec(DEFAULT_MODEL))  # noqa: SLF001
    assert resolved == "/fake/cache/best.pt"
    assert calls == {"repo_id": DEFAULT_MODEL_REPO, "filename": "best.pt"}
