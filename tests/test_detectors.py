"""Unit tests for YOLO registry error tracking and default-model fallback."""

import sys
import types

import numpy as np
import pytest
from fastapi import HTTPException

from yocr.config import Settings
from yocr.detectors import (
    DEFAULT_ICON_PROMPTS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_REPO,
    ICONFINDER_WEIGHTS_URL,
    YOLORegistry,
)
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


def test_resolve_source_passes_http_url_through():
    registry = YOLORegistry(Settings(allow_download=True))
    resolved = registry._resolve_source(registry.spec("iconfinder"))  # noqa: SLF001
    assert resolved == ICONFINDER_WEIGHTS_URL


def test_resolve_source_blocked_without_allow_download(tmp_path):
    registry = YOLORegistry(Settings(models_dir=tmp_path))  # default: downloads off
    with pytest.raises(FileNotFoundError) as excinfo:
        registry._resolve_source(registry.spec("iconfinder"))  # noqa: SLF001
    message = str(excinfo.value)
    assert "make models-download" in message
    assert "YOCR_ALLOW_DOWNLOAD=1" in message


def test_resolve_source_iconfinder_local_file_first(tmp_path):
    (tmp_path / "yolov8s-worldv2.pt").write_bytes(b"world-weights")
    registry = YOLORegistry(Settings(models_dir=tmp_path, allow_download=True))
    resolved = registry._resolve_source(registry.spec("iconfinder"))  # noqa: SLF001
    assert resolved == str((tmp_path / "yolov8s-worldv2.pt").resolve())


def test_warmup_all_preloads_every_registered_model(monkeypatch):
    monkeypatch.setenv("YOCR_PRELOAD_MODELS", "all")
    loaded: list[str] = []

    def fake_load(self, spec):
        loaded.append(spec.name)
        return _FakeYOLO()

    monkeypatch.setattr(YOLORegistry, "_load", fake_load)
    registry = make_context(Settings()).registry  # preload_models defaults to ("all",)
    from yocr.detectors import warmup

    warmup(registry, Settings())
    assert set(loaded) == {"android_ui_detection_yolov8", "ScreenParser", "IconFinder"}
    for key in ("android_ui_detection_yolov8", "screenparser", "iconfinder"):
        _, fetched = registry.get(key)  # preload 后应可即时取到已加载实例
        assert isinstance(fetched, _FakeYOLO)


def test_warmup_empty_env_disables_preload(monkeypatch):
    monkeypatch.setenv("YOCR_PRELOAD_MODELS", "")
    loaded: list[str] = []
    monkeypatch.setattr(YOLORegistry, "_load", lambda self, spec: loaded.append(spec.name) or _FakeYOLO())
    registry = make_context(Settings()).registry
    from yocr.detectors import warmup

    warmup(registry, Settings())
    assert loaded == []


def test_builtin_iconfinder_spec_uses_prompts():
    settings = Settings()
    spec = YOLORegistry(settings).spec("iconfinder")
    assert spec.name == "IconFinder"
    assert spec.prompts == DEFAULT_ICON_PROMPTS
    assert "gear" in spec.prompts[0]


def test_custom_icon_classes_env(monkeypatch):
    monkeypatch.setenv("YOCR_ICON_CLASSES", "gear icon, wifi icon")
    spec = YOLORegistry(Settings()).spec("iconfinder")
    assert spec.prompts == ("gear icon", "wifi icon")


def test_load_applies_prompt_names(monkeypatch):
    class _WorldModel:
        def __init__(self, source: str) -> None:
            self.source = source  # 记录传入 YOLO() 的权重地址
            self.names = {0: "coco-class"}  # 原始 COCO 类别，set_classes 后应被覆盖

        def set_classes(self, prompts) -> None:
            self.names = {i: str(p) for i, p in enumerate(prompts)}

    created: dict[str, object] = {}

    def fake_yolo(source: str) -> _WorldModel:
        model = _WorldModel(source)
        created["source"] = source
        return model

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=fake_yolo))
    monkeypatch.setenv("YOCR_ICON_CLASSES", "gear icon, wifi icon")
    registry = YOLORegistry(Settings(allow_download=True))
    _, model = registry.get("iconfinder")
    assert created["source"] == ICONFINDER_WEIGHTS_URL  # http 直链透传到 YOLO()
    assert model.names == {0: "gear icon", 1: "wifi icon"}
    assert registry.classes("iconfinder") == {"0": "gear icon", "1": "wifi icon"}


def test_resolve_source_downloads_from_hf_when_no_local_file(tmp_path, monkeypatch):
    calls = {}

    def fake_download(repo_id, filename):
        calls["repo_id"], calls["filename"] = repo_id, filename
        return "/fake/cache/best.pt"

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    registry = YOLORegistry(Settings(models_dir=tmp_path, allow_download=True))
    resolved = registry._resolve_source(registry.spec(DEFAULT_MODEL))  # noqa: SLF001
    assert resolved == "/fake/cache/best.pt"
    assert calls == {"repo_id": DEFAULT_MODEL_REPO, "filename": "best.pt"}
