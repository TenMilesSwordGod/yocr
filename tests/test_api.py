"""API smoke tests: no model downloads, OCR preload disabled."""

import base64
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("YOCR_PRELOAD_OCR", "0")
        mp.setenv("YOCR_PRELOAD_MODELS", "")
        mp.setenv("HF_HUB_OFFLINE", "1")

        from yocr.app import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture()
def png_b64(tmp_path_factory) -> str:
    from PIL import Image

    arr = np.zeros((64, 96, 3), dtype=np.uint8)
    path = tmp_path_factory.mktemp("img") / "screen.png"
    Image.fromarray(arr).save(path)
    return base64.b64encode(path.read_bytes()).decode()


def test_healthz(client):
    r = client.get("/api/v1/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "android_ui_detection_yolov8" in body["models"]
    assert "ScreenParser" in body["models"]


def test_models_listing(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    models = {m["name"]: m for m in r.json()["models"]}
    assert models["ScreenParser"]["source"] == "docling-project/ScreenParser"
    assert not any(m["loaded"] for m in models.values())  # nothing preloaded
    assert all(m["error"] is None for m in models.values())  # no failures recorded


def test_models_listing_surfaces_load_error(client):
    ctx = client.app.state.ctx
    ctx.registry._errors["ScreenParser"] = "RuntimeError: offline cache miss"  # noqa: SLF001
    try:
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        models = {m["name"]: m for m in r.json()["models"]}
        assert models["ScreenParser"]["loaded"] is False
        assert "offline cache miss" in models["ScreenParser"]["error"]
        assert models["android_ui_detection_yolov8"]["error"] is None
    finally:
        ctx.registry._errors.pop("ScreenParser", None)  # noqa: SLF001


def test_detect_default_model_missing_weights_404(client, png_b64):
    r = client.post("/api/v1/detect", json={"image_base64": png_b64})
    assert r.status_code == 404
    assert "no usable detection model" in r.json()["detail"]


def test_detect_unknown_alias_404(client, png_b64):
    r = client.post(
        "/api/v1/detect?model=nope",
        json={"image_base64": png_b64},
    )
    assert r.status_code == 404
    assert "available" in r.json()["detail"]


def test_missing_image_400(client):
    r = client.post("/api/v1/ocr")
    assert r.status_code == 400


def test_cache_env_anchored_to_project(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)

    import importlib

    from yocr import app as app_module

    importlib.reload(app_module)  # re-run module-level setdefaults
    assert app_module.os.environ["HF_HOME"].endswith(".cache/huggingface")
    assert app_module.os.environ["PADDLE_PDX_CACHE_HOME"].endswith(".cache/paddlex")


def test_corrupt_base64_image_400(client):
    r = client.post("/api/v1/ocr", json={"image_base64": base64.b64encode(b"junk").decode()})
    assert r.status_code == 400


def _pattern_png(tmp_path, name, width, height, seed):
    import cv2

    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    path = tmp_path / name
    Image.fromarray(arr).save(path)
    return path.read_bytes()


def test_match_template_found(client, tmp_path_factory):
    import cv2

    tmp = tmp_path_factory.mktemp("match")
    template_bytes = _pattern_png(tmp, "tpl.png", 40, 30, seed=7)
    scene = np.zeros((200, 300, 3), dtype=np.uint8)
    tpl = cv2.imdecode(np.frombuffer(template_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    scene[60:90, 100:140] = tpl  # paste at a known offset
    ok, buf = cv2.imencode(".png", scene)
    assert ok
    r = client.post(
        "/api/v1/match",
        files={"file": ("scene.png", buf.tobytes(), "image/png"),
               "template": ("t.png", template_bytes, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["score"] >= 0.99
    assert body["box"]["xyxy"] == [100, 60, 140, 90]
    assert body["box"]["center"] == [120, 75]
    assert body["template"] == {"width": 40, "height": 30}


def test_match_template_not_found(client, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("match2")
    template_bytes = _pattern_png(tmp, "tpl.png", 40, 30, seed=1)
    scene_bytes = _pattern_png(tmp, "scene.png", 200, 300, seed=2)  # unrelated noise
    r = client.post(
        "/api/v1/match",
        files={"file": ("scene.png", scene_bytes, "image/png"),
               "template": ("t.png", template_bytes, "image/png")},
        data={"threshold": "0.99"},
    )
    assert r.status_code == 200
    assert r.json()["found"] is False
    assert r.json()["box"] is None


def test_match_template_missing_template_400(client, png_b64):
    r = client.post("/api/v1/match", json={"image_base64": png_b64})
    assert r.status_code == 400
