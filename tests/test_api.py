"""API smoke tests: no model downloads, OCR preload disabled."""

import base64
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    os.environ["YOCR_PRELOAD_OCR"] = "0"
    os.environ["YOCR_PRELOAD_MODELS"] = ""

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


def test_detect_default_model_missing_weights_404(client, png_b64):
    r = client.post("/api/v1/detect", json={"image_base64": png_b64})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


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


def test_corrupt_base64_image_400(client):
    r = client.post("/api/v1/ocr", json={"image_base64": base64.b64encode(b"junk").decode()})
    assert r.status_code == 400
