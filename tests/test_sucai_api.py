"""Sucai registry + finder API tests (offline, no models)."""

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    sucai_dir = tmp_path_factory.mktemp("sucai-root") / "library"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("YOCR_PRELOAD_OCR", "0")
        mp.setenv("YOCR_PRELOAD_MODELS", "")
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.setenv("YOCR_SUCAI_DIR", str(sucai_dir))

        from yocr.app import create_app
        from yocr.config import get_settings

        # Other test modules may have cached Settings before our env vars applied.
        get_settings.cache_clear()
        app = create_app()
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c
        finally:
            get_settings.cache_clear()


def _pattern_png(width, height, seed):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def test_sucai_crud_roundtrip(client):
    png = _pattern_png(40, 30, seed=7)

    # create with explicit id + describe
    r = client.post("/api/v1/sucai",
                    files={"file": ("btn.png", png, "image/png")},
                    data={"id": "btn-ok", "describe": "确认按钮"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] == "btn-ok"
    assert body["describe"] == "确认按钮"
    assert body["width"] == 40 and body["height"] == 30
    assert body["image_url"].endswith("/api/v1/sucai/btn-ok/image")

    # duplicate id -> 409
    r = client.post("/api/v1/sucai",
                    files={"file": ("x.png", png, "image/png")}, data={"id": "btn-ok"})
    assert r.status_code == 409

    # auto-generated id
    r = client.post("/api/v1/sucai",
                    files={"file": ("x.png", _pattern_png(20, 20, seed=3), "image/png")},
                    data={"describe": "auto id"})
    assert r.status_code == 201
    auto_id = r.json()["id"]
    assert auto_id and auto_id != "btn-ok"

    # invalid id -> 400; corrupt image -> 400
    r = client.post("/api/v1/sucai",
                    files={"file": ("x.png", png, "image/png")}, data={"id": "bad id!"})
    assert r.status_code == 400
    r = client.post("/api/v1/sucai", files={"file": ("x.png", b"junk", "image/png")})
    assert r.status_code == 400

    # list contains both
    items = {i["id"]: i for i in client.get("/api/v1/sucai").json()["items"]}
    assert {"btn-ok", auto_id} <= set(items)

    # image bytes served back identical (normalized PNG)
    img = client.get(f"/api/v1/sucai/{auto_id}/image")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    decoded = cv2.imdecode(np.frombuffer(img.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (20, 20)

    # update describe + replace picture
    r = client.put("/api/v1/sucai/btn-ok",
                   files={"file": ("new.png", _pattern_png(50, 25, seed=9), "image/png")},
                   data={"describe": "更新过的按钮"})
    assert r.status_code == 200
    assert r.json()["describe"] == "更新过的按钮"
    assert r.json()["width"] == 50

    # unknown ids -> 404
    assert client.get("/api/v1/sucai/nope").status_code == 404
    assert client.get("/api/v1/sucai/nope/image").status_code == 404
    assert client.delete("/api/v1/sucai/nope").status_code == 404

    # delete roundtrip
    assert client.delete(f"/api/v1/sucai/{auto_id}").json()["deleted"] is True
    assert client.get(f"/api/v1/sucai/{auto_id}/image").status_code == 404


def test_sucai_find_locates_registered_template(client):
    tpl_bytes = _pattern_png(48, 36, seed=11)
    tpl = cv2.imdecode(np.frombuffer(tpl_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    r = client.post("/api/v1/sucai",
                    files={"file": ("t.png", tpl_bytes, "image/png")},
                    data={"id": "find-me", "describe": "目标素材"})
    assert r.status_code == 201

    scene = np.zeros((200, 300, 3), dtype=np.uint8)
    scene[60:96, 100:148] = tpl
    ok, buf = cv2.imencode(".png", scene)
    assert ok

    r = client.post("/api/v1/sucai/find",
                    files={"file": ("scene.png", buf.tobytes(), "image/png")},
                    params={"threshold": 0.8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sucai_count"] >= 1
    assert body["found_any"] is True

    mine = next(m for m in body["results"] if m["id"] == "find-me")
    assert mine["found"] is True
    assert mine["score"] >= 0.99
    assert mine["box"]["xyxy"] == [100, 60, 148, 96]
    assert mine["center"] == [124, 78]

    results = body["results"]
    assert results == sorted(results, key=lambda m: m["score"], reverse=True)


def test_sucai_find_json_base64_and_threshold(client):
    scene = np.zeros((120, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", scene)
    r = client.post("/api/v1/sucai/find",
                    json={"image_base64": base64.b64encode(buf.tobytes()).decode()},
                    params={"threshold": 0.99})
    assert r.status_code == 200
    body = r.json()
    assert body["found_any"] is False
    # every result reports found=False but still carries its best score
    assert all(m["found"] is False for m in body["results"])
    assert all(m["score"] < 0.99 for m in body["results"])

    # missing image -> 400
    assert client.post("/api/v1/sucai/find").status_code == 400


def test_sucai_persistence_across_restart(client):
    """meta.json + images survive a fresh SucaiStore on the same dir."""
    from yocr.config import Settings
    from yocr.sucai import SucaiStore

    store: SucaiStore = client.app.state.ctx.sucai
    before = {r["id"] for r in store.list()}
    assert before  # earlier tests registered something

    reloaded = SucaiStore(Settings(sucai_dir=store.root).sucai_dir)
    assert {r["id"] for r in reloaded.list()} == before
