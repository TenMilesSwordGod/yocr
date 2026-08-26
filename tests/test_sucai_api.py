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


# ------------------------------------------------- extended edge cases ---
def test_create_id_edge_cases(client):
    png = _pattern_png(10, 10, seed=21)
    # dotted / dashed / digit-leading ids are legal
    for sid in ("a.b_c-d", "9lives"):
        r = client.post("/api/v1/sucai",
                        files={"file": ("x.png", png, "image/png")}, data={"id": sid})
        assert r.status_code == 201, f"{sid}: {r.text}"
    # illegal ids -> 400
    for bad in ("-lead", "_lead", ".lead", "has space", "中文", "x" * 65, "a/b"):
        r = client.post("/api/v1/sucai",
                        files={"file": ("x.png", png, "image/png")}, data={"id": bad})
        assert r.status_code == 400, bad
    # empty id field -> auto-generate instead of erroring
    r = client.post("/api/v1/sucai",
                    files={"file": ("x.png", png, "image/png")}, data={"id": ""})
    assert r.status_code == 201 and r.json()["id"]


def test_create_jpeg_normalized_to_png(client):
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(buf, format="JPEG")
    r = client.post("/api/v1/sucai",
                    files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")},
                    data={"id": "jpeg-src"})
    assert r.status_code == 201
    img = client.get("/api/v1/sucai/jpeg-src/image")
    assert img.content.startswith(b"\x89PNG")


def test_create_grayscale_and_tiny_images(client):
    import io

    from PIL import Image

    gray = io.BytesIO()
    Image.fromarray(np.full((15, 15), 128, dtype=np.uint8)).save(gray, format="PNG")
    r = client.post("/api/v1/sucai",
                    files={"file": ("g.png", gray.getvalue(), "image/png")},
                    data={"id": "gray-img"})
    assert r.status_code == 201 and r.json()["width"] == 15

    tiny = _pattern_png(1, 1, seed=22)
    r = client.post("/api/v1/sucai",
                    files={"file": ("t.png", tiny, "image/png")}, data={"id": "one-px"})
    assert r.status_code == 201


def test_create_empty_file_400(client):
    r = client.post("/api/v1/sucai", files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_create_oversized_image_400(client):
    big = cv2.imencode(".png", np.zeros((10, 2049, 3), dtype=np.uint8))[1].tobytes()
    r = client.post("/api/v1/sucai", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 400
    assert "too large" in r.json()["detail"]


def test_create_describe_is_trimmed(client):
    r = client.post("/api/v1/sucai",
                    files={"file": ("x.png", _pattern_png(8, 8, seed=23), "image/png")},
                    data={"id": "trim-me", "describe": "  padded  "})
    assert r.json()["describe"] == "padded"


def test_healthz_reports_sucai_count(client):
    before = client.get("/api/v1/healthz").json()["sucai_count"]
    client.post("/api/v1/sucai",
                files={"file": ("x.png", _pattern_png(8, 8, seed=24), "image/png")},
                data={"id": "counted"})
    after = client.get("/api/v1/healthz").json()["sucai_count"]
    assert after == before + 1


def test_update_no_fields_400_and_partial_updates(client):
    client.post("/api/v1/sucai",
                files={"file": ("x.png", _pattern_png(20, 10, seed=25), "image/png")},
                data={"id": "up-target", "describe": "original"})
    # neither file nor describe -> 400
    r = client.put("/api/v1/sucai/up-target")
    assert r.status_code == 400

    # describe only: picture untouched
    r = client.put("/api/v1/sucai/up-target", data={"describe": "only text"})
    assert r.status_code == 200 and r.json()["describe"] == "only text"
    assert (r.json()["width"], r.json()["height"]) == (20, 10)

    # picture only: describe untouched
    r = client.put("/api/v1/sucai/up-target",
                   files={"file": ("n.png", _pattern_png(11, 33, seed=26), "image/png")})
    assert r.status_code == 200
    assert (r.json()["width"], r.json()["height"]) == (11, 33)
    assert r.json()["describe"] == "only text"

    # empty describe explicitly clears
    r = client.put("/api/v1/sucai/up-target", data={"describe": ""})
    assert r.status_code == 200 and r.json()["describe"] == ""

    # corrupt replacement image -> 400, record intact
    r = client.put("/api/v1/sucai/up-target", files={"file": ("n.png", b"junk", "image/png")})
    assert r.status_code == 400
    assert client.get("/api/v1/sucai/up-target").json()["describe"] == ""


def test_image_endpoint_404_when_file_missing_on_disk(client, tmp_path):
    client.post("/api/v1/sucai",
                files={"file": ("x.png", _pattern_png(9, 9, seed=27), "image/png")},
                data={"id": "vanish"})
    from yocr.config import get_settings

    image_path = get_settings().sucai_dir / "images" / "vanish.png"
    image_path.unlink()
    assert client.get("/api/v1/sucai/vanish/image").status_code == 404
    # finder skips the orphaned entry instead of 500-ing
    scene = np.zeros((60, 60, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", scene)
    assert ok
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200
    assert "vanish" not in [m["id"] for m in r.json()["results"]]
    client.delete("/api/v1/sucai/vanish")


def test_find_scaled_template_multiscale(client):
    tpl_bytes = _pattern_png(40, 32, seed=31)
    tpl = cv2.imdecode(np.frombuffer(tpl_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    r = client.post("/api/v1/sucai",
                    files={"file": ("t.png", tpl_bytes, "image/png")},
                    data={"id": "scaled"})
    assert r.status_code == 201, r.text
    small = cv2.resize(tpl, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    scene = np.zeros((120, 160, 3), dtype=np.uint8)
    scene[40:64, 50:80] = small  # 30x24
    ok, buf = cv2.imencode(".png", scene)
    assert ok
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", buf.tobytes(), "image/png")},
                    params={"threshold": 0.8})
    mine = next(m for m in r.json()["results"] if m["id"] == "scaled")
    assert mine["found"] is True
    assert abs(mine["scale"] - 0.75) < 0.01
    assert mine["box"]["xyxy"] == [50, 40, 80, 64]


def test_find_template_larger_than_scene_never_found(client):
    client.post("/api/v1/sucai",
                files={"file": ("t.png", _pattern_png(300, 400, seed=32), "image/png")},
                data={"id": "huge"})
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", scene)
    assert ok
    # even at threshold 0 an incomparable template must stay found=False, box=None
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", buf.tobytes(), "image/png")},
                    params={"threshold": 0.0})
    assert r.status_code == 200
    mine = next(m for m in r.json()["results"] if m["id"] == "huge")
    assert mine["found"] is False and mine["box"] is None and mine["score"] == 0.0


def test_find_constant_template_never_found(client):
    # A solid-color sucai must not phantom-match (NCC undefined -> was ~1.0 everywhere)
    solid = cv2.imencode(".png", np.full((30, 40, 3), 77, dtype=np.uint8))[1].tobytes()
    r = client.post("/api/v1/sucai",
                    files={"file": ("s.png", solid, "image/png")}, data={"id": "solid"})
    assert r.status_code == 201
    scene = _pattern_png(120, 160, seed=51)
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", scene, "image/png")},
                    params={"threshold": 0.5})
    assert r.status_code == 200
    mine = next(m for m in r.json()["results"] if m["id"] == "solid")
    assert mine["found"] is False and mine["box"] is None


def test_find_results_sorted_and_top_k(client):
    a_bytes = _pattern_png(40, 30, seed=41)
    a = cv2.imdecode(np.frombuffer(a_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    r = client.post("/api/v1/sucai",
                    files={"file": ("a.png", a_bytes, "image/png")}, data={"id": "sort-a"})
    assert r.status_code == 201, r.text
    client.post("/api/v1/sucai",
                files={"file": ("b.png", _pattern_png(40, 30, seed=42), "image/png")},
                data={"id": "sort-b"})
    scene = np.zeros((200, 200, 3), dtype=np.uint8)
    rng = np.random.default_rng(99)
    scene[:] = rng.integers(0, 255, scene.shape, dtype=np.uint8)  # non-degenerate scene
    scene[10:40, 20:60] = a  # only sort-a present
    ok, buf = cv2.imencode(".png", scene)
    assert ok
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", buf.tobytes(), "image/png")}).json()
    scores = [m["score"] for m in r["results"]]
    assert scores == sorted(scores, reverse=True)
    assert r["results"][0]["id"] == "sort-a"

    top2 = client.post("/api/v1/sucai/find",
                       files={"file": ("s.png", buf.tobytes(), "image/png")},
                       params={"top_k": 2}).json()["results"]
    assert len(top2) == 2 and top2[0]["id"] == "sort-a"


def test_find_accepts_raw_binary_scene(client):
    scene = np.zeros((80, 80, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", scene)
    assert ok
    r = client.post("/api/v1/sucai/find?threshold=0.9",
                    content=buf.tobytes(),
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200
    assert r.json()["image"] == {"width": 80, "height": 80}


def test_find_accepts_image_base64_form_field(client):
    scene = np.zeros((70, 90, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", scene)
    assert ok
    r = client.post("/api/v1/sucai/find",
                    files={"image_base64": (None, base64.b64encode(buf.tobytes()).decode())})
    assert r.status_code == 200
    assert r.json()["image"] == {"width": 90, "height": 70}


def test_find_corrupt_scene_400(client):
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", b"junk", "image/png")})
    assert r.status_code == 400


def test_find_json_body_without_image_400(client):
    r = client.post("/api/v1/sucai/find", json={"unrelated": 1})
    assert r.status_code == 400


def test_find_all_instances_reports_every_occurrence(client):
    tpl_bytes = _pattern_png(40, 30, seed=61)
    tpl = cv2.imdecode(np.frombuffer(tpl_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    r = client.post("/api/v1/sucai",
                    files={"file": ("t.png", tpl_bytes, "image/png")},
                    data={"id": "twice"})
    assert r.status_code == 201, r.text

    scene = np.zeros((200, 300, 3), dtype=np.uint8)
    scene[20:50, 30:70] = tpl    # occurrence 1
    scene[100:130, 200:240] = tpl  # occurrence 2
    ok, buf = cv2.imencode(".png", scene)
    assert ok

    # default: single best location (backward compatible shape)
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", buf.tobytes(), "image/png")},
                    params={"threshold": 0.8}).json()
    mine = next(m for m in r["results"] if m["id"] == "twice")
    assert mine["found"] is True
    assert len(mine["hits"]) == 1

    # all_instances=true: both occurrences, exact boxes, best first
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", buf.tobytes(), "image/png")},
                    params={"threshold": 0.8, "all_instances": "true"}).json()
    mine = next(m for m in r["results"] if m["id"] == "twice")
    assert mine["found"] is True
    assert len(mine["hits"]) == 2
    boxes = [h["box"]["xyxy"] for h in mine["hits"]]
    assert [30, 20, 70, 50] in boxes and [200, 100, 240, 130] in boxes
    assert all(h["score"] >= 0.8 for h in mine["hits"])
    assert mine["box"]["xyxy"] == mine["hits"][0]["box"]["xyxy"]
    assert mine["center"] == mine["hits"][0]["center"]


def test_find_all_instances_nms_merges_overlapping_peaks(client):
    tpl_bytes = _pattern_png(40, 30, seed=62)
    tpl = cv2.imdecode(np.frombuffer(tpl_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    r = client.post("/api/v1/sucai",
                    files={"file": ("t.png", tpl_bytes, "image/png")},
                    data={"id": "nms-me"})
    assert r.status_code == 201

    # two copies shifted by only 4px: same physical spot, NMS must keep one
    scene = np.zeros((150, 150, 3), dtype=np.uint8)
    scene[40:70, 40:80] = tpl
    scene[44:74, 44:84] = tpl
    ok, buf = cv2.imencode(".png", scene)
    assert ok
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", buf.tobytes(), "image/png")},
                    params={"threshold": 0.8, "all_instances": "true"}).json()
    mine = next(m for m in r["results"] if m["id"] == "nms-me")
    assert len(mine["hits"]) == 1


def test_find_scale_refinement_locates_offscale_template(client):
    """True scale 1.02 is not in the coarse list; refinement must nail the box."""
    tpl_bytes = _pattern_png(40, 32, seed=63)
    tpl = cv2.imdecode(np.frombuffer(tpl_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    r = client.post("/api/v1/sucai",
                    files={"file": ("t.png", tpl_bytes, "image/png")},
                    data={"id": "offscale"})
    assert r.status_code == 201, r.text

    small = cv2.resize(tpl, None, fx=1.02, fy=1.02, interpolation=cv2.INTER_AREA)
    scene = np.zeros((150, 200, 3), dtype=np.uint8)
    scene[50:83, 60:101] = small  # 41x33 at (60, 50)
    ok, buf = cv2.imencode(".png", scene)
    assert ok
    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", buf.tobytes(), "image/png")},
                    params={"threshold": 0.8, "all_instances": "true"}).json()
    mine = next(m for m in r["results"] if m["id"] == "offscale")
    assert mine["found"] is True
    assert abs(mine["scale"] - 1.02) <= 0.016
    x1, y1, x2, y2 = mine["box"]["xyxy"]
    assert abs(x1 - 60) <= 2 and abs(y1 - 50) <= 2
    assert abs(x2 - 101) <= 2 and abs(y2 - 83) <= 2


def _tinted_texture(seed, base_bgr):
    """Texture whose grayscale is ~identical across different tints.

    gray = 0.299R + 0.587G + 0.114B ends up equal for both tints (the tint
    rides on a shared luminance texture), so grayscale NCC alone cannot
    tell them apart — only the color gate can.
    """
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 50, (30, 40), dtype=np.uint8).astype(np.int16)
    b, g, r = base_bgr
    img = np.stack(
        [np.clip(b + t, 0, 255), np.clip(g + t, 0, 255), np.clip(r + t, 0, 255)],
        axis=-1,
    ).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_find_color_gate_blocks_cross_color_matches(client):
    """Same texture, different color: gray NCC would cross-match at ~1.0."""
    red = _tinted_texture(71, (0, 0, 200))    # red-dominant, gray ≈ t + 60
    blue = _tinted_texture(71, (255, 0, 0))   # blue-dominant, gray ≈ t + 60
    assert red != blue

    r = client.post("/api/v1/sucai",
                    files={"file": ("red.png", red, "image/png")},
                    data={"id": "red-btn", "describe": "红色按钮"})
    assert r.status_code == 201
    r = client.post("/api/v1/sucai",
                    files={"file": ("blue.png", blue, "image/png")},
                    data={"id": "blue-btn", "describe": "蓝色按钮"})
    assert r.status_code == 201

    red_img = cv2.imdecode(np.frombuffer(red, np.uint8), cv2.IMREAD_COLOR)
    blue_img = cv2.imdecode(np.frombuffer(blue, np.uint8), cv2.IMREAD_COLOR)
    scene = np.zeros((160, 240, 3), dtype=np.uint8)
    scene[20:50, 30:70] = red_img
    scene[90:120, 150:190] = blue_img
    ok, buf = cv2.imencode(".png", scene)
    assert ok

    r = client.post("/api/v1/sucai/find",
                    files={"file": ("s.png", buf.tobytes(), "image/png")},
                    params={"threshold": 0.8, "all_instances": "true"}).json()
    by_id = {m["id"]: m for m in r["results"]}

    # each sucai found exactly once, at its own color's location
    assert by_id["red-btn"]["found"] is True
    assert by_id["blue-btn"]["found"] is True
    assert len(by_id["red-btn"]["hits"]) == 1
    assert len(by_id["blue-btn"]["hits"]) == 1
    assert by_id["red-btn"]["box"]["xyxy"] == [30, 20, 70, 50]
    assert by_id["blue-btn"]["box"]["xyxy"] == [150, 90, 190, 120]
    # the wrong-color location must not score above the threshold
    assert by_id["red-btn"]["hits"][0]["score"] >= 0.8
    assert by_id["blue-btn"]["hits"][0]["score"] >= 0.8
