"""SucaiStore unit tests: id rules, corrupt persistence, ordering, disk edge cases."""

import json

import cv2
import numpy as np
import pytest

from yocr.sucai import ID_PATTERN, SucaiConflict, SucaiError, SucaiStore


@pytest.fixture()
def store(tmp_path):
    return SucaiStore(tmp_path / "lib")


def _png(width=40, height=30, seed=1):
    rng = np.random.default_rng(seed)
    ok, buf = cv2.imencode(".png", rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


# ------------------------------------------------------------ id rules ---
@pytest.mark.parametrize("sid", ["a", "Z9", "btn-ok", "a.b_c-d", "x" * 64, "1"])
def test_validate_id_accepts(sid):
    assert ID_PATTERN.match(sid)


@pytest.mark.parametrize(
    "sid",
    ["", " lead", "-lead", "_lead", ".lead", "has space", "中文id", "x" * 65, "a/b", "a:b"],
)
def test_validate_id_rejects(sid):
    assert not ID_PATTERN.match(sid)


def test_create_with_invalid_id_raises(store):
    with pytest.raises(SucaiError):
        store.create(_png(), sid="bad id!")
    assert store.count() == 0


def test_create_empty_id_autogenerates(store):
    record = store.create(_png(), sid="")
    assert record["id"]


# ------------------------------------------------------------ creation ---
def test_create_rejects_oversized_image(store):
    big = cv2.imencode(".png", np.zeros((10, 2049, 3), dtype=np.uint8))[1].tobytes()
    with pytest.raises(SucaiError, match="too large"):
        store.create(big)


def test_create_rejects_corrupt_image(store):
    with pytest.raises(ValueError):
        store.create(b"not-an-image")


def test_create_normalizes_to_png_and_records_size(store, tmp_path):
    ok, jpg = cv2.imencode(".jpg", np.zeros((20, 25, 3), dtype=np.uint8))
    assert ok
    record = store.create(jpg.tobytes(), sid="jpeg-src")
    on_disk = (tmp_path / "lib" / "images" / "jpeg-src.png").read_bytes()
    assert on_disk.startswith(b"\x89PNG")
    assert record["size_bytes"] == len(on_disk)
    assert (record["width"], record["height"]) == (25, 20)


def test_create_strips_describe(store):
    record = store.create(_png(), describe="  hello world  ")
    assert record["describe"] == "hello world"


def test_duplicate_id_conflicts(store):
    store.create(_png(), sid="dup")
    with pytest.raises(SucaiConflict):
        store.create(_png(seed=2), sid="dup")


def test_auto_ids_are_unique(store):
    ids = {store.create(_png(seed=i))["id"] for i in range(20)}
    assert len(ids) == 20


# ------------------------------------------------------- read / update ---
def test_get_unknown_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.get("nope")


def test_update_describe_only_keeps_image_bytes(store, tmp_path):
    store.create(_png(), sid="u1", describe="old")
    before = (tmp_path / "lib" / "images" / "u1.png").read_bytes()
    record = store.update("u1", describe="new")
    assert record["describe"] == "new"
    assert (tmp_path / "lib" / "images" / "u1.png").read_bytes() == before
    assert record["updated_at"]


def test_update_image_only_keeps_describe(store):
    store.create(_png(), sid="u2", describe="keep me")
    record = store.update("u2", image_bytes=_png(50, 20, seed=5))
    assert record["describe"] == "keep me"
    assert (record["width"], record["height"]) == (50, 20)


def test_update_empty_describe_clears_it(store):
    store.create(_png(), sid="u3", describe="x")
    assert store.update("u3", describe="   ")["describe"] == ""


def test_update_unknown_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.update("nope", describe="x")


def test_update_nothing_raises(store):
    store.create(_png(), sid="u4")
    with pytest.raises(SucaiError):
        store.update("u4")


def test_update_corrupt_image_leaves_record_intact(store, tmp_path):
    store.create(_png(), sid="u5", describe="safe")
    with pytest.raises(ValueError):
        store.update("u5", image_bytes=b"junk")
    assert store.get("u5")["describe"] == "safe"


# ------------------------------------------------------------- delete ----
def test_delete_removes_meta_and_image(store, tmp_path):
    store.create(_png(), sid="d1")
    image = tmp_path / "lib" / "images" / "d1.png"
    assert image.is_file()
    store.delete("d1")
    assert not image.is_file()
    with pytest.raises(KeyError):
        store.get("d1")


def test_delete_unknown_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.delete("nope")


# -------------------------------------------------------------- image ----
def test_read_image_missing_file_raises_filenotfound(store):
    store.create(_png(), sid="gone")
    (store.root / "images" / "gone.png").unlink()
    with pytest.raises(FileNotFoundError):
        store.read_image("gone")


# --------------------------------------------------------- persistence ---
def test_meta_survives_reload(tmp_path):
    store = SucaiStore(tmp_path / "lib")
    store.create(_png(), sid="p1", describe="first")
    reloaded = SucaiStore(tmp_path / "lib")
    assert reloaded.get("p1")["describe"] == "first"


def test_corrupt_meta_starts_empty(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    (root / "meta.json").write_bytes(b"{not json")
    assert SucaiStore(root).list() == []


def test_non_dict_meta_starts_empty(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    (root / "meta.json").write_text('["a", "b"]', encoding="utf-8")
    assert SucaiStore(root).list() == []


def test_malformed_records_skipped_on_load(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    good = {"id": "good", "describe": "", "width": 1, "height": 1,
            "size_bytes": 1, "created_at": "2025-01-01T00:00:00+00:00"}
    (root / "meta.json").write_text(
        json.dumps({"good": good, "str": "oops", "noid": {"width": 2}}), encoding="utf-8"
    )
    loaded = SucaiStore(root)
    assert [r["id"] for r in loaded.list()] == ["good"]
    # and saving after load must not resurrect the malformed entries
    loaded.create(_png(), sid="fresh")
    reloaded = SucaiStore(root)
    assert {r["id"] for r in reloaded.list()} == {"good", "fresh"}


def test_list_ordering_newest_first_with_stable_tiebreak(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()

    def record(sid, created):
        return {"id": sid, "describe": "", "width": 1, "height": 1,
                "size_bytes": 1, "created_at": created}

    (root / "meta.json").write_text(json.dumps({
        "old": record("old", "2024-01-01T00:00:00+00:00"),
        "new": record("new", "2025-06-01T00:00:00+00:00"),
        "mid-b": record("mid-b", "2025-01-01T00:00:00+00:00"),
        "mid-a": record("mid-a", "2025-01-01T00:00:00+00:00"),
    }), encoding="utf-8")
    assert [r["id"] for r in SucaiStore(root).list()] == ["new", "mid-b", "mid-a", "old"]


def test_max_side_constant_matches_guard():
    assert ID_PATTERN.match("a" * 64)
    assert not ID_PATTERN.match("a" * 65)


# ------------------------------------------------------------ category ---
def test_create_stores_stripped_category(store):
    record = store.create(_png(), category="  按钮  ")
    assert record["category"] == "按钮"


def test_create_rejects_over_long_category(store):
    with pytest.raises(SucaiError):
        store.create(_png(), category="x" * 33)


def test_update_category_semantics(store):
    store.create(_png(), sid="c1", category="按钮")
    # None = leave unchanged (needs another field to make the call a real op)
    assert store.update("c1", describe="x", category=None)["category"] == "按钮"
    assert store.update("c1", category="图标")["category"] == "图标"
    assert store.update("c1", category="")["category"] == ""  # cleared


def test_categories_distinct_sorted_and_filter(store):
    store.create(_png(seed=1), sid="a", category="图标")
    store.create(_png(seed=2), sid="b", category="按钮")
    store.create(_png(seed=3), sid="c", category="图标")
    store.create(_png(seed=4), sid="d")
    assert store.categories() == sorted({"图标", "按钮"})
    # same-second creates tie-break by id descending (newest first)
    assert sorted(i["id"] for i in store.list(category="图标")) == ["a", "c"]
    assert [i["id"] for i in store.list(category="")] == ["d"]
    assert len(store.list()) == 4  # no filter
