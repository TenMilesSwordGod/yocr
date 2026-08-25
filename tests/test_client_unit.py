"""Unit tests for the reusable client in examples/yocr_client.py."""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from yocr_client import YocrClient  # noqa: E402


def _client_with(handler) -> YocrClient:
    client = YocrClient("http://test/api/v1")
    client.session = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    return client


def test_http_error_includes_server_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "model 'x' weights not provisioned locally"})

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _client_with(handler).analyze(b"png")
    assert "weights not provisioned locally" in str(excinfo.value)


def test_find_icon_accepts_single_string_label(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["model"] == "iconfinder"
        body = {
            "model": "IconFinder", "image": {"width": 10, "height": 10},
            "elements": [], "found": False, "matched": None,
            "lines": [], "full_text": "", "timing": {"total_ms": 1.0},
        }
        return httpx.Response(200, json=body)

    result = _client_with(handler).find_icon(b"png", labels="a gear settings icon", model="iconfinder")
    assert result is None


def test_match_and_locate_template_roundtrip():
    box = {"xyxy": [10, 20, 30, 40], "xywh": [10, 20, 20, 20], "center": [20, 30]}

    def handler(request: httpx.Request) -> httpx.Response:
        form = request.read()  # multipart body sanity: both parts present
        assert b'name="file"' in form and b'name="template"' in form
        return httpx.Response(200, json={
            "found": True, "score": 0.9, "threshold": 0.8, "scale": 1.0,
            "box": box, "image": {"width": 100, "height": 100},
            "template": {"width": 20, "height": 20}, "timing": {"total_ms": 1.0},
        })

    c = _client_with(handler)
    assert c.locate_template(b"tpl", b"scene") == (20, 30)
