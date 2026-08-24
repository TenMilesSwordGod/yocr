import base64

import numpy as np
import pytest

from yocr.imaging import decode_base64_image, decode_image, encode_png, ensure_bgr


def make_png(tmp_path_factory) -> bytes:
    from PIL import Image

    arr = np.zeros((60, 80, 3), dtype=np.uint8)
    arr[:, :, 0] = 200
    path = tmp_path_factory.mktemp("img") / "t.png"
    Image.fromarray(arr).save(path)
    return path.read_bytes()


def test_decode_png_roundtrip(tmp_path_factory):
    data = make_png(tmp_path_factory)
    img = decode_image(data)
    assert img.shape == (60, 80, 3)
    assert img[10, 10, 2] == 200

    encoded = encode_png(img)
    assert decode_image(encoded).shape == img.shape


def test_decode_base64(tmp_path_factory):
    data = make_png(tmp_path_factory)
    b64 = base64.b64encode(data).decode()
    assert decode_base64_image(b64).shape == (60, 80, 3)


def test_corrupt_payload_raises():
    with pytest.raises(ValueError):
        decode_image(b"not-an-image")
    with pytest.raises(ValueError):
        decode_image(b"")
    with pytest.raises(ValueError):
        decode_base64_image("!!!notb64!!!")


def test_crlf_mangled_png_recovered(tmp_path_factory):
    """Simulate a transport layer that replaced \\n with \\r\\n in the PNG bytes."""
    data = bytearray(make_png(tmp_path_factory))
    mangled = bytes(data.replace(b"\x0a", b"\x0d\x0a"))
    assert decode_image(mangled).shape == (60, 80, 3)


@pytest.mark.parametrize("channels", [1, 4])
def test_ensure_bgr(channels):
    shape = (32, 32) if channels == 1 else (32, 32, channels)
    out = ensure_bgr(np.zeros(shape, dtype=np.uint8))
    assert out.ndim == 3 and out.shape[2] == 3
