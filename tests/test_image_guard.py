"""P0.4/P0.9 — image validation: valid, oversized, spoofed, malformed, dims."""
import io

import pytest
from PIL import Image

from src.image_guard import ImageRejected, validate_image

KW = dict(max_bytes=5 * 1024 * 1024, max_dim=4096, max_pixels=16_777_216,
          provider_long_side=1600)


def _png(w=200, h=150, color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_image():
    out = validate_image(_png(), **KW)
    assert out.mime in ("image/png", "image/jpeg")
    assert out.width > 0 and out.height > 0 and len(out.data) > 0


def test_oversized_bytes_rejected_413():
    with pytest.raises(ImageRejected) as ei:
        validate_image(b"\x89PNG" + b"\x00" * 10, max_bytes=8, **{k: v for k, v in KW.items() if k != "max_bytes"})
    assert ei.value.status == 413


def test_invalid_mime_rejected():
    with pytest.raises(ImageRejected) as ei:
        validate_image(b"this is plain text, not an image", **KW)
    assert ei.value.code == "invalid_image" and ei.value.status == 400


def test_spoofed_extension_rejected():
    """EXE bytes masquerading as an image must fail magic-byte validation."""
    with pytest.raises(ImageRejected) as ei:
        validate_image(b"MZ\x90\x00" + b"\x00" * 100, **KW)
    assert ei.value.code == "invalid_image"


def test_malformed_truncated_image_rejected():
    full = _png()
    with pytest.raises(ImageRejected) as ei:
        validate_image(full[:40], **KW)  # header only, truncated body
    assert ei.value.code == "invalid_image"


def test_oversized_dimensions_rejected():
    with pytest.raises(ImageRejected) as ei:
        validate_image(_png(w=100, h=100), max_dim=64, **{k: v for k, v in KW.items() if k != "max_dim"})
    assert ei.value.code == "image_dimensions_rejected"


def test_decompression_bomb_pixel_cap():
    with pytest.raises(ImageRejected) as ei:
        validate_image(_png(w=400, h=400), max_pixels=1000,
                       **{k: v for k, v in KW.items() if k != "max_pixels"})
    assert ei.value.code == "image_dimensions_rejected"


def test_disallowed_format_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), (0, 0, 255)).save(buf, format="GIF")
    with pytest.raises(ImageRejected) as ei:
        validate_image(buf.getvalue(), **KW)
    assert ei.value.code in ("invalid_image", "unsupported_format")
