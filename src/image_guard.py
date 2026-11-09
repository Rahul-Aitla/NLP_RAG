"""P0.4 — server-side image validation. Never trusts browser accept=/MIME."""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
FORMAT_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class ImageRejected(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class ValidatedImage:
    data: bytes          # re-encoded, optionally downscaled, safe bytes
    mime: str
    width: int
    height: int


def validate_image(
    raw: bytes,
    *,
    max_bytes: int,
    max_dim: int,
    max_pixels: int,
    provider_long_side: int,
) -> ValidatedImage:
    """Validate magic bytes, format, dimensions; return safe re-encoded bytes.

    Raises ImageRejected (safe message, no internals) for: empty, oversized,
    unidentifiable/spoofed, disallowed format, oversized dimensions, or
    decompression-bomb pixel counts. Re-encoding strips malicious metadata
    and normalizes spoofed extensions.
    """
    if not raw:
        raise ImageRejected("empty_image", "No image data received.")
    if len(raw) > max_bytes:
        raise ImageRejected(
            "image_too_large",
            f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit.",
            status=413,
        )
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()  # magic-byte + structural check, no pixel decode
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise ImageRejected(
            "invalid_image", "Uploaded file is not a valid image."
        ) from None

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # full decode inside try: catches truncated/malformed
    except Exception:
        raise ImageRejected(
            "invalid_image", "Uploaded file is not a valid image."
        ) from None

    fmt = (img.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ImageRejected(
            "unsupported_format",
            "Only JPEG, PNG, or WEBP images are accepted.",
        )
    w, h = img.size
    if w <= 0 or h <= 0 or max(w, h) > max_dim or (w * h) > max_pixels:
        raise ImageRejected(
            "image_dimensions_rejected",
            "Image dimensions are outside acceptable limits.",
        )
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    # Bound provider payload: downscale longest side, re-encode to strip
    # metadata / normalize spoofed content.
    longest = max(w, h)
    if longest > provider_long_side:
        scale = provider_long_side / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt != "PNG" else "PNG"
    img.save(buf, format=save_fmt, quality=85, optimize=True)
    out = buf.getvalue()
    if len(out) > max_bytes:
        raise ImageRejected(
            "image_too_large",
            "Processed image exceeds the size limit.",
            status=413,
        )
    return ValidatedImage(
        data=out,
        mime=FORMAT_MIME[save_fmt],
        width=img.size[0],
        height=img.size[1],
    )
