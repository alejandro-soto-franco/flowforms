"""Brand chrome overlay: title, handle, and caption on a rendered frame."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import brand


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def add_chrome(img: np.ndarray, *, title=None, handle=None, caption=None) -> np.ndarray:
    if not any((title, handle, caption)):
        return img
    pil = Image.fromarray(np.asarray(img)[..., :3].astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    if title:
        draw.text((int(0.04 * w), int(0.04 * h)), title,
                  fill=brand.PALETTE["paper"], font=_font(max(18, w // 30)))
    if caption:
        draw.text((int(0.04 * w), int(0.90 * h)), caption,
                  fill=brand.PALETTE["gold"], font=_font(max(12, w // 55)))
    if handle:
        draw.text((int(0.80 * w), int(0.95 * h)), handle,
                  fill=brand.PALETTE["muted"], font=_font(max(12, w // 60)))
    return np.asarray(pil)
