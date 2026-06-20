"""Track C: composite (field over rolling time-series plot) and hero assembly."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import brand
from .diagnostics import Diagnostics

_LABELS = {"enstrophy": r"$\mathcal{Z}(t)$", "energy": r"$E(t)$", "helicity": r"$H(t)$"}


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def _crop_or_pad(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Force an RGB array to exactly (h, w, 3) by cropping and/or padding."""
    img = np.asarray(img)[..., :3]
    ch, cw = img.shape[0], img.shape[1]
    # Crop if larger.
    img = img[: min(ch, h), : min(cw, w), :]
    ch, cw = img.shape[0], img.shape[1]
    if ch == h and cw == w:
        return img
    out = np.zeros((h, w, 3), dtype=img.dtype)
    out[:ch, :cw, :] = img
    return out


def rolling_plot(diag: Diagnostics, quantity: str, t_now: float, *,
                 size_px=(1080, 450), annotations=(), ylabel=None,
                 xlim=None) -> np.ndarray:
    """Render a diagnostic curve up to t_now with a cursor and annotations.

    Pixel dimensions are deterministic and independent of content: the figure
    is created at exactly size_px and the axes are placed with fixed margins
    (no tight_layout, which would reflow when annotations appear). The result
    is cropped/padded to exactly (h, w, 3) so every call for a given size_px
    returns identical dimensions -- required so stacked composite frames stay
    uniform for h264 encoding.
    """
    brand.apply_figure_style()
    dpi = 100
    w, h = int(size_px[0]), int(size_px[1])
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    # Fixed axes box in figure fractions -- constant regardless of content.
    ax = fig.add_axes((0.12, 0.22, 0.84, 0.70))
    t = diag.time
    y = diag.column(quantity)
    ax.plot(t, y, color=brand.PALETTE["muted"], lw=1.0, alpha=0.4)
    mask = t <= t_now
    ax.plot(t[mask], y[mask], color=brand.PALETTE["blue"], lw=2.2)
    ax.axvline(t_now, color=brand.PALETTE["rust"], lw=1.5)
    for (at_t, text) in annotations:
        if t_now >= at_t:
            ax.annotate(text, xy=(at_t, np.interp(at_t, t, y)),
                        xytext=(8, 8), textcoords="offset points",
                        color=brand.PALETTE["gold"], fontsize=11)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel or _LABELS.get(quantity, quantity))
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        ax.set_xlim(t.min(), t.max())
    img = _fig_to_rgb(fig)
    return _crop_or_pad(img, h, w)


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((w, h)))


def stack(top: np.ndarray, bottom: np.ndarray, *, layout: str = "stacked", bg=None) -> np.ndarray:
    top = np.asarray(top)[..., :3]
    bottom = np.asarray(bottom)[..., :3]
    if layout == "stacked":
        w = top.shape[1]
        bh = round(bottom.shape[0] * w / bottom.shape[1])
        bottom_r = _resize(bottom, w, bh)
        return np.vstack([top, bottom_r])
    if layout == "side_by_side":
        h = top.shape[0]
        bw = round(bottom.shape[1] * h / bottom.shape[0])
        bottom_r = _resize(bottom, bw, h)
        return np.hstack([top, bottom_r])
    if layout == "pip":
        out = top.copy()
        ph = top.shape[0] // 3
        pw = round(bottom.shape[1] * ph / bottom.shape[0])
        inset = _resize(bottom, pw, ph)
        out[-ph:, -pw:] = inset
        return out
    raise ValueError(f"unknown layout: {layout!r}")


import imageio.v3 as iio
from pathlib import Path
from PIL import ImageDraw, ImageFont
from . import cine as _cine
from . import camera as _camera
from . import chrome as _chrome
from .scene import Scene


# Frame normalization lives in cine (single source of truth, avoids a circular
# import the other way) and is re-exported here for convenience.
normalize_frames = _cine.normalize_frames


def _time_readout(img: np.ndarray, text: str, *, panel_h: int | None = None) -> np.ndarray:
    """Overlay a small time string at the top-right of the top panel."""
    pil = Image.fromarray(np.asarray(img)[..., :3].astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, w // 45))
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
    except Exception:
        tw = len(text) * 8
    x = w - tw - int(0.03 * w)
    y = int(0.03 * h)
    draw.text((x, y), text, fill=brand.PALETTE["paper"], font=font)
    return np.asarray(pil)


def render_composite_animation(series, diag: Diagnostics, scene: Scene, *,
                               quantity: str = "enstrophy", out, fps: int = 30,
                               top_size=(1080, 810), plot_size=(1080, 360),
                               layout: str = "stacked", orbit: bool = True,
                               annotations=(), formats=("mp4", "webm"),
                               title=None, handle=None, caption=None,
                               time_readout: bool = True):
    n = len(series)
    if n == 0:
        raise ValueError("empty series; nothing to render")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    positions = None
    if orbit:
        center, radius = _camera.bounds_center_radius(series[0].mesh)
        positions = _camera.orbit_positions(center, radius, n)
    times = np.asarray(series.times)
    xlim = (float(times[0]), float(times[-1])) if len(times) else None
    frames_rgb = []
    for i in range(n):
        cam = positions[i] if positions is not None else None
        top = _cine.render_scene(series[i], scene, size=top_size, camera_position=cam)
        bottom = rolling_plot(diag, quantity, float(times[i]),
                              size_px=plot_size, annotations=annotations, xlim=xlim)
        frame = stack(top, bottom, layout=layout)
        frame = _chrome.add_chrome(frame, title=title, handle=handle, caption=caption)
        if time_readout:
            frame = _time_readout(frame, f"t = {float(times[i]):.2f}")
        frames_rgb.append(frame)
    frames_rgb = normalize_frames(frames_rgb)
    written = []
    for fmt in formats:
        path = out.with_suffix(f".{fmt}")
        _cine._imwrite_video(path, frames_rgb, fps=fps)
        written.append(path)
    return written
