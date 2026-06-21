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
                 size_px=(1080, 360), xlim=None, ylim=None) -> np.ndarray:
    """Render a GROWING diagnostic curve: only data with ``t <= t_now``.

    The line grows rightward as t_now advances (xlim right edge = t_now, which
    is "now"; no vertical cursor and no full faded curve). The y-limits are
    FIXED to the full-series range so the axis does not jump while the line
    grows. Dark house style; axis labels in CM math, title/ticks in CM Sans.

    Pixel dimensions are deterministic and independent of content: the figure
    is created at exactly size_px and the axes are placed with fixed margins
    (no tight_layout). The result is cropped/padded to exactly (h, w, 3) so
    every call for a given size_px returns identical dimensions -- required so
    stacked composite frames stay uniform for h264 encoding.

    xlim: if None, use (data t.min(), t_now). If (t0, t1) is passed, the right
    edge still tracks t_now -> (t0, t_now). ylim: if None, the full data range
    of ``quantity``.
    """
    brand.apply_figure_style(dark=True)
    dpi = 100
    w, h = int(size_px[0]), int(size_px[1])
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    # Fixed axes box in figure fractions -- constant regardless of content.
    ax = fig.add_axes((0.12, 0.22, 0.84, 0.70))
    t = diag.time
    y = diag.column(quantity)
    t_now = float(t_now)
    # GROW: only data up to now. Guard the degenerate single-point case so the
    # line is still visible at the very first frame.
    mask = t <= t_now
    if mask.sum() >= 1:
        ax.plot(t[mask], y[mask], color=brand.PALETTE["blue"], lw=2.0)
    t0 = float(xlim[0]) if xlim is not None else float(t.min())
    ax.set_xlim(t0, t_now if t_now > t0 else t0 + 1e-9)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        ymin, ymax = float(np.min(y)), float(np.max(y))
        if ymax <= ymin:
            ymax = ymin + 1e-9
        pad = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.xaxis.set_major_formatter(brand.sans_tick_formatter())
    ax.yaxis.set_major_formatter(brand.sans_tick_formatter())
    ax.tick_params(labelsize=13)
    ax.set_xlabel(r"$t$", fontsize=19)
    ax.set_ylabel(_LABELS.get(quantity, quantity), fontsize=19)
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
from . import cine as _cine
from . import camera as _camera
from . import chrome as _chrome
from .scene import Scene


# Frame normalization lives in cine (single source of truth, avoids a circular
# import the other way) and is re-exported here for convenience.
normalize_frames = _cine.normalize_frames


def render_composite_animation(series, diag: Diagnostics, scene: Scene, *,
                               quantity: str = "enstrophy", out, fps: int = 30,
                               top_size=(1080, 810), plot_size=(1080, 360),
                               layout: str = "stacked", orbit: bool = True,
                               formats=("mp4", "webm"), title=None):
    """Render the cinematic composite: 3D render over a GROWING rolling plot.

    The growing rolling plot conveys time, so there is no per-frame time
    readout and no yellow commentary. Only a subtle CM Sans title is overlaid
    via chrome.
    """
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
    # Grow from the series start; the right edge tracks t_now per frame.
    t0 = float(times[0]) if len(times) else None
    frames_rgb = []
    # Streamline cache: recompute tubes every update_every frames; hold in between.
    tubes_cache = None
    update_every = max(1, scene.streamlines.update_every) if scene.streamlines.enabled else 1
    for i in range(n):
        cam = positions[i] if positions is not None else None
        if scene.streamlines.enabled:
            if tubes_cache is None or i % update_every == 0:
                tubes_cache = _cine.streamline_tubes(series[i], scene)
            st_arg = tubes_cache
        else:
            st_arg = None
        top = _cine.render_scene(series[i], scene, size=top_size, camera_position=cam,
                                 streamline_tubes=st_arg)
        bottom = rolling_plot(diag, quantity, float(times[i]),
                              size_px=plot_size,
                              xlim=(t0, None) if t0 is not None else None)
        frame = stack(top, bottom, layout=layout)
        frame = _chrome.add_chrome(frame, title=title)
        frames_rgb.append(frame)
    frames_rgb = normalize_frames(frames_rgb)
    written = []
    for fmt in formats:
        path = out.with_suffix(f".{fmt}")
        _cine._imwrite_video(path, frames_rgb, fps=fps)
        written.append(path)
    return written
