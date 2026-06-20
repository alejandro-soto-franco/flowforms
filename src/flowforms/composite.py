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


def rolling_plot(diag: Diagnostics, quantity: str, t_now: float, *,
                 size_px=(1080, 450), annotations=(), ylabel=None) -> np.ndarray:
    """Render a diagnostic curve up to t_now with a cursor and annotations."""
    brand.apply_figure_style()
    dpi = 100
    fig, ax = plt.subplots(figsize=(size_px[0] / dpi, size_px[1] / dpi), dpi=dpi)
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
    ax.set_xlim(t.min(), t.max())
    fig.tight_layout()
    return _fig_to_rgb(fig)


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
