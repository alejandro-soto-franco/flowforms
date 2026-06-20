"""Track A: publication figures (matplotlib) on the flowforms brand."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import brand
from .diagnostics import Diagnostics
from .io import Frame
from .spectrum import energy_spectrum

_LABELS = {
    "energy": r"$E(t)$",
    "enstrophy": r"$\mathcal{Z}(t)$",
    "helicity": r"$H(t)$",
    "max_vorticity": r"$\|\omega\|_\infty$",
}


def decay_plot(diag: Diagnostics, quantities=("energy", "enstrophy"), *, ax=None):
    """Plot scalar diagnostics against time on one axis (twin y if needed)."""
    brand.apply_figure_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.figure
    t = diag.time
    for q in quantities:
        ax.plot(t, diag.column(q), label=_LABELS.get(q, q))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("diagnostic")
    ax.legend()
    fig.tight_layout()
    return fig


def spectrum_plot(frame: Frame, *, guide: bool = True, ax=None):
    """Plot E(k) log-log with an optional k^{-5/3} guide line."""
    brand.apply_figure_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.figure
    k, ek = energy_spectrum(frame)
    mask = (k > 0) & (ek > 0)
    ax.loglog(k[mask], ek[mask], marker="o", ms=3, label=r"$E(k)$")
    if guide and mask.sum() > 2:
        kk = k[mask]
        anchor = ek[mask][1] * (kk / kk[1]) ** (-5.0 / 3.0)
        ax.loglog(kk, anchor, ls="--", color=brand.PALETTE["muted"],
                  label=r"$k^{-5/3}$")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.legend()
    fig.tight_layout()
    return fig


def save(fig, path: str | Path) -> None:
    """Save a figure as PNG plus matching PDF and SVG vector outputs."""
    path = Path(path)
    fig.savefig(path)
    for ext in (".pdf", ".svg"):
        fig.savefig(path.with_suffix(ext))
