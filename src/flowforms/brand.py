"""Visual brand for flowforms: palette, colormaps, fonts, and themes.

One palette serves both the figure profile (light background, referee-proof)
and the cinema profile (dark background, emissive). Field colormaps are
perceptually uniform and colorblind-safe: diverging for signed fields
(vorticity, pressure) and sequential for magnitudes (speed, enstrophy, Q).
"""
from __future__ import annotations
import shutil
from functools import lru_cache
from typing import Any

import cmocean
import matplotlib as mpl
import pyvista as pv

PALETTE: dict[str, str] = {
    "ink": "#14171C",       # near-black text on light figures
    "paper": "#FAFAFA",     # light figure background
    "grid_bg": "#FFFFFF",   # axes face for figures
    "cine_bg": "#0B0E14",   # dark cinematic background
    "gold": "#E8B04B",
    "rust": "#C1440E",
    "blue": "#1F6FEB",
    "muted": "#8A93A3",
}

# Ordered accent cycle for multi-series plots (CVD-safe).
ACCENT_CYCLE: list[str] = [PALETTE["blue"], PALETTE["rust"], PALETTE["gold"], PALETTE["muted"]]

_SEQUENTIAL: Any = getattr(cmocean.cm, "thermal")
_DIVERGING: Any = getattr(cmocean.cm, "balance")


def field_cmap(kind: str) -> Any:
    """Return a colormap appropriate for a field kind.

    Signed fields (vorticity, pressure, helicity) get a diverging map;
    everything else (magnitudes, Q, enstrophy) gets a sequential map.
    """
    signed = {"vorticity", "pressure", "helicity", "diverging"}
    return _DIVERGING if kind in signed else _SEQUENTIAL


def field_cmap_for_name(name: str) -> Any:
    """Return the appropriate colormap for a field identified by name.

    Classifies by substring: fields whose name contains 'vorticity', 'vort',
    'pressure', or 'helicity' are treated as signed (diverging map); all
    others (magnitudes, Q, enstrophy, etc.) get the sequential map. This is
    the single source of truth used by both figures._field_kind and cine's
    inline slice-field classification.
    """
    signed_substrings = ("vorticity", "vort", "pressure", "helicity")
    if any(s in name for s in signed_substrings):
        return _DIVERGING
    return _SEQUENTIAL


@lru_cache(maxsize=1)
def latex_available() -> bool:
    """True if a system LaTeX is on PATH (so usetex is safe)."""
    return shutil.which("latex") is not None


def apply_figure_style() -> None:
    """Apply the figure (light, referee-proof) matplotlib style globally."""
    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["grid_bg"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "axes.grid": True,
            "grid.color": "#DfE3E8",
            "grid.linewidth": 0.6,
            "axes.prop_cycle": getattr(mpl, "cycler")(color=ACCENT_CYCLE),
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "text.usetex": latex_available(),
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.frameon": False,
        }
    )


def figure_pv_theme() -> Any:
    """A PyVista theme for light-background publication stills."""
    theme = getattr(pv.themes, "DocumentTheme")()
    theme.background = PALETTE["paper"]
    theme.font.color = PALETTE["ink"]
    theme.cmap = "viridis"
    theme.transparent_background = False
    return theme


# Emissive colormap for volume glow (dark to hot, reads as light emission).
EMISSIVE_CMAP: Any = getattr(cmocean.cm, "thermal")


def cinema_pv_theme() -> Any:
    """A PyVista theme for dark, emissive cinematic renders."""
    theme = getattr(pv.themes, "DarkTheme")()
    theme.background = PALETTE["cine_bg"]
    theme.font.color = PALETTE["paper"]
    theme.cmap = "magma"
    theme.transparent_background = False
    return theme
