"""Visual brand for flowforms: palette, colormaps, fonts, and themes.

One palette serves both the figure profile (light background, referee-proof)
and the cinema profile (dark background, emissive). Field colormaps are
perceptually uniform and colorblind-safe: diverging for signed fields
(vorticity, pressure) and sequential for magnitudes (speed, enstrophy, Q).

Typography follows the house ("polybius") style: under usetex, axis labels
(math symbols) render in Computer Modern math while every upright text
element (title, ticks, legend) renders in Computer Modern Sans. This split is
achieved with ``\\renewcommand{\\familydefault}{\\sfdefault}`` (NOT sansmath,
which would force math into sans too). A no-LaTeX fallback keeps the same
math-CM / text-sans split via mathtext.fontset="cm" and a CM Sans family.
"""
from __future__ import annotations
import os
import shutil
from functools import lru_cache
from typing import Any

import cmocean
import matplotlib as mpl
import pyvista as pv

PALETTE: dict[str, str] = {
    "ink": "#171717",       # near-black text on light figures (polybius)
    "paper": "#ffffff",     # light figure background (polybius)
    "grid_bg": "#ffffff",   # axes face for figures
    "cine_bg": "#0a0a0a",   # dark cinematic background (polybius)
    "text_light": "#ededed",  # light text on dark theme (polybius)
    "edge_dark": "#666666",   # axes edge on dark theme (polybius)
    "edge_light": "#333333",  # axes edge on light theme (polybius)
    "gold": "#E8B04B",      # non-text accent only
    "rust": "#C1440E",
    "blue": "#627eea",      # bright blue readable on dark bg (polybius)
    "muted": "#8A93A3",
}

# Ordered accent cycle for multi-series plots (CVD-safe).
ACCENT_CYCLE: list[str] = [PALETTE["blue"], PALETTE["rust"], PALETTE["gold"], PALETTE["muted"]]

_SEQUENTIAL: Any = getattr(cmocean.cm, "thermal")
_DIVERGING: Any = getattr(cmocean.cm, "balance")

# LaTeX preamble: base packages plus the sans-default switch that gives the
# requested math-CM / text-sans split (upright text -> Latin Modern Sans).
_LATEX_PREAMBLE = (
    r"\usepackage{lmodern}"
    r"\usepackage{amsmath}"
    r"\usepackage{amssymb}"
    r"\renewcommand{\familydefault}{\sfdefault}"
)


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
    """True if a system LaTeX is on PATH (so usetex is *worth attempting*)."""
    return shutil.which("latex") is not None


def cm_sans_font_file() -> str:
    """Return a path to a Computer Modern Sans TTF (cmss10) for VTK/PyVista.

    matplotlib bundles cmss10.ttf under its mpl-data/fonts/ttf. We compute the
    path from matplotlib.__file__ rather than hard-coding it. Rev-2 uses this
    for the 3D scalar bar so the bar font matches the matplotlib sans text.
    """
    base = os.path.dirname(mpl.__file__)
    path = os.path.join(base, "mpl-data", "fonts", "ttf", "cmss10.ttf")
    assert os.path.exists(path), f"cmss10.ttf not found at {path}"
    return path


@lru_cache(maxsize=1)
def _usetex_renders() -> bool:
    """Probe whether usetex can actually render a tiny figure with a math label.

    latex_available() only checks PATH; a broken TeX install or a missing
    package would still raise at draw time. We render a 1-inch figure with a
    math axis label under usetex and return True only if it succeeds. The
    result is cached so the probe runs once per process.
    """
    if not latex_available():
        return False
    import matplotlib.pyplot as plt
    saved = dict(mpl.rcParams)
    try:
        mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "text.latex.preamble": _LATEX_PREAMBLE,
            }
        )
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_subplot(111)
        ax.set_xlabel(r"$\mathcal{Z}(t)$")
        ax.set_ylabel(r"$t$")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False
    finally:
        mpl.rcParams.update(saved)


def apply_figure_style(dark: bool = False) -> None:
    """Apply the house matplotlib style globally.

    Light (publication) by default; ``dark=True`` applies the cinematic dark
    theme used for the composite/rolling plot so it blends with the dark 3D
    render. The font scheme gives axis math symbols in Computer Modern math and
    all upright text (title, ticks, legend) in Computer Modern Sans.

    If usetex is available *and* a probe figure renders, we use usetex with the
    sans-default preamble. Otherwise we fall back to usetex=False with a CM Sans
    family and mathtext.fontset="cm" so the math-CM / text-sans split survives.
    """
    if dark:
        bg = PALETTE["cine_bg"]
        text = PALETTE["text_light"]
        edge = PALETTE["edge_dark"]
    else:
        bg = PALETTE["paper"]
        text = PALETTE["ink"]
        edge = PALETTE["edge_light"]

    common = {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "axes.edgecolor": edge,
        "axes.labelcolor": text,
        "text.color": text,
        "xtick.color": text,
        "ytick.color": text,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": getattr(mpl, "cycler")(color=ACCENT_CYCLE),
        "lines.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
    }

    if _usetex_renders():
        common.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "text.latex.preamble": _LATEX_PREAMBLE,
            }
        )
    else:
        common.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "CMU Sans Serif",
                    "Latin Modern Sans",
                    "cmss10",
                    "DejaVu Sans",
                ],
                "mathtext.fontset": "cm",
            }
        )
    mpl.rcParams.update(common)


def sans_tick_formatter():
    """A tick formatter wrapping numeric tick labels so digits render sans.

    Under usetex, bare numeric ticks render as Computer Modern *math* digits;
    wrapping them in ``\\mathsf{...}`` keeps tick digits in the sans family
    (matching the requested title/tick sans scheme) while axis labels stay CM
    math. Safe with usetex=False too (mathtext understands \\mathsf).
    """
    from matplotlib.ticker import FuncFormatter

    def _fmt(value: float, _pos: int) -> str:
        if value == int(value):
            s = f"{int(value)}"
        else:
            s = f"{value:g}"
        return rf"$\mathsf{{{s}}}$"

    return FuncFormatter(_fmt)


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
    theme.font.color = PALETTE["text_light"]
    theme.cmap = "magma"
    theme.transparent_background = False
    return theme
