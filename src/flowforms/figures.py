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


import pyvista as pv  # noqa: E402

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _field_cmap(field: str):
    """Return the colormap for this field name, delegating to brand."""
    return brand.field_cmap_for_name(field)


def slice_still(frame: Frame, field: str, *, axis: str = "z", index=None, ax=None):
    """Render a 2D slice of a grid scalar field with matplotlib."""
    brand.apply_figure_style()
    if frame.kind != "grid":
        raise ValueError("slice_still requires a grid frame")
    f = frame.derive(field)
    dims = np.array(f.mesh.dimensions)  # (nx, ny, nz)
    nx, ny, nz = (int(d) for d in dims)
    data = f.array(field)
    if data.ndim > 1:
        data = np.linalg.norm(data, axis=1)
    vol = np.transpose(data.reshape(nz, ny, nx), (2, 1, 0))  # (nx,ny,nz)
    a = _AXIS_INDEX[axis]
    if index is None:
        index = vol.shape[a] // 2
    plane = np.take(vol, index, axis=a)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
    else:
        fig = ax.figure
    cmap = _field_cmap(field)
    vmax = float(np.abs(plane).max()) or 1.0
    # Use symmetric limits for signed (diverging) fields.
    signed_substrings = ("vorticity", "vort", "pressure", "helicity")
    is_signed = any(s in field for s in signed_substrings)
    if is_signed:
        im = ax.pcolormesh(plane.T, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    else:
        im = ax.pcolormesh(plane.T, cmap=cmap, shading="auto")
    ax.set_aspect("equal")
    ax.set_title(field)
    fig.colorbar(im, ax=ax, label=field)
    fig.tight_layout()
    return fig


def _offscreen_plotter() -> pv.Plotter:
    pv.OFF_SCREEN = True
    return pv.Plotter(off_screen=True, theme=brand.figure_pv_theme())


def isosurface_still(frame: Frame, field: str, value: float, *, path) -> Path:
    """Render a single isosurface of a grid scalar field, off-screen.

    Works for both native point-data fields and derived/cell-data fields.
    If the field is not in point_data after derivation (e.g. it landed in
    cell_data), we promote via cell_data_to_point_data before contouring.
    """
    from typing import cast as _cast
    f = frame.derive(field)
    mesh = f.mesh
    if field not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()
    contour = _cast(pv.PolyData, mesh.contour([value], scalars=field))
    pl = _offscreen_plotter()
    pl.add_mesh(contour, cmap=_field_cmap(field), smooth_shading=True)
    pl.camera_position = "iso"
    out = Path(path)
    pl.screenshot(str(out))
    pl.close()
    return out


def surface_still(frame: Frame, scalar: str, *, path) -> Path:
    """Render an on-surface scalar map for a .vtp manifold, off-screen."""
    pl = _offscreen_plotter()
    pl.add_mesh(frame.mesh, scalars=scalar, cmap=brand.field_cmap("magnitude"),
                smooth_shading=True, show_edges=False)
    pl.camera_position = "iso"
    out = Path(path)
    pl.screenshot(str(out))
    pl.close()
    return out
