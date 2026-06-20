import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import pyvista as pv
import flowforms.io as fio
import flowforms.figures as figures
from flowforms.io import Frame
from tests.test_spectrum import _single_mode_frame


def _vortical_frame(n=24) -> Frame:
    """A grid frame with non-trivial (non-zero, non-uniform) omega_mag."""
    L = 2 * np.pi
    xs = np.linspace(0, L, n, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    vel = np.zeros((n, n, n, 3))
    vel[..., 0] = np.sin(Y)
    vel[..., 1] = np.cos(X)
    vel[..., 2] = np.sin(Z)
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(L / n,) * 3, origin=(0, 0, 0))
    flat = np.transpose(vel, (2, 1, 0, 3)).reshape(-1, 3)
    grid["velocity"] = flat
    return Frame(grid)


def _has_gl() -> bool:
    try:
        pv.OFF_SCREEN = True
        p = pv.Plotter(off_screen=True)
        p.add_mesh(pv.Sphere())
        p.screenshot(return_img=True)
        p.close()
        return True
    except Exception:
        return False


def test_slice_still_matplotlib():
    frame = _single_mode_frame(n=16, k0=2).derive("velocity_mag")
    fig = figures.slice_still(frame, "velocity_mag", axis="z", index=8)
    assert fig.axes  # has at least one axis
    # colorbar present
    assert any("Colorbar" in type(a).__name__ or a.get_label() == "<colorbar>"
               for a in fig.axes) or len(fig.axes) >= 2


@pytest.mark.skipif(not _has_gl(), reason="no GL/xvfb for off-screen render")
def test_isosurface_still(tmp_path):
    frame = _single_mode_frame(n=24, k0=2).derive("velocity_mag")
    out = figures.isosurface_still(frame, "velocity_mag", 0.5, path=tmp_path / "iso.png")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(not _has_gl(), reason="no GL/xvfb for off-screen render")
def test_surface_still(tmp_path):
    frame = fio.load("tests/fixtures/surface.vtp")
    out = figures.surface_still(frame, "temp", path=tmp_path / "surf.png")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(not _has_gl(), reason="no GL/xvfb for off-screen render")
def test_isosurface_still_omega_mag(tmp_path):
    """C1: isosurface_still must work for omega_mag (derived scalar) on a grid frame.

    The single-mode frame has zero curl, so we use a vortical frame with u=sin(y),
    v=cos(x), w=sin(z) which produces a spatially varying omega_mag with a
    well-populated 50th-percentile isosurface.
    """
    frame = _vortical_frame(n=24)
    derived = frame.derive("omega_mag")
    arr = derived.array("omega_mag")
    # Pick median value: guaranteed to produce a non-empty contour surface.
    val = float(np.percentile(arr, 50))
    out = figures.isosurface_still(frame, "omega_mag", val, path=tmp_path / "iso_omega.png")
    assert out.exists() and out.stat().st_size > 0
