import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import pyvista as pv
import flowforms.io as fio
import flowforms.figures as figures
from tests.test_spectrum import _single_mode_frame


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
