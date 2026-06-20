import numpy as np
import pytest
import pyvista as pv
from flowforms.scene import Scene
import flowforms.cine as cine
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


pytestmark = pytest.mark.skipif(not _has_gl(), reason="no GL/xvfb for off-screen render")


def test_render_grid_scene_returns_image():
    frame = _single_mode_frame(n=24, k0=2)
    scene = Scene.default_grid()
    img = cine.render_scene(frame, scene, size=(256, 256))
    assert isinstance(img, np.ndarray)
    assert img.shape[0] == 256 and img.shape[1] == 256 and img.shape[2] in (3, 4)


def test_toggling_a_layer_changes_pixels():
    frame = _single_mode_frame(n=24, k0=2)
    s_on = Scene.default_grid()
    s_off = Scene.default_grid()
    s_off.glow.enabled = False
    s_off.streamlines.enabled = False
    a = cine.render_scene(frame, s_on, size=(256, 256))
    b = cine.render_scene(frame, s_off, size=(256, 256))
    assert a.shape == b.shape
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) > 0.5
