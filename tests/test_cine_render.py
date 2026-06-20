import numpy as np
import pytest
import pyvista as pv
import flowforms.io as fio
from flowforms.scene import Scene, Glow, Isosurface, Streamlines
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


def test_render_scene_returns_rgb():
    """M3: render_scene must return a 3-channel array."""
    frame = _single_mode_frame(n=24, k0=2)
    img = cine.render_scene(frame, Scene.default_grid(), size=(128, 128))
    assert img.ndim == 3 and img.shape[2] == 3


def test_surface_render_nonexistent_glow_field():
    """I4: rendering a surface frame with a nonexistent glow field must not crash."""
    frame = fio.load("tests/fixtures/surface.vtp")
    scene = Scene.default_surface()
    scene.glow.enabled = True
    scene.glow.field = "NONEXISTENT_FIELD_XYZ"
    img = cine.render_scene(frame, scene, size=(256, 256))
    assert isinstance(img, np.ndarray) and img.shape[2] == 3


def test_surface_nematic_directors_render():
    """I5: rendering a surface frame with nematic directors must not crash and return an image."""
    frame = fio.load("tests/fixtures/surface.vtp")
    # surface.vtp has 'director__nematic'; use default_surface scene so glow is enabled.
    scene = Scene.default_surface()
    scene.glow.field = "temp"
    img = cine.render_scene(frame, scene, size=(256, 256))
    assert isinstance(img, np.ndarray) and img.shape[2] == 3


def test_isosurface_primary_cascade_scene_returns_image():
    """Rev-2: a cascade scene with Q-criterion isosurface primary renders without error."""
    frame = _single_mode_frame(n=24, k0=2)
    scene = Scene(
        isosurface=Isosurface(enabled=True, field="qcriterion", values=()),
        streamlines=Streamlines(enabled=True, n_points=40, opacity=0.15),
    )
    img = cine.render_scene(frame, scene, size=(256, 256))
    assert isinstance(img, np.ndarray)
    assert img.shape[0] == 256 and img.shape[1] == 256 and img.shape[2] == 3


def test_streamlines_faint_opacity_renders_without_error():
    """Rev-2: streamlines with opacity=0.15 (faint) must render without error."""
    frame = _single_mode_frame(n=24, k0=2)
    scene = Scene(
        streamlines=Streamlines(enabled=True, n_points=40, opacity=0.15),
    )
    img = cine.render_scene(frame, scene, size=(128, 128))
    assert isinstance(img, np.ndarray) and img.shape[2] == 3
