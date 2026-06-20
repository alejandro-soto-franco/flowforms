from pathlib import Path
import numpy as np
import pytest
import pyvista as pv
from flowforms.scene import Scene
import flowforms.cine as cine


def _has_gl() -> bool:
    try:
        pv.OFF_SCREEN = True
        p = pv.Plotter(off_screen=True); p.add_mesh(pv.Sphere())
        p.screenshot(return_img=True); p.close(); return True
    except Exception:
        return False


class _FakeSeries:
    """Minimal Series-like: yields a few grid frames with a time axis."""
    def __init__(self, frames):
        self._frames = frames
        self.times = np.linspace(0, 1, len(frames))
    def __len__(self): return len(self._frames)
    def __getitem__(self, i): return self._frames[i]
    def __iter__(self): return iter(self._frames)


@pytest.mark.skipif(not _has_gl(), reason="no GL/xvfb")
def test_render_animation_writes_files(tmp_path):
    from tests.test_spectrum import _single_mode_frame
    frames = [_single_mode_frame(n=16, k0=2) for _ in range(4)]
    series = _FakeSeries(frames)
    scene = Scene.default_grid()
    out = tmp_path / "movie"
    paths = cine.render_animation(series, scene, out=out, fps=8, size=(128, 128),
                                  formats=("mp4",))
    assert all(Path(p).exists() and Path(p).stat().st_size > 0 for p in paths)
