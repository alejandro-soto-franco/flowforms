from pathlib import Path
import numpy as np
import pytest
import pyvista as pv
import pyarrow as pa, pyarrow.parquet as pq
import flowforms.diagnostics as diag
import flowforms.composite as comp
from flowforms.scene import Scene


def _has_gl():
    try:
        pv.OFF_SCREEN = True
        p = pv.Plotter(off_screen=True); p.add_mesh(pv.Sphere())
        p.screenshot(return_img=True); p.close(); return True
    except Exception:
        return False


class _FakeSeries:
    def __init__(self, frames, times):
        self._f = frames; self.times = times
    def __len__(self): return len(self._f)
    def __getitem__(self, i): return self._f[i]
    def __iter__(self): return iter(self._f)


@pytest.mark.skipif(not _has_gl(), reason="no GL/xvfb")
def test_composite_animation_writes(tmp_path):
    from tests.test_spectrum import _single_mode_frame
    n = 4
    frames = [_single_mode_frame(n=16, k0=2) for _ in range(n)]
    times = np.linspace(0, 1, n)
    series = _FakeSeries(frames, times)
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"time": times, "enstrophy": 1 + times}), p)
    d = diag.load(p)
    out = tmp_path / "comp"
    paths = comp.render_composite_animation(
        series, d, Scene.default_grid(), out=out, fps=8,
        top_size=(128, 96), plot_size=(128, 48), formats=("mp4",))
    assert all(Path(x).exists() and Path(x).stat().st_size > 0 for x in paths)
