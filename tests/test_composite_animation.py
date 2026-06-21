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


def test_normalize_frames_uniform_even_shape():
    # Frames of differing shapes must collapse to one common even shape.
    frames = [
        np.zeros((90, 120, 3), dtype=np.uint8),
        np.zeros((91, 121, 3), dtype=np.uint8),
        np.zeros((100, 130, 3), dtype=np.uint8),
    ]
    out = comp.normalize_frames(frames)
    shapes = {a.shape for a in out}
    assert len(shapes) == 1, f"frames not uniform: {shapes}"
    (h, w, c) = out[0].shape
    assert c == 3
    assert h % 2 == 0 and w % 2 == 0
    assert len(out) == len(frames)


def test_normalize_frames_empty():
    assert comp.normalize_frames([]) == []


def test_composite_animation_empty_series_raises(tmp_path):
    class _Empty:
        times = np.array([])
        def __len__(self): return 0
        def __getitem__(self, i): raise IndexError
        def __iter__(self): return iter(())
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"time": [0.0], "enstrophy": [1.0]}), p)
    d = diag.load(p)
    with pytest.raises(ValueError, match="empty series"):
        comp.render_composite_animation(_Empty(), d, Scene.default_grid(),
                                        out=tmp_path / "x", formats=("mp4",))


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


@pytest.mark.skipif(not _has_gl(), reason="no GL/xvfb")
def test_composite_animation_streamline_cache(tmp_path):
    """Tweak 1: update_every=2 with a 6-frame series writes a valid movie."""
    from tests.test_spectrum import _single_mode_frame
    from flowforms.scene import Streamlines, Glow, Background
    n = 6
    frames = [_single_mode_frame(n=16, k0=2) for _ in range(n)]
    times = np.linspace(0, 1, n)
    series = _FakeSeries(frames, times)
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"time": times, "enstrophy": 1 + times}), p)
    d = diag.load(p)
    scene = Scene(
        background=Background(enabled=True),
        glow=Glow(enabled=True, field="omega_mag", opacity=0.35),
        streamlines=Streamlines(enabled=True, vectors="velocity", n_points=20,
                                opacity=0.15, update_every=2),
    )
    out = tmp_path / "comp_cache"
    paths = comp.render_composite_animation(
        series, d, scene, out=out, fps=6,
        top_size=(128, 96), plot_size=(128, 48), formats=("mp4",))
    assert all(Path(x).exists() and Path(x).stat().st_size > 0 for x in paths)
