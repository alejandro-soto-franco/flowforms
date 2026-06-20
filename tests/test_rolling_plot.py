import matplotlib
matplotlib.use("Agg")
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import flowforms.diagnostics as diag
import flowforms.composite as comp


def _diag(tmp_path):
    t = np.linspace(0, 10, 100)
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"time": t, "enstrophy": 1 + np.exp(-(t - 5) ** 2)}), p)
    return diag.load(p)


def test_rolling_plot_shape(tmp_path):
    d = _diag(tmp_path)
    img = comp.rolling_plot(d, "enstrophy", 5.0, size_px=(640, 240))
    assert img.shape[0] == 240 and img.shape[1] == 640 and img.shape[2] in (3, 4)


def test_cursor_advances_changes_pixels(tmp_path):
    d = _diag(tmp_path)
    a = comp.rolling_plot(d, "enstrophy", 1.0, size_px=(640, 240))
    b = comp.rolling_plot(d, "enstrophy", 9.0, size_px=(640, 240))
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) > 0.5
