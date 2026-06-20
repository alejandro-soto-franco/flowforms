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


def test_advancing_changes_pixels(tmp_path):
    d = _diag(tmp_path)
    a = comp.rolling_plot(d, "enstrophy", 1.0, size_px=(640, 240))
    b = comp.rolling_plot(d, "enstrophy", 9.0, size_px=(640, 240))
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) > 0.5


def _line_x_extent(img):
    """Rightmost column (fraction of width) that contains any blue line pixel.

    The growing line is bright blue (#627eea) on a near-black (#0a0a0a) bg, so
    we find columns whose blue channel is clearly above the background.
    """
    arr = img.astype(int)
    # Blue-dominant, bright pixels = the data line.
    line = (arr[..., 2] > 120) & (arr[..., 2] > arr[..., 0] + 30)
    cols = np.where(line.any(axis=0))[0]
    if len(cols) == 0:
        return 0.0
    return float(cols.max()) / img.shape[1]


def test_rolling_plot_grows_with_t_now(tmp_path):
    # The line must span LESS x for a small t_now than for a large t_now,
    # because xlim's right edge is t_now and we only draw t <= t_now.
    d = _diag(tmp_path)
    early = comp.rolling_plot(d, "enstrophy", 2.0, size_px=(640, 240))
    late = comp.rolling_plot(d, "enstrophy", 9.0, size_px=(640, 240))
    assert _line_x_extent(early) < _line_x_extent(late)


def test_rolling_plot_no_vertical_cursor(tmp_path):
    # There must be no full-height vertical cursor line. A rust cursor used to
    # paint a near-full-height column; assert no column is mostly the line color.
    d = _diag(tmp_path)
    img = comp.rolling_plot(d, "enstrophy", 5.0, size_px=(640, 240)).astype(int)
    rust = np.array([0xC1, 0x44, 0x0E])
    near = (np.abs(img - rust).sum(axis=2) < 40)
    col_frac = near.mean(axis=0)
    assert col_frac.max() < 0.5  # no near-full-height cursor column


def test_rolling_plot_deterministic_shape(tmp_path):
    # Two different t_now values must yield identical pixel dims (no reflow).
    d = _diag(tmp_path)
    before = comp.rolling_plot(d, "enstrophy", 1.0, size_px=(640, 240))
    after = comp.rolling_plot(d, "enstrophy", 9.0, size_px=(640, 240))
    assert before.shape == after.shape == (240, 640, 3)


def test_rolling_plot_xlim_right_edge_tracks_t_now(tmp_path):
    # Passing xlim sets the left edge; the right edge still tracks t_now so the
    # line grows. Two t_now values with the same left edge must differ.
    d = _diag(tmp_path)
    a = comp.rolling_plot(d, "enstrophy", 3.0, size_px=(640, 240), xlim=(0.0, None))
    b = comp.rolling_plot(d, "enstrophy", 8.0, size_px=(640, 240), xlim=(0.0, None))
    assert a.shape == b.shape
    assert _line_x_extent(a) < _line_x_extent(b)


def test_rolling_plot_fixed_ylim(tmp_path):
    # An explicit ylim is honored (y-axis does not jump as the line grows).
    d = _diag(tmp_path)
    img = comp.rolling_plot(d, "enstrophy", 5.0, size_px=(640, 240), ylim=(0.0, 3.0))
    assert img.shape == (240, 640, 3)
