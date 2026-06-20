import matplotlib
matplotlib.use("Agg")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import flowforms.diagnostics as diag
import flowforms.figures as figures
from tests._imgcompare import assert_close
from tests.test_spectrum import _single_mode_frame


def _diag(tmp_path):
    n = 50
    t = np.linspace(0, 5, n)
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({
        "time": t, "energy": np.exp(-0.2 * t),
        "enstrophy": 1 + 0.5 * np.sin(2 * t) ** 2, "helicity": np.cos(t),
    }), p)
    return diag.load(p)


def test_decay_plot_structure(tmp_path):
    fig = figures.decay_plot(_diag(tmp_path), quantities=("energy", "enstrophy"))
    ax = fig.axes[0]
    assert ax.get_xlabel() != ""
    assert len(ax.get_lines()) == 2
    assert ax.get_legend() is not None


def test_spectrum_plot_has_guide():
    fig = figures.spectrum_plot(_single_mode_frame(), guide=True)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    # one data line plus one guide line
    assert len(ax.get_lines()) >= 2


def test_save_writes_vector_formats(tmp_path):
    fig = figures.decay_plot(_diag(tmp_path))
    stem = tmp_path / "out"
    figures.save(fig, stem.with_suffix(".png"))
    assert stem.with_suffix(".png").exists()
    assert stem.with_suffix(".pdf").exists()
    assert stem.with_suffix(".svg").exists()


def test_decay_plot_golden(tmp_path):
    fig = figures.decay_plot(_diag(tmp_path), quantities=("energy", "enstrophy"))
    out = tmp_path / "decay.png"
    figures.save(fig, out)
    assert_close(out, "tests/golden/decay.png", max_hamming=10)
