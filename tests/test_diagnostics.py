import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import flowforms.diagnostics as diag


def _make_fixture(path):
    n = 20
    t = np.linspace(0.0, 2.0, n)
    table = pa.table(
        {
            "step": np.arange(n, dtype=np.uint64),
            "time": t,
            "energy": np.exp(-0.1 * t),
            "enstrophy": 1.0 + np.sin(t) ** 2,
            "helicity": np.cos(t),
        }
    )
    pq.write_table(table, path)


def test_load_columns(tmp_path):
    p = tmp_path / "diagnostics.parquet"
    _make_fixture(p)
    d = diag.load(p)
    assert d.time.shape == (20,)
    assert "enstrophy" in d.columns
    assert np.allclose(d.enstrophy, d.column("enstrophy"))
    assert d.energy[0] > d.energy[-1]  # decaying
