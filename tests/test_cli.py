import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from flowforms.cli import main


def test_cli_diagnostics_figure(tmp_path):
    t = np.linspace(0, 3, 30)
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"time": t, "energy": np.exp(-t), "enstrophy": 1 + t * 0}), p)
    out = tmp_path / "fig.png"
    rc = main(["figure", "--diagnostics", str(p), "--quantity", "energy",
               "--quantity", "enstrophy", "--out", str(out)])
    assert rc == 0
    assert out.exists()


def test_cli_recipe(tmp_path):
    t = np.linspace(0, 3, 30)
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"time": t, "energy": np.exp(-t)}), p)
    out = tmp_path / "r.png"
    recipe = tmp_path / "r.toml"
    recipe.write_text(
        f'kind = "decay"\ndiagnostics = "{p}"\nquantities = ["energy"]\nout = "{out}"\n'
    )
    assert main(["figure", "--recipe", str(recipe)]) == 0
    assert out.exists()
