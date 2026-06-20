"""Read vonkarman diagnostics.parquet scalar time-series logs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


class Diagnostics:
    def __init__(self, columns: dict[str, np.ndarray]):
        self._cols = columns

    @property
    def columns(self) -> list[str]:
        return list(self._cols.keys())

    def column(self, name: str) -> np.ndarray:
        if name not in self._cols:
            raise KeyError(f"no diagnostics column {name!r}; have {self.columns}")
        return self._cols[name]

    @property
    def time(self) -> np.ndarray:
        return self.column("time")

    @property
    def energy(self) -> np.ndarray:
        return self.column("energy")

    @property
    def enstrophy(self) -> np.ndarray:
        return self.column("enstrophy")

    @property
    def helicity(self) -> np.ndarray:
        return self.column("helicity")


def load(path: str | Path) -> Diagnostics:
    table = pq.read_table(str(path))
    cols = {name: np.asarray(table.column(name)) for name in table.column_names}
    return Diagnostics(cols)
