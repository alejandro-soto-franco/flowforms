"""Load VTK/PVD output from vonkarman and cartan into a uniform Frame."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, cast
import numpy as np
import pyvista as pv

_VTK_TETRA = 10


def _detect_kind(mesh: pv.DataSet) -> str:
    if isinstance(mesh, pv.ImageData):
        return "grid"
    try:
        celltypes = mesh.celltypes
    except Exception:
        celltypes = np.array([], dtype=np.uint8)
    if celltypes.size and int(celltypes.max()) >= _VTK_TETRA:
        return "volume-mesh"
    return "surface"


class Frame:
    def __init__(self, mesh: pv.DataSet, time: float = 0.0):
        self.mesh = mesh
        self.time = float(time)
        self.kind = _detect_kind(mesh)

    def fields(self) -> list[str]:
        return list(self.mesh.point_data.keys()) + list(self.mesh.cell_data.keys())

    def array(self, name: str) -> np.ndarray:
        return np.asarray(self.mesh[name])

    def derive(self, name: str) -> "Frame":
        if name in self.fields():
            return self
        m = self.mesh
        if name == "velocity_mag":
            m = m.copy()
            m["velocity_mag"] = np.linalg.norm(np.asarray(m["velocity"]), axis=1)
        elif name in ("vorticity", "qcriterion"):
            m = cast(pv.DataSet, m.compute_derivative(scalars="velocity", vorticity=True, qcriterion=True))
        elif name in ("|omega|", "omega_mag"):
            base = self.derive("vorticity").mesh.copy()
            base["omega_mag"] = np.linalg.norm(np.asarray(base["vorticity"]), axis=1)
            m = base
        else:
            raise ValueError(f"unknown derived field: {name}")
        return Frame(cast(pv.DataSet, m), self.time)


class Series:
    def __init__(self, pvd_path: str | Path):
        self._reader = pv.PVDReader(str(pvd_path))
        self.times = np.asarray(self._reader.time_values)

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, i: int) -> Frame:
        self._reader.set_active_time_point(i)
        block = cast(pv.DataSet, self._reader.read()[0])
        return Frame(block, float(self.times[i]))

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self[i]


def load(path: str | Path) -> Frame:
    return Frame(cast(pv.DataSet, pv.read(str(path))))


def load_series(path: str | Path) -> Series:
    return Series(path)
