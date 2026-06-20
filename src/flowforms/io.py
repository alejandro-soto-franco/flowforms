"""Load VTK/PVD output from vonkarman and cartan into a uniform Frame.

Interchange conventions
-----------------------
Grid frames (pv.ImageData / .vti):
  - ``velocity``      -- 3-component velocity vector per point
  - ``vorticity``     -- curl of velocity, derived via pyvista
  - ``omega_mag``     -- magnitude of vorticity (aliases: ``|omega|``)
  - ``velocity_mag``  -- magnitude of velocity
  - ``qcriterion``    -- Q-criterion scalar, derived via pyvista

Surface / mesh frames (.vtp):
  - Arbitrary named scalars and vectors.
  - A vector whose DataArray name ends with ``__nematic`` is a headless
    director field and must be rendered double-ended. Use ``nematic_fields()``
    to enumerate them and ``base_name()`` to strip the suffix.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, cast
import numpy as np
import pyvista as pv

_VTK_TETRA = 10
_NEMATIC_SUFFIX = "__nematic"


def base_name(name: str) -> str:
    """Strip the ``__nematic`` suffix from a field name, if present."""
    if name.endswith(_NEMATIC_SUFFIX):
        return name[: -len(_NEMATIC_SUFFIX)]
    return name


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

    def nematic_fields(self) -> list[str]:
        """Return names of point-data arrays whose name ends with ``__nematic``."""
        return [k for k in self.mesh.point_data.keys() if k.endswith(_NEMATIC_SUFFIX)]

    def array(self, name: str) -> np.ndarray:
        return np.asarray(self.mesh[name])

    def derive(self, name: str) -> "Frame":
        # Canonical resolution: |omega| and omega_mag both resolve to omega_mag.
        canonical = "omega_mag" if name in ("|omega|", "omega_mag") else name

        if canonical in self.fields():
            return self
        m = self.mesh
        if canonical == "velocity_mag":
            m = m.copy()
            m["velocity_mag"] = np.linalg.norm(np.asarray(m["velocity"]), axis=1)
        elif canonical in ("vorticity", "qcriterion"):
            m = cast(
                pv.DataSet,
                m.compute_derivative(scalars="velocity", vorticity=True, qcriterion=True),
            )
        elif canonical == "omega_mag":
            base = self.derive("vorticity").mesh.copy()
            base["omega_mag"] = np.linalg.norm(np.asarray(base["vorticity"]), axis=1)
            m = base
        else:
            raise ValueError(f"unknown derived field: {name!r}")
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
