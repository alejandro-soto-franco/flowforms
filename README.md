# flowforms

Beautiful CFD visualization for the [vonkarman](https://github.com/alejandro-soto-franco/vonkarman)
(periodic-grid pseudospectral Navier-Stokes) and [cartan](https://github.com/alejandro-soto-franco/cartan)
(Riemannian/DEC manifold) solvers.

flowforms consumes a single canonical VTK/PVD interchange emitted by both solvers
and exposes one uniform data model, so the same code drives gridded torus data and
curved triangulated manifolds.

## Status

Foundation layer: the `Frame`/`Series` loader over the VTK/PVD interchange.
Figure, cinema, and composite-animation tracks are planned.

## Install

```bash
uv sync
```

## Usage

```python
import flowforms.io as fio

# A single snapshot (.vti grid, or .vtu/.vtp mesh)
frame = fio.load("snapshot.vti")
print(frame.kind)            # "grid" | "surface" | "volume-mesh"
print(frame.fields())        # available point/cell arrays
speed = frame.derive("velocity_mag").array("velocity_mag")

# A time series (.pvd collection)
series = fio.load_series("run.pvd")
for f in series:             # iterates in simulation-time order
    ...
```

## Interchange conventions

- Grid point data: `velocity`, `vorticity` (3-component), `omega_mag` (scalar).
- Derived fields: `velocity_mag`, `vorticity`, `qcriterion`, `omega_mag`.
- Headless nematic directors are emitted with a `__nematic` name suffix so the
  renderer can draw them double-ended.

## License

MIT
