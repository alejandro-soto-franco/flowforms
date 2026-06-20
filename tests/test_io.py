from pathlib import Path
import numpy as np
import pyvista as pv
import flowforms.io as fio

FIX = Path(__file__).parent / "fixtures"


# Task 8 tests

def test_loads_grid_as_grid_kind():
    frame = fio.load(FIX / "grid.vti")
    assert frame.kind == "grid"
    assert "velocity" in frame.fields()


def test_loads_surface_as_surface_kind():
    frame = fio.load(FIX / "surface.vtp")
    assert frame.kind == "surface"


def test_derive_velocity_magnitude():
    frame = fio.load(FIX / "grid.vti").derive("velocity_mag")
    arr = frame.array("velocity_mag")
    assert arr.ndim == 1
    assert np.all(arr >= 0.0)


# Task 9 test

def test_series_iterates_in_time_order():
    s = fio.load_series(FIX / "seq.pvd")
    assert len(s) == 2
    assert list(s.times) == [0.0, 0.5]
    frames = list(s)
    assert frames[0].time == 0.0
    assert frames[1].kind == "grid"


# Final-fix tests

def _make_shear_frame() -> fio.Frame:
    """Build a small ImageData with u = (y, 0, 0), so curl = (0, 0, -du/dy) = (0, 0, -1)."""
    nx, ny, nz = 6, 6, 6
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    pts = np.array(grid.points)
    # u_x = y, u_y = 0, u_z = 0
    vel = np.zeros((grid.n_points, 3))
    vel[:, 0] = pts[:, 1]
    grid.point_data["velocity"] = vel
    return fio.Frame(grid)


def test_derive_vorticity_computes_curl():
    """Vorticity of u=(y,0,0) should be (0,0,-1) almost everywhere."""
    frame = _make_shear_frame().derive("vorticity")
    vort = frame.array("vorticity")
    assert vort.shape[1] == 3
    # Use median to avoid noisy boundary cells.
    med = np.median(vort, axis=0)
    assert abs(med[0]) < 0.1, f"x-component of curl should be ~0, got {med[0]}"
    assert abs(med[1]) < 0.1, f"y-component of curl should be ~0, got {med[1]}"
    assert abs(med[2] - (-1.0)) < 0.1, f"z-component of curl should be ~-1, got {med[2]}"


def test_derive_qcriterion_runs():
    """Q-criterion can be derived from the same shear field and yields finite values."""
    frame = _make_shear_frame().derive("qcriterion")
    q = frame.array("qcriterion")
    assert q.ndim == 1
    assert np.all(np.isfinite(q))


def test_nematic_field_detected():
    """surface.vtp must contain director__nematic and nematic_fields() must return it."""
    frame = fio.load(FIX / "surface.vtp")
    assert "director__nematic" in frame.fields(), f"fields: {frame.fields()}"
    assert frame.nematic_fields() == ["director__nematic"]


def test_omega_alias():
    """|omega| and omega_mag both resolve to the omega_mag array (non-negative)."""
    frame = fio.load(FIX / "grid.vti")
    arr = frame.derive("|omega|").array("omega_mag")
    assert arr.ndim == 1
    assert np.all(arr >= 0.0)

    # omega_mag alias also works
    arr2 = frame.derive("omega_mag").array("omega_mag")
    assert np.allclose(arr, arr2)
