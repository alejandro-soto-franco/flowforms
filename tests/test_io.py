from pathlib import Path
import numpy as np
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
