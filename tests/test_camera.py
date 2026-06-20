import numpy as np
from flowforms import camera


def test_orbit_positions_count_and_loop():
    pos = camera.orbit_positions((0, 0, 0), 5.0, 12, elevation=20.0)
    assert len(pos) == 12
    # each entry is (position, focal_point, view_up)
    p0 = pos[0]
    assert len(p0) == 3 and len(p0[0]) == 3
    # focal point is the center
    assert np.allclose(p0[1], (0, 0, 0))
    # azimuth wraps: positions are distinct and all at the same radius from center
    radii = [np.linalg.norm(np.array(p[0]) - np.array((0, 0, 0))) for p in pos]
    assert np.allclose(radii, radii[0], rtol=1e-6)


def test_bounds_center_radius():
    import pyvista as pv
    sph = pv.Sphere(radius=2.0, center=(1, 1, 1))
    c, r = camera.bounds_center_radius(sph)
    assert np.allclose(c, (1, 1, 1), atol=0.1)
    assert r > 2.0
