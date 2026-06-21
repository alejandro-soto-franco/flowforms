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


def test_orbit_positions_revolutions_half():
    """revolutions=0.5 sweeps half the azimuth compared to revolutions=1.0."""
    n = 8
    full = camera.orbit_positions((0, 0, 0), 5.0, n, elevation=20.0, revolutions=1.0)
    half = camera.orbit_positions((0, 0, 0), 5.0, n, elevation=20.0, revolutions=0.5)
    # Last position azimuth for full: 2*pi*(n-1)/n; for half: pi*(n-1)/n
    # Compare xy angles of the last position.
    def az(pos_tuple):
        x, y, _ = pos_tuple[0]
        return np.arctan2(y, x)
    # Half revolution last az should be approximately half of full last az
    # (both measured from the same starting angle of 0).
    az_full = az(full[-1]) % (2 * np.pi)
    az_half = az(half[-1]) % (2 * np.pi)
    assert np.isclose(az_half, az_full / 2, atol=1e-9)


def test_orbit_positions_revolutions_default_unchanged():
    """Default (revolutions=1.0) matches the old behaviour exactly."""
    n = 12
    default = camera.orbit_positions((0, 0, 0), 5.0, n, elevation=20.0)
    explicit = camera.orbit_positions((0, 0, 0), 5.0, n, elevation=20.0, revolutions=1.0)
    for a, b in zip(default, explicit):
        assert np.allclose(a[0], b[0])


def test_bounds_center_radius():
    import pyvista as pv
    sph = pv.Sphere(radius=2.0, center=(1, 1, 1))
    c, r = camera.bounds_center_radius(sph)
    assert np.allclose(c, (1, 1, 1), atol=0.1)
    assert r > 2.0
