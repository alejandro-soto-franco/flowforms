"""Camera helpers: orbit paths and data framing for cinematic renders."""
from __future__ import annotations
import numpy as np


def orbit_positions(center, radius, n, *, elevation: float = 25.0,
                    revolutions: float = 1.0):
    """Return n camera_position tuples sweeping revolutions full turns in azimuth.

    revolutions=1.0 (default) gives a seamless loop over 2*pi.
    revolutions<1.0 gives a partial arc (start and end orientations differ).
    """
    cx, cy, cz = center
    el = np.radians(elevation)
    z = radius * np.sin(el)
    r_xy = radius * np.cos(el)
    out = []
    for i in range(n):
        az = 2.0 * np.pi * revolutions * i / n
        pos = (cx + r_xy * np.cos(az), cy + r_xy * np.sin(az), cz + z)
        out.append((pos, (cx, cy, cz), (0.0, 0.0, 1.0)))
    return out


def bounds_center_radius(mesh):
    """Center and a framing radius from a mesh's axis-aligned bounds."""
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
    return center, 1.4 * diag
