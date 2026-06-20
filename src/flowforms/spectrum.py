"""Shell-averaged kinetic energy spectrum E(k) from a grid velocity field."""
from __future__ import annotations
import numpy as np
from .io import Frame


def energy_spectrum(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Compute the shell-averaged energy spectrum of a cubic periodic grid.

    Returns (k, E_k) with integer shell wavenumbers. E(k) is the kinetic
    energy summed over the spherical shell k <= |kvec| < k+1.

    Convention (verified by test_spectrum.py::_single_mode_frame):
    pyvista ImageData stores points in x-fastest order, so the flat
    point array reshapes to (nz, ny, nx, 3) before transposing to
    (nx, ny, nz, 3) for FFT with indexing='ij'.
    """
    if frame.kind != "grid":
        raise ValueError("energy_spectrum requires a grid frame")
    dims = np.array(frame.mesh.dimensions)  # (nx, ny, nz)
    nx, ny, nz = (int(d) for d in dims)
    vel = frame.array("velocity").reshape(nz, ny, nx, 3)
    # back to (nx, ny, nz, 3): x-fastest flat -> reshape(dims[::-1]) gives (nz,ny,nx)
    vel = np.transpose(vel, (2, 1, 0, 3))

    ntot = nx * ny * nz
    uhat = np.fft.fftn(vel, axes=(0, 1, 2)) / ntot
    e_density = 0.5 * np.sum(np.abs(uhat) ** 2, axis=-1)  # per mode

    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    kz = np.fft.fftfreq(nz, d=1.0 / nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    kbins = np.round(kmag).astype(int)

    kmax = kbins.max()
    shells = np.arange(0, kmax + 1)
    ek = np.zeros(kmax + 1)
    flat_bins = kbins.ravel()
    flat_e = e_density.ravel()
    np.add.at(ek, flat_bins, flat_e)
    return shells, ek
