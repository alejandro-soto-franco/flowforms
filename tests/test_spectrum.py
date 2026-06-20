import numpy as np
import pyvista as pv
from flowforms.io import Frame
import flowforms.spectrum as sp


def _single_mode_frame(n=16, k0=2):
    L = 2 * np.pi
    xs = np.linspace(0, L, n, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    u = np.sin(k0 * X)
    vel = np.zeros((n, n, n, 3))
    vel[..., 0] = u
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(L / n,) * 3, origin=(0, 0, 0))
    # pyvista point order is x-fastest: flatten with order matching reshape(dims[::-1]).
    flat = np.transpose(vel, (2, 1, 0, 3)).reshape(-1, 3)
    grid["velocity"] = flat
    return Frame(grid)


def test_spectrum_peaks_at_mode():
    frame = _single_mode_frame(n=16, k0=2)
    k, ek = sp.energy_spectrum(frame)
    assert k.shape == ek.shape
    assert ek[np.argmax(ek)] > 0
    peak_k = k[np.argmax(ek)]
    assert abs(peak_k - 2) <= 1  # energy concentrated near |k| = 2


def test_spectrum_nonnegative():
    frame = _single_mode_frame()
    _, ek = sp.energy_spectrum(frame)
    assert np.all(ek >= 0)
