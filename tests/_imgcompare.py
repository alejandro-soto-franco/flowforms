"""Tiny perceptual hash (average-hash over a DCT-free 8x8 luminance grid).

Robust enough for golden-image regression with a Hamming tolerance, with no
external image-hashing dependency. Accepts a path to a PNG or an HxWxC array.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _to_luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    return img.astype(np.float64)


def _load(path_or_array) -> np.ndarray:
    if isinstance(path_or_array, (str, Path)):
        from matplotlib import image as mpimg
        arr = mpimg.imread(str(path_or_array))
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8)
        return arr
    return np.asarray(path_or_array)


def phash(path_or_array, size: int = 8) -> int:
    luma = _to_luma(_load(path_or_array))
    h, w = luma.shape
    # Block-average down to size x size.
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    small = np.empty((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            block = luma[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            small[i, j] = block.mean() if block.size else 0.0
    bits = (small > 127.5).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def assert_close(actual_png, golden_png, max_hamming: int = 8) -> None:
    d = hamming(phash(actual_png), phash(golden_png))
    assert d <= max_hamming, f"image differs from golden: hamming={d} > {max_hamming}"
