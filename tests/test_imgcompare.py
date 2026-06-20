import numpy as np
from tests._imgcompare import phash, hamming


def test_identical_images_zero_distance():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    assert hamming(phash(img), phash(img.copy())) == 0


def test_small_perturbation_small_distance():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    noisy = img.astype(np.int16)
    noisy[0, 0] = (noisy[0, 0] + 3) % 256
    d = hamming(phash(img), phash(noisy.astype(np.uint8)))
    assert d <= 4


def test_different_images_large_distance():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert hamming(phash(a), phash(b)) > 16
