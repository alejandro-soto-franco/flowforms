import numpy as np
import flowforms.chrome as chrome


def test_add_chrome_preserves_shape_and_marks_pixels():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    out = chrome.add_chrome(img, title="Colliding Vortex Rings",
                            handle="@asf", caption="Re=700 head-on collision")
    assert out.shape == img.shape
    assert out.sum() > 0  # text drawn


def test_add_chrome_noop_when_empty():
    img = np.full((100, 100, 3), 7, dtype=np.uint8)
    out = chrome.add_chrome(img)
    assert np.array_equal(out, img)
