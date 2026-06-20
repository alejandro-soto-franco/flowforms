import numpy as np
import flowforms.chrome as chrome
import flowforms.brand as brand


def test_add_chrome_title_only_marks_pixels():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    out = chrome.add_chrome(img, title="Colliding Vortex Rings")
    assert out.shape == img.shape
    assert out.sum() > 0  # title drawn


def test_add_chrome_uses_cm_sans_font():
    # The title must render with the bundled Computer Modern Sans TTF.
    import os
    assert os.path.exists(brand.cm_sans_font_file())
    font = chrome._font(24)
    # PIL truetype fonts expose the loaded file path.
    assert getattr(font, "path", "").endswith("cmss10.ttf")


def test_add_chrome_title_not_yellow():
    # The title color must be the light grey, never gold/yellow.
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    out = chrome.add_chrome(img, title="Colliding Vortex Rings").astype(int)
    gold = np.array([0xE8, 0xB0, 0x4B])
    drawn = out[out.sum(axis=2) > 0]
    assert len(drawn) > 0
    # No drawn pixel should be close to gold.
    assert np.min(np.abs(drawn - gold).sum(axis=1)) > 60


def test_add_chrome_noop_when_empty():
    img = np.full((100, 100, 3), 7, dtype=np.uint8)
    out = chrome.add_chrome(img)
    assert np.array_equal(out, img)
