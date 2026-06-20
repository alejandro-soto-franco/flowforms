import numpy as np
import flowforms.composite as comp


def test_stacked_dimensions():
    top = np.zeros((400, 600, 3), dtype=np.uint8)
    bottom = np.full((200, 800, 3), 128, dtype=np.uint8)
    out = comp.stack(top, bottom, layout="stacked")
    assert out.shape[1] == 600           # width matched to top
    assert out.shape[0] == 400 + 150     # bottom rescaled to width 600 keeps aspect (200*600/800=150)
    assert out.shape[2] == 3


def test_pip_keeps_top_size():
    top = np.zeros((400, 600, 3), dtype=np.uint8)
    bottom = np.full((200, 200, 3), 255, dtype=np.uint8)
    out = comp.stack(top, bottom, layout="pip")
    assert out.shape[:2] == (400, 600)
    # inset corner is no longer all-zero
    assert out[-1, -1].sum() > 0
