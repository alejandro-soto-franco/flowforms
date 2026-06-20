import matplotlib
matplotlib.use("Agg")
import pyvista as pv
import flowforms.brand as brand


def test_cinema_theme_is_dark():
    theme = brand.cinema_pv_theme()
    assert isinstance(theme, pv.themes.Theme)
    assert theme.background is not None


def test_emissive_cmap_exists():
    assert brand.EMISSIVE_CMAP is not None
