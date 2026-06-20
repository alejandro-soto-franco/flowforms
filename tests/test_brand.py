import matplotlib
matplotlib.use("Agg")
import flowforms.brand as brand


def test_palette_has_core_colors():
    for key in ("ink", "paper", "gold", "rust", "blue", "grid_bg", "cine_bg"):
        assert key in brand.PALETTE
        assert isinstance(brand.PALETTE[key], str) and brand.PALETTE[key].startswith("#")


def test_field_cmap_kinds():
    # Diverging for signed fields, sequential for magnitudes.
    assert brand.field_cmap("vorticity") is not None
    assert brand.field_cmap("pressure") is not None
    assert brand.field_cmap("magnitude") is not None
    assert brand.field_cmap("qcriterion") is not None


def test_apply_figure_style_sets_rcparams():
    brand.apply_figure_style()
    import matplotlib as mpl
    assert mpl.rcParams["figure.dpi"] >= 150
    # usetex must match latex availability (never True without latex).
    assert mpl.rcParams["text.usetex"] == brand.latex_available()


def test_pv_theme():
    import pyvista as pv
    theme = brand.figure_pv_theme()
    assert isinstance(theme, pv.themes.Theme)
