import matplotlib
matplotlib.use("Agg")
import flowforms.brand as brand


def test_palette_has_core_colors():
    for key in ("ink", "paper", "gold", "rust", "blue", "grid_bg", "cine_bg"):
        assert key in brand.PALETTE
        assert isinstance(brand.PALETTE[key], str) and brand.PALETTE[key].startswith("#")


def test_palette_polybius_values():
    assert brand.PALETTE["paper"] == "#ffffff"
    assert brand.PALETTE["ink"] == "#171717"
    assert brand.PALETTE["cine_bg"] == "#0a0a0a"


def test_cm_sans_font_file_exists():
    import os
    path = brand.cm_sans_font_file()
    assert os.path.exists(path) and path.endswith("cmss10.ttf")


def test_field_cmap_kinds():
    # Diverging for signed fields, sequential for magnitudes.
    assert brand.field_cmap("vorticity") is not None
    assert brand.field_cmap("pressure") is not None
    assert brand.field_cmap("magnitude") is not None
    assert brand.field_cmap("qcriterion") is not None


def test_apply_figure_style_sets_rcparams():
    brand.apply_figure_style()
    import matplotlib as mpl
    assert mpl.rcParams["figure.dpi"] == 150
    assert mpl.rcParams["savefig.dpi"] == 200
    # House style: no gridlines, no top/right spines, frameless legend, sans family.
    assert mpl.rcParams["axes.grid"] is False
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["legend.frameon"] is False
    assert mpl.rcParams["font.family"] == ["sans-serif"]
    # usetex is only ever True when latex is on PATH (never True without it).
    if mpl.rcParams["text.usetex"]:
        assert brand.latex_available()


def test_apply_figure_style_dark_theme():
    brand.apply_figure_style(dark=True)
    import matplotlib as mpl
    assert mpl.rcParams["figure.facecolor"] == "#0a0a0a"
    assert mpl.rcParams["axes.facecolor"] == "#0a0a0a"
    assert mpl.rcParams["text.color"] == "#ededed"
    assert mpl.rcParams["axes.edgecolor"] == "#666666"


def test_pv_theme():
    import pyvista as pv
    theme = brand.figure_pv_theme()
    assert isinstance(theme, pv.themes.Theme)
