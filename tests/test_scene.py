from flowforms.scene import Scene, Glow, Streamlines


def test_default_grid_scene_toggles():
    s = Scene.default_grid()
    assert s.glow.enabled
    assert s.streamlines.enabled
    # everything is individually toggleable
    s.arrows.enabled = False
    assert not s.arrows.enabled


def test_scene_layers_have_params():
    g = Glow(enabled=True, field="omega_mag", opacity=0.4)
    assert g.field == "omega_mag" and 0 <= g.opacity <= 1
    st = Streamlines(enabled=True, n_points=200)
    assert st.n_points == 200


def test_enabled_layers_list():
    s = Scene.default_grid()
    names = s.enabled_layers()
    assert "glow" in names and isinstance(names, list)
