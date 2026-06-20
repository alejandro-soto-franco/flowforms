from flowforms.recipes import scene_from_recipe


def test_scene_from_recipe_toggles():
    r = {"glow": {"enabled": True, "field": "omega_mag"},
         "streamlines": {"enabled": False}}
    s = scene_from_recipe(r)
    assert s.glow.enabled and s.glow.field == "omega_mag"
    assert not s.streamlines.enabled
