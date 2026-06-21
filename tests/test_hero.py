import numpy as np
import pytest
import flowforms.hero as hero
from flowforms.diagnostics import Diagnostics


class _FakeSeries:
    def __init__(self, n):
        self.times = np.linspace(0, 1, n)
    def __len__(self): return len(self.times)
    def __getitem__(self, i): raise IndexError
    def __iter__(self): return iter(())


def test_hero_scene_streamlines_disabled():
    """Tweak 1: hero_scene() must have streamlines.enabled == False."""
    s = hero.hero_scene()
    assert not s.streamlines.enabled


def test_poster_frame_selection():
    # The poster frame is the enstrophy peak when no impact_time is given.
    times = np.linspace(0, 10, 50)
    enstrophy = 1 + np.exp(-(times - 6.0) ** 2)
    idx = hero.poster_frame_index(times, enstrophy, impact_time=None)
    assert abs(times[idx] - 6.0) < 0.5


def test_poster_frame_uses_impact_time():
    times = np.linspace(0, 10, 50)
    enstrophy = np.ones_like(times)
    idx = hero.poster_frame_index(times, enstrophy, impact_time=3.0)
    assert abs(times[idx] - 3.0) < 0.2


def test_build_hero_missing_enstrophy_raises(tmp_path):
    # Validated BEFORE any GL rendering, so this is GL-free.
    d = Diagnostics({"time": np.linspace(0, 1, 5), "energy": np.ones(5)})
    series = _FakeSeries(5)
    with pytest.raises(ValueError, match="enstrophy"):
        hero.build_hero(series, d, out_dir=tmp_path / "h")


def test_build_hero_empty_series_raises(tmp_path):
    d = Diagnostics({"time": np.linspace(0, 1, 5), "enstrophy": np.ones(5)})
    series = _FakeSeries(0)
    with pytest.raises(ValueError, match="empty series"):
        hero.build_hero(series, d, out_dir=tmp_path / "h")
