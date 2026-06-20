import numpy as np
import flowforms.hero as hero


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
