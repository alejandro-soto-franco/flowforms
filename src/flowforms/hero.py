"""The flowforms hero: colliding vortex rings over a rolling enstrophy curve."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

from . import composite as _composite
from . import chrome as _chrome
from .scene import Scene


def poster_frame_index(times, values, impact_time=None) -> int:
    times = np.asarray(times)
    if impact_time is not None:
        return int(np.argmin(np.abs(times - impact_time)))
    return int(np.argmax(np.asarray(values)))


def hero_scene() -> Scene:
    s = Scene.default_grid()
    s.glow.enabled = True
    s.glow.field = "omega_mag"
    s.streamlines.enabled = True
    return s


def build_hero(series, diag, *, out_dir, formats=("mp4", "webm"),
               title="Colliding Vortex Rings", handle="", caption="",
               impact_time=None, fps=30) -> dict:
    if "enstrophy" not in diag.columns:
        raise ValueError(
            f"diagnostics missing required 'enstrophy' column; have {diag.columns}")
    if len(series) == 0:
        raise ValueError("empty series; nothing to render")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = hero_scene()
    ann = ((impact_time, "impact"),) if impact_time is not None else ()
    results: dict = {}

    # Portrait 1080x1350 (top 1080x900 + plot 1080x450) and square 1080x1080.
    specs = {
        "portrait": dict(top_size=(1080, 900), plot_size=(1080, 450)),
        "square": dict(top_size=(1080, 720), plot_size=(1080, 360)),
    }
    for name, sz in specs.items():
        paths = _composite.render_composite_animation(
            series, diag, scene, quantity="enstrophy",
            out=out_dir / f"hero_{name}", fps=fps, layout="stacked",
            annotations=ann, formats=formats,
            title=title, handle=handle, caption=caption, time_readout=True,
            **sz)
        results[name] = paths

    # Poster: a single composited frame at the impact / enstrophy-peak time, with chrome.
    idx = poster_frame_index(series.times, diag.column("enstrophy"), impact_time)
    from . import cine as _cine
    top = _cine.render_scene(series[idx], scene, size=(1080, 900))
    bottom = _composite.rolling_plot(diag, "enstrophy", float(series.times[idx]),
                                     size_px=(1080, 450), annotations=ann)
    frame = _composite.stack(top, bottom, layout="stacked")
    frame = _chrome.add_chrome(frame, title=title, handle=handle, caption=caption)
    poster = out_dir / "hero_poster.png"
    Image.fromarray(frame).save(poster)
    results["poster"] = poster
    return results
