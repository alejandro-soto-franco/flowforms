"""TOML viz recipes: declarative figure specs mirroring the solver TOMLs."""
from __future__ import annotations
import tomllib
from pathlib import Path


def load_recipe(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


from .scene import Scene  # noqa: E402


def scene_from_recipe(r: dict) -> Scene:
    """Build a Scene from a recipe dict; unspecified layers keep defaults."""
    s = Scene.default_grid()
    for layer in ("background", "glow", "arrows", "streamlines", "isosurface", "slice"):
        if layer in r:
            target = getattr(s, layer)
            for k, v in r[layer].items():
                if hasattr(target, k):
                    setattr(target, k, v)
    return s
