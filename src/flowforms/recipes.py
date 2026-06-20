"""TOML viz recipes: declarative figure specs mirroring the solver TOMLs."""
from __future__ import annotations
import tomllib
from pathlib import Path


def load_recipe(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)
