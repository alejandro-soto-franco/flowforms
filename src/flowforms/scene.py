"""Declarative scene configuration: independently toggleable cinema layers."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Background:
    enabled: bool = True
    color: str | None = None  # None -> brand cine_bg


@dataclass
class Glow:
    enabled: bool = False
    field: str = "omega_mag"
    opacity: float = 0.35
    cmap: str | None = None  # None -> brand EMISSIVE_CMAP


@dataclass
class Arrows:
    enabled: bool = False
    field: str = "velocity"
    density: float = 0.1   # fraction of points kept
    scale: float = 0.05


@dataclass
class Streamlines:
    enabled: bool = False
    vectors: str = "velocity"
    n_points: int = 200
    radius: float = 0.01
    cmap: str | None = None
    opacity: float = 1.0


@dataclass
class Isosurface:
    enabled: bool = False
    field: str = "qcriterion"
    values: tuple[float, ...] = ()


@dataclass
class Slice:
    enabled: bool = False
    field: str = "omega_mag"
    axis: str = "z"


@dataclass
class Scene:
    background: Background = field(default_factory=Background)
    glow: Glow = field(default_factory=Glow)
    arrows: Arrows = field(default_factory=Arrows)
    streamlines: Streamlines = field(default_factory=Streamlines)
    isosurface: Isosurface = field(default_factory=Isosurface)
    slice: Slice = field(default_factory=Slice)

    def enabled_layers(self) -> list[str]:
        out = []
        for name in ("background", "glow", "arrows", "streamlines", "isosurface", "slice"):
            if getattr(self, name).enabled:
                out.append(name)
        return out

    @classmethod
    def default_grid(cls) -> "Scene":
        return cls(
            background=Background(enabled=True),
            glow=Glow(enabled=True, field="omega_mag", opacity=0.35),
            streamlines=Streamlines(enabled=True, vectors="velocity", n_points=200),
        )

    @classmethod
    def default_surface(cls) -> "Scene":
        return cls(
            background=Background(enabled=True),
            glow=Glow(enabled=True, field="temp", opacity=0.6),
        )
