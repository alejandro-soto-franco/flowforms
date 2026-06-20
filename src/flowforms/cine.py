"""Track B: cinematic layered scene rendering (PyVista off-screen)."""
from __future__ import annotations
from typing import Any
import numpy as np
import pyvista as pv

from . import brand
from .io import Frame
from .scene import Scene


def _grid_layers(pl: pv.Plotter, frame: Frame, scene: Scene) -> None:
    mesh = frame.mesh
    if scene.glow.enabled:
        try:
            f = frame.derive(scene.glow.field)
            cmap = scene.glow.cmap or brand.EMISSIVE_CMAP
            pl.add_volume(f.mesh, scalars=scene.glow.field, cmap=cmap,
                          opacity="linear", opacity_unit_distance=1.0)
        except Exception:
            # add_volume can fail if the scalar range is degenerate (all zeros)
            # or the grid is too small for the fixed-point mapper; skip gracefully.
            pass
    if scene.slice.enabled:
        try:
            f = frame.derive(scene.slice.field)
            # Cast axis to Any to satisfy pyrefly's strict Literal type for `normal`.
            sl: Any = f.mesh.slice(normal=scene.slice.axis)  # type: ignore[arg-type]
            pl.add_mesh(sl, scalars=scene.slice.field,  # type: ignore[arg-type]
                        cmap=brand.field_cmap("vorticity" if "vort" in scene.slice.field else "magnitude"))
        except Exception:
            # slice can fail on degenerate mesh extents; skip gracefully.
            pass
    if scene.isosurface.enabled:
        try:
            f = frame.derive(scene.isosurface.field)
            vals = list(scene.isosurface.values) or [float(np.nanmean(f.array(scene.isosurface.field)))]
            iso: Any = f.mesh.contour(vals, scalars=scene.isosurface.field)
            pl.add_mesh(iso, cmap=brand.field_cmap("magnitude"), smooth_shading=True)  # type: ignore[arg-type]
        except Exception:
            # contour can fail if the isosurface value is outside the scalar range;
            # skip gracefully.
            pass
    if scene.arrows.enabled and scene.arrows.field in mesh.point_data:
        try:
            sub: Any = mesh.glyph(orient=scene.arrows.field, scale=scene.arrows.field,
                                  factor=scene.arrows.scale, tolerance=max(1e-3, 1 - scene.arrows.density))
            pl.add_mesh(sub, color=brand.PALETTE["gold"])  # type: ignore[arg-type]
        except Exception:
            # glyph can fail if the vector field is all-zero or tolerance is invalid;
            # skip gracefully.
            pass
    if scene.streamlines.enabled and scene.streamlines.vectors in mesh.point_data:
        try:
            stream: Any = mesh.streamlines(
                vectors=scene.streamlines.vectors,
                n_points=scene.streamlines.n_points,
                source_radius=float(np.linalg.norm(mesh.length) / 2) or 1.0,
            )
            tubes: Any = stream.tube(radius=scene.streamlines.radius)
            pl.add_mesh(tubes, cmap=scene.streamlines.cmap or brand.field_cmap("magnitude"))  # type: ignore[arg-type]
        except Exception:
            pass  # streamline seeding can fail on degenerate fields; skip the layer


def _surface_layers(pl: pv.Plotter, frame: Frame, scene: Scene) -> None:
    if scene.glow.enabled:
        cmap = scene.glow.cmap or brand.EMISSIVE_CMAP
        pl.add_mesh(frame.mesh, scalars=scene.glow.field, cmap=cmap, smooth_shading=True)
    else:
        pl.add_mesh(frame.mesh, smooth_shading=True)


def render_scene(frame: Frame, scene: Scene, *, size: tuple[int, int] = (1080, 1080),
                 camera_position: Any = None, return_img: bool = True) -> np.ndarray:
    """Render one frame's enabled layers off-screen, return an RGB array."""
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, theme=brand.cinema_pv_theme(),
                    window_size=list(size))
    if scene.background.enabled:
        color: str = scene.background.color or brand.PALETTE["cine_bg"]
        pl.background_color = color  # type: ignore[assignment]
    if frame.kind == "grid":
        _grid_layers(pl, frame, scene)
    else:
        _surface_layers(pl, frame, scene)
    if camera_position is not None:
        pl.camera_position = camera_position
    else:
        pl.camera_position = "iso"
    img = pl.screenshot(return_img=return_img)
    pl.close()
    # screenshot returns None only when return_img=False; with return_img=True it
    # always returns an array. Cast to ndarray to satisfy the declared return type.
    return np.asarray(img)


from pathlib import Path
import imageio.v3 as iio
from . import camera as _camera


def render_animation(series, scene: Scene, *, out, fps: int = 30,
                     size: tuple[int, int] = (1080, 1080), orbit: bool = True,
                     formats: tuple[str, ...] = ("mp4", "webm")) -> list[Path]:
    """Render a Series to one movie per format with an optional orbit camera."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(series)
    positions = None
    if orbit and n > 0:
        center, radius = _camera.bounds_center_radius(series[0].mesh)
        positions = _camera.orbit_positions(center, radius, n)
    frames_rgb = []
    for i in range(n):
        frame = series[i]
        cam = positions[i] if positions is not None else None
        img = render_scene(frame, scene, size=size, camera_position=cam)
        frames_rgb.append(np.asarray(img)[..., :3])
    written = []
    for fmt in formats:
        path = out.with_suffix(f".{fmt}")
        iio.imwrite(path, frames_rgb, fps=fps)
        written.append(path)
    return written
