"""Track B: cinematic layered scene rendering (PyVista off-screen)."""
from __future__ import annotations
from typing import Any
import numpy as np
import pyvista as pv

from . import brand
from .io import Frame
from .scene import Scene


# Sentinel: distinguishes "caller did not pass streamline_tubes" from "caller
# passed None" (which means skip streamlines explicitly).
class _Unset:
    pass


_UNSET = _Unset()


def _display_label(field: str) -> str:
    """Map internal field names to legible display labels for scalar bars.

    VTK cannot render LaTeX/bold, so we use Unicode approximations.
    """
    _MAP = {
        "velocity_mag": "‖u‖",   # ||u||
        "omega_mag": "‖ω‖",  # ||omega||
        "qcriterion": "Q",
    }
    return _MAP.get(field, field)


def _apply_cm_sans_to_scalar_bar(pl: pv.Plotter) -> None:
    """Set the scalar bar actor's title and label text to CM Sans (cmss10.ttf).

    Uses VTK's SetFontFamily(VTK_FONT_FILE) + SetFontFile(path) on both the
    title text property and the label text property. Falls back silently if the
    TTF is missing or VTK refuses the font file; the cinema theme default font
    is used in that case.
    """
    try:
        import vtk
        font_path = brand.cm_sans_font_file()
        actors = pl.renderer.GetActors2D()
        actors.InitTraversal()
        while True:
            actor = actors.GetNextActor2D()
            if actor is None:
                break
            # ScalarBarActor has GetLabelTextProperty and GetTitleTextProperty.
            get_label = getattr(actor, "GetLabelTextProperty", None)
            get_title = getattr(actor, "GetTitleTextProperty", None)
            if get_label is None or get_title is None:
                continue
            for prop in (get_label(), get_title()):
                if prop is None:
                    continue
                prop.SetFontFamily(vtk.VTK_FONT_FILE)  # type: ignore[attr-defined]
                prop.SetFontFile(font_path)
    except Exception:
        # Font wiring is best-effort; fall back to theme default silently.
        pass


def _glow_scalar_bar_args(field: str) -> Any:
    """Scalar-bar styling for the glow field: a readable light-colored label
    on the dark cinematic background. Returned as Any to satisfy pyvista's
    strict ScalarBarArgs TypedDict for add_mesh/add_volume."""
    light = brand.PALETTE["paper"]
    args: Any = dict(title=field, color=light, title_font_size=14,
                     label_font_size=11, n_labels=3, vertical=True,
                     position_x=0.88, position_y=0.10, width=0.08, height=0.55)
    return args


def streamline_tubes(frame: Frame, scene: Scene) -> Any:
    """Compute streamline tube geometry for one frame.

    Returns the PyVista tube mesh on success, or None if seeding fails or the
    vector field is absent. This is the reusable computation that the caching
    layer in render_animation / render_composite_animation calls and holds
    between frames.
    """
    mesh = frame.mesh
    if not scene.streamlines.enabled or scene.streamlines.vectors not in mesh.point_data:
        return None
    try:
        stream: Any = mesh.streamlines(
            vectors=scene.streamlines.vectors,
            n_points=scene.streamlines.n_points,
            source_radius=(mesh.length / 2) or 1.0,
        )
        tubes: Any = stream.tube(radius=scene.streamlines.radius)
        return tubes
    except Exception:
        return None


def _grid_layers(pl: pv.Plotter, frame: Frame, scene: Scene,
                 _streamline_tubes: Any = _UNSET) -> None:
    """Add grid-type scene layers to an already-created Plotter.

    _streamline_tubes controls the streamlines layer:
      - _UNSET (default): compute from frame on the fly (original behavior).
      - None: skip the streamlines layer entirely.
      - a mesh: draw this precomputed tube mesh directly.
    """
    mesh = frame.mesh
    if scene.glow.enabled:
        try:
            f = frame.derive(scene.glow.field)
            cmap = scene.glow.cmap or brand.EMISSIVE_CMAP
            # When an isosurface is present it owns the single scalar bar; a
            # glow-only scene keeps its own bar. This avoids a cluttered
            # double-colorbar in the hero (faint glow + Q-criterion isosurface).
            glow_bar = not scene.isosurface.enabled
            pl.add_volume(f.mesh, scalars=scene.glow.field, cmap=cmap,
                          opacity="linear", opacity_unit_distance=1.0,
                          show_scalar_bar=glow_bar,
                          scalar_bar_args=_glow_scalar_bar_args(scene.glow.field))
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
                        cmap=brand.field_cmap_for_name(scene.slice.field))
        except Exception:
            # slice can fail on degenerate mesh extents; skip gracefully.
            pass
    if scene.isosurface.enabled:
        try:
            f = frame.derive(scene.isosurface.field)
            if scene.isosurface.values:
                vals = list(scene.isosurface.values)
            else:
                # Auto-threshold: use a high percentile of the positive values so
                # we land firmly inside vortex cores rather than at the near-zero
                # mean (which gives garbage geometry for Q-criterion).
                arr = f.array(scene.isosurface.field)
                pos = arr[arr > 0]
                if pos.size > 0:
                    thresh = float(np.percentile(pos, 90))
                else:
                    # Fallback when no positive values: mean + 1.5 std (non-zero
                    # spread still gives a surface; degenerate arrays are caught
                    # by the outer except).
                    thresh = float(np.nanmean(arr) + 1.5 * np.nanstd(arr))
                vals = [thresh]
            iso: Any = f.mesh.contour(vals, scalars=scene.isosurface.field)
            # Color the isosurface by velocity_mag for perceptual richness; fall
            # back to the primary field if velocity is unavailable.
            try:
                f_vel = frame.derive("velocity_mag")
                iso_colored: Any = iso
                # Map velocity_mag onto the isosurface via interpolation.
                iso_colored = iso.sample(f_vel.mesh)
                color_field = "velocity_mag"
            except Exception:
                iso_colored = iso
                color_field = scene.isosurface.field
            sbar_args: Any = dict(
                title=_display_label(color_field),
                color=brand.PALETTE["paper"],
                title_font_size=26,
                label_font_size=12,
                n_labels=3,
                vertical=False,
                position_x=0.30,
                position_y=0.06,
                width=0.40,
                height=0.05,
            )
            pl.add_mesh(  # type: ignore[arg-type]
                iso_colored,
                scalars=color_field,
                cmap=brand.field_cmap("magnitude"),
                smooth_shading=True,
                show_scalar_bar=True,
                scalar_bar_args=sbar_args,
            )
            _apply_cm_sans_to_scalar_bar(pl)
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
    # Streamlines: three modes controlled by _streamline_tubes.
    #   _UNSET -> compute from frame now (original per-frame behavior).
    #   None   -> skip entirely (caller disabled or caching decided to skip).
    #   mesh   -> draw precomputed tubes from the cache.
    if isinstance(_streamline_tubes, _Unset):
        # Original behavior: compute on the fly.
        tubes_to_draw = streamline_tubes(frame, scene)
    else:
        tubes_to_draw = _streamline_tubes
    if tubes_to_draw is not None:
        try:
            pl.add_mesh(  # type: ignore[arg-type]
                tubes_to_draw,
                cmap=scene.streamlines.cmap or brand.field_cmap("magnitude"),
                opacity=scene.streamlines.opacity,
                show_scalar_bar=False,  # decorative layer; isosurface owns the bar
            )
        except Exception:
            pass  # drawing can still fail on degenerate meshes; skip gracefully


def _surface_layers(pl: pv.Plotter, frame: Frame, scene: Scene) -> None:
    mesh = frame.mesh
    if scene.glow.enabled:
        cmap = scene.glow.cmap or brand.EMISSIVE_CMAP
        # Determine which scalar to display: prefer glow.field, fall back to
        # the first available scalar, or render plain geometry if none exist.
        all_scalars = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        glow_field = scene.glow.field
        if glow_field not in all_scalars:
            glow_field = all_scalars[0] if all_scalars else None
        try:
            if glow_field is not None:
                pl.add_mesh(mesh, scalars=glow_field, cmap=cmap,
                            smooth_shading=True, show_scalar_bar=True,
                            scalar_bar_args=_glow_scalar_bar_args(glow_field))
            else:
                pl.add_mesh(mesh, smooth_shading=True)
        except Exception:
            pl.add_mesh(mesh, smooth_shading=True)
    else:
        pl.add_mesh(mesh, smooth_shading=True)

    # Nematic director glyphs: render double-ended directors as two glyph sets
    # (director and its negation) so the field reads as headless/bidirectional.
    for nematic_name in frame.nematic_fields():
        try:
            from .io import base_name as _base_name
            label = _base_name(nematic_name)
            directors = np.asarray(mesh.point_data[nematic_name])
            # Build a copy of the mesh with the negated director vectors.
            neg_mesh = mesh.copy()
            neg_mesh[nematic_name] = -directors
            # Forward glyphs
            fwd: Any = mesh.glyph(orient=nematic_name, scale=False, factor=0.1,
                                  tolerance=0.05)
            pl.add_mesh(fwd, color=brand.PALETTE["gold"])  # type: ignore[arg-type]
            # Backward glyphs (negated direction)
            bwd: Any = neg_mesh.glyph(orient=nematic_name, scale=False, factor=0.1,
                                      tolerance=0.05)
            pl.add_mesh(bwd, color=brand.PALETTE["gold"])  # type: ignore[arg-type]
        except Exception:
            pass  # director rendering is best-effort; skip on failure


def render_scene(frame: Frame, scene: Scene, *, size: tuple[int, int] = (1080, 1080),
                 camera_position: Any = None, return_img: bool = True,
                 streamline_tubes: Any = _UNSET) -> np.ndarray:
    """Render one frame's enabled layers off-screen, return an RGB array.

    streamline_tubes controls the streamlines layer for grid frames:
      - omitted / _UNSET: compute per-frame (original behavior, back-compat).
      - None: skip streamlines entirely for this frame.
      - a PyVista mesh: draw this precomputed tube mesh (enables caching).
    """
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, theme=brand.cinema_pv_theme(),
                    window_size=list(size))
    if scene.background.enabled:
        color: str = scene.background.color or brand.PALETTE["cine_bg"]
        pl.background_color = color  # type: ignore[assignment]
    if frame.kind == "grid":
        _grid_layers(pl, frame, scene, _streamline_tubes=streamline_tubes)
    else:
        _surface_layers(pl, frame, scene)
    if camera_position is not None:
        pl.camera_position = camera_position
    else:
        pl.camera_position = "iso"
    img = pl.screenshot(return_img=return_img)
    pl.close()
    # screenshot returns None only when return_img=False; with return_img=True it
    # always returns an array. Slice to 3-channel RGB here so every caller gets a
    # consistent shape without needing its own [..., :3] guard.
    return np.asarray(img)[..., :3]


from pathlib import Path
import imageio.v3 as iio
from PIL import Image
from . import camera as _camera


_CODECS = {"webm": "libvpx-vp9", "mp4": "libx264", "mov": "libx264"}


def _imwrite_video(path: Path, frames, *, fps: int) -> None:
    """Write a video, picking a container-appropriate codec. imageio-ffmpeg
    otherwise defaults to libx264 for every suffix, which a WebM container
    rejects ('Only VP8/VP9/AV1') and breaks the pipe."""
    fmt = path.suffix.lstrip(".").lower()
    kwargs: dict[str, Any] = {"fps": fps}
    codec = _CODECS.get(fmt)
    if codec is not None:
        kwargs["codec"] = codec
    iio.imwrite(path, frames, **kwargs)


def _even(n: int, multiple: int = 2) -> int:
    """Round n up to the nearest multiple (default 2; 16 is safest for h264)."""
    if n % multiple == 0:
        return n
    return n + (multiple - n % multiple)


def normalize_frames(frames, *, multiple: int = 16) -> list[np.ndarray]:
    """Resize every frame to one common shape whose width/height are a multiple
    of ``multiple`` (>=2, even). Returns a perfectly uniform list so video
    encoders never see a frame-size change mid-stream -- this is the fix for the
    imageio-ffmpeg broken-pipe error caused by inconsistent frame dimensions."""
    arrs = [np.asarray(f)[..., :3] for f in frames]
    if not arrs:
        return arrs
    target_h = _even(max(a.shape[0] for a in arrs), multiple)
    target_w = _even(max(a.shape[1] for a in arrs), multiple)
    out = []
    for a in arrs:
        if a.shape[0] != target_h or a.shape[1] != target_w:
            a = np.asarray(
                Image.fromarray(a.astype(np.uint8)).resize((target_w, target_h)))
        out.append(np.ascontiguousarray(a.astype(np.uint8))[..., :3])
    return out


def render_animation(series, scene: Scene, *, out, fps: int = 30,
                     size: tuple[int, int] = (1080, 1080), orbit: bool = True,
                     formats: tuple[str, ...] = ("mp4", "webm"),
                     orbit_revolutions: float = 1.0) -> list[Path]:
    """Render a Series to one movie per format with an optional orbit camera."""
    n = len(series)
    if n == 0:
        raise ValueError("empty series; nothing to render")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    positions = None
    if orbit:
        center, radius = _camera.bounds_center_radius(series[0].mesh)
        positions = _camera.orbit_positions(center, radius, n,
                                            revolutions=orbit_revolutions)
    frames_rgb = []
    # Streamline cache: recompute tubes every update_every frames; hold in between.
    tubes_cache: Any = None
    update_every = max(1, scene.streamlines.update_every) if scene.streamlines.enabled else 1
    for i in range(n):
        frame = series[i]
        cam = positions[i] if positions is not None else None
        if scene.streamlines.enabled:
            if tubes_cache is None or i % update_every == 0:
                tubes_cache = streamline_tubes(frame, scene)
            st_arg: Any = tubes_cache
        else:
            st_arg = None
        img = render_scene(frame, scene, size=size, camera_position=cam,
                           streamline_tubes=st_arg)
        frames_rgb.append(img)
    frames_rgb = normalize_frames(frames_rgb)
    written = []
    for fmt in formats:
        path = out.with_suffix(f".{fmt}")
        _imwrite_video(path, frames_rgb, fps=fps)
        written.append(path)
    return written
