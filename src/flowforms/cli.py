"""flowforms command-line interface."""
from __future__ import annotations
import argparse
from pathlib import Path

from . import diagnostics as diag
from . import figures
from .recipes import load_recipe


def _render_decay(diagnostics_path, quantities, out) -> None:
    d = diag.load(diagnostics_path)
    fig = figures.decay_plot(d, tuple(quantities) if quantities else ("energy", "enstrophy"))
    figures.save(fig, out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="flowforms")
    sub = parser.add_subparsers(dest="command", required=True)

    fig = sub.add_parser("figure", help="render a publication figure")
    fig.add_argument("--recipe")
    fig.add_argument("--diagnostics")
    fig.add_argument("--quantity", action="append", default=[])
    fig.add_argument("--out")

    cine_p = sub.add_parser("cine", help="render a cinematic animation")
    cine_p.add_argument("--recipe")
    cine_p.add_argument("--series")
    cine_p.add_argument("--out")
    cine_p.add_argument("--fps", type=int, default=30)
    cine_p.add_argument("--glow", action=argparse.BooleanOptionalAction, default=None)
    cine_p.add_argument("--streamlines", action=argparse.BooleanOptionalAction, default=None)
    cine_p.add_argument("--arrows", action=argparse.BooleanOptionalAction, default=None)
    cine_p.add_argument("--slice", action=argparse.BooleanOptionalAction, default=None)
    cine_p.add_argument("--iso", action=argparse.BooleanOptionalAction, default=None)

    movie_p = sub.add_parser("movie", help="composite field-over-plot animation")
    movie_p.add_argument("--recipe")
    movie_p.add_argument("--series")
    movie_p.add_argument("--diagnostics")
    movie_p.add_argument("--quantity", default=None)
    movie_p.add_argument("--out")
    movie_p.add_argument("--fps", type=int, default=30)

    hero_p = sub.add_parser("hero", help="build the colliding-rings hero piece")
    hero_p.add_argument("--recipe")
    hero_p.add_argument("--series")
    hero_p.add_argument("--diagnostics")
    hero_p.add_argument("--out-dir")
    hero_p.add_argument("--impact-time", type=float)
    hero_p.add_argument("--handle", default="")
    hero_p.add_argument("--caption", default="")
    hero_p.add_argument("--title", default="")
    hero_p.add_argument("--quantity", default=None)

    args = parser.parse_args(argv)

    if args.command == "figure":
        if args.recipe:
            r = load_recipe(args.recipe)
            if r.get("kind") == "decay":
                _render_decay(r["diagnostics"], r.get("quantities", []), r["out"])
                return 0
            raise SystemExit(f"unknown recipe kind: {r.get('kind')!r}")
        if args.diagnostics and args.out:
            _render_decay(args.diagnostics, args.quantity, args.out)
            return 0
        raise SystemExit("figure: provide --recipe, or --diagnostics with --out")

    if args.command == "cine":
        from . import cine as cine_mod
        from . import io as fio
        from .recipes import scene_from_recipe
        from .scene import Scene
        if args.recipe:
            r = load_recipe(args.recipe)
            for key in ("series", "out"):
                if key not in r:
                    raise SystemExit(
                        f"cine recipe missing required key {key!r} in {args.recipe!r}")
            scene = scene_from_recipe(r)
            series = fio.load_series(r["series"])
            out = r["out"]
            fps = int(r.get("fps", 30))
        else:
            if not (args.series and args.out):
                raise SystemExit("cine: provide --recipe, or --series with --out")
            scene = Scene.default_grid()
            # BooleanOptionalAction: None means flag not given; keep Scene default.
            if args.glow is not None:
                scene.glow.enabled = args.glow
            if args.streamlines is not None:
                scene.streamlines.enabled = args.streamlines
            if args.arrows is not None:
                scene.arrows.enabled = args.arrows
            if args.slice is not None:
                scene.slice.enabled = args.slice
            if args.iso is not None:
                scene.isosurface.enabled = args.iso
            series = fio.load_series(args.series)
            out = args.out
            fps = args.fps
        cine_mod.render_animation(series, scene, out=out, fps=fps)
        return 0

    if args.command == "movie":
        from . import io as fio, diagnostics as fdiag, composite as fcomp
        from .scene import Scene
        if args.recipe:
            r = load_recipe(args.recipe)
            for key in ("series", "diagnostics", "out"):
                if key not in r:
                    raise SystemExit(
                        f"movie recipe missing required key {key!r} in {args.recipe!r}")
            series_path = r["series"]
            diag_path = r["diagnostics"]
            out = r["out"]
            quantity = r.get("quantity", "enstrophy")
            fps = int(r.get("fps", 30))
        else:
            for attr, name in (("series", "--series"), ("diagnostics", "--diagnostics"), ("out", "--out")):
                if not getattr(args, attr):
                    raise SystemExit(
                        f"movie: provide --recipe, or {name} (missing)")
            series_path = args.series
            diag_path = args.diagnostics
            out = args.out
            quantity = args.quantity or "enstrophy"
            fps = args.fps
        fcomp.render_composite_animation(
            fio.load_series(series_path), fdiag.load(diag_path),
            Scene.default_grid(), quantity=quantity, out=out, fps=fps)
        return 0

    if args.command == "hero":
        from . import io as fio, diagnostics as fdiag, hero as fhero
        if args.recipe:
            r = load_recipe(args.recipe)
            for key in ("series", "diagnostics", "out_dir"):
                if key not in r:
                    raise SystemExit(
                        f"hero recipe missing required key {key!r} in {args.recipe!r}")
            series_path = r["series"]
            diag_path = r["diagnostics"]
            out_dir = r["out_dir"]
            impact_time = r.get("impact_time", args.impact_time)
            handle = r.get("handle", args.handle)
            caption = r.get("caption", args.caption)
            title = r.get("title", args.title)
            quantity = r.get("quantity", "enstrophy")
        else:
            for attr, name in (("series", "--series"), ("diagnostics", "--diagnostics"), ("out_dir", "--out-dir")):
                if not getattr(args, attr):
                    raise SystemExit(
                        f"hero: provide --recipe, or {name} (missing)")
            series_path = args.series
            diag_path = args.diagnostics
            out_dir = args.out_dir
            impact_time = args.impact_time
            handle = args.handle
            caption = args.caption
            title = args.title
            quantity = args.quantity or "enstrophy"
        fhero.build_hero(
            fio.load_series(series_path), fdiag.load(diag_path),
            out_dir=out_dir, impact_time=impact_time,
            handle=handle, caption=caption, title=title)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
