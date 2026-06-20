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
    cine_p.add_argument("--no-glow", action="store_true")
    cine_p.add_argument("--streamlines", action="store_true")
    cine_p.add_argument("--arrows", action="store_true")
    cine_p.add_argument("--slice", action="store_true")
    cine_p.add_argument("--iso", action="store_true")

    movie_p = sub.add_parser("movie", help="composite field-over-plot animation")
    movie_p.add_argument("--series", required=True)
    movie_p.add_argument("--diagnostics", required=True)
    movie_p.add_argument("--quantity", default="enstrophy")
    movie_p.add_argument("--out", required=True)
    movie_p.add_argument("--fps", type=int, default=30)

    hero_p = sub.add_parser("hero", help="build the colliding-rings hero piece")
    hero_p.add_argument("--series", required=True)
    hero_p.add_argument("--diagnostics", required=True)
    hero_p.add_argument("--out-dir", required=True)
    hero_p.add_argument("--impact-time", type=float)
    hero_p.add_argument("--handle", default="")
    hero_p.add_argument("--caption", default="")

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
            scene = scene_from_recipe(r)
            series = fio.load_series(r["series"])
            out = r["out"]
            fps = int(r.get("fps", 30))
        else:
            if not (args.series and args.out):
                raise SystemExit("cine: provide --recipe, or --series with --out")
            scene = Scene.default_grid()
            if args.no_glow:
                scene.glow.enabled = False
            scene.streamlines.enabled = args.streamlines or scene.streamlines.enabled
            scene.arrows.enabled = args.arrows
            scene.slice.enabled = args.slice
            scene.isosurface.enabled = args.iso
            series = fio.load_series(args.series)
            out = args.out
            fps = args.fps
        cine_mod.render_animation(series, scene, out=out, fps=fps)
        return 0

    if args.command == "movie":
        from . import io as fio, diagnostics as fdiag, composite as fcomp
        from .scene import Scene
        fcomp.render_composite_animation(
            fio.load_series(args.series), fdiag.load(args.diagnostics),
            Scene.default_grid(), quantity=args.quantity, out=args.out, fps=args.fps)
        return 0
    if args.command == "hero":
        from . import io as fio, diagnostics as fdiag, hero as fhero
        fhero.build_hero(
            fio.load_series(args.series), fdiag.load(args.diagnostics),
            out_dir=args.out_dir, impact_time=args.impact_time,
            handle=args.handle, caption=args.caption)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
