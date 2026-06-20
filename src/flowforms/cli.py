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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
