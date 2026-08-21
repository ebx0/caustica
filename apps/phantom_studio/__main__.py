"""``python -m apps.phantom_studio`` — launch the local phantom studio."""

from __future__ import annotations

import argparse

from apps.phantom_studio.server import serve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m apps.phantom_studio",
        description="Local web GUI for building simulation-ready phantoms",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--preview-mvox",
        type=float,
        default=10.0,
        help="voxel budget (millions) for interactive previews; full builds ignore it",
    )
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.add_argument("--verbose", action="store_true", help="log every request and full tracebacks")
    args = p.parse_args(argv)
    serve(
        host=args.host,
        port=args.port,
        preview_mvox=args.preview_mvox,
        open_browser=not args.no_browser,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
