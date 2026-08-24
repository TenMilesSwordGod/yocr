"""CLI entrypoint: `yocr serve` or simply `python main.py`."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yocr", description="YOLO+OCR visual service for Android testing")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="start the HTTP server")
    serve.add_argument("--host", default=None, help="bind host (env YOCR_HOST)")
    serve.add_argument("--port", type=int, default=None, help="bind port (env YOCR_PORT)")
    serve.add_argument("--models-dir", default=None, help="directory with .pt models (env YOCR_MODELS_DIR)")
    serve.add_argument("--device", default=None, help="torch device: cpu|cuda:0 (env YOCR_DEVICE)")
    serve.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev only)")
    return parser


def apply_args(args: argparse.Namespace) -> None:
    import os

    if args.host:
        os.environ["YOCR_HOST"] = args.host
    if args.port:
        os.environ["YOCR_PORT"] = str(args.port)
    if getattr(args, "models_dir", None):
        os.environ["YOCR_MODELS_DIR"] = args.models_dir
    if getattr(args, "device", None):
        os.environ["YOCR_DEVICE"] = args.device


def main() -> None:
    import uvicorn

    from .config import get_settings

    args = build_parser().parse_args()
    command = args.command or "serve"
    apply_args(args)
    settings = get_settings()
    uvicorn.run(
        "yocr.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=bool(getattr(args, "reload", False)),
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
