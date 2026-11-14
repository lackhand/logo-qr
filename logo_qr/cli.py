"""Command-line interface for generating crisp square logo-centered QR PNG and SVG files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Sequence

from logo_qr.render import MAGICK_BIN, compute_logo_qr_stats, export_logo_qr_svg, render_logo_qr
from logo_qr.style import VALID_EC_LEVELS, LogoQRDesign, QRPalette, build_layers
from logo_qr.verify import verify_qr_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logo-qr",
        description="Embed central single or multi-layer SVG and raster logos inside scannable QR codes.",
    )
    parser.add_argument("--url", required=True, help="Target URL or text payload to encode")
    parser.add_argument(
        "--logo",
        action="append",
        required=True,
        help="Path to SVG, PNG, or WEBP logo (pass multiple times to stack centered layers in order)",
    )
    parser.add_argument(
        "--layer",
        action="append",
        default=[],
        help="Layer op spec '<idx>: translate(x,y) rotate(<angle>turn|deg|rad) scale(s) die-cut(w)'",
    )
    parser.add_argument("--version", type=int, default=None, help="QR version 4..14 (default: auto)")
    parser.add_argument(
        "--ec-level",
        default="H",
        choices=sorted(VALID_EC_LEVELS),
        help="Reed-Solomon error correction level (default: H)",
    )
    parser.add_argument("--dark-color", default="#1E293B", help="Dark module #RRGGBB hex (BT.601 Y <= 0.20)")
    parser.add_argument("--light-color", default="#FFFFFF", help="Light background #RRGGBB hex (BT.601 Y >= 0.85)")
    parser.add_argument(
        "--base-scale",
        type=float,
        default=0.34,
        help="Base logo box size as fraction of QR width (0.05..0.60, default: 0.34)",
    )
    parser.add_argument("--halo-mod", type=float, default=0.65, help="Clear halo margin around logo in modules")
    parser.add_argument("--module-px", type=int, default=24, help="PNG pixels per QR module (default: 24)")
    parser.add_argument("--quiet-mod", type=int, default=4, help="Quiet zone width in modules (>= 4)")
    parser.add_argument("--png-out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--svg-out", type=Path, default=None, help="Output SVG path")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify generated output via pure-Python OpenCV Aruco and standard QR decoders",
    )
    return parser


def _rasterize_and_verify_svg(svg_path: Path, expected_url: str) -> dict[str, list[str]]:
    if not MAGICK_BIN.exists():
        raise FileNotFoundError(
            f"ImageMagick binary required to verify SVG output not found at {MAGICK_BIN}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_png = Path(tmp) / "verified_svg.png"
        res = subprocess.run(
            [str(MAGICK_BIN), str(svg_path), f"PNG24:{tmp_png}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0 or not tmp_png.exists():
            raise RuntimeError(f"Failed rasterizing SVG {svg_path} via magick: {res.stderr}")
        return verify_qr_image(tmp_png, expected_url)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.png_out is None and args.svg_out is None:
        parser.error("Provide at least one of --png-out or --svg-out")

    try:
        palette = QRPalette(dark_color=args.dark_color, light_color=args.light_color)
        layers = build_layers(args.logo, args.layer)
        design = LogoQRDesign(
            version=args.version,
            ec_level=args.ec_level,
            layers=layers,
            palette=palette,
            base_scale_frac=args.base_scale,
            halo_mod=args.halo_mod,
            module_px=args.module_px,
            quiet_mod=args.quiet_mod,
        )
        rendered = render_logo_qr(design, url=args.url)
        stats = compute_logo_qr_stats(design, rendered)

        if args.png_out is not None:
            args.png_out.parent.mkdir(parents=True, exist_ok=True)
            rendered.image.save(args.png_out)
        if args.svg_out is not None:
            export_logo_qr_svg(design, rendered, args.svg_out)

        if args.verify:
            if args.png_out is not None:
                stats["verified_png"] = verify_qr_image(args.png_out, args.url)
            if args.svg_out is not None:
                stats["verified_svg"] = _rasterize_and_verify_svg(args.svg_out, args.url)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(stats, indent=2))
    return 0
