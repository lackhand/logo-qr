"""Command-line interface for generating logo-centered QR PNG and SVG artwork."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from logo_qr.render import compute_logo_qr_stats, export_logo_qr_svg, render_logo_qr
from logo_qr.style import (
    VALID_EC_LEVELS,
    VALID_LAYOUT_MODES,
    LogoQRDesign,
    QRPalette,
    build_layout_placements,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _verify_png_with_vision(png_path: Path, expected_url: str) -> list[str]:
    vision_bin = REPO_ROOT / "bin" / "vision_qr_check"
    if not vision_bin.exists():
        swift_src = REPO_ROOT / "tools" / "vision_qr_check.swift"
        swiftc = Path("/usr/bin/swiftc")
        if not swift_src.exists() or not swiftc.exists():
            raise FileNotFoundError("vision_qr_check binary missing and cannot auto-compile")
        vision_bin.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(swiftc), "-O", str(swift_src), "-o", str(vision_bin)],
            check=True,
            capture_output=True,
            text=True,
        )
    proc = subprocess.run(
        [str(vision_bin), str(png_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    decoded = list(json.loads(proc.stdout.strip())["vision"])
    if expected_url not in decoded:
        raise RuntimeError(
            f"Apple Vision verification failed for {png_path}: expected {expected_url!r}, got {decoded}"
        )
    return decoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logo-qr",
        description="Generate scannable QR codes with continuous rounded corridors wrapping around central SVG or raster logos.",
    )
    parser.add_argument("--url", required=True, help="Target URL or text payload to encode")
    parser.add_argument(
        "--logo",
        action="append",
        required=True,
        help="Path to SVG, PNG, or WEBP logo (pass twice for two distinct duo logos)",
    )
    parser.add_argument(
        "--layout",
        default="single",
        choices=sorted(VALID_LAYOUT_MODES),
        help="Central logo composition preset",
    )
    parser.add_argument("--version", type=int, default=None, help="QR version 4..14 (default: auto)")
    parser.add_argument(
        "--ec-level",
        default="H",
        choices=sorted(VALID_EC_LEVELS),
        help="Reed-Solomon error correction level (default: H)",
    )
    parser.add_argument("--wall-color", default="#1E293B", help="Module wall #RRGGBB hex (BT.601 Y <= 0.20)")
    parser.add_argument("--accent-color", default="#0F172A", help="Dot/finder accent #RRGGBB hex (Y <= 0.20)")
    parser.add_argument("--background-color", default="#FFFFFF", help="Quiet-zone/paper #RRGGBB hex (Y >= 0.85)")
    parser.add_argument("--frame-color", default=None, help="Optional outer border #RRGGBB hex")
    parser.add_argument("--frame-mod", type=float, default=None, help="Outer frame width in modules (default: 1.6 if --frame-color)")
    parser.add_argument("--scale", type=float, default=None, help="Override logo scale fraction (0.05..0.70)")
    parser.add_argument("--angle-deg", type=float, default=None, help="Override logo rotation angle (-90..90)")
    parser.add_argument("--halo-mod", type=float, default=0.65, help="Clear halo margin around logo in modules")
    parser.add_argument("--module-px", type=int, default=24, help="PNG pixels per QR module (default: 24)")
    parser.add_argument("--no-run-dots", action="store_true", help="Disable accent spine dots on straight walls")
    parser.add_argument("--png-out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--svg-out", type=Path, default=None, help="Output SVG path")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify generated output via macOS Apple Vision QR decoder",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.png_out is None and args.svg_out is None:
        parser.error("Provide at least one of --png-out or --svg-out")

    try:
        palette = QRPalette(
            wall_color=args.wall_color,
            accent_color=args.accent_color,
            background_color=args.background_color,
            frame_color=args.frame_color,
        )
        placements, chamber_r = build_layout_placements(
            args.logo, args.layout, scale_frac=args.scale, angle_deg=args.angle_deg
        )
        frame_mod = (
            args.frame_mod
            if args.frame_mod is not None
            else (1.6 if args.frame_color is not None else 0.0)
        )
        design = LogoQRDesign(
            version=args.version,
            ec_level=args.ec_level,
            placements=placements,
            palette=palette,
            halo_pad_mod=args.halo_mod,
            frame_mod=frame_mod,
            chamber_radius_frac=chamber_r,
            module_px=args.module_px,
            use_run_dots=not args.no_run_dots,
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
                stats["vision_decoded"] = _verify_png_with_vision(args.png_out, args.url)
            else:
                raise RuntimeError("--verify currently requires --png-out")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(stats, indent=2))
    return 0
