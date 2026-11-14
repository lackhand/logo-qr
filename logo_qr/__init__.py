"""Logo-centered artistic QR generator with QArt F2 steering and silhouette cutouts."""

from logo_qr.qart import QArtResult, encode_standard_qr, select_version, solve_qart
from logo_qr.render import (
    RenderedLogoQR,
    compute_logo_qr_stats,
    export_logo_qr_svg,
    load_logo_rgba,
    render_logo_qr,
)
from logo_qr.style import (
    LogoPlacement,
    LogoQRDesign,
    QRPalette,
    bt601_luma,
    build_layout_placements,
    parse_hex_rgb,
)

__all__ = [
    "LogoPlacement",
    "LogoQRDesign",
    "QArtResult",
    "QRPalette",
    "RenderedLogoQR",
    "bt601_luma",
    "build_layout_placements",
    "compute_logo_qr_stats",
    "encode_standard_qr",
    "export_logo_qr_svg",
    "load_logo_rgba",
    "parse_hex_rgb",
    "render_logo_qr",
    "select_version",
    "solve_qart",
]
