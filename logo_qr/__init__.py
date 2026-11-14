"""Logo-centered QR generator with QArt F2 steering and CSS/SVG multi-layer composition."""

from logo_qr.qart import QArtResult, encode_standard_qr, select_version, solve_qart
from logo_qr.render import (
    RenderedLogoQR,
    compute_logo_qr_stats,
    export_logo_qr_svg,
    load_logo_rgba,
    render_logo_qr,
)
from logo_qr.style import (
    Affine2D,
    LogoLayer,
    LogoQRDesign,
    QRPalette,
    bt601_luma,
    build_layers,
    parse_hex_rgb,
    parse_layer_spec,
)
from logo_qr.verify import decode_qr_image, verify_qr_image

__all__ = [
    "Affine2D",
    "LogoLayer",
    "LogoQRDesign",
    "QArtResult",
    "QRPalette",
    "RenderedLogoQR",
    "bt601_luma",
    "build_layers",
    "compute_logo_qr_stats",
    "decode_qr_image",
    "encode_standard_qr",
    "export_logo_qr_svg",
    "load_logo_rgba",
    "parse_hex_rgb",
    "parse_layer_spec",
    "render_logo_qr",
    "select_version",
    "solve_qart",
    "verify_qr_image",
]
