"""TDD suite for crisp square QR modules, multi-layer CSS/SVG transforms, and robustness."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from logo_qr.render import (
    MAGICK_BIN,
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
    build_layers,
    parse_layer_spec,
)
from tests.qr_verifiers import (
    decode_qr_image,
    verify_camera_robustness,
    verify_qr_image,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
STAR_SVG = ASSETS_DIR / "star_badge.svg"
GEM_SVG = ASSETS_DIR / "gem_offset.svg"
TEST_URL = "https://example.com/projects/logo-qr?mode=demo"


def _rasterize_svg_with_magick(svg_path: Path, png_path: Path) -> None:
    res = subprocess.run(
        [str(MAGICK_BIN), str(svg_path), f"PNG24:{png_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0 and png_path.exists(), f"Magick SVG rasterize failed: {res.stderr}"


def test_palette_boundary_validation():
    with pytest.raises(ValueError, match="hex"):
        QRPalette(dark_color="1E293B")
    with pytest.raises(ValueError, match="hex"):
        QRPalette(dark_color="#GGGGGG")
    with pytest.raises(ValueError, match="dark_color"):
        QRPalette(dark_color="#9CA3AF")
    with pytest.raises(ValueError, match="light_color"):
        QRPalette(light_color="#334155")
    with pytest.raises(ValueError, match="light_color"):
        QRPalette(light_color="not-hex")

    pal = QRPalette(dark_color="#6002a0", light_color="#ffffff")
    assert pal.dark_color == "#6002A0"
    assert pal.light_color == "#FFFFFF"
    assert pal.dark_rgb == (96, 2, 160)
    assert pal.light_rgb == (255, 255, 255)


def test_affine2d_and_layer_parser_contracts():
    layers_default = build_layers([STAR_SVG, GEM_SVG])
    assert len(layers_default) == 2
    assert layers_default[0].logo_path == STAR_SVG
    assert layers_default[1].logo_path == GEM_SVG
    assert layers_default[0].transform == Affine2D.identity()
    assert layers_default[1].transform == Affine2D.identity()
    assert layers_default[0].die_cut_mod == 0.0 and layers_default[1].die_cut_mod == 0.0

    # Angle unit equivalence: turn, deg, rad
    _, m_turn, _ = parse_layer_spec("0: rotate(-0.25turn)")
    _, m_deg, _ = parse_layer_spec("0: rotate(-90deg)")
    _, m_rad, _ = parse_layer_spec(f"0: rotate({-math.pi / 2:.16f}rad)")
    for field in ("a", "b", "c", "d", "tx", "ty"):
        assert abs(getattr(m_turn, field) - getattr(m_deg, field)) < 1e-6
        assert abs(getattr(m_turn, field) - getattr(m_rad, field)) < 1e-6

    # Non-commutativity of ordered 2D affine transforms
    _, tr_rot, _ = parse_layer_spec("0: translate(5, 0) rotate(-0.25turn)")
    assert tr_rot.apply(0.0, 0.0) == pytest.approx((5.0, 0.0), abs=1e-6)
    _, rot_tr, _ = parse_layer_spec("0: rotate(-0.25turn) translate(5, 0)")
    assert rot_tr.apply(0.0, 0.0) == pytest.approx((0.0, -5.0), abs=1e-6)

    # translateX, translateY, 1-arg translate, anisotropic scale, and die-cut
    idx, mat, dc = parse_layer_spec(
        "1: translateX(1.5) translateY(-2.0) translate(0.5) scale(1.2, 0.8) die-cut(0.55)"
    )
    assert idx == 1
    assert dc == pytest.approx(0.55)
    assert mat.apply(0.0, 0.0) == pytest.approx((2.0, -2.0), abs=1e-6)
    assert mat.apply(1.0, 1.0) == pytest.approx((3.2, -1.2), abs=1e-6)


@pytest.mark.parametrize(
    "bad_spec",
    [
        "0 rotate(10deg)",
        "-1: rotate(10deg)",
        "0: ",
        "0: rotate(10deg); die-cut(0.5)",
        "0: rotate(10deg) bogus",
        "0: rotate(10deg",
        "0: rotate(-0.25)",
        "0: rotate(0)",
        "0: rotate(10grad)",
        "0: translate(5px, 0)",
        "0: translate(5%, 0)",
        "0: die-cut(1px)",
        "0: scale(0)",
        "0: scale(-1)",
        "0: scale(9)",
        "0: die-cut(-0.1)",
        "0: die-cut(5.0)",
        "0: die-cut(0.4) die-cut(0.5)",
        "0: unknown(1)",
    ],
)
def test_layer_parser_rejects_invalid_specs_fail_fast(bad_spec: str):
    with pytest.raises(ValueError):
        parse_layer_spec(bad_spec)


def test_layer_and_design_boundary_validation(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LogoLayer(logo_path=tmp_path / "does_not_exist.svg")

    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported logo format"):
        LogoLayer(logo_path=bad_ext)

    with pytest.raises(ValueError, match="die_cut_mod"):
        LogoLayer(logo_path=STAR_SVG, die_cut_mod=-0.1)
    with pytest.raises(ValueError, match="scale"):
        LogoLayer(logo_path=STAR_SVG, transform=Affine2D.scale(0.05))
    with pytest.raises(ValueError, match="tx"):
        LogoLayer(logo_path=STAR_SVG, transform=Affine2D.translate(40.0, 0.0))

    with pytest.raises(ValueError, match="layer"):
        build_layers([STAR_SVG, GEM_SVG], ["2: scale(1.0)"])
    with pytest.raises(ValueError, match="Duplicate"):
        build_layers([STAR_SVG], ["0: scale(0.9)", "0: rotate(10deg)"])
    with pytest.raises(ValueError, match="logo"):
        build_layers([])

    valid_layers = build_layers([STAR_SVG])
    with pytest.raises(ValueError, match="layers"):
        LogoQRDesign(layers=())
    with pytest.raises(ValueError, match="ec_level"):
        LogoQRDesign(layers=valid_layers, ec_level="Z")
    with pytest.raises(ValueError, match="base_scale_frac"):
        LogoQRDesign(layers=valid_layers, base_scale_frac=0.90)
    with pytest.raises(ValueError, match="halo_mod"):
        LogoQRDesign(layers=valid_layers, halo_mod=-0.5)
    with pytest.raises(ValueError, match="quiet_mod"):
        LogoQRDesign(layers=valid_layers, quiet_mod=1)
    with pytest.raises(ValueError, match="module_px"):
        LogoQRDesign(layers=valid_layers, module_px=4)

    # Off-grid translation and finder-corner collision must fail fast at render time
    off_grid = LogoQRDesign(
        version=8,
        ec_level="H",
        layers=build_layers([STAR_SVG], ["0: translate(30, 30)"]),
    )
    with pytest.raises(ValueError, match="outside"):
        render_logo_qr(off_grid, url=TEST_URL)

    hit_finder = LogoQRDesign(
        version=8,
        ec_level="H",
        layers=build_layers([STAR_SVG], ["0: translate(20, -20)"]),
    )
    with pytest.raises(ValueError, match="finder"):
        render_logo_qr(hit_finder, url=TEST_URL)


def test_load_logo_rgba_aspect_and_svg_origin(tmp_path: Path):
    svg_img = load_logo_rgba(GEM_SVG, 192)
    assert svg_img.mode == "RGBA" and svg_img.size == (192, 192)
    extrema = svg_img.getchannel("A").getextrema()
    assert extrema[0] == 0 and extrema[1] == 255

    rect_png = tmp_path / "wide.png"
    src = Image.new("RGBA", (160, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(src)
    draw.rounded_rectangle((8, 8, 152, 72), radius=12, fill=(239, 68, 68, 255))
    src.save(rect_png)

    loaded = load_logo_rgba(rect_png, 160)
    assert loaded.size == (160, 160)
    assert loaded.getpixel((80, 10))[3] == 0
    assert loaded.getpixel((80, 80))[3] == 255


def test_default_multi_layer_draw_order_overwrites_lower_layer():
    single_design = LogoQRDesign(version=8, ec_level="H", layers=build_layers([STAR_SVG]))
    stacked_design = LogoQRDesign(
        version=8, ec_level="H", layers=build_layers([STAR_SVG, GEM_SVG])
    )
    r_single = render_logo_qr(single_design, url=TEST_URL)
    r_stacked = render_logo_qr(stacked_design, url=TEST_URL)

    cx = r_single.image.width // 2
    cy = r_single.image.height // 2
    px_single = r_single.image.getpixel((cx, cy))
    px_stacked = r_stacked.image.getpixel((cx, cy))
    # star_badge center is red (#EF4444); gem_offset center is pink (#EC4899)
    assert px_single[0] > 200 and px_single[2] < 100
    assert px_stacked != px_single and px_stacked[2] > 130


def _assert_crisp_square_modules(rendered, design: LogoQRDesign) -> None:
    """Verify non-suppressed dark modules are exact uniform dark_rgb squares with zero ornaments."""
    arr = np.asarray(rendered.image)
    m = design.module_px
    off = rendered.pad_px_1x
    n = rendered.layout.size
    active_dark = (rendered.qart.matrix == 1) & (~rendered.suppressed)
    dark_rgb = np.array(design.palette.dark_rgb, dtype=np.uint8)

    checked = 0
    for r in range(n):
        for c in range(n):
            if active_dark[r, c]:
                block = arr[off + r * m : off + (r + 1) * m, off + c * m : off + (c + 1) * m]
                assert np.array_equal(block, np.broadcast_to(dark_rgb, block.shape)), (
                    f"Non-uniform pixels in dark module ({r}, {c})"
                )
                checked += 1
    assert checked > 100


def test_multilayer_square_qr_luminance_svg_parity_and_robustness(tmp_path: Path):
    raster_png = tmp_path / "badge.png"
    r_img = Image.new("RGBA", (160, 120), (0, 0, 0, 0))
    r_draw = ImageDraw.Draw(r_img)
    r_draw.ellipse((16, 10, 144, 110), fill=(245, 158, 11, 255))
    r_draw.rectangle((48, 36, 112, 84), fill=(37, 99, 235, 255))
    r_img.save(raster_png)

    raster_webp = tmp_path / "overlay.webp"
    w_img = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    w_draw = ImageDraw.Draw(w_img)
    w_draw.polygon([(60, 10), (110, 60), (60, 110), (10, 60)], fill=(16, 185, 129, 255))
    w_img.save(raster_webp, format="WEBP", lossless=True)

    designs = [
        (
            "single_star_v8h",
            LogoQRDesign(
                version=8,
                ec_level="H",
                layers=build_layers([STAR_SVG]),
                palette=QRPalette(dark_color="#1E293B", light_color="#FFFFFF"),
            ),
            False,
        ),
        (
            "stacked_default_v8h",
            LogoQRDesign(
                version=8,
                ec_level="H",
                layers=build_layers([STAR_SVG, GEM_SVG]),
                palette=QRPalette(dark_color="#0F172A", light_color="#FFFFFF"),
            ),
            False,
        ),
        (
            "tilted_twin_svg_turn_v8h",
            LogoQRDesign(
                version=8,
                ec_level="H",
                layers=build_layers(
                    [GEM_SVG, GEM_SVG],
                    [
                        "0: translate(-2.5, -0.4) rotate(-0.05turn) scale(0.86)",
                        "1: translate(2.5, 0.7) rotate(0.06turn) scale(0.88) die-cut(0.55)",
                    ],
                ),
                palette=QRPalette(dark_color="#1E1B4B", light_color="#FFFFFF"),
            ),
            False,
        ),
        (
            "mixed_svg_violet_v9h",
            LogoQRDesign(
                version=9,
                ec_level="H",
                layers=build_layers(
                    [STAR_SVG, GEM_SVG],
                    [
                        "0: translate(-3.0, 0.5) rotate(-32deg) scale(0.86)",
                        "1: translate(3.0, -0.5) rotate(34deg) scale(0.88) die-cut(0.55)",
                    ],
                ),
                palette=QRPalette(dark_color="#6002A0", light_color="#FFFFFF"),
            ),
            False,
        ),
        (
            "raster_png_webp_v8h",
            LogoQRDesign(
                version=8,
                ec_level="H",
                layers=build_layers(
                    [raster_png, raster_webp],
                    [
                        "0: translate(-2.2, 0.2) rotate(-0.04turn) scale(0.84)",
                        "1: translate(2.2, -0.2) rotate(0.05turn) scale(0.80) die-cut(0.50)",
                    ],
                ),
                palette=QRPalette(dark_color="#064030", light_color="#FFFFFF"),
            ),
            True,
        ),
    ]

    for name, design, has_raster in designs:
        rendered = render_logo_qr(design, url=TEST_URL)
        stats = compute_logo_qr_stats(design, rendered)

        assert stats["min_light_y"] >= 0.85, f"{name} min_light_y={stats['min_light_y']}"
        assert stats["max_dark_y"] <= 0.20, f"{name} max_dark_y={stats['max_dark_y']}"
        assert stats["delta_y"] >= 0.65, f"{name} delta_y={stats['delta_y']}"
        assert 0.05 <= stats["suppressed_fraction"] <= 0.22, (
            f"{name} suppressed_fraction={stats['suppressed_fraction']:.3f}"
        )
        assert stats["function_suppressed"] == 0, f"{name} suppressed protected function modules"
        assert stats["emblem_nonbg_pixels"] >= 100, f"{name} emblem did not render cleanly"

        _assert_crisp_square_modules(rendered, design)

        out_sub = tmp_path / name
        out_sub.mkdir(parents=True, exist_ok=True)
        png_path = out_sub / f"{name}.png"
        svg_path = out_sub / f"{name}.svg"
        rendered.image.save(png_path)
        export_logo_qr_svg(design, rendered, svg_path)

        svg_text = svg_path.read_text(encoding="utf-8")
        assert "<svg" in svg_text and "</svg>" in svg_text
        assert '<path id="qr-modules"' in svg_text
        assert design.palette.dark_color in svg_text

        if has_raster:
            assert "data:image/png;base64," in svg_text
        else:
            assert "data:image/" not in svg_text

        ids = re.findall(r'\bid="([^"]+)"', svg_text)
        assert len(ids) == len(set(ids)), f"{name} duplicate SVG IDs found: {ids}"

        # 5-transform physical camera & drape verification on PNG (OpenCV Aruco)
        verify_camera_robustness(
            png_path, TEST_URL, exact=True, required_engines=("opencv_aruco",)
        )
        verify_qr_image(png_path, TEST_URL)

        # Rasterized SVG must also decode cleanly with both Python QR engines
        svg_raster = out_sub / f"{name}_from_svg.png"
        _rasterize_svg_with_magick(svg_path, svg_raster)
        verify_qr_image(svg_raster, TEST_URL)
