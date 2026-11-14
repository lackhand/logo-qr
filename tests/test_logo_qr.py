"""TDD suite for arbitrary central SVG/raster logo QR rendering and robustness."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

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
    LogoPlacement,
    LogoQRDesign,
    QRPalette,
    build_layout_placements,
)
from tests.qr_verifiers import decode_all, run_vision_qr, verify_camera_robustness

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
        QRPalette(wall_color="1E293B")
    with pytest.raises(ValueError, match="hex"):
        QRPalette(wall_color="#GGGGGG")
    with pytest.raises(ValueError, match="wall_color"):
        QRPalette(wall_color="#9CA3AF")
    with pytest.raises(ValueError, match="accent_color"):
        QRPalette(accent_color="#9CA3AF")
    with pytest.raises(ValueError, match="background_color"):
        QRPalette(background_color="#334155")
    with pytest.raises(ValueError, match="frame_color"):
        QRPalette(frame_color="not-hex")


def test_placement_and_design_boundary_validation(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LogoPlacement(logo_path=tmp_path / "does_not_exist.svg")

    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported logo format"):
        LogoPlacement(logo_path=bad_ext)

    with pytest.raises(ValueError, match="scale_frac"):
        LogoPlacement(logo_path=STAR_SVG, scale_frac=0.90)
    with pytest.raises(ValueError, match="scale_frac"):
        LogoPlacement(logo_path=STAR_SVG, scale_frac=0.01)
    with pytest.raises(ValueError, match="angle_deg"):
        LogoPlacement(logo_path=STAR_SVG, angle_deg=120.0)
    with pytest.raises(ValueError, match="dx_frac"):
        LogoPlacement(logo_path=STAR_SVG, dx_frac=0.60)
    with pytest.raises(ValueError, match="rim_mod"):
        LogoPlacement(logo_path=STAR_SVG, rim_mod=-0.1)

    valid_p = (LogoPlacement(logo_path=STAR_SVG),)
    with pytest.raises(ValueError, match="placements"):
        LogoQRDesign(placements=())
    with pytest.raises(ValueError, match="ec_level"):
        LogoQRDesign(placements=valid_p, ec_level="Z")
    with pytest.raises(ValueError, match="halo_pad_mod"):
        LogoQRDesign(placements=valid_p, halo_pad_mod=-0.5)
    with pytest.raises(ValueError, match="quiet_mod"):
        LogoQRDesign(placements=valid_p, quiet_mod=1)
    with pytest.raises(ValueError, match="frame_color"):
        LogoQRDesign(placements=valid_p, frame_mod=1.5, palette=QRPalette(frame_color=None))


def test_build_layout_placements_contracts(tmp_path: Path):
    p_single, ch0 = build_layout_placements([STAR_SVG], "single")
    assert len(p_single) == 1 and ch0 == 0.0

    p_duo1, ch1 = build_layout_placements([STAR_SVG], "duo-cozy")
    assert len(p_duo1) == 2 and ch1 == 0.0

    p_duo2, ch2 = build_layout_placements([STAR_SVG, GEM_SVG], "duo-chamber")
    assert len(p_duo2) == 2 and ch2 > 0.0
    assert p_duo2[0].logo_path == STAR_SVG and p_duo2[1].logo_path == GEM_SVG

    with pytest.raises(ValueError, match="layout"):
        build_layout_placements([STAR_SVG], "unknown-mode")
    with pytest.raises(ValueError, match="logo"):
        build_layout_placements([], "single")
    with pytest.raises(ValueError, match="single"):
        build_layout_placements([STAR_SVG, GEM_SVG], "single")
    with pytest.raises(ValueError, match="at most 2"):
        build_layout_placements([STAR_SVG, GEM_SVG, STAR_SVG], "duo-cozy")


def test_load_logo_rgba_aspect_and_svg_origin(tmp_path: Path):
    svg_img = load_logo_rgba(GEM_SVG, 192)
    assert svg_img.mode == "RGBA" and svg_img.size == (192, 192)
    extrema = svg_img.getchannel("A").getextrema()
    assert extrema[0] == 0 and extrema[1] == 255

    # Non-square transparent PNG should be centered aspect-preserving in 160x160 canvas
    rect_png = tmp_path / "wide.png"
    src = Image.new("RGBA", (160, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(src)
    draw.rounded_rectangle((8, 8, 152, 72), radius=12, fill=(239, 68, 68, 255))
    src.save(rect_png)

    loaded = load_logo_rgba(rect_png, 160)
    assert loaded.size == (160, 160)
    # Top row must remain transparent after aspect-preserving 2:1 centering
    assert loaded.getpixel((80, 10))[3] == 0
    assert loaded.getpixel((80, 80))[3] == 255


def test_arbitrary_logo_layouts_luminance_and_robustness(tmp_path: Path):
    # Dynamic non-square raster logo fixture
    raster_logo = tmp_path / "emblem.png"
    r_img = Image.new("RGBA", (160, 120), (0, 0, 0, 0))
    r_draw = ImageDraw.Draw(r_img)
    r_draw.ellipse((16, 10, 144, 110), fill=(245, 158, 11, 255))
    r_draw.rectangle((48, 36, 112, 84), fill=(37, 99, 235, 255))
    r_img.save(raster_logo)

    p_single, _ = build_layout_placements([STAR_SVG], "single")
    p_cozy, _ = build_layout_placements([GEM_SVG], "duo-cozy")
    p_splayed, _ = build_layout_placements([STAR_SVG], "duo-splayed")
    p_chamber, ch_r = build_layout_placements([STAR_SVG], "duo-chamber")
    p_raster, _ = build_layout_placements([raster_logo], "single", scale_frac=0.33)

    designs = [
        (
            "single_star_v8h",
            LogoQRDesign(
                version=8,
                ec_level="H",
                placements=p_single,
                palette=QRPalette(wall_color="#1E293B", accent_color="#0F172A"),
            ),
            False,
        ),
        (
            "duo_cozy_gem_v8h",
            LogoQRDesign(
                version=8,
                ec_level="H",
                placements=p_cozy,
                palette=QRPalette(wall_color="#1E1B4B", accent_color="#0F172A"),
                halo_pad_mod=0.60,
            ),
            False,
        ),
        (
            "duo_splayed_violet_v9h",
            LogoQRDesign(
                version=9,
                ec_level="H",
                placements=p_splayed,
                palette=QRPalette(
                    wall_color="#6002A0",
                    accent_color="#42037B",
                    frame_color="#9643D4",
                ),
                frame_mod=1.6,
            ),
            False,
        ),
        (
            "duo_chamber_emerald_v9h",
            LogoQRDesign(
                version=9,
                ec_level="H",
                placements=p_chamber,
                palette=QRPalette(
                    wall_color="#064030",
                    accent_color="#022C22",
                    frame_color="#A7F3D0",
                ),
                frame_mod=1.6,
                chamber_radius_frac=ch_r,
            ),
            False,
        ),
        (
            "single_raster_v8h",
            LogoQRDesign(
                version=8,
                ec_level="H",
                placements=p_raster,
                palette=QRPalette(wall_color="#111827", accent_color="#030712"),
            ),
            True,
        ),
    ]

    for name, design, is_raster in designs:
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

        out_sub = tmp_path / name
        out_sub.mkdir(parents=True, exist_ok=True)
        png_path = out_sub / f"{name}.png"
        svg_path = out_sub / f"{name}.svg"
        rendered.image.save(png_path)
        export_logo_qr_svg(design, rendered, svg_path)

        svg_text = svg_path.read_text(encoding="utf-8")
        assert "<svg" in svg_text and "</svg>" in svg_text
        assert design.palette.wall_color in svg_text

        # Vector SVGs must stay pure vector; raster logos embed lossless PNG base64
        if is_raster:
            assert "data:image/png;base64," in svg_text
        else:
            assert "data:image/" not in svg_text

        # Ensure per-placement <defs> IDs were uniquified cleanly when SVG has gradients
        ids = re.findall(r'\bid="([^"]+)"', svg_text)
        assert len(ids) == len(set(ids)), f"{name} duplicate SVG IDs found: {ids}"

        # 5-transform physical camera & drape verification on PNG
        results = verify_camera_robustness(
            png_path, TEST_URL, exact=True, required_engines=("vision",)
        )
        for tname, eng in results.items():
            assert TEST_URL in eng["vision"], f"{name} transform={tname} failed Vision: {eng}"
        assert decode_all(png_path)["vision"] == [TEST_URL]

        # SVG rasterized verification via ImageMagick -> Apple Vision
        svg_raster = out_sub / f"{name}_from_svg.png"
        _rasterize_svg_with_magick(svg_path, svg_raster)
        assert TEST_URL in run_vision_qr(svg_raster), f"{name} exported SVG failed Vision decode"
