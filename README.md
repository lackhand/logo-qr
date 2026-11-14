# logo-qr

Generate scannable **QR codes with continuous rounded corridors** wrapping organically around arbitrary central vector (`.svg`) or raster (`.png`, `.webp`) logos — built for apparel printing, stickers, and posters.

## Highlights

- **Exact Clean URLs (No Fragment Hacks)**: Defaults to Reed-Solomon Error Correction Level **`H`** (`30%` recovery) using **Russ Cox QArt $\mathbb{F}_2$ Gaussian elimination** in `post_terminator` mode to steer free padding bits into a natural collar around the center cutout without altering your URL.
- **Arbitrary SVG & Raster Logos**:
  - Parses any `viewBox="minX minY W H"` SVG cleanly, rewrites per-placement `<defs>` IDs to avoid gradient/clip collisions when composing twin emblems, and emits **pure-vector `.svg` print files** (no embedded bitmap blobs for SVG inputs).
  - Supports transparent `.png` and `.webp` logos aspect-preservingly centered inside the QR matrix.
- **Contour-Hugging Cutouts & Whole-Alignment Suppression**:
  - Uses Euclidean distance transforms on the logo alpha silhouette to carve a tight organic halo (and crisp die-cut separator rims for tilted overlapping duos).
  - Protects finders, separators, timing strips, and format bits, and cleanly drops entire `5×5` interior alignment squares if touched by the logo halo so half-clipped squares never peek out.
- **Contrast Safety by Construction**:
  - Validates BT.601 normalized luminance (`wall_y <= 0.20`, `background_y >= 0.85`, `delta_y >= 0.65`) at configuration time so colored themes stay dark enough for real cameras.
- **Built-in Apple Vision + Camera/Fabric Stress Tests**:
  - Ships a standalone Swift CLI (`tools/vision_qr_check.swift`) invoking macOS `VNDetectBarcodesRequest` (`Revision 1`) alongside simulated 480px downscaling, JPEG blur, 12° 3D perspective tilt, and curved fabric drape tests.

## Quick Start

```bash
# 1. Create virtualenv / install dependencies (Python >= 3.11)
pip install -e ".[dev]"

# Requires ImageMagick (`magick`) in /opt/homebrew/bin/magick for SVG rasterization
# and Xcode Command Line Tools (`swiftc`) on macOS for --verify / Apple Vision tests.

# 2. Generate a single-logo QR code (PNG + pure-vector SVG)
logo-qr \
  --url "https://example.com/your-link" \
  --logo path/to/logo.svg \
  --layout single \
  --wall-color "#1E293B" \
  --accent-color "#0F172A" \
  --png-out out/qr.png \
  --svg-out out/qr.svg \
  --verify

# 3. Tilted duo with a purple corridor palette & outer lavender border
logo-qr \
  --url "https://example.com/your-link" \
  --logo path/to/logo.svg \
  --layout duo-splayed \
  --wall-color "#6002A0" \
  --accent-color "#42037B" \
  --frame-color "#9643D4" \
  --png-out out/duo.png \
  --svg-out out/duo.svg \
  --verify
```

### Layouts

- `single`: Single centered emblem (default `scale=0.35`, `angle=0°`)
- `duo-cozy`: Tight overlapping pair (`-18° / +22°`) with opaque die-cut rim
- `duo-splayed`: Wider jaunty pair (`-34° / +36°`) with die-cut rim
- `duo-chamber`: Overlapping tilted pair (`-24° / +28°`) nested inside a rounded circular doorway chamber

Pass `--logo` twice to pair two different logos in any `duo-*` layout.

## Python API

```python
from logo_qr import LogoQRDesign, QRPalette, build_layout_placements, export_logo_qr_svg, render_logo_qr

placements, chamber_r = build_layout_placements(["logo.svg"], "duo-cozy")
design = LogoQRDesign(
    ec_level="H",
    placements=placements,
    palette=QRPalette(wall_color="#064030", accent_color="#022C22", frame_color="#A7F3D0"),
    frame_mod=1.6,
    chamber_radius_frac=chamber_r,
)
rendered = render_logo_qr(design, url="https://example.com/your-link")
rendered.image.save("out.png")
export_logo_qr_svg(design, rendered, "out.svg")
```

## Running Tests

```bash
make build-verifier   # optional; tests auto-compile bin/vision_qr_check if missing
make test
```

## License

MIT
