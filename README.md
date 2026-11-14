# logo-qr

Embed central single or multi-layer SVG and raster logos inside crisp, scannable square-module QR codes — producing both high-DPI `.png` and pure-vector `.svg` files for web, stickers, posters, and apparel.

## Highlights

- **Exact Clean URLs (No Fragment Hacks)**: Defaults to Reed-Solomon Error Correction Level **`H`** (`30%` recovery) and uses **Russ Cox QArt $\mathbb{F}_2$ Gaussian elimination** (`post_terminator` mode) to steer free padding bits behind the center cutout to match emblem luminance without modifying your payload.
- **Standard Crisp Square QR Modules**: Emits sharp `1×1` square data modules and standard `7×7` square finder patterns (merged into a single compact `<path id="qr-modules">` in `.svg` exports) for maximum camera reliability.
- **Arbitrary Multi-Layer Logos (`--logo` + `--layer`)**:
  - Pass `--logo` multiple times to draw centered overlapping layers in draw order (`0, 1, ...`).
  - Pass `--layer '<idx>: <ops>'` using standard **W3C CSS/SVG transform syntax** (`translate(dx, dy)`, `translateX(dx)`, `translateY(dy)`, `rotate(<angle>turn|deg|rad)`, `scale(sx[, sy])`) plus `die-cut(width)` in QR module units.
  - Parses any `viewBox="minX minY W H"` SVG cleanly, rewrites per-layer `<defs>` IDs to avoid gradient/clip collisions, and emits pure-vector `.svg` output for SVG logos alongside aspect-preserving `.png` and `.webp` support.
- **Contour-Hugging Cutouts & Whole-Alignment Suppression**:
  - Uses Euclidean distance transforms on the composited logo alpha silhouette to carve a clean organic halo around central emblems.
  - Protects finders, separators, timing strips, and format bits, and drops entire `5×5` interior alignment blocks if touched by the logo halo so half-clipped squares never appear.
- **Contrast Safety by Construction**:
  - Validates BT.601 normalized luminance (`dark_y <= 0.20`, `light_y >= 0.85`, `delta_y >= 0.65`) at configuration time so custom dark themes stay scannable by real cameras.
- **Pure-Python QR Verification & Camera/Fabric Stress Suite**:
  - Built-in Python verifier (`logo_qr.verify`) checks both OpenCV Aruco (`cv2.QRCodeDetectorAruco`) and OpenCV standard (`cv2.QRCodeDetector`) decoders on clean exports, and checks OpenCV Aruco across `480px` downscaling, Gaussian JPEG blur, `12°` 3D perspective tilt, and curved fabric drape.

## Quick Start

```bash
# 1. Install dependencies (Python >= 3.11; ImageMagick `magick` used for SVG rasterization)
pip install -e ".[dev]"

# 2. Generate a single-logo QR code (PNG + pure-vector SVG)
logo-qr \
  --url "https://example.com/your-link" \
  --logo path/to/logo.svg \
  --dark-color "#1E293B" \
  --png-out out/qr.png \
  --svg-out out/qr.svg \
  --verify

# 3. Compose two overlapping tilted layers with a die-cut rim on the upper layer
logo-qr \
  --url "https://example.com/your-link" \
  --logo path/to/back.svg \
  --logo path/to/front.svg \
  --layer "0: translate(-2.6, -0.3) rotate(-0.05turn) scale(0.86)" \
  --layer "1: translate(2.6, 0.4) rotate(20deg) scale(0.88) die-cut(0.55)" \
  --dark-color "#6002A0" \
  --png-out out/layered.png \
  --svg-out out/layered.svg \
  --verify
```

### Layer Operations (`--layer`)

Operations in `--layer "<idx>: ..."` compose left-to-right in **QR module units** relative to the matrix center (`+X` right, `+Y` down; positive rotation is clockwise, matching SVG/CSS):

- `translate(dx)` / `translate(dx, dy)` — translate by unitless QR modules
- `translateX(dx)` / `translateY(dy)` — single-axis translation in QR modules
- `rotate(<val>turn|deg|rad)` — rotate with an explicit CSS angle unit (e.g., `-0.25turn` = `-90deg`)
- `scale(s)` / `scale(sx, sy)` — scale around the layer center
- `die-cut(width)` — dilate a `light_color` Euclidean rim (`0.0..3.0` modules) beneath that layer

## Python API

```python
from logo_qr import LogoQRDesign, QRPalette, build_layers, export_logo_qr_svg, render_logo_qr

layers = build_layers(
    ["back.svg", "front.svg"],
    [
        "0: translate(-2.5, -0.4) rotate(-0.05turn) scale(0.86)",
        "1: translate(2.5, 0.7) rotate(0.06turn) scale(0.88) die-cut(0.55)",
    ],
)
design = LogoQRDesign(
    ec_level="H",
    layers=layers,
    palette=QRPalette(dark_color="#064030", light_color="#FFFFFF"),
)
rendered = render_logo_qr(design, url="https://example.com/your-link")
rendered.image.save("out.png")
export_logo_qr_svg(design, rendered, "out.svg")
```

## Running Tests

```bash
make test
```

## License

MIT
