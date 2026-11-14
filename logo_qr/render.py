"""Logo-centered artistic QR renderer (2x-supersampled PNG and pure-vector SVG)."""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import math
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from logo_qr.qart import QArtResult, select_version, solve_qart
from logo_qr.qr_spec import QRMatrixLayout, build_layout
from logo_qr.style import LogoPlacement, LogoQRDesign

MAGICK_BIN = Path("/opt/homebrew/bin/magick")
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

_SKIP_RIM_TAGS = frozenset({"defs", "style", "title", "desc", "metadata", "linearGradient", "radialGradient", "pattern", "clipPath", "mask", "filter"})
_GEOM_TAGS = frozenset({"path", "circle", "ellipse", "rect", "polygon", "polyline", "line"})
_NUM_PREFIX_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


@dataclass(frozen=True)
class RenderedLogoQR:
    """Rendered output paired with solved QR matrix and suppression metadata."""

    image: Image.Image
    qart: QArtResult
    layout: QRMatrixLayout
    suppressed: np.ndarray          # (n, n) bool
    protected_function: np.ndarray  # (n, n) bool
    pad_px_1x: int


def _parse_svg_number(raw: str | None) -> float | None:
    if not raw:
        return None
    m = _NUM_PREFIX_RE.match(raw)
    return float(m.group(1)) if m else None


def _parse_svg_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, width, height) from root <svg> viewBox or width/height."""
    vb_raw = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if vb_raw:
        parts = [p for p in re.split(r"[\s,]+", vb_raw.strip()) if p]
        if len(parts) != 4:
            raise ValueError(f"Malformed SVG viewBox attribute: {vb_raw!r}")
        min_x, min_y, width, height = (float(x) for x in parts)
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"SVG viewBox dimensions must be > 0, got {width}x{height}")
        return min_x, min_y, width, height

    width = _parse_svg_number(root.attrib.get("width"))
    height = _parse_svg_number(root.attrib.get("height"))
    if width is None or height is None or width <= 0.0 or height <= 0.0:
        raise ValueError("SVG must define a valid viewBox or positive width/height attributes")
    return 0.0, 0.0, width, height


@lru_cache(maxsize=64)
def _load_logo_rgba_cached(resolved_path: str, mtime_ns: int, box_px: int) -> Image.Image:
    del mtime_ns  # Cache invalidation key only
    path = Path(resolved_path)
    if path.suffix.lower() == ".svg":
        if not MAGICK_BIN.exists():
            raise FileNotFoundError(
                f"ImageMagick binary required for SVG rasterization not found at {MAGICK_BIN}"
            )
        with tempfile.TemporaryDirectory() as tmp:
            out_png = Path(tmp) / "logo.png"
            res = subprocess.run(
                [
                    str(MAGICK_BIN),
                    "-background",
                    "none",
                    "-density",
                    "1200",
                    resolved_path,
                    "-resize",
                    f"{box_px}x{box_px}",
                    f"PNG32:{out_png}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0 or not out_png.exists():
                raise RuntimeError(f"Failed rendering SVG {resolved_path} via magick: {res.stderr}")
            with Image.open(out_png) as im:
                rgba = im.convert("RGBA").copy()
    else:
        with Image.open(path) as im:
            src = im.convert("RGBA")
            rgba = ImageOps.contain(src, (box_px, box_px), method=Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (box_px, box_px), (0, 0, 0, 0))
    ox = (box_px - rgba.width) // 2
    oy = (box_px - rgba.height) // 2
    canvas.alpha_composite(rgba, (ox, oy))
    return canvas


def load_logo_rgba(logo_path: Path | str, box_px: int) -> Image.Image:
    """Load an SVG or RGBA raster logo fitted aspect-preserving into a square box_px canvas."""
    if box_px <= 0:
        raise ValueError(f"box_px must be > 0, got {box_px}")
    path = Path(logo_path)
    if not path.exists():
        raise FileNotFoundError(f"Logo file not found: {path}")
    resolved = path.resolve()
    mtime_ns = resolved.stat().st_mtime_ns
    return _load_logo_rgba_cached(str(resolved), mtime_ns, int(box_px)).copy()


def _build_protected_and_alignment_masks(
    layout: QRMatrixLayout,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Identify protected QR structure modules (finders/timing/format) and interior alignments."""
    n = layout.size
    in_finder = np.zeros((n, n), dtype=bool)
    for fr, fc in ((0, 0), (0, n - 7), (n - 7, 0)):
        in_finder[fr : fr + 7, fc : fc + 7] = True

    finder_plus_sep = np.zeros((n, n), dtype=bool)
    finder_plus_sep[0:8, 0:8] = True
    finder_plus_sep[0:8, n - 8 : n] = True
    finder_plus_sep[n - 8 : n, 0:8] = True

    timing = np.zeros((n, n), dtype=bool)
    timing[6, :] = True
    timing[:, 6] = True

    valid_centers: list[tuple[int, int]] = []
    alignment_all = np.zeros((n, n), dtype=bool)
    for ar in layout.spec.alignment_centers:
        for ac in layout.spec.alignment_centers:
            if finder_plus_sep[ar - 2 : ar + 3, ac - 2 : ac + 3].any():
                continue
            valid_centers.append((ar, ac))
            alignment_all[ar - 2 : ar + 3, ac - 2 : ac + 3] = True

    protected = (layout.is_function & (~alignment_all)) | finder_plus_sep | timing
    return in_finder, protected, valid_centers


def _dilate_alpha_rim(
    sprite_rgba: Image.Image, rim_px: float, rim_rgb: tuple[int, int, int]
) -> Image.Image:
    """Create a canvas padded with an opaque Euclidean die-cut rim beneath a rotated sprite."""
    pad = max(2, int(math.ceil(rim_px)) + 3)
    w, h = sprite_rgba.size
    canvas = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    canvas.alpha_composite(sprite_rgba, (pad, pad))
    if rim_px <= 0.0:
        return canvas

    arr = np.asarray(canvas)
    fg = (arr[:, :, 3] >= 24).astype(np.uint8)
    dist = cv2.distanceTransform(1 - fg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    rim_alpha = np.clip(rim_px + 0.5 - dist, 0.0, 1.0)
    rim_rgba = np.zeros_like(arr)
    rim_rgba[:, :, 0] = rim_rgb[0]
    rim_rgba[:, :, 1] = rim_rgb[1]
    rim_rgba[:, :, 2] = rim_rgb[2]
    rim_rgba[:, :, 3] = np.round(rim_alpha * 255.0).astype(np.uint8)

    rim_img = Image.fromarray(rim_rgba, mode="RGBA")
    rim_img.alpha_composite(canvas)
    return rim_img


def _composite_emblem_canvas(
    design: LogoQRDesign, grid_ss: int, m_ss: int
) -> tuple[Image.Image, np.ndarray]:
    """Render placed logos, die-cut rims, and optional chamber onto RGBA canvas and silhouette."""
    emblem = Image.new("RGBA", (grid_ss, grid_ss), (0, 0, 0, 0))
    silhouette = np.zeros((grid_ss, grid_ss), dtype=bool)
    bg_rgb = design.palette.background_rgb

    for p in design.placements:
        base_px = max(16, int(round(p.scale_frac * grid_ss)))
        raw = load_logo_rgba(p.logo_path, base_px)
        # Positive angle_deg rotates clockwise, matching SVG rotate(+angle_deg)
        rotated = raw.rotate(-p.angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
        layered = _dilate_alpha_rim(rotated, rim_px=p.rim_mod * m_ss, rim_rgb=bg_rgb)

        cx = grid_ss * (0.5 + p.dx_frac)
        cy = grid_ss * (0.5 + p.dy_frac)
        ox = int(round(cx - layered.width / 2.0))
        oy = int(round(cy - layered.height / 2.0))
        emblem.alpha_composite(layered, (ox, oy))

        la = np.asarray(layered)[:, :, 3] >= 24
        x0 = max(0, ox)
        y0 = max(0, oy)
        x1 = min(grid_ss, ox + layered.width)
        y1 = min(grid_ss, oy + layered.height)
        if x1 > x0 and y1 > y0:
            silhouette[y0:y1, x0:x1] |= la[y0 - oy : y1 - oy, x0 - ox : x1 - ox]

    if design.chamber_radius_frac > 0.0:
        r_ch = design.chamber_radius_frac * grid_ss
        stroke_px = 2.0 * design.wall_radius_frac * m_ss
        yy, xx = np.ogrid[:grid_ss, :grid_ss]
        dist_c = np.hypot(xx - (grid_ss / 2.0), yy - (grid_ss / 2.0))
        silhouette |= dist_c <= (r_ch + stroke_px * 0.5)

    return emblem, silhouette


def _compute_suppressed_modules(
    layout: QRMatrixLayout,
    design: LogoQRDesign,
    silhouette: np.ndarray,
    protected_function: np.ndarray,
    valid_centers: list[tuple[int, int]],
    m_ss: int,
) -> np.ndarray:
    """Suppress QR modules overlapped by logo + halo and drop clipped 5x5 alignment blocks."""
    n = layout.size
    fg = silhouette.astype(np.uint8)
    dist_to_art = cv2.distanceTransform(1 - fg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    halo_px = design.halo_pad_mod * m_ss

    suppressed = np.zeros((n, n), dtype=bool)
    for r in range(n):
        y0, y1 = r * m_ss, (r + 1) * m_ss
        for c in range(n):
            if protected_function[r, c]:
                continue
            x0, x1 = c * m_ss, (c + 1) * m_ss
            if np.any(dist_to_art[y0:y1, x0:x1] <= halo_px):
                suppressed[r, c] = True

    for ar, ac in valid_centers:
        if suppressed[ar - 2 : ar + 3, ac - 2 : ac + 3].any():
            for rr in range(ar - 2, ar + 3):
                for cc in range(ac - 2, ac + 3):
                    if not protected_function[rr, cc]:
                        suppressed[rr, cc] = True

    return suppressed


def _draw_chamber_ring_pil(
    canvas: Image.Image, grid_origin_ss: int, grid_ss: int, m_ss: int, design: LogoQRDesign
) -> None:
    """Draw circular chamber arc with a top doorway and accent endpoint nodes."""
    if design.chamber_radius_frac <= 0.0:
        return
    draw = ImageDraw.Draw(canvas)
    cx = grid_origin_ss + grid_ss / 2.0
    cy = grid_origin_ss + grid_ss / 2.0
    r_ch = design.chamber_radius_frac * grid_ss
    half_w = design.wall_radius_frac * m_ss
    wall_rgb = design.palette.wall_rgb
    accent_rgb = design.palette.accent_rgb

    start_deg = -90.0 + design.chamber_door_half_deg
    end_deg = 270.0 - design.chamber_door_half_deg
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), 180)

    for a in angles:
        px = cx + r_ch * math.cos(a)
        py = cy + r_ch * math.sin(a)
        draw.ellipse((px - half_w, py - half_w, px + half_w, py + half_w), fill=wall_rgb)

    hub_r = 0.65 * m_ss
    for a in (angles[0], angles[-1]):
        hx = cx + r_ch * math.cos(a)
        hy = cy + r_ch * math.sin(a)
        draw.ellipse((hx - hub_r, hy - hub_r, hx + hub_r, hy + hub_r), fill=wall_rgb)
        draw.ellipse(
            (hx - 0.34 * m_ss, hy - 0.34 * m_ss, hx + 0.34 * m_ss, hy + 0.34 * m_ss),
            fill=accent_rgb,
        )

    if design.use_run_dots:
        dot_r = 0.16 * m_ss
        for da in np.linspace(start_deg + 14.0, end_deg - 14.0, 16):
            rad = math.radians(float(da))
            dx = cx + r_ch * math.cos(rad)
            dy = cy + r_ch * math.sin(rad)
            draw.ellipse((dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r), fill=accent_rgb)


def _resolve_version(design: LogoQRDesign, url: str) -> int:
    if design.version is not None:
        return design.version
    return select_version(url, ec_level=design.ec_level, min_version=8, min_free_bytes=8)


def render_logo_qr(design: LogoQRDesign, url: str) -> RenderedLogoQR:
    """Solve QArt matrix around suppressed central silhouette and render 2x-supersampled PNG."""
    version = _resolve_version(design, url)
    layout = build_layout(version, design.ec_level)
    n = layout.size
    m_ss = 2 * design.module_px
    grid_ss = n * m_ss
    pad_px_1x = int(round((design.quiet_mod + design.frame_mod) * design.module_px))
    grid_origin_ss = 2 * pad_px_1x
    total_ss = grid_origin_ss + grid_ss + grid_origin_ss

    wall_rgb = design.palette.wall_rgb
    accent_rgb = design.palette.accent_rgb
    bg_rgb = design.palette.background_rgb
    frame_rgb = design.palette.frame_rgb

    in_finder, protected, valid_centers = _build_protected_and_alignment_masks(layout)
    emblem, silhouette = _composite_emblem_canvas(design, grid_ss, m_ss)
    suppressed = _compute_suppressed_modules(
        layout, design, silhouette, protected, valid_centers, m_ss
    )

    target = np.zeros((n, n), dtype=np.uint8)
    priority = np.full((n, n), 0.1, dtype=np.float64)
    supp_u8 = suppressed.astype(np.uint8)
    collar = (
        cv2.dilate(supp_u8, np.ones((5, 5), np.uint8)) > 0
    ) & (~suppressed) & (~layout.is_function)
    target[collar] = 1
    priority[collar] = 12.0
    priority[suppressed] = 0.0

    qart = solve_qart(url=url, version=version, ec_level=design.ec_level, target=target, priority=priority)
    mat = qart.matrix

    canvas_bg = frame_rgb if (design.frame_mod > 0.0 and frame_rgb is not None) else bg_rgb
    canvas = Image.new("RGB", (total_ss, total_ss), canvas_bg)
    draw = ImageDraw.Draw(canvas)

    if design.frame_mod > 0.0 and frame_rgb is not None:
        frame_ss = int(round(design.frame_mod * m_ss))
        draw.rounded_rectangle(
            (frame_ss, frame_ss, total_ss - 1 - frame_ss, total_ss - 1 - frame_ss),
            radius=int(round(0.35 * m_ss)),
            fill=bg_rgb,
        )

    for fr, fc in ((0, 0), (0, n - 7), (n - 7, 0)):
        x0 = grid_origin_ss + fc * m_ss
        y0 = grid_origin_ss + fr * m_ss
        draw.rounded_rectangle(
            (x0, y0, x0 + 7 * m_ss, y0 + 7 * m_ss),
            radius=int(round(design.finder_outer_radius_frac * m_ss)),
            fill=wall_rgb,
        )
        draw.rounded_rectangle(
            (x0 + m_ss, y0 + m_ss, x0 + 6 * m_ss, y0 + 6 * m_ss),
            radius=int(round(0.45 * m_ss)),
            fill=bg_rgb,
        )
        draw.rounded_rectangle(
            (x0 + 2 * m_ss, y0 + 2 * m_ss, x0 + 5 * m_ss, y0 + 5 * m_ss),
            radius=int(round(0.60 * m_ss)),
            fill=accent_rgb,
        )

    active = (mat == 1) & (~suppressed) & (~in_finder)
    wall_r = design.wall_radius_frac * m_ss

    for r in range(n):
        cy = grid_origin_ss + (r + 0.5) * m_ss
        for c in range(n):
            if not active[r, c]:
                continue
            cx = grid_origin_ss + (c + 0.5) * m_ss
            draw.ellipse((cx - wall_r, cy - wall_r, cx + wall_r, cy + wall_r), fill=wall_rgb)
            if c + 1 < n and active[r, c + 1]:
                draw.rectangle((cx, cy - wall_r, cx + m_ss, cy + wall_r), fill=wall_rgb)
            if r + 1 < n and active[r + 1, c]:
                draw.rectangle((cx - wall_r, cy, cx + wall_r, cy + m_ss), fill=wall_rgb)
            if r + 1 < n and c + 1 < n and active[r + 1, c] and active[r, c + 1] and active[r + 1, c + 1]:
                draw.rectangle((cx, cy, cx + m_ss, cy + m_ss), fill=wall_rgb)

    if design.use_run_dots:
        dot_r = 0.16 * m_ss
        for r in range(n):
            cy = grid_origin_ss + (r + 0.5) * m_ss
            for c in range(n):
                if not active[r, c]:
                    continue
                horiz = (c > 0 and active[r, c - 1]) and (c + 1 < n and active[r, c + 1])
                vert = (r > 0 and active[r - 1, c]) and (r + 1 < n and active[r + 1, c])
                if horiz or vert:
                    cx = grid_origin_ss + (c + 0.5) * m_ss
                    draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r), fill=accent_rgb)

    rgba_full = canvas.convert("RGBA")
    _draw_chamber_ring_pil(rgba_full, grid_origin_ss, grid_ss, m_ss, design)
    rgba_full.alpha_composite(emblem, (grid_origin_ss, grid_origin_ss))

    out_1x = rgba_full.convert("RGB").resize(
        (total_ss // 2, total_ss // 2), resample=Image.Resampling.LANCZOS
    )
    return RenderedLogoQR(
        image=out_1x,
        qart=qart,
        layout=layout,
        suppressed=suppressed,
        protected_function=protected,
        pad_px_1x=pad_px_1x,
    )


def compute_logo_qr_stats(design: LogoQRDesign, rendered: RenderedLogoQR) -> dict[str, object]:
    """Measure suppressed fraction, BT.601 module contrast separation, and center logo coverage."""
    mat = rendered.qart.matrix
    suppressed = rendered.suppressed
    n = rendered.layout.size
    m = design.module_px
    off = rendered.pad_px_1x

    arr = np.asarray(rendered.image, dtype=np.float64) / 255.0
    centers_rc = off + np.arange(n) * m + (m // 2)
    sample_rgb = arr[np.ix_(centers_rc, centers_rc)]
    luma = 0.299 * sample_rgb[:, :, 0] + 0.587 * sample_rgb[:, :, 1] + 0.114 * sample_rgb[:, :, 2]

    active_light = (~suppressed) & (mat == 0)
    active_dark = (~suppressed) & (mat == 1)
    min_light_y = float(np.min(luma[active_light]))
    max_dark_y = float(np.max(luma[active_dark]))

    # Count pixels inside suppressed center modules that differ noticeably from background_rgb
    grid_rgb = np.asarray(rendered.image, dtype=np.int16)[off : off + n * m, off : off + n * m]
    supp_px = np.repeat(np.repeat(suppressed, m, axis=0), m, axis=1)
    bg_arr = np.array(design.palette.background_rgb, dtype=np.int16)
    diff_bg = np.max(np.abs(grid_rgb - bg_arr), axis=2) >= 20
    emblem_nonbg_pixels = int((supp_px & diff_bg).sum())

    return {
        "version": rendered.layout.version,
        "ec_level": rendered.layout.ec_level,
        "size": n,
        "active_modules": int((~suppressed).sum()),
        "suppressed_modules": int(suppressed.sum()),
        "suppressed_fraction": float(suppressed.sum() / (n * n)),
        "function_suppressed": int((suppressed & rendered.protected_function).sum()),
        "min_light_y": min_light_y,
        "max_dark_y": max_dark_y,
        "delta_y": min_light_y - max_dark_y,
        "emblem_nonbg_pixels": emblem_nonbg_pixels,
        "mask": rendered.qart.mask,
        "satisfied_weight": float(rendered.qart.satisfied_weight),
        "total_weight": float(rendered.qart.total_weight),
    }


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _prefix_svg_ids(node: ET.Element, prefix: str) -> None:
    """Rewrite all XML IDs and #id / url(#id) references inside node to prevent collisions."""
    id_map: dict[str, str] = {}
    for elem in node.iter():
        old_id = elem.attrib.get("id")
        if old_id:
            new_id = f"{prefix}{old_id}"
            id_map[old_id] = new_id
            elem.attrib["id"] = new_id

    if not id_map:
        return

    pattern = re.compile(
        r"(url\(\s*#|#|\bhref\s*=\s*['\"]#)(" + "|".join(re.escape(k) for k in sorted(id_map, key=len, reverse=True)) + r")(\b)"
    )

    def _replace_urls(val: str) -> str:
        val = re.sub(
            r"url\(\s*#(" + "|".join(re.escape(k) for k in id_map) + r")\s*\)",
            lambda m: f"url(#{id_map[m.group(1)]})",
            val,
        )
        for old_id, new_id in id_map.items():
            if val == f"#{old_id}":
                return f"#{new_id}"
        return val

    for elem in node.iter():
        for attr, val in list(elem.attrib.items()):
            if attr != "id":
                elem.attrib[attr] = _replace_urls(val)
        if elem.text and "#" in elem.text:
            elem.text = pattern.sub(lambda m: f"{m.group(1)}{id_map[m.group(2)]}{m.group(3)}", elem.text)


def _clone_svg_rim_element(elem: ET.Element, bg_hex: str, stroke_w: str) -> ET.Element | None:
    """Clone drawable SVG geometry elements with an opaque background-colored stroke rim."""
    tag = _local_tag(elem.tag)
    if tag in _SKIP_RIM_TAGS:
        return None

    out = ET.Element(elem.tag, dict(elem.attrib))
    for drop_attr in ("id", "opacity", "fill-opacity", "stroke-opacity", "clip-path", "mask", "filter"):
        out.attrib.pop(drop_attr, None)

    if tag in _GEOM_TAGS:
        out.attrib["fill"] = bg_hex
        out.attrib["stroke"] = bg_hex
        out.attrib["stroke-width"] = stroke_w
        out.attrib["stroke-linejoin"] = "round"
        out.attrib["stroke-linecap"] = "round"
        # Clear inline style fill/stroke properties so presentation attributes dominate cleanly
        if "style" in out.attrib:
            out.attrib.pop("style")

    for child in elem:
        cloned_child = _clone_svg_rim_element(child, bg_hex, stroke_w)
        if cloned_child is not None:
            out.append(cloned_child)
    return out


def _append_svg_logo_group(
    parent: ET.Element,
    placement: LogoPlacement,
    idx: int,
    origin: float,
    n: int,
    bg_hex: str,
) -> None:
    cx_p = origin + n * (0.5 + placement.dx_frac)
    cy_p = origin + n * (0.5 + placement.dy_frac)

    tree = ET.parse(placement.logo_path)
    src_root = tree.getroot()
    min_x, min_y, vb_w, vb_h = _parse_svg_viewbox(src_root)
    scale = (placement.scale_frac * n) / max(vb_w, vb_h)
    vb_cx = min_x + vb_w / 2.0
    vb_cy = min_y + vb_h / 2.0

    group = ET.SubElement(
        parent,
        f"{{{SVG_NS}}}g",
        {
            "id": f"logo-{idx}",
            "transform": (
                f"translate({cx_p:.4f} {cy_p:.4f}) "
                f"rotate({placement.angle_deg:.3f}) "
                f"scale({scale:.6f}) "
                f"translate({-vb_cx:.4f} {-vb_cy:.4f})"
            ),
        },
    )

    if placement.rim_mod > 0.0:
        sw = f"{(2.0 * placement.rim_mod) / scale:.4f}"
        rim_group = ET.SubElement(group, f"{{{SVG_NS}}}g", {"id": f"logo-{idx}-rim"})
        for child in src_root:
            rim_child = _clone_svg_rim_element(child, bg_hex, sw)
            if rim_child is not None:
                rim_group.append(rim_child)

    fg_group = ET.SubElement(group, f"{{{SVG_NS}}}g", {"id": f"logo-{idx}-art"})
    for child in src_root:
        fg_group.append(copy.deepcopy(child))
    _prefix_svg_ids(fg_group, f"logo{idx}-")


def _append_raster_logo_group(
    parent: ET.Element,
    placement: LogoPlacement,
    idx: int,
    origin: float,
    n: int,
    bg_rgb: tuple[int, int, int],
) -> None:
    cx_p = origin + n * (0.5 + placement.dx_frac)
    cy_p = origin + n * (0.5 + placement.dy_frac)
    box_mod = placement.scale_frac * n

    with Image.open(placement.logo_path) as im:
        src = im.convert("RGBA")

    # Apply die-cut rim at high resolution when requested on raster logos
    if placement.rim_mod > 0.0:
        px_per_mod = 512.0 / box_mod
        layered = _dilate_alpha_rim(src, rim_px=placement.rim_mod * px_per_mod, rim_rgb=bg_rgb)
        scale_w = layered.width / px_per_mod
        scale_h = layered.height / px_per_mod
        embed_img = layered
    else:
        aspect = src.width / src.height if src.height else 1.0
        if aspect >= 1.0:
            scale_w = box_mod
            scale_h = box_mod / aspect
        else:
            scale_h = box_mod
            scale_w = box_mod * aspect
        embed_img = src

    buf = BytesIO()
    embed_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    group = ET.SubElement(
        parent,
        f"{{{SVG_NS}}}g",
        {
            "id": f"logo-{idx}",
            "transform": f"translate({cx_p:.4f} {cy_p:.4f}) rotate({placement.angle_deg:.3f})",
        },
    )
    ET.SubElement(
        group,
        f"{{{SVG_NS}}}image",
        {
            "x": f"{-scale_w / 2.0:.4f}",
            "y": f"{-scale_h / 2.0:.4f}",
            "width": f"{scale_w:.4f}",
            "height": f"{scale_h:.4f}",
            "href": f"data:image/png;base64,{b64}",
        },
    )


def export_logo_qr_svg(
    design: LogoQRDesign, rendered: RenderedLogoQR, out_path: Path | str
) -> Path:
    """Export synchronized print-ready SVG (pure-vector for SVG logos, lossless embed for raster)."""
    out_path = Path(out_path)
    layout = rendered.layout
    n = layout.size
    mat = rendered.qart.matrix
    suppressed = rendered.suppressed
    in_finder, _, _ = _build_protected_and_alignment_masks(layout)

    origin = design.quiet_mod + design.frame_mod
    total_mod = n + 2.0 * origin
    wr = design.wall_radius_frac
    active = (mat == 1) & (~suppressed) & (~in_finder)

    wall_hex = design.palette.wall_color
    accent_hex = design.palette.accent_hex
    bg_hex = design.palette.background_color
    frame_hex = design.palette.frame_color

    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": "1200",
            "height": "1200",
            "viewBox": f"0 0 {total_mod:.4f} {total_mod:.4f}",
        },
    )

    outer_fill = frame_hex if (design.frame_mod > 0.0 and frame_hex is not None) else bg_hex
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {"width": f"{total_mod:.4f}", "height": f"{total_mod:.4f}", "fill": outer_fill},
    )

    if design.frame_mod > 0.0 and frame_hex is not None:
        f = design.frame_mod
        ET.SubElement(
            root,
            f"{{{SVG_NS}}}rect",
            {
                "x": f"{f:.4f}",
                "y": f"{f:.4f}",
                "width": f"{total_mod - 2 * f:.4f}",
                "height": f"{total_mod - 2 * f:.4f}",
                "rx": "0.35",
                "fill": bg_hex,
            },
        )

    finders_g = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "finders"})
    for fr, fc in ((0, 0), (0, n - 7), (n - 7, 0)):
        fx, fy = origin + fc, origin + fr
        ET.SubElement(
            finders_g,
            f"{{{SVG_NS}}}rect",
            {
                "x": f"{fx:.4f}",
                "y": f"{fy:.4f}",
                "width": "7",
                "height": "7",
                "rx": f"{design.finder_outer_radius_frac:.3f}",
                "fill": wall_hex,
            },
        )
        ET.SubElement(
            finders_g,
            f"{{{SVG_NS}}}rect",
            {
                "x": f"{fx + 1:.4f}",
                "y": f"{fy + 1:.4f}",
                "width": "5",
                "height": "5",
                "rx": "0.45",
                "fill": bg_hex,
            },
        )
        ET.SubElement(
            finders_g,
            f"{{{SVG_NS}}}rect",
            {
                "x": f"{fx + 2:.4f}",
                "y": f"{fy + 2:.4f}",
                "width": "3",
                "height": "3",
                "rx": "0.60",
                "fill": accent_hex,
            },
        )

    walls_g = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "corridors", "fill": wall_hex})
    for r in range(n):
        cy = origin + r + 0.5
        for c in range(n):
            if not active[r, c]:
                continue
            cx = origin + c + 0.5
            ET.SubElement(
                walls_g,
                f"{{{SVG_NS}}}circle",
                {"cx": f"{cx:.3f}", "cy": f"{cy:.3f}", "r": f"{wr:.3f}"},
            )
            if c + 1 < n and active[r, c + 1]:
                ET.SubElement(
                    walls_g,
                    f"{{{SVG_NS}}}rect",
                    {"x": f"{cx:.3f}", "y": f"{cy - wr:.3f}", "width": "1", "height": f"{2 * wr:.3f}"},
                )
            if r + 1 < n and active[r + 1, c]:
                ET.SubElement(
                    walls_g,
                    f"{{{SVG_NS}}}rect",
                    {"x": f"{cx - wr:.3f}", "y": f"{cy:.3f}", "width": f"{2 * wr:.3f}", "height": "1"},
                )
            if r + 1 < n and c + 1 < n and active[r + 1, c] and active[r, c + 1] and active[r + 1, c + 1]:
                ET.SubElement(
                    walls_g,
                    f"{{{SVG_NS}}}rect",
                    {"x": f"{cx:.3f}", "y": f"{cy:.3f}", "width": "1", "height": "1"},
                )

    if design.use_run_dots:
        dots_g = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "corridor-dots", "fill": accent_hex})
        for r in range(n):
            cy = origin + r + 0.5
            for c in range(n):
                if not active[r, c]:
                    continue
                horiz = (c > 0 and active[r, c - 1]) and (c + 1 < n and active[r, c + 1])
                vert = (r > 0 and active[r - 1, c]) and (r + 1 < n and active[r + 1, c])
                if horiz or vert:
                    cx = origin + c + 0.5
                    ET.SubElement(
                        dots_g,
                        f"{{{SVG_NS}}}circle",
                        {"cx": f"{cx:.3f}", "cy": f"{cy:.3f}", "r": "0.160"},
                    )

    if design.chamber_radius_frac > 0.0:
        cx_c = origin + n / 2.0
        cy_c = origin + n / 2.0
        r_ch = design.chamber_radius_frac * n
        start_deg = -90.0 + design.chamber_door_half_deg
        end_deg = 270.0 - design.chamber_door_half_deg
        a0, a1 = math.radians(start_deg), math.radians(end_deg)
        x0, y0 = cx_c + r_ch * math.cos(a0), cy_c + r_ch * math.sin(a0)
        x1, y1 = cx_c + r_ch * math.cos(a1), cy_c + r_ch * math.sin(a1)
        chamber_g = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "chamber"})
        ET.SubElement(
            chamber_g,
            f"{{{SVG_NS}}}path",
            {
                "d": f"M {x0:.4f} {y0:.4f} A {r_ch:.4f} {r_ch:.4f} 0 1 1 {x1:.4f} {y1:.4f}",
                "fill": "none",
                "stroke": wall_hex,
                "stroke-width": f"{2 * wr:.4f}",
                "stroke-linecap": "round",
            },
        )
        for hx, hy in ((x0, y0), (x1, y1)):
            ET.SubElement(
                chamber_g,
                f"{{{SVG_NS}}}circle",
                {"cx": f"{hx:.4f}", "cy": f"{hy:.4f}", "r": "0.650", "fill": wall_hex},
            )
            ET.SubElement(
                chamber_g,
                f"{{{SVG_NS}}}circle",
                {"cx": f"{hx:.4f}", "cy": f"{hy:.4f}", "r": "0.340", "fill": accent_hex},
            )
        if design.use_run_dots:
            for da in np.linspace(start_deg + 14.0, end_deg - 14.0, 16):
                rad = math.radians(float(da))
                dx = cx_c + r_ch * math.cos(rad)
                dy = cy_c + r_ch * math.sin(rad)
                ET.SubElement(
                    chamber_g,
                    f"{{{SVG_NS}}}circle",
                    {"cx": f"{dx:.4f}", "cy": f"{dy:.4f}", "r": "0.160", "fill": accent_hex},
                )

    logos_g = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "logos"})
    for idx, placement in enumerate(design.placements):
        if placement.logo_path.suffix.lower() == ".svg":
            _append_svg_logo_group(logos_g, placement, idx, origin, n, bg_hex)
        else:
            _append_raster_logo_group(
                logos_g, placement, idx, origin, n, design.palette.background_rgb
            )

    ET.indent(root, space="  ")
    svg_xml = ET.tostring(root, encoding="unicode") + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg_xml, encoding="utf-8")
    return out_path
