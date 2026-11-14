"""Crisp square-module QR renderer with layered SVG/raster composition (PNG and pure-vector SVG)."""

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
from PIL import Image, ImageOps

from logo_qr.qart import QArtResult, _score_mask_penalty, select_version, solve_qart
from logo_qr.qr_spec import QRMatrixLayout, build_layout
from logo_qr.style import LogoLayer, LogoQRDesign

MAGICK_BIN = Path("/opt/homebrew/bin/magick")
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

_SKIP_RIM_TAGS = frozenset(
    {
        "defs",
        "style",
        "title",
        "desc",
        "metadata",
        "linearGradient",
        "radialGradient",
        "pattern",
        "clipPath",
        "mask",
        "filter",
    }
)
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
    """Identify protected QR structure modules (finders/separators/timing/format) and interior alignments."""
    n = layout.size
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
    return finder_plus_sep, protected, valid_centers


def _dilate_alpha_rim(
    sprite_rgba: Image.Image, rim_px: float, rim_rgb: tuple[int, int, int]
) -> Image.Image:
    """Create a canvas padded with an opaque Euclidean die-cut rim beneath a sprite."""
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


def _warp_rgba_premultiplied(
    raw: Image.Image, warp_2x3: np.ndarray, grid_ss: int
) -> np.ndarray:
    """Warp an RGBA image in premultiplied alpha space to avoid dark edge fringing."""
    arr = np.asarray(raw, dtype=np.float32) / 255.0
    pre = np.empty_like(arr)
    pre[:, :, :3] = arr[:, :, :3] * arr[:, :, 3:4]
    pre[:, :, 3] = arr[:, :, 3]

    wp = cv2.warpAffine(
        pre,
        warp_2x3,
        (grid_ss, grid_ss),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0, 0.0),
    )
    alpha = np.clip(wp[:, :, 3], 0.0, 1.0)
    out = np.zeros((grid_ss, grid_ss, 4), dtype=np.uint8)
    nz = alpha > (1.0 / 512.0)
    rgb = np.zeros((grid_ss, grid_ss, 3), dtype=np.float32)
    rgb[nz] = np.clip(wp[:, :, :3][nz] / alpha[nz, None], 0.0, 1.0)
    out[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    out[:, :, 3] = np.round(alpha * 255.0).astype(np.uint8)
    return out


def _composite_emblem_canvas(
    design: LogoQRDesign,
    grid_ss: int,
    m_ss: int,
    n: int,
    finder_plus_sep: np.ndarray,
) -> tuple[Image.Image, np.ndarray]:
    """Render ordered logo layers and die-cut rims onto an RGBA canvas and binary silhouette."""
    emblem = Image.new("RGBA", (grid_ss, grid_ss), (0, 0, 0, 0))
    silhouette = np.zeros((grid_ss, grid_ss), dtype=bool)
    light_rgb = design.palette.light_rgb
    box_mod = design.base_scale_frac * n
    c_dst = (grid_ss - 1) / 2.0
    finder_px = np.repeat(np.repeat(finder_plus_sep, m_ss, axis=0), m_ss, axis=1)

    for idx, layer in enumerate(design.layers):
        src_px = max(
            32, min(2048, int(round(box_mod * m_ss * max(1.0, layer.transform.max_scale))))
        )
        raw = load_logo_rgba(layer.logo_path, src_px)
        c_src = (src_px - 1) / 2.0
        f = (m_ss * box_mod) / src_px
        m = layer.transform
        warp_2x3 = np.array(
            [
                [f * m.a, f * m.c, c_dst + m_ss * m.tx - f * (m.a * c_src + m.c * c_src)],
                [f * m.b, f * m.d, c_dst + m_ss * m.ty - f * (m.b * c_src + m.d * c_src)],
            ],
            dtype=np.float64,
        )
        warped = _warp_rgba_premultiplied(raw, warp_2x3, grid_ss)
        fg_mask = warped[:, :, 3] >= 24
        if not fg_mask.any():
            raise ValueError(f"Layer {idx} transform places logo completely outside the QR matrix")

        warped_img = Image.fromarray(warped, mode="RGBA")
        if layer.die_cut_mod > 0.0:
            dist = cv2.distanceTransform(
                (~fg_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            rim_alpha = np.clip(layer.die_cut_mod * m_ss + 0.5 - dist, 0.0, 1.0)
            rim_rgba = np.zeros_like(warped)
            rim_rgba[:, :, 0] = light_rgb[0]
            rim_rgba[:, :, 1] = light_rgb[1]
            rim_rgba[:, :, 2] = light_rgb[2]
            rim_rgba[:, :, 3] = np.round(rim_alpha * 255.0).astype(np.uint8)
            layer_img = Image.fromarray(rim_rgba, mode="RGBA")
            layer_img.alpha_composite(warped_img)
            layer_sil = np.asarray(layer_img)[:, :, 3] >= 24
        else:
            layer_img = warped_img
            layer_sil = fg_mask

        if (layer_sil & finder_px).any():
            raise ValueError(
                f"Layer {idx} overlaps protected QR finder pattern; reduce scale or translation"
            )

        emblem.alpha_composite(layer_img)
        silhouette |= layer_sil

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
    halo_px = design.halo_mod * m_ss

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


def _resolve_version(design: LogoQRDesign, url: str) -> int:
    if design.version is not None:
        return design.version
    return select_version(url, ec_level=design.ec_level, min_version=8, min_free_bytes=8)


def render_logo_qr(design: LogoQRDesign, url: str) -> RenderedLogoQR:
    """Solve QArt matrix around suppressed central silhouette and render crisp square modules."""
    version = _resolve_version(design, url)
    layout = build_layout(version, design.ec_level)
    n = layout.size
    m_ss = 2 * design.module_px
    grid_ss = n * m_ss
    pad_px_1x = design.quiet_mod * design.module_px
    grid_origin_ss = 2 * pad_px_1x
    total_ss = 2 * grid_origin_ss + grid_ss

    dark_rgb = design.palette.dark_rgb
    light_rgb = design.palette.light_rgb

    finder_plus_sep, protected, valid_centers = _build_protected_and_alignment_masks(layout)
    emblem, silhouette = _composite_emblem_canvas(design, grid_ss, m_ss, n, finder_plus_sep)
    suppressed = _compute_suppressed_modules(
        layout, design, silhouette, protected, valid_centers, m_ss
    )

    # Steer free padding bits to match emblem luminance at suppressed module centers
    emblem_on_light = Image.new("RGBA", (grid_ss, grid_ss), (*light_rgb, 255))
    emblem_on_light.alpha_composite(emblem)
    emblem_arr = np.asarray(emblem_on_light.convert("RGB"), dtype=np.float64) / 255.0
    centers = np.arange(n) * m_ss + (m_ss // 2)
    sample_rgb = emblem_arr[np.ix_(centers, centers)]
    sample_luma = (
        0.299 * sample_rgb[:, :, 0] + 0.587 * sample_rgb[:, :, 1] + 0.114 * sample_rgb[:, :, 2]
    )

    target = (sample_luma < 0.5).astype(np.uint8)
    priority = np.where(suppressed & (~layout.is_function), 10.0, 0.0)
    best_qart: QArtResult | None = None
    best_score = float("-inf")
    for mask_id in range(8):
        cand = solve_qart(
            url=url,
            version=version,
            ec_level=design.ec_level,
            target=target,
            priority=priority,
            masks=[mask_id],
        )
        score = cand.satisfied_weight - 0.05 * _score_mask_penalty(cand.matrix)
        if score > best_score:
            best_score = score
            best_qart = cand
    assert best_qart is not None
    qart = best_qart
    mat = qart.matrix

    canvas_arr = np.full((total_ss, total_ss, 3), light_rgb, dtype=np.uint8)
    active = (mat == 1) & (~suppressed)
    for r in range(n):
        y0 = grid_origin_ss + r * m_ss
        for c in range(n):
            if active[r, c]:
                x0 = grid_origin_ss + c * m_ss
                canvas_arr[y0 : y0 + m_ss, x0 : x0 + m_ss] = dark_rgb

    rgba_full = Image.fromarray(canvas_arr, mode="RGB").convert("RGBA")
    rgba_full.alpha_composite(emblem, (grid_origin_ss, grid_origin_ss))

    out_1x = rgba_full.convert("RGB").resize(
        (total_ss // 2, total_ss // 2), resample=Image.Resampling.BOX
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

    grid_rgb = np.asarray(rendered.image, dtype=np.int16)[off : off + n * m, off : off + n * m]
    supp_px = np.repeat(np.repeat(suppressed, m, axis=0), m, axis=1)
    light_arr = np.array(design.palette.light_rgb, dtype=np.int16)
    diff_bg = np.max(np.abs(grid_rgb - light_arr), axis=2) >= 20
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
        r"(url\(\s*#|#|\bhref\s*=\s*['\"]#)("
        + "|".join(re.escape(k) for k in sorted(id_map, key=len, reverse=True))
        + r")(\b)"
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


def _clone_svg_rim_element(elem: ET.Element, light_hex: str, stroke_w: str) -> ET.Element | None:
    """Clone drawable SVG geometry elements with an opaque light-colored die-cut stroke rim."""
    tag = _local_tag(elem.tag)
    if tag in _SKIP_RIM_TAGS:
        return None

    out = ET.Element(elem.tag, dict(elem.attrib))
    for drop_attr in ("id", "opacity", "fill-opacity", "stroke-opacity", "clip-path", "mask", "filter"):
        out.attrib.pop(drop_attr, None)

    if tag in _GEOM_TAGS:
        out.attrib["fill"] = light_hex
        out.attrib["stroke"] = light_hex
        out.attrib["stroke-width"] = stroke_w
        out.attrib["stroke-linejoin"] = "round"
        out.attrib["stroke-linecap"] = "round"
        if "style" in out.attrib:
            out.attrib.pop("style")

    for child in elem:
        cloned_child = _clone_svg_rim_element(child, light_hex, stroke_w)
        if cloned_child is not None:
            out.append(cloned_child)
    return out


def _append_svg_layer_group(
    parent: ET.Element,
    layer: LogoLayer,
    idx: int,
    origin: float,
    n: int,
    base_scale_frac: float,
    light_hex: str,
) -> None:
    cx_qr = origin + n / 2.0
    cy_qr = origin + n / 2.0
    box_mod = base_scale_frac * n

    tree = ET.parse(layer.logo_path)
    src_root = tree.getroot()
    min_x, min_y, vb_w, vb_h = _parse_svg_viewbox(src_root)
    vb_scale = box_mod / max(vb_w, vb_h)
    vb_cx = min_x + vb_w / 2.0
    vb_cy = min_y + vb_h / 2.0
    m = layer.transform

    group = ET.SubElement(
        parent,
        f"{{{SVG_NS}}}g",
        {
            "id": f"layer-{idx}",
            "transform": (
                f"translate({cx_qr:.4f} {cy_qr:.4f}) "
                f"matrix({m.a:.6f} {m.b:.6f} {m.c:.6f} {m.d:.6f} {m.tx:.4f} {m.ty:.4f}) "
                f"scale({vb_scale:.6f}) "
                f"translate({-vb_cx:.4f} {-vb_cy:.4f})"
            ),
        },
    )

    if layer.die_cut_mod > 0.0:
        sw = f"{(2.0 * layer.die_cut_mod) / (vb_scale * math.sqrt(m.det)):.4f}"
        rim_group = ET.SubElement(group, f"{{{SVG_NS}}}g", {"id": f"layer-{idx}-rim"})
        for child in src_root:
            rim_child = _clone_svg_rim_element(child, light_hex, sw)
            if rim_child is not None:
                rim_group.append(rim_child)

    fg_group = ET.SubElement(group, f"{{{SVG_NS}}}g", {"id": f"layer-{idx}-art"})
    for child in src_root:
        fg_group.append(copy.deepcopy(child))
    _prefix_svg_ids(fg_group, f"layer{idx}-")


def _append_raster_layer_group(
    parent: ET.Element,
    layer: LogoLayer,
    idx: int,
    origin: float,
    n: int,
    base_scale_frac: float,
    light_rgb: tuple[int, int, int],
) -> None:
    cx_qr = origin + n / 2.0
    cy_qr = origin + n / 2.0
    box_mod = base_scale_frac * n
    m = layer.transform

    with Image.open(layer.logo_path) as im:
        src = ImageOps.contain(im.convert("RGBA"), (512, 512), method=Image.Resampling.LANCZOS)

    px_per_mod = 512.0 / box_mod
    if layer.die_cut_mod > 0.0:
        rim_px = (layer.die_cut_mod * px_per_mod) / math.sqrt(m.det)
        embed_img = _dilate_alpha_rim(src, rim_px=rim_px, rim_rgb=light_rgb)
    else:
        embed_img = src

    scale_w = embed_img.width / px_per_mod
    scale_h = embed_img.height / px_per_mod

    buf = BytesIO()
    embed_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    group = ET.SubElement(
        parent,
        f"{{{SVG_NS}}}g",
        {
            "id": f"layer-{idx}",
            "transform": (
                f"translate({cx_qr:.4f} {cy_qr:.4f}) "
                f"matrix({m.a:.6f} {m.b:.6f} {m.c:.6f} {m.d:.6f} {m.tx:.4f} {m.ty:.4f})"
            ),
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

    origin = design.quiet_mod
    total_mod = n + 2 * origin
    px_dim = total_mod * design.module_px
    active = (mat == 1) & (~suppressed)

    dark_hex = design.palette.dark_color
    light_hex = design.palette.light_color

    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": str(px_dim),
            "height": str(px_dim),
            "viewBox": f"0 0 {total_mod} {total_mod}",
        },
    )

    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {"width": str(total_mod), "height": str(total_mod), "fill": light_hex},
    )

    cmds: list[str] = []
    for r in range(n):
        c = 0
        while c < n:
            if active[r, c]:
                c0 = c
                while c < n and active[r, c]:
                    c += 1
                cmds.append(f"M{origin + c0} {origin + r}h{c - c0}v1h-{c - c0}z")
            else:
                c += 1

    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {"id": "qr-modules", "fill": dark_hex, "d": "".join(cmds)},
    )

    layers_g = ET.SubElement(root, f"{{{SVG_NS}}}g", {"id": "layers"})
    for idx, layer in enumerate(design.layers):
        if layer.logo_path.suffix.lower() == ".svg":
            _append_svg_layer_group(
                layers_g,
                layer,
                idx,
                origin,
                n,
                design.base_scale_frac,
                light_hex,
            )
        else:
            _append_raster_layer_group(
                layers_g,
                layer,
                idx,
                origin,
                n,
                design.base_scale_frac,
                design.palette.light_rgb,
            )

    ET.indent(root, space="  ")
    svg_xml = ET.tostring(root, encoding="unicode") + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg_xml, encoding="utf-8")
    return out_path
