"""Multi-decoder QR verification and simulated phone-camera / fabric stress harness."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from logo_qr.verify import decode_qr_image, verify_qr_image


def _make_perspective_tilt(img: Image.Image, angle_deg: float = 12.0) -> Image.Image:
    """Apply a realistic ~12-degree off-axis 3D projective warp on a light background."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    pad = int(max(w, h) * 0.22)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (242, 242, 242))
    canvas.paste(rgb, (pad, pad))
    cw, ch = canvas.size
    arr = np.array(canvas)

    theta = np.deg2rad(angle_deg)
    phi = np.deg2rad(angle_deg * 0.65)
    f = float(max(cw, ch)) * 2.2
    cx, cy = cw / 2.0, ch / 2.0

    src_pts = np.array(
        [[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]],
        dtype=np.float32,
    )

    dst_pts = []
    for x, y in src_pts:
        dx = x - cx
        dy = y - cy
        x1 = dx * np.cos(theta)
        z1 = -dx * np.sin(theta)
        y2 = dy * np.cos(phi) - z1 * np.sin(phi)
        z2 = dy * np.sin(phi) + z1 * np.cos(phi)
        scale = f / (f + z2)
        dst_pts.append([cx + x1 * scale, cy + y2 * scale])
    dst_pts_arr = np.array(dst_pts, dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts_arr)
    warped = cv2.warpPerspective(
        arr,
        H,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(242, 242, 242),
    )
    return Image.fromarray(warped)


def _make_fabric_drape(img: Image.Image) -> Image.Image:
    """Simulate realistic curved fabric drape (~0.6% amplitude, 0.75 cycles)."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    pad = int(max(w, h) * 0.14)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (242, 242, 242))
    canvas.paste(rgb, (pad, pad))
    cw, ch = canvas.size
    arr = np.array(canvas)

    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    amp = 0.006 * float(cw)
    phase_x = 2.0 * np.pi * 0.75 * (xx / float(cw))
    phase_y = 2.0 * np.pi * 0.75 * (yy / float(ch))

    map_x = xx + amp * np.sin(phase_y + 0.4 * np.cos(phase_x))
    map_y = yy + 0.6 * amp * np.sin(phase_x)

    warped = cv2.remap(
        arr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(242, 242, 242),
    )
    return Image.fromarray(warped)


def generate_camera_transforms(image_path: str | Path, work_dir: Path) -> dict[str, Path]:
    """Create simulated phone-camera and curved fabric transforms inside work_dir."""
    src_path = Path(image_path)
    img = Image.open(src_path).convert("RGB")
    transforms: dict[str, Path] = {}

    clean_path = work_dir / "clean.png"
    img.save(clean_path, format="PNG")
    transforms["clean"] = clean_path

    down_path = work_dir / "downscaled_480.png"
    img.resize((480, 480), resample=Image.Resampling.BILINEAR).save(down_path, format="PNG")
    transforms["downscaled_480"] = down_path

    blur_base = img.resize((900, 900), resample=Image.Resampling.BILINEAR)
    blurred = blur_base.filter(ImageFilter.GaussianBlur(radius=1.2))
    blur_path = work_dir / "blurred.jpg"
    blurred.save(blur_path, format="JPEG", quality=75)
    transforms["blurred"] = blur_path

    persp = _make_perspective_tilt(img, angle_deg=12.0)
    persp_path = work_dir / "perspective_tilt.png"
    persp.save(persp_path, format="PNG")
    transforms["perspective_tilt"] = persp_path

    drape = _make_fabric_drape(img)
    drape_path = work_dir / "fabric_drape.png"
    drape.save(drape_path, format="PNG")
    transforms["fabric_drape"] = drape_path

    return transforms


def _matches(decoded: list[str], expected: str, exact: bool) -> bool:
    if exact:
        return expected in decoded
    return any(s.startswith(expected) for s in decoded)


def verify_camera_robustness(
    image_path: str | Path,
    expected_url: str,
    exact: bool = True,
    required_engines: tuple[str, ...] = ("opencv_aruco",),
) -> dict[str, dict[str, list[str]]]:
    """Verify required Python barcode engines decode the expected URL across all 5 transforms."""
    results: dict[str, dict[str, list[str]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        transforms = generate_camera_transforms(image_path, Path(tmp))
        for tname, tpath in transforms.items():
            decoded = decode_qr_image(tpath)
            results[tname] = decoded
            for engine in required_engines:
                if not _matches(decoded[engine], expected_url, exact=exact):
                    raise AssertionError(
                        f"Transform '{tname}' failed on {engine} for {image_path}: "
                        f"expected {'exact' if exact else 'prefix'} {expected_url!r}, "
                        f"got {decoded}"
                    )
    return results


__all__ = [
    "decode_qr_image",
    "generate_camera_transforms",
    "verify_camera_robustness",
    "verify_qr_image",
]
