"""Pure-Python multi-decoder QR verification built on OpenCV."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=1)
def _get_aruco_detector() -> object:
    return cv2.QRCodeDetectorAruco()


@lru_cache(maxsize=1)
def _get_standard_detector() -> object:
    return cv2.QRCodeDetector()


def _decode_cv_detector(detector: object, img_bgr: np.ndarray) -> list[str]:
    text, _, _ = detector.detectAndDecode(img_bgr)
    if text:
        return [text]
    ok, decoded_info, _, _ = detector.detectAndDecodeMulti(img_bgr)
    if ok and decoded_info:
        return [s for s in decoded_info if s]
    return []


def decode_qr_image(image_path: str | Path) -> dict[str, list[str]]:
    """Decode QR payloads using pure-Python OpenCV Aruco and OpenCV standard detectors."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load image with cv2: {path}")
    return {
        "opencv_aruco": _decode_cv_detector(_get_aruco_detector(), img_bgr),
        "opencv": _decode_cv_detector(_get_standard_detector(), img_bgr),
    }


def verify_qr_image(image_path: str | Path, expected_payload: str) -> dict[str, list[str]]:
    """Assert expected_payload decodes cleanly on both opencv_aruco and opencv detectors."""
    decoded = decode_qr_image(image_path)
    if (
        expected_payload not in decoded["opencv_aruco"]
        or expected_payload not in decoded["opencv"]
    ):
        raise RuntimeError(
            f"QR verification failed for {image_path}: "
            f"expected {expected_payload!r}, got {decoded}"
        )
    return decoded
