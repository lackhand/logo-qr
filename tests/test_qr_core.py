"""Core unit tests for ISO/IEC 18004 QR primitives, Reed-Solomon, and QArt F2 solving."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from logo_qr.qart import encode_standard_qr, select_version, solve_qart
from logo_qr.qr_spec import (
    GF_EXP,
    build_layout,
    get_version_spec,
    gf_mul,
    rs_encode_block,
)
from tests.qr_verifiers import decode_qr_image

DEMO_URL = "https://example.com/qr/demo?item=42"


def _save_matrix_png(matrix: np.ndarray, path: Path, scale: int = 10, quiet: int = 4) -> None:
    n = matrix.shape[0]
    canvas = np.ones((n + 2 * quiet, n + 2 * quiet), dtype=np.uint8) * 255
    canvas[quiet : quiet + n, quiet : quiet + n] = np.where(matrix == 1, 0, 255)
    img = Image.fromarray(canvas, mode="L").resize(
        ((n + 2 * quiet) * scale, (n + 2 * quiet) * scale),
        resample=Image.Resampling.NEAREST,
    )
    img.save(path)


def test_gf256_and_reed_solomon_vector():
    assert gf_mul(0, 123) == 0
    assert gf_mul(1, 123) == 123
    assert gf_mul(0x8E, 0x02) == (0x1C ^ 0x1D)

    data = list(range(1, 35))
    ec = rs_encode_block(data, 10)
    assert len(ec) == 10

    full = data + ec
    for j in range(10):
        root = GF_EXP[j]
        val = 0
        for cw in full:
            val = gf_mul(val, root) ^ cw
        assert val == 0, f"RS syndrome non-zero at alpha^{j}"


@pytest.mark.parametrize("version", range(1, 15))
@pytest.mark.parametrize("ec_level", ["L", "M", "Q", "H"])
def test_all_table9_version_specs_and_decode(version: int, ec_level: str):
    spec = get_version_spec(version, ec_level)
    layout = build_layout(version, ec_level)
    expected_cw = sum(d + e for d, e in spec.blocks)
    assert len(layout.codeword_coords) == expected_cw

    msg = "HI-QR" if spec.total_data_codewords < 38 else DEMO_URL
    mat = encode_standard_qr(msg, version=version, ec_level=ec_level, mask=1)
    assert mat.shape == (spec.size, spec.size)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"v{version}_{ec_level}.png"
        _save_matrix_png(mat, p, scale=10, quiet=4)
        decoded = decode_qr_image(p)
        for engine in ("opencv_aruco", "opencv"):
            assert msg in decoded[engine], f"V{version}-{ec_level} failed on {engine}: {decoded}"


def test_auto_mask_and_select_version():
    v = select_version(DEMO_URL, ec_level="H", min_version=5, min_free_bytes=8)
    assert v >= 5
    mat = encode_standard_qr(DEMO_URL, version=v, ec_level="H", mask=None)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "auto_mask.png"
        _save_matrix_png(mat, p, scale=10, quiet=4)
        decoded = decode_qr_image(p)
        assert DEMO_URL in decoded["opencv_aruco"]
        assert DEMO_URL in decoded["opencv"]

        for mask_id in range(8):
            m_mat = encode_standard_qr(DEMO_URL, version=5, ec_level="H", mask=mask_id)
            mp = Path(tmp) / f"mask_{mask_id}.png"
            _save_matrix_png(m_mat, mp, scale=10, quiet=4)
            m_dec = decode_qr_image(mp)
            assert DEMO_URL in m_dec["opencv_aruco"]
            assert DEMO_URL in m_dec["opencv"]

    with pytest.raises(ValueError, match="exceeds"):
        select_version("x" * 1000, ec_level="H")


def test_qart_post_terminator_and_url_fragment():
    version = 9
    spec = get_version_spec(version, "L")
    n = spec.size

    target_zero = np.zeros((n, n), dtype=np.uint8)
    priority = np.zeros((n, n), dtype=np.float64)
    priority[14:39, 14:32] = 10.0

    res0 = solve_qart(DEMO_URL, version, "L", target_zero, priority, mode="post_terminator")
    assert res0.decoded_payload == DEMO_URL
    assert res0.satisfied_weight >= 0.90 * res0.total_weight

    res_frag = solve_qart(DEMO_URL, version, "L", target_zero, priority, mode="url_fragment")
    assert res_frag.decoded_payload.startswith(DEMO_URL + "#")
    frag = res_frag.decoded_payload.split("#", 1)[1]
    assert len(frag) > 10
    assert all(ch in "@ABCDEFGHIJKLMNO" for ch in frag)

    with tempfile.TemporaryDirectory() as tmp:
        for idx, res in enumerate((res0, res_frag)):
            p = Path(tmp) / f"qart_{idx}.png"
            _save_matrix_png(res.matrix, p, scale=10, quiet=4)
            decoded = decode_qr_image(p)
            for engine in ("opencv", "opencv_aruco"):
                assert res.decoded_payload in decoded[engine], (
                    f"QArt {idx} failed {engine}: {decoded}"
                )
