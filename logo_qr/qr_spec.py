"""ISO/IEC 18004 QR Code specification primitives (Versions 1..14, Levels L/M/Q/H)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# GF(256) with primitive polynomial 0x11D (x^8 + x^4 + x^3 + x^2 + 1)
GF_EXP: list[int] = [0] * 512
GF_LOG: list[int] = [0] * 256

_x = 1
for _i in range(255):
    GF_EXP[_i] = _x
    GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    GF_EXP[_i] = GF_EXP[_i - 255]


def gf_mul(a: int, b: int) -> int:
    """Multiply two elements in GF(256)."""
    if a == 0 or b == 0:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def rs_generator_poly(ec_count: int) -> list[int]:
    """Build Reed-Solomon generator polynomial of degree ec_count over GF(256)."""
    poly = [1]
    for i in range(ec_count):
        nxt = [0] * (len(poly) + 1)
        root = GF_EXP[i]
        for j, coef in enumerate(poly):
            nxt[j] ^= coef
            nxt[j + 1] ^= gf_mul(coef, root)
        poly = nxt
    return poly


_RS_GEN_CACHE: dict[int, list[int]] = {}


def rs_encode_block(data: list[int] | bytes, ec_count: int) -> list[int]:
    """Compute ec_count Reed-Solomon parity bytes for a single data block."""
    if ec_count not in _RS_GEN_CACHE:
        _RS_GEN_CACHE[ec_count] = rs_generator_poly(ec_count)
    gen = _RS_GEN_CACHE[ec_count]
    msg = list(data) + [0] * ec_count
    data_len = len(data)
    for i in range(data_len):
        coef = msg[i]
        if coef != 0:
            for j in range(1, ec_count + 1):
                msg[i + j] ^= gf_mul(gen[j], coef)
    return msg[data_len:]


ALIGNMENT_CENTERS: dict[int, list[int]] = {
    1: [],
    2: [6, 18],
    3: [6, 22],
    4: [6, 26],
    5: [6, 30],
    6: [6, 34],
    7: [6, 22, 38],
    8: [6, 24, 42],
    9: [6, 26, 46],
    10: [6, 28, 50],
    11: [6, 30, 54],
    12: [6, 32, 58],
    13: [6, 34, 62],
    14: [6, 26, 46, 66],
}

# ISO/IEC 18004 Table 9: (version, ec_level) -> list of (data_codewords, ec_codewords) per block
_BLOCK_SPECS: dict[tuple[int, str], list[tuple[int, int]]] = {
    (1, "L"): [(19, 7)],
    (1, "M"): [(16, 10)],
    (1, "Q"): [(13, 13)],
    (1, "H"): [(9, 17)],
    (2, "L"): [(34, 10)],
    (2, "M"): [(28, 16)],
    (2, "Q"): [(22, 22)],
    (2, "H"): [(16, 28)],
    (3, "L"): [(55, 15)],
    (3, "M"): [(44, 26)],
    (3, "Q"): [(17, 18)] * 2,
    (3, "H"): [(13, 22)] * 2,
    (4, "L"): [(80, 20)],
    (4, "M"): [(32, 18)] * 2,
    (4, "Q"): [(24, 26)] * 2,
    (4, "H"): [(9, 16)] * 4,
    (5, "L"): [(108, 26)],
    (5, "M"): [(43, 24)] * 2,
    (5, "Q"): [(15, 18)] * 2 + [(16, 18)] * 2,
    (5, "H"): [(11, 22)] * 2 + [(12, 22)] * 2,
    (6, "L"): [(68, 18)] * 2,
    (6, "M"): [(27, 16)] * 4,
    (6, "Q"): [(19, 24)] * 4,
    (6, "H"): [(15, 28)] * 4,
    (7, "L"): [(78, 20)] * 2,
    (7, "M"): [(31, 18)] * 4,
    (7, "Q"): [(14, 18)] * 2 + [(15, 18)] * 4,
    (7, "H"): [(13, 26)] * 4 + [(14, 26)] * 1,
    (8, "L"): [(97, 24)] * 2,
    (8, "M"): [(38, 22)] * 2 + [(39, 22)] * 2,
    (8, "Q"): [(18, 22)] * 4 + [(19, 22)] * 2,
    (8, "H"): [(14, 26)] * 4 + [(15, 26)] * 2,
    (9, "L"): [(116, 30)] * 2,
    (9, "M"): [(36, 22)] * 3 + [(37, 22)] * 2,
    (9, "Q"): [(16, 20)] * 4 + [(17, 20)] * 4,
    (9, "H"): [(12, 24)] * 4 + [(13, 24)] * 4,
    (10, "L"): [(68, 18)] * 2 + [(69, 18)] * 2,
    (10, "M"): [(43, 26)] * 4 + [(44, 26)] * 1,
    (10, "Q"): [(19, 24)] * 6 + [(20, 24)] * 2,
    (10, "H"): [(15, 28)] * 6 + [(16, 28)] * 2,
    (11, "L"): [(81, 20)] * 4,
    (11, "M"): [(50, 30)] * 1 + [(51, 30)] * 4,
    (11, "Q"): [(22, 28)] * 4 + [(23, 28)] * 4,
    (11, "H"): [(12, 24)] * 3 + [(13, 24)] * 8,
    (12, "L"): [(92, 24)] * 2 + [(93, 24)] * 2,
    (12, "M"): [(36, 22)] * 6 + [(37, 22)] * 2,
    (12, "Q"): [(20, 26)] * 4 + [(21, 26)] * 6,
    (12, "H"): [(14, 28)] * 7 + [(15, 28)] * 4,
    (13, "L"): [(107, 26)] * 4,
    (13, "M"): [(37, 22)] * 8 + [(38, 22)] * 1,
    (13, "Q"): [(20, 24)] * 8 + [(21, 24)] * 4,
    (13, "H"): [(11, 22)] * 12 + [(12, 22)] * 4,
    (14, "L"): [(115, 30)] * 3 + [(116, 30)] * 1,
    (14, "M"): [(40, 24)] * 4 + [(41, 24)] * 5,
    (14, "Q"): [(16, 20)] * 11 + [(17, 20)] * 5,
    (14, "H"): [(12, 24)] * 11 + [(13, 24)] * 5,
}

EC_FORMAT_BITS: dict[str, int] = {
    "L": 0b01,
    "M": 0b00,
    "Q": 0b11,
    "H": 0b10,
}

MASK_FUNCTIONS: dict[int, Callable[[int, int], int]] = {
    0: lambda r, c: 1 if (r + c) % 2 == 0 else 0,
    1: lambda r, c: 1 if r % 2 == 0 else 0,
    2: lambda r, c: 1 if c % 3 == 0 else 0,
    3: lambda r, c: 1 if (r + c) % 3 == 0 else 0,
    4: lambda r, c: 1 if ((r // 2) + (c // 3)) % 2 == 0 else 0,
    5: lambda r, c: 1 if ((r * c) % 2) + ((r * c) % 3) == 0 else 0,
    6: lambda r, c: 1 if (((r * c) % 2) + ((r * c) % 3)) % 2 == 0 else 0,
    7: lambda r, c: 1 if (((r + c) % 2) + ((r * c) % 3)) % 2 == 0 else 0,
}


@dataclass(frozen=True)
class VersionSpec:
    version: int
    ec_level: str
    size: int
    blocks: list[tuple[int, int]]
    alignment_centers: list[int]
    total_data_codewords: int
    total_ec_codewords: int
    total_codewords: int


@dataclass
class QRMatrixLayout:
    version: int
    ec_level: str
    size: int
    spec: VersionSpec
    is_function: np.ndarray
    fixed_values: np.ndarray
    codeword_coords: list[list[tuple[int, int]]]
    remainder_coords: list[tuple[int, int]]
    tx_to_block: list[tuple[int, bool, int]]
    block_to_tx: dict[tuple[int, bool, int], int]


def get_version_spec(version: int, ec_level: str) -> VersionSpec:
    """Return validated VersionSpec for supported versions 1..14 and levels L/M/Q/H."""
    key = (version, ec_level.upper())
    if key not in _BLOCK_SPECS:
        raise ValueError(f"Unsupported QR version/EC level: {key}")
    blocks = _BLOCK_SPECS[key]
    size = 4 * version + 17
    total_data = sum(d for d, _ in blocks)
    total_ec = sum(e for _, e in blocks)
    return VersionSpec(
        version=version,
        ec_level=ec_level.upper(),
        size=size,
        blocks=blocks,
        alignment_centers=ALIGNMENT_CENTERS[version],
        total_data_codewords=total_data,
        total_ec_codewords=total_ec,
        total_codewords=total_data + total_ec,
    )


def compute_format_bits(ec_level: str, mask_id: int) -> int:
    """Compute 15-bit BCH(15,5) + 0x5412 masked format information word."""
    if mask_id not in MASK_FUNCTIONS:
        raise ValueError(f"Invalid QR mask id: {mask_id}")
    ec_bits = EC_FORMAT_BITS[ec_level.upper()]
    data = (ec_bits << 3) | mask_id
    rem = data << 10
    gen = 0x537
    for shift in range(4, -1, -1):
        if rem & (1 << (shift + 10)):
            rem ^= gen << shift
    return ((data << 10) | rem) ^ 0x5412


def compute_version_bits(version: int) -> int:
    """Compute 18-bit Golay(18,6) version information word for V >= 7."""
    if version < 7:
        raise ValueError("Version info bits only exist for Version >= 7")
    rem = version << 12
    gen = 0x1F25
    for shift in range(5, -1, -1):
        if rem & (1 << (shift + 12)):
            rem ^= gen << shift
    return (version << 12) | rem


def _place_finder(is_fn: np.ndarray, fixed: np.ndarray, r0: int, c0: int) -> None:
    n = is_fn.shape[0]
    for dr in range(-1, 8):
        for dc in range(-1, 8):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < n and 0 <= c < n:
                is_fn[r, c] = True
                if 0 <= dr <= 6 and 0 <= dc <= 6:
                    on_border = dr in (0, 6) or dc in (0, 6)
                    in_center = 2 <= dr <= 4 and 2 <= dc <= 4
                    fixed[r, c] = 1 if (on_border or in_center) else 0
                else:
                    fixed[r, c] = 0


def _place_alignment(is_fn: np.ndarray, fixed: np.ndarray, cr: int, cc: int) -> None:
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = cr + dr, cc + dc
            is_fn[r, c] = True
            on_border = abs(dr) == 2 or abs(dc) == 2
            is_center = dr == 0 and dc == 0
            fixed[r, c] = 1 if (on_border or is_center) else 0


def build_layout(version: int, ec_level: str) -> QRMatrixLayout:
    """Construct QR function pattern mask and codeword bit coordinate table."""
    spec = get_version_spec(version, ec_level)
    n = spec.size
    is_fn = np.zeros((n, n), dtype=bool)
    fixed = np.full((n, n), -1, dtype=np.int8)

    _place_finder(is_fn, fixed, 0, 0)
    _place_finder(is_fn, fixed, 0, n - 7)
    _place_finder(is_fn, fixed, n - 7, 0)

    for idx in range(8, n - 8):
        val = 1 if idx % 2 == 0 else 0
        is_fn[6, idx] = True
        fixed[6, idx] = val
        is_fn[idx, 6] = True
        fixed[idx, 6] = val

    centers = spec.alignment_centers
    for cr in centers:
        for cc in centers:
            if (cr <= 8 and cc <= 8) or (cr <= 8 and cc >= n - 9) or (cr >= n - 9 and cc <= 8):
                continue
            _place_alignment(is_fn, fixed, cr, cc)

    dark_r = 4 * version + 9
    is_fn[dark_r, 8] = True
    fixed[dark_r, 8] = 1

    for i in range(9):
        if i != 6:
            is_fn[8, i] = True
            is_fn[i, 8] = True
    for i in range(n - 8, n):
        is_fn[8, i] = True
    for i in range(n - 7, n):
        is_fn[i, 8] = True

    if version >= 7:
        v_bits = compute_version_bits(version)
        for i in range(18):
            bit = (v_bits >> i) & 1
            r_tr, c_tr = i // 3, n - 11 + (i % 3)
            is_fn[r_tr, c_tr] = True
            fixed[r_tr, c_tr] = bit
            r_bl, c_bl = n - 11 + (i % 3), i // 3
            is_fn[r_bl, c_bl] = True
            fixed[r_bl, c_bl] = bit

    bit_coords: list[tuple[int, int]] = []
    col = n - 1
    direction = -1
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(n - 1, -1, -1) if direction == -1 else range(n)
        for r in rows:
            for c in (col, col - 1):
                if not is_fn[r, c]:
                    bit_coords.append((r, c))
        direction *= -1
        col -= 2

    total_cw_bits = spec.total_codewords * 8
    if len(bit_coords) < total_cw_bits:
        raise RuntimeError(
            f"Layout bit capacity {len(bit_coords)} < required {total_cw_bits} for V{version}-{ec_level}"
        )

    codeword_coords = [bit_coords[i * 8 : (i + 1) * 8] for i in range(spec.total_codewords)]
    remainder_coords = bit_coords[total_cw_bits:]

    tx_to_block: list[tuple[int, bool, int]] = []
    block_to_tx: dict[tuple[int, bool, int], int] = {}

    max_d = max(d for d, _ in spec.blocks)
    for i in range(max_d):
        for b_idx, (d_len, _) in enumerate(spec.blocks):
            if i < d_len:
                tx_idx = len(tx_to_block)
                entry = (b_idx, False, i)
                tx_to_block.append(entry)
                block_to_tx[entry] = tx_idx

    max_e = max(e for _, e in spec.blocks)
    for i in range(max_e):
        for b_idx, (_, e_len) in enumerate(spec.blocks):
            if i < e_len:
                tx_idx = len(tx_to_block)
                entry = (b_idx, True, i)
                tx_to_block.append(entry)
                block_to_tx[entry] = tx_idx

    if len(tx_to_block) != spec.total_codewords:
        raise RuntimeError("Interleaving count mismatch")

    return QRMatrixLayout(
        version=version,
        ec_level=spec.ec_level,
        size=n,
        spec=spec,
        is_function=is_fn,
        fixed_values=fixed,
        codeword_coords=codeword_coords,
        remainder_coords=remainder_coords,
        tx_to_block=tx_to_block,
        block_to_tx=block_to_tx,
    )


def apply_format_bits(matrix: np.ndarray, ec_level: str, mask_id: int) -> None:
    """Write 15-bit format information into both reserved strips of matrix in-place."""
    n = matrix.shape[0]
    fmt = compute_format_bits(ec_level, mask_id)

    c1_coords = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
        (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
    ]
    c2_coords = [
        (n - 1, 8), (n - 2, 8), (n - 3, 8), (n - 4, 8), (n - 5, 8), (n - 6, 8), (n - 7, 8),
        (8, n - 8), (8, n - 7), (8, n - 6), (8, n - 5), (8, n - 4), (8, n - 3), (8, n - 2), (8, n - 1),
    ]
    for i in range(15):
        bit = (fmt >> (14 - i)) & 1
        r1, c1 = c1_coords[i]
        r2, c2 = c2_coords[i]
        matrix[r1, c1] = bit
        matrix[r2, c2] = bit
