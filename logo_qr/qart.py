"""QR encoding, automatic version selection, and Russ Cox QArt F2 solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from logo_qr.qr_spec import (
    MASK_FUNCTIONS,
    QRMatrixLayout,
    apply_format_bits,
    build_layout,
    get_version_spec,
    rs_encode_block,
)


@dataclass
class QArtResult:
    version: int
    ec_level: str
    mask: int
    mode: str
    decoded_payload: str
    matrix: np.ndarray
    data_codewords: list[int]
    all_codewords: list[int]
    satisfied_weight: float
    total_weight: float
    matched_mask: np.ndarray


def _char_count_bit_len(version: int) -> int:
    return 8 if version <= 9 else 16


def _encode_byte_header_and_payload(payload_bytes: bytes, version: int) -> list[int]:
    """Build byte-mode prefix bits: [0100][count][payload_bytes]."""
    bits: list[int] = [0, 1, 0, 0]
    c_len = _char_count_bit_len(version)
    n_bytes = len(payload_bytes)
    for sh in range(c_len - 1, -1, -1):
        bits.append((n_bytes >> sh) & 1)
    for b in payload_bytes:
        for sh in range(7, -1, -1):
            bits.append((b >> sh) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> list[int]:
    out: list[int] = []
    for i in range(0, len(bits), 8):
        val = 0
        for b in range(8):
            val = (val << 1) | bits[i + b]
        out.append(val)
    return out


def select_version(
    payload: str,
    ec_level: str = "H",
    min_version: int = 5,
    min_free_bytes: int = 8,
) -> int:
    """Pick the smallest QR version in [min_version..14] that fits payload cleanly."""
    if not payload:
        raise ValueError("payload must be non-empty")
    if not (1 <= min_version <= 14):
        raise ValueError(f"min_version must be in [1, 14], got {min_version}")
    if min_free_bytes < 0:
        raise ValueError(f"min_free_bytes must be >= 0, got {min_free_bytes}")

    ec = ec_level.upper()
    n_bytes = len(payload.encode("utf-8"))
    for headroom in (min_free_bytes, 0):
        for v in range(min_version, 15):
            spec = get_version_spec(v, ec)
            overhead_bytes = 2 if v <= 9 else 3
            if overhead_bytes + n_bytes + headroom <= spec.total_data_codewords:
                return v

    raise ValueError(
        f"Payload ({n_bytes} bytes) exceeds V14-{ec} data capacity"
    )


def _assemble_all_codewords(layout: QRMatrixLayout, data_codewords: list[int]) -> list[int]:
    """Split data_codewords into RS blocks, compute EC bytes, and interleave."""
    spec = layout.spec
    if len(data_codewords) != spec.total_data_codewords:
        raise ValueError(
            f"Expected {spec.total_data_codewords} data codewords, got {len(data_codewords)}"
        )

    blocks_data: list[list[int]] = []
    blocks_ec: list[list[int]] = []
    offset = 0
    for d_len, e_len in spec.blocks:
        blk = data_codewords[offset : offset + d_len]
        offset += d_len
        blocks_data.append(blk)
        blocks_ec.append(rs_encode_block(blk, e_len))

    all_cw = [0] * spec.total_codewords
    for tx_idx, (b_idx, is_ec, local_idx) in enumerate(layout.tx_to_block):
        all_cw[tx_idx] = (
            blocks_ec[b_idx][local_idx] if is_ec else blocks_data[b_idx][local_idx]
        )
    return all_cw


def _render_unmasked_matrix(layout: QRMatrixLayout, all_codewords: list[int]) -> np.ndarray:
    """Place fixed function bits (except format info) and unmasked codeword bits."""
    n = layout.size
    mat = np.zeros((n, n), dtype=np.uint8)
    fixed_mask = layout.fixed_values >= 0
    mat[fixed_mask] = layout.fixed_values[fixed_mask].astype(np.uint8)
    for cw_idx, coords in enumerate(layout.codeword_coords):
        byte_val = all_codewords[cw_idx]
        for bit_pos, (r, c) in enumerate(coords):
            mat[r, c] = (byte_val >> (7 - bit_pos)) & 1
    return mat


def _apply_mask_and_format(
    unmasked: np.ndarray, layout: QRMatrixLayout, mask_id: int
) -> np.ndarray:
    """Apply QR data mask and format bits to produce final binary matrix."""
    n = layout.size
    mat = unmasked.copy()
    mfn = MASK_FUNCTIONS[mask_id]
    for r in range(n):
        for c in range(n):
            if not layout.is_function[r, c]:
                mat[r, c] ^= mfn(r, c)
    apply_format_bits(mat, layout.ec_level, mask_id)
    return mat


def _score_mask_penalty(mat: np.ndarray) -> int:
    """Evaluate ISO/IEC 18004 mask penalty score for a finished binary matrix."""
    n = mat.shape[0]
    penalty = 0

    for line in (mat, mat.T):
        for row in line:
            run = 1
            for idx in range(1, n):
                if row[idx] == row[idx - 1]:
                    run += 1
                else:
                    if run >= 5:
                        penalty += 3 + (run - 5)
                    run = 1
            if run >= 5:
                penalty += 3 + (run - 5)

    same_2x2 = (
        (mat[:-1, :-1] == mat[:-1, 1:])
        & (mat[:-1, :-1] == mat[1:, :-1])
        & (mat[:-1, :-1] == mat[1:, 1:])
    )
    penalty += 3 * int(same_2x2.sum())

    pat_a = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0], dtype=np.uint8)
    pat_b = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
    for line in (mat, mat.T):
        for row in line:
            for j in range(0, n - 10):
                seg = row[j : j + 11]
                if np.array_equal(seg, pat_a) or np.array_equal(seg, pat_b):
                    penalty += 40

    dark_pct = (100.0 * float(mat.sum())) / float(n * n)
    prev5 = int(dark_pct // 5) * 5
    next5 = prev5 + 5
    penalty += 10 * min(abs(prev5 - 50) // 5, abs(next5 - 50) // 5)
    return penalty


def encode_standard_qr(
    text: str, version: int, ec_level: str = "M", mask: int | None = None
) -> np.ndarray:
    """Encode text as a standard ISO/IEC 18004 QR matrix using byte mode."""
    layout = build_layout(version, ec_level)
    payload = text.encode("utf-8")
    bits = _encode_byte_header_and_payload(payload, version)
    max_bits = layout.spec.total_data_codewords * 8
    if len(bits) > max_bits:
        raise ValueError(
            f"Payload ({len(payload)} bytes -> {len(bits)} bits) exceeds V{version}-{ec_level} capacity ({max_bits} bits)"
        )

    term_len = min(4, max_bits - len(bits))
    bits.extend([0] * term_len)
    while len(bits) % 8 != 0:
        bits.append(0)
    pad_bytes = [0xEC, 0x11]
    pad_idx = 0
    while len(bits) < max_bits:
        b = pad_bytes[pad_idx % 2]
        for sh in range(7, -1, -1):
            bits.append((b >> sh) & 1)
        pad_idx += 1

    data_cw = _bits_to_bytes(bits)
    all_cw = _assemble_all_codewords(layout, data_cw)
    unmasked = _render_unmasked_matrix(layout, all_cw)
    if mask is not None:
        return _apply_mask_and_format(unmasked, layout, mask)

    best_mat: np.ndarray | None = None
    best_pen: int | None = None
    for mask_id in range(8):
        cand = _apply_mask_and_format(unmasked, layout, mask_id)
        pen = _score_mask_penalty(cand)
        if best_pen is None or pen < best_pen:
            best_pen = pen
            best_mat = cand
    assert best_mat is not None
    return best_mat


def _prepare_qart_variables(
    url: str,
    layout: QRMatrixLayout,
    mode: str,
    fragment_bytes: int | None,
) -> tuple[list[int], list[int], str]:
    """Build base_bits, free_bit_indices, and expected decoded_payload."""
    version = layout.version
    max_data_bytes = layout.spec.total_data_codewords
    max_bits = max_data_bytes * 8

    if mode == "post_terminator":
        payload = url.encode("utf-8")
        bits = _encode_byte_header_and_payload(payload, version)
        if len(bits) + 4 > max_bits:
            raise ValueError(
                f"URL too long for V{version}-{layout.ec_level} post_terminator mode"
            )
        bits.extend([0, 0, 0, 0])
        while len(bits) % 8 != 0:
            bits.append(0)
        free_start = len(bits)
        bits.extend([0] * (max_bits - free_start))
        free_indices = list(range(free_start, max_bits))
        return bits, free_indices, url

    if mode == "url_fragment":
        header_bytes = 2 if version <= 9 else 3
        prefix_bytes = url.encode("utf-8") + b"#"
        avail_frag = max_data_bytes - header_bytes - len(prefix_bytes)
        if avail_frag <= 0:
            raise ValueError(
                f"URL too long for V{version}-{layout.ec_level} url_fragment mode"
            )
        k_frag = avail_frag if fragment_bytes is None else min(fragment_bytes, avail_frag)
        if k_frag <= 0:
            raise ValueError("fragment_bytes must be positive")
        full_payload = prefix_bytes + bytes([0x40] * k_frag)
        bits = _encode_byte_header_and_payload(full_payload, version)
        prefix_bit_len = len(_encode_byte_header_and_payload(prefix_bytes, version))
        free_indices: list[int] = []
        for fi in range(k_frag):
            byte_start = prefix_bit_len + fi * 8
            free_indices.extend([byte_start + 4, byte_start + 5, byte_start + 6, byte_start + 7])

        bits.extend([0, 0, 0, 0])
        while len(bits) % 8 != 0:
            bits.append(0)
        pad_bytes = [0xEC, 0x11]
        pad_idx = 0
        while len(bits) < max_bits:
            b = pad_bytes[pad_idx % 2]
            for sh in range(7, -1, -1):
                bits.append((b >> sh) & 1)
            pad_idx += 1
        return bits, free_indices, full_payload.decode("ascii")

    raise ValueError(f"Unsupported QArt mode: {mode!r}")


def _build_sensitivity_matrix(layout: QRMatrixLayout, free_indices: list[int]) -> np.ndarray:
    """Return (N, N) object array of Python int bitmasks for F2 sensitivities."""
    n = layout.size
    spec = layout.spec
    sens = np.zeros((n, n), dtype=object)

    byte_to_block: list[tuple[int, int]] = []
    for b_idx, (d_len, _) in enumerate(spec.blocks):
        for local_idx in range(d_len):
            byte_to_block.append((b_idx, local_idx))

    for var_id, global_bit_idx in enumerate(free_indices):
        var_mask = 1 << var_id
        global_byte_idx = global_bit_idx // 8
        bit_in_byte = global_bit_idx % 8
        b_idx, local_byte_idx = byte_to_block[global_byte_idx]
        d_len, e_len = spec.blocks[b_idx]

        tx_data = layout.block_to_tx[(b_idx, False, local_byte_idx)]
        r_d, c_d = layout.codeword_coords[tx_data][bit_in_byte]
        sens[r_d, c_d] |= var_mask

        unit_block = [0] * d_len
        unit_block[local_byte_idx] = 1 << (7 - bit_in_byte)
        ec_delta = rs_encode_block(unit_block, e_len)
        for ec_idx, ec_byte in enumerate(ec_delta):
            if ec_byte == 0:
                continue
            tx_ec = layout.block_to_tx[(b_idx, True, ec_idx)]
            ec_coords = layout.codeword_coords[tx_ec]
            for eb in range(8):
                if (ec_byte >> (7 - eb)) & 1:
                    r_e, c_e = ec_coords[eb]
                    sens[r_e, c_e] |= var_mask

    return sens


def _solve_f2_priority_system(
    pixel_order: list[tuple[int, int]],
    sens: np.ndarray,
    rhs_grid: np.ndarray,
    num_vars: int,
) -> list[int]:
    """Greedy priority-ordered Gaussian elimination over F2 using int bitsets."""
    basis: dict[int, tuple[int, int]] = {}

    for r, c in pixel_order:
        row_mask = int(sens[r, c])
        if row_mask == 0:
            continue
        row_rhs = int(rhs_grid[r, c])
        while row_mask:
            pivot = row_mask.bit_length() - 1
            existing = basis.get(pivot)
            if existing is None:
                basis[pivot] = (row_mask, row_rhs)
                break
            b_mask, b_rhs = existing
            row_mask ^= b_mask
            row_rhs ^= b_rhs

    x_int = 0
    for pivot in sorted(basis.keys()):
        row_mask, row_rhs = basis[pivot]
        lower = row_mask ^ (1 << pivot)
        parity = (lower & x_int).bit_count() & 1
        if row_rhs ^ parity:
            x_int |= 1 << pivot

    return [(x_int >> k) & 1 for k in range(num_vars)]


def solve_qart(
    url: str,
    version: int,
    ec_level: str,
    target: np.ndarray,
    priority: np.ndarray,
    mode: str = "post_terminator",
    fragment_bytes: int | None = None,
    masks: list[int] | None = None,
) -> QArtResult:
    """Solve QR data bits via F2 elimination to match target[r, c] weighted by priority."""
    layout = build_layout(version, ec_level)
    n = layout.size
    if target.shape != (n, n) or priority.shape != (n, n):
        raise ValueError(f"target and priority must have shape {(n, n)}")

    base_bits, free_indices, base_decoded = _prepare_qart_variables(
        url, layout, mode=mode, fragment_bytes=fragment_bytes
    )
    base_data_cw = _bits_to_bytes(base_bits)
    base_all_cw = _assemble_all_codewords(layout, base_data_cw)
    base_unmasked = _render_unmasked_matrix(layout, base_all_cw)

    sens = _build_sensitivity_matrix(layout, free_indices)

    candidates: list[tuple[float, int, int]] = []
    center = (n - 1) / 2.0
    total_weight = 0.0
    for r in range(n):
        for c in range(n):
            p = float(priority[r, c])
            if p > 0.0:
                total_weight += p
                if not layout.is_function[r, c]:
                    dist2 = (r - center) ** 2 + (c - center) ** 2
                    candidates.append((p, -dist2, r * n + c))
    candidates.sort(reverse=True)
    pixel_order = [(rc // n, rc % n) for _, _, rc in candidates]

    mask_candidates = list(range(8)) if masks is None else masks
    best_result: QArtResult | None = None

    for mask_id in mask_candidates:
        mfn = MASK_FUNCTIONS[mask_id]
        rhs_grid = np.zeros((n, n), dtype=np.uint8)
        for r in range(n):
            for c in range(n):
                if not layout.is_function[r, c]:
                    rhs_grid[r, c] = int(target[r, c]) ^ mfn(r, c) ^ int(base_unmasked[r, c])

        x_vec = _solve_f2_priority_system(pixel_order, sens, rhs_grid, len(free_indices))
        solved_bits = list(base_bits)
        for k, val in enumerate(x_vec):
            if val:
                solved_bits[free_indices[k]] ^= 1

        data_cw = _bits_to_bytes(solved_bits)
        all_cw = _assemble_all_codewords(layout, data_cw)
        unmasked = _render_unmasked_matrix(layout, all_cw)
        final_mat = _apply_mask_and_format(unmasked, layout, mask_id)

        matched = final_mat == target
        satisfied_weight = float(np.sum(priority[matched]))

        if mode == "url_fragment":
            c_len = _char_count_bit_len(version)
            start = 4 + c_len
            msg_bytes = bytes(
                int("".join(str(b) for b in solved_bits[start + i * 8 : start + (i + 1) * 8]), 2)
                for i in range(len(base_decoded))
            )
            decoded_payload = msg_bytes.decode("ascii")
        else:
            decoded_payload = url

        candidate = QArtResult(
            version=version,
            ec_level=layout.ec_level,
            mask=mask_id,
            mode=mode,
            decoded_payload=decoded_payload,
            matrix=final_mat,
            data_codewords=data_cw,
            all_codewords=all_cw,
            satisfied_weight=satisfied_weight,
            total_weight=total_weight,
            matched_mask=matched,
        )
        if best_result is None or candidate.satisfied_weight > best_result.satisfied_weight:
            best_result = candidate

    assert best_result is not None
    return best_result
