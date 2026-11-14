"""End-to-end CLI tests and repository content hygiene checks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.qr_verifiers import run_vision_qr

REPO_ROOT = Path(__file__).resolve().parent.parent
STAR_SVG = REPO_ROOT / "tests" / "assets" / "star_badge.svg"

# Hex-encoded legacy terms so literal strings never appear in source or tests
FORBIDDEN_HEX_TERMS = (
    "7375646f6b75",              # legacy puzzle term
    "7261742072756e",            # two-word maze title
    "7261745f72756e",            # underscored maze title
    "6c6f6769632d6d617374657273",  # specific portal domain
    "303030696669",              # specific puzzle id
    "7375646f6b75706164",        # specific app name
    "66696e6b7a",                # character name
    "63757063616b65",            # old emblem subject
)

EXCLUDED_DIRS = {".git", "bin", "__pycache__", ".pytest_cache", "dist", "build"}


def test_cli_generates_and_verifies_svg_and_png(tmp_path: Path):
    png_out = tmp_path / "cli_out.png"
    svg_out = tmp_path / "cli_out.svg"
    url = "https://example.com/cli-test"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "logo_qr",
            "--url",
            url,
            "--logo",
            str(STAR_SVG),
            "--layout",
            "duo-cozy",
            "--wall-color",
            "#1E293B",
            "--accent-color",
            "#0F172A",
            "--frame-color",
            "#CBD5E1",
            "--png-out",
            str(png_out),
            "--svg-out",
            str(svg_out),
            "--verify",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}\n{proc.stdout}"
    assert png_out.exists() and svg_out.exists()
    assert url in run_vision_qr(png_out)

    bad_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "logo_qr",
            "--url",
            url,
            "--logo",
            str(STAR_SVG),
            "--wall-color",
            "#FFFFFF",
            "--png-out",
            str(tmp_path / "bad.png"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_proc.returncode != 0


def test_repository_hygiene_no_forbidden_substrings():
    forbidden = [bytes.fromhex(h).decode("ascii") for h in FORBIDDEN_HEX_TERMS]

    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for fname in files:
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(REPO_ROOT)).lower()
            for term in forbidden:
                assert term not in rel, f"Forbidden term in path {rel}"

            raw = fpath.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8", errors="ignore").lower()
            for term in forbidden:
                assert term not in text, f"Forbidden term found in {rel}"
