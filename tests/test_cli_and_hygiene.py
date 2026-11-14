"""End-to-end CLI tests, zero-zombie checks, and repository content hygiene checks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from logo_qr.verify import verify_qr_image

REPO_ROOT = Path(__file__).resolve().parent.parent
STAR_SVG = REPO_ROOT / "tests" / "assets" / "star_badge.svg"
GEM_SVG = REPO_ROOT / "tests" / "assets" / "gem_offset.svg"

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

LEGACY_REMOVED_SYMBOLS = (
    "LogoPlacement",
    "build_layout_placements",
    "VALID_LAYOUT_MODES",
    "duo-cozy",
    "duo-splayed",
    "duo-chamber",
    "wall_color",
    "accent_color",
    "use_run_dots",
    "chamber_radius_frac",
)

EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", "dist", "build"}


def test_cli_multilayer_generates_and_verifies_png_and_svg(tmp_path: Path):
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
            "--logo",
            str(GEM_SVG),
            "--layer",
            "0: translate(-2.5, -0.3) rotate(-0.05turn) scale(0.86)",
            "--layer",
            "1: translate(2.5, 0.4) rotate(0.06turn) scale(0.88) die-cut(0.55)",
            "--dark-color",
            "#1E293B",
            "--light-color",
            "#FFFFFF",
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
    verify_qr_image(png_out, url)

    svg_only = tmp_path / "svg_only.svg"
    proc_svg = subprocess.run(
        [
            sys.executable,
            "-m",
            "logo_qr",
            "--url",
            url,
            "--logo",
            str(STAR_SVG),
            "--svg-out",
            str(svg_only),
            "--verify",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_svg.returncode == 0, f"CLI --svg-out --verify failed: {proc_svg.stderr}\n{proc_svg.stdout}"
    assert svg_only.exists()

    for bad_args in (
        ["--dark-color", "#FFFFFF", "--png-out", str(tmp_path / "bad_color.png")],
        ["--layer", "0: rotate(-0.25)", "--png-out", str(tmp_path / "bad_angle.png")],
        ["--layer", "3: scale(1.0)", "--png-out", str(tmp_path / "bad_idx.png")],
    ):
        bad_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "logo_qr",
                "--url",
                url,
                "--logo",
                str(STAR_SVG),
                *bad_args,
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert bad_proc.returncode != 0


def test_repository_hygiene_no_forbidden_substrings_or_zombies():
    assert not (REPO_ROOT / "tools").exists(), "Legacy tools/ directory must not exist"
    assert not (REPO_ROOT / "bin").exists(), "Legacy bin/ directory must not exist"

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
            text = raw.decode("utf-8", errors="ignore")
            lowered = text.lower()
            for term in forbidden:
                assert term not in lowered, f"Forbidden term found in {rel}"
            if rel.startswith("logo_qr/"):
                for legacy in LEGACY_REMOVED_SYMBOLS:
                    assert legacy not in text, f"Legacy removed symbol {legacy!r} found in {rel}"

    git_log = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "--all", "--format=%B"],
        capture_output=True,
        text=True,
        check=True,
    )
    log_lower = git_log.stdout.lower()
    for term in forbidden:
        assert term not in log_lower, "Forbidden term found in git commit history"
