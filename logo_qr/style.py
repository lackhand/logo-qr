"""Validated color palette, placement, and layout presets for logo-centered QR codes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

VALID_EC_LEVELS = frozenset({"L", "M", "Q", "H"})
SUPPORTED_LOGO_SUFFIXES = frozenset({".svg", ".png", ".webp"})
VALID_LAYOUT_MODES = frozenset({"single", "duo-cozy", "duo-splayed", "duo-chamber"})
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def parse_hex_rgb(hex_color: str, field_name: str = "color") -> tuple[int, int, int]:
    """Parse a strict #RRGGBB hex string into an (R, G, B) tuple in 0..255."""
    if not isinstance(hex_color, str) or not _HEX_RE.match(hex_color):
        raise ValueError(f"{field_name} must be a strict #RRGGBB hex color, got {hex_color!r}")
    return (
        int(hex_color[1:3], 16),
        int(hex_color[3:5], 16),
        int(hex_color[5:7], 16),
    )


def bt601_luma(rgb: tuple[int, int, int]) -> float:
    """Compute ITU-R BT.601 normalized luminance Y in [0.0, 1.0]."""
    r, g, b = rgb
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


@dataclass(frozen=True)
class QRPalette:
    """Contrast-validated palette for QR corridors, accents, quiet zone, and optional frame."""

    wall_color: str = "#1E293B"
    accent_color: str | None = "#0F172A"
    background_color: str = "#FFFFFF"
    frame_color: str | None = None

    def __post_init__(self) -> None:
        wall_rgb = parse_hex_rgb(self.wall_color, "wall_color")
        object.__setattr__(self, "wall_color", self.wall_color.upper())

        bg_rgb = parse_hex_rgb(self.background_color, "background_color")
        object.__setattr__(self, "background_color", self.background_color.upper())

        wall_y = bt601_luma(wall_rgb)
        if wall_y > 0.20:
            raise ValueError(
                f"wall_color {self.wall_color} has BT.601 luma {wall_y:.3f} > 0.20 (must be dark for reliable scanning)"
            )

        max_dark_y = wall_y
        if self.accent_color is not None:
            accent_rgb = parse_hex_rgb(self.accent_color, "accent_color")
            object.__setattr__(self, "accent_color", self.accent_color.upper())
            accent_y = bt601_luma(accent_rgb)
            if accent_y > 0.20:
                raise ValueError(
                    f"accent_color {self.accent_color} has BT.601 luma {accent_y:.3f} > 0.20"
                )
            max_dark_y = max(max_dark_y, accent_y)

        bg_y = bt601_luma(bg_rgb)
        if bg_y < 0.85:
            raise ValueError(
                f"background_color {self.background_color} has BT.601 luma {bg_y:.3f} < 0.85 (must be light)"
            )

        delta_y = bg_y - max_dark_y
        if delta_y < 0.65:
            raise ValueError(
                f"QR contrast separation delta_y={delta_y:.3f} < 0.65 between {self.background_color} and dark modules"
            )

        if self.frame_color is not None:
            parse_hex_rgb(self.frame_color, "frame_color")
            object.__setattr__(self, "frame_color", self.frame_color.upper())

    @property
    def wall_rgb(self) -> tuple[int, int, int]:
        return parse_hex_rgb(self.wall_color, "wall_color")

    @property
    def accent_rgb(self) -> tuple[int, int, int]:
        hex_val = self.accent_color if self.accent_color is not None else self.wall_color
        return parse_hex_rgb(hex_val, "accent_color")

    @property
    def accent_hex(self) -> str:
        return self.accent_color if self.accent_color is not None else self.wall_color

    @property
    def background_rgb(self) -> tuple[int, int, int]:
        return parse_hex_rgb(self.background_color, "background_color")

    @property
    def frame_rgb(self) -> tuple[int, int, int] | None:
        if self.frame_color is None:
            return None
        return parse_hex_rgb(self.frame_color, "frame_color")


@dataclass(frozen=True)
class LogoPlacement:
    """Single logo placement inside the QR matrix (clockwise-positive angles)."""

    logo_path: Path | str
    dx_frac: float = 0.0
    dy_frac: float = 0.0
    scale_frac: float = 0.34
    angle_deg: float = 0.0
    rim_mod: float = 0.0

    def __post_init__(self) -> None:
        path = Path(self.logo_path)
        if not path.exists():
            raise FileNotFoundError(f"Logo file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_LOGO_SUFFIXES:
            raise ValueError(
                f"Unsupported logo format {path.suffix!r}; expected one of {sorted(SUPPORTED_LOGO_SUFFIXES)}"
            )
        object.__setattr__(self, "logo_path", path)

        if not (0.05 <= self.scale_frac <= 0.70):
            raise ValueError(f"scale_frac must be in [0.05, 0.70], got {self.scale_frac}")
        if not (-90.0 <= self.angle_deg <= 90.0):
            raise ValueError(f"angle_deg must be in [-90, 90], got {self.angle_deg}")
        if not (-0.45 <= self.dx_frac <= 0.45):
            raise ValueError(f"dx_frac must be in [-0.45, 0.45], got {self.dx_frac}")
        if not (-0.45 <= self.dy_frac <= 0.45):
            raise ValueError(f"dy_frac must be in [-0.45, 0.45], got {self.dy_frac}")
        if not (0.0 <= self.rim_mod <= 2.5):
            raise ValueError(f"rim_mod must be in [0.0, 2.5], got {self.rim_mod}")


@dataclass(frozen=True)
class LogoQRDesign:
    """Full configuration for rendering a logo-centered QR code."""

    version: int | None = None
    ec_level: str = "H"
    placements: tuple[LogoPlacement, ...] = ()
    palette: QRPalette = QRPalette()
    halo_pad_mod: float = 0.65
    frame_mod: float = 0.0
    chamber_radius_frac: float = 0.0
    chamber_door_half_deg: float = 22.0
    module_px: int = 24
    quiet_mod: int = 4
    wall_radius_frac: float = 0.44
    finder_outer_radius_frac: float = 0.85
    use_run_dots: bool = True

    def __post_init__(self) -> None:
        if self.version is not None and not (4 <= self.version <= 14):
            raise ValueError(f"version must be None or in [4, 14], got {self.version}")
        ec = self.ec_level.upper()
        if ec not in VALID_EC_LEVELS:
            raise ValueError(
                f"ec_level must be one of {sorted(VALID_EC_LEVELS)}, got {self.ec_level!r}"
            )
        object.__setattr__(self, "ec_level", ec)
        if not self.placements:
            raise ValueError("placements must contain at least one LogoPlacement")
        if not (0.0 <= self.halo_pad_mod <= 3.0):
            raise ValueError(f"halo_pad_mod must be in [0.0, 3.0], got {self.halo_pad_mod}")
        if not (0.0 <= self.frame_mod <= 4.0):
            raise ValueError(f"frame_mod must be in [0.0, 4.0], got {self.frame_mod}")
        if self.frame_mod > 0.0 and self.palette.frame_color is None:
            raise ValueError("frame_mod > 0 requires palette.frame_color to be set")
        if not (0.0 <= self.chamber_radius_frac <= 0.35):
            raise ValueError(
                f"chamber_radius_frac must be in [0.0, 0.35], got {self.chamber_radius_frac}"
            )
        if self.module_px < 8:
            raise ValueError(f"module_px must be >= 8, got {self.module_px}")
        if self.quiet_mod < 4:
            raise ValueError(f"quiet_mod must be >= 4 for ISO compliance, got {self.quiet_mod}")
        if not (0.30 <= self.wall_radius_frac <= 0.50):
            raise ValueError(f"wall_radius_frac must be in [0.30, 0.50], got {self.wall_radius_frac}")
        if not (0.0 <= self.finder_outer_radius_frac <= 2.5):
            raise ValueError(
                f"finder_outer_radius_frac must be in [0.0, 2.5], got {self.finder_outer_radius_frac}"
            )


def build_layout_placements(
    logos: Sequence[Path | str],
    layout_mode: str,
    scale_frac: float | None = None,
    angle_deg: float | None = None,
) -> tuple[tuple[LogoPlacement, ...], float]:
    """Construct placements and chamber_radius_frac for standard single/duo presets."""
    if not logos:
        raise ValueError("At least one logo path is required")
    if layout_mode not in VALID_LAYOUT_MODES:
        raise ValueError(
            f"Unsupported layout mode {layout_mode!r}; expected one of {sorted(VALID_LAYOUT_MODES)}"
        )

    if layout_mode == "single":
        if len(logos) != 1:
            raise ValueError(f"single layout accepts exactly 1 logo path, got {len(logos)}")
        scale = 0.35 if scale_frac is None else scale_frac
        angle = 0.0 if angle_deg is None else angle_deg
        return (
            (LogoPlacement(logo_path=logos[0], scale_frac=scale, angle_deg=angle),),
            0.0,
        )

    if len(logos) > 2:
        raise ValueError(f"{layout_mode} layout accepts at most 2 logo paths, got {len(logos)}")
    first = logos[0]
    second = logos[1] if len(logos) == 2 else logos[0]

    if layout_mode == "duo-cozy":
        base_scale = 0.295 if scale_frac is None else scale_frac
        ang_l = -18.0 if angle_deg is None else -abs(angle_deg)
        ang_r = 22.0 if angle_deg is None else abs(angle_deg)
        return (
            (
                LogoPlacement(logo_path=first, dx_frac=-0.060, dy_frac=-0.010, scale_frac=base_scale, angle_deg=ang_l),
                LogoPlacement(
                    logo_path=second,
                    dx_frac=0.058,
                    dy_frac=0.022,
                    scale_frac=min(0.70, base_scale + 0.010),
                    angle_deg=ang_r,
                    rim_mod=0.55,
                ),
            ),
            0.0,
        )

    if layout_mode == "duo-splayed":
        base_scale = 0.295 if scale_frac is None else scale_frac
        ang_l = -34.0 if angle_deg is None else -abs(angle_deg)
        ang_r = 36.0 if angle_deg is None else abs(angle_deg)
        return (
            (
                LogoPlacement(logo_path=first, dx_frac=-0.072, dy_frac=0.015, scale_frac=base_scale, angle_deg=ang_l),
                LogoPlacement(
                    logo_path=second,
                    dx_frac=0.070,
                    dy_frac=-0.012,
                    scale_frac=min(0.70, base_scale + 0.010),
                    angle_deg=ang_r,
                    rim_mod=0.55,
                ),
            ),
            0.0,
        )

    # duo-chamber
    base_scale = 0.242 if scale_frac is None else scale_frac
    ang_l = -24.0 if angle_deg is None else -abs(angle_deg)
    ang_r = 28.0 if angle_deg is None else abs(angle_deg)
    return (
        (
            LogoPlacement(logo_path=first, dx_frac=-0.048, dy_frac=0.010, scale_frac=base_scale, angle_deg=ang_l),
            LogoPlacement(
                logo_path=second,
                dx_frac=0.048,
                dy_frac=0.022,
                scale_frac=min(0.70, base_scale + 0.008),
                angle_deg=ang_r,
                rim_mod=0.50,
            ),
        ),
        0.212,
    )
