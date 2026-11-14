"""Validated color palette, 2D affine transforms, and CSS/SVG-style multi-layer composition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Sequence

VALID_EC_LEVELS = frozenset({"L", "M", "Q", "H"})
SUPPORTED_LOGO_SUFFIXES = frozenset({".svg", ".png", ".webp"})
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_LAYER_HEADER_RE = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$")
_TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z0-9-]*)\s*\(([^()]*)\)")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_ANGLE_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(turn|deg|rad)$")


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
    """Contrast-validated two-tone QR module palette."""

    dark_color: str = "#1E293B"
    light_color: str = "#FFFFFF"

    def __post_init__(self) -> None:
        dark_rgb = parse_hex_rgb(self.dark_color, "dark_color")
        object.__setattr__(self, "dark_color", self.dark_color.upper())

        light_rgb = parse_hex_rgb(self.light_color, "light_color")
        object.__setattr__(self, "light_color", self.light_color.upper())

        dark_y = bt601_luma(dark_rgb)
        if dark_y > 0.20:
            raise ValueError(
                f"dark_color {self.dark_color} has BT.601 luma {dark_y:.3f} > 0.20 (must be dark for reliable scanning)"
            )

        light_y = bt601_luma(light_rgb)
        if light_y < 0.85:
            raise ValueError(
                f"light_color {self.light_color} has BT.601 luma {light_y:.3f} < 0.85 (must be light)"
            )

        delta_y = light_y - dark_y
        if delta_y < 0.65:
            raise ValueError(
                f"QR contrast separation delta_y={delta_y:.3f} < 0.65 between {self.light_color} and {self.dark_color}"
            )

    @property
    def dark_rgb(self) -> tuple[int, int, int]:
        return parse_hex_rgb(self.dark_color, "dark_color")

    @property
    def light_rgb(self) -> tuple[int, int, int]:
        return parse_hex_rgb(self.light_color, "light_color")


@dataclass(frozen=True)
class Affine2D:
    """2D homogeneous affine matrix matching SVG matrix(a, b, c, d, tx, ty) in QR module units."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d", "tx", "ty"):
            val = float(getattr(self, name))
            if not math.isfinite(val):
                raise ValueError(f"Affine2D component {name} must be finite, got {val!r}")
            object.__setattr__(self, name, val)
        if self.det <= 1e-6:
            raise ValueError(
                f"Affine2D determinant must be > 1e-6 (non-degenerate, orientation-preserving), got {self.det}"
            )

    @property
    def det(self) -> float:
        return self.a * self.d - self.b * self.c

    @property
    def max_scale(self) -> float:
        """Largest singular value of the linear 2x2 submatrix."""
        s_plus = math.hypot(self.a + self.d, self.b - self.c)
        s_minus = math.hypot(self.a - self.d, self.b + self.c)
        return 0.5 * (s_plus + s_minus)

    @classmethod
    def identity(cls) -> Affine2D:
        return cls()

    @classmethod
    def translate(cls, dx: float, dy: float = 0.0) -> Affine2D:
        dx_f = float(dx)
        dy_f = float(dy)
        if not math.isfinite(dx_f) or not (-30.0 <= dx_f <= 30.0):
            raise ValueError(f"Translation tx must be finite in [-30.0, 30.0], got {dx!r}")
        if not math.isfinite(dy_f) or not (-30.0 <= dy_f <= 30.0):
            raise ValueError(f"Translation ty must be finite in [-30.0, 30.0], got {dy!r}")
        return cls(a=1.0, b=0.0, c=0.0, d=1.0, tx=dx_f, ty=dy_f)

    @classmethod
    def rotate_deg(cls, deg: float) -> Affine2D:
        deg_f = float(deg)
        if not math.isfinite(deg_f):
            raise ValueError(f"Rotation angle must be finite, got {deg!r}")
        rad = math.radians(deg_f)
        ca = math.cos(rad)
        sa = math.sin(rad)
        return cls(a=ca, b=sa, c=-sa, d=ca, tx=0.0, ty=0.0)

    @classmethod
    def rotate_turn(cls, turns: float) -> Affine2D:
        turns_f = float(turns)
        if not math.isfinite(turns_f):
            raise ValueError(f"Rotation turns must be finite, got {turns!r}")
        return cls.rotate_deg(turns_f * 360.0)

    @classmethod
    def rotate_rad(cls, rad: float) -> Affine2D:
        rad_f = float(rad)
        if not math.isfinite(rad_f):
            raise ValueError(f"Rotation radians must be finite, got {rad!r}")
        return cls.rotate_deg(math.degrees(rad_f))

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> Affine2D:
        sx_f = float(sx)
        sy_f = sx_f if sy is None else float(sy)
        if not math.isfinite(sx_f) or not (0.10 <= sx_f <= 4.0):
            raise ValueError(f"scale sx must be finite in [0.10, 4.0], got {sx!r}")
        if not math.isfinite(sy_f) or not (0.10 <= sy_f <= 4.0):
            raise ValueError(f"scale sy must be finite in [0.10, 4.0], got {sy!r}")
        return cls(a=sx_f, b=0.0, c=0.0, d=sy_f, tx=0.0, ty=0.0)

    def __matmul__(self, other: Affine2D) -> Affine2D:
        if not isinstance(other, Affine2D):
            return NotImplemented
        return Affine2D(
            a=self.a * other.a + self.c * other.b,
            b=self.b * other.a + self.d * other.b,
            c=self.a * other.c + self.c * other.d,
            d=self.b * other.c + self.d * other.d,
            tx=self.a * other.tx + self.c * other.ty + self.tx,
            ty=self.b * other.tx + self.d * other.ty + self.ty,
        )

    def apply(self, x: float, y: float) -> tuple[float, float]:
        """Transform point (x, y) from layer-local coordinates into QR-center module coordinates."""
        return (
            self.a * x + self.c * y + self.tx,
            self.b * x + self.d * y + self.ty,
        )


@dataclass(frozen=True)
class LogoLayer:
    """Single logo layer with an ordered 2D affine transform and optional die-cut rim."""

    logo_path: Path | str
    transform: Affine2D = Affine2D()
    die_cut_mod: float = 0.0

    def __post_init__(self) -> None:
        path = Path(self.logo_path)
        if not path.exists():
            raise FileNotFoundError(f"Logo file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_LOGO_SUFFIXES:
            raise ValueError(
                f"Unsupported logo format {path.suffix!r}; expected one of {sorted(SUPPORTED_LOGO_SUFFIXES)}"
            )
        object.__setattr__(self, "logo_path", path)

        if not isinstance(self.transform, Affine2D):
            raise ValueError(f"transform must be an Affine2D instance, got {type(self.transform).__name__}")
        if not (0.10 <= self.transform.max_scale <= 4.0):
            raise ValueError(
                f"Composed layer scale {self.transform.max_scale:.3f} must be in [0.10, 4.0]"
            )
        if not (-30.0 <= self.transform.tx <= 30.0):
            raise ValueError(f"Layer translation tx must be in [-30.0, 30.0], got {self.transform.tx}")
        if not (-30.0 <= self.transform.ty <= 30.0):
            raise ValueError(f"Layer translation ty must be in [-30.0, 30.0], got {self.transform.ty}")

        dc = float(self.die_cut_mod)
        if not math.isfinite(dc) or not (0.0 <= dc <= 3.0):
            raise ValueError(f"die_cut_mod must be finite in [0.0, 3.0], got {self.die_cut_mod!r}")
        object.__setattr__(self, "die_cut_mod", dc)


def _parse_unitless_float(raw: str, fn_name: str) -> float:
    token = raw.strip()
    if not _FLOAT_RE.match(token):
        raise ValueError(
            f"{fn_name}() expects unitless QR-module numbers (no 'px', '%', or unit suffixes), got {token!r}"
        )
    val = float(token)
    if not math.isfinite(val):
        raise ValueError(f"{fn_name}() argument must be finite, got {token!r}")
    return val


def _parse_css_angle_deg(raw: str) -> float:
    token = raw.strip()
    m = _ANGLE_RE.match(token)
    if not m:
        raise ValueError(
            f"rotate() requires an explicit angle unit ('turn', 'deg', or 'rad'), got {token!r}"
        )
    value = float(m.group(1))
    unit = m.group(2)
    if not math.isfinite(value):
        raise ValueError(f"rotate() angle must be finite, got {token!r}")
    if unit == "turn":
        return value * 360.0
    if unit == "deg":
        return value
    return math.degrees(value)


def _split_fn_args(arg_str: str) -> list[str]:
    stripped = arg_str.strip()
    if not stripped:
        return []
    parts = [part.strip() for part in stripped.split(",")]
    if any(not p for p in parts):
        raise ValueError(f"Malformed empty argument in {arg_str!r}")
    return parts


def parse_layer_spec(spec: str) -> tuple[int, Affine2D, float]:
    """Parse '<idx>: <css/svg ops>' into (layer_index, Affine2D, die_cut_mod)."""
    if not isinstance(spec, str):
        raise ValueError(f"Layer spec must be a string, got {type(spec).__name__}")
    header_m = _LAYER_HEADER_RE.match(spec)
    if not header_m:
        raise ValueError(
            f"Layer spec must begin with '<index>: <operations>', got {spec!r}"
        )

    layer_idx = int(header_m.group(1))
    body = header_m.group(2).strip()

    transform = Affine2D.identity()
    die_cut_mod: float | None = None
    pos = 0
    op_count = 0
    length = len(body)

    while pos < length:
        if body[pos].isspace():
            pos += 1
            continue
        tok_m = _TOKEN_RE.match(body, pos)
        if not tok_m:
            raise ValueError(
                f"Invalid token near {body[pos:]!r} in layer spec {spec!r}"
            )
        fn_name = tok_m.group(1)
        args = _split_fn_args(tok_m.group(2))
        op_count += 1
        pos = tok_m.end()

        if fn_name == "translate":
            if len(args) not in (1, 2):
                raise ValueError(f"translate() expects 1 or 2 arguments, got {len(args)}")
            dx = _parse_unitless_float(args[0], "translate")
            dy = 0.0 if len(args) == 1 else _parse_unitless_float(args[1], "translate")
            transform = transform @ Affine2D.translate(dx, dy)
        elif fn_name == "translateX":
            if len(args) != 1:
                raise ValueError(f"translateX() expects 1 argument, got {len(args)}")
            dx = _parse_unitless_float(args[0], "translateX")
            transform = transform @ Affine2D.translate(dx, 0.0)
        elif fn_name == "translateY":
            if len(args) != 1:
                raise ValueError(f"translateY() expects 1 argument, got {len(args)}")
            dy = _parse_unitless_float(args[0], "translateY")
            transform = transform @ Affine2D.translate(0.0, dy)
        elif fn_name == "rotate":
            if len(args) != 1:
                raise ValueError(f"rotate() expects 1 argument, got {len(args)}")
            deg = _parse_css_angle_deg(args[0])
            transform = transform @ Affine2D.rotate_deg(deg)
        elif fn_name == "scale":
            if len(args) not in (1, 2):
                raise ValueError(f"scale() expects 1 or 2 arguments, got {len(args)}")
            sx = _parse_unitless_float(args[0], "scale")
            sy = None if len(args) == 1 else _parse_unitless_float(args[1], "scale")
            transform = transform @ Affine2D.scale(sx, sy)
        elif fn_name == "die-cut":
            if die_cut_mod is not None:
                raise ValueError(f"Duplicate die-cut() operation in layer spec {spec!r}")
            if len(args) != 1:
                raise ValueError(f"die-cut() expects 1 argument, got {len(args)}")
            dc = _parse_unitless_float(args[0], "die-cut")
            if not (0.0 <= dc <= 3.0):
                raise ValueError(f"die-cut() width must be in [0.0, 3.0], got {dc}")
            die_cut_mod = dc
        else:
            raise ValueError(
                f"Unsupported layer operation {fn_name!r}; expected one of "
                "translate, translateX, translateY, rotate, scale, die-cut"
            )

    if op_count == 0:
        raise ValueError(f"Empty layer spec {spec!r}: expected transform or die-cut operation")
    return layer_idx, transform, (0.0 if die_cut_mod is None else die_cut_mod)


def build_layers(
    logos: Sequence[Path | str], layer_specs: Sequence[str] = ()
) -> tuple[LogoLayer, ...]:
    """Compose centered ordered LogoLayers from --logo paths and optional --layer overrides."""
    if not logos:
        raise ValueError("At least one logo path is required")

    specs_by_idx: dict[int, tuple[Affine2D, float]] = {}
    for spec in layer_specs:
        idx, transform, die_cut_mod = parse_layer_spec(spec)
        if idx >= len(logos):
            raise ValueError(
                f"Invalid layer index {idx}: out of range for {len(logos)} logo(s) (valid: 0..{len(logos) - 1})"
            )
        if idx in specs_by_idx:
            raise ValueError(f"Duplicate --layer specification for layer index {idx}")
        specs_by_idx[idx] = (transform, die_cut_mod)

    layers: list[LogoLayer] = []
    for idx, logo_path in enumerate(logos):
        transform, die_cut_mod = specs_by_idx.get(idx, (Affine2D.identity(), 0.0))
        layers.append(LogoLayer(logo_path=logo_path, transform=transform, die_cut_mod=die_cut_mod))
    return tuple(layers)


@dataclass(frozen=True)
class LogoQRDesign:
    """Full configuration for rendering a crisp square QR code around layered logos."""

    version: int | None = None
    ec_level: str = "H"
    layers: tuple[LogoLayer, ...] = ()
    palette: QRPalette = QRPalette()
    base_scale_frac: float = 0.34
    halo_mod: float = 0.65
    module_px: int = 24
    quiet_mod: int = 4

    def __post_init__(self) -> None:
        if self.version is not None and not (4 <= self.version <= 14):
            raise ValueError(f"version must be None or in [4, 14], got {self.version}")
        ec = self.ec_level.upper()
        if ec not in VALID_EC_LEVELS:
            raise ValueError(
                f"ec_level must be one of {sorted(VALID_EC_LEVELS)}, got {self.ec_level!r}"
            )
        object.__setattr__(self, "ec_level", ec)
        if not self.layers:
            raise ValueError("layers must contain at least one LogoLayer")
        if not (0.05 <= self.base_scale_frac <= 0.60):
            raise ValueError(f"base_scale_frac must be in [0.05, 0.60], got {self.base_scale_frac}")
        if not (0.0 <= self.halo_mod <= 3.0):
            raise ValueError(f"halo_mod must be in [0.0, 3.0], got {self.halo_mod}")
        if self.module_px < 8:
            raise ValueError(f"module_px must be >= 8, got {self.module_px}")
        if self.quiet_mod < 4:
            raise ValueError(f"quiet_mod must be >= 4 for ISO compliance, got {self.quiet_mod}")
