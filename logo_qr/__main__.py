"""Allow `python3 -m logo_qr` execution."""

from __future__ import annotations

import sys

from logo_qr.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
