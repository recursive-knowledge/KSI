#!/usr/bin/env python3
"""Compatibility wrapper for benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from arc_prep import prepare_arc_workspace_payloads as _impl  # noqa: E402

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
main = _impl.main


if __name__ == "__main__":
    main()
