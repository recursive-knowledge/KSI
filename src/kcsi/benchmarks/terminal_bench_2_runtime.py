"""Compatibility alias for the Terminal-Bench 2 runtime implementation.

The trial runner lives under :mod:`kcsi.runtime.terminal_bench_2_trial`, where
runtime ownership belongs. Keep this module as an alias so existing imports and
monkeypatch paths continue to resolve to the implementation module.
"""

from __future__ import annotations

import sys

from ..runtime import terminal_bench_2_trial as _impl

setattr(sys.modules[__package__], "terminal_bench_2_runtime", _impl)
sys.modules[__name__] = _impl
