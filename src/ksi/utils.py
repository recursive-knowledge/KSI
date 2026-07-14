"""Shared utility helpers used across the ksi package."""

from __future__ import annotations

from typing import Any


def to_int(value: Any, default: int = 0) -> int:
    """Coerce *value* to ``int``, returning *default* on failure or ``None``."""
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default
