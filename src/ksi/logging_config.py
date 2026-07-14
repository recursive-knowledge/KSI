"""Centralized logging configuration for the ``ksi`` package.

This module provides a single :func:`configure_logging` entry point that
sets up the package-level logger (``logging.getLogger("ksi")``) with a
consistent formatter and a single stream handler. The helper is idempotent,
so calling it multiple times (e.g., from different CLI entry points) does
not attach duplicate handlers.

The default level is read from the ``KSI_LOG_LEVEL`` environment
variable and falls back to ``INFO`` if the variable is unset or invalid.
Callers may override the level explicitly by passing ``level=...``.

Usage::

    from ksi.logging_config import configure_logging
    configure_logging()

    import logging
    log = logging.getLogger(__name__)
    log.info("hello")
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"
_PACKAGE_LOGGER_NAME = "ksi"


def _resolve_level(level: Optional[Union[int, str]]) -> int:
    """Resolve a logging level from arg/env, defaulting to ``INFO``."""
    if level is None:
        level = os.environ.get("KSI_LOG_LEVEL")
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    try:
        # Accept both numeric strings ("20") and names ("INFO", "debug").
        if isinstance(level, str) and level.strip().isdigit():
            return int(level.strip())
        resolved = logging.getLevelName(str(level).upper())
        if isinstance(resolved, int):
            return resolved
    except Exception:
        pass
    return logging.INFO


def configure_logging(level: Optional[Union[int, str]] = None) -> logging.Logger:
    """Configure the ``ksi`` package logger.

    Parameters
    ----------
    level:
        Optional explicit log level (int or string). If omitted, the level
        is read from the ``KSI_LOG_LEVEL`` env var and defaults to
        ``INFO``.

    Returns
    -------
    logging.Logger
        The configured package logger.

    Notes
    -----
    - Idempotent: repeat calls only adjust the level; handlers are not
      duplicated.
    - Attaches a single :class:`logging.StreamHandler` with a fixed
      formatter (``DEFAULT_FORMAT``).
    - Sets ``propagate=True`` so messages bubble up to the root logger,
      enabling pytest's caplog and other root-level handlers to observe them.
    """
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    resolved = _resolve_level(level)
    logger.setLevel(resolved)
    # Keep propagation enabled so pytest's caplog and any caller that
    # configures the root logger can still observe these messages.
    logger.propagate = True

    # Idempotency: only attach our handler once. We tag the handler so
    # subsequent calls can recognize and reuse it.
    for handler in logger.handlers:
        if getattr(handler, "_ksi_configured", False):
            handler.setLevel(resolved)
            return logger

    handler = logging.StreamHandler()
    handler.setLevel(resolved)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))
    handler._ksi_configured = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


__all__ = ["configure_logging", "DEFAULT_FORMAT", "DEFAULT_DATEFMT"]
