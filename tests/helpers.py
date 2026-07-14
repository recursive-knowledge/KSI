"""Shared test helpers for the Knowledge-Centric Self-Improvement test suite.

Provides the ``_load_by_path`` utility used by most test modules to load
``kcsi/`` modules under synthetic package names, avoiding the
``src/kcsi`` vs ``kcsi/`` collision enforced by conftest.py.
"""

from __future__ import annotations

import importlib.util
import sys

from conftest import REPO_ROOT

_ROOT = REPO_ROOT  # project root


def _load_by_path(unique_name: str, rel_path: str, package: str | None = None):
    """Load a Python module from a filesystem path under a synthetic module name.

    Parameters
    ----------
    unique_name:
        The name to register in ``sys.modules`` (e.g. ``"kcsi_pkg.models"``).
    rel_path:
        Path relative to the project root (e.g. ``"src/kcsi/models.py"``).
    package:
        Value to assign to ``mod.__package__`` so that relative imports
        resolve correctly.  Set to the synthetic parent package name
        (e.g. ``"kcsi_pkg"``).  Note: ``mod.__spec__.parent`` is a
        read-only computed property derived from ``spec.name``, so
        ``unique_name`` must already reflect the correct dotted hierarchy
        to keep ``__package__`` and ``__spec__.parent`` in agreement.

    Returns
    -------
    types.ModuleType
        The loaded module object.
    """
    abs_path = _ROOT / rel_path
    spec = importlib.util.spec_from_file_location(unique_name, abs_path, submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    if package:
        mod.__package__ = package
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod
