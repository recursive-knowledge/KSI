"""Regression guard for the benchmarks<->tasks reentrant-import invariant.

``kcsi.benchmarks.register_all()`` must yield the canonical five-source
tuple no matter which package a fresh interpreter imports first:
``benchmarks/__init__.py`` defines ``register_all``/``_registered`` BEFORE
its eager evaluator imports (which reenter ``kcsi.tasks``), and
``tasks/loaders.py`` computes ``SUPPORTED_TASK_SOURCES`` after the wiring
block. Reordering either silently breaks the losing import direction while
the rest of the suite (which always enters via ``kcsi.tasks``) stays green.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

CANONICAL = "('swebench_pro', 'arc', 'polyglot', 'terminal_bench_2', 'custom')"

_IMPORT_ORDERS = {
    "tasks_first": "import kcsi.tasks.loaders, kcsi.benchmarks",
    "benchmarks_first": "import kcsi.benchmarks, kcsi.tasks.loaders",
    "benchmarks_loaders_first": "import kcsi.benchmarks.loaders, kcsi.tasks.loaders",
    "registry_only": "import kcsi.tasks.loaders",
}


@pytest.mark.parametrize("label", sorted(_IMPORT_ORDERS))
def test_supported_task_sources_invariant_under_import_order(label: str) -> None:
    code = (
        f"{_IMPORT_ORDERS[label]}; "
        "from kcsi.tasks.registry import supported_task_sources; "
        "print(tuple(supported_task_sources()))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{label}: import failed:\n{proc.stderr}"
    assert proc.stdout.strip() == CANONICAL, f"{label}: got {proc.stdout.strip()!r}"
