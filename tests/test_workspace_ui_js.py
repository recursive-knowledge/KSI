"""Pytest wrapper that runs the Node unit tests under ``tests/js/``.

Covers:
  * ``grid.test.mjs`` — ARC workspace UI pure helpers
    (``assertGrid`` / ``cloneGrid`` / ``zeroGrid``) that validate and
    manipulate prediction grids.
  * ``token_accumulation.test.mjs`` — regression test for the 2026-04
    zero-token reporting bug in the agent-runner streaming accumulator.

All ``*.test.mjs`` files under ``tests/js/`` are auto-discovered by
``node --test`` so new JS tests are picked up without changes here.

Skipped (not failed) if ``node`` is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from conftest import REPO_ROOT

JS_TESTS_DIR = REPO_ROOT / "tests" / "js"


def test_workspace_ui_grid_js():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; skipping JS tests")

    test_files = sorted(JS_TESTS_DIR.glob("*.test.mjs"))
    if not test_files:
        pytest.skip(f"no JS test files under {JS_TESTS_DIR}")

    result = subprocess.run(
        [node, "--test", *[str(p) for p in test_files]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"node --test failed (exit {result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
