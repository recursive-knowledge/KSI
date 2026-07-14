"""Exercises scripts/dev/check_worktree_discipline.sh end-to-end.

The script is a lightweight smoke-test agents run at the start of a
parallel-isolation run. We cover:
  - happy path: passing the script the worktree it lives in -> exit 0.
  - mismatch path: passing the script a directory that is NOT a git
    worktree -> exit 1 with the FAIL prefix on stderr.
  - missing-arg path: no arg -> exit 1.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_worktree_discipline.sh"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"
    # Not asserting exec bit here (git preserves it but test envs vary);
    # we always run it via `bash SCRIPT` anyway.


def test_happy_path_matches_current_worktree():
    """Passing the repo root to the script resolves to its own git
    top-level → OK."""
    result = _run(str(REPO_ROOT))
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_mismatch_path_fails(tmp_path: Path):
    """A fresh tmp directory is not a git repo → FAIL."""
    result = _run(str(tmp_path))
    assert result.returncode != 0, (
        f"expected non-zero exit when pointing at a non-git directory; stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "FAIL" in result.stderr


def test_missing_arg_fails():
    """No argument → FAIL."""
    result = _run()
    assert result.returncode != 0
    assert "FAIL" in result.stderr


@pytest.mark.parametrize("bad_path", ["/nonexistent/path/should-not-exist-12345"])
def test_nonexistent_expected_root_fails(bad_path: str):
    result = _run(bad_path)
    assert result.returncode != 0
    assert "FAIL" in result.stderr
