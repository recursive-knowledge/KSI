"""Exercises scripts/typecheck_agent_runner.sh strict/CI fail-hard behavior.

Regression guard for issue #1226: in CI (strict) mode the agent-runner
typecheck must FAIL HARD when its tooling is unavailable — missing npm/node,
a failed ``npm ci``, or a missing ``tsc`` — instead of failing open (exit 0)
and letting CI go green without ever typechecking agent-runner/src. Outside
strict mode the local soft-skip convenience is preserved.

Tests are hermetic: they never run a real ``npm ci`` against the network. We
simulate missing tooling by handing the script a PATH that omits npm/node, and
simulate a failing ``npm ci`` with a stub ``npm`` that exits nonzero.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "typecheck_agent_runner.sh"

# Minimal coreutils the script needs before it reaches the npm/node check.
_CORE_UTILS = ("bash", "env", "cat", "dirname", "pwd", "mkdir", "cp", "ln", "md5sum", "awk")


def _make_bin_dir(tmp_path: Path, *, with_npm_node: bool, npm_fails: bool) -> Path:
    """Build an isolated bin dir with only the tools we want visible."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in _CORE_UTILS:
        real = _which(name)
        if real:
            (bindir / name).symlink_to(real)
    if with_npm_node:
        node = _which("node")
        if node:
            (bindir / "node").symlink_to(node)
        npm_path = bindir / "npm"
        if npm_fails:
            npm_path.write_text("#!/bin/bash\nexit 1\n")
            npm_path.chmod(npm_path.stat().st_mode | stat.S_IEXEC)
        else:
            real_npm = _which("npm")
            if real_npm:
                npm_path.symlink_to(real_npm)
    return bindir


def _which(name: str) -> str | None:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / name
        if cand.exists():
            return str(cand)
    return None


def _run(env: dict[str, str], path: str) -> subprocess.CompletedProcess:
    full_env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": path}
    full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
        env=full_env,
    )


def test_script_exists():
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"


def test_strict_ci_fails_hard_when_npm_missing(tmp_path):
    """CI=true + no npm/node on PATH -> nonzero (fail hard, not fail open)."""
    bindir = _make_bin_dir(tmp_path, with_npm_node=False, npm_fails=False)
    result = _run({"CI": "true"}, str(bindir))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "npm/node not on PATH" in result.stderr
    assert "failing" in result.stderr


def test_strict_via_dedicated_flag_fails_hard_when_npm_missing(tmp_path):
    """KSI_TYPECHECK_STRICT=1 also triggers fail-hard, independent of CI."""
    bindir = _make_bin_dir(tmp_path, with_npm_node=False, npm_fails=False)
    result = _run({"KSI_TYPECHECK_STRICT": "1"}, str(bindir))
    assert result.returncode != 0, result.stdout + result.stderr


def test_local_soft_skip_when_npm_missing(tmp_path):
    """Without CI/strict, missing npm/node -> soft skip (exit 0)."""
    bindir = _make_bin_dir(tmp_path, with_npm_node=False, npm_fails=False)
    result = _run({}, str(bindir))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipping typecheck" in result.stderr


def test_strict_fails_hard_when_npm_ci_fails(tmp_path):
    """CI=true + a failing `npm ci` -> nonzero (fail hard). Hermetic: the
    stub npm exits 1 without touching the network. Uses an isolated cache so
    it never sees a warm node_modules."""
    bindir = _make_bin_dir(tmp_path, with_npm_node=True, npm_fails=True)
    cache = tmp_path / "cache"
    result = _run(
        {"CI": "true", "KSI_TYPECHECK_CACHE": str(cache)},
        str(bindir) + os.pathsep + os.environ.get("PATH", ""),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "npm ci failed" in result.stderr


def test_local_soft_skip_when_npm_ci_fails(tmp_path):
    """Without CI/strict, a failing `npm ci` -> soft skip (exit 0)."""
    bindir = _make_bin_dir(tmp_path, with_npm_node=True, npm_fails=True)
    cache = tmp_path / "cache"
    result = _run(
        {"KSI_TYPECHECK_CACHE": str(cache)},
        str(bindir) + os.pathsep + os.environ.get("PATH", ""),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_copied_script(env: dict[str, str], scripts_dir: Path) -> subprocess.CompletedProcess:
    """Run a copy of the script whose sibling ``runtime_runner/agent-runner/src``
    does not exist, so the missing-SRC_DIR branch is exercised. Uses the real
    PATH so tooling presence is not the reason for any skip/fail."""
    copied = scripts_dir / "typecheck_agent_runner.sh"
    copied.write_text(SCRIPT.read_text())
    full_env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": os.environ.get("PATH", ""),
    }
    full_env.update(env)
    return subprocess.run(
        ["bash", str(copied)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(scripts_dir.parent),
        env=full_env,
    )


def test_strict_ci_fails_hard_when_src_dir_missing(tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    result = _run_copied_script({"CI": "true"}, scripts_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "not found" in result.stderr


def test_local_soft_skip_when_src_dir_missing(tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    result = _run_copied_script({}, scripts_dir)
    assert result.returncode == 0, result.stdout + result.stderr


def test_skip_flag_bypasses_even_in_strict_mode(tmp_path):
    """KSI_TYPECHECK_SKIP=1 short-circuits before the tooling checks, even
    under CI=true — an explicit opt-out is always honored."""
    bindir = _make_bin_dir(tmp_path, with_npm_node=False, npm_fails=False)
    result = _run({"CI": "true", "KSI_TYPECHECK_SKIP": "1"}, str(bindir))
    assert result.returncode == 0, result.stdout + result.stderr
