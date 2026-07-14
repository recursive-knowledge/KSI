"""Guard: the Python and TypeScript ``claude-agent-sdk`` pins stay in lockstep.

The SDK is consumed on BOTH sides of the runtime:

  - Python host: ``claude_agent_sdk`` (``src/ksi/tasks/loaders.py``) — task
    classification drives the bundled Claude Code CLI via ``query()``.
  - TS agent-runner: ``@anthropic-ai/claude-agent-sdk``
    (``runtime_runner/agent-runner/``) — the in-container agent loop drives its
    own bundled CLI via ``query()``.

Anthropic publishes the npm and PyPI packages with matching version numbers, and
each version bundles a specific Claude Code CLI. If the two pins drift, the two
sides run different bundled CLIs and can silently disagree on message shapes /
token reporting at runtime (issue #947). The "keep them in lockstep" rule used
to live only as a comment in ``pyproject.toml``; this test enforces it.

The Python side of truth is ``uv.lock`` (what actually gets installed), NOT the
``pyproject.toml`` range — the range deliberately allows a window, but the
resolved/locked version must equal the exact TS pin.
"""

from __future__ import annotations

import json
import tomllib

from conftest import REPO_ROOT
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
AGENT_RUNNER_PKG = REPO_ROOT / "runtime_runner" / "agent-runner" / "package.json"
AGENT_RUNNER_LOCK = REPO_ROOT / "runtime_runner" / "agent-runner" / "package-lock.json"

_PY_PKG = "claude-agent-sdk"
_TS_PKG = "@anthropic-ai/claude-agent-sdk"


def _python_locked_version() -> str:
    data = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    matches = [p for p in data.get("package", []) if p.get("name") == _PY_PKG]
    assert matches, f"{_PY_PKG} not found in {UV_LOCK} — did the dependency get removed?"
    assert len(matches) == 1, f"{_PY_PKG} appears {len(matches)} times in {UV_LOCK}"
    version = matches[0].get("version")
    assert version, f"{_PY_PKG} entry in {UV_LOCK} has no version"
    return version


def _ts_pinned_version() -> str:
    pkg = json.loads(AGENT_RUNNER_PKG.read_text(encoding="utf-8"))
    version = pkg.get("dependencies", {}).get(_TS_PKG)
    assert version, f"{_TS_PKG} not found in {AGENT_RUNNER_PKG} dependencies"
    return version


def _ts_locked_version() -> str:
    # lockfileVersion 3: resolved versions live under packages[<install path>].
    lock = json.loads(AGENT_RUNNER_LOCK.read_text(encoding="utf-8"))
    entry = lock.get("packages", {}).get(f"node_modules/{_TS_PKG}", {})
    version = entry.get("version")
    assert version, f"{_TS_PKG} not resolved in {AGENT_RUNNER_LOCK}"
    return version


def _pyproject_specifier() -> SpecifierSet:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    for dep in data.get("project", {}).get("dependencies", []):
        req = Requirement(dep)
        if req.name == _PY_PKG:
            return req.specifier
    raise AssertionError(f"no {_PY_PKG} dependency specifier found in {PYPROJECT}")


def test_python_and_ts_claude_agent_sdk_pins_match() -> None:
    py = _python_locked_version()
    ts = _ts_pinned_version()
    assert py == ts, (
        "claude-agent-sdk versions have drifted between the Python host and the "
        f"TS agent-runner (issue #947):\n"
        f"  Python (uv.lock):              {py}\n"
        f"  TS (agent-runner/package.json): {ts}\n"
        "Anthropic ships the npm and PyPI packages with matching version numbers "
        "and each bundles a specific Claude Code CLI. Bump BOTH sides to the same "
        "version: set the exact pin in runtime_runner/agent-runner/package.json "
        "(and re-run `npm install` there) and re-lock Python with "
        '`uv lock --upgrade-package "claude-agent-sdk==<version>"`.'
    )


def test_ts_pin_is_an_exact_version() -> None:
    # The TS side must pin an exact version (no ^/~/range) so the lockstep
    # comparison above is meaningful.
    ts = _ts_pinned_version()
    assert ts[:1].isdigit(), (
        f"{_TS_PKG} should be an exact version pin (e.g. '0.1.77'), got {ts!r}. "
        "A range pin would make the lockstep guard ambiguous."
    )


def test_pyproject_range_admits_the_locked_version() -> None:
    # The pyproject specifier must actually PERMIT the locked version (parse the
    # range, don't just check a substring) — so a future narrowing of the cap
    # (e.g. `<0.1.77`) that excludes the locked/TS pin fails this guard.
    spec = _pyproject_specifier()
    locked = _python_locked_version()
    assert spec.contains(locked, prereleases=True), (
        f"pyproject specifier {_PY_PKG}{spec} does not admit the locked version {locked} "
        f"(from {UV_LOCK}); widen the pyproject range or re-lock."
    )


def test_ts_package_lock_matches_package_json_pin() -> None:
    # The package.json pin and its lockfile must agree, else `npm ci` installs a
    # version the lockstep guard never checked.
    pinned = _ts_pinned_version()
    locked = _ts_locked_version()
    assert pinned == locked, (
        f"{_TS_PKG} drift between package.json pin ({pinned}) and package-lock.json "
        f"resolved version ({locked}); re-run `npm install` in runtime_runner/agent-runner."
    )
