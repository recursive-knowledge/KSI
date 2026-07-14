"""Tests for TSC compile-error surfacing from the container entrypoint.

Background
----------
When the container entrypoint recompiles the agent-runner (``npx tsc
--outDir /tmp/dist``) because the mounted-source hash differs from the
baked-dist hash, ``tsc`` failures used to exit 2 with the actual compile
diagnostic completely swallowed by shell error-propagation. The host then
raised ``RuntimeError("Shared container runner failed (exit=2): Container
exited with code 2: ")`` with 30 chars of empty suffix — undiagnosable.

The observability patch adds a framed diagnostic block to the entrypoint's
stderr stream before inheriting the exit code:

    ====TSC_COMPILE_FAILED====
    <first 40 lines of captured tsc stdout+stderr>
    ====END_TSC_COMPILE_FAILED====

These tests exercise:

1. The host-side helper that recognises the framed block and extracts a
   readable snippet into the raised ``RuntimeError`` message (which the
   engine writes as ``trace.error`` / ``error_text``).
2. Backward-compat — stderr without the header is passed through unchanged.
3. Truncated / malformed frames don't crash the host.
4. A shell-level integration test for the entrypoint snippet itself that
   stubs ``tsc`` with a failing script and verifies the framed diagnostic
   appears on stderr and the entrypoint exits 2.
"""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest
from conftest import REPO_ROOT

from kcsi.runtime.normalize import extract_tsc_compile_error

FAKE_TSC_ERROR = (
    "src/index.ts(42,15): error TS2322: Type 'string' is not assignable to type 'number'.\n"
    "src/helpers.ts(7,3): error TS2304: Cannot find name 'foo'.\n"
    "Found 2 errors in 2 files.\n"
)


# ── Host-side extraction helper ─────────────────────────────────────────────


def test_container_host_surfaces_tsc_compile_failed_header():
    """Host helper pulls the TS error body out of a framed stderr block."""
    stderr = (
        "some earlier log line\n"
        "====TSC_COMPILE_FAILED====\n"
        f"{FAKE_TSC_ERROR}"
        "====END_TSC_COMPILE_FAILED====\n"
        "Container exited with code 2: \n"
    )
    excerpt = extract_tsc_compile_error(stderr)
    assert excerpt is not None
    assert "error TS2322" in excerpt
    assert "Type 'string' is not assignable to type 'number'" in excerpt
    # The frame markers themselves shouldn't be in the excerpt body.
    assert "====TSC_COMPILE_FAILED====" not in excerpt
    assert "====END_TSC_COMPILE_FAILED====" not in excerpt


def test_container_host_ignores_absence_of_header():
    """Without the header, helper returns None — callers keep the old value."""
    stderr = "npm ERR! something unrelated\nContainer exited with code 1: some other failure\n"
    assert extract_tsc_compile_error(stderr) is None


def test_container_host_ignores_empty_stderr():
    assert extract_tsc_compile_error("") is None
    assert extract_tsc_compile_error(None) is None  # type: ignore[arg-type]


def test_container_host_handles_truncated_diagnostic_block():
    """Header present, footer missing — best-effort excerpt, no crash."""
    stderr = (
        f"====TSC_COMPILE_FAILED====\n{FAKE_TSC_ERROR}"
        # No footer — simulates stream truncation (e.g., stderr buffer limit).
    )
    excerpt = extract_tsc_compile_error(stderr)
    assert excerpt is not None
    # Best-effort: still captures the visible content.
    assert "error TS2322" in excerpt


def test_container_host_handles_header_with_no_body():
    """Empty diagnostic block shouldn't crash."""
    stderr = "====TSC_COMPILE_FAILED====\n====END_TSC_COMPILE_FAILED====\n"
    excerpt = extract_tsc_compile_error(stderr)
    # Either None or empty string is acceptable; the key is no crash.
    assert excerpt is None or excerpt == ""


# ── Entrypoint-script integration test ──────────────────────────────────────


def _locate_entrypoint() -> Path:
    return REPO_ROOT / "container" / "entrypoint.sh"


def test_entrypoint_file_exists_and_is_executable():
    entrypoint = _locate_entrypoint()
    assert entrypoint.exists(), (
        f"container/entrypoint.sh is missing at {entrypoint}; both Dockerfiles should COPY this script."
    )
    st = entrypoint.stat()
    assert st.st_mode & stat.S_IXUSR, "entrypoint.sh must be executable"


def test_entrypoint_dockerfile_references_are_consistent():
    """Both Dockerfiles must COPY the external entrypoint.sh (not inline it)."""
    root = REPO_ROOT
    for df in ("container/Dockerfile", "container/Dockerfile.bench"):
        text = (root / df).read_text(encoding="utf-8")
        assert "entrypoint.sh" in text, f"{df} must reference entrypoint.sh"
        # Guard: the inline printf'ing entrypoint in the old Dockerfiles is gone.
        assert "printf '#!/bin/bash" not in text, (
            f"{df} still bakes an inline entrypoint; switch to COPY container/entrypoint.sh ..."
        )


def test_entrypoint_prints_tsc_diagnostic_on_compile_failure(tmp_path):
    """Shell-level integration: stub tsc to fail, assert framing + exit code.

    This spawns the real ``container/entrypoint.sh`` in a sandbox that
    mimics the /app layout (``src/`` with a .ts file whose hash won't match
    the cached ``/tmp/dist/.src_hash``) and shims ``tsc`` with a failing
    script on ``$PATH``. We drive only the recompile branch — the
    entrypoint's ``node /tmp/dist/index.js`` step is replaced with a stub
    that can't run (node may not exist) but we expect the script to exit
    during tsc failure, *before* reaching node.
    """
    entrypoint = _locate_entrypoint()
    if not entrypoint.exists():  # guarded by the existence test above
        pytest.skip("entrypoint.sh not present")

    # Build an isolated /app stand-in.
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    src_dir = app_dir / "src"
    src_dir.mkdir()
    (src_dir / "foo.ts").write_text("export const x: number = 1;\n", encoding="utf-8")

    # Stub `tsc` (invoked via `npx tsc`): an `npx` shim that exits 2 with
    # the fake compile error on stderr. `npx tsc ...` in the entrypoint
    # goes through $PATH, so a shim named `npx` first on PATH wins.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npx_shim = bin_dir / "npx"
    npx_shim.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            # Ignore all args; simulate tsc compile failure.
            echo {FAKE_TSC_ERROR!r} >&2
            exit 2
            """
        ),
        encoding="utf-8",
    )
    os.chmod(npx_shim, 0o755)

    # Also stub the md5sum/find helpers to produce a deterministic hash that
    # won't match (force the recompile branch).
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        # Let the entrypoint cd into our fake runner root.
        "KCSI_RUNNER_ROOT": str(app_dir),
    }

    # We pipe an empty stdin so `cat > /tmp/input.json` (the last step of the
    # entrypoint) doesn't block. We expect the script to exit 2 BEFORE reaching
    # that step because tsc fails.
    script = entrypoint.read_text(encoding="utf-8")
    wrapper = tmp_path / "run.sh"
    modified = script
    # Redirect the dist cache into tmp_path so the test is hermetic.
    modified = modified.replace("/tmp/dist", str(tmp_path / "dist"))
    modified = modified.replace("/tmp/input.json", str(tmp_path / "input.json"))
    modified = modified.replace("/tmp/tsc-compile.log", str(tmp_path / "tsc-compile.log"))
    wrapper.write_text(modified, encoding="utf-8")
    os.chmod(wrapper, 0o755)

    result = subprocess.run(
        ["bash", str(wrapper)],
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2, (
        f"expected exit 2, got {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "====TSC_COMPILE_FAILED====" in result.stderr, "entrypoint should emit a TSC_COMPILE_FAILED header on stderr"
    assert "====END_TSC_COMPILE_FAILED====" in result.stderr, "entrypoint should emit a matching END footer"
    assert "error TS2322" in result.stderr, "entrypoint should include captured tsc error body in the framing"
