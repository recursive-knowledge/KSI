"""Regression harness for the Teleport SWE-bench Pro instance.

Replays two real patches through the upstream SWE-bench Pro evaluator and asserts
the known resolved/unresolved outcomes have not silently flipped:

  * ``fixtures/swebench_pro/teleport/swarm_patch.diff`` (expected **unresolved**)
    KSI's 2-gen x 3-task smoke produced this patch. It flips the target
    FAIL_TO_PASS ``TestMux/SSHProxyHelloSignature`` to PASSED but regresses the
    PASS_TO_PASS pair ``TestMux/DisableTLS`` and ``TestMux/TLSSSH``, so the
    instance ends up unresolved.

  * ``fixtures/swebench_pro/teleport/dgm_patch.diff`` (expected **resolved**)
    DGM's gen-1 patch for the same instance cleanly resolves it.

Both patches target instance
``instance_gravitational__teleport-af5e2517de7d18406b614e413aca61c319312171-vee9b09fb20c43af7e520f57e9239bbcf46b7113d``
(issue #461 handoff goal 5).

The test is marked ``@pytest.mark.slow`` so it is excluded from the default
``uv run pytest`` run. It is also skipped with a clear reason when any of
the prerequisites is missing:

  * Docker daemon unreachable
  * Pinned SWE-bench Pro evaluator repo not installed under
    ``benchmarks/swebench_pro/evaluator``
  * Raw sample dataset (``benchmarks/swebench_pro/dataset/test.jsonl``) not present

To run explicitly with Docker and the evaluator available::

    uv run pytest -m slow tests/eval/test_swebench_pro_teleport_regression.py -xvs

If either assertion fails, investigate in this order:
  1. The evaluator revision under ``benchmarks/swebench_pro/evaluator`` — a
     silent revision drift is the most common cause.
  2. The ``jefzda/swebench_pro_<instance>`` Docker image for this instance
     (regeneration can change test behaviour).
  3. An intentional fixture refresh — update the .diff files and bump
     ``fixtures/swebench_pro/teleport/meta.json``.
  4. A real agent-quality change — if Swarm's patch now resolves the instance,
     that is a positive signal worth investigating; do not simply flip the
     assertion without a linked experiment.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import FIXTURES_DIR, REPO_ROOT

from ksi.benchmarks.swebench_pro import SwebenchProEvaluator, _default_swebench_pro_repo_root
from ksi.benchmarks.swebench_pro_external import EVALUATOR_REVISION, SETUP_COMMAND, SWEBENCH_FAILURE_STATUSES
from ksi.models import TaskSpec

TELEPORT_INSTANCE_ID = (
    "instance_gravitational__teleport-"
    "af5e2517de7d18406b614e413aca61c319312171-"
    "vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
)

FIXTURE_DIR = FIXTURES_DIR / "swebench_pro" / "teleport"
KSI_PATCH_PATH = FIXTURE_DIR / "swarm_patch.diff"
DGM_PATCH_PATH = FIXTURE_DIR / "dgm_patch.diff"
META_PATH = FIXTURE_DIR / "meta.json"

# 30-minute ceiling per patch — Teleport builds are slow even when fully cached.
PER_PATCH_TIMEOUT_SEC = 30 * 60


def _docker_available() -> bool:
    """Return True iff ``docker info`` succeeds in a short window."""
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _evaluator_installed() -> bool:
    """Return True iff the pinned SWE-bench Pro evaluator is present."""
    repo_root = _default_swebench_pro_repo_root()
    return (repo_root / "swe_bench_pro_eval.py").is_file() and (repo_root / "run_scripts").is_dir()


def _raw_sample_path() -> Path:
    """Return the canonical raw sample JSONL shipped with the repo."""
    return REPO_ROOT / "benchmarks" / "swebench_pro" / "dataset" / "test.jsonl"


def _load_raw_sample_row() -> dict[str, Any] | None:
    raw_path = _raw_sample_path()
    if not raw_path.is_file():
        return None
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("instance_id") or "") == TELEPORT_INSTANCE_ID:
                return row
    return None


def test_teleport_regression_fixtures_are_present() -> None:
    """Fixtures must always ship — no Docker required."""
    assert KSI_PATCH_PATH.is_file(), KSI_PATCH_PATH
    assert DGM_PATCH_PATH.is_file(), DGM_PATCH_PATH
    assert META_PATH.is_file(), META_PATH

    swarm_patch = KSI_PATCH_PATH.read_text(encoding="utf-8")
    dgm_patch = DGM_PATCH_PATH.read_text(encoding="utf-8")
    assert swarm_patch.lstrip().startswith("diff --git "), "swarm_patch.diff must begin with 'diff --git '"
    assert dgm_patch.lstrip().startswith("diff --git "), "dgm_patch.diff must begin with 'diff --git '"

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert meta["instance_id"] == TELEPORT_INSTANCE_ID
    assert meta["fixtures"]["swarm_patch.diff"]["expected_resolved"] is False
    assert meta["fixtures"]["dgm_patch.diff"]["expected_resolved"] is True


@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.parametrize(
    "patch_path, expected_resolved, label",
    [
        (KSI_PATCH_PATH, False, "swarm"),
        (DGM_PATCH_PATH, True, "dgm"),
    ],
    ids=["ksi-unresolved", "dgm-resolved"],
)
def test_teleport_patch_resolution_matches_fixture(
    patch_path: Path,
    expected_resolved: bool,
    label: str,
) -> None:
    """Replay each fixture patch through the evaluator and assert the known outcome."""
    if not patch_path.is_file():
        pytest.skip(f"missing fixture patch: {patch_path}")
    if not _docker_available():
        pytest.skip("Docker daemon not reachable; skipping SWE-bench Pro Teleport replay")
    if not _evaluator_installed():
        pytest.skip(
            f"SWE-bench Pro evaluator not installed under benchmarks/swebench_pro/evaluator; run: {SETUP_COMMAND}"
        )
    raw_row = _load_raw_sample_row()
    if raw_row is None:
        pytest.skip(
            "Teleport instance row missing from benchmarks/swebench_pro/dataset/test.jsonl; "
            "run: uv run python benchmarks/scripts/dataprep/prepare_swebench_pro_repo_cache.py"
        )

    task = TaskSpec(
        id=TELEPORT_INSTANCE_ID,
        repo=str(raw_row.get("repo") or "gravitational/teleport"),
        prompt="teleport-regression-harness",
        metadata={
            "task_source": "swebench_pro",
            "fail_to_pass": raw_row.get("fail_to_pass"),
            "pass_to_pass": raw_row.get("pass_to_pass"),
            "base_commit": raw_row.get("base_commit"),
        },
    )

    evaluator = SwebenchProEvaluator(
        raw_sample_path=str(_raw_sample_path()),
        timeout_sec=PER_PATCH_TIMEOUT_SEC,
        use_local_docker=True,
    )

    patch_text = patch_path.read_text(encoding="utf-8")
    result = evaluator.evaluate(task=task, model_output=patch_text)

    status = result.get("swebench_status")
    if status in SWEBENCH_FAILURE_STATUSES:
        pytest.skip(
            f"SWE-bench Pro evaluator returned non-ok status {status!r} for {label}; "
            f"evaluator is likely unreachable or image missing. Full result: {result}"
        )

    assert status == "ok", f"evaluator returned unexpected status: {result}"
    observed_resolved = bool(result.get("resolved"))

    # The load-bearing assertion: known outcomes must not flip silently.
    assert observed_resolved == expected_resolved, (
        f"Teleport regression: {label} patch expected resolved={expected_resolved} "
        f"but got {observed_resolved}. "
        f"Revision pinned at {EVALUATOR_REVISION}. "
        f"Investigate in order: evaluator repo revision, jefzda/swebench_pro_* image drift, "
        f"intentional fixture refresh (update meta.json), or a real agent-quality change. "
        f"Full result summary: status={status} "
        f"tests_status={(result.get('instance_report') or {}).get('tests_status')}"
    )
