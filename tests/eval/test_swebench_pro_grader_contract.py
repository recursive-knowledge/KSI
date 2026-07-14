"""Non-docker invocation-contract smoke for the external SWE-bench Pro grader.

Addresses #1138 (part 1). CI's other 11 SWE-bench Pro grading tests all
monkeypatch the grader, so nothing exercises the *real* upstream entrypoint.
This module adds a cheap, docker-free contract check that runs against the
actual checked-out evaluator (populated by the CI setup step, absent in local
dev) and pins two invariants:

  1. The repo-revision guard (`_swebench_pro_revision_error`) passes for the
     checked-out evaluator — i.e. it is at the revision our code expects.
  2. The real upstream grader's argparse accepts every flag our
     `SwebenchProEvaluator.evaluate()` passes it.

The real-docker grading run (part 2) is a deferred scheduled job, not this
PR-gating smoke.

These tests `pytest.skip` cleanly when the evaluator checkout is absent, so they
only truly run in CI where the setup step ran. The call-site coverage guard at
the bottom reads only our own source, so it runs everywhere.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pytest

import kcsi.benchmarks.swebench_pro as swebench_pro_module
from kcsi.benchmarks.swebench_pro import _swebench_pro_revision_error
from kcsi.benchmarks.swebench_pro_external import DEFAULT_EVALUATOR_RELATIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_DIR = REPO_ROOT / DEFAULT_EVALUATOR_RELATIVE
GRADER_SCRIPT = EVALUATOR_DIR / "swe_bench_pro_eval.py"

_HAS_EVALUATOR = GRADER_SCRIPT.is_file()
_SKIP_REASON = f"SWE-bench Pro evaluator checkout absent ({GRADER_SCRIPT}); CI-only grader contract smoke (#1138)"

# The full, maximally-configured argv that ``SwebenchProEvaluator.evaluate()``
# can pass to ``swe_bench_pro_eval.py``. Mirrors the ``cmd = [...]`` list in
# ``src/kcsi/benchmarks/swebench_pro.py`` plus its conditional ``--use_local_docker`` /
# ``--block_network`` / ``--docker_platform`` appends. The
# ``test_evaluate_call_site_flags_are_covered`` guard below fails if evaluate()
# ever threads a flag not represented here, so this stays honest across changes.
_EVALUATE_ARGV = [
    "--raw_sample_path",
    "sample.csv",
    "--patch_path",
    "patches.json",
    "--output_dir",
    "out",
    "--dockerhub_username",
    "jefzda",
    "--scripts_dir",
    "run_scripts",
    "--num_workers",
    "1",
    "--use_local_docker",
    "--block_network",
    "--docker_platform",
    "linux/amd64",
]


def _load_grader_parse_args():
    """Extract just the upstream ``parse_args`` function and exec it in isolation.

    The full ``swe_bench_pro_eval.py`` imports heavy runtime deps (docker/modal),
    so we can't import it in a docker-free CI job. ``parse_args`` only depends on
    ``argparse``, so extracting and exec-ing its source exercises the real
    upstream argparse without pulling in the rest of the module.
    """
    source = GRADER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args":
            func_src = ast.get_source_segment(source, node)
            assert func_src is not None
            namespace: dict[str, object] = {"argparse": argparse}
            exec(func_src, namespace)  # noqa: S102 - trusted, pinned upstream source
            return namespace["parse_args"]
    raise AssertionError(f"parse_args() not found in upstream grader {GRADER_SCRIPT}")


@pytest.mark.skipif(not _HAS_EVALUATOR, reason=_SKIP_REASON)
def test_revision_guard_passes_for_checked_out_evaluator() -> None:
    """The checked-out evaluator must satisfy our repo-revision guard."""
    assert _swebench_pro_revision_error(EVALUATOR_DIR) is None


@pytest.mark.skipif(not _HAS_EVALUATOR, reason=_SKIP_REASON)
def test_grader_argparse_accepts_every_evaluate_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real upstream grader argparse must accept every flag evaluate() sends."""
    parse_args = _load_grader_parse_args()
    monkeypatch.setattr(sys, "argv", ["swe_bench_pro_eval.py", *_EVALUATE_ARGV])
    ns = parse_args()

    assert ns.raw_sample_path == "sample.csv"
    assert ns.patch_path == "patches.json"
    assert ns.output_dir == "out"
    assert ns.dockerhub_username == "jefzda"
    assert ns.scripts_dir == "run_scripts"
    assert ns.num_workers == 1
    assert ns.use_local_docker is True
    assert ns.block_network is True
    assert ns.docker_platform == "linux/amd64"


def test_evaluate_call_site_flags_are_covered() -> None:
    """Drift guard: every ``--flag`` literal in evaluate()'s subprocess cmd is
    represented in ``_EVALUATE_ARGV``.

    Runs everywhere (reads only our own source) so a new grader flag added to
    evaluate() can't silently outrun the CI contract test above.
    """
    source = Path(swebench_pro_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    evaluate = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
    call_site_flags = {
        node.value
        for node in ast.walk(evaluate)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("--")
    }
    covered = {arg for arg in _EVALUATE_ARGV if arg.startswith("--")}
    missing = call_site_flags - covered
    assert not missing, (
        f"evaluate() passes grader flags not covered by _EVALUATE_ARGV / the contract test: {sorted(missing)}"
    )


def test_ci_evaluator_cache_key_tracks_patch_inputs() -> None:
    """A changed local evaluator patch must invalidate the CI evaluator cache."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "benchmarks/swebench_pro/evaluator_patches/*.patch" in workflow
    assert "benchmarks/scripts/dataprep/setup_swebench_pro_evaluator.py" in workflow
