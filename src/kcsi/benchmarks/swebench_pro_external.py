"""Shared constants for the external SWE-bench Pro evaluator checkout."""

from __future__ import annotations

from pathlib import Path

EVALUATOR_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
EVALUATOR_REVISION = "0c64e26f00b9c190432de7fc520c8ceed5c25518"
DATASET_NAME = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
REVISION_MARKER = ".kcsi-evaluator-revision"
PATCH_STATE_MARKER = ".kcsi-evaluator-patches.json"
SETUP_COMMAND = "uv run python benchmarks/scripts/dataprep/setup_swebench_pro_evaluator.py"

DEFAULT_EVALUATOR_RELATIVE = Path("benchmarks") / "swebench_pro" / "evaluator"
DEFAULT_DATASET_RELATIVE = Path("benchmarks") / "swebench_pro" / "dataset" / "test.jsonl"
EVALUATOR_PATCHES_RELATIVE = Path("benchmarks") / "swebench_pro" / "evaluator_patches"

# Producer/consumer contract for the evaluator's ``swebench_status`` field.
#
# ``src/kcsi/eval/swebench_pro.py`` (producer) tags every eval-result dict with a
# ``swebench_status`` string. ``score_swebench_from_eval`` in
# ``src/kcsi/orchestrator/scoring.py`` (consumer) maps any FAILURE status to
# ``None`` (unscored) — deliberately NOT a genuine ``0.0`` — so the engine skips
# the attempt instead of counting an infra failure as a solved-zero. Centralizing
# the literals here keeps that contract explicit instead of relying on two copies
# of the same string tuple staying in sync.
#
# NOTE: this vocabulary uses ``harness_timeout`` — distinct from the generic
# eval-level ``status`` field consumed elsewhere in scoring.py, which uses the
# bare ``timeout``. The two vocabularies are intentionally NOT shared.
SWEBENCH_STATUS_OK = "ok"
SWEBENCH_FAILURE_STATUSES = (
    "harness_timeout",
    "harness_failed",
    "no_patch",
    # capture_failed: the agent made edits but the workspace-diff capture came
    # back empty (e.g. broken submodule gitlink) and no patch was recoverable.
    # Unscored (None) like no_patch — an infra failure, not a real 0.0.
    "capture_failed",
    "missing_report",
    "oom_killed",
)
