"""Single source of truth for the generic eval-results -> score precedence.

Previously this precedence chain was duplicated between ``_score_from_eval``
(``src/ksi/orchestrator/engine.py``, generic fallback leg) and ``_extract_score``
(``src/ksi/memory/store.py``). Both now delegate here.

The native_score step uses the STRICT ``isinstance(ns, (int, float))`` guard:
a non-numeric, non-None native_score falls through to the later precedence
steps rather than being passed to ``float()`` (which would have crashed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..benchmarks.swebench_pro_external import SWEBENCH_FAILURE_STATUSES
from ..benchmarks.terminal_bench_2 import TB2_VERIFIER_UNSCORED_STATUSES

if TYPE_CHECKING:
    from ..models import TaskSpec


def score_from_eval_results(eval_r: dict[str, Any]) -> float | None:
    """Extract a numeric score from a generic eval-results dict (higher is better).

    Precedence:
    1. ``native_score`` (numeric only)
    2. ``resolved`` (bool -> 1.0 / 0.0)
    3. ``instance_report.resolved`` (bool -> 1.0 / 0.0)
    4. ``pass`` (bool -> 1.0 / 0.0)

    Returns ``None`` when none of the above are present.
    """
    ns = eval_r.get("native_score")
    if isinstance(ns, (int, float)):
        return float(ns)
    # Only a real ``bool`` is an authoritative verdict: a non-bool ``resolved``
    # (e.g. the string ``"false"``, which is truthy) would otherwise be misread,
    # so fall through to the later precedence steps instead of trusting it.
    if isinstance(eval_r.get("resolved"), bool):
        return 1.0 if eval_r["resolved"] else 0.0
    instance_report = eval_r.get("instance_report")
    if isinstance(instance_report, dict):
        if isinstance(instance_report.get("resolved"), bool):
            return 1.0 if instance_report["resolved"] else 0.0
    if "pass" in eval_r and eval_r["pass"] is not None:
        return 1.0 if bool(eval_r["pass"]) else 0.0
    return None


def score_swebench_from_eval(eval_result: dict[str, Any], *, task: "TaskSpec | None" = None) -> float | None:
    """SWE-bench scorer: defer to the harness's authoritative ``resolved``
    verdict.

    The upstream harness resolves an instance via an exact-string subset check
    ``(FAIL_TO_PASS | PASS_TO_PASS) <= passed_tests`` and emits the result as a
    per-instance ``resolved`` boolean — and that boolean IS the published
    SWE-bench metric. We score from it directly rather than re-deriving an
    independent per-test tally: re-derivation re-runs the SAME rule over a
    separately-read ``output.json`` and only adds a divergence surface — a
    stale/empty/name-skewed read marks expected tests ``unknown`` and would
    force a false 0 even though the harness resolved the instance (the
    resolved-but-unknown bias). The ``tests_status`` tally is therefore
    DIAGNOSTIC only; a dedicated resolved-but-divergent-tally counter is tracked
    separately rather than here.

    Precedence:
    1. Known harness/infra failure statuses -> None (no trustworthy verdict,
       unscored — matches ``score_tb2_from_eval``'s infra-failure gating).
    2. ``instance_report.resolved`` -> 1.0 / 0.0 (authoritative verdict).
    3. Eval-level failure statuses (``status`` / ``swebench_status``) -> None.
    4. ``run_summary`` resolved_ids / unresolved_ids -> 1.0 / 0.0.
    5. Fallback (no verdict available): all-clear ``tests_status`` tally -> 1.0,
       else 0.0. Unreachable for real evaluator output, which always carries
       ``resolved`` (step 2); retained for malformed/verdict-less reports.

    NOTE (re-baseline): unlike the prior binary scorer, the verdict is now
    authoritative in BOTH directions, so historical SWE-bench numbers can move
    up OR down (announced, never silent):
    - UP: an instance the harness marked ``resolved`` scores 1.0 even when the
      local tally shows unknown/failed/skipped expected tests — moving
      wrongly-zeroed instances up (the resolved-but-unknown case;
      ``test_resolved_true_with_unknown_tally_scores_one``). This also covers a
      genuine ``failure`` tally alongside ``resolved: True`` — a read divergence
      by the harness's subset rule that the verdict overrides to 1.0
      (``test_resolved_true_with_failure_tally_scores_one``).
    - DOWN: a ``resolved: False`` verdict scores 0.0 even when the local tally
      looks all-clear, overriding what the old all-or-nothing tally would have
      scored 1.0 (``test_resolved_false_scores_zero_regardless_of_tally``).
    """
    instance_report = eval_result.get("instance_report")
    if isinstance(instance_report, dict):
        # Check for known failure statuses first. These are infra/harness
        # failures with no trustworthy verdict -> None (unscored), not a
        # fabricated 0.0 that would feed _best_scores/distillation/forum as a
        # genuine agent failure (mirrors score_tb2_from_eval).
        status = instance_report.get("status")
        if status in set(SWEBENCH_FAILURE_STATUSES) | {"timeout"}:
            return None

        # Authoritative verdict, ahead of the local tally: the harness already
        # computed (F2P | P2P) <= passed for this instance, so trust it over any
        # re-derivation — which can only diverge downward via a misread. Require a
        # real ``bool``: a stray non-bool ``resolved`` (e.g. the truthy string
        # ``"false"``) falls through to the tally rather than scoring a false 1.0.
        if isinstance(instance_report.get("resolved"), bool):
            return 1.0 if instance_report["resolved"] else 0.0

        # Fallback tally (verdict-absent reports only): all-or-nothing pass.
        tests = instance_report.get("tests_status", {})
        if isinstance(tests, dict):
            f2p = tests.get("FAIL_TO_PASS", {}) if isinstance(tests.get("FAIL_TO_PASS"), dict) else {}
            p2p = tests.get("PASS_TO_PASS", {}) if isinstance(tests.get("PASS_TO_PASS"), dict) else {}
            f2p_failure = f2p.get("failure", []) if isinstance(f2p.get("failure"), list) else []
            p2p_failure = p2p.get("failure", []) if isinstance(p2p.get("failure"), list) else []
            f2p_skipped = f2p.get("skipped", []) if isinstance(f2p.get("skipped"), list) else []
            p2p_skipped = p2p.get("skipped", []) if isinstance(p2p.get("skipped"), list) else []
            f2p_unknown = f2p.get("unknown", []) if isinstance(f2p.get("unknown"), list) else []
            p2p_unknown = p2p.get("unknown", []) if isinstance(p2p.get("unknown"), list) else []
            f2p_success = f2p.get("success", []) if isinstance(f2p.get("success"), list) else []
            p2p_success = p2p.get("success", []) if isinstance(p2p.get("success"), list) else []
            total = (
                len(f2p_success)
                + len(f2p_failure)
                + len(f2p_skipped)
                + len(f2p_unknown)
                + len(p2p_success)
                + len(p2p_failure)
                + len(p2p_skipped)
                + len(p2p_unknown)
            )
            if total > 0:
                # Binary: resolved only when ALL tests pass
                all_clear = (
                    len(f2p_failure) == 0
                    and len(p2p_failure) == 0
                    and len(f2p_skipped) == 0
                    and len(p2p_skipped) == 0
                    and len(f2p_unknown) == 0
                    and len(p2p_unknown) == 0
                )
                return 1.0 if all_clear else 0.0

    # Check for eval-level failure statuses. The swebench_pro evaluator
    # emits its failure statuses under ``swebench_status`` (never a
    # top-level ``status``). Same infra-failure -> None rule as the
    # instance_report gate above.
    swebench_status = eval_result.get("swebench_status")
    if swebench_status in SWEBENCH_FAILURE_STATUSES:
        return None

    run_summary = eval_result.get("run_summary")
    if isinstance(run_summary, dict) and task is not None:
        task_id = task.id
        resolved_ids = set(run_summary.get("resolved_ids", []) or [])
        unresolved_ids = set(run_summary.get("unresolved_ids", []) or [])
        if task_id in resolved_ids:
            return 1.0
        if task_id in unresolved_ids:
            return 0.0
    return 0.0


def score_tb2_from_eval(eval_result: dict[str, Any], *, task: "TaskSpec | None" = None) -> float | None:
    """Terminal-Bench-2 scorer: distinguish *unscored* from a genuine ``0.0``.

    A genuine failure (verifier ran, reward 0) carries ``native_score=0.0``;
    the verifier-never-ran cases -- a crash / OOM / exec timeout before the
    verifier stage (``TB2_VERIFIER_MISSING_STATUS``), or strict mode refusing an
    untrusted-toolchain fallback (``TB2_VERIFIER_FAIL_CLOSED_STATUS``) --
    instead carry ``native_score=None`` and ``resolved=False``. This scorer must
    still gate on those statuses: the generic ``native_score`` precedence would
    skip the absent score and read ``resolved=False`` as a real ``0.0`` failure,
    feeding it into ``_best_scores`` / ``record_attempt`` / distillation and
    contaminating the multi-generation learning signal as if the agent had been
    tried and failed.

    Return ``None`` (unscored) for those statuses so the engine skips the
    ``_best_scores`` update and preserves the prior best; defer to the generic
    precedence otherwise (a real reward, including a genuine ``0.0``). This
    mirrors ``score_swebench_from_eval``'s infra-failure gating.
    """
    status = str(eval_result.get("status") or "").strip()
    if status in TB2_VERIFIER_UNSCORED_STATUSES:
        return None
    return score_from_eval_results(eval_result)


def score_polyglot_from_eval(eval_result: dict[str, Any], *, task: "TaskSpec | None" = None) -> float | None:
    """Polyglot scorer: distinguish infra failure / no-submission from a
    genuine agent failure.

    ``status == "timeout"`` means the Docker test-run subprocess was killed
    before producing a trustworthy pass/fail. ``status == "no_solution"``
    means the agent produced no extractable solution at all -- nothing was
    graded, mirroring ``swebench_pro``'s ``no_patch`` status (both represent
    "the agent ran and produced nothing scorable," not a trustworthy 0.0
    failure verdict). ``status == "setup_failed"`` means the pre-test setup
    step (e.g. ``npm install``) itself exited nonzero before the test ever
    ran -- a nonzero setup exit is chained via ``&&`` ahead of the test
    command, so it would otherwise be indistinguishable from a genuine test
    failure. All three score ``None`` (unscored) so the engine
    preserves the prior best instead of recording a fabricated ``0.0`` into
    ``_best_scores``/distillation/forum. Every other status (``"ok"``,
    ``"skip_docker"``) is a genuine, trustworthy verdict and defers to the
    generic precedence.

    ARC's analogous true no-verdict case is ``"no_runtime_submission"``; an
    executed ARC run that simply never submits is a real failed attempt and keeps
    the evaluator's numeric ``0.0``.
    """
    if str(eval_result.get("status") or "").strip() in ("timeout", "no_solution", "setup_failed"):
        return None
    return score_from_eval_results(eval_result)


_ARC_UNSCORED_STATUSES = (
    "no_runtime_submission",
    "missing_reference",
    "missing_reference_output",
    "invalid_reference_output",
)


def score_arc_from_eval(eval_result: dict[str, Any], *, task: "TaskSpec | None" = None) -> float | None:
    """ARC scorer: distinguish a true infra failure / broken reference data /
    no-runtime-submission run from a genuine (possibly zero) trial-based score.

    ``arc_session.py`` always emits a numeric ``native_score``, but two
    categories carry no trustworthy verdict: ``"no_runtime_submission"`` (no
    tool_trace captured at all -- the runtime never ran), and the
    missing/invalid-reference-data statuses (the task's own reference grids are
    absent or malformed -- a dataset bug, not an agent failure, and identical on
    every attempt at this task). Score ``None`` (unscored) for those so the
    engine preserves the prior best instead of recording a fabricated ``0.0``.
    ``"no_submission"`` means the agent ran but never submitted an accepted ARC
    trial; that is a real failed attempt and keeps the evaluator's numeric
    ``0.0``.
    """
    if str(eval_result.get("status") or "").strip() in _ARC_UNSCORED_STATUSES:
        return None
    return score_from_eval_results(eval_result)


def _attach_registry_scorers() -> None:
    """Attach the built-in per-source scorers to their registered specs.

    Mirrors ``ksi.benchmarks.loaders.attach_benchmark_loaders`` / approach_diagnosis wiring:
    keeps ``ksi.tasks.registry`` import-light while populating the
    ``score_from_eval`` hook here. Idempotent.
    """
    from dataclasses import replace as dataclass_replace

    from ..tasks.registry import REGISTRY, register_task_source

    spec = REGISTRY.get("swebench_pro")
    if spec is not None and spec.score_from_eval is None:
        register_task_source(dataclass_replace(spec, score_from_eval=score_swebench_from_eval), replace=True)

    tb2_spec = REGISTRY.get("terminal_bench_2")
    if tb2_spec is not None and tb2_spec.score_from_eval is None:
        register_task_source(dataclass_replace(tb2_spec, score_from_eval=score_tb2_from_eval), replace=True)

    polyglot_spec = REGISTRY.get("polyglot")
    if polyglot_spec is not None and polyglot_spec.score_from_eval is None:
        register_task_source(dataclass_replace(polyglot_spec, score_from_eval=score_polyglot_from_eval), replace=True)

    arc_spec = REGISTRY.get("arc")
    if arc_spec is not None and arc_spec.score_from_eval is None:
        register_task_source(dataclass_replace(arc_spec, score_from_eval=score_arc_from_eval), replace=True)


_attach_registry_scorers()
