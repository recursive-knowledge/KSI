"""Per-source attempt-failure diagnosis formatters.

``engine._build_approach_diagnosis`` builds a short failure-diagnosis summary
that is stored in memory instead of the raw model output (so agents learn what
to AVOID rather than copy a subtly-wrong patch). The benchmark-specific body of
that summary used to live in a ``task_source ==`` if/elif chain inside the
engine; this module holds those legs as standalone formatters and attaches them
to their ``TaskSourceSpec`` via the same post-hoc wiring pattern used for
``TaskSourceSpec.loader`` (``kcsi.benchmarks.loaders.attach_benchmark_loaders``).

Each formatter shares one contract::

    formatter(*, trace, eval_result, outcome, seed_test_files) -> list[str]

and returns the lines that sit between the generic ``Outcome:`` header and the
``Next attempt:`` footer the engine adds. Sources without a formatter
(``polyglot``, unknown) fall back to the engine's generic
eval-status default — the legs here are moved verbatim from the engine to keep
the rendered diagnosis byte-identical.
"""

from __future__ import annotations

import re
from typing import Any

from kcsi.tasks.registry import REGISTRY, TaskSourceSpec, register_task_source


def _arc(*, trace: Any, eval_result: dict[str, Any], outcome: str, seed_test_files: bool) -> list[str]:
    parts: list[str] = []
    status = eval_result.get("status") or ""
    if status:
        parts.append(f"Eval status: {status}")
    total = eval_result.get("arc_total_count")
    correct = eval_result.get("arc_correct_count")
    ratio = eval_result.get("arc_pass_ratio")
    if total is not None:
        parts.append(f"ARC score: {correct or 0}/{total} tests correct (pass_ratio={ratio})")

    trial_count = 0
    runtime_meta = trace.runtime_meta or {}
    for item in (runtime_meta.get("arc_submit_trial_results") or [])[:4]:
        if not isinstance(item, dict):
            continue
        reason = item.get("reason") or item.get("status") or "unknown"
        if reason not in {"ok", "accepted"}:
            trial_count += 1
    if trial_count:
        parts.append(f"Rejected hidden ARC trial(s): {trial_count}")
        parts.append("Next check: derive the transformation from visible train pairs before spending another trial.")
    elif outcome != "resolved" and status == "parse_error":
        # Legacy/replayed traces only: the strict trace scorer no longer
        # emits `parse_error` for ARC (statuses are ok / no_runtime_submission /
        # no_submission / missing_reference*), so this branch is dead for
        # current runs and kept only to diagnose pre-strict-scorer stored eval_results.
        err = str(eval_result.get("error") or "")[:180]
        parts.append(f"Rejected output format: {err}")
        parts.append("Next check: return exactly the required grid/list-of-grids JSON.")
    return parts


def _swebench(*, trace: Any, eval_result: dict[str, Any], outcome: str, seed_test_files: bool) -> list[str]:
    parts: list[str] = []
    instance_report = eval_result.get("instance_report", {})
    tests_status = instance_report.get("tests_status", {})

    if tests_status:
        ftp = tests_status.get("FAIL_TO_PASS", {})
        ptp = tests_status.get("PASS_TO_PASS", {})
        ftp_pass = ftp.get("success", [])
        ftp_fail = ftp.get("failure", [])
        ftp_unknown = ftp.get("unknown", [])
        ptp_fail = ptp.get("failure", [])
        ptp_unknown = ptp.get("unknown", [])

        unknown_tests = [*ftp_unknown, *ptp_unknown]
        # Upstream-strict (seed_test_files=False): never emit test names
        # — they're eval signal the agent must not see in MEMORY.md or via
        # the MCP query tool's full_memory_trace_condensed payload.
        # Counts preserve the diagnostic signal needed by the swarm without
        # exposing the test identifiers themselves. Mirrors the
        # _best_attempt_summary anonymization.
        if seed_test_files:
            if ftp_pass:
                parts.append(f"Tests now passing (good): {', '.join(ftp_pass)}")
            if ftp_fail:
                parts.append(f"Tests STILL FAILING (the patch did NOT fix these): {', '.join(ftp_fail)}")
            if ptp_fail:
                parts.append(f"Tests REGRESSED (the patch BROKE these): {', '.join(ptp_fail)}")
            if unknown_tests:
                parts.append(f"Expected tests not observed in parser output: {', '.join(unknown_tests)}")
        else:
            if ftp_pass:
                parts.append(f"Target tests now passing (good): {len(ftp_pass)}")
            if ftp_fail:
                parts.append(f"Target tests STILL FAILING: {len(ftp_fail)}")
            if ptp_fail:
                parts.append(f"Previously-passing tests REGRESSED: {len(ptp_fail)}")
            if unknown_tests:
                parts.append(f"Expected tests not observed in parser output: {len(unknown_tests)}")
        if ftp_pass and not ftp_fail and not ptp_fail and not unknown_tests:
            parts.append("All target tests pass and no regressions.")
    elif outcome != "resolved":
        status = eval_result.get("status") or eval_result.get("swebench_status") or ""
        if status:
            parts.append(f"Eval status: {status}")

    output = trace.model_output or ""
    changed_files: list[str] = []
    for m in re.finditer(r"diff --git a/\S+ b/(\S+)", output):
        f = m.group(1)
        if f not in changed_files:
            changed_files.append(f)
    if changed_files:
        parts.append(f"Files modified: {', '.join(changed_files)}")
    return parts


def _terminal_bench_2(*, trace: Any, eval_result: dict[str, Any], outcome: str, seed_test_files: bool) -> list[str]:
    parts: list[str] = []
    runtime_meta = trace.runtime_meta or {}
    parts.append(
        "TB2 verifier summary: "
        f"reward={runtime_meta.get('reward')} "
        f"agent_exit={runtime_meta.get('agent_exit_code')} "
        f"verifier_exit={runtime_meta.get('verifier_exit_code')}"
    )
    if trace.tool_trace:
        parts.append(f"Shell steps executed: {len(trace.tool_trace)}")
        recent_commands: list[str] = []
        for step in trace.tool_trace[-3:]:
            if not isinstance(step, dict):
                continue
            tool_input = step.get("tool_input") or {}
            if not isinstance(tool_input, dict):
                continue
            command = str(tool_input.get("command") or "").strip()
            if command:
                recent_commands.append(command[:180])
        if recent_commands:
            parts.append("Recent commands: " + " | ".join(recent_commands))
    return parts


_FORMATTERS = {
    "arc": _arc,
    "swebench_pro": _swebench,
    "terminal_bench_2": _terminal_bench_2,
}


def _attach_registry_formatters() -> None:
    """Attach the built-in diagnosis formatters to their registered specs.

    ``kcsi.tasks.registry`` stays free of engine/runtime imports while the
    ``approach_diagnosis`` hook is populated here, per the ``loader`` precedent.
    Idempotent: only sets the hook when it is still ``None``.
    """
    from dataclasses import replace as dataclass_replace

    for name, formatter in _FORMATTERS.items():
        spec: TaskSourceSpec | None = REGISTRY.get(name)
        if spec is not None and spec.approach_diagnosis is None:
            register_task_source(dataclass_replace(spec, approach_diagnosis=formatter), replace=True)


_attach_registry_formatters()
