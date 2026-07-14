"""Single source of truth for the default information-parity field policy.

Three default upstream-strict agent-facing
surfaces (MCP redactor, distillation prompts, condensed trace) all derive their
"held-out vs declared feedback" decision from the constants here so the policy
can't drift across copies. This module does not ban self-improvement feedback; it
keeps hidden grader material out of adaptive surfaces that are reported as
upstream-strict.
"""

from __future__ import annotations

import re
from typing import Any

# Hidden test-RUNNER output tails (eval_results) for upstream-strict no-feedback benchmarks.
HIDDEN_TEST_RUNNER_TAIL_KEYS: tuple[str, ...] = (
    "test_stdout_tail",
    "test_stderr_tail",  # polyglot
    "swebench_stdout_tail",
    "swebench_stderr_tail",  # SWE-bench Pro
)

# terminal_bench_2 hidden-pytest verifier output carried in attempt_meta.
HIDDEN_ATTEMPT_META_KEYS: tuple[str, ...] = (
    "verifier_stdout_tail",
    "verifier_stderr_tail",
    "verifier_clues",
    "failure_signature",
)

# Allow-list for ARC per-test entries: a no-feedback ARC solver may legitimately
# carry forward only the per-test index + correctness flag. Everything else
# (gold `detail`, any future nested answer field) fails closed.
ARC_PER_TEST_SAFE_KEYS: frozenset[str] = frozenset({"test_index", "correct"})

# Exact top-level eval_results keys that are answer/test-contract payloads when
# present. ARC per-test details are also structurally projected below.
HIDDEN_EVAL_ANSWER_KEYS: tuple[str, ...] = (
    "detail",
    "expected",
    "expected_shape",
)

_HIDDEN_TEXT_MARKERS: tuple[str, ...] = HIDDEN_TEST_RUNNER_TAIL_KEYS + HIDDEN_ATTEMPT_META_KEYS
_SAFE_TEXT_RESUME_KEYS: tuple[str, ...] = (
    "reward",
    "native_score",
    "agent_exit",
    "agent_exit_code",
    "verifier_exit",
    "verifier_exit_code",
    "tool_count",
    "recent_commands",
    "verified_outcome",
    "last_state_change",
    "proposed change",
    "predicted outcome",
    "insight",
    "next",
    "safe",
)


def _as_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if value is None:
        return 0
    return 1


def _redact_tests_status_names(tests_status: Any) -> dict[str, Any]:
    if not isinstance(tests_status, dict):
        return {}
    out: dict[str, Any] = {}
    if "observed_count" in tests_status:
        out["observed_count"] = tests_status.get("observed_count")
    for suite in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        bucket = tests_status.get(suite)
        if not isinstance(bucket, dict):
            continue
        suite_counts: dict[str, Any] = {}
        for key in ("success", "failure", "skipped", "unknown"):
            count_key = f"{key}_count"
            if count_key in bucket:
                suite_counts[count_key] = bucket.get(count_key)
            elif key in bucket:
                suite_counts[count_key] = _as_count(bucket.get(key))
        out[suite] = suite_counts
    return out


def _redact_instance_report(report: dict[str, Any]) -> None:
    tests = report.pop("tests", None)
    if isinstance(tests, list):
        report["tests_count"] = len(tests)
    tests_status = report.get("tests_status")
    if isinstance(tests_status, dict):
        report["tests_status"] = _redact_tests_status_names(tests_status)


def _redact_hidden_text_values(value: Any) -> Any:
    if isinstance(value, str):
        return redact_solver_hidden_text(value)
    if isinstance(value, list):
        return [_redact_hidden_text_values(item) for item in value]
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _redact_hidden_text_values(item)
        return value
    return value


def redact_solver_hidden_text(value: Any) -> str:
    """Strip stale hidden-output key/value fragments from derived text fields.

    New attempts avoid baking these fields into ``trace_condensed`` at the
    source, but older DB rows may already contain fragments such as
    ``failure_signature=...``. Those derived strings are rendered to MEMORY.md
    and distillation prompts, so sanitize them at the read boundary too.
    """
    text = str(value or "")
    if not any(marker in text for marker in _HIDDEN_TEXT_MARKERS):
        return text
    cleaned = text
    for marker in _HIDDEN_TEXT_MARKERS:
        marker_re = re.escape(marker)
        # Hidden verifier/test output often contains semicolons itself. Strip
        # from the hidden key through continuation fragments until the next
        # known-safe summary field, so a value like
        # ``verifier_stdout_tail=expected x; got y; tool_count=3`` does not
        # leave ``got y`` behind.
        safe_keys = "|".join(re.escape(key) for key in _SAFE_TEXT_RESUME_KEYS)
        next_field = rf"(?=(?:[;,\n]\s*)(?:{safe_keys})\s*[:=]|$)"
        flags = re.DOTALL | re.IGNORECASE
        cleaned = re.sub(rf"(?:^|[;,\n]\s*){marker_re}\s*[:=]\s*.*?{next_field}", "", cleaned, flags=flags)
        cleaned = re.sub(rf"{marker_re}\s*[:=]\s*.*?{next_field}", "", cleaned, flags=flags)
    if any(marker in cleaned for marker in _HIDDEN_TEXT_MARKERS):
        return "[hidden verifier detail redacted]"
    return re.sub(r"\s*;\s*;", ";", cleaned).strip(" ;,\n")


def redact_solver_hidden_eval_fields(page: dict[str, Any]) -> dict[str, Any]:
    """Apply the upstream-strict redaction policy to a knowledge page.

    Grader answers, hidden test contracts, and hidden verifier transcripts are
    stripped before the page reaches an in-container forum agent. Declared
    experience signals such as outcome scalars, anonymized counts, agent-owned
    commands, and ordinary reflections are preserved. Mutates and returns
    ``page``.

    Callers MUST pass a freshly built page (e.g. KnowledgeStore.query_task's
    per-call json.loads result), never a cached/shared dict.
    """
    for attempt in page.get("attempts") or []:
        content = attempt.get("content")
        if not isinstance(content, dict):
            continue
        eval_results = content.get("eval_results")
        if isinstance(eval_results, dict):
            for key in HIDDEN_TEST_RUNNER_TAIL_KEYS:
                eval_results.pop(key, None)
            for key in HIDDEN_EVAL_ANSWER_KEYS:
                eval_results.pop(key, None)
            per_test = eval_results.get("arc_per_test")
            if isinstance(per_test, list):
                eval_results["arc_per_test"] = [
                    {k: entry[k] for k in ARC_PER_TEST_SAFE_KEYS if k in entry}
                    for entry in per_test
                    if isinstance(entry, dict)
                ]
            instance_report = eval_results.get("instance_report")
            if isinstance(instance_report, dict):
                _redact_instance_report(instance_report)
        attempt_meta = content.get("attempt_meta")
        if isinstance(attempt_meta, dict):
            for key in HIDDEN_ATTEMPT_META_KEYS:
                attempt_meta.pop(key, None)
        for text_key in ("trace_condensed", "reflection"):
            text_value = content.get(text_key)
            if isinstance(text_value, str):
                content[text_key] = redact_solver_hidden_text(text_value)
    for bucket_name in ("discussion", "insights", "distilled"):
        bucket = page.get(bucket_name)
        if isinstance(bucket, list):
            for item in bucket:
                if isinstance(item, dict):
                    _redact_hidden_text_values(item)
    return page
