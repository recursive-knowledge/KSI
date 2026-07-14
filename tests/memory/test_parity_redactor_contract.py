"""Contract test: the MCP knowledge redactor must cover every hidden-test-runner
output key any evaluator emits.

Parity rule (src/kcsi/memory/parity.py): adaptive surfaces may contain only
information from the declared phase/split feedback channel. In default
upstream-strict runs, each maintained no-feedback benchmark (polyglot /
SWE-bench / terminal_bench_2 — all run hidden pytest) emits hidden test-runner
transcripts under a ``*_stdout_tail`` / ``*_stderr_tail`` key, which must be
stripped by ``_redact_solver_hidden_eval_fields`` before any agent-facing exit.

This test fails CLOSED: a NEW evaluator that adds a ``*_tail`` key must classify
it as either stripped (hidden test/verifier output) or safe (the agent's OWN run
output / container teardown). An unclassified key trips CI here instead of
silently leaking the test contract into the next generation's seed.
"""

from __future__ import annotations

import json
import re

from conftest import REPO_ROOT

from kcsi.memory.mcp_server import (
    _HIDDEN_ATTEMPT_META_KEYS,
    _HIDDEN_TEST_RUNNER_TAIL_KEYS,
    _redact_solver_hidden_eval_fields,
)
from kcsi.memory.parity import HIDDEN_EVAL_ANSWER_KEYS

_REPO = REPO_ROOT

# Tail keys the redactor strips (hidden test / verifier transcript).
_STRIPPED_TAILS = frozenset(_HIDDEN_TEST_RUNNER_TAIL_KEYS) | {
    k for k in _HIDDEN_ATTEMPT_META_KEYS if k.endswith(("_stdout_tail", "_stderr_tail"))
}

# Tail keys that are NOT a leak in the upstream-strict channel:
#   agent_*  : the agent's OWN process stdout/stderr (terminal_bench_2_runtime)
#   cleanup_*: the container-teardown step's output, not test assertions (polyglot)
# Adding an entry here is a deliberate "this tail belongs to the declared
# feedback channel" decision.
_SAFE_TAILS = frozenset(
    {
        "agent_stdout_tail",
        "agent_stderr_tail",
        "cleanup_stdout_tail",
        "cleanup_stderr_tail",
    }
)

# Matches both quoted dict keys ("test_stdout_tail") and kwarg form
# (test_stdout_tail=...), since evaluators use both.
_TAIL_RE = re.compile(r"\b([a-z][a-z0-9_]*_(?:stdout|stderr)_tail)\b")

# Where evaluators/runtimes/memory tools produce eval or agent-facing output
# keys, plus the distillation prompt that *renders* attempt content to the
# distill LLM. The distill renderers protect via render-time field selection (not
# the redactor), so a renderer that names a hidden tail/answer key must trip here;
# tests/distillation/test_distill_parity_contract.py adds the behavioral guard
# against a key-less raw dump that this name-based scan cannot see.
_SCAN = (
    "src/kcsi/eval",
    "src/kcsi/runtime",
    "src/kcsi/orchestrator/engine.py",
    "src/kcsi/memory",
    "src/kcsi/distillation",
)


def _discover_tail_keys() -> set[str]:
    keys: set[str] = set()
    for rel in _SCAN:
        p = _REPO / rel
        files = p.rglob("*.py") if p.is_dir() else [p]
        for f in files:
            keys.update(_TAIL_RE.findall(f.read_text(encoding="utf-8")))
    return keys


def test_every_eval_tail_key_is_classified():
    discovered = _discover_tail_keys()
    # Guard against the regex silently matching nothing (e.g. a refactor to a
    # computed key) — the known hidden tails must be discoverable.
    assert {"test_stdout_tail", "swebench_stdout_tail", "verifier_stdout_tail"} <= discovered, (
        f"sanity scan failed; discovered={sorted(discovered)}"
    )
    unclassified = discovered - (_STRIPPED_TAILS | _SAFE_TAILS)
    assert not unclassified, (
        f"Unclassified stdout/stderr tail key(s): {sorted(unclassified)}. "
        "If it is hidden test/verifier output outside the declared channel, add it to "
        "_HIDDEN_TEST_RUNNER_TAIL_KEYS or _HIDDEN_ATTEMPT_META_KEYS in "
        "src/kcsi/memory/parity.py; if it is declared agent/runtime feedback, add "
        "it to _SAFE_TAILS here. See src/kcsi/memory/parity.py."
    )


def test_redactor_strips_all_declared_hidden_keys():
    """Canary coverage: the redactor actually removes every key it declares, so
    an edit that drops one from the loop is caught."""
    eval_results: dict = {"resolved": False, "native_score": 0.0}
    for key in _HIDDEN_TEST_RUNNER_TAIL_KEYS:
        eval_results[key] = f"CANARY_{key}"
    eval_results["arc_per_test"] = [{"correct": False, "detail": {"first_mismatch": {"expected": 4242}}}]
    attempt_meta: dict = {"reward": 0.0}
    for key in _HIDDEN_ATTEMPT_META_KEYS:
        attempt_meta[key] = f"CANARY_{key}"
    page = {"attempts": [{"content": {"eval_results": eval_results, "attempt_meta": attempt_meta}}]}

    blob = json.dumps(_redact_solver_hidden_eval_fields(page))
    assert "CANARY_" not in blob  # every declared hidden key removed
    assert "4242" not in blob  # ARC gold answer removed
    assert '"resolved"' in blob and '"reward"' in blob  # scalars retained


def test_unknown_nested_arc_answer_field_fails_closed():
    """A future nested answer key under arc_per_test must NOT survive — the
    allow-list projection is the structural guarantee (deny-lists can't catch
    a key nobody remembered to add)."""
    from kcsi.memory.parity import redact_solver_hidden_eval_fields

    page = {
        "attempts": [
            {
                "content": {
                    "eval_results": {
                        "arc_per_test": [{"test_index": 0, "correct": False, "leaked_expected_grid": [[7]]}]
                    }
                }
            }
        ]
    }
    blob = json.dumps(redact_solver_hidden_eval_fields(page))
    assert "leaked_expected_grid" not in blob
    assert "[[7]]" not in blob and "[[7]]".replace(" ", "") not in blob.replace(" ", "")


# Answer-shaped key-name fragments that must be either stripped or explicitly
# allow-listed. Catches the non-`_tail` leak class (e.g. arc_per_test[].detail)
# in quoted keys and ``dict(answer_key=...)`` / keyword-style emitters.
_ANSWER_KEY_RE = re.compile(
    r"""["']((?:[a-z][a-z0-9_]*_)?(?:expected|gold|answer|solution|detail)(?:_[a-z0-9]+)*)["']|"""
    r"""(?:dict\(|,\s*)((?:[a-z][a-z0-9_]*_)?(?:expected|gold|answer|solution|detail)(?:_[a-z0-9]+)*)\s*="""
)

_STRIPPED_ANSWER_KEYS = frozenset(HIDDEN_EVAL_ANSWER_KEYS) | {
    # ARC grader detail keys are hidden answers; top-level exact keys are
    # stripped and arc_per_test entries are structurally allow-list projected.
    "detail",
    "expected",
    "expected_shape",
}

# Keys/identifiers that look answer-shaped but are confirmed inside the declared
# channel / not a leak. Extend only with a one-line rationale.
_ANSWER_KEY_ALLOWLIST = frozenset(
    {
        # Coordination counters, not answer keys.
        "agents_expected",
        "expected_agents",
        "expected_agents_set",
        "expected_count",
        "forum_expected_agents",
        # ARC evaluator-local variable, not an agent-facing key.
        "expected_grids",
        # TB2 trial-local variable (tuple-unpacked from
        # _extract_trusted_bash_from_image in runtime/terminal_bench_2_trial.py);
        # feeds verifier_trusted_bash_detail, never a delivered dict key.
        "extract_detail",
        # TB2 runtime_meta / trial_result.json audit-only keys (verifier-toolchain
        # and reward-readout diagnostics). Neither is selected into the
        # agent-facing attempt_meta allowlist projection (_tb2_attempt_meta in
        # orchestrator/attempt_events.py) nor into eval_results, so they never
        # reach MEMORY.md / distillation / forum — audit sidecar only.
        "verifier_trusted_bash_detail",
        "reward_readout_detail",
        # polyglot_harness status VALUE ("no_solution" = the agent shipped no
        # solution files); an outcome label, not a grader answer.
        "no_solution",
        # Solver-owned artifact names/metadata, not grader answers.
        "solution",
        "solution_files",
        "solution_source",
        "workspace_solution_files",
        # Terminal-Bench 2 verifier diagnostics, not task answer keys.
        "extract_detail",
        "reward_readout_detail",
        "verifier_trusted_bash_detail",
    }
)


def test_answer_key_regex_catches_exact_and_prefixed_keys():
    samples = [
        '"detail"',
        '"expected"',
        '"expected_shape"',
        '"gold_grid"',
        '"answer"',
        '"task_solution"',
        "dict(gold_grid=[[1]])",
        "foo(x=1, gold_grid=[[1]])",
    ]
    for sample in samples:
        assert _ANSWER_KEY_RE.search(sample), sample


def _answer_key_matches(text: str) -> set[str]:
    out: set[str] = set()
    for match in _ANSWER_KEY_RE.finditer(text):
        key = match.group(1) or match.group(2)
        if key:
            out.add(key)
    return out


def test_answer_shaped_keys_are_classified():
    discovered: set[str] = set()
    for rel in _SCAN:
        p = _REPO / rel
        files = p.rglob("*.py") if p.is_dir() else [p]
        for f in files:
            discovered.update(_answer_key_matches(f.read_text(encoding="utf-8")))
    unclassified = discovered - (_STRIPPED_ANSWER_KEYS | _ANSWER_KEY_ALLOWLIST)
    assert not unclassified, (
        f"Answer-shaped key(s) {sorted(unclassified)} found under {_SCAN}. "
        "If a grader answer, ensure the redactor strips/allow-list-projects it; "
        "then add to _ANSWER_KEY_ALLOWLIST here with a rationale. "
        "See src/kcsi/memory/parity.py."
    )
