"""Information-parity guard for the MCP `knowledge` tool.

Forum-phase agents (MCP_TOOLSET='forum') can call the `knowledge` tool, which
returns prior attempts' eval_results, and the forum's posts are distilled into
the next generation's seed (MEMORY.md). The knowledge loop is designed to carry
declared experience forward; leakage is not "any feedback", but information from
outside the phase/split feedback channel being claimed.

For default upstream-strict runs, `handle_knowledge` strips these classes of
content for benchmarks whose solver-attempt channel gets no hidden test feedback
(so the content is effectively hidden grader/test-contract material):
  1. the ARC gold answer in ``arc_per_test[].detail`` (expected grid shape +
     a gold cell value); ``arc_submit_trial`` returns no correctness.
  2. the polyglot / SWE-bench hidden-test-RUNNER tails
     (``test_*_tail`` / ``swebench_*_tail``) — ``pytest --tb=long`` etc. echo
     the failing assertion SOURCE and its gold expected value.
  3. terminal_bench_2 verifier output/clues in ``attempt_meta`` — hidden pytest
     evidence that the benchmark forbids the solver from reading.
Retained as legitimate signal: SWE-bench anonymized counts, terminal_bench_2
outcome scalars/recent commands, and declared outcome scalars.
"""

from __future__ import annotations

import json

from kcsi.memory.mcp_server import _query_from_snapshot, _redact_solver_hidden_eval_fields, handle_knowledge

# Distinctive canaries.
_EXPECTED_CELL = 4242  # ARC gold cell value — must NOT survive.
_TEST_OUTPUT_CANARY = "secretcanarytoken_expected_foldr_eq_99"  # polyglot tail — must NOT survive.


def _page_with_arc_and_output_attempt() -> dict:
    return {
        "task_id": "t1",
        "attempts": [
            {
                "gen": 1,
                "agent_id": "a",
                "score": 0.0,
                "content": {
                    "eval_results": {
                        "status": "ok",
                        "resolved": False,
                        "native_score": 0.0,
                        "arc_correct_count": 1,
                        "arc_total_count": 2,
                        "arc_per_test": [
                            {"test_index": 0, "correct": True, "detail": ""},
                            {
                                "test_index": 1,
                                "correct": False,
                                "detail": {
                                    "reason": "cell_mismatch",
                                    "expected_shape": [3, 3],
                                    "first_mismatch": {"row": 0, "col": 0, "expected": _EXPECTED_CELL, "submitted": 3},
                                },
                            },
                        ],
                        # polyglot hidden-test-runner tail — echoes assertion source
                        # + gold expected value; the solver gets no tests → STRIPPED.
                        "test_stdout_tail": f"FAILED list_ops_test.py::test_foldr - {_TEST_OUTPUT_CANARY}",
                        "test_stderr_tail": f"AssertionError: {_TEST_OUTPUT_CANARY}",
                    }
                },
            }
        ],
        "discussion": [],
        "insights": [],
        "distilled": [],
    }


def test_redact_strips_arc_answer_and_polyglot_test_tails():
    page = _redact_solver_hidden_eval_fields(_page_with_arc_and_output_attempt())
    ev = page["attempts"][0]["content"]["eval_results"]
    # The ARC gold ANSWER is gone.
    assert all("detail" not in t for t in ev["arc_per_test"])
    # The polyglot hidden-test-runner tail is gone (echoes assertion source + gold value).
    assert "test_stdout_tail" not in ev
    assert "test_stderr_tail" not in ev
    # Outcome scalars + per-test correct flag retained.
    assert ev["resolved"] is False
    assert ev["native_score"] == 0.0
    assert ev["arc_correct_count"] == 1
    assert [t["correct"] for t in ev["arc_per_test"]] == [True, False]


class _StubStore:
    def query_task(self, task_id, *, entry_types=None, experiment=None, limit=None):  # noqa: ANN001, D102
        return _page_with_arc_and_output_attempt()


def test_handle_knowledge_strips_arc_answer_and_polyglot_tail():
    page = handle_knowledge(knowledge_store=_StubStore(), task_id="t1")
    blob = json.dumps(page)
    # Neither the ARC gold cell nor the polyglot test-runner tail survives.
    assert str(_EXPECTED_CELL) not in blob
    assert _TEST_OUTPUT_CANARY not in blob
    # Outcome scalars are present (legitimate signal).
    assert '"resolved"' in blob
    assert '"native_score"' in blob


def test_handle_query_routes_page_through_redactor():
    """`query` is a second forum-facing exit of the same attempt page; it must
    apply the same ARC-answer redaction as `handle_knowledge` (defense-in-depth,
    so a future passthrough of eval_results can't reopen the leak)."""
    from kcsi.memory import mcp_server

    captured: dict = {}
    orig = mcp_server._knowledge_attempts_to_query_rows

    def _spy(page):
        # Snapshot the page AFTER redaction to assert the gold answer is gone.
        captured["arc_per_test"] = page["attempts"][0]["content"]["eval_results"]["arc_per_test"]
        return orig(page)

    mcp_server._knowledge_attempts_to_query_rows = _spy
    try:
        mcp_server.handle_query(store=None, task_id="t1", knowledge_store=_StubStore())
    finally:
        mcp_server._knowledge_attempts_to_query_rows = orig

    # By the time rows are built, the ARC gold answer has been stripped from the page.
    assert all("detail" not in t for t in captured["arc_per_test"])


def test_handle_query_redacts_related_attempt_content_from_fts():
    from kcsi.memory import mcp_server

    class _RelatedStore:
        _vec_enabled = False

        def query_task(self, task_id, *, entry_types=None, experiment=None, limit=None):  # noqa: ANN001, D102
            return {"task_id": task_id, "attempts": [], "discussion": [], "insights": [], "distilled": []}

        def fts_search(self, query, *, max_results=5, experiment=None, raw_match=False):  # noqa: ANN001, D102
            return [
                {
                    "task_id": "related-task",
                    "entry_type": "attempt",
                    "content": {
                        "eval_results": {
                            "resolved": False,
                            "native_score": 0.0,
                            "arc_per_test": [
                                {
                                    "test_index": 0,
                                    "correct": False,
                                    "detail": {"first_mismatch": {"expected": _EXPECTED_CELL}},
                                }
                            ],
                            "test_stdout_tail": _TEST_OUTPUT_CANARY,
                        },
                        "attempt_meta": {
                            "reward": 0.0,
                            "failure_signature": f"AssertionError {_TB2_VERIFIER_CANARY}",
                            "verifier_stdout_tail": f"verifier: {_TB2_VERIFIER_CANARY}",
                        },
                    },
                }
            ]

    result = mcp_server.handle_query(
        store=None,
        task_id="t1",
        knowledge_store=_RelatedStore(),
        semantic_query="related task",
    )
    blob = json.dumps(result["related"])

    assert str(_EXPECTED_CELL) not in blob
    assert _TEST_OUTPUT_CANARY not in blob
    assert _TB2_VERIFIER_CANARY not in blob
    related_eval = result["related"][0]["content"]["eval_results"]
    assert related_eval["arc_per_test"] == [{"test_index": 0, "correct": False}]
    assert related_eval["native_score"] == 0.0


def test_handle_query_short_task_id_fallback_returns_and_redacts_related():
    """Regression: the task-id FTS fallback must still fire for a short or
    digit-only task id.

    With no ``semantic_query`` the ``related`` retrieval derives its FTS query
    from the task id itself (``task_id_fallback``). The narrowing filter keeps
    only >=3-char non-numeric tokens; a short id like ``t1`` (or a digit-only
    ARC id) has none, which previously emptied the query and returned ZERO
    related rows — silently disabling the fallback. The fix falls back to the
    raw operator-filtered tokens so such ids still retrieve rows, and the
    redaction path (which only runs on returned rows) is still exercised here.
    """
    from kcsi.memory import mcp_server

    class _RelatedStore:
        _vec_enabled = False

        def query_task(self, task_id, *, entry_types=None, experiment=None, limit=None):  # noqa: ANN001, D102
            return {"task_id": task_id, "attempts": [], "discussion": [], "insights": [], "distilled": []}

        def fts_search(self, query, *, max_results=5, experiment=None, raw_match=False):  # noqa: ANN001, D102
            # A non-empty MATCH query must reach the store; an empty query is
            # the pre-fix failure mode that returned no related rows at all.
            assert query.strip(), "task_id_fallback must not emit an empty FTS query"
            return [
                {
                    "task_id": "related-task",
                    "entry_type": "attempt",
                    "content": {
                        "eval_results": {
                            "resolved": False,
                            "native_score": 0.0,
                            "arc_per_test": [
                                {
                                    "test_index": 0,
                                    "correct": False,
                                    "detail": {"first_mismatch": {"expected": _EXPECTED_CELL}},
                                }
                            ],
                            "test_stdout_tail": _TEST_OUTPUT_CANARY,
                        },
                        "attempt_meta": {
                            "reward": 0.0,
                            "failure_signature": f"AssertionError {_TB2_VERIFIER_CANARY}",
                            "verifier_stdout_tail": f"verifier: {_TB2_VERIFIER_CANARY}",
                        },
                    },
                }
            ]

    # Short id (no semantic_query) -> task_id_fallback path.
    result = mcp_server.handle_query(
        store=None,
        task_id="t1",
        knowledge_store=_RelatedStore(),
    )

    # The fallback still fires: a related row is returned (pre-fix: []).
    assert result["retrieval_mode"] == "fts"
    assert len(result["related"]) == 1

    blob = json.dumps(result["related"])
    assert str(_EXPECTED_CELL) not in blob
    assert _TEST_OUTPUT_CANARY not in blob
    assert _TB2_VERIFIER_CANARY not in blob
    related_eval = result["related"][0]["content"]["eval_results"]
    assert related_eval["arc_per_test"] == [{"test_index": 0, "correct": False}]
    assert related_eval["native_score"] == 0.0

    # A purely digit-only task id (every token dropped by the non-digit
    # narrowing) must also fall back to the raw token and retrieve rows.
    result_digits = mcp_server.handle_query(
        store=None,
        task_id="12345",
        knowledge_store=_RelatedStore(),
    )
    assert len(result_digits["related"]) == 1


def test_snapshot_query_redacts_related_summary_text_fields():
    snapshot = {
        "query_records_by_task": {},
        "related_summaries": [
            {
                "task_id": "related-task",
                "approach": (
                    "TB2 attempt summary: reward=0.0; "
                    f"failure_signature={_TB2_VERIFIER_CANARY}; "
                    f"verifier_clues=['{_TB2_VERIFIER_CANARY}']; "
                    "tool_count=3"
                ),
                "outcome": "unresolved",
                "score": 0.0,
                "lessons": f"keep reward=0.0; verifier_stdout_tail={_TB2_VERIFIER_CANARY}",
            }
        ],
    }

    result = _query_from_snapshot(snapshot=snapshot, task_id="t1")
    blob = json.dumps(result["related"])

    assert _TB2_VERIFIER_CANARY not in blob
    assert "failure_signature" not in blob
    assert "verifier_clues" not in blob
    assert "verifier_stdout_tail" not in blob
    assert "reward=0.0" in blob
    assert "tool_count=3" in blob


def test_query_redacts_flat_trace_condensed_from_legacy_store():
    from kcsi.memory import mcp_server

    class _LegacyMemoryStore:
        def query_task_memory(self, *, task_id, experiment=None, limit=None):  # noqa: ANN001, D102
            return [
                {
                    "gen": 1,
                    "agent_id": "a",
                    "task_id": task_id,
                    "eval_results": {"native_score": 0.0, "resolved": False},
                    "full_memory_trace_condensed": (
                        "TB2 attempt summary: reward=0.0; "
                        f"failure_signature={_TB2_VERIFIER_CANARY}; "
                        f"verifier_clues=['{_TB2_VERIFIER_CANARY}']; "
                        "tool_count=1"
                    ),
                    "attempt_history": [],
                }
            ]

    result = mcp_server.handle_query(store=_LegacyMemoryStore(), task_id="t1")
    trace = result["records"][0]["full_memory_trace_condensed"]

    assert _TB2_VERIFIER_CANARY not in trace
    assert "failure_signature" not in trace
    assert "verifier_clues" not in trace
    assert "reward=0.0" in trace
    assert "tool_count=1" in trace


def test_snapshot_query_redacts_flat_trace_condensed():
    snapshot = {
        "query_records_by_task": {
            "t1": [
                {
                    "gen": 1,
                    "agent_id": "a",
                    "task_id": "t1",
                    "eval_results": {"native_score": 0.0, "resolved": False},
                    "full_memory_trace_condensed": (
                        "TB2 attempt summary: reward=0.0; "
                        f"verifier_stdout_tail=expected foo; got {_TB2_VERIFIER_CANARY}; "
                        "tool_count=1"
                    ),
                }
            ]
        }
    }

    result = _query_from_snapshot(snapshot=snapshot, task_id="t1")
    trace = result["records"][0]["full_memory_trace_condensed"]

    assert _TB2_VERIFIER_CANARY not in trace
    assert "verifier_stdout_tail" not in trace
    assert "reward=0.0" in trace
    assert "tool_count=1" in trace


def test_redactor_scrubs_stale_non_attempt_buckets():
    page = {
        "attempts": [],
        "discussion": [{"text": f"verifier_stdout_tail=expected; got {_TB2_VERIFIER_CANARY}; safe=1"}],
        "insights": [{"text": f"failure_signature={_TB2_VERIFIER_CANARY}; safe=1"}],
        "distilled": [
            {
                "text": f"verifier_stderr_tail=trace {_TB2_VERIFIER_CANARY}; safe=1",
                "bundle": {"checks": [f"verifier_clues={_TB2_VERIFIER_CANARY}; safe=1"]},
            }
        ],
    }

    out = _redact_solver_hidden_eval_fields(page)
    blob = json.dumps(out)

    assert _TB2_VERIFIER_CANARY not in blob
    assert "verifier_stdout_tail" not in blob
    assert "failure_signature" not in blob
    assert "verifier_stderr_tail" not in blob
    assert "verifier_clues" not in blob
    assert "safe=1" in blob


def test_handle_query_strips_legacy_store_attempt_history_arc_answers():
    """Older MemoryStore rows can already contain raw attempt_history entries;
    query must sanitize those read-side payloads too."""
    from kcsi.memory import mcp_server

    class _LegacyMemoryStore:
        def query_task_memory(self, *, task_id, experiment=None, limit=None):  # noqa: ANN001, D102
            return [
                {
                    "gen": 1,
                    "agent_id": "a",
                    "task_id": task_id,
                    "eval_results": {"native_score": 0.0, "resolved": False},
                    "attempt_history": [
                        {
                            "status": "ok",
                            "arc_per_test": [
                                {
                                    "test_index": 0,
                                    "correct": False,
                                    "detail": {"first_mismatch": {"expected": _EXPECTED_CELL}},
                                }
                            ],
                        }
                    ],
                }
            ]

    result = mcp_server.handle_query(store=_LegacyMemoryStore(), task_id="t1")
    blob = json.dumps(result)
    assert str(_EXPECTED_CELL) not in blob
    assert "detail" not in blob
    assert result["records"][0]["attempt_history"][0]["arc_per_test"] == [{"test_index": 0, "correct": False}]


def test_snapshot_query_strips_legacy_attempt_history_arc_answers():
    snapshot = {
        "query_records_by_task": {
            "t1": [
                {
                    "gen": 1,
                    "agent_id": "a",
                    "task_id": "t1",
                    "eval_results": {"native_score": 0.0, "resolved": False},
                    "attempt_history": [
                        {
                            "arc_per_test": [
                                {
                                    "test_index": 0,
                                    "correct": False,
                                    "detail": {"first_mismatch": {"expected": _EXPECTED_CELL}},
                                }
                            ]
                        }
                    ],
                }
            ]
        }
    }
    result = _query_from_snapshot(snapshot=snapshot, task_id="t1")
    blob = json.dumps(result)
    assert str(_EXPECTED_CELL) not in blob
    assert "detail" not in blob
    assert result["records"][0]["attempt_history"][0]["arc_per_test"] == [{"test_index": 0, "correct": False}]


def test_handle_knowledge_no_store_is_safe():
    page = handle_knowledge(knowledge_store=None, task_id="t1")
    assert page == {"task_id": "t1", "attempts": [], "discussion": [], "insights": [], "distilled": []}


# --- SWE-bench Pro: hidden-test-runner tails STRIPPED, hidden test names anonymized ---

_SWE_TAIL_CANARY = "secretcanarytoken_swebench_runner_traceback"
_SWE_TEST_NAME = "tests/secretcanarytoken_test_widget.py::test_hidden_behavior"


def _page_with_swebench_attempt() -> dict:
    return {
        "task_id": "swe1",
        "attempts": [
            {
                "gen": 1,
                "agent_id": "a",
                "score": 0.0,
                "content": {
                    "eval_results": {
                        "swebench_status": "ok",
                        "resolved": False,
                        "native_score": 0.0,
                        "swebench_stdout_tail": f"FAILED {_SWE_TEST_NAME} - {_SWE_TAIL_CANARY}",
                        "swebench_stderr_tail": f"AssertionError: {_SWE_TAIL_CANARY}",
                        "instance_report": {
                            "status": "ok",
                            "resolved": False,
                            "tests": [{"name": _SWE_TEST_NAME, "status": "FAILED"}],
                            "tests_status": {
                                "observed_count": 1,
                                "FAIL_TO_PASS": {
                                    "success": [],
                                    "failure": [_SWE_TEST_NAME],
                                    "skipped": [],
                                    "unknown": [],
                                },
                            },
                        },
                    }
                },
            }
        ],
        "discussion": [],
        "insights": [],
        "distilled": [],
    }


def test_swebench_runner_tails_and_test_names_stripped_to_counts():
    """SWE-bench hidden-test-RUNNER tails echo the failing assertion source +
    gold expected value; upstream-strict solvers never get the FAIL_TO_PASS tests,
    so those tails are outside the declared feedback channel and are stripped.
    Hidden test NAMES are stripped to counts for upstream-strict comparability."""
    page = _redact_solver_hidden_eval_fields(_page_with_swebench_attempt())
    ev = page["attempts"][0]["content"]["eval_results"]
    # Hidden-test-runner tails gone.
    assert "swebench_stdout_tail" not in ev
    assert "swebench_stderr_tail" not in ev
    assert _SWE_TAIL_CANARY not in json.dumps(page)
    # Test identifiers gone; outcome scalars and anonymized counts retained.
    assert _SWE_TEST_NAME not in json.dumps(page)
    assert ev["instance_report"]["tests_count"] == 1
    assert ev["instance_report"]["tests_status"]["observed_count"] == 1
    assert ev["instance_report"]["tests_status"]["FAIL_TO_PASS"]["failure_count"] == 1
    assert ev["resolved"] is False


# --- terminal_bench_2: hidden-pytest verifier output in attempt_meta STRIPPED ---

_TB2_VERIFIER_CANARY = "secretcanarytoken_verifier_failure_clue"


def _page_with_tb2_attempt() -> dict:
    return {
        "task_id": "tb1",
        "attempts": [
            {
                "gen": 1,
                "agent_id": "a",
                "score": 0.0,
                "content": {
                    "eval_results": {"status": "ok", "resolved": False, "native_score": 0.0},
                    "attempt_meta": {
                        "task_source": "terminal_bench_2",
                        "reward": 0.0,
                        "verifier_exit_code": 1,
                        "verified_outcome": "Verifier unresolved with reward 0.",
                        "recent_commands": ["ls -la", "cat solution.txt"],
                        "failure_signature": f"AssertionError {_TB2_VERIFIER_CANARY}",
                        "verifier_clues": [f"expected {_TB2_VERIFIER_CANARY}"],
                        "verifier_stdout_tail": f"verifier: {_TB2_VERIFIER_CANARY}",
                        "verifier_stderr_tail": f"trace: {_TB2_VERIFIER_CANARY}",
                    },
                },
            }
        ],
        "discussion": [],
        "insights": [],
        "distilled": [],
    }


def test_tb2_verifier_output_stripped_from_attempt_meta():
    """terminal_bench_2's verifier runs hidden pytest the benchmark forbids the
    agent from reading, so its tails + extracted clues echo held-out test
    assertions → stripped. Outcome scalars + the agent's own commands retained."""
    page = _redact_solver_hidden_eval_fields(_page_with_tb2_attempt())
    meta = page["attempts"][0]["content"]["attempt_meta"]
    # Held-out verifier content gone.
    assert "verifier_stdout_tail" not in meta
    assert "verifier_stderr_tail" not in meta
    assert "verifier_clues" not in meta
    assert "failure_signature" not in meta
    assert _TB2_VERIFIER_CANARY not in json.dumps(page)
    # Outcome scalars + the agent's own commands retained.
    assert meta["reward"] == 0.0
    assert meta["verifier_exit_code"] == 1
    assert meta["verified_outcome"] == "Verifier unresolved with reward 0."
    assert meta["recent_commands"] == ["ls -la", "cat solution.txt"]


def test_knowledge_attempts_to_seed_records_strips_hidden_fields_without_mutating_source():
    from kcsi.orchestrator.enrichment_phase import _knowledge_attempts_to_seed_records

    page = _page_with_tb2_attempt()
    page["attempts"][0]["content"]["trace_condensed"] = (
        "TB2 attempt summary: reward=0.0; "
        f"failure_signature={_TB2_VERIFIER_CANARY}; "
        f"verifier_clues=['{_TB2_VERIFIER_CANARY}']; "
        "tool_count=2"
    )
    page["attempts"][0]["content"]["eval_results"] = {
        "status": "ok",
        "native_score": 0.0,
        "resolved": False,
        "arc_per_test": [
            {
                "test_index": 0,
                "correct": False,
                "detail": {"first_mismatch": {"expected": _EXPECTED_CELL}},
            }
        ],
        "test_stdout_tail": _TEST_OUTPUT_CANARY,
    }

    records = _knowledge_attempts_to_seed_records(page)
    blob = json.dumps(records)
    assert str(_EXPECTED_CELL) not in blob
    assert _TEST_OUTPUT_CANARY not in blob
    assert _TB2_VERIFIER_CANARY not in blob
    assert "failure_signature" not in blob
    assert "verifier_clues" not in blob
    assert "reward=0.0" in records[0]["full_memory_trace_condensed"]
    assert records[0]["eval_results"]["arc_per_test"] == [{"test_index": 0, "correct": False}]
    assert page["attempts"][0]["content"]["eval_results"]["arc_per_test"][0]["detail"]
    assert page["attempts"][0]["content"]["attempt_meta"]["verifier_stdout_tail"]
