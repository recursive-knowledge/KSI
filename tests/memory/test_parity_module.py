from ksi.memory.parity import (
    ARC_PER_TEST_SAFE_KEYS,
    HIDDEN_ATTEMPT_META_KEYS,
    HIDDEN_EVAL_ANSWER_KEYS,
    HIDDEN_TEST_RUNNER_TAIL_KEYS,
    redact_solver_hidden_eval_fields,
    redact_solver_hidden_text,
)


def test_arc_per_test_projected_to_safe_keys_drops_unknown_nested_answer():
    """The redactor must allow-list arc_per_test entries to {test_index, correct},
    so the demonstrated `detail` leak AND any FUTURE nested answer key fail closed."""
    page = {
        "attempts": [
            {
                "content": {
                    "eval_results": {
                        "resolved": False,
                        "arc_per_test": [
                            {
                                "test_index": 1,
                                "correct": False,
                                "detail": {"first_mismatch": {"expected": 4242}},
                                "future_gold_field": {"answer_grid": [[4242]]},
                            }
                        ],
                    }
                }
            }
        ]
    }
    out = redact_solver_hidden_eval_fields(page)
    entry = out["attempts"][0]["content"]["eval_results"]["arc_per_test"][0]
    assert set(entry) <= ARC_PER_TEST_SAFE_KEYS
    assert entry == {"test_index": 1, "correct": False}


def test_known_tail_keys_still_stripped():
    page = {"attempts": [{"content": {"eval_results": {"test_stdout_tail": "X", "resolved": False}}}]}
    out = redact_solver_hidden_eval_fields(page)
    assert "test_stdout_tail" not in out["attempts"][0]["content"]["eval_results"]
    assert out["attempts"][0]["content"]["eval_results"]["resolved"] is False


def test_top_level_answer_keys_are_stripped():
    page = {
        "attempts": [
            {
                "content": {
                    "eval_results": {
                        "resolved": False,
                        "expected": "secretcanarytoken_expected",
                        "expected_shape": [3, 3],
                        "detail": {"gold": "secretcanarytoken_detail"},
                    }
                }
            }
        ]
    }

    out = redact_solver_hidden_eval_fields(page)
    eval_results = out["attempts"][0]["content"]["eval_results"]
    blob = str(eval_results)

    assert "secretcanarytoken" not in blob
    assert "expected" not in eval_results
    assert "expected_shape" not in eval_results
    assert "detail" not in eval_results
    assert eval_results["resolved"] is False


def test_declared_experience_signals_are_retained():
    page = {
        "attempts": [
            {
                "content": {
                    "eval_results": {
                        "resolved": False,
                        "native_score": 0.25,
                        "agent_stdout_tail": "public smoke test failed: missing CLI flag",
                        "self_generated_test_output": "unit test reproduced the parse error",
                    },
                    "attempt_meta": {
                        "reward": 0.25,
                        "agent_exit_code": 1,
                        "recent_commands": ["pytest tests/test_cli.py -q"],
                    },
                    "trace_condensed": "reward=0.25; recent_commands=pytest tests/test_cli.py -q; next=fix CLI flag",
                    "reflection": "The public smoke test and my local unit test point to the CLI parser.",
                }
            }
        ]
    }

    out = redact_solver_hidden_eval_fields(page)
    content = out["attempts"][0]["content"]

    assert content["eval_results"]["resolved"] is False
    assert content["eval_results"]["native_score"] == 0.25
    assert content["eval_results"]["agent_stdout_tail"] == "public smoke test failed: missing CLI flag"
    assert content["eval_results"]["self_generated_test_output"] == "unit test reproduced the parse error"
    assert content["attempt_meta"]["reward"] == 0.25
    assert content["attempt_meta"]["recent_commands"] == ["pytest tests/test_cli.py -q"]
    assert "next=fix CLI flag" in content["trace_condensed"]
    assert "public smoke test" in content["reflection"]


def test_stale_hidden_trace_text_is_scrubbed():
    text = (
        "TB2 attempt summary: reward=0.0 agent_exit=0; "
        "failure_signature=secretcanarytoken_old_failure; "
        "verifier_clues=['secretcanarytoken_old_clue']; "
        "tool_count=3"
    )

    redacted = redact_solver_hidden_text(text)

    assert "secretcanarytoken" not in redacted
    assert "failure_signature" not in redacted
    assert "verifier_clues" not in redacted
    assert "reward=0.0" in redacted
    assert "tool_count=3" in redacted


def test_stale_hidden_trace_text_with_semicolons_is_scrubbed():
    text = (
        "TB2 attempt summary: reward=0.0; "
        "verifier_stdout_tail=Expected foo; got secretcanarytoken_after_semi; "
        "tool_count=3"
    )

    redacted = redact_solver_hidden_text(text)

    assert "secretcanarytoken" not in redacted
    assert "verifier_stdout_tail" not in redacted
    assert "Expected foo" not in redacted
    assert "got " not in redacted
    assert "reward=0.0" in redacted
    assert "tool_count=3" in redacted


def test_stale_hidden_trace_text_with_newlines_is_scrubbed():
    text = "prefix; verifier_stdout_tail=line1\nsecretcanarytoken_multiline\nnext=ok"

    redacted = redact_solver_hidden_text(text)

    assert "secretcanarytoken" not in redacted
    assert "verifier_stdout_tail" not in redacted
    assert "line1" not in redacted
    assert "prefix" in redacted
    assert "next=ok" in redacted


def test_redactor_scrubs_reflection_text():
    page = {
        "attempts": [
            {
                "content": {
                    "reflection": (
                        "I checked reward=0.0; verifier_stdout_tail=Expected foo; "
                        "got secretcanarytoken_reflection; proposed change: retry config"
                    )
                }
            }
        ]
    }

    out = redact_solver_hidden_eval_fields(page)
    reflection = out["attempts"][0]["content"]["reflection"]

    assert "secretcanarytoken" not in reflection
    assert "verifier_stdout_tail" not in reflection
    assert "Expected foo" not in reflection
    assert "reward=0.0" in reflection
    assert "proposed change: retry config" in reflection


def test_policy_constants_exported():
    assert "test_stdout_tail" in HIDDEN_TEST_RUNNER_TAIL_KEYS
    assert "verifier_stdout_tail" in HIDDEN_ATTEMPT_META_KEYS
    assert "expected" in HIDDEN_EVAL_ANSWER_KEYS
    assert ARC_PER_TEST_SAFE_KEYS == frozenset({"test_index", "correct"})
