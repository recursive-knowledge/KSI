from ksi.benchmarks.arc_session import ArcSessionEvaluator
from ksi.models import TaskSpec


def _task_with_tests(test_pairs):
    return TaskSpec(
        id="arc-task-1",
        repo="",
        prompt="arc",
        metadata={
            "task_source": "arc",
            "arc_test_pairs": test_pairs,
        },
    )


def test_evaluator_no_submission_no_trace_is_infra_zero():
    """Canonical scoring has no text-output fallback. A model_output with no
    tool trace cannot be scored canonically (the runtime captured nothing), so
    it is an infra-failure 0: native_score 0, resolved False, and
    scored_from_runtime_trials left ABSENT so paper analysis filters it."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 1], [1, 1]]}])
    result = evaluator.evaluate(task=task, model_output="[[1,1],[1,1]]")
    assert result["status"] == "no_runtime_submission"
    assert result["resolved"] is False
    assert result["native_score"] == 0.0
    assert result["arc_total_count"] == 1
    assert "scored_from_runtime_trials" not in result


def test_evaluator_prose_output_no_trace_is_infra_zero():
    """Prose / unparseable model_output is no longer recovered — same canonical
    infra-zero as any other no-trace run."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1]]}])
    result = evaluator.evaluate(task=task, model_output="I think answer is one.")
    assert result["status"] == "no_runtime_submission"
    assert result["resolved"] is False
    assert result["native_score"] == 0.0


def test_evaluator_trace_without_submission_scores_zero_canonical():
    """A tool trace that sets a grid but never submits gets a distinct
    ``"no_submission"`` status (the agent ran but did not formally submit):
    scored_from_runtime_trials True, native_score 0, resolved False. The
    orchestrator's score_arc_from_eval keeps this status as a scored failed
    attempt."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 1], [1, 1]]}])
    result = evaluator.evaluate(
        task=task,
        model_output="",
        tool_trace=[_set_grid_call([[1, 1], [1, 1]])],  # set but never submitted
    )
    assert result["status"] == "no_submission"
    assert result["resolved"] is False
    assert result["native_score"] == 0.0
    assert result["scored_from_runtime_trials"] is True


def test_arc_session_reconstructs_blind_runtime_trials():
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 2], [3, 4]]}])
    tool_trace = [
        _set_grid_call([[1, 2], [3, 4]]),
        {
            "type": "message",
            "text": "Now let me submit this trial:",
        },
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="I solved it with the tool.",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["status"] == "ok"
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert result["scored_from_runtime_trials"] is True


# ---------------------------------------------------------------------------
# Canonical ARC Prize blind-scoring: the runtime no longer emits ``correct``
# on the arc_submit_trial response. The scorer reconstructs correctness from
# the tool_trace itself.
# ---------------------------------------------------------------------------


def _set_grid_call(grid):
    return {
        "type": "tool_call",
        "tool_name": "arc_set_output_grid",
        "tool_input": {"grid": grid},
        "tool_output": '{"status": "ok"}',
    }


def _submit_call():
    return {
        "type": "tool_call",
        "tool_name": "arc_submit_trial",
        "tool_input": {},
        "tool_output": '{"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0}',
    }


def _next_test_call(*, ok: bool = True):
    if ok:
        output = '{"status": "ok", "current_test_index": 1, "test_count": 2}'
    else:
        output = '{"status": "no_next_test_input"}'
    return {
        "type": "tool_call",
        "tool_name": "arc_next_test_input",
        "tool_input": {},
        "tool_output": output,
    }


def _resize_call(height, width):
    return {
        "type": "tool_call",
        "tool_name": "arc_resize_output_grid",
        "tool_input": {"height": height, "width": width},
        "tool_output": f'{{"status": "ok", "old_shape": [0, 0], "new_shape": [{height}, {width}]}}',
    }


def test_scorer_reconstructs_correct_verdict_from_tool_trace_single_test():
    """Blind-scoring path: runtime_meta has submit entries WITHOUT ``correct``
    (canonical ARC). Scorer must walk the tool trace, pair set_output_grid
    with the following submit_trial, and compare against the expected grid."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 2], [3, 4]]}])
    tool_trace = [
        _set_grid_call([[1, 2], [3, 4]]),  # matches expected
        _submit_call(),
    ]
    # runtime_meta carries the blind-shaped submit result (no `correct` field).
    result = evaluator.evaluate(
        task=task,
        model_output="",  # scorer should NOT need the model_output here
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["status"] == "ok"
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert result["scored_from_runtime_trials"] is True
    # The verdict came from trace reconstruction, not a legacy ``correct`` field.
    assert result["arc_per_test"][0]["source"] == "trace_reconstruction"


def test_scorer_reconstructs_wrong_verdict_from_tool_trace():
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 2], [3, 4]]}])
    tool_trace = [
        _set_grid_call([[9, 9], [9, 9]]),  # wrong
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["status"] == "ok"
    assert result["resolved"] is False
    assert result["native_score"] == 0.0
    assert result["arc_per_test"][0]["correct"] is False


def test_scorer_reconstructs_multi_test_from_tool_trace_with_next_test():
    """Multi-test reconstruction: set_output_grid → submit → next_test →
    set_output_grid → submit. Advance cursor after next_test so the second
    submit is scored against test_index=1."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests(
        [
            {"input": [[0]], "output": [[1]]},
            {"input": [[0]], "output": [[2]]},
        ]
    )
    tool_trace = [
        _set_grid_call([[1]]),  # matches test 0
        _submit_call(),
        _next_test_call(),
        _set_grid_call([[2]]),  # matches test 1
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0, "trial_count": 1, "trials_remaining": 1},
                {"status": "ok", "test_index": 1, "trial_count": 1, "trials_remaining": 1},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is True
    assert result["arc_correct_count"] == 2
    assert result["arc_total_count"] == 2


def test_scorer_reconstructs_multi_test_native_unprefixed_tool_names():
    """Native ARC synthesizer emits tool names `arc_set_output_grid`,
    `arc_submit_trial`, `arc_next_test_input`. The scorer must reconstruct the
    cursor advance on `arc_next_test_input`, so a 2-test task with both grids
    correct scores 1.0 / resolved=True (issue #694)."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests(
        [
            {"input": [[0]], "output": [[1]]},
            {"input": [[0]], "output": [[2]]},
        ]
    )
    tool_trace = [
        _set_grid_call([[1]]),  # matches test 0
        _submit_call(),
        _next_test_call(),
        _set_grid_call([[2]]),  # matches test 1
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0, "trial_count": 1, "trials_remaining": 1},
                {"status": "ok", "test_index": 1, "trial_count": 1, "trials_remaining": 1},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["native_score"] == 1.0
    assert result["resolved"] is True
    assert result["arc_correct_count"] == 2
    assert result["arc_total_count"] == 2


def test_scorer_reconstructs_multi_test_partial_credit():
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests(
        [
            {"input": [[0]], "output": [[1]]},
            {"input": [[0]], "output": [[2]]},
        ]
    )
    tool_trace = [
        _set_grid_call([[1]]),  # correct for test 0
        _submit_call(),
        _next_test_call(),
        _set_grid_call([[9]]),  # wrong for test 1
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0},
                {"status": "ok", "test_index": 1},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is False
    assert result["arc_correct_count"] == 1
    assert result["arc_total_count"] == 2
    assert result["arc_pass_ratio"] == 0.5


def test_scorer_multi_test_unsubmitted_test_gets_full_per_test_row():
    """When the agent submits some but not all tests, arc_per_test still
    carries a row per test input (the unsubmitted one recorded as wrong),
    matching the no-submission paths. The unsubmitted test counts as 0."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests(
        [
            {"input": [[0]], "output": [[1]]},
            {"input": [[0]], "output": [[2]]},
        ]
    )
    tool_trace = [
        _set_grid_call([[1]]),  # correct for test 0
        _submit_call(),
        # never advanced to / submitted test 1
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={"tool_trace": tool_trace},
    )
    assert result["resolved"] is False
    assert result["arc_correct_count"] == 1
    assert result["arc_total_count"] == 2
    assert result["arc_pass_ratio"] == 0.5
    by_index = {item["test_index"]: item for item in result["arc_per_test"]}
    assert set(by_index) == {0, 1}
    assert by_index[0]["correct"] is True
    assert by_index[1]["correct"] is False
    assert by_index[1]["detail"] == "no accepted scoring attempt"


def test_scorer_reconstruction_submit_without_set_grid_counts_as_wrong():
    """If the agent calls submit_trial without a preceding set_output_grid,
    we have nothing to compare — treat it as a wrong submission rather than
    silently dropping it."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1]]}])
    tool_trace = [_submit_call()]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is False
    assert result["arc_per_test"][0]["correct"] is False


def test_scorer_reconstruction_keeps_best_on_wrong_then_right():
    """Two submits on the same test_index: wrong then right. Keep-best must
    still apply (the second correct trial wins)."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[7]]}])
    tool_trace = [
        _set_grid_call([[0]]),
        _submit_call(),
        _set_grid_call([[7]]),
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0},
                {"status": "ok", "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is True
    assert result["arc_per_test"][0]["correct"] is True


def test_scorer_reconstruction_enforces_max_trials():
    """Trace reconstruction must not award credit for submissions beyond the
    ARC per-test trial budget."""
    evaluator = ArcSessionEvaluator()
    task = TaskSpec(
        id="arc-budget",
        repo="",
        prompt="arc",
        metadata={
            "task_source": "arc",
            "arc_max_trials": 2,
            "arc_test_pairs": [{"input": [[0]], "output": [[7]]}],
        },
    )
    tool_trace = [
        _set_grid_call([[0]]),
        _submit_call(),
        _set_grid_call([[1]]),
        _submit_call(),
        _set_grid_call([[7]]),
        _submit_call(),
    ]

    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0},
                {"status": "ok", "test_index": 0},
                {"status": "ok", "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )

    assert result["status"] == "ok"
    assert result["resolved"] is False
    assert result["arc_per_test"][0]["correct"] is False


def test_scorer_reconstruction_honors_runtime_effective_max_trials_override():
    """#1046: the agent can call ``arc_load_task(max_trials=5)``, which the
    live session genuinely enforces (arc_semantics.py) independent of the
    task-metadata-derived default. The scorer must honor that effective
    override (recorded in ``runtime_meta["arc_effective_max_trials"]`` at
    session-load time) rather than re-deriving a stale value from task
    metadata, or it wrongly discards a correct later submission as
    over-budget."""
    evaluator = ArcSessionEvaluator()
    task = TaskSpec(
        id="arc-budget-override",
        repo="",
        prompt="arc",
        metadata={
            "task_source": "arc",
            # Task metadata still says 2 -- stale relative to the live
            # session's actual (agent-overridden) budget of 5.
            "arc_max_trials": 2,
            "arc_test_pairs": [{"input": [[0]], "output": [[7]]}],
        },
    )
    tool_trace = [
        _set_grid_call([[0]]),
        _submit_call(),
        _set_grid_call([[1]]),
        _submit_call(),
        _set_grid_call([[7]]),
        _submit_call(),
    ]

    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_effective_max_trials": 5,
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0},
                {"status": "ok", "test_index": 0},
                {"status": "ok", "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )

    assert result["status"] == "ok"
    assert result["resolved"] is True
    assert result["arc_per_test"][0]["correct"] is True


def test_scorer_reconstruction_ignores_trial_limit_submit():
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[7]]}])
    over_budget_submit = {
        "type": "tool_call",
        "tool_name": "arc_submit_trial",
        "tool_input": {},
        "tool_output": ('{"status": "trial_limit_exceeded", "trial_count": 2, "trials_remaining": 0, "test_index": 0}'),
    }
    tool_trace = [
        _set_grid_call([[0]]),
        _submit_call(),
        _set_grid_call([[7]]),
        over_budget_submit,
    ]

    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0},
                {"status": "trial_limit_exceeded", "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )

    assert result["status"] == "ok"
    assert result["resolved"] is False
    assert result["arc_per_test"][0]["correct"] is False


def test_scorer_reconstruction_rejected_submit_blocks_model_output_fallback():
    """Only a rejected submit exists (no accepted submission for any test), so
    this is the "no_submission" case, not a genuine wrong-answer "ok"."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[7]]}])
    rejected_submit = {
        "type": "tool_call",
        "tool_name": "arc_submit_trial",
        "tool_input": {},
        "tool_output": '{"status": "trial_limit_exceeded", "test_index": 0}',
    }

    result = evaluator.evaluate(
        task=task,
        model_output="[[7]]",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "trial_limit_exceeded", "test_index": 0},
            ],
            "tool_trace": [rejected_submit],
        },
    )

    assert result["status"] == "no_submission"
    assert result["resolved"] is False
    assert result["scored_from_runtime_trials"] is True
    assert result["arc_per_test"][0]["detail"] == "no accepted scoring attempt"


def test_scorer_reconstruction_enforces_budget_per_test_index():
    evaluator = ArcSessionEvaluator()
    task = TaskSpec(
        id="arc-budget-multi",
        repo="",
        prompt="arc",
        metadata={
            "task_source": "arc",
            "arc_max_trials": 2,
            "arc_test_pairs": [
                {"input": [[0]], "output": [[7]]},
                {"input": [[1]], "output": [[8]]},
            ],
        },
    )
    tool_trace = [
        _set_grid_call([[0]]),
        _submit_call(),
        _set_grid_call([[1]]),
        _submit_call(),
        _set_grid_call([[7]]),
        _submit_call(),
        _next_test_call(),
        _set_grid_call([[8]]),
        _submit_call(),
    ]

    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0},
                {"status": "ok", "test_index": 0},
                {"status": "ok", "test_index": 0},
                {"status": "ok", "test_index": 1},
            ],
            "tool_trace": tool_trace,
        },
    )

    assert result["resolved"] is False
    assert result["native_score"] == 0.5
    by_index = {item["test_index"]: item for item in result["arc_per_test"]}
    assert by_index[0]["correct"] is False
    assert by_index[1]["correct"] is True


def test_no_trace_with_runtime_meta_results_is_infra_zero():
    """Blind-shaped runtime_meta submit entries but NO tool_trace: the scorer
    cannot reconstruct a submission, and there is no text fallback, so this is
    an infra-failure 0 (scored_from_runtime_trials absent) rather than a
    model_output recovery."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 1], [1, 1]]}])
    result = evaluator.evaluate(
        task=task,
        model_output="[[1,1],[1,1]]",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "test_index": 0},
            ],
        },
    )
    assert result["status"] == "no_runtime_submission"
    assert result["resolved"] is False
    assert result["native_score"] == 0.0
    assert "scored_from_runtime_trials" not in result


def test_missing_reference_sets_scored_from_runtime_trials_false():
    evaluator = ArcSessionEvaluator()
    task = TaskSpec(
        id="arc-task-1",
        repo="",
        prompt="arc",
        metadata={"task_source": "arc"},
    )
    result = evaluator.evaluate(task=task, model_output="[[1]]")
    assert result["status"] == "missing_reference"
    assert result["scored_from_runtime_trials"] is False


def test_trial_path_keeps_scored_from_runtime_trials_true():
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[5]]}])
    result = evaluator.evaluate(
        task=task,
        model_output="",
        tool_trace=[_set_grid_call([[5]]), _submit_call()],
    )
    assert result["resolved"] is True
    assert result["scored_from_runtime_trials"] is True


# ---------------------------------------------------------------------------
# #1041: arc_resize_output_grid must be replayed during trace reconstruction
# with the same overlap-copy/zero-fill semantics as
# ArcSession.resize_output_grid — a resize followed by submit with no further
# arc_set_output_grid call must be scored against the resized grid, not the
# stale pre-resize grid.
# ---------------------------------------------------------------------------


def test_scorer_reconstruction_resize_up_without_reset_zero_fills_new_cells():
    """Agent sets a 2x2 grid, resizes UP to 3x3 (relying on zero-padding for
    the new cells), then submits with no further set_output_grid call. The
    reconstruction must replay the resize: overlap copied top-left, new cells
    zero-filled — not silently keep scoring the stale 2x2 grid."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 2, 0], [3, 4, 0], [0, 0, 0]]}])
    tool_trace = [
        _set_grid_call([[1, 2], [3, 4]]),
        _resize_call(3, 3),
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert result["arc_per_test"][0]["correct"] is True


def test_scorer_reconstruction_resize_down_without_reset_crops_overlap():
    """Agent sets a 3x3 grid, resizes DOWN (crops) to 2x2, then submits with
    no further set_output_grid call. The reconstruction must replay the crop
    (top-left 2x2 kept), not keep comparing the stale 3x3 grid."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 2], [4, 5]]}])
    tool_trace = [
        _set_grid_call([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        _resize_call(2, 2),
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert result["arc_per_test"][0]["correct"] is True


def test_scorer_reconstruction_resize_then_full_reset_still_scores_reset_grid():
    """Regression guard for the observed real-world pattern (2/112 traces):
    resize followed by a full arc_set_output_grid before submit. The later
    full set must win — resize tracking must not corrupt this case."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[9, 9], [9, 9]]}])
    tool_trace = [
        _set_grid_call([[1, 2], [3, 4]]),
        _resize_call(3, 3),
        _set_grid_call([[9, 9], [9, 9]]),  # full reset after resize
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert result["arc_per_test"][0]["correct"] is True


def test_scorer_reconstruction_resize_before_any_set_grid_uses_default_zero_base():
    """Resizing with no prior set_output_grid call must base the resize on
    ArcSession's actual default output grid (a 3x3 zero grid), matching
    ArcSession.__init__ / next_test_input's reset — not treat the missing
    grid as unrecoverable."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[0, 0], [0, 0]]}])
    tool_trace = [
        _resize_call(2, 2),  # no preceding set_output_grid
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert result["arc_per_test"][0]["correct"] is True


def test_scorer_reconstruction_invalid_resize_dimensions_are_skipped():
    """An invalid resize (matching ArcSession.resize_output_grid's ValueError
    guard) must leave the last-known grid untouched rather than crash the
    reconstruction or silently null it out."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 2], [3, 4]]}])
    tool_trace = [
        _set_grid_call([[1, 2], [3, 4]]),
        _resize_call(0, 5),  # invalid: height < 1
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert result["arc_per_test"][0]["correct"] is True


def test_scorer_reconstruction_resize_native_unprefixed_tool_names():
    """The native `arc_resize_output_grid` tool name must be recognized too,
    mirroring the set/submit tool-name coverage tested elsewhere."""
    evaluator = ArcSessionEvaluator()
    task = _task_with_tests([{"input": [[0]], "output": [[1, 2, 0], [3, 4, 0], [0, 0, 0]]}])
    tool_trace = [
        _set_grid_call([[1, 2], [3, 4]]),
        _resize_call(3, 3),
        _submit_call(),
    ]
    result = evaluator.evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "arc_submit_trial_results": [
                {"status": "ok", "trial_count": 1, "trials_remaining": 1, "test_index": 0},
            ],
            "tool_trace": tool_trace,
        },
    )
    assert result["resolved"] is True
    assert result["native_score"] == 1.0
    assert result["arc_per_test"][0]["correct"] is True
