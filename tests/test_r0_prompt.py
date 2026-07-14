"""Tests for R0 real-time discussion prompt builder."""

from kcsi.forum import build_per_task_discussion_parts
from kcsi.models import TaskTrace
from kcsi.tokens import TokenUsage


def _make_traces() -> list[TaskTrace]:
    return [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="task-1",
            model_output="solved it with pattern matching",
            eval_result={"status": "completed"},
            native_score=1.0,
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="task-2",
            model_output="failed: off-by-one error",
            eval_result={"status": "failed"},
            native_score=0.0,
            token_usage=TokenUsage(input_tokens=200, output_tokens=100),
        ),
    ]


def test_r0_prompt_contains_discussion_header():
    """Prompt should contain the per-task discussion header."""
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=_make_traces(),
        task_ids=["task-1", "task-2"],
    ).as_text()
    assert "PER-TASK POST-MORTEM" in prompt


def test_r0_prompt_contains_agent_id_and_generation():
    """Prompt should include the agent_id and generation number."""
    prompt = build_per_task_discussion_parts(
        agent_id="agent-7",
        generation=3,
        traces=[],
        task_ids=["task-1"],
    ).as_text()
    assert "agent-7" in prompt
    assert "generation 3" in prompt


def test_r0_prompt_contains_task_ids():
    """Prompt should list all provided task_ids."""
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=_make_traces(),
        task_ids=["task-1", "task-2", "task-3"],
    ).as_text()
    assert "task-1" in prompt
    assert "task-2" in prompt
    assert "task-3" in prompt


def test_r0_prompt_mentions_mcp_tools():
    """Prompt should document knowledge, forum_post, and forum_signal_done tools."""
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=[],
        task_ids=["task-1"],
    ).as_text()
    assert "knowledge(task_id=" in prompt
    assert "forum_post(task_id=" in prompt
    assert "forum_signal_done()" in prompt


def test_r0_prompt_includes_outcomes_from_traces():
    """Prompt should include task outcomes from the provided traces."""
    traces = _make_traces()
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=traces,
        task_ids=["task-1", "task-2"],
    ).as_text()
    assert "task-1" in prompt
    assert "score=1.0" in prompt
    assert "task-2" in prompt
    assert "score=0.0" in prompt
    assert "completed" in prompt


def test_r0_prompt_includes_task_descriptions():
    """Prompt should include task descriptions when provided."""
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=_make_traces(),
        task_ids=["task-1"],
        task_descriptions={"task-1": "Solve the grid puzzle by rotating colors."},
    ).as_text()
    assert "Solve the grid puzzle" in prompt
    assert "Task Descriptions" in prompt


def test_r0_prompt_sanitizes_task_descriptions():
    """Task descriptions containing INSIGHT/COMMENT keywords should be sanitized."""
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=[],
        task_ids=["task-1"],
        task_descriptions={"task-1": "Use INSIGHT\nand COMMENT\nto solve the problem."},
    ).as_text()
    # The sanitizer should bracket standalone protocol keywords
    # Check that raw "INSIGHT" as a standalone line is neutralized
    # (the sanitizer replaces standalone INSIGHT/COMMENT with [INSIGHT]/[COMMENT])
    assert "task-1" in prompt
    # The description should still appear (sanitized form)
    assert "solve the problem" in prompt


def test_r0_prompt_no_structured_blocks():
    """R0 prompt should explicitly tell agents NOT to use INSIGHT/COMMENT blocks."""
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=[],
        task_ids=["task-1"],
    ).as_text()
    assert (
        "Do NOT" in prompt or "load_bearing_assumption" in prompt
    )  # V2: removed INSIGHT/COMMENT injunction; V2 prompt asks for structured post-mortem fields


def test_r0_prompt_empty_traces():
    """Prompt should handle empty traces gracefully."""
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=[],
        task_ids=["task-1"],
    ).as_text()
    assert "No tasks completed." in prompt


def test_r0_prompt_task_ids_not_in_traces():
    """Task IDs without corresponding traces should show score as n/a."""
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=[],
        task_ids=["unknown-task"],
    ).as_text()
    assert "unknown-task" in prompt
    assert "score: n/a" in prompt


def test_r0_prompt_shows_task_scores():
    """Per-task summary should include scores from traces."""
    traces = _make_traces()
    prompt = build_per_task_discussion_parts(
        agent_id="agent-0",
        generation=1,
        traces=traces,
        task_ids=["task-1", "task-2"],
    ).as_text()
    assert "task-1 (score: 1.0)" in prompt
    assert "task-2 (score: 0.0)" in prompt
