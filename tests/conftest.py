"""Shared pytest fixtures for mock runtime, evaluator, LLM, and task helpers."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ksi.models import TaskSpec
from ksi.runtime.types import RuntimeResult
from ksi.tokens import LLMResponse, TokenUsage

# Repo-root / fixtures anchors so test files stay depth-independent: a test
# can `from conftest import REPO_ROOT, FIXTURES_DIR` regardless of which
# tests/ subdir it lives in (conftest imports as a bare module via the
# pythonpath = ["tests", "."] setting). See issue #900.
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _build_mock_runtime(native_session_memory="transcript content here"):
    """Build a mock runtime returning a canned RuntimeResult."""
    rt = MagicMock()
    rt.run_task.return_value = RuntimeResult(
        output="Fixed the bug by changing line 42",
        tool_trace=[],
        runtime_meta={"native_session_memory": native_session_memory, "session_scope": "task"},
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )
    return rt


def _build_mock_evaluator(
    *,
    resolved: bool = True,
    native_score: float = 1.0,
    task_type: str = "swebench",
    score_by_task: dict[str, float] | None = None,
):
    """Build a mock evaluator.

    By default returns a uniformly-resolved result, which is convenient for
    pipeline-shape tests. Callers that want to exercise scoring logic should
    pass ``resolved=False`` / ``native_score=0.0``, or a ``score_by_task``
    dict keyed by ``task.id`` to vary the score per task. Without this knob
    the default fixture would mask grader regressions because every task is
    marked "solved" regardless of trace content.
    """
    ev = MagicMock()
    if score_by_task is not None:

        def _evaluate(trace=None, task=None, **_kwargs):
            tid = ""
            if task is not None:
                tid = getattr(task, "id", "") or ""
            if not tid and trace is not None:
                tid = getattr(trace, "task_id", "") or ""
            score = float(score_by_task.get(tid, native_score))
            return {
                "resolved": bool(score >= 1.0 if resolved is None else (resolved and score >= 1.0)),
                "native_score": score,
                "task_type": task_type,
            }

        ev.evaluate.side_effect = _evaluate
    else:
        ev.evaluate.return_value = {
            "resolved": resolved,
            "native_score": native_score,
            "task_type": task_type,
        }
    return ev


def _build_mock_llm():
    """Build a mock LLM with canned side-effect routing."""
    llm = MagicMock()

    def llm_side_effect(system, user, **kwargs):
        if "transferable insight" in user.lower():
            return json.dumps(
                {
                    "text": "Always verify cache invalidation after ORM changes",
                    "workstream": "django-orm",
                    "confidence": "high",
                }
            ), TokenUsage(input_tokens=200, output_tokens=30)
        elif "Available Tasks" in user or "claimed_tasks" in user.lower() or "workstream" in user.lower():
            return json.dumps({"claimed_tasks": ["task-0"]}), TokenUsage(input_tokens=50, output_tokens=20)
        elif "task_results" in user.lower() or "propose" in user.lower():
            return json.dumps(
                {
                    "insights": [{"text": "Use breakpoints", "workstream": "debugging", "confidence": "high"}],
                    "workstream_claim": "debugging",
                    "proposed_workstreams": ["debugging"],
                }
            ), TokenUsage(input_tokens=100, output_tokens=50)
        elif "bucket" in user.lower() or "cluster" in user.lower() or "distill" in user.lower():
            return json.dumps(
                {
                    "buckets": [
                        {"label": "general", "task_ids": ["task-0"], "insight_summary": "General debugging insights"}
                    ],
                }
            ), TokenUsage(input_tokens=150, output_tokens=60)
        elif "bundle" in user.lower() or "condense" in user.lower():
            return json.dumps(
                {
                    "bundle_summary": "Key debugging insights from this generation",
                    "shared_insight_bundle": [{"text": "Use breakpoints for debugging", "confidence": "high"}],
                }
            ), TokenUsage(input_tokens=120, output_tokens=50)
        else:
            return json.dumps(
                {
                    "proposals": [],
                    "workstream_claim": "debugging",
                }
            ), TokenUsage(input_tokens=80, output_tokens=40)

    def _llm_response_side_effect(system, user, **kwargs):
        text, usage = llm_side_effect(system, user, **kwargs)
        return LLMResponse(text=text, usage=usage)

    llm.call.side_effect = _llm_response_side_effect
    return llm


def _build_make_tasks(n: int) -> list[TaskSpec]:
    """Create n simple TaskSpec objects."""
    return [TaskSpec(id=f"task-{i}", repo="r", prompt=f"Fix bug {i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Pytest fixtures — each returns the *builder function* so callers can
# customise arguments:  ``rt = mock_runtime()`` or
# ``rt = mock_runtime(native_session_memory="")``.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_runtime():
    return _build_mock_runtime


@pytest.fixture
def mock_evaluator():
    return _build_mock_evaluator


@pytest.fixture
def mock_llm():
    return _build_mock_llm


@pytest.fixture
def make_tasks():
    return _build_make_tasks


@pytest.fixture
def structured_insight() -> dict:
    """A canonical structured Insight dict matching the cross-task
    distillation schema. Reused across test files to keep the round-trip
    / render contracts in sync.
    """
    return {
        "text": "When a full-width zero band splits the grid, partition by row-run signature.",
        "applies_when": "ARC tasks where one or more rows are entirely zero, splitting bands.",
        "does_not_apply_when": "Grids without zero rows or where zeros are sparse rather than banded.",
        "confidence": "high",
        "evidence": [
            {"task_id": "abc123", "post_id": 42, "quote": "Zero-band detected at row 7."},
            {"task_id": "def456", "post_id": 91, "quote": "Two zero rows split the grid into three bands."},
        ],
    }
