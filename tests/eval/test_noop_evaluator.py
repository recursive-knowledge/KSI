"""Unit tests for NoopEvaluator."""

from __future__ import annotations

from kcsi.eval.noop import NoopEvaluator
from kcsi.models import TaskSpec


class TestNoopEvaluator:
    def test_returns_zero_score(self):
        evaluator = NoopEvaluator()
        task = TaskSpec(id="test-1", prompt="test prompt")
        result = evaluator.evaluate(task=task, model_output="some output")
        assert result["native_score"] == 0.0

    def test_returns_not_resolved(self):
        evaluator = NoopEvaluator()
        task = TaskSpec(id="test-1", prompt="test prompt")
        result = evaluator.evaluate(task=task, model_output="some output")
        assert result["resolved"] is False

    def test_returns_instance_id(self):
        evaluator = NoopEvaluator()
        task = TaskSpec(id="my-task-42", prompt="fix this")
        result = evaluator.evaluate(task=task, model_output="patch")
        assert result["instance_id"] == "my-task-42"

    def test_returns_status_not_evaluated(self):
        evaluator = NoopEvaluator()
        task = TaskSpec(id="t1", prompt="p")
        result = evaluator.evaluate(task=task, model_output="out")
        assert result["status"] == "not_evaluated"
