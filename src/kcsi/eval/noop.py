from __future__ import annotations

from dataclasses import dataclass

from ..models import TaskSpec


@dataclass
class NoopEvaluator:
    def evaluate(self, *, task: TaskSpec, model_output: str, **kwargs) -> dict:
        return {
            "status": "not_evaluated",
            "instance_id": task.id,
            "native_score": 0.0,
            "resolved": False,
        }
