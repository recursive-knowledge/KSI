"""Population sizing for the maintained task-driven path."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import GenerationConfig


def _normalize_count(value: int) -> int:
    return max(0, int(value))


@dataclass(frozen=True)
class TaskStrategy:
    """Task mode sizes the population directly from remaining tasks."""

    def next_agent_count(
        self,
        generation: int,
        remaining_tasks: int,
    ) -> int:
        return _normalize_count(remaining_tasks)


def make_strategy(config: GenerationConfig) -> TaskStrategy:
    """Construct the maintained task-driven population strategy."""
    return TaskStrategy()
