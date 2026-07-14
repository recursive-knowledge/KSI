"""Forum discussion prompt builders (per-task and cross-task)."""

from .prompt import (
    ForumPromptParts,
    build_cross_task_discussion_parts,
    build_per_task_discussion_parts,
)

__all__ = [
    "ForumPromptParts",
    "build_cross_task_discussion_parts",
    "build_per_task_discussion_parts",
]
