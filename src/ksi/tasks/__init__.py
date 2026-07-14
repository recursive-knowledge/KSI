from . import path_validation as _path_validation  # noqa: F401  # wires TaskSourceSpec.validate_tasks_path hooks
from .loaders import (
    CLASSIFY_MAX_WORKERS,
    SUPPORTED_TASK_SOURCES,
    SWEBENCH_CATEGORIES,
    classify_task_with_llm,
    classify_tasks,
    load_categories_json,
    load_eval_records_for_source,
    load_tasks_for_source,
)
from .registry import (
    TaskSourceSpec,
    get_spec,
    register_task_source,
    resolve_source,
    supported_task_sources,
    upstream_strict_task_sources,
)

__all__ = [
    "CLASSIFY_MAX_WORKERS",
    "SWEBENCH_CATEGORIES",
    "SUPPORTED_TASK_SOURCES",
    "TaskSourceSpec",
    "classify_task_with_llm",
    "classify_tasks",
    "get_spec",
    "load_categories_json",
    "load_eval_records_for_source",
    "load_tasks_for_source",
    "register_task_source",
    "resolve_source",
    "supported_task_sources",
    "upstream_strict_task_sources",
]
