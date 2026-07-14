from . import registry  # noqa: F401  (import populates the evaluator REGISTRY)
from .noop import NoopEvaluator
from .registry import (
    EvaluatorSpec,
    build_evaluator,
    get_evaluator_spec,
    register_evaluator,
    resolve_evaluator,
    supported_evaluators,
)

# Retained name (3 internal importers); now the single source of truth is the registry.
SUPPORTED_EVALUATORS: tuple[str, ...] = supported_evaluators()

__all__ = [
    "NoopEvaluator",
    "SUPPORTED_EVALUATORS",
    "EvaluatorSpec",
    "register_evaluator",
    "resolve_evaluator",
    "get_evaluator_spec",
    "supported_evaluators",
    "build_evaluator",
]
