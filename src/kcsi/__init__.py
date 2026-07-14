"""KCSI (knowledge-centric self-improvement) orchestration package.

Public surface
--------------
- :func:`kcsi.run` — programmatic entry point (see :mod:`kcsi.api`).
- :class:`~kcsi.orchestrator.engine.GenerationalOrchestrator` — the engine.
- Config / data types: :class:`GenerationConfig`, :class:`TaskSpec`, :class:`TaskTrace`, ...
- Extension registration: :func:`register_evaluator`, :func:`register_runtime`,
  :func:`register_task_source`, and :func:`register_strategy` — register new
  seams without editing core.
- Programmatic construction: :func:`build_evaluator` / :func:`build_runtime`
  build a registered component from CLI defaults + keyword overrides, no
  argparse ``Namespace`` required.

Build the forum/distillation LLM caller with
``from kcsi.runtime.llm import build_llm_caller``; runtime classes live under
:mod:`kcsi.runtime` and reference benchmark evaluator classes live under
:mod:`kcsi.benchmarks`.
"""

from .api import run
from .errors import KcsiError
from .eval import EvaluatorSpec, build_evaluator, register_evaluator
from .models import (
    AgentState,
    ArcEvalResult,
    Assignment,
    EvalResult,
    GenerationConfig,
    Insight,
    TaskSpec,
    TaskTrace,
)
from .orchestrator import StrategySpec, register_strategy
from .orchestrator.engine import GenerationalOrchestrator
from .runtime import RuntimeSpec, build_runtime, register_runtime
from .tasks import TaskSourceSpec, register_task_source

__all__ = [
    "AgentState",
    "ArcEvalResult",
    "Assignment",
    "EvalResult",
    "GenerationConfig",
    "Insight",
    "TaskSpec",
    "TaskTrace",
    "GenerationalOrchestrator",
    "run",
    "KcsiError",
    "register_evaluator",
    "register_runtime",
    "register_task_source",
    "register_strategy",
    "EvaluatorSpec",
    "RuntimeSpec",
    "StrategySpec",
    "TaskSourceSpec",
    "build_evaluator",
    "build_runtime",
]
