"""Public programmatic entry point for kcsi.

The CLI (`kcsi.cli`) is one front door; this module is the other. It lets a
researcher drive a generational run from Python or a notebook without going
through argparse:

    import kcsi
    from kcsi.eval import NoopEvaluator
    from kcsi.runtime import KcsiContainerExecutor
    from kcsi.runtime.llm import build_llm_caller

    config = kcsi.GenerationConfig(num_generations=1, num_agents=1)
    llm = build_llm_caller(provider="anthropic", model="claude-sonnet-4-6")
    runtime = KcsiContainerExecutor(command=[...], working_dir=".")
    traces = kcsi.run(config, tasks, runtime=runtime, evaluator=NoopEvaluator(), llm=llm)

`run` is a thin, behavior-preserving wrapper around
`GenerationalOrchestrator` — it constructs the orchestrator with the same
arguments the CLI uses and calls ``.run(tasks=...)``.
Construct ``runtime`` / ``evaluator`` / ``llm`` yourself (the registry factories
and ``build_llm_caller`` are the building blocks); this keeps the API explicit
about what each run depends on.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from .distillation._removed_env import assert_no_removed_channel_env
from .models import GenerationConfig, TaskSpec, TaskTrace
from .orchestrator.engine import GenerationalOrchestrator

if TYPE_CHECKING:
    from .protocols import Evaluator, LLMCaller, PersistenceObserver, RuntimeExecutor

__all__ = ["run"]


def run(
    config: GenerationConfig,
    tasks: list[TaskSpec],
    *,
    runtime: "RuntimeExecutor",
    evaluator: "Evaluator",
    llm: "LLMCaller",
    persistence: "PersistenceObserver | None" = None,
    working_dir: str = ".",
) -> list[TaskTrace]:
    """Run a generational self-improvement loop and return the task traces.

    This is the programmatic equivalent of invoking the CLI: it constructs a
    `GenerationalOrchestrator` with the same arguments and executes ``tasks``.
    The orchestrator construction and run are identical to the CLI path;
    CLI-only conveniences are NOT applied — provider
    profile / environment loading and signal handling are the caller's
    responsibility, so set any such environment yourself before calling ``run``.

    .. important::
        Unlike the CLI, ``run`` does NOT derive a default knowledge DB path.
        ``config.knowledge_db_path`` defaults to ``""`` (see
        `GenerationConfig`), and the orchestrator SILENTLY disables the entire
        knowledge substrate when it is empty — task attempts are not recorded,
        and retrieval, distillation, and cross-generation seeding never run
        (only raw per-generation attempts happen). The CLI never hits this
        because it always resolves a per-experiment default path
        (``cli._prepare_knowledge_db_path``). To get the knowledge loop, set
        ``config.knowledge_db_path`` to a writable ``*_knowledge.sqlite`` path
        (one per concurrent experiment). ``run`` emits a ``UserWarning`` when
        the path is empty so the degrade is visible; pass a path to silence it.

    Two further CLI-only conveniences are the caller's responsibility (no code
    is run for them here): loading a provider profile / auth environment (the
    container host validates provider auth and raises if it is missing — set
    ``MODEL_PROVIDER`` / ``*_API_KEY`` / etc. yourself), and any task-source
    runtime delegation the CLI wires up (e.g. TB2). See ``docs/programmatic_api.md``.

    Parameters
    ----------
    config:
        The run configuration (generations, agents, paths, ...).
    tasks:
        The tasks to attempt.
    runtime:
        How agents execute (e.g. ``KcsiContainerExecutor``). Build via the
        runtime registry or construct directly.
    evaluator:
        How attempts are scored (e.g. ``ArcSessionEvaluator``). Build via the
        evaluator registry or construct directly.
    llm:
        The forum/distillation LLM caller (see
        `kcsi.runtime.llm.build_llm_caller`).
    persistence:
        Optional `PersistenceObserver` for transcripts, token accounting, and
        lifecycle callbacks. ``None`` disables persistence.
    working_dir:
        Project root used to resolve runtime/runner paths. Defaults to ``"."``.

    Returns
    -------
    list[TaskTrace]
        One trace per executed task attempt.
    """
    assert_no_removed_channel_env()
    # The orchestrator disables the whole knowledge substrate (attempts,
    # retrieval, distillation, seeding) when knowledge_db_path is empty
    # (engine._initialize_stores guards on `if knowledge_db_path:`). The CLI
    # never trips this — it resolves a default path — so surface the silent
    # degrade instead of letting the loop quietly become a raw-attempts run.
    if not str(getattr(config, "knowledge_db_path", "") or "").strip():
        warnings.warn(
            "kcsi.run: config.knowledge_db_path is empty, so the knowledge loop "
            "is DISABLED — task attempts are not recorded and retrieval, "
            "distillation, and cross-generation seeding will not run. Set "
            "config.knowledge_db_path to a writable '*_knowledge.sqlite' path "
            "(one per concurrent experiment) to enable it. The CLI derives this "
            "default automatically; the programmatic API does not.",
            UserWarning,
            stacklevel=2,
        )
    orchestrator = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=persistence,
        working_dir=working_dir,
    )
    return orchestrator.run(tasks=tasks)
