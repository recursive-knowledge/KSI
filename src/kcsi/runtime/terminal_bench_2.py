from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..models import TaskSpec
from ..tasks.registry import resolve_source
from .terminal_bench_2_trial import run_terminal_bench_2_trial
from .types import RuntimeResult

log = logging.getLogger(__name__)


@dataclass
class TerminalBench2Executor:
    """Thin runtime wrapper around the TB2 trial runner."""

    # The CLI always sets agent_mode="kcsi" (src/kcsi/cli.py); "oracle" /
    # "noop" / "command" remain reachable through run_terminal_bench_2_trial
    # for debug / oracle-validation flows but are not used in production.
    agent_mode: str = "kcsi"
    agent_command: str = ""
    output_root: str = ""
    keep_container: bool = False
    env: dict[str, str] = field(default_factory=dict)
    fallback_runtime: Any | None = None
    # Raw-attempts memory mode (ablation), mirroring
    # KcsiContainerExecutor.memory_seed_raw_attempts. When True, MEMORY.md is
    # built from ONLY the raw prior-attempt model_output + eval detail -- no
    # distilled bundles, no insights, no condensed-approach reflection.
    # Defaults False (preserves existing behavior).
    memory_seed_raw_attempts: bool = False

    def close(self) -> None:
        """Close the fallback runtime if one was provided."""
        if self.fallback_runtime is not None and hasattr(self.fallback_runtime, "close"):
            self.fallback_runtime.close()

    def run_task(
        self,
        *,
        generation: int,
        agent_id: str,
        task: TaskSpec,
        cross_task_shared_container: bool = False,
        cross_task_r1_callback: Callable[..., Any] | None = None,
        **kwargs,
    ) -> RuntimeResult:
        task_source = str((task.metadata or {}).get("task_source") or "").strip().lower()
        _spec = resolve_source(task_source)
        if _spec is None or not _spec.delegates_runtime:
            if self.fallback_runtime is None:
                raise ValueError(f"task {task.id!r} has task_source={task_source!r}, expected 'terminal_bench_2'")
            log.info(
                "Delegating non-TB2 task to fallback runtime task_id=%s task_source=%s",
                task.id,
                task_source,
            )
            return self.fallback_runtime.run_task(
                generation=generation,
                agent_id=agent_id,
                task=task,
                cross_task_shared_container=cross_task_shared_container,
                cross_task_r1_callback=cross_task_r1_callback,
                **kwargs,
            )
        result = run_terminal_bench_2_trial(
            task=task,
            agent_mode=self.agent_mode,
            agent_command=self.agent_command or None,
            output_dir=self.output_root or None,
            keep_container=self.keep_container,
            provider_env=self.env,
            generation=generation,
            agent_id=agent_id,
            seed_package=kwargs.get("agent_seed_package"),
            raw_mode=self.memory_seed_raw_attempts,
        )
        runtime_meta = dict(result.runtime_meta)
        runtime_meta.update(
            {
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task.id,
                "status": "success",
                "runner": "terminal_bench_2_executor",
            }
        )
        return RuntimeResult(
            output=(
                f"tb2 trial finished for {task.id}: reward={result.reward} "
                f"agent_exit={result.agent_exit_code} verifier_exit={result.verifier_exit_code}"
            ),
            tool_trace=result.tool_trace,
            runtime_meta=runtime_meta,
            token_usage=result.token_usage,
        )
