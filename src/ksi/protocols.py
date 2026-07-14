"""Shared protocol definitions for the ksi package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from .models import AgentState, Insight, TaskSpec, TaskTrace
from .runtime.types import RuntimeResult
from .tokens import LLMResponse, TokenUsage, TokenUsageDict


class ForumMessageContent(TypedDict, total=False):
    """Content payload recorded by :meth:`PersistenceObserver.on_forum_message`.

    ``total=False``: all keys are optional. The keys below are those emitted
    today (forum agent error reporting, ``message_type="error"``); extend this
    TypedDict when a new ``message_type`` records additional fields.
    """

    phase: str
    error: str
    error_type: str


class LLMCaller(Protocol):
    # Callers that support provider structured output advertise it with this
    # class attribute. When passed ``json_schema=`` such a caller populates
    # ``LLMResponse.parsed`` with the provider-validated dict (or ``None`` if it
    # could not be parsed). Callers without structured-output support omit the
    # attribute (defaults False via the engine's ``getattr(..., False)`` gate)
    # and never receive a ``json_schema``; ``LLMResponse.parsed`` is then always
    # ``None``.
    supports_json_schema: bool

    def call(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None, **kwargs: Any
    ) -> LLMResponse: ...


class RuntimeExecutor(Protocol):
    def run_task(
        self,
        *,
        generation: int,
        agent_id: str,
        task: TaskSpec,
        cross_task_shared_container: bool = False,
        cross_task_r1_callback: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> str | RuntimeResult: ...


class Evaluator(Protocol):
    # Return type is ``dict[str, Any]`` — the honest reflection of what every
    # concrete evaluator returns. The richer ``EvalResult`` TypedDict
    # (in ``models``) documents the cross-evaluator keys and is the declared type
    # of ``TaskTrace.eval_result`` for consumers that know the shape; evaluators
    # additionally emit evaluator-specific keys (e.g. terminal_bench_2 ``reward``,
    # swebench ``patch_source``) not enumerated there, so pinning the Protocol to
    # ``EvalResult`` would falsely reject conforming implementations.
    def evaluate(self, *, task: TaskSpec, model_output: str, **kwargs: Any) -> dict[str, Any]: ...


class PersistenceObserver(Protocol):
    def on_generation_start(self, *, generation: int, agents: list[AgentState]) -> None: ...
    def on_assignment(self, *, generation: int, assigned: dict[str, list[str]], total_tasks: int = 0) -> None: ...
    def on_task_status(self, *, generation: int, agent_id: str, task_id: str, status: str) -> None: ...
    def on_task_trace(self, trace: TaskTrace) -> None: ...
    def on_forum_message(
        self,
        *,
        generation: int,
        round_num: int,
        agent_id: str,
        message_type: str,
        content_json: ForumMessageContent,
        token_usage: TokenUsageDict,
    ) -> None: ...
    def on_native_memory(self, *, generation: int, agent_id: str, content: str) -> None: ...
    def on_insight(self, *, generation: int, agent_id: str, insight: Insight) -> None: ...
    def on_generation_end(self, *, generation: int, agents: list[AgentState]) -> None: ...
    def on_run_end(self, *, token_summary: TokenUsage) -> None: ...
