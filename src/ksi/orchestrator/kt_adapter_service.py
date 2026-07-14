"""KT adapter-transfer memo service.

Extracted verbatim from ``GenerationalOrchestrator`` (engine.py): the
generation-loop orchestrator carried ~200 LOC of knowledge-transfer
adapter-memo logic (build / repair / per-(generation, task) memoization) that
is used only by the execution phase. It now lives here behind an explicit,
injected ``KtAdapterService`` so the engine stays a sequencer and the KT-memo
logic is independently testable.

Behavior is preserved byte-for-byte, including the polyglot information-parity
guard, the JSON-repair retry, the deterministic asset fallback, and the auth
fail-fast. The service is constructed with the engine's ``_llm_call`` bound
method and its ``TokenAccumulator`` so token accounting continues to land under
``agent_id`` with the ``kt_adapter`` / ``kt_adapter_repair`` phase labels.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from typing import Any, Callable

from ..discussion.prompts import extract_json
from ..errors import AuthenticationFailure
from ..errors import is_auth_error as _is_auth_error
from ..models import AgentState, TaskSpec
from ..prompts.kt_adapter import build_kt_adapter_prompts
from ..tokens import LLMResponse, TokenAccumulator
from . import kt_adapter as _kt_adapter  # deterministic KT-adapter helpers

log = logging.getLogger(__name__)


class KtAdapterService:
    """Build and memoize per-task KT adapter-transfer memos.

    Injected dependencies:
      - ``llm_call``: the engine's ``_llm_call`` (``system``/``user``/``context``
        keyword call returning an :class:`LLMResponse`).
      - ``accumulator_getter``: returns the engine's *current*
        :class:`TokenAccumulator`, so KT LLM cost is recorded under the
        ``kt_adapter`` / ``kt_adapter_repair`` phases. Late-bound (a getter,
        not the object captured at construction) because ``run()`` reassigns
        ``engine.accumulator`` each run; capturing the ctor-time instance would
        orphan the recordings — matching the ``llm_call`` late-binding.
    """

    def __init__(
        self,
        *,
        llm_call: Callable[..., LLMResponse],
        accumulator_getter: Callable[[], TokenAccumulator],
    ) -> None:
        self._llm_call = llm_call
        self._accumulator_getter = accumulator_getter
        # The cross-task bundle is shared across all agents in a generation, so
        # a memo depends only on (generation, task_id); cache to avoid N agents
        # × 1 LLM call producing near-identical memos.
        self._memo_cache: dict[tuple[int, str], dict[str, Any]] = {}
        self._memo_lock = threading.Lock()

    def get_or_build_memo(
        self,
        *,
        generation: int,
        agent: AgentState,
        task: TaskSpec,
        cross_task: dict[str, Any],
    ) -> dict[str, Any] | None:
        # The cross-task bundle is shared across all agents in a generation,
        # so the adapter memo depends only on (generation, task_id). Cache it
        # to avoid N agents × 1 LLM call producing near-identical memos.
        cache_key = (int(generation), str(task.id))
        with self._memo_lock:
            cached = self._memo_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        memo = self._build_memo(generation=generation, agent=agent, task=task, cross_task=cross_task)
        if isinstance(memo, dict) and memo:
            with self._memo_lock:
                self._memo_cache.setdefault(cache_key, copy.deepcopy(memo))
        return memo

    def _build_memo(
        self,
        *,
        generation: int,
        agent: AgentState,
        task: TaskSpec,
        cross_task: dict[str, Any],
    ) -> dict[str, Any] | None:
        asset_payload = _kt_adapter.adapter_bundle_payload(cross_task)
        if not asset_payload:
            return None
        task_source = str((task.metadata or {}).get("task_source") or "").strip().lower()
        # Information parity is now enforced at the source: adapter_task_payload
        # strips ARC test outputs (passes test inputs only) and omits polyglot's
        # hidden test_files/build_files, so the memo payload is a subset of what
        # the solver sees. See src/ksi/memory/parity.py and the parity contract
        # test in tests/test_kt_adapter_parity.py.
        task_payload = _kt_adapter.adapter_task_payload(task)
        system, user = build_kt_adapter_prompts(
            task_source=task_source,
            task_payload=task_payload,
            asset_payload=asset_payload,
        )
        try:
            resp = self._llm_call(
                system=system,
                user=user,
                context={
                    "phase": "kt_adapter",
                    "generation": generation,
                    "agent_id": agent.id,
                    "task_id": task.id,
                },
            )
            raw, usage = resp.text, resp.usage
            self._accumulator_getter().record_lifecycle(generation, agent.id, "kt_adapter", usage)
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                try:
                    parsed = extract_json(raw)
                except Exception:
                    repaired = self._repair_output(
                        generation=generation,
                        agent=agent,
                        task=task,
                        system=system,
                        user=user,
                        raw_output=raw,
                        parse_exc=exc,
                    )
                    if repaired:
                        return repaired
                    fallback = _kt_adapter.build_fallback_memo(task=task, cross_task=cross_task)
                    if fallback:
                        log.warning(
                            "[ENGINE] kt adapter fell back to deterministic memo for task=%s after parse failure",
                            task.id,
                        )
                    return fallback
            memo = _kt_adapter.normalize_memo(parsed)
            if memo:
                memo["_memo_source"] = "adapter_llm"
                return memo
            fallback = _kt_adapter.build_fallback_memo(task=task, cross_task=cross_task)
            if fallback:
                log.warning(
                    "[ENGINE] kt adapter produced unusable memo for task=%s; using deterministic asset fallback",
                    task.id,
                )
            return fallback
        except Exception as exc:
            if _is_auth_error(exc):
                log.error(
                    "[ENGINE] LLM auth failure during kt_adapter — aborting run. agent=%s task=%s error=%s",
                    agent.id,
                    task.id,
                    exc,
                )
                raise AuthenticationFailure(f"LLM authentication failed for kt_adapter: {exc}") from exc
            log.warning("[ENGINE] kt adapter generation failed for task=%s: %s", task.id, exc)
            fallback = _kt_adapter.build_fallback_memo(task=task, cross_task=cross_task)
            if fallback:
                log.warning(
                    "[ENGINE] kt adapter LLM call failed for task=%s; using deterministic asset fallback",
                    task.id,
                )
                return fallback
            return None

    def _repair_output(
        self,
        *,
        generation: int,
        agent: AgentState,
        task: TaskSpec,
        system: str,
        user: str,
        raw_output: str,
        parse_exc: Exception,
    ) -> dict[str, Any] | None:
        prior = raw_output or ""
        max_chars = 8000
        if len(prior) > max_chars:
            prior = prior[:max_chars] + "\n... [truncated for repair retry] ...\n"
        repair_user = (
            f"{user}\n\n"
            "---\n"
            "REPAIR CONTEXT:\n"
            "Your previous response failed to parse as JSON.\n"
            f"Parse error: {parse_exc!r}\n\n"
            f"Previous output (possibly truncated):\n{prior}\n\n"
            "Re-emit ONLY one valid JSON object matching the required schema from the system prompt. "
            "No prose. No markdown fences. No commentary."
        )
        try:
            retry_resp = self._llm_call(
                system=system,
                user=repair_user,
                context={
                    "phase": "kt_adapter_repair",
                    "generation": generation,
                    "agent_id": agent.id,
                    "task_id": task.id,
                },
            )
            retry_raw, retry_usage = retry_resp.text, retry_resp.usage
            self._accumulator_getter().record_lifecycle(generation, agent.id, "kt_adapter_repair", retry_usage)
            try:
                repaired = json.loads(retry_raw)
            except (json.JSONDecodeError, ValueError):
                repaired = extract_json(retry_raw)
            memo = _kt_adapter.normalize_memo(repaired)
            if memo:
                memo["_memo_source"] = "adapter_repair"
                log.info("[ENGINE] kt adapter repaired output for task=%s after parse error", task.id)
            return memo
        except Exception as exc:
            log.warning(
                "[ENGINE] kt adapter repair failed for task=%s (orig parse=%r, repair=%r)",
                task.id,
                parse_exc,
                exc,
            )
            return None
