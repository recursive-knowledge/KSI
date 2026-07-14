"""Tests for the cross-task R0->R1 shared-container feature.

Covers three layers:

* **Coordinator unit tests** — :class:`_CrossTaskR1Coordinator` happy path
  (all sentinels arrive, drain runs, per-agent prompts get computed) and
  graceful degrade (some agents miss the deadline).
* **Engine wiring** — when the feature flag is on, the cross-task forum service
  routes through its shared-container path and the runtime sees the
  ``cross_task_shared_container=True`` kwarg + a callable
  ``cross_task_r1_callback``. Per-round token usage from the container
  envelope's ``cross_task_round_<n>_result`` blocks gets attributed to
  the right ``token_phases`` slugs.
* **CLI flag** — the ``--cross-task-shared-container`` argparse plumbing.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kcsi.models import GenerationConfig, TaskTrace
from kcsi.orchestrator.engine import (
    ForumValidationError,
    GenerationalOrchestrator,
    NoopPersistence,
    _coerce_round_usage,
    _CrossTaskR1Coordinator,
)
from kcsi.runtime.barrier import BarrierEvent
from kcsi.runtime.types import RuntimeResult
from kcsi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import cross_task_forum

# ── _CrossTaskR1Coordinator unit tests ─────────────────────────────────────


def _make_event(agent_id: str, payload_extra: dict | None = None) -> BarrierEvent:
    payload = {"agent_id": agent_id, "schema": "cross_task_r1.v1"}
    if payload_extra:
        payload.update(payload_extra)
    return BarrierEvent(
        sentinel_path=Path(f"/tmp/.barrier.cross_task_r1.{agent_id}.ready"),
        response_path=Path(f"/tmp/.barrier.cross_task_r1.{agent_id}.response"),
        payload=payload,
    )


def test_coordinator_happy_path_all_sentinels_arrive():
    """All expected agents signal -> coordinator drains, builds prompts,
    every agent's BarrierWatcher callback returns the per-agent response."""
    drain_calls = []
    prompt_calls = []

    def drain():
        drain_calls.append(time.time())

    def builder(agent_id):
        prompt_calls.append(agent_id)
        return f"R1 prompt for {agent_id}"

    coord = _CrossTaskR1Coordinator(
        expected_agent_ids=["a", "b"],
        prompt_builder=builder,
        timeout_sec=5.0,
        on_drain=drain,
    )
    coord.start()

    results: dict[str, dict] = {}

    def call_a():
        results["a"] = coord.on_sentinel(_make_event("a"))

    def call_b():
        results["b"] = coord.on_sentinel(_make_event("b"))

    t_a = threading.Thread(target=call_a, daemon=True)
    t_b = threading.Thread(target=call_b, daemon=True)
    t_a.start()
    t_b.start()
    t_a.join(timeout=10.0)
    t_b.join(timeout=10.0)

    assert "a" in results and "b" in results
    assert results["a"]["r1_prompt_text"] == "R1 prompt for a"
    assert results["b"]["r1_prompt_text"] == "R1 prompt for b"
    assert results["a"]["agent_id"] == "a"
    assert results["b"]["agent_id"] == "b"
    assert len(drain_calls) == 1, "drain must run exactly once"
    assert sorted(prompt_calls) == ["a", "b"]
    assert coord.timed_out_agents == set()
    coord.stop()


def test_coordinator_timeout_some_agents_miss():
    """When the coordinator timeout fires before all agents signal, the
    arrived agents still get responses; missing ones are tracked but
    naturally never invoked the callback."""
    drain_calls = []

    coord = _CrossTaskR1Coordinator(
        expected_agent_ids=["a", "b", "c"],
        prompt_builder=lambda aid: f"R1 for {aid}",
        timeout_sec=0.5,
        on_drain=lambda: drain_calls.append(1),
    )
    coord.start()

    # Only "a" signals; "b" and "c" never do.
    result_a = coord.on_sentinel(_make_event("a"))

    assert result_a["r1_prompt_text"] == "R1 for a"
    assert "b" in coord.timed_out_agents
    assert "c" in coord.timed_out_agents
    assert "a" not in coord.timed_out_agents
    assert len(drain_calls) == 1
    coord.stop()


def test_coordinator_stop_waits_for_drain_to_finish():
    """stop() must not return while the coordinator drain is still about to mutate health."""
    release_drain = threading.Event()
    drain_finished = []

    def drain():
        release_drain.wait(timeout=2.0)
        drain_finished.append(1)

    coord = _CrossTaskR1Coordinator(
        expected_agent_ids=["a"],
        prompt_builder=lambda aid: f"R1 for {aid}",
        timeout_sec=5.0,
        on_drain=drain,
    )
    coord.start()
    join_timeouts = []
    orig_join = coord._coordinator_thread.join

    def join_spy(timeout=None):
        join_timeouts.append(timeout)
        return orig_join(timeout=timeout)

    coord._coordinator_thread.join = join_spy  # type: ignore[method-assign]
    release_drain.set()
    coord.stop()

    assert drain_finished == [1]
    # stop() now bounds the join at timeout_sec + 30 (was an unbounded join)
    # so a stuck forum-bus drain can't hang the engine thread indefinitely.
    assert join_timeouts == [coord._timeout_sec + 30]


def test_coordinator_propagates_builder_exception_per_agent():
    """A prompt_builder exception for one agent doesn't block the others
    from receiving a response — the failed agent just gets {error: ...}."""

    def builder(aid):
        if aid == "boom":
            raise ValueError("synthetic builder failure")
        return f"R1 for {aid}"

    coord = _CrossTaskR1Coordinator(
        expected_agent_ids=["ok", "boom"],
        prompt_builder=builder,
        timeout_sec=2.0,
    )
    coord.start()

    results: dict[str, dict] = {}

    def call(aid):
        results[aid] = coord.on_sentinel(_make_event(aid))

    threads = [threading.Thread(target=call, args=(aid,), daemon=True) for aid in ["ok", "boom"]]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert results["ok"]["r1_prompt_text"] == "R1 for ok"
    assert "error" in results["boom"]
    assert "synthetic builder failure" in results["boom"]["error"]
    coord.stop()


def test_coerce_round_usage_handles_missing_keys():
    """``_coerce_round_usage`` must default to zero on missing/non-dict
    inputs without raising — defensive parse for envelope shapes."""
    assert _coerce_round_usage(None) == TokenUsage()
    assert _coerce_round_usage({}) == TokenUsage()
    assert _coerce_round_usage({"tokenUsage": None}) == TokenUsage()
    block = {
        "tokenUsage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 80,
        },
    }
    usage = _coerce_round_usage(block)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cache_creation_input_tokens == 20
    assert usage.cache_read_input_tokens == 80


def test_coerce_round_usage_handles_garbage_values():
    """Non-numeric values must drop silently to 0 — never raise."""
    block = {
        "tokenUsage": {
            "input_tokens": "lots",
            "output_tokens": None,
            "cache_creation_input_tokens": "",
            "cache_read_input_tokens": 42,
        },
    }
    usage = _coerce_round_usage(block)
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cache_creation_input_tokens == 0
    assert usage.cache_read_input_tokens == 42


# ── Engine-wiring tests ───────────────────────────────────────────────────


def _make_orch_with_shared_container(tmp_path) -> tuple[GenerationalOrchestrator, MagicMock]:
    runtime = MagicMock()
    db_path = str(tmp_path / "knowledge.sqlite")
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"resolved": True, "native_score": 1.0}
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps({"claimed_tasks": []}),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        knowledge_db_path=db_path,
    )
    config.cross_task_forum_rounds = 2
    config.cross_task_shared_container = True
    config.cross_task_forum_timeout_sec = 60
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    return orch, runtime


def test_engine_routes_to_shared_container_when_flag_on(tmp_path):
    """When ``cross_task_shared_container=True`` and rounds>=2, the engine
    invokes ``run_task`` once per agent (NOT once per agent per round)
    AND passes ``cross_task_shared_container=True`` and a
    ``cross_task_r1_callback`` callable through to the runtime."""
    orch, runtime = _make_orch_with_shared_container(tmp_path)

    captured_kwargs: list[dict] = []
    captured_task_metadata: list[dict] = []

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        captured_kwargs.append(kwargs)
        captured_task_metadata.append(dict(task.metadata or {}))
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    runtime.run_task.side_effect = fake_run_task

    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    # 2 agents x 1 dispatch (NOT 2 dispatches per agent for legacy 2-round).
    assert runtime.run_task.call_count == 2, (
        f"expected 2 dispatches (one per agent in shared mode), got {runtime.run_task.call_count}"
    )
    for kw in captured_kwargs:
        assert kw.get("cross_task_shared_container") is True, (
            f"shared-container kwarg not threaded; saw kwargs={list(kw.keys())}"
        )
        assert callable(kw.get("cross_task_r1_callback")), (
            f"cross_task_r1_callback not threaded as callable; got {kw.get('cross_task_r1_callback')!r}"
        )
    assert captured_task_metadata
    for metadata in captured_task_metadata:
        assert metadata.get("forum_task_ids") == ["t1", "t2"]


def test_engine_legacy_path_when_flag_off(tmp_path):
    """When the flag is off, the engine takes the legacy two-dispatch
    path (one per agent per round)."""
    orch, runtime = _make_orch_with_shared_container(tmp_path)
    orch.config.cross_task_shared_container = False

    runtime.run_task.return_value = RuntimeResult(
        output="",
        tool_trace=[],
        runtime_meta={},
        token_usage=TokenUsage(input_tokens=1, output_tokens=1),
    )

    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    # Legacy path: 2 agents x 2 rounds = 4 dispatches.
    assert runtime.run_task.call_count == 4, (
        f"legacy path should dispatch per-agent per-round (4); got {runtime.run_task.call_count}"
    )


def test_engine_persists_per_round_token_usage_when_envelope_has_both_blocks(tmp_path):
    """When the container envelope ships ``cross_task_round_0_result`` and
    ``cross_task_round_1_result`` in runtime_meta, the engine attributes
    R0 tokens to ``cross_task_forum_round_0`` and R1 tokens to
    ``cross_task_forum_round_1`` separately."""
    orch, runtime = _make_orch_with_shared_container(tmp_path)

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        # Simulate a successful R0+R1 envelope.
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={
                "cross_task_round_0_result": {
                    "resultText": "",
                    "toolTrace": [],
                    "tokenUsage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_creation_input_tokens": 200,
                        "cache_read_input_tokens": 0,
                    },
                    "signaledDone": True,
                },
                "cross_task_round_1_result": {
                    "resultText": "",
                    "toolTrace": [],
                    "tokenUsage": {
                        "input_tokens": 30,
                        "output_tokens": 15,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 350,
                    },
                    "signaledDone": True,
                },
                "cross_task_shared_container_meta": {
                    "enabled": True,
                    "r1_captured": True,
                },
            },
            token_usage=TokenUsage(
                input_tokens=130,
                output_tokens=65,
                cache_creation_input_tokens=200,
                cache_read_input_tokens=350,
            ),
        )

    runtime.run_task.side_effect = fake_run_task

    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    cross_task_forum(orch, generation=1, traces=traces)

    # Inspect lifecycle accumulator: it must show distinct R0 and R1
    # phase rows for each agent.
    lifecycle = orch.accumulator._entries  # type: ignore[attr-defined]  # internal but stable
    # Find the cross-task rows. Lifecycle entries use source_id "__lc:<call_type>".
    r0_rows = [(k, v) for k, v in lifecycle.items() if k[2] == "__lc:cross_task_forum_round_0"]
    r1_rows = [(k, v) for k, v in lifecycle.items() if k[2] == "__lc:cross_task_forum_round_1"]
    assert len(r0_rows) == 2, f"expected 2 R0 rows (one per agent), got {len(r0_rows)}"
    assert len(r1_rows) == 2, f"expected 2 R1 rows (one per agent), got {len(r1_rows)}"
    # Per-agent R0 input tokens should equal 100 (the per-round-block value),
    # NOT the 130 aggregate. This confirms the engine is using
    # ``_coerce_round_usage`` instead of the top-level token_usage when
    # the per-round blocks are present.
    for _, usage in r0_rows:
        assert usage.input_tokens == 100, f"R0 row should reflect per-round usage (100), got {usage.input_tokens}"
    for _, usage in r1_rows:
        assert usage.input_tokens == 30, f"R1 row should reflect per-round usage (30), got {usage.input_tokens}"


def test_engine_treats_r1_barrier_timeout_as_graceful_degrade_not_failure(tmp_path):
    """A missing R1 caused by a barrier timeout must NOT look like a failure.

    This is the documented contract (``_CrossTaskR1Coordinator``'s docstring
    and the TS-side "R1 absence is OK" comment): when the coordinator's own
    barrier times out waiting for a response, agents whose containers never
    got an R1 prompt back fall through to an R0-only envelope. R0 already
    ran to completion, so this is graceful degradation, not an error -- the
    engine must not count it against ``forum_agent_failures`` or raise.
    """
    orch, runtime = _make_orch_with_shared_container(tmp_path)

    runtime.run_task.return_value = RuntimeResult(
        output="",
        tool_trace=[],
        runtime_meta={
            "cross_task_shared_container_meta": {
                "enabled": True,
                "r1_captured": False,
                "note": "cross_task_r1: host barrier response not received within 30000ms",
                "timed_out": True,
            },
        },
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )

    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    # Must not raise.
    cross_task_forum(orch, generation=1, traces=traces)

    lifecycle = orch.accumulator._entries  # type: ignore[attr-defined]
    r0_rows = [k for k in lifecycle if k[2] == "__lc:cross_task_forum_round_0"]
    r1_rows = [k for k in lifecycle if k[2] == "__lc:cross_task_forum_round_1"]
    assert len(r0_rows) == 2
    assert len(r1_rows) == 0, f"R1 row must NOT be recorded when R1 block is absent; got {r1_rows}"
    # A graceful barrier timeout must not even create a phase-health bucket
    # for this generation (knowledge_phase_health_by_generation() only
    # records generations with at least one recorded failure).
    assert orch.knowledge_phase_health_by_generation().get(1, {}).get("forum_agent_failures", 0) == 0


def test_engine_treats_r1_genuine_error_as_shared_container_failure(tmp_path):
    """A missing R1 caused by a genuine error must still fail the round.

    Unlike a graceful barrier timeout, an actual error capturing R1 (e.g.
    the R1 turn threw, or the host response was missing ``r1_prompt_text``)
    has no ``timed_out`` flag set, so it must still be recorded as a
    forum-agent failure and raise when every dispatch has that shape.
    """
    orch, runtime = _make_orch_with_shared_container(tmp_path)

    runtime.run_task.return_value = RuntimeResult(
        output="",
        tool_trace=[],
        runtime_meta={
            "cross_task_shared_container_meta": {
                "enabled": True,
                "r1_captured": False,
                "note": "cross_task_r1: R1 turn threw: Error: boom",
            },
        },
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )

    traces = [
        TaskTrace(
            generation=1,
            agent_id="agent-0",
            task_id="t1",
            model_output="",
            eval_result={"resolved": False},
            native_score=0.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
        TaskTrace(
            generation=1,
            agent_id="agent-1",
            task_id="t2",
            model_output="",
            eval_result={"resolved": True},
            native_score=1.0,
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        ),
    ]

    with pytest.raises(ForumValidationError, match="all cross-task forum agents failed"):
        cross_task_forum(orch, generation=1, traces=traces)

    lifecycle = orch.accumulator._entries  # type: ignore[attr-defined]
    r0_rows = [k for k in lifecycle if k[2] == "__lc:cross_task_forum_round_0"]
    r1_rows = [k for k in lifecycle if k[2] == "__lc:cross_task_forum_round_1"]
    assert len(r0_rows) == 2
    assert len(r1_rows) == 0, f"R1 row must NOT be recorded when R1 block is absent; got {r1_rows}"
    assert orch.knowledge_phase_health_by_generation()[1]["forum_agent_failures"] == 2


# ── CLI flag tests ────────────────────────────────────────────────────────


def test_cli_cross_task_shared_container_flag_default_off():
    from kcsi.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["--task-source", "polyglot", "--tasks-path", "dummy.json"],
    )
    assert args.cross_task_shared_container is False


def test_cli_cross_task_shared_container_flag_bare_enables():
    from kcsi.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "polyglot",
            "--tasks-path",
            "dummy.json",
            "--cross-task-shared-container",
        ],
    )
    assert args.cross_task_shared_container is True


def test_cli_cross_task_shared_container_flag_accepts_bool_words():
    from kcsi.cli import _build_parser

    base = ["--task-source", "polyglot", "--tasks-path", "dummy.json"]
    for raw, expected in [("true", True), ("false", False), ("0", False), ("1", True), ("yes", True), ("no", False)]:
        parser = _build_parser()
        args = parser.parse_args([*base, "--cross-task-shared-container", raw])
        assert args.cross_task_shared_container is expected, (raw, expected)
