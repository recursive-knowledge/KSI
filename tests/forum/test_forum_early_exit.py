"""Tests for forum early-exit wait-loop.

The orchestrator starts a background watcher during each per-task /
cross-task forum round.  When every expected agent has signalled done via
``forum_signal_done`` (persisted on the ForumBus JSONL), the watcher short-
circuits the remaining container timeout and triggers a targeted
``docker stop``.

The hard backstop remains ``--forum-timeout-sec`` /
``--cross-task-forum-timeout-sec`` inside the container runtime — the
watcher can only finish *early*, never late.

These tests exercise the watcher in isolation (no Docker / no real
containers) so the assertions are deterministic and fast.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

from ksi.memory.forum_bus import ForumBus
from ksi.models import GenerationConfig, TaskTrace
from ksi.orchestrator.engine import (
    GenerationalOrchestrator,
    NoopPersistence,
    _all_expected_signalled,
    _forum_container_prefix,
    _ForumEarlyExitWatcher,
    _read_done_agent_ids,
    _read_done_signals,
)
from ksi.runtime.types import RuntimeResult
from ksi.tokens import LLMResponse, TokenUsage
from tests.orchestrator_phase_helpers import per_task_forum

# ---------------------------------------------------------------------------
# Signal-parsing helpers
# ---------------------------------------------------------------------------


def test_read_done_signals_expands_task_ids(tmp_path):
    """`forum_signal_done` may carry multiple task_ids per event — each one
    should be expanded to a separate ``(task_id, agent_id)`` observation.
    """
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(
        round_num=0,
        agent_id="agent-0",
        message_type="done",
        content={"task_ids": ["t1", "t2"]},
    )
    bus.append(
        round_num=0,
        agent_id="agent-1",
        message_type="done",
        content={"task_ids": ["t2"]},
    )
    observed = _read_done_signals(bus)
    assert observed["t1"] == {"agent-0"}
    assert observed["t2"] == {"agent-0", "agent-1"}


def test_read_done_signals_with_round_ignores_roundless_events(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=None, agent_id="agent-0", message_type="done", content={"task_ids": ["t1"]})
    bus.append(round_num=1, agent_id="agent-1", message_type="done", content={"task_ids": ["t1"]})

    observed = _read_done_signals(bus, round_num=1)

    assert observed["t1"] == {"agent-1"}


def test_read_done_agent_ids_ignores_task_payload(tmp_path):
    """Cross-task events carry empty task_ids — every agent_id still counts."""
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=0, agent_id="agent-0", message_type="done", content={"task_ids": []})
    bus.append(round_num=0, agent_id="agent-1", message_type="done", content={})
    assert _read_done_agent_ids(bus) == {"agent-0", "agent-1"}


def test_read_done_agent_ids_with_round_ignores_roundless_events(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=None, agent_id="agent-0", message_type="done", content={})
    bus.append(round_num=1, agent_id="agent-1", message_type="done", content={})

    assert _read_done_agent_ids(bus, round_num=1) == {"agent-1"}


# ---------------------------------------------------------------------------
# _all_expected_signalled logic
# ---------------------------------------------------------------------------


def test_all_expected_signalled_per_task_true(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(
        round_num=0,
        agent_id="a",
        message_type="done",
        content={"task_ids": ["t1"]},
    )
    bus.append(
        round_num=0,
        agent_id="b",
        message_type="done",
        content={"task_ids": ["t1"]},
    )
    assert _all_expected_signalled(
        forum_bus=bus,
        expected={"t1": {"a", "b"}},
    )


def test_all_expected_signalled_per_task_partial_false(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(
        round_num=0,
        agent_id="a",
        message_type="done",
        content={"task_ids": ["t1"]},
    )
    # Only 1 of 2 expected agents has signalled.
    assert not _all_expected_signalled(
        forum_bus=bus,
        expected={"t1": {"a", "b"}},
    )


def test_all_expected_signalled_cross_task_true(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=0, agent_id="a", message_type="done", content={})
    bus.append(round_num=0, agent_id="b", message_type="done", content={"task_ids": []})
    assert _all_expected_signalled(
        forum_bus=bus,
        expected={},
        agent_only=True,
        expected_agents={"a", "b"},
    )


def test_all_expected_signalled_cross_task_partial_false(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=0, agent_id="a", message_type="done", content={})
    # Missing "b"
    assert not _all_expected_signalled(
        forum_bus=bus,
        expected={},
        agent_only=True,
        expected_agents={"a", "b"},
    )


# ---------------------------------------------------------------------------
# Container-prefix derivation
# ---------------------------------------------------------------------------


def test_forum_container_prefix_per_task():
    pfx = _forum_container_prefix(
        experiment_name="myexp",
        generation=2,
        phase="per_task",
    )
    assert pfx == "ksi-runtime-task--myexp--forum--g2"


def test_forum_container_prefix_cross_task():
    pfx = _forum_container_prefix(
        experiment_name="myexp",
        generation=2,
        phase="cross_task",
    )
    assert pfx == "ksi-runtime-task--myexp--cross-task-forum--g2"


# ---------------------------------------------------------------------------
# _ForumEarlyExitWatcher: short-circuits on all-done
# ---------------------------------------------------------------------------


def _start_watcher(bus, *, expected=None, expected_agents=None, agent_only=False, round_num=None, poll_sec=0.1):
    stop_event = threading.Event()
    triggered = threading.Event()
    watcher = _ForumEarlyExitWatcher(
        forum_bus=bus,
        expected=expected,
        expected_agents=expected_agents,
        agent_only=agent_only,
        round_num=round_num,
        stop_event=stop_event,
        triggered_event=triggered,
        container_name_prefixes=["ksi-runtime-task--test-noop"],
        poll_interval_sec=poll_sec,
        phase_label="unittest",
    )
    return watcher, stop_event, triggered


def test_watcher_triggers_when_all_agents_signal(tmp_path):
    """Watcher sets triggered_event once every expected pair appears."""
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    # Pre-populate signals
    bus.append(
        round_num=0,
        agent_id="a",
        message_type="done",
        content={"task_ids": ["t1"]},
    )
    bus.append(
        round_num=0,
        agent_id="b",
        message_type="done",
        content={"task_ids": ["t1"]},
    )
    watcher, stop_event, triggered = _start_watcher(
        bus,
        expected={"t1": {"a", "b"}},
        poll_sec=0.05,
    )
    # Stub the docker-stop helper so tests don't touch the daemon
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        # Wait up to 5s for the watcher to notice and trigger
        assert triggered.wait(timeout=5.0), "watcher did not trigger on all-done"
        stop_event.set()
        watcher.join(timeout=5.0)
        assert not watcher.is_alive()


def test_watcher_round_scoped_does_not_use_knowledge_fallback(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    knowledge = MagicMock()
    knowledge.get_done_status.return_value = {"all_done": True}
    stop_event = threading.Event()
    triggered = threading.Event()
    watcher = _ForumEarlyExitWatcher(
        forum_bus=bus,
        expected={"t1": {"a"}},
        round_num=1,
        stop_event=stop_event,
        triggered_event=triggered,
        container_name_prefixes=[],
        poll_interval_sec=0.05,
        knowledge=knowledge,
        experiment="e",
        generation=1,
        phase_label="unittest",
    )

    assert not watcher._all_signalled()
    knowledge.get_done_status.assert_not_called()


def test_watcher_respects_hard_timeout_when_no_signals(tmp_path):
    """If no agent signals done, the watcher never triggers — it exits only
    when the outer forum phase sets ``stop_event``.  This simulates the hard
    timeout path where the backstop is still the container timeout.
    """
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    watcher, stop_event, triggered = _start_watcher(
        bus,
        expected={"t1": {"a", "b"}},
        poll_sec=0.05,
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        # Sleep longer than several poll intervals; watcher should NOT trigger.
        time.sleep(0.3)
        assert not triggered.is_set(), "watcher fired without any done signals"
        # Simulate the forum phase finishing — stop_event tells watcher to exit.
        stop_event.set()
        watcher.join(timeout=5.0)
        assert not watcher.is_alive()
        assert not triggered.is_set()


def test_watcher_partial_signals_no_trigger(tmp_path):
    """Only 2 of 3 expected agents signal — watcher must stay asleep."""
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(
        round_num=0,
        agent_id="a",
        message_type="done",
        content={"task_ids": ["t1"]},
    )
    bus.append(
        round_num=0,
        agent_id="b",
        message_type="done",
        content={"task_ids": ["t1"]},
    )
    watcher, stop_event, triggered = _start_watcher(
        bus,
        expected={"t1": {"a", "b", "c"}},
        poll_sec=0.05,
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        time.sleep(0.3)
        assert not triggered.is_set(), "watcher fired on partial signals"
        stop_event.set()
        watcher.join(timeout=5.0)


def test_watcher_triggers_after_late_signal(tmp_path):
    """Signals arrive mid-run; watcher catches up on the next poll."""
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    watcher, stop_event, triggered = _start_watcher(
        bus,
        expected={"t1": {"a", "b"}},
        poll_sec=0.05,
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        time.sleep(0.15)  # a few polls where no signals exist
        assert not triggered.is_set()
        # Now land both signals; next poll should trigger.
        bus.append(
            round_num=0,
            agent_id="a",
            message_type="done",
            content={"task_ids": ["t1"]},
        )
        bus.append(
            round_num=0,
            agent_id="b",
            message_type="done",
            content={"task_ids": ["t1"]},
        )
        assert triggered.wait(timeout=5.0), "watcher didn't catch up on late signals"
        stop_event.set()
        watcher.join(timeout=5.0)


def test_watcher_cross_task_agent_only_trigger(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    # Cross-task done events — task_ids payload is empty (matches real MCP).
    bus.append(round_num=0, agent_id="a", message_type="done", content={})
    bus.append(round_num=0, agent_id="b", message_type="done", content={"task_ids": []})
    watcher, stop_event, triggered = _start_watcher(
        bus,
        expected_agents={"a", "b"},
        agent_only=True,
        poll_sec=0.05,
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        assert triggered.wait(timeout=5.0)
        stop_event.set()
        watcher.join(timeout=5.0)


def test_watcher_cross_task_partial_no_trigger(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=0, agent_id="a", message_type="done", content={})
    watcher, stop_event, triggered = _start_watcher(
        bus,
        expected_agents={"a", "b"},
        agent_only=True,
        poll_sec=0.05,
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        time.sleep(0.3)
        assert not triggered.is_set()
        stop_event.set()
        watcher.join(timeout=5.0)


# ---------------------------------------------------------------------------
# _ForumEarlyExitWatcher: quorum-based early exit (issue #1045)
#
# A straggler minority (observed in real campaign traces -- see #1045) can
# permanently block the all-required watcher above from ever firing, forcing
# the full hard-timeout wait. ``quorum_pct``/``quorum_grace_sec`` let callers
# opt into exiting once only a threshold fraction of expected agents have
# signalled, after a grace window from the moment that fraction was reached.
# ---------------------------------------------------------------------------


def _start_watcher_with_quorum(
    bus,
    *,
    expected=None,
    expected_agents=None,
    agent_only=False,
    round_num=None,
    poll_sec=0.05,
    quorum_pct=100.0,
    quorum_grace_sec=0.0,
):
    stop_event = threading.Event()
    triggered = threading.Event()
    watcher = _ForumEarlyExitWatcher(
        forum_bus=bus,
        expected=expected,
        expected_agents=expected_agents,
        agent_only=agent_only,
        round_num=round_num,
        stop_event=stop_event,
        triggered_event=triggered,
        container_name_prefixes=["ksi-runtime-task--test-noop"],
        poll_interval_sec=poll_sec,
        phase_label="unittest",
        quorum_pct=quorum_pct,
        quorum_grace_sec=quorum_grace_sec,
    )
    return watcher, stop_event, triggered


def test_watcher_default_quorum_100_never_cuts_off_stragglers(tmp_path):
    """Default quorum_pct=100 must behave exactly like the pre-#1045
    all-required watcher: no trigger until every expected agent signals,
    even with a grace window configured and even well past that window."""
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=0, agent_id="a", message_type="done", content={})
    bus.append(round_num=0, agent_id="b", message_type="done", content={})
    bus.append(round_num=0, agent_id="c", message_type="done", content={})
    # Only 3 of 4 expected agents signal ("d" is the straggler).
    watcher, stop_event, triggered = _start_watcher_with_quorum(
        bus,
        expected_agents={"a", "b", "c", "d"},
        agent_only=True,
        quorum_grace_sec=0.1,  # grace configured, but quorum_pct still 100 (default)
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        time.sleep(0.4)  # well past the grace window
        assert not triggered.is_set(), "default quorum_pct=100 must require every agent"
        stop_event.set()
        watcher.join(timeout=5.0)


def test_watcher_quorum_below_threshold_no_trigger(tmp_path):
    """Below the configured quorum, the watcher stays asleep even after the
    grace window would have elapsed."""
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=0, agent_id="a", message_type="done", content={})
    # Only 1 of 4 signalled (25%), below a 75% quorum.
    watcher, stop_event, triggered = _start_watcher_with_quorum(
        bus,
        expected_agents={"a", "b", "c", "d"},
        agent_only=True,
        quorum_pct=75.0,
        quorum_grace_sec=0.1,
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        time.sleep(0.4)
        assert not triggered.is_set(), "watcher fired below the configured quorum"
        stop_event.set()
        watcher.join(timeout=5.0)


def test_watcher_quorum_triggers_after_grace_window_cross_task(tmp_path):
    """Cross-task (agent_only) mode: once >=75% of expected agents have
    signalled done, the watcher fires after the grace window elapses --
    even though one agent (the straggler) never signals."""
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=0, agent_id="a", message_type="done", content={})
    bus.append(round_num=0, agent_id="b", message_type="done", content={})
    bus.append(round_num=0, agent_id="c", message_type="done", content={})
    # 3 of 4 expected agents signalled -> 75%; "d" never signals.
    watcher, stop_event, triggered = _start_watcher_with_quorum(
        bus,
        expected_agents={"a", "b", "c", "d"},
        agent_only=True,
        quorum_pct=75.0,
        quorum_grace_sec=0.1,
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        assert triggered.wait(timeout=5.0), "watcher never fired on quorum+grace"
        stop_event.set()
        watcher.join(timeout=5.0)
        assert not watcher.is_alive()


def test_watcher_quorum_triggers_after_grace_window_per_task(tmp_path):
    """Per-task (task-scoped) mode also honors the quorum path."""
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=0, agent_id="a", message_type="done", content={"task_ids": ["t1"]})
    bus.append(round_num=0, agent_id="b", message_type="done", content={"task_ids": ["t1"]})
    # 2 of 3 expected agents on t1 -> 66.7%; "c" never signals.
    watcher, stop_event, triggered = _start_watcher_with_quorum(
        bus,
        expected={"t1": {"a", "b", "c"}},
        quorum_pct=60.0,
        quorum_grace_sec=0.1,
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        assert triggered.wait(timeout=5.0), "per-task watcher never fired on quorum+grace"
        stop_event.set()
        watcher.join(timeout=5.0)


def test_watcher_quorum_does_not_trigger_before_grace_elapses(tmp_path):
    """Quorum is met immediately, but the watcher must still wait out the
    full grace window before triggering -- it must not fire on the very
    first poll after quorum is reached."""
    bus = ForumBus(db_path=str(tmp_path / "m.sqlite"), experiment="e", generation=1)
    bus.append(round_num=0, agent_id="a", message_type="done", content={})
    bus.append(round_num=0, agent_id="b", message_type="done", content={})
    bus.append(round_num=0, agent_id="c", message_type="done", content={})
    watcher, stop_event, triggered = _start_watcher_with_quorum(
        bus,
        expected_agents={"a", "b", "c", "d"},
        agent_only=True,
        poll_sec=0.05,
        quorum_pct=75.0,
        quorum_grace_sec=1.0,  # long enough that an immediate fire would be caught
    )
    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", return_value=0):
        watcher.start()
        time.sleep(0.3)  # quorum reached well before this, but grace has not elapsed
        assert not triggered.is_set(), "watcher fired before the grace window elapsed"
        stop_event.set()
        watcher.join(timeout=5.0)


# ---------------------------------------------------------------------------
# CLI flag-gating
# ---------------------------------------------------------------------------


def test_cli_forum_early_exit_default_off():
    from ksi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--task-source", "arc", "--tasks-path", "/tmp/x"])
    assert args.forum_early_exit is False


def test_cli_forum_early_exit_can_opt_in():
    from ksi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--task-source", "arc", "--tasks-path", "/tmp/x", "--forum-early-exit", "true"])
    assert args.forum_early_exit is True


def test_cli_forum_early_exit_off():
    from ksi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--task-source", "arc", "--tasks-path", "/tmp/x", "--forum-early-exit", "off"])
    assert args.forum_early_exit is False


def test_cli_forum_early_exit_poll_override():
    from ksi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--task-source", "arc", "--tasks-path", "/tmp/x", "--forum-early-exit-poll-sec", "1.5"])
    assert args.forum_early_exit_poll_sec == 1.5


def test_cli_forum_early_exit_quorum_pct_default_is_100():
    """Default must preserve the pre-#1045 all-required behavior."""
    from ksi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--task-source", "arc", "--tasks-path", "/tmp/x"])
    assert args.forum_early_exit_quorum_pct == 100.0


def test_cli_forum_early_exit_quorum_pct_override():
    from ksi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--task-source", "arc", "--tasks-path", "/tmp/x", "--forum-early-exit-quorum-pct", "80"])
    assert args.forum_early_exit_quorum_pct == 80.0


def test_cli_forum_early_exit_quorum_grace_sec_default_is_zero():
    from ksi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--task-source", "arc", "--tasks-path", "/tmp/x"])
    assert args.forum_early_exit_quorum_grace_sec == 0.0


def test_cli_forum_early_exit_quorum_grace_sec_override():
    from ksi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["--task-source", "arc", "--tasks-path", "/tmp/x", "--forum-early-exit-quorum-grace-sec", "45"]
    )
    assert args.forum_early_exit_quorum_grace_sec == 45.0


# ---------------------------------------------------------------------------
# End-to-end: per-task forum dispatch logs early-exit
# ---------------------------------------------------------------------------


def _make_trace(generation: int, agent_id: str, task_id: str) -> TaskTrace:
    return TaskTrace(
        generation=generation,
        agent_id=agent_id,
        task_id=task_id,
        model_output="patch",
        eval_result={"resolved": True},
        native_score=1.0,
        token_usage=TokenUsage(input_tokens=1, output_tokens=1),
    )


def _make_orch(tmp_path, runtime, *, early_exit_enabled: bool = True) -> GenerationalOrchestrator:
    db_path = str(tmp_path / "memory.sqlite")
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
        # Pin explicitly: the ForumBus writes below use experiment="default",
        # and the dataclass default is now "ksi" (CLI parity, #732).
        experiment_name="default",
    )
    config.per_task_forum_rounds = 1
    config.cross_task_forum_rounds = 1
    config.forum_early_exit = bool(early_exit_enabled)
    config.forum_early_exit_poll_sec = 0.05
    orch = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )
    return orch


def test_per_task_forum_phase_triggers_early_exit(tmp_path):
    """End-to-end: when fake agents pre-write done signals to the ForumBus,
    the engine's per-task forum phase logs an early-exit.

    We patch ``_stop_forum_containers`` to a spy so the test doesn't depend
    on a live docker daemon, and we check it was called with the per-task
    prefix once the watcher fires.
    """
    runtime = MagicMock()

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        metadata = getattr(task, "metadata", {}) or {}
        if metadata.get("task_source") == "per_task_forum":
            # Each agent writes a ``done`` signal for all tasks it owns.
            bus = ForumBus(
                db_path=str(tmp_path / "memory.sqlite"),
                experiment="default",
                generation=generation,
            )
            forum_task_ids = list(metadata.get("forum_task_ids") or [])
            bus.append(
                round_num=int(metadata.get("forum_round", 0)),
                agent_id=agent_id,
                message_type="done",
                content={"task_ids": forum_task_ids},
            )
            # Slow work so the watcher has a chance to observe and stop us.
            time.sleep(0.25)
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    orch = _make_orch(tmp_path, runtime)

    traces = [
        _make_trace(1, "agent-0", "t1"),
        _make_trace(1, "agent-1", "t1"),
    ]

    called: list[list[str]] = []

    def fake_stop(prefixes):
        called.append(list(prefixes))
        return 0

    with patch("ksi.orchestrator.forum_runtime._stop_forum_containers", side_effect=fake_stop):
        per_task_forum(orch, generation=1, traces=traces)

    # The watcher should have fired once (both agents signalled done on t1)
    # and called _stop_forum_containers with the per-task forum prefix.
    assert called, "early-exit watcher never called _stop_forum_containers"
    assert any(any("forum--g1" in p and "cross-task" not in p for p in prefixes) for prefixes in called), (
        f"unexpected prefixes {called!r}"
    )


def test_per_task_forum_phase_no_early_exit_when_flag_off(tmp_path):
    """When ``config.forum_early_exit=False``, no watcher is started and
    ``_stop_forum_containers`` is never called -- behavior reduces to the
    legacy hard-timeout-only path.
    """
    runtime = MagicMock()

    def fake_run_task(*, generation, agent_id, task, **kwargs):
        metadata = getattr(task, "metadata", {}) or {}
        if metadata.get("task_source") == "per_task_forum":
            bus = ForumBus(
                db_path=str(tmp_path / "memory.sqlite"),
                experiment="default",
                generation=generation,
            )
            bus.append(
                round_num=0,
                agent_id=agent_id,
                message_type="done",
                content={"task_ids": list(metadata.get("forum_task_ids") or [])},
            )
        return RuntimeResult(
            output="",
            tool_trace=[],
            runtime_meta={},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    runtime.run_task.side_effect = fake_run_task

    orch = _make_orch(tmp_path, runtime, early_exit_enabled=False)

    traces = [
        _make_trace(1, "agent-0", "t1"),
        _make_trace(1, "agent-1", "t1"),
    ]

    called: list[list[str]] = []

    with patch(
        "ksi.orchestrator.forum_runtime._stop_forum_containers",
        side_effect=lambda p: called.append(list(p)) or 0,
    ):
        per_task_forum(orch, generation=1, traces=traces)

    assert not called, "early-exit was disabled but _stop_forum_containers was still called"
