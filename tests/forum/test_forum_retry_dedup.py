"""Forum-retry duplicate-post dedup (issue #541).

When ``_run_retryable_forum_task`` retries a transiently-failed forum agent,
the failed attempt may have already written ``forum_post`` / ``insight`` /
``comment`` / ``done`` events to ``ForumBus`` before the SDK iterator
drained or the scheduled-task ended without a result event. Each event has
a fresh ``fb-<uuid>`` external_id, so the existing ``bulk_has_external_ids``
dedup in ``_drain_forum_bus`` does not catch them — the retry's events
land alongside the failed-attempt events and both are persisted into
``KnowledgeStore``.

The fix records failed-attempt event_ids in a per-bus sidecar
(``<stem>.stale_events.jsonl``) and the drain skips them. These tests pin
that contract end-to-end without touching the runtime/MCP layers.
"""

from __future__ import annotations

from pathlib import Path

from kcsi.memory.forum_bus import ForumBus
from kcsi.memory.knowledge_store import KnowledgeStore
from kcsi.orchestrator.engine import (
    _drain_forum_bus,
    _run_retryable_forum_task,
)
from kcsi.runtime.normalize import SilentAgentRuntimeError
from kcsi.runtime.types import RuntimeResult
from kcsi.tokens import TokenUsage


def _make_bus(tmp_path: Path, generation: int = 1) -> ForumBus:
    return ForumBus(
        db_path=str(tmp_path / "memory.sqlite"),
        experiment="test",
        generation=generation,
    )


def _make_ks(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(str(tmp_path / "knowledge.sqlite"), default_experiment="test")


# ---------------------------------------------------------------------------
# ForumBus stale-event sidecar contract
# ---------------------------------------------------------------------------


class TestForumBusStaleEvents:
    def test_mark_stale_persists_event_ids(self, tmp_path):
        bus = _make_bus(tmp_path)
        ev_a = bus.append(round_num=0, agent_id="a", message_type="post", content={"text": "x"})
        ev_b = bus.append(round_num=0, agent_id="a", message_type="post", content={"text": "y"})
        bus.mark_stale([ev_a["event_id"], ev_b["event_id"]], reason="failed_attempt")
        stale = bus.read_stale_event_ids()
        assert ev_a["event_id"] in stale
        assert ev_b["event_id"] in stale

    def test_mark_stale_empty_is_noop(self, tmp_path):
        bus = _make_bus(tmp_path)
        bus.mark_stale([], reason="failed_attempt")
        assert bus.read_stale_event_ids() == set()

    def test_mark_stale_handles_missing_sidecar(self, tmp_path):
        bus = _make_bus(tmp_path)
        # No prior writes — read_stale_event_ids should return empty, not raise.
        assert bus.read_stale_event_ids() == set()

    def test_read_only_bus_mark_stale_silently_skipped(self, tmp_path):
        bus = _make_bus(tmp_path)
        # Simulate read-only filesystem — mark_stale must never raise even
        # when the bus directory is read-only (e.g. inside a container with
        # a read-only bind mount).
        bus._read_only = True  # type: ignore[attr-defined]
        bus.mark_stale(["fb-1"], reason="failed_attempt")
        # The read path still works on whatever was written previously
        # (here nothing was written), so it returns empty.
        assert bus.read_stale_event_ids() == set()


# ---------------------------------------------------------------------------
# _drain_forum_bus skips stale events
# ---------------------------------------------------------------------------


class TestDrainSkipsStale:
    def test_drain_skips_stale_post_events(self, tmp_path):
        bus = _make_bus(tmp_path)
        ks = _make_ks(tmp_path)
        try:
            stale_post = bus.append(
                round_num=0,
                agent_id="a",
                message_type="post",
                content={"task_id": "t1", "text": "stale content"},
            )
            fresh_post = bus.append(
                round_num=0,
                agent_id="a",
                message_type="post",
                content={"task_id": "t1", "text": "fresh content"},
            )
            bus.mark_stale([stale_post["event_id"]], reason="failed_attempt")

            count = _drain_forum_bus(forum_bus=bus, knowledge=ks, generation=1, experiment="test")
            assert count == 1  # only the fresh post

            posts = ks.query_task("t1", entry_types=["post"])["discussion"]
            texts = [p.get("text") for p in posts]
            assert "fresh content" in texts
            assert "stale content" not in texts
            assert fresh_post["event_id"]  # sanity
        finally:
            ks.close()

    def test_drain_skips_stale_across_message_types(self, tmp_path):
        """A single failed attempt may have written posts AND insights AND
        a done event before crashing. All four message types must be
        marked stale uniformly when the bus owner records them."""
        bus = _make_bus(tmp_path)
        ks = _make_ks(tmp_path)
        try:
            stale_post = bus.append(
                round_num=1,
                agent_id="a",
                message_type="post",
                content={"task_id": "t1", "text": "stale post"},
            )
            stale_insight = bus.append(
                round_num=1,
                agent_id="a",
                message_type="insight",
                content={
                    "text": "stale insight",
                    "scope": "task",
                    "confidence": "medium",
                    "evidence_task_ids": ["t1"],
                },
            )
            stale_done = bus.append(
                round_num=1,
                agent_id="a",
                message_type="done",
                content={"task_ids": ["t1"]},
            )
            # Then the retry succeeds and writes fresh versions.
            fresh_post = bus.append(
                round_num=1,
                agent_id="a",
                message_type="post",
                content={"task_id": "t1", "text": "fresh post"},
            )
            bus.mark_stale(
                [stale_post["event_id"], stale_insight["event_id"], stale_done["event_id"]],
                reason="failed_attempt",
            )

            count = _drain_forum_bus(forum_bus=bus, knowledge=ks, generation=1, experiment="test")
            # Only fresh_post drained as a knowledge row (done events don't
            # increment the count returned by drain).
            assert count == 1

            posts = ks.query_task("t1", entry_types=["post"])["discussion"]
            assert any(p.get("text") == "fresh post" for p in posts)
            assert not any(p.get("text") == "stale post" for p in posts)

            insights = ks.query_task("t1", entry_types=["insight"])["insights"]
            assert not any(i.get("text") == "stale insight" for i in insights)
            assert fresh_post["event_id"]
        finally:
            ks.close()


# ---------------------------------------------------------------------------
# _run_retryable_forum_task marks failed-attempt events stale
# ---------------------------------------------------------------------------


class TestRetryHelperMarksStale:
    def test_success_after_retry_marks_failed_attempt_events_stale(self, tmp_path):
        bus = _make_bus(tmp_path)
        ks = _make_ks(tmp_path)
        agent_id = "agent-0"
        round_num = 0

        attempt_state = {"count": 0, "stale_event_id": ""}

        def run_once() -> RuntimeResult:
            attempt_state["count"] += 1
            if attempt_state["count"] == 1:
                # Simulate the failed attempt writing to ForumBus before
                # the SDK iterator drains. The orchestrator catches the
                # SilentAgentRuntimeError after the subprocess exits.
                ev = bus.append(
                    round_num=round_num,
                    agent_id=agent_id,
                    message_type="post",
                    content={"task_id": "t1", "text": "stale from attempt 1"},
                )
                attempt_state["stale_event_id"] = ev["event_id"]
                raise SilentAgentRuntimeError(
                    "iterator drained without result event",
                    runtime_meta={"status": "error", "tokens_source": "per_turn_sum"},
                )
            # Success: the retried container writes its own post.
            bus.append(
                round_num=round_num,
                agent_id=agent_id,
                message_type="post",
                content={"task_id": "t1", "text": "fresh from attempt 2"},
            )
            return RuntimeResult(
                output="ok",
                tool_trace=[],
                runtime_meta={"status": "success"},
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            )

        try:
            token_usage, runtime_meta, _ = _run_retryable_forum_task(
                run_once=run_once,
                generation=1,
                agent_id=agent_id,
                phase_label="per-task discussion task t1",
                attempts=2,
                forum_bus=bus,
                forum_round=round_num,
            )

            assert attempt_state["count"] == 2
            stale = bus.read_stale_event_ids()
            assert attempt_state["stale_event_id"] in stale, "failed-attempt event_id was not marked stale"

            count = _drain_forum_bus(forum_bus=bus, knowledge=ks, generation=1, experiment="test")
            assert count == 1
            posts = ks.query_task("t1", entry_types=["post"])["discussion"]
            texts = [p.get("text") for p in posts]
            assert "fresh from attempt 2" in texts
            assert "stale from attempt 1" not in texts
            # token_usage and runtime_meta integrity preserved (#540 contract).
            assert token_usage.input_tokens == 10
            assert "retry_attempts" in runtime_meta
        finally:
            ks.close()

    def test_terminal_failure_marks_all_attempts_stale(self, tmp_path):
        bus = _make_bus(tmp_path)
        ks = _make_ks(tmp_path)
        agent_id = "agent-1"
        round_num = 0

        attempt_state = {"count": 0, "ids": []}

        def run_once() -> RuntimeResult:
            attempt_state["count"] += 1
            ev = bus.append(
                round_num=round_num,
                agent_id=agent_id,
                message_type="post",
                content={
                    "task_id": "t1",
                    "text": f"partial content attempt {attempt_state['count']}",
                },
            )
            attempt_state["ids"].append(ev["event_id"])
            raise SilentAgentRuntimeError(
                "iterator drained without result event",
                runtime_meta={"status": "error", "tokens_source": "per_turn_sum"},
            )

        try:
            _, runtime_meta, _ = _run_retryable_forum_task(
                run_once=run_once,
                generation=1,
                agent_id=agent_id,
                phase_label="per-task discussion task t1",
                attempts=2,
                forum_bus=bus,
                forum_round=round_num,
            )

            assert attempt_state["count"] == 2
            stale = bus.read_stale_event_ids()
            for ev_id in attempt_state["ids"]:
                assert ev_id in stale

            count = _drain_forum_bus(forum_bus=bus, knowledge=ks, generation=1, experiment="test")
            assert count == 0  # every attempt's posts are stale
            posts = ks.query_task("t1", entry_types=["post"])["discussion"]
            assert posts == []
            assert "forum_error" in runtime_meta
        finally:
            ks.close()

    def test_first_try_success_no_stale_writes(self, tmp_path):
        bus = _make_bus(tmp_path)
        agent_id = "agent-2"
        round_num = 0

        def run_once() -> RuntimeResult:
            bus.append(
                round_num=round_num,
                agent_id=agent_id,
                message_type="post",
                content={"task_id": "t1", "text": "first-try ok"},
            )
            return RuntimeResult(
                output="ok",
                tool_trace=[],
                runtime_meta={"status": "success"},
                token_usage=TokenUsage(input_tokens=5, output_tokens=2),
            )

        _run_retryable_forum_task(
            run_once=run_once,
            generation=1,
            agent_id=agent_id,
            phase_label="per-task discussion task t1",
            attempts=2,
            forum_bus=bus,
            forum_round=round_num,
        )
        assert bus.read_stale_event_ids() == set()

    def test_retry_helper_only_marks_this_agents_events(self, tmp_path):
        """Concurrent agents share one ForumBus. A retry for agent A must
        not mark agent B's events stale even if B's events were appended
        between A's pre-attempt seq and A's failure."""
        bus = _make_bus(tmp_path)
        agent_a = "agent-A"
        agent_b = "agent-B"
        round_num = 0

        # B writes a post first (interleaved background activity).
        ev_b = bus.append(
            round_num=round_num,
            agent_id=agent_b,
            message_type="post",
            content={"task_id": "t1", "text": "B's content"},
        )

        attempt_state = {"count": 0, "a_id": ""}

        def run_once() -> RuntimeResult:
            attempt_state["count"] += 1
            if attempt_state["count"] == 1:
                # B writes another post during A's attempt (concurrent).
                bus.append(
                    round_num=round_num,
                    agent_id=agent_b,
                    message_type="post",
                    content={"task_id": "t1", "text": "B during A's attempt"},
                )
                ev_a = bus.append(
                    round_num=round_num,
                    agent_id=agent_a,
                    message_type="post",
                    content={"task_id": "t1", "text": "A's stale"},
                )
                attempt_state["a_id"] = ev_a["event_id"]
                raise SilentAgentRuntimeError(
                    "drained",
                    runtime_meta={"status": "error", "tokens_source": "per_turn_sum"},
                )
            return RuntimeResult(
                output="ok",
                tool_trace=[],
                runtime_meta={"status": "success"},
                token_usage=TokenUsage(),
            )

        _run_retryable_forum_task(
            run_once=run_once,
            generation=1,
            agent_id=agent_a,
            phase_label="per-task discussion task t1",
            attempts=2,
            forum_bus=bus,
            forum_round=round_num,
        )

        stale = bus.read_stale_event_ids()
        assert attempt_state["a_id"] in stale
        # B's events MUST NOT be marked stale.
        assert ev_b["event_id"] not in stale


# ---------------------------------------------------------------------------
# Backwards compatibility: forum_bus / forum_round are optional
# ---------------------------------------------------------------------------


def test_retry_helper_without_forum_bus_does_not_break(tmp_path):
    """Retry helper must keep working when forum_bus/forum_round aren't
    plumbed in (legacy callers, future non-bus retry paths)."""
    state = {"count": 0}

    def run_once() -> RuntimeResult:
        state["count"] += 1
        return RuntimeResult(
            output="ok",
            tool_trace=[],
            runtime_meta={"status": "success"},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    token_usage, runtime_meta, output = _run_retryable_forum_task(
        run_once=run_once,
        generation=1,
        agent_id="a",
        phase_label="phase",
        attempts=2,
    )
    assert state["count"] == 1
    assert token_usage.input_tokens == 1
    assert output == "ok"
