"""Tests for _drain_forum_bus() and handle_forum_post() ForumBus-only behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from kcsi.memory.forum_bus import ForumBus
from kcsi.memory.knowledge_store import KnowledgeStore
from kcsi.memory.store import MemoryStore
from kcsi.orchestrator.engine import _coerce_post_ref, _drain_forum_bus


def _make_knowledge_store(tmp_path: Path) -> KnowledgeStore:
    db_path = str(tmp_path / "knowledge.sqlite")
    return KnowledgeStore(db_path, default_experiment="test")


def _make_forum_bus(tmp_path: Path, *, generation: int = 1) -> ForumBus:
    db_path = str(tmp_path / "memory.sqlite")
    return ForumBus(db_path=db_path, experiment="test", generation=generation)


# ---------------------------------------------------------------------------
# _drain_forum_bus tests
# ---------------------------------------------------------------------------


class TestDrainForumBus:
    def test_drain_insight_events(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-1",
                message_type="insight",
                content={
                    "text": "discovered a pattern",
                    "scope": "task",
                    "confidence": "high",
                    "evidence_task_ids": ["task-42"],
                },
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            assert count == 1
            # query_task returns {"insights": [...], "discussion": [...], ...}
            result = ks.query_task("task-42", entry_types=["insight"])
            assert len(result["insights"]) >= 1
            assert result["insights"][0]["text"] == "discovered a pattern"
        finally:
            ks.close()

    def test_drain_post_events(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-2",
                message_type="post",
                content={
                    "task_id": "task-99",
                    "text": "I think we should try approach B",
                },
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=2,
                experiment="test",
            )
            assert count == 1
            result = ks.query_task("task-99", entry_types=["post"])
            assert len(result["discussion"]) >= 1
            assert result["discussion"][0]["text"] == "I think we should try approach B"
        finally:
            ks.close()

    def test_cross_task_post_forced_under_sentinel(self, tmp_path):
        """A cross-task forum post whose content carries a REAL task_id (an agent
        that ignored the "post under __cross_task__" instruction — observed with
        gpt-5.4-mini) must still be persisted under CROSS_TASK_SENTINEL, so the
        cross-task distiller (which reads query_task(CROSS_TASK_SENTINEL)) finds
        it. Mirrors the done-signal path and the documented drain contract
        (forum_phase.py: "Drains land under CROSS_TASK_SENTINEL")."""
        from kcsi.memory.knowledge_store import CROSS_TASK_SENTINEL

        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-noncompliant",
                message_type="post",
                content={"task_id": "de493100", "text": "cross-task observation"},
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=2,
                experiment="test",
                source_phase="cross_task_forum",
            )
            assert count == 1
            # Found under the sentinel, NOT under the model's self-reported id.
            sentinel = ks.query_task(CROSS_TASK_SENTINEL, entry_types=["post"])
            assert len(sentinel["discussion"]) == 1
            assert sentinel["discussion"][0]["text"] == "cross-task observation"
            leaked = ks.query_task("de493100", entry_types=["post"])
            assert len(leaked["discussion"]) == 0
        finally:
            ks.close()

    def test_per_task_post_keeps_real_task_id(self, tmp_path):
        """Non-regression: per-task forum posts must keep their real task_id
        (the sentinel override is scoped to source_phase='cross_task_forum')."""
        from kcsi.memory.knowledge_store import CROSS_TASK_SENTINEL

        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-1",
                message_type="post",
                content={"task_id": "task-77", "text": "per-task note"},
            )
            _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
                source_phase="per_task_forum",
            )
            assert len(ks.query_task("task-77", entry_types=["post"])["discussion"]) == 1
            assert len(ks.query_task(CROSS_TASK_SENTINEL, entry_types=["post"])["discussion"]) == 0
        finally:
            ks.close()

    def test_drain_threads_post_author_native_score(self, tmp_path):
        """The per-task drain records the post author's own task score
        (keyed by (task_id, agent_id)) so the distiller can weight high-score
        authors over low-score authors. Authors absent from the map (e.g.
        cross-task drains) leave native_score None."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="solver",
                message_type="post",
                content={"task_id": "task-99", "text": "the winning approach"},
            )
            bus.append(
                round_num=0,
                agent_id="unknown",
                message_type="post",
                content={"task_id": "task-99", "text": "no score recorded"},
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=2,
                experiment="test",
                source_phase="per_task_forum",
                native_score_by_task_agent={("task-99", "solver"): 1.0},
            )
            assert count == 2
            result = ks.query_task("task-99", entry_types=["post"])
            by_agent = {p["agent_id"]: p for p in result["discussion"]}
            assert by_agent["solver"]["native_score"] == 1.0
            assert by_agent["unknown"]["native_score"] is None
        finally:
            ks.close()

    def test_drain_on_drop_fires_on_swallowed_knowledge_write(self, tmp_path, monkeypatch):
        """#740 H3: a knowledge-row write that raises is swallowed (the drain does
        NOT raise as a whole), but on_drop surfaces it so a partial drain is visible
        instead of reading as a clean, healthy generation."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-x",
                message_type="post",
                content={"task_id": "task-1", "text": "a post that fails to land"},
            )

            def _boom(*_a, **_k):
                raise RuntimeError("knowledge write failed")

            # The drain batches every knowledge-row write into a single
            # transaction; each event's write runs inside its own SAVEPOINT
            # via ``_record_post_locked``, so injecting the failure there
            # exercises the per-event rollback + on_drop accounting.
            monkeypatch.setattr(ks, "_record_post_locked", _boom)
            dropped = []
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
                on_drop=dropped.append,
            )
            assert count == 0  # nothing landed
            assert dropped == [1]  # exactly one knowledge-row drop surfaced
        finally:
            ks.close()

    def test_drain_on_drop_dedupes_failed_event_counter_across_repeat_drains(self, tmp_path, monkeypatch):
        """Repeated drains may retry a transient failed write, but health should
        count a failed event id once so multi-drain forum phases do not inflate
        the same degradation."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-x",
                message_type="post",
                content={"task_id": "task-1", "text": "a post that keeps failing"},
            )

            def _boom(*_a, **_k):
                raise RuntimeError("knowledge write failed")

            seen_drop_ids: set[str] = set()

            def should_count(event_id: str) -> bool:
                if event_id in seen_drop_ids:
                    return False
                seen_drop_ids.add(event_id)
                return True

            monkeypatch.setattr(ks, "_record_post_locked", _boom)
            dropped = []
            first = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
                on_drop=dropped.append,
                drop_dedupe_fn=should_count,
            )
            second = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
                on_drop=dropped.append,
                drop_dedupe_fn=should_count,
            )

            assert first == 0
            assert second == 0
            assert dropped == [1]
        finally:
            ks.close()

    def test_drain_batches_into_single_transaction(self, tmp_path):
        """E1 (#949): N knowledge-row events drain in ONE writer-thread
        transaction, not N. Spying ``_run_write`` is the directest proxy —
        one round-trip == one commit/fsync."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            for i in range(4):
                bus.append(
                    round_num=0,
                    agent_id=f"agent-{i}",
                    message_type="post",
                    content={"task_id": "task-1", "text": f"post {i}"},
                )

            calls = {"n": 0}
            orig = ks._run_write

            def _counting(fn):
                calls["n"] += 1
                return orig(fn)

            ks._run_write = _counting  # type: ignore[method-assign]
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            assert count == 4
            # Exactly one writer-thread dispatch for all four post writes.
            assert calls["n"] == 1
            result = ks.query_task("task-1", entry_types=["post"])
            assert len(result["discussion"]) == 4
        finally:
            ks.close()

    def test_drain_savepoint_isolation_one_failure_others_land(self, tmp_path):
        """A single failing event rolls back to its SAVEPOINT only; the other
        events in the same batch still commit (preserves #740 partial-drain)."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            for i in range(3):
                bus.append(
                    round_num=0,
                    agent_id=f"agent-{i}",
                    message_type="post",
                    content={"task_id": "task-1", "text": f"post {i}"},
                )

            orig_locked = ks._record_post_locked

            def _selective(**kwargs):
                if kwargs.get("text") == "post 1":
                    raise RuntimeError("middle write fails")
                return orig_locked(**kwargs)

            ks._record_post_locked = _selective  # type: ignore[method-assign]
            dropped = []
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
                on_drop=dropped.append,
            )
            assert count == 2  # post 0 and post 2 landed
            assert dropped == [1]  # post 1 surfaced as a drop
            texts = {p["text"] for p in ks.query_task("task-1", entry_types=["post"])["discussion"]}
            assert texts == {"post 0", "post 2"}
        finally:
            ks.close()

    def test_drain_folds_done_signals_into_single_transaction(self, tmp_path):
        """E1 (#949): ``done``/signal_done control rows are folded into the
        SAME batched transaction as the post/insight writes — the whole drain
        is one writer-thread dispatch, and discussion_done is still populated."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-1",
                message_type="post",
                content={"task_id": "task-A", "text": "a post"},
            )
            bus.append(
                round_num=0,
                agent_id="agent-1",
                message_type="done",
                content={"task_ids": ["task-A", "task-B"]},
            )

            calls = {"n": 0}
            orig = ks._run_write

            def _counting(fn):
                calls["n"] += 1
                return orig(fn)

            ks._run_write = _counting  # type: ignore[method-assign]
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=3,
                experiment="test",
            )
            # One post landed (done events never count); the post AND both
            # signal_done rows committed in a single dispatch.
            assert count == 1
            assert calls["n"] == 1
            for tid in ("task-A", "task-B"):
                status = ks.get_done_status(task_id=tid, generation=3, expected_agents=1, experiment="test")
                assert status["agents_done"] == 1
        finally:
            ks.close()

    def test_drain_on_drop_not_called_on_clean_drain(self, tmp_path):
        """on_drop must NOT fire when every event lands — a healthy drain stays
        silent so a clean generation isn't flagged degraded (#740)."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-ok",
                message_type="post",
                content={"task_id": "task-ok", "text": "a post that lands fine"},
            )
            dropped = []
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
                on_drop=dropped.append,
            )
            assert count == 1
            assert dropped == []
        finally:
            ks.close()

    def test_drain_on_drop_fires_when_stale_event_read_fails(self, tmp_path, monkeypatch):
        """If stale-id loading fails, stale retry events may be ingested; surface that."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-stale",
                message_type="post",
                content={"task_id": "task-stale", "text": "possibly stale"},
            )

            def _boom():
                raise RuntimeError("stale sidecar unavailable")

            monkeypatch.setattr(bus, "read_stale_event_ids", _boom)
            dropped = []
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
                on_drop=dropped.append,
            )
            assert count == 1
            assert dropped == [1]
        finally:
            ks.close()

    def test_drain_persists_raw_forum_events_when_store_provided(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        store = MemoryStore(str(tmp_path / "memory_events.sqlite"), default_experiment="test")
        try:
            bus.append(
                round_num=0,
                agent_id="agent-raw",
                message_type="post",
                content={
                    "task_id": "task-raw",
                    "text": "raw forum text",
                },
            )
            bus.append(
                round_num=0,
                agent_id="agent-raw",
                message_type="done",
                content={"task_ids": ["task-raw"]},
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=2,
                experiment="test",
                forum_store=store,
            )
            assert count == 1
            rows = store.list_forum_messages(2, experiment="test")
            assert [row["message_type"] for row in rows] == ["done", "post"]
            assert "raw forum text" in rows[1]["content"]
        finally:
            store.close()
            ks.close()

    def test_drain_comment_events(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-3",
                message_type="comment",
                content={
                    "task_id": "task-10",
                    "text": "Good point, but consider edge cases",
                    "parent_post_id": 5,
                },
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            assert count == 1
            result = ks.query_task("task-10", entry_types=["post"])
            assert len(result["discussion"]) >= 1
        finally:
            ks.close()

    def test_done_events_do_not_increment_count(self, tmp_path):
        """Done events aren't counted as knowledge rows but still persist
        to ``discussion_done`` when task_ids are present."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-1",
                message_type="done",
                content={},
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            # count tracks knowledge rows (insights/posts/comments); done
            # events are control signals and remain unaccounted by count.
            assert count == 0
        finally:
            ks.close()

    def test_drain_done_persists_to_discussion_done(self, tmp_path):
        """Regression: forum_signal_done events with task_ids must land in
        ``discussion_done`` so audits can detect early termination."""
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-done",
                message_type="done",
                content={"task_ids": ["task-A", "task-B"]},
            )
            _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=4,
                experiment="test",
            )
            status_a = ks.get_done_status(
                task_id="task-A",
                generation=4,
                expected_agents=1,
                experiment="test",
            )
            status_b = ks.get_done_status(
                task_id="task-B",
                generation=4,
                expected_agents=1,
                experiment="test",
            )
            assert status_a["agents_done"] == 1
            assert status_b["agents_done"] == 1
        finally:
            ks.close()

    def test_drain_done_without_task_ids_persists_as_cross_task(self, tmp_path):
        """Regression: a done event with an empty ``task_ids`` list is a
        cross-task forum signal (cross-task rooms don't scope agents to any
        task). It must land in ``discussion_done`` under ``CROSS_TASK_SENTINEL``
        so analytics can distinguish cross-task completions from per-task
        completions. Previously the drain filter skipped these rows entirely,
        causing ~70% undercount of forum_signal_done signals in sweeps."""
        from kcsi.memory.knowledge_store import CROSS_TASK_SENTINEL

        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-cross",
                message_type="done",
                content={},
            )
            _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            status = ks.get_done_status(
                task_id=CROSS_TASK_SENTINEL,
                generation=1,
                expected_agents=1,
                experiment="test",
            )
            assert status["agents_done"] == 1
            # Unrelated task_ids are unaffected.
            other = ks.get_done_status(
                task_id="task-unrelated",
                generation=1,
                expected_agents=1,
                experiment="test",
            )
            assert other["agents_done"] == 0
        finally:
            ks.close()

    def test_drain_done_with_task_ids_does_not_write_cross_task_sentinel(self, tmp_path):
        """A per-task done event (non-empty ``task_ids``) must NOT also write
        a row under ``CROSS_TASK_SENTINEL`` — only cross-task signals land
        under the sentinel so analytics can separate the two scopes cleanly."""
        from kcsi.memory.knowledge_store import CROSS_TASK_SENTINEL

        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-per-task",
                message_type="done",
                content={"task_ids": ["task-A"]},
            )
            _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            per_task = ks.get_done_status(
                task_id="task-A",
                generation=1,
                expected_agents=1,
                experiment="test",
            )
            cross = ks.get_done_status(
                task_id=CROSS_TASK_SENTINEL,
                generation=1,
                expected_agents=1,
                experiment="test",
            )
            assert per_task["agents_done"] == 1
            assert cross["agents_done"] == 0
        finally:
            ks.close()

    def test_empty_bus_returns_zero(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            assert count == 0
        finally:
            ks.close()

    def test_multiple_agents_drained(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            # Agent 1 posts an insight
            bus.append(
                round_num=0,
                agent_id="agent-alpha",
                message_type="insight",
                content={
                    "text": "insight from alpha",
                    "scope": "global",
                    "confidence": "low",
                    "evidence_task_ids": ["task-1"],
                },
            )
            # Agent 2 posts a comment
            bus.append(
                round_num=0,
                agent_id="agent-beta",
                message_type="post",
                content={
                    "task_id": "task-2",
                    "text": "post from beta",
                },
            )
            # Agent 3 posts an insight
            bus.append(
                round_num=0,
                agent_id="agent-gamma",
                message_type="insight",
                content={
                    "text": "insight from gamma",
                    "scope": "task",
                    "confidence": "medium",
                    "evidence_task_ids": ["task-3"],
                },
            )
            # A done signal (should be skipped)
            bus.append(
                round_num=0,
                agent_id="agent-alpha",
                message_type="done",
                content={},
            )

            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            assert count == 3  # 2 insights + 1 post, done skipped
        finally:
            ks.close()

    def test_insight_without_evidence_uses_forum_task_id(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-1",
                message_type="insight",
                content={
                    "text": "general insight",
                    "scope": "global",
                    "confidence": "medium",
                },
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            assert count == 1
            # Should use __forum__ as task_id when no evidence_task_ids
            result = ks.query_task("__forum__", entry_types=["insight"])
            assert len(result["insights"]) >= 1
        finally:
            ks.close()

    def test_post_without_task_id_uses_forum(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            bus.append(
                round_num=0,
                agent_id="agent-1",
                message_type="post",
                content={
                    "text": "a post without task_id",
                },
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            assert count == 1
            result = ks.query_task("__forum__", entry_types=["post"])
            assert len(result["discussion"]) >= 1
        finally:
            ks.close()


# ---------------------------------------------------------------------------
# handle_forum_post tests — ForumBus-only, no KnowledgeStore write
# ---------------------------------------------------------------------------


class TestHandleForumPostBusOnly:
    def test_writes_to_forum_bus_only(self, tmp_path):
        """handle_forum_post should write to ForumBus, NOT KnowledgeStore."""
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            result = handle_forum_post(
                knowledge_store=ks,
                forum_bus=bus,
                task_id="task-50",
                text="hello world",
                agent_id="agent-1",
                generation=1,
                experiment="test",
            )
            assert result["status"] == "ok"
            assert result["task_id"] == "task-50"

            # ForumBus should have the event
            events = bus.read_events()
            assert len(events) == 1
            assert events[0].message_type == "post"
            assert events[0].content["text"] == "hello world"

            # KnowledgeStore should NOT have any post (drain not called yet)
            ks_result = ks.query_task("task-50", entry_types=["post"])
            assert len(ks_result["discussion"]) == 0
        finally:
            ks.close()

    def test_returns_event_id_from_bus(self, tmp_path):
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="test",
            agent_id="agent-1",
            generation=1,
        )
        assert result["entry_id"] is not None
        assert result["entry_id"].startswith("fb-")

    def test_no_bus_returns_none_entry_id(self, tmp_path):
        from kcsi.memory.mcp_server import handle_forum_post

        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=None,
            task_id="task-1",
            text="test",
            agent_id="agent-1",
            generation=1,
        )
        assert result["entry_id"] is None
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# handle_forum_post dedup tests — "one post per agent per task per round"
#
# The forum prompt instructs agents to post exactly one post-mortem per task
# per round (and exactly one message per round on the cross-task page), but
# prior to this guard nothing enforced it server-side: a second forum_post
# call from the same agent for the same task_id+round_num was silently
# accepted, writing a second ForumBus event that inflated that agent's
# weight in the distillation input. See issue #1044.
# ---------------------------------------------------------------------------


class TestHandleForumPostDedup:
    def test_second_post_same_agent_task_round_rejected(self, tmp_path):
        """A second forum_post from the same agent for the same task_id and
        round_num must be rejected, not silently accepted as a duplicate."""
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        first = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="first post-mortem",
            agent_id="agent-1",
            generation=1,
            round_num=0,
        )
        assert first["status"] == "ok"

        with pytest.raises(ValueError, match="already posted"):
            handle_forum_post(
                knowledge_store=None,
                forum_bus=bus,
                task_id="task-1",
                text="second post-mortem (duplicate)",
                agent_id="agent-1",
                generation=1,
                round_num=0,
            )

        # Only the first post landed on the bus.
        events = [ev for ev in bus.read_events() if ev.message_type == "post"]
        assert len(events) == 1
        assert events[0].content["text"] == "first post-mortem"

    def test_single_post_per_agent_task_round_still_works(self, tmp_path):
        """Non-regression: a normal single post per agent/task/round is unaffected."""
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="only post",
            agent_id="agent-1",
            generation=1,
            round_num=0,
        )
        assert result["status"] == "ok"
        events = [ev for ev in bus.read_events() if ev.message_type == "post"]
        assert len(events) == 1

    def test_same_agent_different_round_not_blocked(self, tmp_path):
        """The dedup guard is scoped to (agent, task_id, round) — a later round
        from the same agent on the same task is a fresh post, not a duplicate."""
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="round 0 post",
            agent_id="agent-1",
            generation=1,
            round_num=0,
        )
        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="round 1 post",
            agent_id="agent-1",
            generation=1,
            round_num=1,
        )
        assert result["status"] == "ok"
        events = [ev for ev in bus.read_events() if ev.message_type == "post"]
        assert len(events) == 2

    def test_same_agent_different_task_not_blocked(self, tmp_path):
        """The dedup guard is per task_id — the same agent posting to a second
        task in the same round is expected behavior, not a duplicate."""
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="task-1 post",
            agent_id="agent-1",
            generation=1,
            round_num=0,
        )
        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-2",
            text="task-2 post",
            agent_id="agent-1",
            generation=1,
            round_num=0,
        )
        assert result["status"] == "ok"
        events = [ev for ev in bus.read_events() if ev.message_type == "post"]
        assert len(events) == 2

    def test_different_agent_same_task_round_not_blocked(self, tmp_path):
        """Non-regression: the guard is per-agent, not global to the page."""
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="agent-1 post",
            agent_id="agent-1",
            generation=1,
            round_num=0,
        )
        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="agent-2 post",
            agent_id="agent-2",
            generation=1,
            round_num=0,
        )
        assert result["status"] == "ok"
        events = [ev for ev in bus.read_events() if ev.message_type == "post"]
        assert len(events) == 2

    def test_retry_repost_after_stale_mark_not_blocked(self, tmp_path):
        """A forum-task retry (issue #541) re-posts for the same
        (agent_id, task_id, round_num) after the failed attempt's earlier
        post was marked stale. The dedup guard must not treat that stale
        event as a still-live "already posted" — otherwise every legitimate
        post-mortem retry for a task that succeeded earlier in the same
        forum-task attempt is permanently rejected."""
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        first = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="post from the attempt that later failed",
            agent_id="agent-1",
            generation=1,
            round_num=0,
        )
        assert first["status"] == "ok"
        bus.mark_stale([first["entry_id"]], reason="failed_attempt")

        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-1",
            text="retry's fresh post",
            agent_id="agent-1",
            generation=1,
            round_num=0,
        )
        assert result["status"] == "ok"
        events = [ev for ev in bus.read_events() if ev.message_type == "post"]
        assert len(events) == 2
        assert events[-1].content["text"] == "retry's fresh post"


# ---------------------------------------------------------------------------
# parent_post_id / reply_to coercion tests
# ---------------------------------------------------------------------------


class TestCoercePostRef:
    """Unit tests for ``_coerce_post_ref`` — the helper that normalizes
    parent/reply-to post references before they reach the INTEGER columns
    in ``knowledge.parent_id``/``reply_to``.
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            ("null", None),
            ("NULL", None),
            ("None", None),
            ("none", None),
            ("undefined", None),
            ("nil", None),
            ("", None),
            ("   ", None),
            ("not-a-number", None),
            (True, None),  # bools masquerading as ints are rejected
            (False, None),
            (5, 5),
            (0, 0),
            (-3, -3),
            ("7", 7),  # quoted integer from a JSON tool call
            (" 42 ", 42),  # whitespace-padded quoted integer
            (3.0, 3),
        ],
    )
    def test_coerce_post_ref(self, value, expected):
        assert _coerce_post_ref(value) == expected


class TestDrainStringParentIdCoercion:
    """Regression: drain must coerce string ``"null"`` parent_post_id to
    real SQL NULL so threaded-reply joins aren't silently orphaned.
    """

    def test_string_null_parent_post_id_stored_as_sql_null(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            # An LLM tool call occasionally emits the literal string "null"
            # (JSON-stringified) instead of a real integer/omission.  The
            # MCP schema declares this as an integer, but SQLite's type
            # affinity would otherwise store the string as TEXT.
            bus.append(
                round_num=0,
                agent_id="agent-junk",
                message_type="post",
                content={
                    "task_id": "task-x",
                    "text": "a post whose parent_post_id was sent as the string 'null'",
                    "parent_post_id": "null",
                },
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=3,
                experiment="test",
            )
            assert count == 1

            # Inspect the raw SQLite row — both the value AND typeof() must
            # be NULL so threading queries join correctly.
            conn = ks._connection()
            row = conn.execute(
                "SELECT parent_id, typeof(parent_id) FROM knowledge WHERE entry_type='post' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            assert row is not None
            assert row["parent_id"] is None
            assert row["typeof(parent_id)"] == "null"
        finally:
            ks.close()

    def test_quoted_integer_parent_post_id_coerced_to_int(self, tmp_path):
        bus = _make_forum_bus(tmp_path)
        ks = _make_knowledge_store(tmp_path)
        try:
            parent_id = ks.record_post(
                task_id="task-thread",
                agent_id="agent-root",
                generation=1,
                text="root post",
                experiment="test",
            )
            bus.append(
                round_num=0,
                agent_id="agent-replier",
                message_type="post",
                content={
                    "task_id": "task-thread",
                    "text": "a reply whose parent_post_id was sent as the string '1'",
                    "parent_post_id": str(parent_id),
                    "reply_to": str(parent_id),
                },
            )
            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="test",
            )
            assert count == 1

            conn = ks._connection()
            row = conn.execute(
                "SELECT parent_id, reply_to, typeof(parent_id), typeof(reply_to) "
                "FROM knowledge WHERE entry_type='post' AND agent_id='agent-replier'"
            ).fetchone()
            assert row is not None
            assert row["parent_id"] == parent_id
            assert row["reply_to"] == parent_id
            assert row["typeof(parent_id)"] == "integer"
            assert row["typeof(reply_to)"] == "integer"
        finally:
            ks.close()


class TestHandleForumPostCoercion:
    """``handle_forum_post`` also coerces junk parent_post_id inputs so
    ForumBus events never carry the string "null" forward to the drain.
    """

    def test_string_null_parent_post_id_normalized_at_mcp_layer(self, tmp_path):
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-42",
            text="test",
            parent_post_id="null",  # type: ignore[arg-type]
            agent_id="agent-1",
            generation=1,
        )
        assert result["status"] == "ok"

        events = bus.read_events()
        assert len(events) == 1
        payload = events[0].content
        assert payload["parent_post_id"] is None
        assert "reply_to" not in payload

    def test_quoted_integer_parent_post_id_normalized_at_mcp_layer(self, tmp_path):
        from kcsi.memory.mcp_server import handle_forum_post

        bus = _make_forum_bus(tmp_path)
        handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="task-42",
            text="test",
            parent_post_id="7",  # type: ignore[arg-type]
            agent_id="agent-1",
            generation=1,
        )
        events = bus.read_events()
        assert len(events) == 1
        assert events[0].content["parent_post_id"] == 7
        assert events[0].content["reply_to"] == 7
