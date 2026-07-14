"""Tests for unified KnowledgeStore MCP tool handlers."""

from __future__ import annotations

from unittest.mock import MagicMock

from kcsi.memory.forum_bus import ForumBus
from kcsi.memory.knowledge_store import KnowledgeStore
from kcsi.memory.mcp_server import (
    _build_tools,
    handle_forum_post,
    handle_forum_signal_done,
    handle_forum_signal_done_v2,
    handle_knowledge,
)
from kcsi.orchestrator.engine import _drain_forum_bus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_knowledge_store(tmp_path, *, experiment: str = "exp1") -> KnowledgeStore:
    db_path = str(tmp_path / "knowledge.sqlite")
    ks = KnowledgeStore(db_path, default_experiment=experiment)
    return ks


# ---------------------------------------------------------------------------
# handle_knowledge
# ---------------------------------------------------------------------------


class TestHandleKnowledge:
    def test_returns_correct_structure(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)
        try:
            ks.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                native_score=0.5,
                model_output="some output",
                experiment="exp1",
            )
            result = handle_knowledge(
                knowledge_store=ks,
                task_id="task-1",
                experiment="exp1",
            )
            assert result["task_id"] == "task-1"
            assert "attempts" in result
            assert "discussion" in result
            assert "insights" in result
            assert "distilled" in result
            assert len(result["attempts"]) == 1
        finally:
            ks.close()

    def test_with_include_filter_attempts(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)
        try:
            ks.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                experiment="exp1",
            )
            ks.record_post(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="hello",
                experiment="exp1",
            )
            result = handle_knowledge(
                knowledge_store=ks,
                task_id="task-1",
                include="attempts",
                experiment="exp1",
            )
            assert len(result["attempts"]) == 1
            assert len(result["discussion"]) == 0
        finally:
            ks.close()

    def test_with_include_filter_discussion(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)
        try:
            ks.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                experiment="exp1",
            )
            ks.record_post(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="hello",
                experiment="exp1",
            )
            result = handle_knowledge(
                knowledge_store=ks,
                task_id="task-1",
                include="discussion",
                experiment="exp1",
            )
            assert len(result["attempts"]) == 0
            assert len(result["discussion"]) == 1
        finally:
            ks.close()

    def test_with_include_filter_insights(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)
        try:
            ks.record_insight(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="important finding",
                experiment="exp1",
            )
            result = handle_knowledge(
                knowledge_store=ks,
                task_id="task-1",
                include="insights",
                experiment="exp1",
            )
            assert len(result["insights"]) == 1
            assert len(result["attempts"]) == 0
        finally:
            ks.close()

    def test_with_include_filter_distilled(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)
        try:
            ks.record_distillation(
                task_id="task-1",
                generation=1,
                assets=[{"asset_type": "tip", "text": "use caching"}],
                experiment="exp1",
            )
            result = handle_knowledge(
                knowledge_store=ks,
                task_id="task-1",
                include="distilled",
                experiment="exp1",
            )
            assert len(result["distilled"]) == 1
            assert len(result["attempts"]) == 0
        finally:
            ks.close()

    def test_none_store_returns_empty(self):
        result = handle_knowledge(
            knowledge_store=None,
            task_id="task-1",
        )
        assert result["task_id"] == "task-1"
        assert result["attempts"] == []
        assert result["discussion"] == []
        assert result["insights"] == []
        assert result["distilled"] == []

    def test_excluded_task_returns_empty_without_store_query(self):
        ks = MagicMock()

        result = handle_knowledge(
            knowledge_store=ks,
            task_id="conv5__q1",
            experiment="exp1",
            exclude_task_ids=frozenset({"conv5__q1"}),
        )

        assert result == {
            "task_id": "conv5__q1",
            "attempts": [],
            "discussion": [],
            "insights": [],
            "distilled": [],
        }
        ks.query_task.assert_not_called()


# ---------------------------------------------------------------------------
# handle_forum_post
# ---------------------------------------------------------------------------


class TestHandleForumPost:
    def test_writes_to_forum_bus_only(self, tmp_path):
        """handle_forum_post writes to ForumBus only, not KnowledgeStore directly."""
        ks = _make_knowledge_store(tmp_path)
        bus = ForumBus(
            db_path=str(tmp_path / "forum.sqlite"),
            experiment="exp1",
            generation=1,
        )
        try:
            result = handle_forum_post(
                knowledge_store=ks,
                forum_bus=bus,
                task_id="task-1",
                text="My observation",
                agent_id="agent-0",
                generation=1,
                experiment="exp1",
            )
            assert result["status"] == "ok"
            assert result["task_id"] == "task-1"
            assert result["entry_id"] is not None
            # entry_id should be a ForumBus event_id (fb- prefix)
            assert result["entry_id"].startswith("fb-")

            # KnowledgeStore should NOT have the post yet (no drain)
            page = ks.query_task("task-1", experiment="exp1")
            assert len(page["discussion"]) == 0

            # ForumBus should have the event
            events = bus.read_events()
            assert len(events) == 1
            assert events[0].message_type == "post"
            assert events[0].content["text"] == "My observation"

            # After draining, KnowledgeStore should have the post
            _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="exp1",
            )
            page = ks.query_task("task-1", experiment="exp1")
            assert len(page["discussion"]) == 1
            assert page["discussion"][0]["text"] == "My observation"
        finally:
            ks.close()

    def test_with_parent_post_id(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)
        bus = ForumBus(
            db_path=str(tmp_path / "forum.sqlite"),
            experiment="exp1",
            generation=1,
        )
        try:
            # First post
            r1 = handle_forum_post(
                knowledge_store=ks,
                forum_bus=bus,
                task_id="task-1",
                text="Original post",
                agent_id="agent-0",
                generation=1,
                experiment="exp1",
            )
            # Reply
            r2 = handle_forum_post(
                knowledge_store=ks,
                forum_bus=bus,
                task_id="task-1",
                text="Reply to original",
                parent_post_id=r1["entry_id"],
                agent_id="agent-1",
                generation=1,
                experiment="exp1",
            )
            assert r2["status"] == "ok"

            # Drain bus into KnowledgeStore
            _drain_forum_bus(
                forum_bus=bus,
                knowledge=ks,
                generation=1,
                experiment="exp1",
            )

            page = ks.query_task("task-1", experiment="exp1")
            assert len(page["discussion"]) == 2
        finally:
            ks.close()

    def test_none_stores_still_returns_ok(self):
        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=None,
            task_id="task-1",
            text="hello",
            agent_id="agent-0",
            generation=1,
        )
        assert result["status"] == "ok"
        assert result["entry_id"] is None


# ---------------------------------------------------------------------------
# handle_forum_signal_done
# ---------------------------------------------------------------------------


class TestHandleForumSignalDone:
    def test_v2_alias_points_to_canonical_handler(self):
        assert handle_forum_signal_done_v2 is handle_forum_signal_done

    def test_signals_done_on_all_task_ids(self, tmp_path):
        ks = _make_knowledge_store(tmp_path)
        bus = MagicMock()
        try:
            result = handle_forum_signal_done(
                knowledge_store=ks,
                forum_bus=bus,
                agent_id="agent-0",
                generation=1,
                task_ids={"task-1", "task-2"},
                experiment="exp1",
            )
            assert result["status"] == "done"
            assert result["agent_id"] == "agent-0"

            # Verify done status for each task
            for tid in ("task-1", "task-2"):
                status = ks.get_done_status(task_id=tid, generation=1, expected_agents=1, experiment="exp1")
                assert status["agents_done"] == 1
                assert status["all_done"] is True

            # Verify ForumBus.append called
            bus.append.assert_called_once()
            call_kwargs = bus.append.call_args[1]
            assert call_kwargs["message_type"] == "done"
        finally:
            ks.close()

    def test_none_stores(self):
        result = handle_forum_signal_done(
            knowledge_store=None,
            forum_bus=None,
            agent_id="agent-0",
            generation=1,
            task_ids={"task-1"},
        )
        assert result["status"] == "done"


# ---------------------------------------------------------------------------
# _build_tools toolset membership
# ---------------------------------------------------------------------------


class TestBuildToolsKnowledgeTools:
    def test_forum_toolset_includes_knowledge_tools(self):
        tools = _build_tools("forum")
        names = {t["name"] for t in tools}
        assert "knowledge" in names
        assert "forum_post" in names
        assert "forum_signal_done" in names
        # Also has the legacy tools
        assert "query" in names
        assert "forum_read" in names

    def test_all_toolset_includes_knowledge_tools(self):
        tools = _build_tools("all")
        names = {t["name"] for t in tools}
        assert "knowledge" in names
        assert "forum_post" in names
        assert "forum_signal_done" in names
        # Also has the legacy tools
        assert "query" in names
        assert "forum_read" in names

    def test_task_toolset_excludes_knowledge_tools(self):
        tools = _build_tools("task")
        names = {t["name"] for t in tools}
        assert "knowledge" not in names
        assert "forum_post" not in names
        assert "forum_signal_done" not in names
