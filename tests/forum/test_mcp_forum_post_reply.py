"""Tests for the MCP forum_post handler surfacing parent_post_id as reply_to."""

import json
import tempfile
from pathlib import Path

from kcsi.memory.forum_bus import ForumBus
from kcsi.memory.mcp_server import handle_forum_post


def _make_bus(tmp: str):
    db_path = Path(tmp) / "k.sqlite"
    db_path.touch()
    return ForumBus(db_path=str(db_path), experiment="test-exp", generation=0)


def test_handle_forum_post_passes_reply_to_through_parent_post_id():
    with tempfile.TemporaryDirectory() as tmp:
        bus = _make_bus(tmp)
        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="t1",
            text="hi there",
            parent_post_id=7,
            agent_id="a1",
            generation=0,
        )
        assert result["status"] == "ok"
        events = list(Path(tmp).glob("**/*.events.jsonl"))
        assert events, "ForumBus did not create events.jsonl"
        payload = json.loads(events[0].read_text().strip().splitlines()[0])
        assert payload["content"]["reply_to"] == 7


def test_handle_forum_post_without_parent_post_id_omits_reply_to():
    with tempfile.TemporaryDirectory() as tmp:
        bus = _make_bus(tmp)
        handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="t1",
            text="plain post",
            parent_post_id=None,
            agent_id="a1",
            generation=0,
        )
        events = list(Path(tmp).glob("**/*.events.jsonl"))
        payload = json.loads(events[0].read_text().strip().splitlines()[0])
        # No reply_to means either absent or None — either is fine downstream.
        assert payload["content"].get("reply_to") is None
