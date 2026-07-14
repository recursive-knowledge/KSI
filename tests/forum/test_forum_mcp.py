"""Tests for forum MCP tool handlers."""

import io
import json
import os
from unittest import mock

from ksi.memory.forum_bus import ForumBus
from ksi.memory.mcp_server import (
    handle_forum_post,
    handle_forum_read,
)
from ksi.memory.mcp_server import (
    main as mcp_main,
)
from ksi.memory.store import MemoryStore


def test_forum_post_and_read(tmp_path):
    """Post insight messages and read them back."""
    db_path = str(tmp_path / "test.sqlite")
    store = MemoryStore(db_path)
    bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="default", generation=1)
    bus.clear()
    try:
        bus.append(
            round_num=None,
            agent_id="agent-0",
            message_type="insight",
            content={"text": "Django caching lesson", "scope": "task", "evidence_task_ids": ["task-0"]},
        )
        bus.append(
            round_num=None,
            agent_id="agent-1",
            message_type="insight",
            content={"text": "Testing coverage lesson", "scope": "meta", "evidence_task_ids": ["task-1"]},
        )
        result = handle_forum_read(forum_bus=bus)
        assert len(result) == 2
        agent_ids = {r["agent_id"] for r in result}
        assert agent_ids == {"agent-0", "agent-1"}
        assert all(r["message_type"] == "insight" for r in result)
    finally:
        store.close()


def test_forum_read_includes_unified_post_events(tmp_path):
    """Unified forum_post writes message_type=post; readers must surface it."""
    bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="default", generation=1)
    bus.clear()
    bus.append(
        round_num=0,
        agent_id="agent-0",
        message_type="post",
        content={"task_id": "task-0", "text": "unified forum post"},
    )
    result = handle_forum_read(forum_bus=bus)
    assert len(result) == 1
    assert result[0]["message_type"] == "post"
    assert result[0]["content"]["text"] == "unified forum post"


def test_forum_read_skips_malformed_scalar_rows(tmp_path):
    bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="default", generation=1)
    bus.clear()
    bus.append(round_num=1, agent_id="agent-0", message_type="post", content={"text": "good-1"})
    with bus._events_path.open("a", encoding="utf-8") as fp:
        fp.write(
            json.dumps(
                {
                    "event_id": "bad-round",
                    "generation": 1,
                    "round_num": "not-an-int",
                    "agent_id": "agent-1",
                    "message_type": "post",
                    "content": {"text": "bad-round"},
                }
            )
            + "\n"
        )
        fp.write(
            json.dumps(
                {
                    "event_id": "bad-generation",
                    "generation": "not-an-int",
                    "round_num": 1,
                    "agent_id": "agent-2",
                    "message_type": "post",
                    "content": {"text": "bad-generation"},
                }
            )
            + "\n"
        )
    bus.append(round_num=1, agent_id="agent-3", message_type="post", content={"text": "good-2"})

    result = handle_forum_read(forum_bus=bus, round_num=1)

    assert [row["content"]["text"] for row in result] == ["good-2", "good-1"]


def test_forum_read_up_to_round_includes_prior_rounds(tmp_path):
    """Round-2 reads should include round-1 posts for comment targeting."""
    db_path = str(tmp_path / "test.sqlite")
    store = MemoryStore(db_path)
    bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="default", generation=1)
    bus.clear()
    try:
        bus.append(
            round_num=1,
            agent_id="agent-0",
            message_type="insight",
            content={
                "insight_id": "ins-agent-0-r1",
                "text": "r1 insight",
                "scope": "task",
                "evidence_task_ids": ["task-0"],
            },
        )
        bus.append(
            round_num=2,
            agent_id="agent-1",
            message_type="comment",
            content={
                "comment_id": "c-agent-1-r2",
                "target_insight_id": "ins-agent-0-r1",
                "text": "r2 comment",
                "referenced_insight_ids": ["ins-agent-0-r1"],
            },
        )
        only_r2 = handle_forum_read(forum_bus=bus, round_num=2, up_to_round=False)
        up_to_r2 = handle_forum_read(forum_bus=bus, round_num=2, up_to_round=True)
        assert len(only_r2) == 1
        assert only_r2[0]["message_type"] == "comment"
        assert len(up_to_r2) == 2
        assert any(m["message_type"] == "insight" for m in up_to_r2)
        assert any(m["message_type"] == "comment" for m in up_to_r2)
    finally:
        store.close()


def test_forum_read_accepts_deprecated_forum_store_param(tmp_path):
    """handle_forum_read accepts (but ignores) deprecated forum_store param."""
    db_path = str(tmp_path / "test.sqlite")
    bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="default", generation=1)
    bus.clear()
    bus.append(
        round_num=None,
        agent_id="agent-0",
        message_type="insight",
        content={"text": "test insight", "scope": "task", "evidence_task_ids": ["task-0"]},
    )
    result = handle_forum_read(
        forum_bus=bus,
        round_num=None,
        forum_store=None,  # deprecated param
        generation=1,
        experiment="default",
    )
    assert len(result) == 1


def test_round0_task_exec_forum_insight_includes_targetable_insight_id(tmp_path):
    """Round-0 task-exec insight payload should include insight_id for round-2 targeting."""
    db_path = str(tmp_path / "test.sqlite")
    store = MemoryStore(db_path)
    try:
        store.insert_forum_message(
            generation=1,
            agent_id="agent-0",
            message_type="insight",
            round_num=0,
            experiment="default",
            content={
                "insight_id": "ins-round0-target",
                "scope": "task",
                "text": "task-exec insight",
                "evidence_task_ids": ["task-0"],
                "source_phase": "task_exec",
            },
        )
        rows = store.list_forum_messages(1, experiment="default")

        def _to_int(value, default=-1):
            try:
                if value is None:
                    return default
                return int(value)
            except Exception:
                return default

        round0 = [r for r in rows if _to_int(r.get("round_num"), -1) == 0]
        assert len(round0) == 1
        raw = round0[0].get("content")
        payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
        assert payload.get("insight_id") == "ins-round0-target"
    finally:
        store.close()


def test_forum_signal_done_v2_embeds_task_ids_in_bus(tmp_path):
    """``handle_forum_signal_done`` must embed task_ids in the ForumBus
    event so the orchestrator drain can persist them to ``discussion_done``.

    This guards the end-to-end path: container-side MCP handler → ForumBus
    → host-side drain → ``discussion_done`` table. Before the fix, done
    events were written with empty content and the drain explicitly
    skipped them, so audits saw 0 signal_done across every generation.
    """
    from ksi.memory.knowledge_store import KnowledgeStore
    from ksi.memory.mcp_server import handle_forum_signal_done

    bus = ForumBus(
        db_path=str(tmp_path / "memory.sqlite"),
        experiment="default",
        generation=2,
    )
    bus.clear()
    ks_ro = KnowledgeStore(
        str(tmp_path / "knowledge.sqlite"),
        default_experiment="default",
        read_only=False,
    )
    try:
        # Simulate the container path: read-only KnowledgeStore (the
        # container holds this), writable ForumBus (JSONL bind-mount).
        ks_ro._read_only = True  # force the container-side guard path
        result = handle_forum_signal_done(
            knowledge_store=ks_ro,
            forum_bus=bus,
            agent_id="agent-alpha",
            generation=2,
            task_ids={"task-1", "task-2"},
            experiment="default",
            round_num=0,
        )
        assert result["status"] == "done"
        assert result["ks_persisted"] is False  # read-only guard held
        assert result["bus_wrote"] is True
        assert set(result["task_ids"]) == {"task-1", "task-2"}

        events = bus.read_events()
        done_events = [e for e in events if e.message_type == "done"]
        assert len(done_events) == 1
        assert set(done_events[0].content.get("task_ids", [])) == {"task-1", "task-2"}
    finally:
        ks_ro.close()


def test_forum_signal_done_v2_logs_invocation(tmp_path, caplog):
    """Telemetry: ``handle_forum_signal_done`` must log invocation and
    result so the next audit can distinguish "never called" from
    "called but swallowed"."""
    import logging

    from ksi.memory.mcp_server import handle_forum_signal_done

    bus = ForumBus(
        db_path=str(tmp_path / "memory.sqlite"),
        experiment="default",
        generation=1,
    )
    bus.clear()
    try:
        with caplog.at_level(logging.INFO, logger="ksi.memory.mcp_server"):
            handle_forum_signal_done(
                knowledge_store=None,
                forum_bus=bus,
                agent_id="agent-t1",
                generation=1,
                task_ids={"t1"},
                experiment="exp",
                round_num=0,
            )
        messages = [r.getMessage() for r in caplog.records]
        # Invocation log: carries agent/generation/task_count so we can
        # count how many agents ever called the tool in a run.
        assert any("forum_signal_done invoked" in m and "agent=agent-t1" in m for m in messages)
        # Outcome log: ks_persisted/bus_wrote lets us tell "never called"
        # from "called but write path failed".
        assert any("forum_signal_done result" in m and "bus_wrote=True" in m for m in messages)
    finally:
        pass


def test_forum_post_cross_task_round1_requires_evidence_task_ids(tmp_path):
    """Cross-task R1 must carry non-empty evidence_task_ids in the JSON body.

    This is the §2.2 grounding contract wired into the live ``forum_post``
    tool: ungrounded cross-task insights are rejected at the MCP boundary
    rather than delegated to the agent.
    """
    bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="default", generation=1)
    bus.clear()

    # Missing evidence_task_ids → reject.
    try:
        handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="__cross_task__",
            text=json.dumps({"text": "Watch for hidden state mutations.", "scope": "meta"}),
            agent_id="agent-0",
            generation=1,
            round_num=1,
            allowed_task_ids={"task-0", "task-1"},
        )
        assert False, "Expected ValueError for missing evidence_task_ids"
    except ValueError as exc:
        assert "evidence_task_ids" in str(exc)

    # Empty evidence_task_ids → reject.
    try:
        handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="__cross_task__",
            text=json.dumps({"text": "Watch for hidden state mutations.", "evidence_task_ids": []}),
            agent_id="agent-0",
            generation=1,
            round_num=1,
            allowed_task_ids={"task-0", "task-1"},
        )
        assert False, "Expected ValueError for empty evidence_task_ids"
    except ValueError as exc:
        assert "evidence_task_ids" in str(exc)

    # Unknown task id → reject.
    try:
        handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="__cross_task__",
            text=json.dumps({"text": "Watch for hidden state mutations.", "evidence_task_ids": ["task-99"]}),
            agent_id="agent-0",
            generation=1,
            round_num=1,
            allowed_task_ids={"task-0", "task-1"},
        )
        assert False, "Expected ValueError for unknown evidence_task_id"
    except ValueError as exc:
        assert "Unknown evidence_task_ids" in str(exc)

    # Non-JSON body in cross-task R1 → reject.
    try:
        handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="__cross_task__",
            text="just a freeform sentence with no JSON",
            agent_id="agent-0",
            generation=1,
            round_num=1,
            allowed_task_ids={"task-0", "task-1"},
        )
        assert False, "Expected ValueError for non-JSON cross-task R1 body"
    except ValueError as exc:
        assert "non-JSON" in str(exc) or "evidence_task_ids" in str(exc)

    # Well-formed JSON with grounded evidence → accepted.
    result = handle_forum_post(
        knowledge_store=None,
        forum_bus=bus,
        task_id="__cross_task__",
        text=json.dumps({"text": "Watch for hidden state mutations.", "evidence_task_ids": ["task-0"]}),
        agent_id="agent-0",
        generation=1,
        round_num=1,
        allowed_task_ids={"task-0", "task-1"},
    )
    assert result["status"] == "ok"

    # Per-task posts in round 1 are not gated by this contract.
    result_per_task = handle_forum_post(
        knowledge_store=None,
        forum_bus=bus,
        task_id="task-0",
        text='{"load_bearing_assumption": "..."}',
        agent_id="agent-0",
        generation=1,
        round_num=1,
        allowed_task_ids={"task-0", "task-1"},
    )
    assert result_per_task["status"] == "ok"

    # Enforcement can be disabled. Uses a different agent_id from the prior
    # cross-task round-1 posts above — the one-post-per-agent-per-task-per-round
    # dedup guard (issue #1044) is an orthogonal concern from evidence
    # enforcement, and agent-0 already has a round-1 __cross_task__ post.
    result_off = handle_forum_post(
        knowledge_store=None,
        forum_bus=bus,
        task_id="__cross_task__",
        text="freeform",
        agent_id="agent-1",
        generation=1,
        round_num=1,
        allowed_task_ids={"task-0", "task-1"},
        enforce_evidence=False,
    )
    assert result_off["status"] == "ok"


def test_forum_post_cross_task_round1_accepts_prompt_schema_body(tmp_path):
    """A body that exactly follows the cross-task prompt schema is accepted.

    Regression for review finding 621-1: the agent-facing schema in
    ``build_cross_task_discussion_parts`` and the ``handle_forum_post``
    enforcement must agree on ``evidence_task_ids``. A prompt-faithful body
    (all five top-level fields, including ``evidence_task_ids``) must pass
    on the first attempt under ``enforce_evidence=True``. This test bridges
    the prompt and the enforcement so they cannot drift apart silently.
    """
    bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="default", generation=1)
    bus.clear()

    prompt_conforming_body = {
        "concrete_primitive": ("rust .iter().map(|c| c.to_digit(10)).collect::<Option<Vec<_>>>()"),
        "task_grounding": {
            "task_id": "task-0",
            "where_it_appeared": (
                "I parsed each digit with to_digit(10) and collected into "
                "an Option<Vec<_>> to short-circuit on any non-digit char."
            ),
            "evidence_post_id": None,
        },
        "transfer_claim": (
            "AGREE with post #1: task-1's parser should collect into Option<Vec<_>> to fail fast on malformed input."
        ),
        "anti_meta_self_check": ("This names the exact collect turbofish, not generic 'error handling'."),
        "evidence_task_ids": ["task-0"],
    }

    result = handle_forum_post(
        knowledge_store=None,
        forum_bus=bus,
        task_id="__cross_task__",
        text=json.dumps(prompt_conforming_body),
        agent_id="agent-0",
        generation=1,
        round_num=1,
        allowed_task_ids={"task-0", "task-1"},
        enforce_evidence=True,
    )
    assert result["status"] == "ok"


def test_mcp_tools_list_includes_forum_tools(tmp_path):
    """The MCP server should list all forum tools via JSON-RPC tools/list."""
    db = str(tmp_path / "test.sqlite")
    # Simulate a tools/list JSON-RPC request via stdin
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    fake_stdin = io.StringIO(request + "\n")
    fake_stdout = io.StringIO()

    env = {
        "KNOWLEDGE_DB_PATH": db,
        "FORUM_GENERATION": "0",
        "FORUM_AGENT_ID": "",
        "FORUM_EXPECTED_AGENTS": "0",
    }
    with mock.patch.dict(os.environ, env), mock.patch("sys.stdin", fake_stdin), mock.patch("sys.stdout", fake_stdout):
        mcp_main()

    output = fake_stdout.getvalue().strip()
    response = json.loads(output)
    assert response["id"] == 1
    tool_names = [t["name"] for t in response["result"]["tools"]]
    assert "forum_read" in tool_names
    assert "forum_post_insight" not in tool_names
    assert "forum_post_comment" not in tool_names
    assert "forum_get_status" not in tool_names
    # forum_signal_done is now included via unified knowledge tools
    assert "forum_signal_done" in tool_names
    assert "knowledge" in tool_names
    assert "forum_post" in tool_names
    # Also verify memory tools are present
    assert "query" in tool_names


def test_mcp_forum_round2_blocks_insight_posts(tmp_path):
    """Round-2 forum MCP policy should reject insight posting."""
    db = str(tmp_path / "test.sqlite")
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "forum_post_insight",
                "arguments": {"text": "should be blocked"},
            },
        }
    )
    fake_stdin = io.StringIO(request + "\n")
    fake_stdout = io.StringIO()
    env = {
        "KNOWLEDGE_DB_PATH": db,
        "FORUM_GENERATION": "1",
        "FORUM_ROUND": "2",
        "FORUM_AGENT_ID": "agent-0",
        "FORUM_EXPECTED_AGENTS": "2",
    }
    with mock.patch.dict(os.environ, env), mock.patch("sys.stdin", fake_stdin), mock.patch("sys.stdout", fake_stdout):
        mcp_main()

    response = json.loads(fake_stdout.getvalue().strip())
    assert response["id"] == 1
    assert "error" in response
    assert response["error"]["message"] == "Unknown tool: forum_post_insight"


def test_mcp_forum_round1_blocks_comment_posts(tmp_path):
    """Round-1 forum MCP policy should reject comment posting."""
    db = str(tmp_path / "test.sqlite")
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "forum_post_comment",
                "arguments": {
                    "target_insight_id": "ins-any",
                    "text": "should be blocked",
                },
            },
        }
    )
    fake_stdin = io.StringIO(request + "\n")
    fake_stdout = io.StringIO()
    env = {
        "KNOWLEDGE_DB_PATH": db,
        "FORUM_GENERATION": "1",
        "FORUM_ROUND": "1",
        "FORUM_AGENT_ID": "agent-0",
        "FORUM_EXPECTED_AGENTS": "2",
    }
    with mock.patch.dict(os.environ, env), mock.patch("sys.stdin", fake_stdin), mock.patch("sys.stdout", fake_stdout):
        mcp_main()

    response = json.loads(fake_stdout.getvalue().strip())
    assert response["id"] == 2
    assert "error" in response
    assert response["error"]["message"] == "Unknown tool: forum_post_comment"


def test_mcp_task_toolset_hides_forum_tools(tmp_path):
    """Task toolset should reject forum tool calls."""
    db = str(tmp_path / "test.sqlite")
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "forum_read",
                "arguments": {},
            },
        }
    )
    fake_stdin = io.StringIO(request + "\n")
    fake_stdout = io.StringIO()
    env = {
        "KNOWLEDGE_DB_PATH": db,
        "MCP_TOOLSET": "task",
    }
    with mock.patch.dict(os.environ, env), mock.patch("sys.stdin", fake_stdin), mock.patch("sys.stdout", fake_stdout):
        mcp_main()

    response = json.loads(fake_stdout.getvalue().strip())
    assert response["id"] == 3
    assert "error" in response
    assert "Unknown tool" in response["error"]["message"]


def test_mcp_forum_signal_done_succeeds_in_all_rounds(tmp_path):
    """forum_signal_done is now a unified knowledge tool, available in all rounds."""
    db = str(tmp_path / "test.sqlite")
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "forum_signal_done",
                "arguments": {},
            },
        }
    )
    fake_stdin = io.StringIO(request + "\n")
    fake_stdout = io.StringIO()
    env = {
        "KNOWLEDGE_DB_PATH": db,
        "FORUM_GENERATION": "1",
        "FORUM_ROUND": "1",
        "FORUM_AGENT_ID": "agent-0",
        "FORUM_EXPECTED_AGENTS": "2",
    }
    with mock.patch.dict(os.environ, env), mock.patch("sys.stdin", fake_stdin), mock.patch("sys.stdout", fake_stdout):
        mcp_main()

    response = json.loads(fake_stdout.getvalue().strip())
    assert response["id"] == 4
    assert "result" in response
    result = json.loads(response["result"]["content"][0]["text"])
    assert result["status"] == "done"
    assert result["agent_id"] == "agent-0"


def test_mcp_forum_post_dispatch_passes_allowed_task_ids():
    """Regression test: the JSON-RPC dispatch for `forum_post` must call
    `handle_forum_post(..., allowed_task_ids=forum_task_ids, ...)`.

    Without that kwarg, `handle_forum_post`'s round-1 cross-task grounding
    check (the `if allowed_task_ids:` branch that flags `Unknown
    evidence_task_ids`) is dead code at runtime — a future refactor that
    drops the kwarg would silently disable enforcement and no existing
    test of `handle_forum_post` alone would fail. This test guards the
    wire-up by reading the dispatch source directly.
    """
    import inspect

    from ksi.memory import mcp_server

    src = inspect.getsource(mcp_server._run_server)
    assert 'elif tool_name == "forum_post":' in src, "forum_post dispatch branch removed; update this regression test"
    # Slice between the forum_post elif and the next elif to scope the
    # assertion to the forum_post branch only.
    branch_start = src.index('elif tool_name == "forum_post":')
    rest = src[branch_start + len('elif tool_name == "forum_post":') :]
    next_branch = rest.find("elif tool_name ==")
    branch = rest[: next_branch if next_branch >= 0 else len(rest)]
    assert "handle_forum_post(" in branch, "forum_post branch no longer calls handle_forum_post"
    assert "allowed_task_ids=forum_task_ids" in branch, (
        "forum_post dispatch must pass `allowed_task_ids=forum_task_ids` "
        "or the round-1 cross-task grounding check is dead code"
    )
    assert "enforce_evidence=enforce_forum_protocol" in branch, (
        "forum_post dispatch must pass `enforce_evidence=enforce_forum_protocol` "
        "or the §2.2 grounding contract is not honored"
    )
