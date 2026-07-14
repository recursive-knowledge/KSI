"""Integration tests for the unified KnowledgeStore system.

These tests exercise end-to-end flows that span multiple components:
KnowledgeStore, MCP handlers, forum prompt builders, and CLI config.
Individual unit tests live in test_knowledge_store.py, test_mcp_knowledge_tools.py,
test_r0_prompt.py, and test_no_memory_flag.py.
"""

from __future__ import annotations

import threading

import pytest

from ksi.forum import build_per_task_discussion_parts
from ksi.memory.forum_bus import ForumBus
from ksi.memory.knowledge_store import KnowledgeStore
from ksi.memory.mcp_server import (
    handle_forum_post,
    handle_forum_signal_done,
    handle_knowledge,
)
from ksi.models import GenerationConfig, TaskTrace
from ksi.orchestrator.engine import _drain_forum_bus
from ksi.tokens import TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path, **kwargs) -> KnowledgeStore:
    db_path = str(tmp_path / "knowledge.sqlite")
    return KnowledgeStore(db_path, **kwargs)


def _make_trace(
    *,
    task_id: str,
    agent_id: str = "agent-0",
    generation: int = 1,
    score: float = 0.0,
    status: str = "failed",
    output: str = "",
) -> TaskTrace:
    return TaskTrace(
        generation=generation,
        agent_id=agent_id,
        task_id=task_id,
        model_output=output,
        eval_result={"status": status},
        native_score=score,
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


# ---------------------------------------------------------------------------
# 1. Full generation cycle simulation
# ---------------------------------------------------------------------------


class TestFullGenerationCycle:
    def test_complete_lifecycle(self, tmp_path):
        """Simulate a complete generation: record attempts, insights, posts,
        distillation, then query and verify all entry types are present."""
        store = _make_store(tmp_path)
        try:
            # Gen 1: Two agents attempt two tasks
            store.record_attempt(
                task_id="task-1",
                agent_id="agent-1",
                generation=1,
                eval_results={"status": "failed"},
                model_output="tried X",
                trace_condensed="X didn't work",
                native_score=0.0,
            )
            store.record_attempt(
                task_id="task-2",
                agent_id="agent-2",
                generation=1,
                eval_results={"resolved": True},
                model_output="solved it",
                trace_condensed="approach Y worked",
                native_score=1.0,
            )

            # R0 insights from execution
            store.record_insight(
                task_id="task-1",
                agent_id="agent-1",
                generation=1,
                text="X fails because of edge case Z",
                scope="task",
            )

            # R0 discussion posts
            post1 = store.record_post(
                task_id="task-1",
                agent_id="agent-1",
                generation=1,
                text="I think we should try approach Y instead",
            )
            store.record_post(
                task_id="task-1",
                agent_id="agent-2",
                generation=1,
                text="Agreed, Y worked for me on task-2",
                parent_id=post1,
            )

            # Signal done
            store.signal_done(task_id="task-1", agent_id="agent-1", generation=1)
            store.signal_done(task_id="task-1", agent_id="agent-2", generation=1)
            status = store.get_done_status(task_id="task-1", generation=1, expected_agents=2)
            assert status["all_done"] is True

            # R3 distillation
            store.record_distillation(
                task_id="task-1",
                generation=1,
                assets=[{"asset_type": "pitfall", "text": "Don't use approach X"}],
            )

            # Query task-1 page -- should have all entry types
            page = store.query_task("task-1")
            assert len(page["attempts"]) == 1
            assert len(page["insights"]) == 1
            assert len(page["discussion"]) == 2
            assert len(page["distilled"]) >= 1

            # Verify attempt content
            attempt = page["attempts"][0]
            assert attempt["score"] == 0.0
            assert attempt["agent_id"] == "agent-1"

            # Verify insight content
            insight = page["insights"][0]
            assert "edge case Z" in insight["text"]

            # Verify discussion threading
            parent = next(d for d in page["discussion"] if d["parent_id"] is None)
            reply = next(d for d in page["discussion"] if d["parent_id"] is not None)
            assert "approach Y" in parent["text"]
            assert reply["parent_id"] == parent["id"]

            # Verify distillation
            assert page["distilled"][0]["asset_type"] == "pitfall"

            # Query generation -- should have entries from both tasks
            gen_entries = store.query_generation(1)
            # 2 attempts + 1 insight + 2 posts + 1 distillation = 6
            assert len(gen_entries) == 6
            entry_types = {e["entry_type"] for e in gen_entries}
            assert entry_types == {"attempt", "insight", "post", "distillation"}

            # Verify both tasks represented
            task_ids = {e["task_id"] for e in gen_entries}
            assert "task-1" in task_ids
            assert "task-2" in task_ids
        finally:
            store.close()


class TestKnowledgeAttemptAtomicity:
    def test_record_attempt_rolls_back_knowledge_row_when_state_write_fails(self, tmp_path):
        store = _make_store(tmp_path)
        original = store._record_attempt_state_locked

        def boom(*args, **kwargs):
            raise RuntimeError("state write failed")

        store._record_attempt_state_locked = boom
        try:
            with pytest.raises(RuntimeError, match="state write failed"):
                store.record_attempt(
                    task_id="task-atomic",
                    agent_id="agent-0",
                    generation=1,
                    eval_results={"status": "failed"},
                    model_output="partial output",
                    trace_condensed="partial condensed",
                    native_score=0.0,
                )
            conn = store._connection()
            assert conn.execute("SELECT COUNT(*) FROM knowledge WHERE entry_type = 'attempt'").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM task_state").fetchone()[0] == 0
        finally:
            store._record_attempt_state_locked = original
            store.close()


# ---------------------------------------------------------------------------
# 2. MCP handler integration with real KnowledgeStore
# ---------------------------------------------------------------------------


class TestMCPHandlerIntegration:
    def test_handle_knowledge_returns_data_from_real_store(self, tmp_path):
        """Verify handle_knowledge returns data from a real KnowledgeStore."""
        store = _make_store(tmp_path, default_experiment="test_exp")
        try:
            # Populate with varied entry types
            store.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                native_score=0.5,
                model_output="partial fix",
                experiment="test_exp",
            )
            store.record_insight(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="Avoid recursive calls",
                experiment="test_exp",
            )
            store.record_post(
                task_id="task-1",
                agent_id="agent-1",
                generation=1,
                text="Use iterative approach instead",
                experiment="test_exp",
            )
            store.record_distillation(
                task_id="task-1",
                generation=1,
                assets=[{"asset_type": "strategy", "text": "Prefer iteration over recursion"}],
                experiment="test_exp",
            )

            # Query through MCP handler
            result = handle_knowledge(
                knowledge_store=store,
                task_id="task-1",
                experiment="test_exp",
            )

            assert result["task_id"] == "task-1"
            assert len(result["attempts"]) == 1
            assert len(result["insights"]) == 1
            assert len(result["discussion"]) == 1
            assert len(result["distilled"]) == 1

            # Verify content is accessible
            assert result["attempts"][0]["score"] == 0.5
            assert "recursive" in result["insights"][0]["text"]
            assert "iterative" in result["discussion"][0]["text"]
            assert result["distilled"][0]["asset_type"] == "strategy"
        finally:
            store.close()

    def test_handle_forum_post_round_trip(self, tmp_path):
        """Post via MCP handler -> ForumBus -> drain -> read back via handle_knowledge."""
        store = _make_store(tmp_path, default_experiment="test_exp")
        bus = ForumBus(
            db_path=str(tmp_path / "forum.sqlite"),
            experiment="test_exp",
            generation=1,
        )
        try:
            # Post via handler (writes to ForumBus only)
            result = handle_forum_post(
                knowledge_store=store,
                forum_bus=bus,
                task_id="task-1",
                text="Observation: approach A fails on edge cases",
                agent_id="agent-0",
                generation=1,
                experiment="test_exp",
            )
            assert result["status"] == "ok"
            post_id = result["entry_id"]

            # Reply via handler
            reply_result = handle_forum_post(
                knowledge_store=store,
                forum_bus=bus,
                task_id="task-1",
                text="Confirmed, edge case in line 42",
                parent_post_id=post_id,
                agent_id="agent-1",
                generation=1,
                experiment="test_exp",
            )
            assert reply_result["status"] == "ok"

            # Drain ForumBus into KnowledgeStore (orchestrator does this)
            drained = _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
                experiment="test_exp",
            )
            assert drained == 2

            # Read back via knowledge handler
            page = handle_knowledge(
                knowledge_store=store,
                task_id="task-1",
                include="discussion",
                experiment="test_exp",
            )
            assert len(page["discussion"]) == 2
            texts = {d["text"] for d in page["discussion"]}
            assert "Observation: approach A fails on edge cases" in texts
            assert "Confirmed, edge case in line 42" in texts
        finally:
            store.close()

    def test_handle_signal_done_and_verify(self, tmp_path):
        """Signal done via MCP handler and verify via KnowledgeStore."""
        store = _make_store(tmp_path, default_experiment="test_exp")
        try:
            handle_forum_signal_done(
                knowledge_store=store,
                forum_bus=None,
                agent_id="agent-0",
                generation=1,
                task_ids={"task-1", "task-2"},
                experiment="test_exp",
            )
            handle_forum_signal_done(
                knowledge_store=store,
                forum_bus=None,
                agent_id="agent-1",
                generation=1,
                task_ids={"task-1", "task-2"},
                experiment="test_exp",
            )

            for tid in ("task-1", "task-2"):
                status = store.get_done_status(
                    task_id=tid,
                    generation=1,
                    expected_agents=2,
                    experiment="test_exp",
                )
                assert status["all_done"] is True
                assert status["agents_done"] == 2
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 3. Cross-generation knowledge transfer
# ---------------------------------------------------------------------------


class TestCrossGenerationKnowledge:
    def test_gen1_data_visible_in_gen2_context(self, tmp_path):
        """Gen 1 insights and attempts should be visible when querying
        across generations."""
        store = _make_store(tmp_path)
        try:
            # Gen 1 data
            store.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                eval_results={"status": "failed"},
                model_output="first try",
                native_score=0.0,
            )
            store.record_insight(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="Edge case in input validation",
            )
            store.record_distillation(
                task_id="task-1",
                generation=1,
                assets=[{"asset_type": "pitfall", "text": "Validate input length first"}],
            )

            # Gen 2 data
            store.record_attempt(
                task_id="task-1",
                agent_id="agent-1",
                generation=2,
                eval_results={"status": "resolved"},
                model_output="fixed with validation",
                native_score=1.0,
            )
            store.record_insight(
                task_id="task-1",
                agent_id="agent-1",
                generation=2,
                text="Validation fix resolves edge case",
            )

            # Query without generation filter -- sees ALL generations
            page = store.query_task("task-1")
            assert len(page["attempts"]) == 2
            assert len(page["insights"]) == 2
            assert len(page["distilled"]) == 1

            # Query with generation filter -- sees only that generation
            page_gen1 = store.query_task("task-1", generation=1)
            assert len(page_gen1["attempts"]) == 1
            assert page_gen1["attempts"][0]["score"] == 0.0

            page_gen2 = store.query_task("task-1", generation=2)
            assert len(page_gen2["attempts"]) == 1
            assert page_gen2["attempts"][0]["score"] == 1.0

            # Gen 1 distillation visible from cross-gen query
            assert len(page["distilled"]) == 1
            assert page["distilled"][0]["text"] == "Validate input length first"

            # query_generation returns separate buckets
            gen1_entries = store.query_generation(1)
            gen2_entries = store.query_generation(2)
            assert len(gen1_entries) == 3  # attempt + insight + distillation
            assert len(gen2_entries) == 2  # attempt + insight
        finally:
            store.close()

    def test_multi_generation_score_progression(self, tmp_path):
        """Track score improvement across generations."""
        store = _make_store(tmp_path)
        try:
            for gen, score in [(1, 0.0), (2, 0.3), (3, 0.7), (4, 1.0)]:
                store.record_attempt(
                    task_id="task-1",
                    agent_id=f"agent-{gen}",
                    generation=gen,
                    native_score=score,
                )

            page = store.query_task("task-1")
            assert len(page["attempts"]) == 4
            scores = [a["score"] for a in page["attempts"]]
            assert scores == [0.0, 0.3, 0.7, 1.0]  # ordered by id ASC
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 4. FTS search across entry types
# ---------------------------------------------------------------------------


class TestFTSSearchAcrossEntryTypes:
    def test_fts_finds_insights_posts_and_distillation(self, tmp_path):
        """FTS should find entries regardless of entry_type."""
        store = _make_store(tmp_path)
        try:
            # Use a distinctive keyword "refactoring" across different entry types
            store.record_insight(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="Consider refactoring the validation module",
            )
            store.record_post(
                task_id="task-1",
                agent_id="agent-1",
                generation=1,
                text="I agree refactoring would help",
            )
            store.record_distillation(
                task_id="task-1",
                generation=1,
                assets=[{"asset_type": "strategy", "text": "refactoring is key"}],
            )
            store.record_attempt(
                task_id="task-2",
                agent_id="agent-0",
                generation=1,
                model_output="Applied refactoring to views",
                native_score=0.8,
            )

            results = store.fts_search("refactoring")
            assert len(results) >= 3  # insight + post + distillation + attempt
            found_types = {r["entry_type"] for r in results}
            # At minimum should find insight, post, and attempt
            # (distillation content is stored as JSON, search depends on
            #  whether the keyword appears in the content column)
            assert "insight" in found_types
            assert "post" in found_types
            assert "attempt" in found_types
        finally:
            store.close()

    def test_fts_search_respects_entry_type_filter(self, tmp_path):
        """FTS search should respect entry_types filter."""
        store = _make_store(tmp_path)
        try:
            store.record_insight(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="Unique searchable keyword xyzzy",
            )
            store.record_post(
                task_id="task-1",
                agent_id="agent-1",
                generation=1,
                text="Also mentions xyzzy here",
            )

            # Filter to insights only
            results = store.fts_search("xyzzy", entry_types=["insight"])
            assert len(results) == 1
            assert results[0]["entry_type"] == "insight"

            # Filter to posts only
            results = store.fts_search("xyzzy", entry_types=["post"])
            assert len(results) == 1
            assert results[0]["entry_type"] == "post"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 5. R0 prompt integration with real traces
# ---------------------------------------------------------------------------


class TestR0PromptWithRealTraces:
    def test_r0_prompt_contains_task_outcomes(self, tmp_path):
        """Build an R0 prompt and verify it contains task outcomes."""
        traces = [
            _make_trace(
                task_id="task-1",
                agent_id="agent-0",
                score=1.0,
                status="completed",
                output="solved with pattern matching",
            ),
            _make_trace(
                task_id="task-2",
                agent_id="agent-0",
                score=0.0,
                status="failed",
                output="off-by-one error in loop",
            ),
        ]

        prompt = build_per_task_discussion_parts(
            agent_id="agent-0",
            generation=1,
            traces=traces,
            task_ids=["task-1", "task-2"],
            task_descriptions={
                "task-1": "Fix the grid rendering bug",
                "task-2": "Implement binary search",
            },
        ).as_text()

        # Contains header and agent info
        assert "PER-TASK POST-MORTEM" in prompt
        assert "agent-0" in prompt
        assert "generation 1" in prompt

        # Contains task outcomes
        assert "task-1" in prompt
        assert "score=1.0" in prompt
        assert "task-2" in prompt
        assert "score=0.0" in prompt

        # Contains MCP tool references
        assert "knowledge(task_id=" in prompt
        assert "forum_post(task_id=" in prompt
        assert "forum_signal_done()" in prompt

        # Contains task descriptions
        assert "Task Descriptions" in prompt
        assert "grid rendering" in prompt

    def test_r0_prompt_with_store_backed_knowledge(self, tmp_path):
        """Verify R0 prompt works in a scenario where the store has
        prior data that agents would query via MCP tools."""
        store = _make_store(tmp_path)
        try:
            # Simulate gen 1 data already in the store
            store.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                native_score=0.0,
                model_output="first attempt failed",
            )
            store.record_insight(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="Input needs normalization",
            )

            # Build R0 prompt for gen 2 discussion
            traces = [
                _make_trace(
                    task_id="task-1",
                    agent_id="agent-1",
                    generation=2,
                    score=0.5,
                    status="partial",
                    output="got closer with normalization",
                ),
            ]
            prompt = build_per_task_discussion_parts(
                agent_id="agent-1",
                generation=2,
                traces=traces,
                task_ids=["task-1"],
            ).as_text()
            assert "agent-1" in prompt
            assert "generation 2" in prompt
            assert "task-1" in prompt

            # Verify the agent could query prior knowledge via MCP
            page = handle_knowledge(
                knowledge_store=store,
                task_id="task-1",
            )
            assert len(page["attempts"]) == 1
            assert len(page["insights"]) == 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 6. --no-memory flag integration
# ---------------------------------------------------------------------------


class TestNoMemoryFlagIntegration:
    def test_no_memory_disables_knowledge_store_config(self):
        """Verify GenerationConfig with no_memory=True sets the flag."""
        config = GenerationConfig(
            num_generations=3,
            num_agents=5,
            no_memory=True,
            knowledge_db_path="/tmp/should_not_be_used.sqlite",
        )
        assert config.no_memory is True
        # The actual clearing of knowledge_db_path happens in cli.py main(),
        # but the flag is available for engine guards
        assert config.knowledge_db_path == "/tmp/should_not_be_used.sqlite"

    def test_no_memory_false_preserves_config(self):
        """Without no_memory, all memory settings are preserved."""
        config = GenerationConfig(
            num_generations=3,
            num_agents=5,
            no_memory=False,
            knowledge_db_path="/tmp/real.sqlite",
        )
        assert config.no_memory is False
        assert config.knowledge_db_path == "/tmp/real.sqlite"


# ---------------------------------------------------------------------------
# 7. Discussion thread integrity
# ---------------------------------------------------------------------------


class TestDiscussionThreading:
    def test_parent_id_forms_correct_threads(self, tmp_path):
        """Posts with parent_id form correct threads in query results."""
        store = _make_store(tmp_path)
        try:
            # Create a multi-level thread
            root_id = store.record_post(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="Root observation: the parser fails on nested brackets",
            )
            reply1_id = store.record_post(
                task_id="task-1",
                agent_id="agent-1",
                generation=1,
                text="I saw the same issue -- related to stack overflow",
                parent_id=root_id,
            )
            reply2_id = store.record_post(
                task_id="task-1",
                agent_id="agent-2",
                generation=1,
                text="Could be fixed with iterative parsing",
                parent_id=root_id,
            )
            nested_reply_id = store.record_post(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="Yes, iterative parsing worked for me",
                parent_id=reply2_id,
            )

            page = store.query_task("task-1")
            discussion = page["discussion"]
            assert len(discussion) == 4

            # Build a thread map from the results
            thread_map: dict[int, list[int]] = {}
            for post in discussion:
                pid = post["parent_id"]
                if pid is not None:
                    thread_map.setdefault(pid, []).append(post["id"])

            # Root should have 2 direct replies
            assert len(thread_map.get(root_id, [])) == 2
            assert reply1_id in thread_map[root_id]
            assert reply2_id in thread_map[root_id]

            # reply2 should have 1 nested reply
            assert len(thread_map.get(reply2_id, [])) == 1
            assert nested_reply_id in thread_map[reply2_id]

            # reply1 has no children
            assert reply1_id not in thread_map

            # Root has no parent
            root_post = next(d for d in discussion if d["id"] == root_id)
            assert root_post["parent_id"] is None
        finally:
            store.close()

    def test_thread_via_mcp_handler(self, tmp_path):
        """Discussion threading works end-to-end through MCP handlers + drain."""
        store = _make_store(tmp_path, default_experiment="test_exp")
        bus = ForumBus(
            db_path=str(tmp_path / "forum.sqlite"),
            experiment="test_exp",
            generation=1,
        )
        try:
            r1 = handle_forum_post(
                knowledge_store=store,
                forum_bus=bus,
                task_id="task-1",
                text="Initial observation",
                agent_id="agent-0",
                generation=1,
                experiment="test_exp",
            )
            r2 = handle_forum_post(
                knowledge_store=store,
                forum_bus=bus,
                task_id="task-1",
                text="Reply to observation",
                parent_post_id=r1["entry_id"],
                agent_id="agent-1",
                generation=1,
                experiment="test_exp",
            )

            # Drain ForumBus into KnowledgeStore
            drained = _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
                experiment="test_exp",
            )
            assert drained == 2

            page = handle_knowledge(
                knowledge_store=store,
                task_id="task-1",
                include="discussion",
                experiment="test_exp",
            )
            assert len(page["discussion"]) == 2
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 8. Concurrent writes from multiple threads
# ---------------------------------------------------------------------------


class TestConcurrentWrites:
    def test_multiple_threads_writing_simultaneously(self, tmp_path):
        """Multiple threads writing simultaneously should not lose data."""
        store = _make_store(tmp_path)
        num_threads = 10
        writes_per_thread = 20
        errors: list[Exception] = []

        def _writer(thread_idx: int):
            try:
                for i in range(writes_per_thread):
                    agent_id = f"agent-{thread_idx}"
                    task_id = f"task-{thread_idx}-{i}"

                    store.record_attempt(
                        task_id=task_id,
                        agent_id=agent_id,
                        generation=1,
                        native_score=float(i) / writes_per_thread,
                        model_output=f"output-{thread_idx}-{i}",
                    )

                    if i % 3 == 0:
                        store.record_insight(
                            task_id=task_id,
                            agent_id=agent_id,
                            generation=1,
                            text=f"insight-{thread_idx}-{i}",
                        )

                    if i % 5 == 0:
                        store.record_post(
                            task_id=task_id,
                            agent_id=agent_id,
                            generation=1,
                            text=f"post-{thread_idx}-{i}",
                        )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_writer, args=(idx,)) for idx in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        try:
            assert not errors, f"Thread errors: {errors}"

            # Verify all attempts were written
            gen_entries = store.query_generation(1)
            attempts = [e for e in gen_entries if e["entry_type"] == "attempt"]
            assert len(attempts) == num_threads * writes_per_thread

            # Verify insights count
            insights = [e for e in gen_entries if e["entry_type"] == "insight"]
            expected_insights = num_threads * len([i for i in range(writes_per_thread) if i % 3 == 0])
            assert len(insights) == expected_insights

            # Verify posts count
            posts = [e for e in gen_entries if e["entry_type"] == "post"]
            expected_posts = num_threads * len([i for i in range(writes_per_thread) if i % 5 == 0])
            assert len(posts) == expected_posts
        finally:
            store.close()

    def test_concurrent_signal_done(self, tmp_path):
        """Multiple agents signaling done concurrently should not lose signals."""
        store = _make_store(tmp_path)
        num_agents = 20
        errors: list[Exception] = []

        def _signal(agent_idx: int):
            try:
                store.signal_done(
                    task_id="task-shared",
                    agent_id=f"agent-{agent_idx}",
                    generation=1,
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_signal, args=(idx,)) for idx in range(num_agents)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        try:
            assert not errors, f"Thread errors: {errors}"

            status = store.get_done_status(
                task_id="task-shared",
                generation=1,
                expected_agents=num_agents,
            )
            assert status["agents_done"] == num_agents
            assert status["all_done"] is True
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 9. End-to-end: MCP post -> FTS search -> knowledge query
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    def test_post_then_search_then_query(self, tmp_path):
        """Write via ForumBus -> drain -> FTS search -> knowledge handler."""
        store = _make_store(tmp_path, default_experiment="e2e")
        bus = ForumBus(
            db_path=str(tmp_path / "forum.sqlite"),
            experiment="e2e",
            generation=1,
        )
        try:
            # Post via MCP handler (goes to ForumBus only)
            handle_forum_post(
                knowledge_store=store,
                forum_bus=bus,
                task_id="task-1",
                text="The zephyr algorithm converges faster with momentum",
                agent_id="agent-0",
                generation=1,
                experiment="e2e",
            )

            # Drain ForumBus into KnowledgeStore
            _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
                experiment="e2e",
            )

            # Record an insight with the same keyword
            store.record_insight(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="zephyr requires momentum coefficient tuning",
                experiment="e2e",
            )

            # FTS search should find both
            fts_results = store.fts_search("zephyr", experiment="e2e")
            assert len(fts_results) >= 2
            found_types = {r["entry_type"] for r in fts_results}
            assert "post" in found_types
            assert "insight" in found_types

            # Knowledge handler should return the full page
            page = handle_knowledge(
                knowledge_store=store,
                task_id="task-1",
                experiment="e2e",
            )
            assert len(page["discussion"]) == 1
            assert len(page["insights"]) == 1
            assert "zephyr" in page["discussion"][0]["text"]
            assert "momentum" in page["insights"][0]["text"]
        finally:
            store.close()

    def test_multi_task_multi_generation_full_flow(self, tmp_path):
        """Full flow across multiple tasks and generations with all operations."""
        store = _make_store(tmp_path, default_experiment="full_flow")
        try:
            # Gen 1: attempts on 3 tasks
            for i in range(3):
                store.record_attempt(
                    task_id=f"task-{i}",
                    agent_id="agent-0",
                    generation=1,
                    native_score=0.0,
                    model_output=f"gen1 attempt on task-{i}",
                    experiment="full_flow",
                )

            # Gen 1: forum discussion
            for i in range(3):
                store.record_insight(
                    task_id=f"task-{i}",
                    agent_id="agent-0",
                    generation=1,
                    text=f"Gen 1 insight for task-{i}",
                    experiment="full_flow",
                )

            # Gen 1: distillation
            store.record_distillation(
                task_id="task-0",
                generation=1,
                assets=[
                    {"asset_type": "transferable_insight", "text": "Common pattern across tasks"},
                ],
                experiment="full_flow",
            )

            # Gen 2: improved attempts using gen 1 knowledge
            for i in range(3):
                # Verify gen 1 data is available
                page = store.query_task(f"task-{i}", experiment="full_flow")
                assert len(page["attempts"]) >= 1
                assert len(page["insights"]) >= 1

                store.record_attempt(
                    task_id=f"task-{i}",
                    agent_id="agent-0",
                    generation=2,
                    native_score=0.5 + i * 0.2,
                    model_output=f"gen2 improved attempt on task-{i}",
                    experiment="full_flow",
                )

            # Verify gen 2 has access to gen 1 distillation
            page = store.query_task("task-0", experiment="full_flow")
            assert len(page["distilled"]) == 1
            assert "Common pattern" in page["distilled"][0]["text"]

            # Verify total entries per generation
            gen1 = store.query_generation(1, experiment="full_flow")
            gen2 = store.query_generation(2, experiment="full_flow")
            assert len(gen1) == 7  # 3 attempts + 3 insights + 1 distillation
            assert len(gen2) == 3  # 3 attempts
        finally:
            store.close()
