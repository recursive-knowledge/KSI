# tests/test_memory_mcp_server.py
"""Tests for the memory MCP server tool handlers."""

import pytest


def _make_populated_store(tmp_path):
    from ksi.memory.store import MemoryStore

    db_path = str(tmp_path / "test.sqlite")
    store = MemoryStore(db_path)
    store.upsert_task_memory_record(
        experiment="exp1",
        generation=1,
        agent_id="agent-0",
        task_id="task-1",
        eval_results={"resolved": True, "status": "ok", "native_score": 1.0},
        final_model_output="<patch>diff --git ...</patch>",
        full_memory_trace='{"event":"tool_call","tool":"Read"}',
        full_memory_trace_condensed="Prefer migration cache invalidation first.",
        task_specific_insights=["Check migration cache invalidation."],
        attempt_event={"status": "ok", "resolved": True, "native_score": 1.0},
    )
    return store


class TestMCPHandlers:
    def test_default_embedding_model_is_embeddinggemma(self, monkeypatch):
        import importlib

        import ksi.memory.embeddings as embeddings

        monkeypatch.delenv("KSI_EMBEDDING_MODEL", raising=False)
        reloaded = importlib.reload(embeddings)
        try:
            assert reloaded._DEFAULT_EMBEDDING_MODEL == "google/embeddinggemma-300m"
        finally:
            importlib.reload(embeddings)

    def test_build_tools_include_memory_and_forum_toolset(self):
        from ksi.memory.mcp_server import _build_tools

        names = {tool["name"] for tool in _build_tools()}
        assert "query" in names
        assert "search" not in names  # search tool removed; related summaries pre-injected
        assert "forum_read" in names
        assert "forum_post_insight" not in names
        assert "forum_post_comment" not in names
        assert "forum_get_status" not in names
        # forum_signal_done is now included via knowledge_tools
        assert "forum_signal_done" in names
        assert "knowledge" in names
        assert "forum_post" in names
        assert "query_workstream" not in names

        query_tool = next(tool for tool in _build_tools() if tool["name"] == "query")
        assert "query" in query_tool["inputSchema"]["properties"]
        assert "semantic vector search" in query_tool["description"]

    def test_handle_query(self, tmp_path):
        from ksi.memory.mcp_server import handle_query

        store = _make_populated_store(tmp_path)
        try:
            result = handle_query(store=store, task_id="task-1")
            assert result is not None
            assert result["task_id"] == "task-1"
            assert len(result["records"]) >= 1
            assert len(result["insights"]) >= 1
            rec = result["records"][0]
            assert "full_memory_trace_condensed" in rec
            assert "task_specific_insights" in rec
            assert "full_memory_trace" not in rec
            assert "final_model_output" not in rec
            assert "attempt_history" in rec
        finally:
            store.close()

    def test_handle_query_missing(self, tmp_path):
        from ksi.memory.mcp_server import handle_query

        store = _make_populated_store(tmp_path)
        try:
            result = handle_query(store=store, task_id="nonexistent")
            assert result is not None
            assert result["task_id"] == "nonexistent"
            assert result["records"] == []
            assert result["insights"] == []
        finally:
            store.close()

    def test_handle_query_uses_knowledge_store_without_runtime_store(self):
        from ksi.memory.mcp_server import handle_query

        class FakeKnowledgeStore:
            _vec_enabled = False

            def query_task(self, task_id, *, entry_types, experiment=None, limit=50):
                assert task_id == "task-1"
                assert entry_types == ["attempt"]
                assert experiment == "exp1"
                assert limit == 8
                return {
                    "task_id": "task-1",
                    "attempts": [
                        {
                            "gen": 1,
                            "agent_id": "agent-0",
                            "score": 1.0,
                            "content": {
                                "eval_results": {"resolved": True},
                                "trace_condensed": "knowledge attempt",
                                "insights": ["preserve the parser fix"],
                            },
                        }
                    ],
                }

        result = handle_query(
            store=None,
            knowledge_store=FakeKnowledgeStore(),
            task_id="task-1",
            experiment="exp1",
        )

        assert result["records"][0]["full_memory_trace_condensed"] == "knowledge attempt"
        assert result["records"][0]["native_score"] == 1.0
        assert result["insights"][0]["text"] == "preserve the parser fix"

    def test_handle_query_includes_semantic_related_when_available(self, tmp_path):
        from ksi.memory.mcp_server import handle_query

        class FakeEmbedder:
            def embed(self, text):
                assert text == "cache"
                return [0.1, 0.2]

        class FakeKnowledgeStore:
            _vec_enabled = True

            def vec_search(self, embedding, *, max_results, experiment=None):
                assert embedding == [0.1, 0.2]
                assert max_results == 5
                assert experiment == "exp1"
                return [{"task_id": "related", "distance": 0.01}]

        store = _make_populated_store(tmp_path)
        try:
            result = handle_query(
                store=store,
                task_id="task-1",
                experiment="exp1",
                knowledge_store=FakeKnowledgeStore(),
                semantic_embedder=FakeEmbedder(),
                semantic_query="cache",
            )
            assert result["semantic_enabled"] is True
            assert result["retrieval_mode"] == "semantic"
            assert result["semantic_query"] == "cache"
            assert result["semantic_result_count"] == 1
            assert result["semantic_error"] == ""
            assert result["related"] == [{"task_id": "related", "distance": 0.01}]
        finally:
            store.close()

    def test_handle_query_marks_semantic_available_even_without_hits(self, tmp_path):
        from ksi.memory.mcp_server import handle_query

        class FakeEmbedder:
            def embed(self, text):
                assert text == "cache"
                return [0.1, 0.2]

        class FakeKnowledgeStore:
            _vec_enabled = True

            def vec_search(self, embedding, *, max_results, experiment=None):
                return []

        store = _make_populated_store(tmp_path)
        try:
            result = handle_query(
                store=store,
                task_id="task-1",
                experiment="exp1",
                knowledge_store=FakeKnowledgeStore(),
                semantic_embedder=FakeEmbedder(),
                semantic_query="cache",
            )
            assert result["semantic_enabled"] is True
            assert result["semantic_query"] == "cache"
            assert result["semantic_result_count"] == 0
            assert result["related"] == []
        finally:
            store.close()

    def test_handle_query_reports_semantic_disabled_when_vec_unavailable(self, tmp_path):
        from ksi.memory.mcp_server import handle_query

        class FakeEmbedder:
            def embed(self, text):
                raise AssertionError("embed should not be called when vectors are unavailable")

        class FakeKnowledgeStore:
            _vec_enabled = False

            def vec_search(self, embedding, *, max_results, experiment=None):
                raise AssertionError("vec_search should not be called when vectors are unavailable")

        store = _make_populated_store(tmp_path)
        try:
            result = handle_query(
                store=store,
                task_id="task-1",
                experiment="exp1",
                knowledge_store=FakeKnowledgeStore(),
                semantic_embedder=FakeEmbedder(),
                semantic_query="cache",
            )
            assert result["semantic_enabled"] is False
            # No fts_search on this fake store, so the fallback yields nothing,
            # but the mode is still reported as the lexical fallback.
            assert result["retrieval_mode"] == "fts"
            assert result["semantic_query"] == "cache"
            assert result["semantic_result_count"] == 0
            assert result["related"] == []
        finally:
            store.close()

    def test_handle_query_falls_back_to_fts_when_vec_unavailable(self, tmp_path):
        """No embedder/vec → FTS-backed related items with semantic-compatible shape."""
        from ksi.memory.mcp_server import handle_query

        captured: dict[str, object] = {}

        class FakeKnowledgeStore:
            _vec_enabled = False

            def query_task(self, task_id, *, entry_types, experiment=None, limit=50):
                return {"task_id": task_id, "attempts": []}

            def vec_search(self, *_a, **_k):
                raise AssertionError("vec_search must not be called without a vector index")

            def fts_search(self, query, *, max_results, experiment=None, raw_match=False):
                captured["query"] = query
                captured["max_results"] = max_results
                captured["experiment"] = experiment
                # fts_search rows carry created_at (not distance).
                return [
                    {
                        "id": 7,
                        "task_id": "related-task",
                        "agent_id": "agent-2",
                        "entry_type": "insight",
                        "source_phase": "distill",
                        "content": {"text": "lexical hit"},
                        "native_score": 0.5,
                        "generation": 1,
                        "created_at": "2026-01-01",
                    }
                ]

        result = handle_query(
            store=None,
            task_id="task-1",
            experiment="exp1",
            knowledge_store=FakeKnowledgeStore(),
            semantic_embedder=None,
            semantic_query="boundary partition",
        )
        assert result["semantic_enabled"] is False
        assert result["retrieval_mode"] == "fts"
        assert result["semantic_result_count"] == 1
        # Semantic was never attempted (no embedder), so its error field must
        # stay empty — an FTS-side error must not leak into ``semantic_error``.
        assert result["semantic_error"] == ""
        assert captured == {
            "query": "boundary OR partition",
            "max_results": 5,
            "experiment": "exp1",
        }
        item = result["related"][0]
        # Shape compatibility: FTS items carry every key a vec_search item has,
        # including a (None) distance, plus FTS's own created_at.
        assert item["task_id"] == "related-task"
        assert item["distance"] is None
        for key in (
            "id",
            "task_id",
            "agent_id",
            "entry_type",
            "source_phase",
            "content",
            "native_score",
            "generation",
        ):
            assert key in item

    def test_handle_query_fts_fallback_uses_task_id_when_no_query_text(self, tmp_path):
        """Even without explicit query text, the FTS fallback retrieves on task_id."""
        from ksi.memory.mcp_server import handle_query

        seen: dict[str, object] = {}

        class FakeKnowledgeStore:
            _vec_enabled = False

            def query_task(self, task_id, *, entry_types, experiment=None, limit=50):
                return {"task_id": task_id, "attempts": []}

            def fts_search(self, query, *, max_results, experiment=None, raw_match=False):
                seen["query"] = query
                if query == "django__django":
                    return [{"task_id": "django__django-2", "content": {}, "created_at": "x"}]
                return [{"task_id": "suffix-only-1", "content": {}, "created_at": "x"}]

        result = handle_query(
            store=None,
            task_id="django__django-1",
            knowledge_store=FakeKnowledgeStore(),
            semantic_embedder=None,
        )
        assert seen["query"] == "django__django"
        assert result["retrieval_mode"] == "fts"
        assert [item["task_id"] for item in result["related"]] == ["django__django-2"]

    def test_handle_query_embedding_failure_falls_back_to_fts(self, tmp_path):
        """A runtime embedding exception degrades to FTS rather than empty/crash."""
        from ksi.memory.mcp_server import handle_query

        class FailingEmbedder:
            def embed(self, _text):
                raise RuntimeError("embedding service down")

        class FakeKnowledgeStore:
            _vec_enabled = True

            def query_task(self, task_id, *, entry_types, experiment=None, limit=50):
                return {"task_id": task_id, "attempts": []}

            def vec_search(self, *_a, **_k):
                raise AssertionError("vec_search should not run if embed() raised")

            def fts_search(self, query, *, max_results, experiment=None, raw_match=False):
                return [{"task_id": "fts-hit", "content": {}}]

        result = handle_query(
            store=None,
            task_id="task-1",
            knowledge_store=FakeKnowledgeStore(),
            semantic_embedder=FailingEmbedder(),
            semantic_query="cache invalidation",
        )
        # semantic_available was True (vec enabled + embedder), but the embed
        # call raised; the per-call fallback must populate related from FTS.
        assert result["semantic_enabled"] is True
        assert result["retrieval_mode"] == "fts"
        assert result["semantic_error"] != ""
        assert result["related"][0]["task_id"] == "fts-hit"
        assert result["related"][0]["distance"] is None

    def test_forum_protocol_state_requires_retrieval_before_per_task_post(self):
        from ksi.memory.mcp_server import ForumProtocolState

        state = ForumProtocolState()
        assert state.missing_for_post("task-1") == [
            "knowledge(task_id='task-1')",
            "query(task_id='task-1', query='...')",
        ]

        state.mark_knowledge("task-1")
        assert state.missing_for_post("task-1") == ["query(task_id='task-1', query='...')"]

        state.mark_query("task-1", "failing assertion and changed file")
        assert state.missing_for_post("task-1") == []

    def test_forum_protocol_state_requires_cross_task_semantic_query(self):
        from ksi.memory.mcp_server import ForumProtocolState

        state = ForumProtocolState()
        assert state.missing_for_post("__cross_task__") == ["query(task_id='__cross_task__', query='...')"]

        state.mark_query("__cross_task__", "shared failure mode across tasks")
        assert state.missing_for_post("__cross_task__") == []

    def test_snapshot_query(self):
        from ksi.memory.mcp_server import _query_from_snapshot

        snapshot = {
            "query_records_by_task": {
                "task-1": [
                    {
                        "gen": 1,
                        "agent_id": "agent-0",
                        "task_id": "task-1",
                        "eval_results": {"resolved": True, "status": "ok", "native_score": 1.0},
                        "full_memory_trace_condensed": "Check separator rows.",
                        "task_specific_insights": ["Look for separator rows first."],
                        "attempt_history": [{"status": "ok"}],
                        "updated_at": "2026-01-01",
                    }
                ]
            },
            "search_rows": [
                {
                    "id": "sum-1",
                    "experiment": "exp1",
                    "agent_id": "agent-0",
                    "generation": 1,
                    "task_id": "task-1",
                    "repo": "arc",
                    "approach": "Use separator rows to partition the grid",
                    "key_files": "[]",
                    "outcome": "resolved",
                    "score": 1.0,
                    "lessons": '["Check separator rows first"]',
                    "created_at": "2026-01-01",
                }
            ],
        }

        query_result = _query_from_snapshot(snapshot=snapshot, task_id="task-1")
        assert query_result["task_id"] == "task-1"
        assert len(query_result["records"]) == 1
        assert len(query_result["insights"]) == 1

    def test_snapshot_query_uses_semantic_related_when_available(self):
        from ksi.memory.mcp_server import _query_from_snapshot

        class FakeEmbedder:
            def embed(self, text):
                assert text == "boundary pattern"
                return [0.1, 0.2]

        class FakeKnowledgeStore:
            _vec_enabled = True

            def vec_search(self, embedding, *, max_results, experiment=None):
                assert embedding == [0.1, 0.2]
                assert max_results == 5
                assert experiment == "exp1"
                return [{"task_id": "related", "distance": 0.02}]

        result = _query_from_snapshot(
            snapshot={"query_records_by_task": {}},
            task_id="task-1",
            experiment="exp1",
            knowledge_store=FakeKnowledgeStore(),
            semantic_embedder=FakeEmbedder(),
            semantic_query="boundary pattern",
        )
        assert result["semantic_enabled"] is True
        assert result["retrieval_mode"] == "semantic"
        assert result["semantic_query"] == "boundary pattern"
        assert result["semantic_result_count"] == 1
        assert result["related"] == [{"task_id": "related", "distance": 0.02}]

    def test_snapshot_query_falls_back_to_fts_when_vec_unavailable(self):
        from ksi.memory.mcp_server import _query_from_snapshot

        class FakeKnowledgeStore:
            _vec_enabled = False

            def fts_search(self, query, *, max_results, experiment=None, raw_match=False):
                assert query == "boundary OR pattern"
                assert raw_match is True
                return [{"task_id": "lex", "content": {"text": "hit"}}]

        result = _query_from_snapshot(
            snapshot={"query_records_by_task": {}},
            task_id="task-1",
            experiment="exp1",
            knowledge_store=FakeKnowledgeStore(),
            semantic_embedder=None,
            semantic_query="boundary pattern",
        )
        assert result["semantic_enabled"] is False
        assert result["retrieval_mode"] == "fts"
        assert result["related"][0]["task_id"] == "lex"
        assert result["related"][0]["distance"] is None


class TestFTSFallbackEndToEnd:
    """End-to-end: real KnowledgeStore with vec disabled → handle_query uses FTS."""

    def _make_knowledge_store(self, tmp_path):
        from ksi.memory.knowledge_store import KnowledgeStore

        db_path = str(tmp_path / "knowledge.sqlite")
        store = KnowledgeStore(db_path, default_experiment="exp1", enable_vec=False)
        # Cross-task knowledge the current task should be able to retrieve.
        store.record_attempt(
            task_id="django__django-100",
            agent_id="agent-0",
            generation=1,
            model_output="Fixed the QuerySet boundary partition bug in views.py",
            native_score=1.0,
            experiment="exp1",
        )
        store.record_insight(
            task_id="sphinx__sphinx-200",
            agent_id="agent-1",
            generation=1,
            text="Boundary partition rows must be detected before transform",
            experiment="exp1",
        )
        return store

    def test_handle_query_returns_fts_related_when_no_embedder(self, tmp_path):
        from ksi.memory.mcp_server import handle_query

        store = self._make_knowledge_store(tmp_path)
        try:
            result = handle_query(
                store=None,
                task_id="task-current",
                experiment="exp1",
                knowledge_store=store,
                semantic_embedder=None,
                semantic_query="boundary partition",
            )
            assert result["semantic_enabled"] is False
            assert result["retrieval_mode"] == "fts"
            assert result["semantic_result_count"] >= 1
            task_ids = {item["task_id"] for item in result["related"]}
            assert {"django__django-100", "sphinx__sphinx-200"} & task_ids
            for item in result["related"]:
                # Shape compatibility with semantic items.
                assert "distance" in item
                assert item["distance"] is None
        finally:
            store.close()

    def test_handle_query_fts_fallback_survives_special_chars(self, tmp_path):
        """FTS5 special characters in the query must not raise (real sanitizer)."""
        from ksi.memory.mcp_server import handle_query

        store = self._make_knowledge_store(tmp_path)
        try:
            for q in (
                "boundary OR partition",
                'c++ "templates"',
                "*(){}[]",
                "NEAR/3 foo",
                "a AND b NOT c",
            ):
                result = handle_query(
                    store=None,
                    task_id="task-current",
                    experiment="exp1",
                    knowledge_store=store,
                    semantic_embedder=None,
                    semantic_query=q,
                )
                # No FTS5 syntax error raised; the lexical fallback engaged and
                # the semantic error field stays clean.
                assert result["retrieval_mode"] == "fts", q
                assert result["semantic_error"] == "", q
        finally:
            store.close()


class TestIsRedundant:
    """Tests for _is_redundant word-overlap deduplication."""

    def test_high_overlap_returns_true(self):
        from ksi.memory.mcp_server import _is_redundant

        accepted = ["check the migration cache invalidation first"]
        assert _is_redundant("check migration cache invalidation", accepted) is True

    def test_distinct_text_returns_false(self):
        from ksi.memory.mcp_server import _is_redundant

        accepted = ["check the migration cache invalidation first"]
        assert _is_redundant("refactor the matrix multiplication code", accepted) is False

    def test_empty_candidate_always_redundant(self):
        from ksi.memory.mcp_server import _is_redundant

        assert _is_redundant("", []) is True
        assert _is_redundant("   ", ["anything"]) is True

    def test_blank_candidate_always_redundant(self):
        from ksi.memory.mcp_server import _is_redundant

        assert _is_redundant("  \t\n  ", ["some text"]) is True


class TestQueryFromSnapshotDedup:
    """Tests that _query_from_snapshot deduplicates identical insights across records."""

    def test_identical_insights_across_records_are_deduped(self):
        from ksi.memory.mcp_server import _query_from_snapshot

        snapshot = {
            "query_records_by_task": {
                "task-1": [
                    {
                        "gen": 1,
                        "agent_id": "agent-0",
                        "task_id": "task-1",
                        "eval_results": {"resolved": False, "native_score": 0.0},
                        "full_memory_trace_condensed": "First attempt.",
                        "task_specific_insights": ["Check separator rows first."],
                        "attempt_history": [],
                        "updated_at": "2026-01-01",
                    },
                    {
                        "gen": 2,
                        "agent_id": "agent-1",
                        "task_id": "task-1",
                        "eval_results": {"resolved": False, "native_score": 0.0},
                        "full_memory_trace_condensed": "Second attempt.",
                        "task_specific_insights": ["Check separator rows first."],
                        "attempt_history": [],
                        "updated_at": "2026-01-02",
                    },
                ]
            },
            "related_summaries": [],
        }
        result = _query_from_snapshot(snapshot=snapshot, task_id="task-1")
        assert len(result["records"]) == 2
        # Insights should be deduplicated: both records have the same insight text
        assert len(result["insights"]) == 1
        assert result["insights"][0]["text"] == "Check separator rows first."


class TestQueryFromSnapshotRelated:
    """Tests for related_summaries field in _query_from_snapshot."""

    def test_returns_related_field_from_snapshot(self):
        from ksi.memory.mcp_server import _query_from_snapshot

        related = [
            {"task_id": "task-2", "approach": "use grid coloring", "score": 0.5},
            {"task_id": "task-3", "approach": "edge detection", "score": 0.8},
        ]
        snapshot = {
            "query_records_by_task": {
                "task-1": [
                    {
                        "gen": 1,
                        "agent_id": "agent-0",
                        "task_id": "task-1",
                        "eval_results": {"resolved": False},
                        "full_memory_trace_condensed": "",
                        "task_specific_insights": [],
                        "attempt_history": [],
                        "updated_at": "",
                    }
                ]
            },
            "related_summaries": related,
        }
        result = _query_from_snapshot(snapshot=snapshot, task_id="task-1")
        assert result["related"] == related

    def test_related_summaries_capped_at_5(self):
        from ksi.memory.mcp_server import _query_from_snapshot

        related = [{"task_id": f"task-{i}", "approach": f"approach-{i}"} for i in range(10)]
        snapshot = {
            "query_records_by_task": {"task-1": []},
            "related_summaries": related,
        }
        result = _query_from_snapshot(snapshot=snapshot, task_id="task-1")
        assert len(result["related"]) == 5


class TestQueryExcludesHoldoutTaskIds:
    """Hold-out probe: forum/task ``query`` retrieval must be able to exclude
    a set of task ids (the engine's hold-out set) so hold-out content never
    surfaces in training agents' retrieval — in BOTH the semantic vec path
    and the lexical FTS fallback, for both query entry points."""

    class _Embedder:
        def embed(self, text):
            return [0.1, 0.2]

    def test_handle_query_vec_excludes_task_ids(self, tmp_path):
        from ksi.memory.mcp_server import handle_query

        class FakeKnowledgeStore:
            _vec_enabled = True

            def query_task(self, task_id, *, entry_types, experiment=None, limit=50):
                return {"task_id": task_id, "attempts": []}

            def vec_search(self, embedding, *, max_results, experiment=None):
                return [
                    {"task_id": "h1", "distance": 0.01},
                    {"task_id": "t2", "distance": 0.02},
                ]

        result = handle_query(
            store=None,
            task_id="t1",
            experiment="exp1",
            knowledge_store=FakeKnowledgeStore(),
            semantic_embedder=self._Embedder(),
            semantic_query="cache",
            exclude_task_ids=frozenset({"h1"}),
        )
        assert [r["task_id"] for r in result["related"]] == ["t2"]

    def test_handle_query_fts_fallback_excludes_task_ids(self, tmp_path):
        from ksi.memory.mcp_server import handle_query

        class FakeKnowledgeStore:
            _vec_enabled = False

            def query_task(self, task_id, *, entry_types, experiment=None, limit=50):
                return {"task_id": task_id, "attempts": []}

            def fts_search(self, query, *, max_results, experiment=None, raw_match=False):
                return [
                    {"task_id": "h1", "content": {}, "created_at": "x"},
                    {"task_id": "t2", "content": {}, "created_at": "x"},
                ]

        result = handle_query(
            store=None,
            task_id="t1",
            experiment="exp1",
            knowledge_store=FakeKnowledgeStore(),
            semantic_query="cache",
            exclude_task_ids=frozenset({"h1"}),
        )
        assert result["retrieval_mode"] == "fts"
        assert [r["task_id"] for r in result["related"]] == ["t2"]

    def test_query_from_snapshot_vec_excludes_task_ids(self, tmp_path):
        from ksi.memory.mcp_server import _query_from_snapshot

        class FakeKnowledgeStore:
            _vec_enabled = True

            def vec_search(self, embedding, *, max_results, experiment=None):
                return [
                    {"task_id": "h1", "distance": 0.01},
                    {"task_id": "t2", "distance": 0.02},
                ]

        result = _query_from_snapshot(
            snapshot={"query_records_by_task": {}, "related_summaries": []},
            task_id="t1",
            experiment="exp1",
            knowledge_store=FakeKnowledgeStore(),
            semantic_embedder=self._Embedder(),
            semantic_query="cache",
            exclude_task_ids=frozenset({"h1"}),
        )
        assert [r["task_id"] for r in result["related"]] == ["t2"]

    def test_query_from_snapshot_fts_fallback_excludes_task_ids(self, tmp_path):
        from ksi.memory.mcp_server import _query_from_snapshot

        class FakeKnowledgeStore:
            _vec_enabled = False

            def fts_search(self, query, *, max_results, experiment=None, raw_match=False):
                return [
                    {"task_id": "h1", "content": {}, "created_at": "x"},
                    {"task_id": "t2", "content": {}, "created_at": "x"},
                ]

        result = _query_from_snapshot(
            snapshot={"query_records_by_task": {}, "related_summaries": []},
            task_id="t1",
            experiment="exp1",
            knowledge_store=FakeKnowledgeStore(),
            # No embedder + no vec -> FTS fallback path.
            semantic_query="cache",
            exclude_task_ids=frozenset({"h1"}),
        )
        assert result["retrieval_mode"] == "fts"
        assert [r["task_id"] for r in result["related"]] == ["t2"]

    def test_handle_query_excludes_exact_task_records(self):
        from ksi.memory.mcp_server import handle_query

        class FakeKnowledgeStore:
            _vec_enabled = False

            def query_task(self, *args, **kwargs):
                raise AssertionError("excluded exact task id must not query the store")

        result = handle_query(
            store=None,
            task_id="h1",
            experiment="exp1",
            knowledge_store=FakeKnowledgeStore(),
            semantic_query="leaky sibling answer",
            exclude_task_ids=frozenset({"h1"}),
        )

        assert result["records"] == []
        assert result["insights"] == []
        assert result["related"] == []
        assert result["retrieval_mode"] == "excluded"

    def test_query_from_snapshot_excludes_exact_task_records(self):
        from ksi.memory.mcp_server import _query_from_snapshot

        result = _query_from_snapshot(
            snapshot={
                "query_records_by_task": {
                    "h1": [
                        {
                            "task_id": "h1",
                            "full_memory_trace_condensed": "ANSWER: leaky sibling answer",
                        }
                    ]
                },
                "related_summaries": [],
            },
            task_id="h1",
            exclude_task_ids=frozenset({"h1"}),
        )

        assert result["records"] == []
        assert result["insights"] == []
        assert result["related"] == []
        assert result["retrieval_mode"] == "excluded"

    def test_resolve_exclude_task_ids_merges_env_and_snapshot(self, monkeypatch):
        from ksi.memory.mcp_server import _resolve_exclude_task_ids

        monkeypatch.setenv("MEMORY_EXCLUDE_TASK_IDS", "h1, ,h2")
        assert _resolve_exclude_task_ids({"exclude_task_ids": ["h2", "h3", ""]}) == frozenset({"h1", "h2", "h3"})
        assert _resolve_exclude_task_ids(None) == frozenset({"h1", "h2"})
        monkeypatch.delenv("MEMORY_EXCLUDE_TASK_IDS")
        assert _resolve_exclude_task_ids(None) == frozenset()

    def test_forum_read_excludes_task_ids_from_bus(self, tmp_path):
        from ksi.memory.forum_bus import ForumBus
        from ksi.memory.mcp_server import handle_forum_read

        bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="exp1", generation=1)
        bus.clear()
        bus.append(
            round_num=0,
            agent_id="agent-h",
            message_type="post",
            content={"task_id": "h1", "text": "ANSWER: leaky sibling"},
        )
        bus.append(
            round_num=0,
            agent_id="agent-t",
            message_type="post",
            content={"task_id": "t2", "text": "safe foreign conversation"},
        )

        messages = handle_forum_read(
            forum_bus=bus,
            round_num=0,
            exclude_task_ids=frozenset({"h1"}),
        )

        assert [(m["content"]["task_id"], m["content"]["text"]) for m in messages] == [
            ("t2", "safe foreign conversation")
        ]

    def test_forum_post_rejects_excluded_or_unassigned_task_ids(self, tmp_path):
        from ksi.memory.forum_bus import ForumBus
        from ksi.memory.mcp_server import handle_forum_post

        bus = ForumBus(db_path=str(tmp_path / "forum.sqlite"), experiment="exp1", generation=1)

        with pytest.raises(ValueError, match="excluded"):
            handle_forum_post(
                knowledge_store=None,
                forum_bus=bus,
                task_id="h1",
                text="poison sibling page",
                agent_id="agent-0",
                generation=1,
                allowed_task_ids={"t1"},
                exclude_task_ids=frozenset({"h1"}),
            )

        with pytest.raises(ValueError, match="not assigned"):
            handle_forum_post(
                knowledge_store=None,
                forum_bus=bus,
                task_id="h2",
                text="unassigned sibling page",
                agent_id="agent-0",
                generation=1,
                allowed_task_ids={"t1"},
            )

        result = handle_forum_post(
            knowledge_store=None,
            forum_bus=bus,
            task_id="__cross_task__",
            text="cross-task candidate",
            agent_id="agent-0",
            generation=1,
            allowed_task_ids={"t1"},
            round_num=0,
        )

        assert result["status"] == "ok"


class TestRunServerFrameValidation:
    """Regression tests for issue #1224: malformed JSON-RPC frames must
    produce a JSON-RPC error response, never crash the server."""

    def _drive(self, monkeypatch, frames):
        import io
        import json

        from ksi.memory import mcp_server

        stdin = io.StringIO("".join(f + "\n" for f in frames))
        stdout = io.StringIO()
        monkeypatch.setattr(mcp_server.sys, "stdin", stdin)
        monkeypatch.setattr(mcp_server.sys, "stdout", stdout)

        mcp_server._run_server(
            store=None,
            snapshot=None,
            forum_bus=None,
            forum_generation=0,
            forum_round=0,
            forum_agent_id="agent-0",
            forum_expected_agents=0,
            memory_experiment="",
            toolset="all",
            forum_task_ids=set(),
            semantic_embedder=None,
        )
        return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]

    def test_top_level_array_frame_returns_invalid_request(self, monkeypatch):
        responses = self._drive(monkeypatch, ["[]"])
        assert len(responses) == 1
        assert responses[0]["error"]["code"] == -32600
        assert responses[0]["id"] is None

    def test_scalar_frame_returns_invalid_request(self, monkeypatch):
        responses = self._drive(monkeypatch, ["5"])
        assert len(responses) == 1
        assert responses[0]["error"]["code"] == -32600

    def test_array_params_returns_invalid_params(self, monkeypatch):
        frame = '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":[]}'
        responses = self._drive(monkeypatch, [frame])
        assert len(responses) == 1
        assert responses[0]["error"]["code"] == -32602
        assert responses[0]["id"] == 1

    def test_array_arguments_returns_invalid_params(self, monkeypatch):
        frame = '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"query","arguments":[]}}'
        responses = self._drive(monkeypatch, [frame])
        assert len(responses) == 1
        assert responses[0]["error"]["code"] == -32602
        assert responses[0]["id"] == 2

    def test_malformed_frame_does_not_abort_subsequent_frames(self, monkeypatch):
        frames = ["[]", '{"jsonrpc":"2.0","id":9,"method":"tools/list"}']
        responses = self._drive(monkeypatch, frames)
        assert responses[0]["error"]["code"] == -32600
        assert responses[1]["id"] == 9
        assert "tools" in responses[1]["result"]


class TestResolveToolset:
    """Issue #1264: an unknown MCP_TOOLSET must fail closed, not grant "all"."""

    @pytest.mark.parametrize("value", ["all", "memory", "task", "forum"])
    def test_valid_values_pass_through(self, monkeypatch, value):
        from ksi.memory.mcp_server import _resolve_toolset

        monkeypatch.setenv("MCP_TOOLSET", value)
        assert _resolve_toolset() == value

    def test_unset_defaults_to_all(self, monkeypatch):
        from ksi.memory.mcp_server import _resolve_toolset

        monkeypatch.delenv("MCP_TOOLSET", raising=False)
        assert _resolve_toolset() == "all"

    def test_empty_defaults_to_all(self, monkeypatch):
        from ksi.memory.mcp_server import _resolve_toolset

        monkeypatch.setenv("MCP_TOOLSET", "  ")
        assert _resolve_toolset() == "all"

    def test_case_and_whitespace_normalized(self, monkeypatch):
        from ksi.memory.mcp_server import _resolve_toolset

        monkeypatch.setenv("MCP_TOOLSET", " Forum ")
        assert _resolve_toolset() == "forum"

    def test_unknown_value_fails_closed(self, monkeypatch):
        from ksi.memory.mcp_server import _resolve_toolset

        monkeypatch.setenv("MCP_TOOLSET", "taks")
        with pytest.raises(SystemExit, match="Invalid MCP_TOOLSET 'taks'"):
            _resolve_toolset()
