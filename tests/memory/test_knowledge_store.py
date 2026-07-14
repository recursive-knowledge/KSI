"""Tests for src/kcsi/memory/knowledge_store.py — unified KnowledgeStore."""

import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

from kcsi.memory.knowledge_store import KnowledgeStore


def _make_store(tmp_path, **kwargs):
    db_path = str(tmp_path / "test_knowledge.sqlite")
    return KnowledgeStore(db_path, **kwargs)


def _run_count(store: KnowledgeStore, experiment: str) -> int:
    row = store._execute(
        "SELECT COUNT(*) AS count FROM runs WHERE experiment = ?",
        (experiment,),
        fetchone=True,
    )
    return int(row["count"]) if row else 0


# ---------------------------------------------------------------------------
# 1. Schema creation
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    def test_creates_tables(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            tables = store._execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
                fetchall=True,
            )
            names = {r["name"] for r in tables}
            assert "runs" in names
            assert "generations" in names
            assert "agents" in names
            assert "knowledge" in names
            assert "discussion_done" in names
        finally:
            store.close()

    def test_creates_fts_table(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            tables = store._execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
                fetchall=True,
            )
            names = {r["name"] for r in tables}
            assert "knowledge_fts" in names
        finally:
            store.close()

    def test_locked_is_reentrant_without_repeated_advisory_flock(self, tmp_path, monkeypatch):
        import kcsi.memory._store_common as store_common_mod

        store = _make_store(tmp_path)
        calls: list[int] = []

        class FakeFcntl:
            LOCK_EX = 1
            LOCK_NB = 2
            LOCK_UN = 8

            @staticmethod
            def flock(_fd, flags):
                calls.append(flags)

        # The advisory flock now lives in the shared ``_store_common._locked_guard``
        # (the lock helper was deduped out of store.py/knowledge_store.py).
        monkeypatch.setattr(store_common_mod, "fcntl", FakeFcntl)
        try:
            with store._locked():
                with store._locked():
                    store._connection().execute("SELECT 1").fetchone()

            assert calls == [FakeFcntl.LOCK_EX | FakeFcntl.LOCK_NB, FakeFcntl.LOCK_UN]
            assert int(getattr(store._lock_state, "depth", 0) or 0) == 0
        finally:
            store.close()

    def test_reopen_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test_knowledge.sqlite")
        store1 = KnowledgeStore(db_path)
        store1.record_attempt(
            task_id="t1",
            agent_id="a1",
            generation=1,
            eval_results={"pass": True},
            native_score=1.0,
        )
        store1.close()

        # Reopen — schema should be created idempotently
        store2 = KnowledgeStore(db_path)
        try:
            result = store2.query_task("t1")
            assert len(result["attempts"]) == 1
        finally:
            store2.close()


class TestReadOnlyRunLookups:
    def test_empty_read_apis_do_not_create_runs(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            assert _run_count(store, "missing") == 0

            assert store.get_latest_task_generation(experiment="missing") == 0
            assert store.get_best_scores(experiment="missing") == {}
            assert store.list_task_summaries(experiment="missing") == []
            assert store.count_seed_snapshots(generation=1, experiment="missing") == 0

            assert _run_count(store, "missing") == 0
        finally:
            store.close()


class TestListTaskSummariesTruncation:
    """#1040: LIMIT 200 (default) silently dropped candidates off the bottom
    of the recency-ordered result with no log line — must warn when it
    actually truncates, and stay silent otherwise."""

    def _seed_tasks(self, store: KnowledgeStore, n: int, *, experiment: str = "exp") -> None:
        for i in range(n):
            store.record_attempt(
                task_id=f"t{i}",
                agent_id="a1",
                generation=1,
                model_output="x",
                native_score=0.5,
                experiment=experiment,
            )

    def test_no_warning_when_under_limit(self, tmp_path, caplog):
        store = _make_store(tmp_path)
        try:
            self._seed_tasks(store, 3)
            with caplog.at_level("WARNING", logger="kcsi.memory.knowledge_store"):
                rows = store.list_task_summaries(experiment="exp", limit=200)
            assert len(rows) == 3
            assert not any("list_task_summaries" in rec.message for rec in caplog.records)
        finally:
            store.close()

    def test_warns_when_limit_truncates(self, tmp_path, caplog):
        store = _make_store(tmp_path)
        try:
            self._seed_tasks(store, 5)
            with caplog.at_level("WARNING", logger="kcsi.memory.knowledge_store"):
                rows = store.list_task_summaries(experiment="exp", limit=3)
            assert len(rows) == 3, "result must still be capped at the requested limit"
            assert any(
                "list_task_summaries" in rec.message and "truncat" in rec.message.lower() for rec in caplog.records
            ), f"expected a truncation warning, got: {[rec.message for rec in caplog.records]}"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 2. record_attempt + query_task round-trip
# ---------------------------------------------------------------------------


class TestRecordAttempt:
    def test_round_trip(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            entry_id = store.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                eval_results={"resolved": True},
                model_output="diff --git ...",
                trace_condensed="Applied patch to views.py",
                insights=["Use QuerySet.filter instead of get"],
                native_score=1.0,
            )
            assert isinstance(entry_id, int)
            assert entry_id > 0

            result = store.query_task("task-1")
            assert result["task_id"] == "task-1"
            assert len(result["attempts"]) == 1

            attempt = result["attempts"][0]
            assert attempt["gen"] == 1
            assert attempt["agent_id"] == "agent-0"
            assert attempt["score"] == 1.0
            assert attempt["content"]["eval_results"]["resolved"] is True
            assert attempt["content"]["model_output"] == "diff --git ..."
            assert "QuerySet" in attempt["content"]["insights"][0]
        finally:
            store.close()

    def test_record_attempt_dedups_on_external_id(self, tmp_path):
        """A second record_attempt with the same external_id returns the
        existing entry id and adds no new knowledge/attempt row."""
        store = _make_store(tmp_path)
        try:
            first = store.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                native_score=1.0,
                external_id="legacy:task_memory:42",
            )

            def _knowledge_count() -> int:
                row = store._execute("SELECT COUNT(*) AS c FROM knowledge", (), fetchone=True)
                return int(row["c"]) if row else 0

            def _attempt_count() -> int:
                row = store._execute("SELECT COUNT(*) AS c FROM attempts", (), fetchone=True)
                return int(row["c"]) if row else 0

            assert _knowledge_count() == 1
            assert _attempt_count() == 1

            second = store.record_attempt(
                task_id="task-1",
                agent_id="agent-1",  # different agent/gen: must still dedup
                generation=2,
                native_score=0.0,
                external_id="legacy:task_memory:42",
            )

            assert second == first
            assert _knowledge_count() == 1
            assert _attempt_count() == 1
        finally:
            store.close()

    def test_multiple_attempts(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                native_score=0.0,
            )
            store.record_attempt(
                task_id="task-1",
                agent_id="agent-1",
                generation=1,
                native_score=1.0,
            )
            result = store.query_task("task-1")
            assert len(result["attempts"]) == 2
            scores = [a["score"] for a in result["attempts"]]
            assert 0.0 in scores
            assert 1.0 in scores
        finally:
            store.close()

    def test_query_task_filter_by_generation(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="task-1",
                agent_id="a1",
                generation=1,
                native_score=0.5,
            )
            store.record_attempt(
                task_id="task-1",
                agent_id="a1",
                generation=2,
                native_score=1.0,
            )
            result = store.query_task("task-1", generation=1)
            assert len(result["attempts"]) == 1
            assert result["attempts"][0]["score"] == 0.5
        finally:
            store.close()

    def test_record_attempt_persists_repo_for_list_task_summaries(self, tmp_path):
        """``repo`` threaded through record_attempt lands in list_task_summaries.

        Regression test for issue #1039 (knowledge-retrieval.md #1):
        ``list_task_summaries`` used to hardcode ``"repo": ""`` regardless of
        what was recorded — the real ``TaskSpec.repo`` value never made it
        past the ``tasks`` table.
        """
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="django__django-12345",
                agent_id="agent-0",
                generation=1,
                native_score=1.0,
                repo="django/django",
            )
            summaries = store.list_task_summaries()
            assert len(summaries) == 1
            assert summaries[0]["repo"] == "django/django"
        finally:
            store.close()

    def test_record_attempt_supersede_updates_content_in_place(self, tmp_path):
        """A second record_attempt with the same external_id and
        ``supersede=True`` overwrites the first row's content instead of
        being skipped or creating a duplicate.

        Regression test for issue #1039 (trace-mining.md #1): the execution
        phase's early resume-safety write persists a placeholder attempt
        (``insights=[]``, empty ``reflection``) under a stable external_id;
        the engine's later, richer write (real insight_text / reflection /
        lessons) must supersede it in place, not be silently dropped.
        """
        store = _make_store(tmp_path)
        try:
            first = store.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                native_score=1.0,
                trace_condensed="Approach: (no output). Score: 1.0. Insight: (pending reflection)",
                insights=[],
                reflection="",
                external_id="attempt:task-1:agent-0:1",
            )

            def _knowledge_count() -> int:
                row = store._execute("SELECT COUNT(*) AS c FROM knowledge", (), fetchone=True)
                return int(row["c"]) if row else 0

            def _attempt_count() -> int:
                row = store._execute("SELECT COUNT(*) AS c FROM attempts", (), fetchone=True)
                return int(row["c"]) if row else 0

            assert _knowledge_count() == 1
            assert _attempt_count() == 1

            second = store.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                native_score=1.0,
                trace_condensed="Approach: applied a patch. Score: 1.0. Insight: use QuerySet.filter",
                insights=["Use QuerySet.filter instead of get"],
                reflection="Assumed the ORM lazily evaluates; confirmed via test output.",
                external_id="attempt:task-1:agent-0:1",
                supersede=True,
            )

            assert second == first
            assert _knowledge_count() == 1
            assert _attempt_count() == 1

            result = store.query_task("task-1")
            assert len(result["attempts"]) == 1
            attempt = result["attempts"][0]
            assert "pending reflection" not in attempt["content"]["trace_condensed"]
            assert "QuerySet" in attempt["content"]["trace_condensed"]
            assert attempt["content"]["insights"] == ["Use QuerySet.filter instead of get"]
            assert attempt["content"]["reflection"] == "Assumed the ORM lazily evaluates; confirmed via test output."
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 3. record_insight + query_task round-trip
# ---------------------------------------------------------------------------


class TestRecordInsight:
    def test_round_trip(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            entry_id = store.record_insight(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="Use batch processing for large datasets",
                scope="meta",
                confidence="high",
                evidence_task_ids=["task-2", "task-3"],
                round_num=0,
            )
            assert isinstance(entry_id, int)

            result = store.query_task("task-1")
            assert len(result["insights"]) == 1

            insight = result["insights"][0]
            assert insight["agent_id"] == "agent-0"
            assert "batch processing" in insight["text"]
            assert insight["scope"] == "meta"
        finally:
            store.close()

    def test_discussion_insight(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_insight(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="Round 1 insight",
                round_num=1,
            )
            result = store.query_task("task-1")
            assert len(result["insights"]) == 1
        finally:
            store.close()

    def test_rich_insight_round_trips_without_truncation(self, tmp_path):
        """A structured ~2000-char insight (hypothesis + evidence + rule) must
        round-trip intact — we widened the prompt's expected length and must
        not clip the text at the storage layer below the documented target.
        """
        store = _make_store(tmp_path)
        try:
            rich = (
                "Hypothesis: on grids where a 1-cell border of color C surrounds "
                "a rectangular region, the transformation preserves C and recolors "
                "the interior according to the dominant non-C color from the 3x3 "
                "training exemplar. "
            )
            # Pad to ~2000 chars with substantive-looking filler sentences so
            # we approximate the prompt's stated upper bound.
            rich = (
                rich
                + (
                    "Evidence: training pair 0 shows a blue border at rows "
                    "(0,5) x cols (0,5), interior shifts red->green. "
                    "Training pair 1 confirms the same frame invariant with "
                    "a different interior palette. "
                )
                * 10
            )
            # Trim to exactly 2000 to mirror the prompt guidance.
            rich = rich[:2000]
            assert 1900 < len(rich) <= 2000

            store.record_insight(
                task_id="task-rich",
                agent_id="agent-0",
                generation=1,
                text=rich,
                scope="task",
                confidence="high",
                round_num=0,
            )
            result = store.query_task("task-rich")
            assert len(result["insights"]) == 1
            assert result["insights"][0]["text"] == rich
        finally:
            store.close()

    def test_insight_text_stored_verbatim_no_truncation(self, tmp_path):
        """Insights are the primary transfer signal and must be stored
        verbatim — even a long body is never clipped at storage time.
        """
        store = _make_store(tmp_path)
        try:
            long_text = "x" * 9000
            store.record_insight(
                task_id="task-cap",
                agent_id="agent-0",
                generation=1,
                text=long_text,
                round_num=0,
            )
            result = store.query_task("task-cap")
            assert len(result["insights"]) == 1
            stored = result["insights"][0]["text"]
            assert stored == long_text
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 4. record_post with threading (parent_id)
# ---------------------------------------------------------------------------


class TestRecordPost:
    def test_post_with_reply(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            parent_id = store.record_post(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                text="I think we should try approach X",
                round_num=1,
            )
            reply_id = store.record_post(
                task_id="task-1",
                agent_id="agent-1",
                generation=1,
                text="I agree, approach X is better",
                parent_id=parent_id,
                round_num=1,
            )

            result = store.query_task("task-1")
            assert len(result["discussion"]) == 2

            parent_post = next(p for p in result["discussion"] if p["id"] == parent_id)
            reply_post = next(p for p in result["discussion"] if p["id"] == reply_id)

            assert parent_post["parent_id"] is None
            assert reply_post["parent_id"] == parent_id
            assert "approach X" in parent_post["text"]
            assert "I agree" in reply_post["text"]
        finally:
            store.close()

    def test_query_task_returns_native_score_for_posts(self, tmp_path):
        """query_task's discussion entries must carry the post author's
        native_score so the per-task distiller can weight high-score authors
        over low-score authors when claims conflict
        (``distiller._load_per_task_posts`` -> ``prompts._fmt_posts`` renders
        ``author_score``). If the field is dropped from the returned dict the
        weighting silently never fires (``p.get("native_score")`` -> None).
        """
        store = _make_store(tmp_path)
        try:
            scored_id = store.record_post(
                task_id="task-1",
                agent_id="agent-solver",
                generation=1,
                text="I solved it this way",
                round_num=0,
                native_score=1.0,
            )
            unscored_id = store.record_post(
                task_id="task-1",
                agent_id="agent-unknown",
                generation=1,
                text="not sure",
                round_num=0,
            )

            result = store.query_task("task-1", generation=1, entry_types=["post"])
            by_id = {p["id"]: p for p in result["discussion"]}
            assert by_id[scored_id]["native_score"] == 1.0
            assert by_id[unscored_id]["native_score"] is None

            # query_tasks (batched) must match query_task byte-for-byte.
            batched = store.query_tasks(["task-1"], generation=1, entry_types=["post"])
            batched_by_id = {p["id"]: p for p in batched["task-1"]["discussion"]}
            assert batched_by_id[scored_id]["native_score"] == 1.0
            assert batched_by_id[unscored_id]["native_score"] is None
        finally:
            store.close()

    def test_query_task_returns_round_num_for_posts(self, tmp_path):
        """query_task's discussion entries must carry round_num (#1043).

        Same-generation, multi-round forum consumers (per-task and
        cross-task) filter peer posts by `post.get("round_num") <
        current_round` — if the field is silently dropped from the
        returned dict, that filter always sees None and drops every post,
        making multi-round peer context permanently empty regardless of
        how the round is threaded through the prompt builders.
        """
        store = _make_store(tmp_path)
        try:
            store.record_post(
                task_id="task-1",
                agent_id="agent-0",
                generation=2,
                text="round 0 post",
                round_num=0,
            )
            result = store.query_task("task-1", generation=2, entry_types=["post"])
            assert result["discussion"][0]["round_num"] == 0

            # query_tasks (batched) must match query_task byte-for-byte.
            batched = store.query_tasks(["task-1"], generation=2, entry_types=["post"])
            assert batched["task-1"]["discussion"][0]["round_num"] == 0
        finally:
            store.close()

    def test_threaded_post_from_multiple_threads(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            errors = []

            def _post(agent_id, text):
                try:
                    store.record_post(
                        task_id="task-1",
                        agent_id=agent_id,
                        generation=1,
                        text=text,
                        round_num=1,
                    )
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=_post, args=(f"agent-{i}", f"Message from agent {i}")) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert not errors, f"Thread errors: {errors}"

            result = store.query_task("task-1")
            assert len(result["discussion"]) == 5
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 5. record_distillation + query_task
# ---------------------------------------------------------------------------


class TestRecordDistillation:
    def test_round_trip(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            assets = [
                {"asset_type": "strategy", "text": "Use divide-and-conquer on grid tasks"},
                {"asset_type": "pattern", "text": "Look for symmetry in input/output pairs"},
            ]
            entry_id = store.record_distillation(
                task_id="task-1",
                generation=1,
                assets=assets,
            )
            assert isinstance(entry_id, int)

            result = store.query_task("task-1")
            assert len(result["distilled"]) == 2
            assert result["distilled"][0]["asset_type"] == "strategy"
            assert result["distilled"][1]["asset_type"] == "pattern"
            assert result["distilled"][0]["gen"] == 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 6. query_generation filtering
# ---------------------------------------------------------------------------


class TestQueryGeneration:
    def test_bulk_read(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                native_score=0.5,
            )
            store.record_insight(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="insight 1",
            )
            store.record_post(
                task_id="t1",
                agent_id="a2",
                generation=1,
                text="discussion post",
                round_num=1,
            )
            # Different generation — should not appear
            store.record_attempt(
                task_id="t2",
                agent_id="a1",
                generation=2,
                native_score=1.0,
            )

            entries = store.query_generation(1)
            assert len(entries) == 3
            types = {e["entry_type"] for e in entries}
            assert types == {"attempt", "insight", "post"}
        finally:
            store.close()

    def test_filter_by_source_phase(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                native_score=0.5,
            )
            store.record_post(
                task_id="t1",
                agent_id="a2",
                generation=1,
                text="post",
                round_num=1,
            )

            entries = store.query_generation(1, source_phase="execution")
            assert len(entries) == 1
            assert entries[0]["entry_type"] == "attempt"

            entries = store.query_generation(1, source_phase="discussion")
            assert len(entries) == 1
            assert entries[0]["entry_type"] == "post"
        finally:
            store.close()

    def test_filter_by_entry_types(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                native_score=0.5,
            )
            store.record_insight(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="insight",
            )
            store.record_post(
                task_id="t1",
                agent_id="a2",
                generation=1,
                text="post",
                round_num=1,
            )

            entries = store.query_generation(1, entry_types=["attempt", "insight"])
            assert len(entries) == 2
            types = {e["entry_type"] for e in entries}
            assert types == {"attempt", "insight"}
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 7. signal_done + get_done_status
# ---------------------------------------------------------------------------


class TestDiscussionDone:
    def test_signal_and_status(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            status = store.get_done_status(
                task_id="t1",
                generation=1,
                expected_agents=3,
            )
            assert status["agents_done"] == 0
            assert status["agents_expected"] == 3
            assert status["all_done"] is False

            store.signal_done(task_id="t1", agent_id="a1", generation=1)
            store.signal_done(task_id="t1", agent_id="a2", generation=1)

            status = store.get_done_status(
                task_id="t1",
                generation=1,
                expected_agents=3,
            )
            assert status["agents_done"] == 2
            assert status["all_done"] is False

            store.signal_done(task_id="t1", agent_id="a3", generation=1)

            status = store.get_done_status(
                task_id="t1",
                generation=1,
                expected_agents=3,
            )
            assert status["agents_done"] == 3
            assert status["all_done"] is True
        finally:
            store.close()

    def test_idempotent_signal(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.signal_done(task_id="t1", agent_id="a1", generation=1)
            store.signal_done(task_id="t1", agent_id="a1", generation=1)  # duplicate

            status = store.get_done_status(
                task_id="t1",
                generation=1,
                expected_agents=2,
            )
            assert status["agents_done"] == 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 8. FTS5 search
# ---------------------------------------------------------------------------


class TestFTS5:
    def test_fts_index_populated(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="django__django-12345",
                agent_id="agent-0",
                generation=1,
                model_output="Fixed the QuerySet bug in views.py",
                native_score=1.0,
            )
            store.record_insight(
                task_id="sphinx__sphinx-999",
                agent_id="agent-1",
                generation=1,
                text="Sphinx autodoc needs explicit module path",
            )

            # Query FTS directly
            rows = store._execute(
                """
                SELECT rowid, task_id, content
                FROM knowledge_fts
                WHERE knowledge_fts MATCH ?
                ORDER BY rank
                LIMIT 10
                """,
                ("QuerySet",),
                fetchall=True,
            )
            assert len(rows) >= 1
            assert "django" in rows[0]["task_id"]

            # Search for insight content
            rows = store._execute(
                """
                SELECT rowid, task_id, content
                FROM knowledge_fts
                WHERE knowledge_fts MATCH ?
                ORDER BY rank
                LIMIT 10
                """,
                ("Sphinx",),
                fetchall=True,
            )
            assert len(rows) >= 1
            assert "sphinx" in rows[0]["task_id"]
        finally:
            store.close()

    def test_reopen_rebuilds_stale_fts_index(self, tmp_path):
        db_path = str(tmp_path / "test_knowledge.sqlite")
        store = KnowledgeStore(db_path)
        try:
            store.record_post(
                task_id="task-fts",
                agent_id="agent-0",
                generation=1,
                text="needle phrase survives rebuild",
            )
        finally:
            store.close()

        conn = sqlite3.connect(db_path)
        try:
            conn.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES ('delete-all')")
            conn.commit()
        finally:
            conn.close()

        reopened = KnowledgeStore(db_path)
        try:
            results = reopened.fts_search("needle")
            assert len(results) == 1
            assert results[0]["task_id"] == "task-fts"
        finally:
            reopened.close()


# ---------------------------------------------------------------------------
# 9. Read-only mode
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_read_only_queries_work(self, tmp_path):
        db_path = str(tmp_path / "test_knowledge.sqlite")
        # Populate with writable store
        store_w = KnowledgeStore(db_path)
        store_w.record_attempt(
            task_id="t1",
            agent_id="a1",
            generation=1,
            native_score=1.0,
        )
        store_w.close()

        # Open read-only
        store_r = KnowledgeStore(db_path, read_only=True)
        try:
            result = store_r.query_task("t1")
            assert len(result["attempts"]) == 1
            assert result["attempts"][0]["score"] == 1.0
        finally:
            store_r.close()

    def test_read_only_no_lock_file(self, tmp_path):
        db_path = str(tmp_path / "test_knowledge.sqlite")
        store_w = KnowledgeStore(db_path)
        store_w.close()

        store_r = KnowledgeStore(db_path, read_only=True)
        try:
            assert store_r._lock_fd is None
            assert store_r._writer_queue is None
            assert store_r._writer_thread is None
        finally:
            store_r.close()


# ---------------------------------------------------------------------------
# 10. Invalid entry_type / source_phase
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_entry_type(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            with pytest.raises(ValueError, match="Invalid entry_type"):
                with store._locked():
                    store._insert_knowledge_locked(
                        run_id=1,
                        generation=1,
                        task_id="t1",
                        agent_id="a1",
                        entry_type="invalid_type",
                        source_phase="execution",
                        content="{}",
                    )
        finally:
            store.close()

    def test_invalid_source_phase(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            with pytest.raises(ValueError, match="Invalid source_phase"):
                with store._locked():
                    store._insert_knowledge_locked(
                        run_id=1,
                        generation=1,
                        task_id="t1",
                        agent_id="a1",
                        entry_type="attempt",
                        source_phase="invalid_phase",
                        content="{}",
                    )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 11. Batch mode and query pressure
# ---------------------------------------------------------------------------


class TestBatchModeWriterThread:
    """Batched-write contract when the writer thread is already started.

    Renamed from `TestBatchMode` to avoid Python class shadowing with the
    later (line ~814) `TestBatchMode`, which silently dropped these two
    tests from pytest collection.
    """

    def test_batched_rejects_caller_thread_use_after_writer_started(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store._start_writer()
            with pytest.raises(RuntimeError, match="writer thread"):
                with store._batched():
                    pass
        finally:
            store.close()

    def test_batched_allows_writer_thread_operation_after_start(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store._start_writer()

            def write_in_batch():
                with store._batched():
                    store._conn.execute("SELECT 1").fetchone()
                return True

            assert store._run_write(write_in_batch) is True
        finally:
            store.close()


class TestQueryTaskPressure:
    def test_query_task_keeps_newest_rows_under_bucket_limit(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            for idx in range(3):
                store.record_post(
                    task_id="task-pressure",
                    agent_id=f"agent-{idx}",
                    generation=1,
                    text=f"post-{idx}",
                )

            result = store.query_task(
                "task-pressure",
                limit=2,
                entry_types=["post"],
            )

            assert [p["text"] for p in result["discussion"]] == ["post-1", "post-2"]
        finally:
            store.close()


# ---------------------------------------------------------------------------
# solved_task_ids (bulk solved-set for the distill skip)
# ---------------------------------------------------------------------------


class TestSolvedTaskIds:
    def test_resolved_only_and_score_only_both_detected(self, tmp_path):
        """Pins the semantics a naive get_best_scores swap would break:

        a task is solved if ANY attempt has eval_results.resolved is True
        OR native_score >= threshold — resolved-only tasks (score below
        threshold) must still count.
        """
        store = _make_store(tmp_path)
        try:
            # Solved via resolved=True only — score well below threshold.
            store.record_attempt(
                task_id="t-resolved",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": True},
                native_score=0.2,
            )
            # Solved via score only — resolved is False.
            store.record_attempt(
                task_id="t-score",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": False},
                native_score=1.0,
            )
            # Unsolved: resolved falsy, score below threshold.
            store.record_attempt(
                task_id="t-unsolved",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": False},
                native_score=0.4,
            )
            # Unsolved: no eval_results, no score.
            store.record_attempt(
                task_id="t-empty",
                agent_id="a1",
                generation=1,
            )

            solved = store.solved_task_ids(
                ["t-resolved", "t-score", "t-unsolved", "t-empty"],
                threshold=1.0,
            )
            assert solved == {"t-resolved", "t-score"}
        finally:
            store.close()

    def test_non_bool_resolved_values_do_not_count(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t-string-true",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": "true"},
                native_score=0.2,
            )
            store.record_attempt(
                task_id="t-int-one",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": 1},
                native_score=0.2,
            )

            solved = store.solved_task_ids(["t-string-true", "t-int-one"], threshold=1.0)

            assert solved == set()
        finally:
            store.close()

    def test_any_attempt_counts(self, tmp_path):
        """One solving attempt among many failures marks the task solved."""
        store = _make_store(tmp_path)
        try:
            store.record_attempt(task_id="t1", agent_id="a1", generation=1, native_score=0.0)
            store.record_attempt(
                task_id="t1",
                agent_id="a2",
                generation=2,
                eval_results={"resolved": True},
                native_score=0.0,
            )
            store.record_attempt(task_id="t1", agent_id="a3", generation=3, native_score=0.1)
            assert store.solved_task_ids(["t1"]) == {"t1"}
        finally:
            store.close()

    def test_threshold_above_one_drops_score_branch_not_resolved(self, tmp_path):
        """threshold > 1.0: score 1.0 no longer solves, resolved=True still does."""
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t-score-1.0",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": False},
                native_score=1.0,
            )
            store.record_attempt(
                task_id="t-resolved-low",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": True},
                native_score=0.5,
            )
            solved = store.solved_task_ids(["t-score-1.0", "t-resolved-low"], threshold=2.0)
            assert solved == {"t-resolved-low"}
        finally:
            store.close()

    def test_restricted_to_given_task_ids_and_experiment(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t-in",
                agent_id="a1",
                generation=1,
                native_score=1.0,
                experiment="exp_a",
            )
            store.record_attempt(
                task_id="t-other",
                agent_id="a1",
                generation=1,
                native_score=1.0,
                experiment="exp_a",
            )
            # Same task id solved in a DIFFERENT experiment must not leak.
            store.record_attempt(
                task_id="t-cross-exp",
                agent_id="a1",
                generation=1,
                native_score=1.0,
                experiment="exp_b",
            )

            solved = store.solved_task_ids(["t-in", "t-cross-exp", "t-missing"], experiment="exp_a")
            assert solved == {"t-in"}
        finally:
            store.close()

    def test_empty_task_ids_returns_empty_set(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            assert store.solved_task_ids([]) == set()
        finally:
            store.close()

    def test_malformed_content_row_degrades_per_row_not_whole_query(self, tmp_path):
        """One corrupt content row must not abort the bulk query.

        Without the json_valid() guard, json_extract raises
        ``sqlite3.OperationalError: malformed JSON`` for the whole chunk and
        the engine fallback disables skip-solved for the entire generation.
        Parity with the historical per-task loop: the corrupt row loses only
        its ``resolved`` branch (content -> {}), native_score still counts.
        """
        store = _make_store(tmp_path)
        try:
            # Corrupt row, but solved via native_score — must survive.
            store.record_attempt(
                task_id="t-corrupt-score",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": True},
                native_score=1.0,
            )
            # Corrupt row solved ONLY via resolved — degrades to unsolved
            # (same as the old per-task _json_loads fallback), not an error.
            store.record_attempt(
                task_id="t-corrupt-resolved",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": True},
                native_score=0.2,
            )
            # Clean rows in the same chunk must be unaffected.
            store.record_attempt(
                task_id="t-clean",
                agent_id="a1",
                generation=1,
                eval_results={"resolved": True},
                native_score=0.2,
            )
            store._execute(
                "UPDATE knowledge SET content = '{broken' "
                "WHERE entry_type = 'attempt' AND task_id IN ('t-corrupt-score', 't-corrupt-resolved')",
            )

            solved = store.solved_task_ids(
                ["t-corrupt-score", "t-corrupt-resolved", "t-clean"],
                threshold=1.0,
            )
            assert solved == {"t-corrupt-score", "t-clean"}
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestExperimentIsolation:
    def test_different_experiments_isolated(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                native_score=1.0,
                experiment="exp_a",
            )
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                native_score=0.5,
                experiment="exp_b",
            )

            result_a = store.query_task("t1", experiment="exp_a")
            assert len(result_a["attempts"]) == 1
            assert result_a["attempts"][0]["score"] == 1.0

            result_b = store.query_task("t1", experiment="exp_b")
            assert len(result_b["attempts"]) == 1
            assert result_b["attempts"][0]["score"] == 0.5
        finally:
            store.close()


class TestDbPathProperty:
    def test_db_path(self, tmp_path):
        db_path = str(tmp_path / "test.sqlite")
        store = KnowledgeStore(db_path)
        try:
            assert store.db_path == db_path
        finally:
            store.close()


class TestBatchMode:
    def test_batched_rejects_caller_thread_use(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            with pytest.raises(RuntimeError, match="writer thread"):
                with store._batched():
                    store._ensure_run("exp")
        finally:
            store.close()

    def test_batched_allows_single_writer_thread_operation(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            commit_batch_states: list[bool] = []
            original_commit = store._commit

            def tracking_commit() -> None:
                commit_batch_states.append(store._batch_mode)
                original_commit()

            store._commit = tracking_commit  # type: ignore[method-assign]

            def _op() -> int:
                with store._locked():
                    with store._batched():
                        run_id = store._ensure_run("exp")
                        store._ensure_generation(run_id, 1)
                        store._ensure_agent(run_id, "agent-0")
                        return run_id

            run_id = store._run_write(_op)

            assert isinstance(run_id, int)
            assert commit_batch_states == [True, True, True]
            assert store._batch_mode is False
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 11. Vector search (_init_vec, vec_search)
# ---------------------------------------------------------------------------


def _make_embedding(dimensions: int, value: float = 0.0) -> list[float]:
    """Create a simple embedding vector of given dimensions."""
    return [value] * dimensions


def _make_unit_embedding(dimensions: int, index: int) -> list[float]:
    """Create a unit vector with 1.0 at the given index, 0.0 elsewhere."""
    vec = [0.0] * dimensions
    vec[index % dimensions] = 1.0
    return vec


class TestInitVec:
    def test_init_vec_succeeds(self, tmp_path):
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=32)
        try:
            assert store._vec_enabled is True
            assert store._vec_dimensions == 32

            # Verify the virtual table exists
            tables = store._execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
                fetchall=True,
            )
            names = {r["name"] for r in tables}
            assert "knowledge_vec" in names
        finally:
            store.close()

    def test_vec_disabled_by_default(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            assert store._vec_enabled is False
        finally:
            store.close()

    def test_read_only_store_enables_existing_vec_table(self, tmp_path):
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        dim = 32
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=dim)
        try:
            emb = _make_unit_embedding(dim, 0)
            store.record_insight(
                task_id="task-a",
                agent_id="agent-0",
                generation=1,
                text="boundary pattern",
                embedding=emb,
            )
        finally:
            store.close()

        ro_store = KnowledgeStore(db_path, read_only=True, enable_vec=True, vec_dimensions=dim)
        try:
            assert ro_store._vec_enabled is True
            results = ro_store.vec_search(_make_unit_embedding(dim, 0), max_results=5)
            assert len(results) == 1
            assert results[0]["task_id"] == "task-a"
        finally:
            ro_store.close()


class TestVecSearch:
    def test_nearest_neighbor_ordering(self, tmp_path):
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        dim = 32
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=dim)
        try:
            # Insert two entries with different embeddings
            emb_a = _make_unit_embedding(dim, 0)  # [1, 0, 0, ...]
            emb_b = _make_unit_embedding(dim, 1)  # [0, 1, 0, ...]

            store.record_insight(
                task_id="task-a",
                agent_id="a1",
                generation=1,
                text="insight A",
                embedding=emb_a,
            )
            store.record_insight(
                task_id="task-b",
                agent_id="a1",
                generation=1,
                text="insight B",
                embedding=emb_b,
            )

            # Search with emb_a — task-a should be closest
            results = store.vec_search(emb_a, max_results=5)
            assert len(results) == 2
            assert results[0]["task_id"] == "task-a"
            assert results[0]["distance"] < results[1]["distance"]
        finally:
            store.close()

    def test_filter_by_entry_types(self, tmp_path):
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        dim = 32
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=dim)
        try:
            emb = _make_embedding(dim, 0.5)

            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                model_output="patch",
                native_score=1.0,
                embedding=emb,
            )
            store.record_insight(
                task_id="t2",
                agent_id="a1",
                generation=1,
                text="an insight",
                embedding=emb,
            )

            # Filter to insights only
            results = store.vec_search(emb, entry_types=["insight"])
            assert len(results) == 1
            assert results[0]["entry_type"] == "insight"
            assert results[0]["task_id"] == "t2"

            # Filter to attempts only
            results = store.vec_search(emb, entry_types=["attempt"])
            assert len(results) == 1
            assert results[0]["entry_type"] == "attempt"
        finally:
            store.close()

    def test_filter_by_experiment_mixed_experiments(self, tmp_path):
        """Experiment filtering returns the same rows as before the JOIN.

        Entries from two experiments share the vec index; filtering by each
        experiment must return exactly that experiment's entries (post-filter
        selection semantics), and no filter returns all of them.
        """
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        dim = 32
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=dim)
        try:
            emb = _make_unit_embedding(dim, 0)
            id_a = store.record_insight(
                task_id="task-exp-a",
                agent_id="a1",
                generation=1,
                text="insight in exp-a",
                embedding=emb,
                experiment="exp-a",
            )
            id_b = store.record_insight(
                task_id="task-exp-b",
                agent_id="a1",
                generation=1,
                text="insight in exp-b",
                embedding=emb,
                experiment="exp-b",
            )

            results_a = store.vec_search(emb, max_results=5, experiment="exp-a")
            assert [r["id"] for r in results_a] == [id_a]
            assert results_a[0]["task_id"] == "task-exp-a"

            results_b = store.vec_search(emb, max_results=5, experiment="exp-b")
            assert [r["id"] for r in results_b] == [id_b]
            assert results_b[0]["task_id"] == "task-exp-b"

            # No experiment filter: both rows come back.
            results_all = store.vec_search(emb, max_results=5)
            assert {r["id"] for r in results_all} == {id_a, id_b}

            # Unknown experiment matches nothing.
            assert store.vec_search(emb, max_results=5, experiment="exp-none") == []
        finally:
            store.close()

    def test_filtered_search_returns_k_valid_rows_despite_nearer_excluded(self, tmp_path):
        """#1122: a filter that excludes the nearest neighbors must not
        under-return.

        Insert many ``attempt`` rows all closest to the query, plus a few
        farther ``insight`` rows. With ``max_results=2`` the fixed
        ``max_results * 5 == 10`` over-fetch returns only the 12 nearer
        attempts, so the naive post-filter yields ZERO insights even though
        valid insights exist. The escalation-to-full-table guard must recover
        all up-to-k valid rows.
        """
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        dim = 32
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=dim)
        try:
            near = _make_unit_embedding(dim, 0)  # query direction
            far = _make_unit_embedding(dim, 1)  # orthogonal → larger distance

            # 12 attempts at distance 0 (> the 10-row over-fetch window).
            for i in range(12):
                store.record_attempt(
                    task_id=f"attempt-{i}",
                    agent_id="a1",
                    generation=1,
                    model_output=f"patch {i}",
                    embedding=near,
                )
            # 3 farther insights — the rows the filter wants to keep.
            insight_ids = [
                store.record_insight(
                    task_id=f"insight-{j}",
                    agent_id="a1",
                    generation=1,
                    text=f"insight {j}",
                    embedding=far,
                )
                for j in range(3)
            ]

            results = store.vec_search(near, max_results=2, entry_types=["insight"])
            assert len(results) == 2
            assert all(r["entry_type"] == "insight" for r in results)
            assert {r["id"] for r in results}.issubset(set(insight_ids))
        finally:
            store.close()

    def test_returns_empty_when_vec_disabled(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            results = store.vec_search([0.0] * 32)
            assert results == []
        finally:
            store.close()

    def test_record_attempt_accepts_embedding_when_vec_disabled(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            entry_id = store.record_attempt(
                task_id="task-1",
                agent_id="agent-0",
                generation=1,
                model_output="attempt",
                embedding=[0.0] * 32,
            )
            assert isinstance(entry_id, int)
            assert store.vec_search([0.0] * 32) == []
        finally:
            store.close()

    def test_record_vector_status_round_trip(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            row_id = store.record_vector_status(
                phase="run_summary",
                status="enabled",
                generation=2,
                embedding_count=3,
                skipped_count=1,
                experiment="vec-exp",
            )
            assert row_id > 0
            row = store._execute(
                """
                select vs.phase, vs.status, vs.embedding_count, vs.skipped_count
                from vector_status vs
                join runs r on r.id = vs.run_id
                where r.experiment = 'vec-exp'
                """,
                fetchone=True,
            )
            assert row == {
                "phase": "run_summary",
                "status": "enabled",
                "embedding_count": 3,
                "skipped_count": 1,
            }
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 12. FTS search (fts_search)
# ---------------------------------------------------------------------------


class TestFtsSearch:
    def test_basic_fts_search(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="django__django-12345",
                agent_id="agent-0",
                generation=1,
                model_output="Fixed the QuerySet bug in views.py",
                native_score=1.0,
            )
            store.record_insight(
                task_id="sphinx__sphinx-999",
                agent_id="agent-1",
                generation=1,
                text="Sphinx autodoc needs explicit module path",
            )

            results = store.fts_search("QuerySet")
            assert len(results) >= 1
            assert results[0]["task_id"] == "django__django-12345"
            assert results[0]["entry_type"] == "attempt"
        finally:
            store.close()

    def test_fts_search_with_entry_types_filter(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                model_output="contains searchterm here",
                native_score=0.5,
            )
            store.record_insight(
                task_id="t2",
                agent_id="a1",
                generation=1,
                text="also contains searchterm here",
            )

            # Filter to insights only
            results = store.fts_search("searchterm", entry_types=["insight"])
            assert len(results) == 1
            assert results[0]["entry_type"] == "insight"
        finally:
            store.close()

    def test_fts_search_sanitizes_special_chars(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                model_output="special content for testing",
                native_score=1.0,
            )

            # Query with special chars that would break FTS5 if not sanitized
            results = store.fts_search("special* content()")
            assert len(results) >= 1
            assert results[0]["task_id"] == "t1"
        finally:
            store.close()

    def test_fts_search_sanitizes_operators(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                model_output="some unique testword here",
                native_score=1.0,
            )

            # FTS5 operators should be stripped
            results = store.fts_search("NOT OR AND NEAR testword")
            assert len(results) >= 1
            assert results[0]["task_id"] == "t1"
        finally:
            store.close()

    def test_fts_search_empty_query_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                model_output="some content",
                native_score=1.0,
            )
            results = store.fts_search("***")
            assert results == []
        finally:
            store.close()

    def test_fts_search_with_experiment_filter(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                model_output="shared keyword here",
                native_score=1.0,
                experiment="exp_a",
            )
            store.record_attempt(
                task_id="t2",
                agent_id="a1",
                generation=1,
                model_output="shared keyword here too",
                native_score=0.5,
                experiment="exp_b",
            )

            results = store.fts_search("keyword", experiment="exp_a")
            assert len(results) == 1
            assert results[0]["task_id"] == "t1"

            results = store.fts_search("keyword", experiment="exp_b")
            assert len(results) == 1
            assert results[0]["task_id"] == "t2"
        finally:
            store.close()

    def test_fts_fallback_related_or_joins_multiword_query(self, tmp_path):
        """Regression: the ``related`` FTS fallback must OR-join query terms.

        Before the fix the caller passed the raw multi-word query through the
        AND-collapsing sanitizer, so a query whose terms live in *different*
        rows matched nothing. OR-joining (via ``fts_search(raw_match=True)``)
        returns every row containing any term.
        """
        from kcsi.memory.mcp_server import _fts_fallback_related

        store = _make_store(tmp_path)
        try:
            store.record_attempt(
                task_id="django__django-1",
                agent_id="a1",
                generation=1,
                model_output="Fixed the QuerySet ordering bug",
                native_score=1.0,
            )
            store.record_insight(
                task_id="sphinx__sphinx-2",
                agent_id="a2",
                generation=1,
                text="Sphinx autodoc needs an explicit module path",
            )

            # "queryset" and "autodoc" occur in *different* rows: an implicit
            # AND matches neither; OR-joining matches both.
            related, error = _fts_fallback_related(store, "queryset autodoc", max_results=5)
            assert error == ""
            task_ids = {r["task_id"] for r in related}
            assert task_ids == {"django__django-1", "sphinx__sphinx-2"}
        finally:
            store.close()


class TestSanitizeFtsQuery:
    def test_removes_special_chars(self):
        assert KnowledgeStore._sanitize_fts_query("hello*world()") == "hello world"

    def test_removes_fts_operators(self):
        assert KnowledgeStore._sanitize_fts_query("NOT this OR that AND NEAR") == "this that"

    def test_collapses_whitespace(self):
        assert KnowledgeStore._sanitize_fts_query("  hello   world  ") == "hello world"

    def test_empty_after_sanitize(self):
        assert KnowledgeStore._sanitize_fts_query("***") == ""

    def test_preserves_normal_text(self):
        assert KnowledgeStore._sanitize_fts_query("QuerySet filter bug") == "QuerySet filter bug"


# ---------------------------------------------------------------------------
# 13. Embedding coverage for attempts + insights
#
# Regression guard for the live-run finding that attempt and insight rows
# were written to the ``knowledge`` table but never indexed in
# ``knowledge_vec`` (0/449 coverage in Haiku baseline sweep).  These tests
# use a mock embedder (returns a deterministic constant vector) to prove
# that both entry types reach ``knowledge_vec`` when an embedding is
# supplied, and gracefully skip the vec insert when no embedding is given
# (embedder-not-ready case).
# ---------------------------------------------------------------------------


class TestAttemptInsightEmbeddingCoverage:
    def _vec_count(self, store, knowledge_id: int) -> int:
        """Count rows in ``knowledge_vec`` for a given knowledge rowid."""
        rows = store._execute(
            "SELECT COUNT(*) AS n FROM knowledge_vec WHERE knowledge_rowid = ?",
            (knowledge_id,),
            fetchall=True,
        )
        return int(rows[0]["n"]) if rows else 0

    def test_attempt_with_embedding_lands_in_vec(self, tmp_path):
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        dim = 32
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=dim)
        try:
            # Mocked embedder: returns a zero vector of the correct dim.
            emb = _make_embedding(dim, 0.0)
            attempt_id = store.record_attempt(
                task_id="task-attempt-embed",
                agent_id="agent-0",
                generation=1,
                eval_results={"resolved": True},
                model_output="patch contents",
                trace_condensed="Approach: batch. Score: 1.0. Insight: x",
                native_score=1.0,
                embedding=emb,
            )
            assert attempt_id > 0
            # Core claim: the vec row exists for this attempt.
            assert self._vec_count(store, attempt_id) == 1
            # And vec_search can retrieve it.
            results = store.vec_search(emb, max_results=5)
            assert any(r["id"] == attempt_id for r in results)
            assert any(r["entry_type"] == "attempt" for r in results)
        finally:
            store.close()

    def test_insight_with_embedding_lands_in_vec(self, tmp_path):
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        dim = 32
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=dim)
        try:
            emb = _make_embedding(dim, 0.0)
            insight_id = store.record_insight(
                task_id="task-insight-embed",
                agent_id="agent-1",
                generation=1,
                text="Normalise inputs before applying transforms",
                scope="task",
                confidence="high",
                round_num=0,
                embedding=emb,
            )
            assert insight_id > 0
            assert self._vec_count(store, insight_id) == 1
            results = store.vec_search(emb, entry_types=["insight"], max_results=5)
            assert any(r["id"] == insight_id for r in results)
        finally:
            store.close()

    def test_attempt_without_embedding_still_writes_knowledge_row(self, tmp_path):
        """Embedder-not-ready: attempt write path must not block on embedding."""
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        dim = 32
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=dim)
        try:
            # No embedding provided (simulates `_maybe_embed` returning None
            # because the embedder is still loading in the background).
            attempt_id = store.record_attempt(
                task_id="task-no-embed",
                agent_id="agent-0",
                generation=1,
                native_score=0.5,
                embedding=None,
            )
            assert attempt_id > 0
            # Knowledge row exists but vec row does NOT.
            result = store.query_task("task-no-embed")
            assert len(result["attempts"]) == 1
            assert self._vec_count(store, attempt_id) == 0
        finally:
            store.close()

    def test_insight_without_embedding_still_writes_knowledge_row(self, tmp_path):
        sqlite_vec = pytest.importorskip("sqlite_vec")
        db_path = str(tmp_path / "test_vec.sqlite")
        dim = 32
        store = KnowledgeStore(db_path, enable_vec=True, vec_dimensions=dim)
        try:
            insight_id = store.record_insight(
                task_id="task-insight-noembed",
                agent_id="agent-0",
                generation=1,
                text="Ordering matters when the output grid is rotated",
                embedding=None,
            )
            assert insight_id > 0
            result = store.query_task("task-insight-noembed")
            assert len(result["insights"]) == 1
            assert self._vec_count(store, insight_id) == 0
        finally:
            store.close()

    def test_record_attempt_embedding_ignored_when_vec_disabled(self, tmp_path):
        """Passing an embedding when vec is disabled must not raise."""
        store = _make_store(tmp_path)  # enable_vec=False by default
        try:
            attempt_id = store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                native_score=1.0,
                embedding=[0.0] * 32,  # silently ignored
            )
            assert attempt_id > 0
            result = store.query_task("t1")
            assert len(result["attempts"]) == 1
        finally:
            store.close()


# Busy-retry helper
#
# `_call_with_busy_retry` absorbs millisecond-scale `database is locked`
# contention between the engine and the optional runtime-DB sidecar.
# Non-retryable errors propagate immediately so they remain loud.
# ---------------------------------------------------------------------------


class TestBusyRetry:
    def test_transient_lock_is_retried_and_succeeds(self, tmp_path, caplog, monkeypatch):
        store = _make_store(tmp_path)
        # Speed the test up — collapse the backoff schedule so "transient
        # retries" complete in well under a second instead of ~3s total.
        monkeypatch.setattr(KnowledgeStore, "_BUSY_RETRY_DELAYS_SEC", (0.0, 0.0, 0.0, 0.0, 0.0))
        try:
            calls = {"n": 0}

            def fn():
                calls["n"] += 1
                if calls["n"] < 3:
                    raise sqlite3.OperationalError("database is locked")
                return "ok"

            with caplog.at_level("WARNING", logger="kcsi.memory.knowledge_store"):
                assert store._call_with_busy_retry(fn) == "ok"

            assert calls["n"] == 3, "fn should be called until it stops raising"
            # The first transient lock should produce one warning so chronic
            # contention is observable.
            assert any("transient SQLite lock" in rec.message for rec in caplog.records), (
                "first retry should emit a warning"
            )
        finally:
            store.close()

    def test_non_transient_error_is_not_retried(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            calls = {"n": 0}

            def fn():
                calls["n"] += 1
                # Schema/syntax-class OperationalError — must NOT be retried;
                # retrying would mask a real bug under a budget of latency.
                raise sqlite3.OperationalError("no such column: bogus")

            with pytest.raises(sqlite3.OperationalError, match="no such column"):
                store._call_with_busy_retry(fn)
            assert calls["n"] == 1, "non-transient errors must propagate on first call"
        finally:
            store.close()

    def test_unrelated_exception_is_not_retried(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            calls = {"n": 0}

            def fn():
                calls["n"] += 1
                raise ValueError("programming error")

            with pytest.raises(ValueError, match="programming error"):
                store._call_with_busy_retry(fn)
            assert calls["n"] == 1
        finally:
            store.close()

    def test_persistent_lock_exhausts_retries_then_raises(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        monkeypatch.setattr(KnowledgeStore, "_BUSY_RETRY_DELAYS_SEC", (0.0, 0.0, 0.0))
        try:
            calls = {"n": 0}

            def fn():
                calls["n"] += 1
                raise sqlite3.OperationalError("database is busy")

            with pytest.raises(sqlite3.OperationalError, match="database is busy"):
                store._call_with_busy_retry(fn)
            # Initial call + len(_BUSY_RETRY_DELAYS_SEC) retries = 4 total.
            assert calls["n"] == 4
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Writer-thread batching — `record_*` writes must each route through ONE
# writer-queue dispatch. Previously the chain was 4 dispatches per record
# (run / generation / agent / insert), which serialized 4N round-trips
# through the single writer thread for N drained forum events.
# ---------------------------------------------------------------------------


class TestWriterBatching:
    @staticmethod
    def _wrap_writer_queue(store):
        """Replace the writer-queue with a dispatch-counting wrapper."""
        original_queue = store._writer_queue
        if original_queue is None:
            pytest.skip("writer queue not active in this mode")

        class CountingQueue:
            def __init__(self, inner):
                self.inner = inner
                self.dispatch_count = 0

            def put(self, item):
                self.dispatch_count += 1
                return self.inner.put(item)

            def get(self):
                return self.inner.get()

            def qsize(self):
                return self.inner.qsize()

        wrapper = CountingQueue(original_queue)
        store._writer_queue = wrapper
        return wrapper

    def test_record_attempt_uses_single_writer_dispatch(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            counter = self._wrap_writer_queue(store)
            store.record_attempt(
                task_id="t1",
                agent_id="a1",
                generation=1,
                native_score=0.5,
            )
            assert counter.dispatch_count == 1, (
                f"record_attempt should be 1 writer-queue round-trip; got {counter.dispatch_count}"
            )
        finally:
            store.close()

    def test_record_insight_uses_single_writer_dispatch(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            counter = self._wrap_writer_queue(store)
            store.record_insight(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="Use breakpoints",
                round_num=0,
            )
            assert counter.dispatch_count == 1
        finally:
            store.close()

    def test_record_post_uses_single_writer_dispatch(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            counter = self._wrap_writer_queue(store)
            store.record_post(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="hello",
            )
            assert counter.dispatch_count == 1
        finally:
            store.close()

    def test_record_distillation_uses_single_writer_dispatch(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            counter = self._wrap_writer_queue(store)
            store.record_distillation(
                task_id="t1",
                generation=1,
                bundle={"transferable_insights": ["x"]},
                scope="per_task",
            )
            assert counter.dispatch_count == 1
        finally:
            store.close()

    def test_signal_done_uses_single_writer_dispatch(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            counter = self._wrap_writer_queue(store)
            store.signal_done(task_id="t1", agent_id="a1", generation=1)
            assert counter.dispatch_count == 1
        finally:
            store.close()

    def test_batched_failure_rolls_back_partial_writes(self, tmp_path, monkeypatch):
        """If a multi-statement closure under _batched() raises mid-way,
        the prerequisite rows must NOT be left committed.

        Pre-fix: _batched.__exit__ always committed in finally — if the
        final insert raised, ensure_run/_generation/_agent INSERTs were
        committed without the corresponding knowledge row. Post-fix:
        _batched rolls back the connection on exception.
        """
        store = _make_store(tmp_path)
        try:
            # Force the inner _insert_knowledge_locked to raise after the
            # _ensure_* helpers have already inserted prerequisite rows.
            original_insert = store._insert_knowledge_locked

            def boom(*args, **kwargs):
                raise RuntimeError("simulated insert failure")

            monkeypatch.setattr(store, "_insert_knowledge_locked", boom)

            # Pre-condition: no runs row for "rollback-test".
            pre_runs = store._execute(
                "SELECT COUNT(*) AS cnt FROM runs WHERE experiment = ?",
                ("rollback-test",),
                fetchone=True,
            )
            assert pre_runs["cnt"] == 0

            with pytest.raises(RuntimeError, match="simulated insert failure"):
                store.record_post(
                    task_id="t1",
                    agent_id="a1",
                    generation=1,
                    text="will-fail",
                    experiment="rollback-test",
                )

            # Post-condition: the failed write left NO runs row behind
            # — _batched rolled back.
            post_runs = store._execute(
                "SELECT COUNT(*) AS cnt FROM runs WHERE experiment = ?",
                ("rollback-test",),
                fetchone=True,
            )
            assert post_runs["cnt"] == 0, (
                "rollback failed: prerequisite runs row was committed despite the inner insert raising"
            )

            # And restoring the patch + retrying must succeed cleanly
            # (connection is not in a wedged state).
            monkeypatch.setattr(store, "_insert_knowledge_locked", original_insert)
            store.record_post(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="works-now",
                experiment="rollback-test",
            )
            page = store.query_task("t1", experiment="rollback-test")
            assert len(page["discussion"]) == 1
            assert page["discussion"][0]["text"] == "works-now"
        finally:
            store.close()

    def test_one_hundred_record_posts_use_one_hundred_dispatches(self, tmp_path):
        """Pre-fix: 100 events × 4 ops/record = 400 writer-queue round-trips.
        Post-fix: 100 events × 1 op/record = 100 round-trips.

        Concrete proof of the savings rather than relying on per-record
        assertions to compose.
        """
        store = _make_store(tmp_path)
        try:
            counter = self._wrap_writer_queue(store)
            for i in range(100):
                store.record_post(
                    task_id="t1",
                    agent_id="a1",
                    generation=1,
                    text=f"post-{i}",
                    external_id=f"ext-{i}",
                )
            assert counter.dispatch_count == 100, (
                f"100 record_post calls should produce 100 writer dispatches "
                f"(was 400 before this PR); got {counter.dispatch_count}"
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Concurrent write safety: parallel agents (the paper's central scaling
# claim) write to the same knowledge DB. None of the writes may be lost
# under contention.
# ---------------------------------------------------------------------------


class TestConcurrentWriteSafety:
    def test_parallel_record_attempt_does_not_drop_writes(self, tmp_path):
        """20 threads, 25 record_attempt calls each → 500 rows persisted.

        The paper's parallel-disposable-agent design races multiple agents
        through ``record_attempt`` per generation. SQLite + the store's
        single-writer thread + advisory flock must serialize correctly with
        zero dropped rows. The threshold is strict: every issued write must
        appear in the round-trip query, otherwise the scaling story is sand.
        """
        store = _make_store(tmp_path)
        n_threads = 20
        n_writes_per_thread = 25

        errors: list[BaseException] = []
        barrier = threading.Barrier(n_threads)

        def writer(thread_idx: int) -> None:
            barrier.wait()
            try:
                for i in range(n_writes_per_thread):
                    store.record_attempt(
                        task_id=f"task-{thread_idx}",
                        agent_id=f"agent-{thread_idx}-{i}",
                        generation=1,
                        eval_results={"resolved": False},
                        model_output=f"output-{thread_idx}-{i}",
                        native_score=float(i) / n_writes_per_thread,
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        try:
            threads = [threading.Thread(target=writer, args=(t,), name=f"writer-{t}") for t in range(n_threads)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
                assert not thread.is_alive(), f"writer thread {thread.name} hung"

            assert not errors, f"writer threads raised: {errors!r}"

            total = 0
            for t in range(n_threads):
                result = store.query_task(f"task-{t}")
                total += len(result["attempts"])
            assert total == n_threads * n_writes_per_thread, (
                f"expected {n_threads * n_writes_per_thread} attempts, got {total}"
            )
        finally:
            store.close()

    def test_parallel_record_post_does_not_drop_writes(self, tmp_path):
        """Same invariant for ``record_post`` (forum-drain path)."""
        store = _make_store(tmp_path)
        n_threads = 10
        n_writes_per_thread = 30
        errors: list[BaseException] = []
        barrier = threading.Barrier(n_threads)

        def writer(thread_idx: int) -> None:
            barrier.wait()
            try:
                for i in range(n_writes_per_thread):
                    store.record_post(
                        task_id="shared-task",
                        agent_id=f"agent-{thread_idx}",
                        generation=1,
                        text=f"post-{thread_idx}-{i}",
                        external_id=f"ext-{thread_idx}-{i}",
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        try:
            threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
                assert not t.is_alive()

            assert not errors, f"writer threads raised: {errors!r}"

            # query_task's per-bucket default is 50; pass a higher limit so the
            # assertion compares total written vs total visible, not vs the cap.
            result = store.query_task("shared-task", limit=n_threads * n_writes_per_thread * 2)
            assert len(result["discussion"]) == n_threads * n_writes_per_thread, (
                f"expected {n_threads * n_writes_per_thread} posts, got {len(result['discussion'])}"
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 16. _run_write cancellation (issue #778, ported from store.py / issue #767)
# ---------------------------------------------------------------------------


class _FastWaitEvent(threading.Event):
    """Event whose wait() blocks at most 0.3s (still returning immediately
    once set) — compresses _run_write's 120s stall floor for tests."""

    def wait(self, timeout=None):
        return super().wait(timeout=0.3)


class TestRunWriteCancellation:
    """_run_write must be at-most-once: a write whose stall timeout fires is
    cancelled, not left in the queue to apply later (issue #778). Before the
    fix, a timed-out closure stayed queued and executed on writer recovery,
    so a knowledge write reported as failed could silently apply later — an
    inconsistency hazard on the authoritative DB."""

    @staticmethod
    def _occupy_writer(store, claimed: threading.Event, release: threading.Event) -> threading.Thread:
        """Block the single writer thread until ``release`` is set. Returns
        only after the worker has actually claimed the blocker closure, so
        subsequent writes deterministically queue behind it. The events must
        be real (created before the _FastWaitEvent patch)."""

        def _blocker():
            claimed.set()
            release.wait(5.0)

        def _target():
            try:
                store._run_write(_blocker)
            except RuntimeError:
                pass  # under _FastWaitEvent the blocker submission itself times out

        t = threading.Thread(target=_target)
        t.start()
        assert claimed.wait(2.0), "worker never claimed the blocker closure"
        return t

    def test_timed_out_write_is_cancelled_and_never_applies(self, tmp_path):
        store = _make_store(tmp_path)
        executed = []
        claimed, release = threading.Event(), threading.Event()
        with patch("threading.Event", _FastWaitEvent):
            blocker = self._occupy_writer(store, claimed, release)
            with pytest.raises(RuntimeError, match="write cancelled"):
                store._run_write(lambda: executed.append(1))
        release.set()
        blocker.join(timeout=5)
        store._run_write(lambda: None)  # barrier: queue fully drained
        assert executed == []
        store.close()

    def test_running_write_gets_grace_period_instead_of_cancel(self, tmp_path):
        """A closure the worker is already executing cannot be cancelled; the
        caller waits one more window and consumes the result normally."""
        store = _make_store(tmp_path)
        with patch("threading.Event", _FastWaitEvent):
            ret = store._run_write(lambda: time.sleep(0.45) or 42)
        assert ret == 42
        store.close()

    def test_unclaimable_running_write_raises_indeterminate(self, tmp_path):
        """If the closure is still executing after the grace window, the error
        must be the dedicated WriteIndeterminateError so callers know the
        write may still land and must not retry."""
        from kcsi.errors import WriteIndeterminateError

        store = _make_store(tmp_path)
        finished = []

        def _slow():
            time.sleep(1.2)
            finished.append(1)

        with patch("threading.Event", _FastWaitEvent):
            with pytest.raises(WriteIndeterminateError, match="may still be applied"):
                store._run_write(_slow)
        store._run_write(lambda: None)  # barrier: waits for _slow to finish
        assert finished == [1]  # the message was truthful — it did apply
        store.close()
