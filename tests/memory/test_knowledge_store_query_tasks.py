"""Parity tests for KnowledgeStore.query_tasks (the batched query_task).

query_tasks replaces the engine's N+1 per-task loop with a single
``WHERE task_id IN (...)`` pass using ``ROW_NUMBER() OVER (PARTITION BY ...)``.
These tests pin that the batched result is byte-identical to calling
``query_task`` once per id, including the per-bucket ``limit`` semantics and
the ``generation`` / ``entry_types`` filters.
"""

from kcsi.memory.knowledge_store import KnowledgeStore


def _seed_store(store: KnowledgeStore) -> list[str]:
    """Populate a small multi-task, multi-type, multi-generation fixture.

    Returns the list of task ids written.
    """
    task_ids = ["alpha", "beta", "gamma"]
    for tid in task_ids:
        # More attempts than the bucket limit used below (8) for `alpha`, so the
        # ROW_NUMBER per-partition cap is actually exercised, not a no-op.
        n_attempts = 12 if tid == "alpha" else 3
        for i in range(n_attempts):
            store.record_attempt(
                task_id=tid,
                agent_id=f"agent-{i % 3}",
                generation=1 + (i % 2),
                eval_results={"resolved": i == 0},
                model_output=f"diff for {tid} #{i}",
                trace_condensed=f"trace {tid} {i}",
                insights=[f"insight {tid} {i}"],
                native_score=float(i) / 10.0,
            )
        # Discussion posts.
        for j in range(2):
            store.record_post(
                task_id=tid,
                agent_id=f"agent-{j}",
                generation=1,
                text=f"post {tid} {j}",
            )
        # Execution-time insights (round_num == 0).
        store.record_insight(
            task_id=tid,
            agent_id="agent-0",
            generation=2,
            text=f"std insight {tid}",
            scope="task",
        )
        # A distillation row (legacy assets wire format).
        store.record_distillation(
            task_id=tid,
            generation=2,
            assets=[{"asset_type": "rule", "text": f"rule {tid}"}],
        )
    return task_ids


def _per_task(store: KnowledgeStore, task_ids, **kwargs) -> dict:
    return {tid: store.query_task(tid, **kwargs) for tid in task_ids}


class TestQueryTasksParity:
    def test_matches_per_task_default_buckets(self, tmp_path):
        store = KnowledgeStore(tmp_path / "k.sqlite")
        try:
            task_ids = _seed_store(store)
            batched = store.query_tasks(task_ids, limit=8)
            expected = _per_task(store, task_ids, limit=8)
            assert batched == expected
            # Per-bucket cap actually engaged for the heavy task.
            assert len(batched["alpha"]["attempts"]) == 8
            assert len(batched["beta"]["attempts"]) == 3
        finally:
            store.close()

    def test_matches_per_task_with_entry_types(self, tmp_path):
        store = KnowledgeStore(tmp_path / "k.sqlite")
        try:
            task_ids = _seed_store(store)
            kwargs = {"entry_types": ["attempt", "insight"], "limit": 8}
            batched = store.query_tasks(task_ids, **kwargs)
            expected = _per_task(store, task_ids, **kwargs)
            assert batched == expected
            # The engine's enrich path uses exactly this filter; discussion and
            # distilled stay empty.
            for tid in task_ids:
                assert batched[tid]["discussion"] == []
                assert batched[tid]["distilled"] == []
                assert batched[tid]["insights"]  # insight rows present
        finally:
            store.close()

    def test_matches_per_task_with_generation_filter(self, tmp_path):
        store = KnowledgeStore(tmp_path / "k.sqlite")
        try:
            task_ids = _seed_store(store)
            kwargs = {"generation": 1, "limit": 50}
            batched = store.query_tasks(task_ids, **kwargs)
            expected = _per_task(store, task_ids, **kwargs)
            assert batched == expected
        finally:
            store.close()

    def test_missing_and_duplicate_ids(self, tmp_path):
        store = KnowledgeStore(tmp_path / "k.sqlite")
        try:
            _seed_store(store)
            # Unknown id -> empty page; duplicate id -> single entry (no
            # double-counting); blank id -> dropped.
            result = store.query_tasks(["alpha", "alpha", "unknown", ""], limit=8)
            assert set(result) == {"alpha", "unknown"}
            assert result["unknown"] == store.query_task("unknown", limit=8)
            assert result["alpha"] == store.query_task("alpha", limit=8)
        finally:
            store.close()

    def test_empty_input_returns_empty_mapping(self, tmp_path):
        store = KnowledgeStore(tmp_path / "k.sqlite")
        try:
            assert store.query_tasks([]) == {}
        finally:
            store.close()
