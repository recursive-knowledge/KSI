"""Tests for bulk forum drain optimizations (C5+F2)."""

from kcsi.memory.knowledge_store import KnowledgeStore


def _make_store(tmp_path, **kwargs):
    db_path = str(tmp_path / "test_knowledge.sqlite")
    return KnowledgeStore(db_path, **kwargs)


class TestBulkHasExternalIds:
    def test_returns_set_of_existing_ids(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_post(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="hello",
                external_id="ext-1",
            )
            store.record_post(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="world",
                external_id="ext-2",
            )
            result = store.bulk_has_external_ids(
                ["ext-1", "ext-2", "ext-3", "ext-4"],
            )
            assert result == {"ext-1", "ext-2"}
        finally:
            store.close()

    def test_empty_input_returns_empty_set(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            assert store.bulk_has_external_ids([]) == set()
        finally:
            store.close()

    def test_filters_by_experiment(self, tmp_path):
        store = _make_store(tmp_path, default_experiment="exp-A")
        try:
            store.record_post(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="hello",
                external_id="ext-1",
                experiment="exp-A",
            )
            store.record_post(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="world",
                external_id="ext-2",
                experiment="exp-B",
            )
            result = store.bulk_has_external_ids(
                ["ext-1", "ext-2"],
                experiment="exp-A",
            )
            assert result == {"ext-1"}
        finally:
            store.close()

    def test_chunks_large_id_lists_to_avoid_sqlite_variable_limit(self, tmp_path):
        """SQLite caps placeholders per statement (999 in older builds).

        At 50 tasks × 100 posts/round, a single drain easily exceeds 999
        events. Verify the bulk lookup works on lists > the chunk size
        AND > the conservative 999 default. Also verify the chunk cap is
        actually applied — a single SELECT with 1500+ placeholders would
        fail on older SQLite, so success here proves chunking works.
        """
        store = _make_store(tmp_path)
        try:
            # Insert 1500 posts so half should be found, half missing.
            for i in range(750):
                store.record_post(
                    task_id="t1",
                    agent_id="a1",
                    generation=1,
                    text=f"hello-{i}",
                    external_id=f"ext-{i}",
                )
            ids_to_check = [f"ext-{i}" for i in range(1500)]
            result = store.bulk_has_external_ids(ids_to_check)
            # Exactly the first 750 are real; the rest never existed.
            assert len(result) == 750
            assert "ext-0" in result
            assert "ext-749" in result
            assert "ext-750" not in result
            assert "ext-1499" not in result
        finally:
            store.close()


class TestEnsureRefs:
    def test_returns_cached_ids(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            run_id, gen_id, agent_id_ref = store.ensure_refs(
                experiment="default",
                generation=1,
                agent_id="agent-0",
            )
            assert isinstance(run_id, int)
            assert isinstance(gen_id, int)
            assert isinstance(agent_id_ref, int)
        finally:
            store.close()

    def test_repeated_calls_return_same_ids(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            r1 = store.ensure_refs(
                experiment="default",
                generation=1,
                agent_id="agent-0",
            )
            r2 = store.ensure_refs(
                experiment="default",
                generation=1,
                agent_id="agent-0",
            )
            assert r1 == r2
        finally:
            store.close()


from unittest.mock import MagicMock

from kcsi.orchestrator.engine import _drain_forum_bus


class _FakeEvent:
    def __init__(self, message_type, event_id, agent_id, content, round_num=0):
        self.message_type = message_type
        self.event_id = event_id
        self.agent_id = agent_id
        self.content = content
        self.round_num = round_num


class TestDrainBulkOptimizations:
    def test_bulk_dedup_skips_existing_external_ids(self, tmp_path):
        """F2: drain should prefetch external IDs, not query per event."""
        store = _make_store(tmp_path)
        try:
            store.record_post(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="existing",
                external_id="dup-1",
            )
            events = [
                _FakeEvent("post", "dup-1", "a1", {"text": "dup", "task_id": "t1"}),
                _FakeEvent("post", "new-1", "a1", {"text": "new", "task_id": "t1"}),
            ]
            bus = MagicMock()
            bus.read_events.return_value = events
            bus.read_stale_event_ids.return_value = set()

            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
            )
            assert count == 1, "duplicate event should be skipped"
        finally:
            store.close()

    def test_drain_1000_events_writes_all(self, tmp_path):
        """Stress test: 1000 events should all drain successfully."""
        store = _make_store(tmp_path)
        try:
            events = [_FakeEvent("post", f"e-{i}", "a1", {"text": f"t-{i}", "task_id": "t1"}) for i in range(1000)]
            bus = MagicMock()
            bus.read_events.return_value = events
            bus.read_stale_event_ids.return_value = set()

            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
            )
            assert count == 1000
        finally:
            store.close()

    def test_embed_fn_called_per_event(self, tmp_path):
        """Backward-compat: per-event embed_fn still works when no batch fn provided."""
        store = _make_store(tmp_path)
        try:
            events = [_FakeEvent("post", f"e-{i}", "a1", {"text": f"text-{i}", "task_id": "t1"}) for i in range(10)]
            embed_calls = []

            def counting_embed(text):
                embed_calls.append(text)
                return [0.1] * 10

            bus = MagicMock()
            bus.read_events.return_value = events
            bus.read_stale_event_ids.return_value = set()

            _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
                embed_fn=counting_embed,
            )
            assert len(embed_calls) == 10
        finally:
            store.close()

    def test_batch_embed_fn_called_once_for_all_events(self, tmp_path):
        """When batch_embed_fn is provided, drain calls it ONCE for all
        embeddable events instead of N per-event embed_fn calls.

        This is the audit C5+F2 dominant-cost fix: per-event
        SentenceTransformer.encode() ran 100-400s of single-threaded CPU
        per drain at production scale; one batched encode amortizes that.
        """
        store = _make_store(tmp_path)
        try:
            events = [_FakeEvent("post", f"e-{i}", "a1", {"text": f"text-{i}", "task_id": "t1"}) for i in range(10)]
            batch_calls = []
            per_event_calls = []

            def batch_embed(texts):
                batch_calls.append(list(texts))
                return [[0.1] * 10 for _ in texts]

            def per_event_embed(text):
                per_event_calls.append(text)
                return [0.2] * 10

            bus = MagicMock()
            bus.read_events.return_value = events
            bus.read_stale_event_ids.return_value = set()

            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
                embed_fn=per_event_embed,
                batch_embed_fn=batch_embed,
            )
            assert count == 10
            assert len(batch_calls) == 1, f"batch_embed_fn should be called exactly once; got {len(batch_calls)}"
            assert len(batch_calls[0]) == 10, "all 10 embeddable texts should land in the single batch call"
            assert len(per_event_calls) == 0, "per-event embed_fn must NOT be called when batch_embed_fn is provided"
        finally:
            store.close()

    def test_batch_embed_skips_already_existing_external_ids(self, tmp_path):
        """Batch embed must skip events that will be deduped — no point
        burning encode cycles on rows that won't be inserted."""
        store = _make_store(tmp_path)
        try:
            # Pre-seed one event so it's already in the store.
            store.record_post(
                task_id="t1",
                agent_id="a1",
                generation=1,
                text="dup-text",
                external_id="dup-1",
            )
            events = [
                _FakeEvent("post", "dup-1", "a1", {"text": "dup-text", "task_id": "t1"}),
                _FakeEvent("post", "new-1", "a1", {"text": "new-text", "task_id": "t1"}),
                _FakeEvent("post", "new-2", "a1", {"text": "new-text-2", "task_id": "t1"}),
            ]
            batch_calls = []

            def batch_embed(texts):
                batch_calls.append(list(texts))
                return [[0.1] * 10 for _ in texts]

            bus = MagicMock()
            bus.read_events.return_value = events
            bus.read_stale_event_ids.return_value = set()

            _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
                batch_embed_fn=batch_embed,
            )
            # Only 2 new events should reach the batch — the duplicate
            # is filtered out before embedding.
            assert len(batch_calls) == 1
            assert sorted(batch_calls[0]) == ["new-text", "new-text-2"]
        finally:
            store.close()

    def test_batch_embed_skips_stale_event_ids(self, tmp_path):
        """Stale (failed-retry) events must also be filtered before embed."""
        store = _make_store(tmp_path)
        try:
            events = [
                _FakeEvent("post", "stale-1", "a1", {"text": "stale", "task_id": "t1"}),
                _FakeEvent("post", "fresh-1", "a1", {"text": "fresh", "task_id": "t1"}),
            ]
            batch_calls = []

            def batch_embed(texts):
                batch_calls.append(list(texts))
                return [[0.1] * 10 for _ in texts]

            bus = MagicMock()
            bus.read_events.return_value = events
            bus.read_stale_event_ids.return_value = {"stale-1"}

            _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
                batch_embed_fn=batch_embed,
            )
            assert len(batch_calls) == 1
            assert batch_calls[0] == ["fresh"]
        finally:
            store.close()

    def test_batch_embed_failure_falls_through_without_aborting_drain(self, tmp_path):
        """If batch_embed_fn raises, the drain still persists rows (with
        no embeddings) — embedding failure must never stop the audit
        write path."""
        store = _make_store(tmp_path)
        try:
            events = [
                _FakeEvent("post", "e-1", "a1", {"text": "t1", "task_id": "t1"}),
                _FakeEvent("post", "e-2", "a1", {"text": "t2", "task_id": "t1"}),
            ]

            def boom(_texts):
                raise RuntimeError("embedder offline")

            bus = MagicMock()
            bus.read_events.return_value = events
            bus.read_stale_event_ids.return_value = set()

            count = _drain_forum_bus(
                forum_bus=bus,
                knowledge=store,
                generation=1,
                batch_embed_fn=boom,
            )
            assert count == 2, "drain must persist all events even when embedder fails"
        finally:
            store.close()
