"""run_drain_batch resolves run/generation refs once per drain (#978 M2).

run_id and gen_id are constant across every op in a drain, so they are resolved
ONCE (in the outer transaction, before any per-op SAVEPOINT) and cached; the
per-op ``_ensure_run/gen_locked`` then short-circuit instead of re-running
``INSERT OR IGNORE`` + ``SELECT`` per event.
"""

import tempfile
from pathlib import Path

from ksi.memory.knowledge_store import KnowledgeStore


def _post_ops(ks: KnowledgeStore, *, generation: int, n: int):
    return [
        (
            lambda i=i: ks._record_post_locked(
                task_id=f"t{i}",
                agent_id=f"a{i}",
                generation=generation,
                text="hello",
                experiment="exp",
            )
        )
        for i in range(n)
    ]


def _count_ref_inserts(seen: list[str]) -> tuple[int, int]:
    runs = sum(1 for s in seen if "INSERT OR IGNORE INTO runs" in s)
    gens = sum(1 for s in seen if "INSERT OR IGNORE INTO generations" in s)
    return runs, gens


def test_drain_resolves_run_and_generation_once():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            ops = _post_ops(ks, generation=1, n=6)
            seen: list[str] = []
            ks._conn.set_trace_callback(lambda stmt: seen.append(stmt))
            try:
                results = ks.run_drain_batch(ops, experiment="exp", generation=1)
            finally:
                ks._conn.set_trace_callback(None)

            assert all(ok for ok, _ in results)
            runs, gens = _count_ref_inserts(seen)
            # Resolved once for the whole drain, not once per op.
            assert runs == 1, f"expected 1 runs INSERT OR IGNORE, got {runs}"
            assert gens == 1, f"expected 1 generations INSERT OR IGNORE, got {gens}"
            # The posts actually landed with the correct run/generation.
            page = ks.query_task("t0", entry_types=["post"])
            assert page["discussion"], "post should be persisted"
            assert page["discussion"][0]["generation"] == 1
            # Cache is cleared after the drain (never leaks to later writes).
            assert ks._drain_ensured_run is None
            assert ks._drain_ensured_gens == {}
        finally:
            ks.close()


def test_drain_without_hint_resolves_refs_per_op():
    """Omitting experiment+generation preserves the prior per-op resolution."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:
            ops = _post_ops(ks, generation=1, n=4)
            seen: list[str] = []
            ks._conn.set_trace_callback(lambda stmt: seen.append(stmt))
            try:
                results = ks.run_drain_batch(ops)  # no hint
            finally:
                ks._conn.set_trace_callback(None)

            assert all(ok for ok, _ in results)
            runs, gens = _count_ref_inserts(seen)
            # Each op resolves its own refs → one INSERT OR IGNORE per op. This
            # is the redundant work the hinted path eliminates (proves teeth).
            assert runs == 4, f"expected 4 per-op runs inserts, got {runs}"
            assert gens == 4, f"expected 4 per-op generations inserts, got {gens}"
        finally:
            ks.close()


def test_drain_ref_resolution_is_rollback_safe():
    """A failing op rolls back to its savepoint but must NOT orphan the run/gen
    rows (resolved before any savepoint) — later ops still commit correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        ks = KnowledgeStore(str(Path(tmp) / "k.sqlite"), default_experiment="exp")
        try:

            def _boom():
                raise RuntimeError("simulated per-op failure")

            ops = [
                lambda: ks._record_post_locked(
                    task_id="t0", agent_id="a0", generation=2, text="first", experiment="exp"
                ),
                _boom,
                lambda: ks._record_post_locked(
                    task_id="t1", agent_id="a1", generation=2, text="third", experiment="exp"
                ),
            ]
            results = ks.run_drain_batch(ops, experiment="exp", generation=2)

            assert [ok for ok, _ in results] == [True, False, True]
            # Both good posts landed (the middle rollback didn't orphan run/gen).
            assert ks.query_task("t0", entry_types=["post"])["discussion"]
            assert ks.query_task("t1", entry_types=["post"])["discussion"]
            assert ks._drain_ensured_run is None
            assert ks._drain_ensured_gens == {}
        finally:
            ks.close()
