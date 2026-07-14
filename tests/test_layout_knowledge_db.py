"""Unit tests for per-experiment knowledge/runtime DB path derivation.

Covers the knowledge/runtime DB path helpers that live in ``ksi.layout``.
They are reused by ``ksi.orchestrator.engine`` and
``ksi.runtime.container_host`` to produce a per-experiment
``<stem>_knowledge.sqlite`` and ``<stem>_runtime.sqlite`` paths instead of a
single shared ``knowledge.sqlite``.

This prevents two campaigns that previously shared one legacy memory directory
directory (e.g. ``baseline_haiku_arc1`` + ``baseline_haiku_arc2``) from
clobbering each other's knowledge rows into one file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ksi.layout import (
    default_knowledge_db_path,
    default_runtime_db_path,
    derive_legacy_sibling,
    derive_runtime_sibling,
    legacy_flat_knowledge_db_path,
)


def test_default_runtime_db_path_uses_runtime_suffix():
    result = default_runtime_db_path("my exp")
    assert result.name == "my_exp_runtime.sqlite"


class TestDeriveLegacySibling:
    def test_canonical_memory_stem_becomes_per_experiment_knowledge(self):
        # Given: /x/y/foo_memory.sqlite
        # Expect: /x/y/foo_knowledge.sqlite
        result = derive_legacy_sibling("/x/y/foo_memory.sqlite", "knowledge")
        assert result == str(Path("/x/y/foo_knowledge.sqlite").resolve())

    def test_canonical_runtime_stem_becomes_per_experiment_knowledge(self):
        result = derive_legacy_sibling("/x/y/foo_runtime.sqlite", "knowledge")
        assert result == str(Path("/x/y/foo_knowledge.sqlite").resolve())

    def test_legacy_swarms_sqlite_preserves_shared_name(self):
        # Historical pre-rename artifact name: "swarms.sqlite" ->
        # /x/y/knowledge.sqlite (unchanged); deliberate keep.
        result = derive_legacy_sibling("/x/y/swarms.sqlite", "knowledge")
        assert result == str(Path("/x/y/knowledge.sqlite").resolve())

    def test_legacy_task_memory_sqlite_preserves_shared_name(self):
        result = derive_legacy_sibling("/x/y/task_memory.sqlite", "knowledge")
        assert result == str(Path("/x/y/knowledge.sqlite").resolve())

    def test_arbitrary_stem_without_memory_suffix_uses_stem_prefix(self):
        # e.g. "mem.sqlite" -> "mem_knowledge.sqlite"
        result = derive_legacy_sibling("/x/y/mem.sqlite", "knowledge")
        assert result == str(Path("/x/y/mem_knowledge.sqlite").resolve())

    def test_forum_kind_uses_same_rules(self):
        # The helper is reused for forum + task_docs migration paths; verify
        # the kind parameter propagates cleanly.
        result = derive_legacy_sibling("/x/y/exp_memory.sqlite", "forum")
        assert result == str(Path("/x/y/exp_forum.sqlite").resolve())

    def test_two_experiments_get_distinct_knowledge_paths(self):
        # Regression: baseline_haiku_arc1 + baseline_haiku_arc2 sharing a
        # memory dir must NOT collapse into a single knowledge.sqlite.
        arc1 = derive_legacy_sibling("/runtime_state/memory/baseline_haiku_arc1_memory.sqlite", "knowledge")
        arc2 = derive_legacy_sibling("/runtime_state/memory/baseline_haiku_arc2_memory.sqlite", "knowledge")
        assert arc1 != arc2
        assert arc1.endswith("baseline_haiku_arc1_knowledge.sqlite")
        assert arc2.endswith("baseline_haiku_arc2_knowledge.sqlite")

    def test_non_sqlite_extension_falls_back_to_kind_filename(self):
        # Non-.sqlite paths fall back to "<parent>/<kind>.sqlite" (legacy shape).
        result = derive_legacy_sibling("/x/y/foo.db", "knowledge")
        assert result == str(Path("/x/y/knowledge.sqlite").resolve())


def test_default_knowledge_db_path_uses_per_experiment_subdir():
    result = default_knowledge_db_path("my exp")
    assert result.name == "my_exp_knowledge.sqlite"
    assert result.parent.name == "my_exp"  # per-experiment subdir
    assert result.parent.parent.name == "knowledge"  # under runtime_state/knowledge/


def test_runtime_sibling_of_default_stays_in_subdir():
    kdb = default_knowledge_db_path("exp1")
    sib = Path(derive_runtime_sibling(str(kdb)))
    assert sib.parent == kdb.parent  # same isolated subdir
    assert sib.name == "exp1_runtime.sqlite"


def test_legacy_flat_knowledge_db_path_is_flat():
    result = legacy_flat_knowledge_db_path("my exp")
    assert result.name == "my_exp_knowledge.sqlite"
    assert result.parent.name == "knowledge"  # flat — no per-exp subdir


class TestCliReExport:
    """The cli module kept ``_derive_legacy_sibling`` as a backward-compat alias."""

    def test_cli_alias_matches_layout_helper(self):
        from ksi.cli import _derive_legacy_sibling
        from ksi.layout import derive_legacy_sibling as canonical

        # Same callable, not a re-implementation.
        assert _derive_legacy_sibling is canonical


def _make_wal_db(path: Path, marker: str) -> None:
    """Write a real WAL-mode SQLite DB with one row so migration can be
    checked for data preservation (and so -wal/-shm sidecars may be present)."""
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (marker,))
        conn.commit()
    finally:
        conn.close()


def _read_marker(path: Path) -> str:
    import sqlite3

    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("SELECT v FROM t").fetchone()[0]
    finally:
        conn.close()


class TestResolveKnowledgeDbBackCompat:
    """#923 M1: new runs use the per-exp subdir; a pre-existing flat-layout DB is
    MIGRATED into the isolated subdir on first resume so the container directory
    mount stops exposing sibling experiments' DBs."""

    def test_pure_resolve_has_no_filesystem_side_effects(self, monkeypatch, tmp_path):
        """The pure resolver returns the subdir path WITHOUT creating dirs or
        migrating a legacy flat DB — that is the whole point of the split (#982
        #3). The destructive work lives in _migrate_legacy_flat_knowledge_db."""
        monkeypatch.setattr("ksi.layout.RUNTIME_KNOWLEDGE_DIR", tmp_path)
        legacy = tmp_path / "expA_knowledge.sqlite"
        _make_wal_db(legacy, "expA")
        from ksi.cli import _resolve_knowledge_db_path

        result = Path(_resolve_knowledge_db_path("", "expA"))
        assert result == (tmp_path / "expA" / "expA_knowledge.sqlite").resolve()
        assert not result.parent.exists()  # no dir created
        assert legacy.exists()  # legacy flat DB NOT migrated by the pure resolver

    def test_uses_subdir_when_nothing_exists(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ksi.layout.RUNTIME_KNOWLEDGE_DIR", tmp_path)
        from ksi.cli import _prepare_knowledge_db_path

        result = Path(_prepare_knowledge_db_path("", "expA"))
        assert result == (tmp_path / "expA" / "expA_knowledge.sqlite").resolve()
        assert result.parent.is_dir()  # parent dir created

    def test_migrates_legacy_flat_into_isolated_subdir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ksi.layout.RUNTIME_KNOWLEDGE_DIR", tmp_path)
        legacy = tmp_path / "expA_knowledge.sqlite"
        legacy_runtime = tmp_path / "expA_runtime.sqlite"
        _make_wal_db(legacy, "expA-knowledge")
        _make_wal_db(legacy_runtime, "expA-runtime")
        from ksi.cli import _prepare_knowledge_db_path

        result = Path(_prepare_knowledge_db_path("", "expA"))

        # Resolved to the isolated per-experiment subdir, data preserved.
        subdir_db = (tmp_path / "expA" / "expA_knowledge.sqlite").resolve()
        assert result == subdir_db
        assert _read_marker(result) == "expA-knowledge"
        # The flat copies are gone (moved, not duplicated).
        assert not legacy.exists()
        assert not legacy_runtime.exists()
        # The runtime sibling rode along into the same isolated dir.
        moved_runtime = result.parent / "expA_runtime.sqlite"
        assert moved_runtime.exists()
        assert _read_marker(moved_runtime) == "expA-runtime"

    def test_legacy_resume_does_not_mount_dir_with_sibling_dbs(self, monkeypatch, tmp_path):
        """Core #923 M1 isolation guarantee for the legacy-resume path: the
        directory the container will mount (``dirname(resolved)``) must contain
        ONLY this experiment's DB, never a concurrent sibling experiment's DB."""
        monkeypatch.setattr("ksi.layout.RUNTIME_KNOWLEDGE_DIR", tmp_path)
        # expA: legacy flat layout being resumed.
        _make_wal_db(tmp_path / "expA_knowledge.sqlite", "expA")
        # expB: an unrelated sibling experiment sharing the flat dir.
        _make_wal_db(tmp_path / "expB_knowledge.sqlite", "expB")
        from ksi.cli import _prepare_knowledge_db_path

        result = Path(_prepare_knowledge_db_path("", "expA"))
        mounted_dir = result.parent  # == path.dirname(dbPath) the runner mounts

        # Only expA's knowledge DB lives in the mounted dir — expB is absent.
        knowledge_dbs = sorted(p.name for p in mounted_dir.glob("*_knowledge.sqlite"))
        assert knowledge_dbs == ["expA_knowledge.sqlite"]
        assert not (mounted_dir / "expB_knowledge.sqlite").exists()
        # The sibling's DB is untouched at its original flat location.
        assert (tmp_path / "expB_knowledge.sqlite").exists()

    def test_migration_failure_fails_closed(self, monkeypatch, tmp_path):
        """#966: if migration fails with the main DB still at the flat path,
        resolving it in place would bind-mount the shared dir and re-expose
        sibling experiments. The resume must FAIL CLOSED (raise) rather than
        silently mount the shared dir — the operator recovers by moving the DB
        or passing --knowledge-db-path."""
        monkeypatch.setattr("ksi.layout.RUNTIME_KNOWLEDGE_DIR", tmp_path)
        legacy = tmp_path / "expA_knowledge.sqlite"
        _make_wal_db(legacy, "expA")
        import ksi.cli as cli

        def _boom(*_a, **_k):
            raise OSError("simulated migration failure")

        monkeypatch.setattr(cli, "_migrate_legacy_flat_db_to_subdir", _boom)
        with pytest.raises(RuntimeError, match="refusing to resume against the shared"):
            cli._prepare_knowledge_db_path("", "expA")
        # The flat DB is left untouched (no silent reset, no shared mount).
        assert legacy.exists()

    def test_partial_migration_failure_resumes_against_subdir(self, monkeypatch, tmp_path):
        """If the main DB already moved into the subdir but a LATER step fails
        (e.g. the runtime-sibling move), the resume must point at the subdir —
        not the now-empty flat path, which would silently reset the run."""
        monkeypatch.setattr("ksi.layout.RUNTIME_KNOWLEDGE_DIR", tmp_path)
        legacy = tmp_path / "expA_knowledge.sqlite"
        legacy_runtime = tmp_path / "expA_runtime.sqlite"
        _make_wal_db(legacy, "expA-knowledge")
        _make_wal_db(legacy_runtime, "expA-runtime")
        import ksi.cli as cli

        real_move = cli._checkpoint_and_move_sqlite
        calls = {"n": 0}

        def _move_then_boom(src, dst):
            # First call (the main knowledge DB) succeeds; the second (the
            # runtime-audit sibling) raises, leaving the main DB in the subdir.
            calls["n"] += 1
            if calls["n"] == 1:
                return real_move(src, dst)
            raise OSError("simulated sibling-move failure")

        monkeypatch.setattr(cli, "_checkpoint_and_move_sqlite", _move_then_boom)
        result = Path(cli._prepare_knowledge_db_path("", "expA"))

        subdir_db = (tmp_path / "expA" / "expA_knowledge.sqlite").resolve()
        assert result == subdir_db  # resumed against the subdir, not the flat path
        assert _read_marker(result) == "expA-knowledge"  # data preserved
        assert not legacy.exists()  # the main DB really moved

    def test_prefers_subdir_when_both_exist(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ksi.layout.RUNTIME_KNOWLEDGE_DIR", tmp_path)
        subdir_db = tmp_path / "expA" / "expA_knowledge.sqlite"
        subdir_db.parent.mkdir(parents=True)
        subdir_db.write_text("")
        (tmp_path / "expA_knowledge.sqlite").write_text("")
        from ksi.cli import _prepare_knowledge_db_path

        result = Path(_prepare_knowledge_db_path("", "expA"))
        assert result == subdir_db.resolve()

    def test_explicit_path_is_never_rewritten(self, tmp_path):
        from ksi.cli import _resolve_knowledge_db_path

        explicit = tmp_path / "custom" / "mine_knowledge.sqlite"
        result = Path(_resolve_knowledge_db_path(str(explicit), "expA"))
        assert result == explicit.resolve()
