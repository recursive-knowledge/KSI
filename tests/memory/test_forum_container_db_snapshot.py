"""Regression tests for the forum-container knowledge-DB snapshot path.

Root cause being pinned (macOS + Docker Desktop): forum containers used to
bind-mount the LIVE knowledge DB and its ``-wal``/``-shm`` sidecars, and the
in-container MCP server opened them with ``read_only=True``. In WAL mode even a
read-only SQLite connection WRITES to the ``-shm`` wal-index (read-marks,
recovery rebuilds) and takes POSIX locks — but a VirtioFS bind mount shares
neither the host's advisory locks nor a coherent mmap of the ``-shm``. The
container-side "reader" therefore silently fought the host writer over the
wal-index, misdirecting WAL appends/checkpoint backfills into wrong page
offsets: ``sqlite3.DatabaseError: database disk image is malformed`` on the
authoritative store (or a host SIGBUS inside ``walIndexAppend`` /
``walIndexReadHdr``), scaling with workstream count.

The fix: the live WAL trio never crosses the container boundary. Forum
containers mount a standalone point-in-time snapshot produced by
``_store_common.snapshot_db_for_container_mount`` (``VACUUM INTO`` on its own
read-only connection), and the MCP server opens it ``read_only=True,
immutable=True`` — no locks, no wal-index, no shared state.

See tests/runtime/test_container_host.py::TestForumKnowledgeDbSnapshot for the
payload/mount-contract side of the same regression.
"""

from __future__ import annotations

import sqlite3

import pytest

from ksi.memory._store_common import snapshot_db_for_container_mount
from ksi.memory.knowledge_store import KnowledgeStore


def _wal_trio(db_path):
    return [db_path, db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")]


def test_snapshot_is_standalone_and_complete_while_writer_is_live(tmp_path):
    """Snapshot cut while the writer holds the DB open must be a single
    self-contained file carrying every committed row — including rows that
    still live only in the source's ``-wal``.
    """
    db = tmp_path / "exp_knowledge.sqlite"
    store = KnowledgeStore(str(db), default_experiment="exp")
    try:
        for i in range(5):
            store.record_post(
                task_id="t1",
                agent_id="agent-0",
                generation=1,
                text=f"post {i}",
                source_phase="per_task_forum",
            )
        # Live WAL trio exists: the writer connection is open and rows are in
        # the -wal (autocheckpoint threshold not reached).
        for p in _wal_trio(db):
            assert p.exists(), f"precondition: live {p.name} missing"

        snap = tmp_path / "knowledge_ro_snapshot_agent-0_deadbeef.sqlite"
        snapshot_db_for_container_mount(db, snap)

        # Standalone: single rollback-journal file, no sidecars to share.
        assert snap.exists()
        assert not snap.with_name(snap.name + "-wal").exists()
        assert not snap.with_name(snap.name + "-shm").exists()
        header = snap.read_bytes()[:100]
        assert header[18] == 1 and header[19] == 1, "snapshot must not be in WAL mode"
    finally:
        store.close()

    # Complete: all committed rows visible through the store's own read path.
    ro = KnowledgeStore(str(snap), read_only=True, immutable=True, default_experiment="exp")
    try:
        page = ro.query_task("t1", entry_types=["post"], experiment="exp")
        assert len(page["discussion"]) == 5
    finally:
        ro.close()


def test_snapshot_overwrites_stale_dest(tmp_path):
    """A stale file at the destination (crashed prior run) must not survive:
    ``VACUUM INTO`` refuses to overwrite, so the helper unlinks first.
    """
    db = tmp_path / "exp_knowledge.sqlite"
    store = KnowledgeStore(str(db), default_experiment="exp")
    try:
        store.record_insight(task_id="t1", agent_id="agent-0", generation=1, text="x")
        snap = tmp_path / "snap.sqlite"
        snap.write_bytes(b"stale garbage from a crashed run")
        snapshot_db_for_container_mount(db, snap)
        conn = sqlite3.connect(f"file:{snap}?mode=ro&immutable=1", uri=True)
        try:
            assert conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] == 1
        finally:
            conn.close()
    finally:
        store.close()


def test_snapshot_failure_leaves_no_partial_file(tmp_path):
    """A failed snapshot (source is not a database) must raise AND leave no
    dest file behind for a container to mount.
    """
    not_a_db = tmp_path / "exp_knowledge.sqlite"
    not_a_db.write_bytes(b"this is not a sqlite database")
    snap = tmp_path / "snap.sqlite"
    with pytest.raises(sqlite3.Error):
        snapshot_db_for_container_mount(not_a_db, snap)
    assert not snap.exists(), "half-written snapshot must be cleaned up"


def test_immutable_requires_read_only(tmp_path):
    """``immutable=1`` promises SQLite the file never changes; on a writable
    store that would be a lie (stale/torn reads at best). Fail loudly.
    """
    with pytest.raises(ValueError, match="immutable=True requires read_only=True"):
        KnowledgeStore(str(tmp_path / "k.sqlite"), immutable=True)


def test_immutable_open_takes_no_locks_and_creates_no_sidecars(tmp_path):
    """The immutable read path must not create ``-wal``/``-shm`` (there is no
    shared WAL state to fight over — the property the container mount relies
    on).
    """
    db = tmp_path / "exp_knowledge.sqlite"
    store = KnowledgeStore(str(db), default_experiment="exp")
    try:
        store.record_post(task_id="t1", agent_id="agent-0", generation=1, text="p")
        snap = tmp_path / "snap.sqlite"
        snapshot_db_for_container_mount(db, snap)
    finally:
        store.close()

    ro = KnowledgeStore(str(snap), read_only=True, immutable=True, default_experiment="exp")
    try:
        ro.query_task("t1", experiment="exp")
        assert not snap.with_name(snap.name + "-wal").exists()
        assert not snap.with_name(snap.name + "-shm").exists()
    finally:
        ro.close()
