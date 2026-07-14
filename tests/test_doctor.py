"""Smoke tests for `kcsi-doctor` — it must run and report without crashing,
regardless of whether Docker/Node/keys are present on the machine."""

from __future__ import annotations

import kcsi.doctor as doctor


def test_parse_node_version_accepts_node_cli_output() -> None:
    assert doctor._parse_node_version("v22.16.0") == (22, 16, 0)
    assert doctor._parse_node_version("22.16.0") == (22, 16, 0)
    assert doctor._parse_node_version("not-a-version") is None


def test_node_version_support_matches_runtime_package_engines() -> None:
    assert doctor._node_version_is_supported((22, 16, 0))
    assert doctor._node_version_is_supported((22, 20, 0))
    assert not doctor._node_version_is_supported((22, 15, 9))
    assert not doctor._node_version_is_supported((23, 0, 0))


def test_check_node_rejects_old_version(monkeypatch, capsys) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(doctor, "_run", lambda cmd, timeout=15.0: (0, "v20.11.1"))

    r = doctor.Report()
    doctor._check_node(r)

    assert r.hard_failures == 1
    assert doctor.NODE_ENGINE_RANGE in capsys.readouterr().out


def test_doctor_main_returns_int(capsys) -> None:
    rc = doctor.doctor_main([])
    out = capsys.readouterr().out
    assert isinstance(rc, int)
    # Section headers are always printed.
    for header in ("Core", "Runtime", "Providers", "Optional"):
        assert header in out
    assert "Python" in out


def test_report_counts_hard_failures() -> None:
    r = doctor.Report()
    assert r.hard_failures == 0
    r.ok("fine")
    r.warn("meh")
    assert r.hard_failures == 0
    r.fail("broken", fix="do the thing")
    assert r.hard_failures == 1


def test_check_vector_status_missing_db(tmp_path, capsys) -> None:
    r = doctor.Report()
    doctor._check_vector_status(r, str(tmp_path / "nope.sqlite"))
    assert r.hard_failures == 1
    assert "not found" in capsys.readouterr().out


def test_check_vector_status_reads_knowledge_db(tmp_path, capsys) -> None:
    """Latest row per phase is reported; degraded phases warn, enabled phases pass."""
    from kcsi.memory.knowledge_store import KnowledgeStore

    db_path = str(tmp_path / "exp_knowledge.sqlite")
    store = KnowledgeStore(db_path, default_experiment="exp")
    try:
        store.record_vector_status(phase="init", status="degraded", detail="sqlite-vec unavailable")
        store.record_vector_status(phase="init", status="enabled", detail="knowledge_vec ready")
        store.record_vector_status(
            phase="embedder",
            status="degraded",
            detail="no HF_TOKEN",
            embedding_count=3,
            skipped_count=2,
        )
    finally:
        store.close()

    r = doctor.Report()
    doctor._check_vector_status(r, db_path)
    out = capsys.readouterr().out
    assert r.hard_failures == 0
    # Latest init row wins (enabled), embedder row is degraded -> warn.
    assert "vector_status[init]" in out
    assert "knowledge_vec ready" in out
    assert "sqlite-vec unavailable" not in out
    assert "vector_status[embedder]" in out
    assert "no HF_TOKEN" in out
    assert "3 embedded, 2 skipped" in out


def test_check_vector_status_not_a_knowledge_db(tmp_path, capsys) -> None:
    """A SQLite file without the vector_status table warns instead of crashing."""
    import sqlite3

    db_path = tmp_path / "other.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.commit()
    conn.close()

    r = doctor.Report()
    doctor._check_vector_status(r, str(db_path))
    assert r.hard_failures == 0
    assert "vector_status table missing" in capsys.readouterr().out
