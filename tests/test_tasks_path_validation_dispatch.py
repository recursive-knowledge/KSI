"""Seam 4 (issue #741): per-source --tasks-path validation dispatch.

The CLI-level tests pin the exact ``parser.error`` messages each task source
emits for a bad ``--tasks-path`` (green before and after the refactor that moves
the validation off the cli ``path_kind`` if/elif chain onto a
``TaskSourceSpec.validate_tasks_path`` hook). The hook-level tests exercise the
new spec API and fail until the field + dispatch exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kcsi import cli as cli_module
from kcsi.tasks import get_spec


def _run_cli(monkeypatch, tmp_path, *, source, evaluator, tasks_path):
    monkeypatch.setattr(cli_module, "_ensure_trace_dir", lambda _experiment_name: None)
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main(
            [
                "--task-source",
                source,
                "--tasks-path",
                str(tasks_path),
                "--evaluator",
                evaluator,
                "--knowledge-db-path",
                str(tmp_path / "knowledge.sqlite"),
                "--runtime-db-path",
                str(tmp_path / "runtime.sqlite"),
            ]
        )
    assert excinfo.value.code == 2


# ── CLI-level characterization (byte-identical messages, green before/after) ──


def test_arc_missing_path_message(tmp_path, capsys, monkeypatch):
    missing = tmp_path / "nope"
    _run_cli(monkeypatch, tmp_path, source="arc", evaluator="arc_session", tasks_path=missing)
    err = capsys.readouterr().err
    assert f"--tasks-path for --task-source arc must exist (json file or directory): {missing}" in err


def test_arc_non_json_file_message(tmp_path, capsys, monkeypatch):
    f = tmp_path / "tasks.txt"
    f.write_text("[]")
    _run_cli(monkeypatch, tmp_path, source="arc", evaluator="arc_session", tasks_path=f)
    err = capsys.readouterr().err
    assert f"--tasks-path for --task-source arc file input must be .json: {f}" in err


def test_polyglot_directory_message(tmp_path, capsys, monkeypatch):
    d = tmp_path / "poly"
    d.mkdir()
    _run_cli(monkeypatch, tmp_path, source="polyglot", evaluator="polyglot_harness", tasks_path=d)
    err = capsys.readouterr().err
    assert f"--tasks-path for --task-source polyglot must be an existing .json file: {d}" in err


def test_swebench_pro_missing_message(tmp_path, capsys, monkeypatch):
    missing = tmp_path / "nope"
    _run_cli(monkeypatch, tmp_path, source="swebench_pro", evaluator="swebench_pro", tasks_path=missing)
    err = capsys.readouterr().err
    assert f"--tasks-path for --task-source swebench_pro must exist: {missing}" in err


def test_swebench_pro_bad_suffix_message(tmp_path, capsys, monkeypatch):
    f = tmp_path / "tasks.json"
    f.write_text("[]")
    _run_cli(monkeypatch, tmp_path, source="swebench_pro", evaluator="swebench_pro", tasks_path=f)
    err = capsys.readouterr().err
    assert f"--tasks-path for --task-source swebench_pro must be .parquet, .csv, or .jsonl: {f}" in err


def test_terminal_bench_2_message(tmp_path, capsys, monkeypatch):
    d = tmp_path / "tb2"
    d.mkdir()
    _run_cli(monkeypatch, tmp_path, source="terminal_bench_2", evaluator="terminal_bench_2", tasks_path=d)
    err = capsys.readouterr().err
    assert f"--tasks-path for --task-source terminal_bench_2 must be an existing .json task map: {d}" in err


# ── New capability: validation lives on the spec hook ─────────────────────────


def test_validate_tasks_path_hook_returns_messages(tmp_path):
    arc = get_spec("arc").validate_tasks_path
    missing = tmp_path / "nope"
    assert arc(missing, evals_path=None) == (
        f"--tasks-path for --task-source arc must exist (json file or directory): {missing}"
    )
    good = tmp_path / "t.json"
    good.write_text("[]")
    assert arc(good, evals_path=None) is None


def test_validate_tasks_path_hook_swebench_evals(tmp_path):
    spec = get_spec("swebench_pro").validate_tasks_path
    f = tmp_path / "t.parquet"
    f.write_text("x")
    bad_evals = tmp_path / "e.csv"
    bad_evals.write_text("x")
    assert spec(f, evals_path=bad_evals) == f"--evals-path must be a parquet file (.parquet): {bad_evals}"
    assert spec(f, evals_path=None) is None


def test_custom_source_validate_hook_is_dispatched(tmp_path):
    """A registered source's ``validate_tasks_path`` hook drives CLI path
    validation with no ``path_kind`` edit in the cli."""
    from kcsi.tasks import registry

    def _validate(tasks_path: Path, *, evals_path):
        return None if tasks_path.suffix == ".weird" else f"need .weird, got {tasks_path}"

    spec = registry.TaskSourceSpec(name="pv_fake_src", validate_tasks_path=_validate)
    registry.register_task_source(spec)
    try:
        resolved = registry.get_spec("pv_fake_src")
        assert resolved.validate_tasks_path(Path("x.txt"), evals_path=None) == "need .weird, got x.txt"
        assert resolved.validate_tasks_path(Path("x.weird"), evals_path=None) is None
    finally:
        for key in spec.all_names():
            registry.REGISTRY.pop(key, None)
