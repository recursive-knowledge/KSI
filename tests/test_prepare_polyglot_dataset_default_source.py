"""Pin the default polyglot subset source to the committed 50-task map (#1132).

The canonical polyglot pool is now the repo-committed
``benchmarks/polyglot/task_maps/polyglot_medium_50_seed0_ids.json`` rather than
a live HyperAgents@main URL, so ``prepare_polyglot_dataset.py`` materialises a
reproducible dataset without a network fetch. ``POLYGLOT_SUBSET_URL`` still
overrides the default for anyone who wants a live/remote subset.
"""

import importlib
import json
import subprocess

import pytest
from conftest import REPO_ROOT

from benchmarks.scripts.dataprep import prepare_polyglot_dataset as ppd

_COMMITTED_MAP = REPO_ROOT / "benchmarks" / "polyglot" / "task_maps" / "polyglot_medium_50_seed0_ids.json"
_COMMITTED_META = _COMMITTED_MAP.with_name(f"{_COMMITTED_MAP.stem}.meta.json")
_COMMITTED_SOURCE_COMMIT = json.loads(_COMMITTED_META.read_text(encoding="utf-8"))["source_commit"]


def test_default_subset_url_resolves_to_committed_map():
    module = importlib.reload(ppd)
    assert module.DEFAULT_SUBSET_URL == str(_COMMITTED_MAP)
    # It is a local path (not an http(s) URL), so _fetch_subset reads it directly.
    assert not module.DEFAULT_SUBSET_URL.startswith(("http://", "https://"))


def test_default_source_fetches_the_committed_50_ids():
    module = importlib.reload(ppd)
    fetched = module._fetch_subset(module.DEFAULT_SUBSET_URL)
    committed = json.loads(_COMMITTED_MAP.read_text(encoding="utf-8"))
    assert fetched == committed
    assert len(fetched) == 50


def test_default_source_commit_resolves_from_committed_meta(monkeypatch):
    monkeypatch.delenv("POLYGLOT_SOURCE_COMMIT", raising=False)
    module = importlib.reload(ppd)
    try:
        assert module._source_commit_for_subset(module.DEFAULT_SUBSET_URL) == _COMMITTED_SOURCE_COMMIT
        assert module._resolve_source_commit(module.DEFAULT_SUBSET_URL, None) == _COMMITTED_SOURCE_COMMIT
    finally:
        importlib.reload(module)


def test_polyglot_source_commit_env_override_wins(monkeypatch):
    monkeypatch.setenv("POLYGLOT_SOURCE_COMMIT", "custom-commit")
    module = importlib.reload(ppd)
    try:
        assert module._resolve_source_commit(module.DEFAULT_SUBSET_URL, None) == "custom-commit"
    finally:
        monkeypatch.delenv("POLYGLOT_SOURCE_COMMIT", raising=False)
        importlib.reload(module)


def test_remote_subset_requires_source_commit(tmp_path, monkeypatch):
    module = importlib.reload(ppd)
    monkeypatch.setattr(module, "_fetch_subset", lambda url: ["python__poker"])

    with pytest.raises(SystemExit, match="remote polyglot subset URLs must be paired with --source-commit"):
        module.prepare_dataset(
            subset_url="https://example.com/tasks.json",
            output=tmp_path / "polyglot.json",
            repo_cache="",
        )


def test_remote_subset_can_be_marked_mutable_for_exploration(tmp_path, monkeypatch):
    module = importlib.reload(ppd)
    _stub_extraction(module, tmp_path, monkeypatch)

    def fake_clone(dest, *, source_commit=None):
        assert source_commit is None
        dest.mkdir(parents=True)

    monkeypatch.setattr(module, "_clone_repo", fake_clone)

    out = tmp_path / "polyglot.json"
    tasks = module.prepare_dataset(
        subset_url="https://example.com/tasks.json",
        output=out,
        repo_cache=str(tmp_path / "cache"),
        allow_partial=True,
        allow_mutable_source=True,
    )

    assert len(tasks) == 1
    assert out.exists()


def test_clone_repo_fetches_the_pinned_commit(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(ppd.subprocess, "run", fake_run)

    ppd._clone_repo(tmp_path / "polyglot-benchmark", source_commit=_COMMITTED_SOURCE_COMMIT)

    assert calls == [
        ["git", "init", str(tmp_path / "polyglot-benchmark")],
        ["git", "-C", str(tmp_path / "polyglot-benchmark"), "remote", "add", "origin", ppd.BENCHMARK_REPO],
        ["git", "-C", str(tmp_path / "polyglot-benchmark"), "fetch", "--depth=1", "origin", _COMMITTED_SOURCE_COMMIT],
        ["git", "-C", str(tmp_path / "polyglot-benchmark"), "checkout", "--detach", _COMMITTED_SOURCE_COMMIT],
    ]


def test_cached_repo_must_match_the_pinned_commit(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        assert cmd == ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(cmd, 0, stdout="different-commit\n")

    monkeypatch.setattr(ppd.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=f"expected {_COMMITTED_SOURCE_COMMIT}"):
        ppd._ensure_repo_at_commit(tmp_path, _COMMITTED_SOURCE_COMMIT)


def test_polyglot_subset_url_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("POLYGLOT_SUBSET_URL", "https://example.com/custom.json")
    module = importlib.reload(ppd)
    try:
        assert module.DEFAULT_SUBSET_URL == "https://example.com/custom.json"
    finally:
        monkeypatch.delenv("POLYGLOT_SUBSET_URL", raising=False)
        importlib.reload(module)


def _stub_extraction(module, tmp_path, monkeypatch):
    """Two requested ids; only the python one extracts (go__wordy 'missing')."""
    monkeypatch.setattr(module, "_fetch_subset", lambda url: ["python__poker", "go__wordy"])
    monkeypatch.setattr(
        module,
        "_extract_exercise",
        lambda root, lang, ex: {"task_id": f"{lang}__{ex}", "language": lang} if lang == "python" else None,
    )


def test_prepare_dataset_fails_closed_on_partial_extraction(tmp_path, monkeypatch):
    # #1281: a partial materialization must NOT silently write a short dataset
    # that a sweep then runs under the full "medium-50" label.
    module = importlib.reload(ppd)
    _stub_extraction(module, tmp_path, monkeypatch)
    out = tmp_path / "polyglot.json"
    with pytest.raises(SystemExit):
        module.prepare_dataset(subset_url="x", output=out, repo_cache=str(tmp_path))
    assert not out.exists(), "short dataset must not be written when failing closed"


def test_prepare_dataset_fail_closed_checks_missing_dirs_before_extraction(tmp_path, monkeypatch):
    module = importlib.reload(ppd)
    monkeypatch.setattr(module, "_fetch_subset", lambda url: ["python__poker", "go__wordy"])
    (tmp_path / "python" / "exercises" / "practice" / "poker").mkdir(parents=True)

    def fail_extract(*args, **kwargs):
        raise AssertionError("_extract_exercise should not run when fail-closed preflight finds a missing dir")

    monkeypatch.setattr(module, "_extract_exercise", fail_extract)

    out = tmp_path / "polyglot.json"
    with pytest.raises(SystemExit):
        module.prepare_dataset(subset_url="x", output=out, repo_cache=str(tmp_path))
    assert not out.exists()


def test_prepare_dataset_allow_partial_writes_short_set(tmp_path, monkeypatch):
    module = importlib.reload(ppd)
    _stub_extraction(module, tmp_path, monkeypatch)
    out = tmp_path / "polyglot.json"
    tasks = module.prepare_dataset(subset_url="x", output=out, repo_cache=str(tmp_path), allow_partial=True)
    assert len(tasks) == 1
    assert out.exists()


def test_prepare_dataset_combines_source_pin_with_allow_partial(tmp_path, monkeypatch):
    module = importlib.reload(ppd)
    _stub_extraction(module, tmp_path, monkeypatch)
    clone_calls: list[tuple[str, str]] = []

    def fake_clone(dest, *, source_commit=None):
        clone_calls.append((str(dest), source_commit))
        dest.mkdir(parents=True)

    monkeypatch.setattr(module, "_clone_repo", fake_clone)

    out = tmp_path / "polyglot.json"
    tasks = module.prepare_dataset(
        subset_url="x",
        output=out,
        repo_cache=str(tmp_path / "cache"),
        source_commit="pinned-source",
        allow_partial=True,
    )

    assert len(tasks) == 1
    assert out.exists()
    assert clone_calls == [(str(tmp_path / "cache"), "pinned-source")]


def test_cli_accepts_source_commit_and_allow_partial_together(tmp_path, monkeypatch):
    module = importlib.reload(ppd)
    _stub_extraction(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_clone_repo", lambda dest, *, source_commit=None: dest.mkdir(parents=True))

    out = tmp_path / "polyglot.json"

    assert (
        module.main(
            [
                "--subset-url",
                "x",
                "--output",
                str(out),
                "--repo-cache",
                str(tmp_path / "cache"),
                "--source-commit",
                "pinned-source",
                "--allow-partial",
            ]
        )
        == 0
    )
    assert out.exists()
