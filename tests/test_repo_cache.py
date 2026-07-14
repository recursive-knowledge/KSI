"""Tests for src/ksi/tasks/repo_cache.py -- SWE-bench repo snapshot preparation."""

from __future__ import annotations

import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ksi.models import TaskSpec
from ksi.tasks.repo_cache import (
    _commit_reachable,
    _find_local_source_clone,
    _prepare_one_repo,
    _run,
    _warn_if_uninitialized_submodules,
    parse_test_files_from_before_cmd,
    prepare_swebench_repo_snapshots,
)

VALID_SHA = "a" * 40


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------
class TestRun:
    def test_successful_command(self, tmp_path):
        # A simple command that should succeed
        _run(["echo", "hello"])

    def test_failed_command_raises(self):
        with pytest.raises(subprocess.CalledProcessError):
            _run(["false"])

    def test_uses_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("KSI_GIT_TIMEOUT_SECONDS", "10")
        with patch("ksi.tasks.repo_cache.subprocess.Popen") as mock_popen:
            mock_popen.return_value.communicate.return_value = ("", "")
            mock_popen.return_value.returncode = 0
            _run(["git", "status"])
            mock_popen.assert_called_once()
            assert mock_popen.return_value.communicate.call_args.kwargs["timeout"] == 10

    def test_default_timeout(self, monkeypatch):
        monkeypatch.delenv("KSI_GIT_TIMEOUT_SECONDS", raising=False)
        with patch("ksi.tasks.repo_cache.subprocess.Popen") as mock_popen:
            mock_popen.return_value.communicate.return_value = ("", "")
            mock_popen.return_value.returncode = 0
            _run(["git", "status"])
            assert mock_popen.return_value.communicate.call_args.kwargs["timeout"] == 300


# ---------------------------------------------------------------------------
# _prepare_one_repo
# ---------------------------------------------------------------------------
class TestPrepareOneRepo:
    def test_rejects_invalid_repo_name(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid repo name"):
            _prepare_one_repo(target=tmp_path / "out", repo="has spaces/repo", base_commit=VALID_SHA)

    def test_rejects_repo_with_slashes(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid repo name"):
            _prepare_one_repo(target=tmp_path / "out", repo="a/b/c", base_commit=VALID_SHA)

    @patch("ksi.tasks.repo_cache._run")
    def test_clones_when_no_existing_repo(self, mock_run, tmp_path):
        target = tmp_path / "myrepo"
        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        # Should clone and then checkout
        calls = mock_run.call_args_list
        clone_call = calls[0]
        assert "clone" in clone_call.args[0]
        assert "https://github.com/owner/repo.git" in clone_call.args[0]

    @patch("ksi.tasks.repo_cache._run")
    def test_fetches_when_existing_repo(self, mock_run, tmp_path):
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        calls = mock_run.call_args_list
        fetch_call = calls[0]
        assert "fetch" in fetch_call.args[0]

    @patch("ksi.tasks.repo_cache._run")
    def test_offline_mode_skips_fetch(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.setenv("KSI_REPO_CACHE_OFFLINE", "true")
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        # Should NOT have a fetch call -- only checkout and clean
        for c in mock_run.call_args_list:
            assert "fetch" not in c.args[0]

    @patch("ksi.tasks.repo_cache._run")
    def test_offline_mode_no_clone_raises(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.setenv("KSI_REPO_CACHE_OFFLINE", "true")
        target = tmp_path / "new_repo"

        with pytest.raises(RuntimeError, match="offline mode"):
            _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

    @patch("ksi.tasks.repo_cache._run")
    def test_checkout_requires_base_commit(self, mock_run, tmp_path):
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        with pytest.raises(ValueError, match="base_commit must be a full 40-character hexadecimal commit SHA"):
            _prepare_one_repo(target=target, repo="owner/repo", base_commit="")

        mock_run.assert_not_called()

    @pytest.mark.parametrize(
        "base_commit",
        [
            " ",
            "HEAD",
            "main",
            "refs/heads/main",
            "abc123",
            "g" * 40,
        ],
    )
    @patch("ksi.tasks.repo_cache._run")
    def test_rejects_symbolic_or_malformed_base_commit_before_git(self, mock_run, tmp_path, base_commit):
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        with pytest.raises(ValueError, match="base_commit must be a full 40-character hexadecimal commit SHA"):
            _prepare_one_repo(target=target, repo="owner/repo", base_commit=base_commit)

        mock_run.assert_not_called()

    @patch("ksi.tasks.repo_cache._run")
    def test_initializes_submodules_after_checkout(self, mock_run, tmp_path):
        # Some SWE-bench Pro repos vendor an import-required dependency as a git
        # submodule (openlibrary exposes `infogami` via a top-level symlink into
        # vendor/infogami). A plain clone+checkout leaves the submodule empty, so
        # the symlink dangles and the repo's own conftest.py fails to import ->
        # the agent cannot run any repo test. Prep must init submodules.
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        cmds = [c.args[0] for c in mock_run.call_args_list]
        submodule_cmds = [c for c in cmds if "submodule" in c]
        assert submodule_cmds, "expected a `git submodule update --init` call during prep"
        assert submodule_cmds[0][-3:] == ["update", "--init", "--recursive"]
        assert "url.https://github.com/.insteadOf=git://github.com/" in submodule_cmds[0]
        # Task-controlled .gitmodules URLs must not open non-https transports
        # on the host (issue #1264): the fetch is pinned to https-only.
        assert "protocol.allow=never" in submodule_cmds[0]
        assert "protocol.https.allow=always" in submodule_cmds[0]
        # Must run AFTER the base_commit checkout (so .gitmodules is resolved at
        # the pinned tree) and after clean (so clean doesn't wipe it).
        checkout_idx = next(i for i, c in enumerate(cmds) if "checkout" in c)
        clean_idx = next(i for i, c in enumerate(cmds) if "clean" in c)
        submodule_idx = next(i for i, c in enumerate(cmds) if "submodule" in c)
        assert submodule_idx > checkout_idx
        assert submodule_idx > clean_idx

    @patch("ksi.tasks.repo_cache._run")
    def test_offline_mode_skips_submodule_init(self, mock_run, tmp_path, monkeypatch):
        # Submodule init fetches from the submodule's own remote; offline mode
        # can't reach the network, so it must be skipped (best-effort).
        monkeypatch.setenv("KSI_REPO_CACHE_OFFLINE", "true")
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        for c in mock_run.call_args_list:
            assert "submodule" not in c.args[0]

    @patch("ksi.tasks.repo_cache._run")
    def test_submodule_timeout_does_not_abort_prep(self, mock_run, tmp_path):
        # `_run` raises `TimeoutExpired` on a hung submodule fetch -- a sibling of
        # `CalledProcessError`, not a subclass. Submodule init is best-effort, so
        # a hung clone must NOT propagate and abort the whole batch's repo prep.
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        def _side_effect(cmd):
            if "submodule" in cmd:
                raise subprocess.TimeoutExpired(cmd, 300)

        mock_run.side_effect = _side_effect

        # Must not raise.
        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

    def test_offline_warns_on_uninitialized_submodules(self, tmp_path, caplog):
        # Offline mode can't populate submodules; rather than silently leave the
        # agent with the dangling-symlink import failure, prep must warn loudly
        # when the existing cache still has uninitialized submodules.
        target = tmp_path / "myrepo"
        target.mkdir()
        with patch("ksi.tasks.repo_cache.subprocess.run") as mock_sp:
            mock_sp.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="-cb117b0aa vendor/infogami\n 1a2b3c4dd vendor/other\n",
                stderr="",
            )
            with caplog.at_level("WARNING"):
                _warn_if_uninitialized_submodules(target, "owner/repo")
        warnings = [r.message for r in caplog.records if "uninitialized submodule" in r.message]
        assert warnings, "expected a warning naming the uninitialized submodule"
        assert "vendor/infogami" in warnings[0]
        # The initialized entry (no `-` prefix) must not be flagged.
        assert "vendor/other" not in warnings[0]

    @pytest.mark.skipif(shutil.which("git") is None, reason="requires a real git binary")
    def test_prepare_one_repo_shared_clone_populates_submodule(self, tmp_path, monkeypatch):
        # End-to-end, hermetic (no network): drive the real _prepare_one_repo
        # through the `git clone --shared` local-source path and assert the
        # submodule's content is actually populated -- guards the load-bearing
        # behavior (that `remote set-url origin` before `submodule update
        # --init --recursive` resolves the submodule and that --shared doesn't
        # break population) that the unit tests only mock.
        monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@example.com")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@example.com")
        # Allow file-protocol submodule fetches (blocked by default since
        # CVE-2022-39253) for every git call in this process, incl. the one
        # inside _prepare_one_repo.
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")

        def g(*args, cwd):
            subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)

        # Submodule origin holding the import-required content.
        sub = tmp_path / "sub_origin"
        sub.mkdir()
        g("init", "-q", cwd=sub)
        (sub / "f.txt").write_text("hi")
        g("add", "f.txt", cwd=sub)
        g("commit", "-q", "-m", "init", cwd=sub)

        # Upstream superproject that vendors the submodule at an absolute path.
        superp = tmp_path / "super_origin"
        superp.mkdir()
        g("init", "-q", cwd=superp)
        (superp / "readme").write_text("x")
        g("add", "readme", cwd=superp)
        g("commit", "-q", "-m", "base", cwd=superp)
        g("submodule", "add", str(sub), "vendor/sub", cwd=superp)
        g("commit", "-q", "-m", "add sub", cwd=superp)
        base = subprocess.run(
            ["git", "-C", str(superp), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Local source clone in the cache dir, tagged with the GitHub origin URL
        # so _find_local_source_clone matches it and clones it via --shared.
        cache = tmp_path / "cache"
        cache.mkdir()
        source = cache / "source_repo"
        g("clone", "-q", str(superp), str(source), cwd=cache)
        g("remote", "set-url", "origin", "https://github.com/owner/repo.git", cwd=source)

        target = cache / "owner__repo"
        _prepare_one_repo(target=target, repo="owner/repo", base_commit=base)

        assert (target / "vendor" / "sub" / "f.txt").read_text() == "hi"

    @pytest.mark.skipif(shutil.which("git") is None, reason="requires a real git binary")
    def test_prepare_one_repo_blocks_file_protocol_submodule(self, tmp_path, monkeypatch, caplog):
        # Issue #1264: `.gitmodules` URLs are task-controlled and the submodule
        # fetch runs ON THE HOST (outside container egress isolation). The
        # command-scoped `protocol.allow=never` + `protocol.https.allow=always`
        # config must block a `file://` submodule URL: prep completes (init is
        # best-effort), warns, and deinits the failed submodule instead of
        # fetching from the host filesystem. Unlike the populated-submodule
        # test above, no `protocol.file.allow=always` escape is set here.
        monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@example.com")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@example.com")

        def g(*args, cwd):
            subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)

        sub = tmp_path / "sub_origin"
        sub.mkdir()
        g("init", "-q", cwd=sub)
        (sub / "f.txt").write_text("hi")
        g("add", "f.txt", cwd=sub)
        g("commit", "-q", "-m", "init", cwd=sub)

        superp = tmp_path / "super_origin"
        superp.mkdir()
        g("init", "-q", cwd=superp)
        (superp / "readme").write_text("x")
        g("add", "readme", cwd=superp)
        g("commit", "-q", "-m", "base", cwd=superp)
        # Vendor the submodule with an explicit file:// URL, allowing the file
        # protocol for this SETUP command only.
        g("-c", "protocol.file.allow=always", "submodule", "add", f"file://{sub}", "vendor/sub", cwd=superp)
        g("commit", "-q", "-m", "add sub", cwd=superp)
        base = subprocess.run(
            ["git", "-C", str(superp), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        cache = tmp_path / "cache"
        cache.mkdir()
        source = cache / "source_repo"
        g("clone", "-q", str(superp), str(source), cwd=cache)
        g("remote", "set-url", "origin", "https://github.com/owner/repo.git", cwd=source)

        target = cache / "owner__repo"
        with caplog.at_level("WARNING"):
            _prepare_one_repo(target=target, repo="owner/repo", base_commit=base)

        # The submodule must NOT be populated from the host filesystem.
        assert not (target / "vendor" / "sub" / "f.txt").exists()
        messages = [r.getMessage() for r in caplog.records]
        assert any("Submodule init failed" in m for m in messages), messages
        # The failed submodule is deinited so the tree stays clean.
        status = subprocess.run(
            ["git", "-C", str(target), "submodule", "status"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert status.stdout.startswith("-"), status.stdout

    @patch("ksi.tasks.repo_cache._commit_reachable", return_value=True)
    @patch("ksi.tasks.repo_cache._run")
    def test_checkout_uses_detached_validated_commit(self, mock_run, mock_reach, tmp_path):
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA.upper())

        checkout_calls = [c.args[0] for c in mock_run.call_args_list if "checkout" in c.args[0]]
        assert checkout_calls == [["git", "-C", str(target), "checkout", "-f", "--detach", VALID_SHA]]

    @patch("ksi.tasks.repo_cache._commit_reachable", return_value=True)
    @patch("ksi.tasks.repo_cache._run")
    def test_skips_pr_ref_fetch_when_commit_reachable(self, mock_run, mock_reach, tmp_path):
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        # Should NOT have a PR-ref fetch call
        for c in mock_run.call_args_list:
            args = c.args[0]
            assert not any("refs/pull" in str(a) for a in args)

    @patch("ksi.tasks.repo_cache._commit_reachable", return_value=False)
    @patch("ksi.tasks.repo_cache._run")
    def test_falls_back_to_pr_refs_when_commit_unreachable(self, mock_run, mock_reach, tmp_path):
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        # Should have attempted a PR-ref fetch before checkout
        pr_ref_calls = [c for c in mock_run.call_args_list if any("refs/pull" in str(a) for a in c.args[0])]
        assert len(pr_ref_calls) == 1
        assert "fetch" in pr_ref_calls[0].args[0]

    @patch("ksi.tasks.repo_cache._commit_reachable", return_value=False)
    @patch("ksi.tasks.repo_cache._run")
    def test_pr_ref_fetch_failure_is_non_fatal(self, mock_run, mock_reach, tmp_path):
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        # Make PR-ref fetch fail, but normal calls succeed
        def side_effect(cmd):
            if any("refs/pull" in str(a) for a in cmd):
                raise subprocess.CalledProcessError(1, cmd, stderr="pr ref fetch denied")
            return None

        mock_run.side_effect = side_effect

        # Should not raise — fetch failure is logged and swallowed so checkout can still be attempted
        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        # Checkout must still have been attempted after the failed PR-ref fetch
        checkout_calls = [c for c in mock_run.call_args_list if "checkout" in c.args[0]]
        assert any(VALID_SHA in c.args[0] for c in checkout_calls)

    @patch("ksi.tasks.repo_cache._commit_reachable")
    @patch("ksi.tasks.repo_cache._run")
    def test_offline_mode_skips_pr_ref_fetch(self, mock_run, mock_reach, tmp_path, monkeypatch):
        monkeypatch.setenv("KSI_REPO_CACHE_OFFLINE", "true")
        target = tmp_path / "myrepo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        # Offline mode must not probe PR refs or call reachability (it would need network)
        mock_reach.assert_not_called()
        for c in mock_run.call_args_list:
            args = c.args[0]
            assert not any("refs/pull" in str(a) for a in args)


# ---------------------------------------------------------------------------
# _commit_reachable
# ---------------------------------------------------------------------------
class TestCommitReachable:
    def test_returns_true_on_success(self, tmp_path):
        with patch("ksi.tasks.repo_cache.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="commit\n")
            assert _commit_reachable(tmp_path, VALID_SHA) is True

    def test_returns_false_for_non_commit_object(self, tmp_path):
        with patch("ksi.tasks.repo_cache.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="tree\n")
            assert _commit_reachable(tmp_path, VALID_SHA) is False

    def test_returns_false_when_git_fails(self, tmp_path):
        with patch("ksi.tasks.repo_cache.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, ["git"])
            assert _commit_reachable(tmp_path, VALID_SHA) is False

    def test_returns_false_on_timeout(self, tmp_path):
        with patch("ksi.tasks.repo_cache.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git"], timeout=5)
            assert _commit_reachable(tmp_path, VALID_SHA) is False


# ---------------------------------------------------------------------------
# _find_local_source_clone
# ---------------------------------------------------------------------------
class TestFindLocalSourceClone:
    def test_returns_none_for_empty_dir(self, tmp_path):
        result = _find_local_source_clone(repos_cache_dir=tmp_path, remote="https://github.com/owner/repo.git")
        assert result is None

    def test_finds_matching_clone(self, tmp_path):
        clone_dir = tmp_path / "existing"
        (clone_dir / ".git").mkdir(parents=True)

        with patch("ksi.tasks.repo_cache.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.stdout = "https://github.com/owner/repo.git\n"
            mock_run.return_value = mock_proc

            result = _find_local_source_clone(
                repos_cache_dir=tmp_path,
                remote="https://github.com/owner/repo.git",
            )
            assert result == clone_dir

    def test_skips_non_matching_clone(self, tmp_path):
        clone_dir = tmp_path / "other"
        (clone_dir / ".git").mkdir(parents=True)

        with patch("ksi.tasks.repo_cache.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.stdout = "https://github.com/other/project.git\n"
            mock_run.return_value = mock_proc

            result = _find_local_source_clone(
                repos_cache_dir=tmp_path,
                remote="https://github.com/owner/repo.git",
            )
            assert result is None


# ---------------------------------------------------------------------------
# prepare_swebench_repo_snapshots
# ---------------------------------------------------------------------------
class TestPrepareSwebenchRepoSnapshots:
    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_skips_non_swebench_tasks(self, mock_prep, tmp_path):
        tasks = [
            TaskSpec(
                id="t1",
                repo="owner/repo",
                prompt="fix",
                metadata={"task_source": "arc"},
            ),
        ]
        prepare_swebench_repo_snapshots(tasks=tasks, repos_cache_dir=tmp_path)
        mock_prep.assert_not_called()

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_skips_tasks_without_repo(self, mock_prep, tmp_path):
        tasks = [
            TaskSpec(
                id="t1",
                repo="",
                prompt="fix",
                metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
            ),
        ]
        prepare_swebench_repo_snapshots(tasks=tasks, repos_cache_dir=tmp_path)
        mock_prep.assert_not_called()

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_prepares_swebench_task(self, mock_prep, tmp_path):
        task = TaskSpec(
            id="django__django-12345",
            repo="django/django",
            prompt="fix the bug",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA.upper()},
        )
        prepare_swebench_repo_snapshots(tasks=[task], repos_cache_dir=tmp_path)

        mock_prep.assert_called_once()
        call_kwargs = mock_prep.call_args.kwargs
        assert call_kwargs["repo"] == "django/django"
        assert call_kwargs["base_commit"] == VALID_SHA
        # Should set repo_path in metadata
        assert "repo_path" in task.metadata

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_rejects_missing_base_commit(self, mock_prep, tmp_path):
        task = TaskSpec(
            id="task-missing-base",
            repo="demo/repo",
            prompt="fix the bug",
            metadata={"task_source": "swebench_pro"},
        )
        with pytest.raises(ValueError, match="base_commit must be a full 40-character hexadecimal commit SHA"):
            prepare_swebench_repo_snapshots(tasks=[task], repos_cache_dir=tmp_path)

        mock_prep.assert_not_called()

    @pytest.mark.parametrize(
        "base_commit",
        [
            "",
            " ",
            "HEAD",
            "main",
            "refs/heads/main",
            "abc123",
            "g" * 40,
        ],
    )
    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_rejects_invalid_loaded_base_commit_before_prepare(self, mock_prep, tmp_path, base_commit):
        task = TaskSpec(
            id="task-invalid-base",
            repo="demo/repo",
            prompt="fix the bug",
            metadata={"task_source": "swebench_pro", "base_commit": base_commit},
        )

        with pytest.raises(ValueError, match="base_commit must be a full 40-character hexadecimal commit SHA"):
            prepare_swebench_repo_snapshots(tasks=[task], repos_cache_dir=tmp_path)

        mock_prep.assert_not_called()
        assert "repo_path" not in task.metadata

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_prepares_swebench_pro_task(self, mock_prep, tmp_path):
        task = TaskSpec(
            id="instance_demo__repo-123",
            repo="demo/repo",
            prompt="fix the bug",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
        )
        prepare_swebench_repo_snapshots(tasks=[task], repos_cache_dir=tmp_path)

        mock_prep.assert_called_once()
        assert "repo_path" in task.metadata

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_creates_cache_dir(self, mock_prep, tmp_path):
        cache_dir = tmp_path / "new_cache"
        tasks = [
            TaskSpec(
                id="t1",
                repo="owner/repo",
                prompt="fix",
                metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
            ),
        ]
        prepare_swebench_repo_snapshots(tasks=tasks, repos_cache_dir=cache_dir)
        assert cache_dir.exists()

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_propagates_before_repo_set_cmd_to_prepare(self, mock_prep, tmp_path):
        before_cmd = (
            "git reset --hard " + VALID_SHA + "\n"
            "git clean -fd\n"
            "git checkout " + ("b" * 40) + " -- path/to/test_a.go path/to/test_b.go"
        )
        task = TaskSpec(
            id="t1",
            repo="demo/repo",
            prompt="fix",
            metadata={
                "task_source": "swebench_pro",
                "base_commit": VALID_SHA,
                "before_repo_set_cmd": before_cmd,
            },
        )
        prepare_swebench_repo_snapshots(tasks=[task], repos_cache_dir=tmp_path)
        mock_prep.assert_called_once()
        call_kwargs = mock_prep.call_args.kwargs
        assert call_kwargs["before_repo_set_cmd"] == before_cmd

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_seed_test_files_default_false(self, mock_prep, tmp_path):
        """Upstream-strict default: seed_test_files=False unless explicitly opted in."""
        task = TaskSpec(
            id="t1",
            repo="demo/repo",
            prompt="fix",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
        )
        prepare_swebench_repo_snapshots(tasks=[task], repos_cache_dir=tmp_path)
        mock_prep.assert_called_once()
        assert mock_prep.call_args.kwargs["seed_test_files"] is False

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_seed_test_files_propagated_when_true(self, mock_prep, tmp_path):
        """DGM-equivalent opt-in: seed_test_files=True flows through to _prepare_one_repo."""
        task = TaskSpec(
            id="t1",
            repo="demo/repo",
            prompt="fix",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
        )
        prepare_swebench_repo_snapshots(
            tasks=[task],
            repos_cache_dir=tmp_path,
            seed_test_files=True,
        )
        mock_prep.assert_called_once()
        assert mock_prep.call_args.kwargs["seed_test_files"] is True


class TestPrepareOneRepoSeedingGate:
    """``_prepare_one_repo`` only seeds when seed_test_files=True."""

    @patch("ksi.tasks.repo_cache._seed_baseline_test_files")
    @patch("ksi.tasks.repo_cache._run")
    @patch("ksi.tasks.repo_cache._commit_reachable", return_value=True)
    def test_default_does_not_seed(self, _mock_reachable, _mock_run, mock_seed, tmp_path):
        from ksi.tasks.repo_cache import _prepare_one_repo

        target = tmp_path / "demo"
        target.mkdir()
        (target / ".git").mkdir()
        _prepare_one_repo(
            target=target,
            repo="demo/repo",
            base_commit=VALID_SHA,
            before_repo_set_cmd="git checkout " + ("b" * 40) + " -- t.go",
        )
        mock_seed.assert_not_called()

    @patch("ksi.tasks.repo_cache._seed_baseline_test_files")
    @patch("ksi.tasks.repo_cache._run")
    @patch("ksi.tasks.repo_cache._commit_reachable", return_value=True)
    def test_seed_true_invokes_seeder(self, _mock_reachable, _mock_run, mock_seed, tmp_path):
        from ksi.tasks.repo_cache import _prepare_one_repo

        target = tmp_path / "demo"
        target.mkdir()
        (target / ".git").mkdir()
        before_cmd = "git checkout " + ("b" * 40) + " -- t.go"
        _prepare_one_repo(
            target=target,
            repo="demo/repo",
            base_commit=VALID_SHA,
            before_repo_set_cmd=before_cmd,
            seed_test_files=True,
        )
        mock_seed.assert_called_once_with(target, before_cmd)


class TestParseTestFilesFromBeforeCmd:
    """Reliable extraction of grader-cherry-picked test files."""

    def test_extracts_commit_and_files(self):
        cmd = (
            "git reset --hard " + VALID_SHA + "\n"
            "git clean -fd\n"
            "git checkout " + ("b" * 40) + " -- path/test_a.go other/test_b.go"
        )
        commit, files = parse_test_files_from_before_cmd(cmd)
        assert commit == "b" * 40
        assert files == ["path/test_a.go", "other/test_b.go"]

    def test_empty_input_returns_empty(self):
        assert parse_test_files_from_before_cmd("") == ("", [])
        assert parse_test_files_from_before_cmd("   ") == ("", [])

    def test_rejects_non_sha_commit_token(self):
        cmd = "git checkout HEAD -- path/test.go"
        assert parse_test_files_from_before_cmd(cmd) == ("", [])

    def test_rejects_missing_separator(self):
        cmd = "git checkout " + VALID_SHA + " path/test.go"
        assert parse_test_files_from_before_cmd(cmd) == ("", [])

    def test_only_uses_last_line(self):
        # Earlier destructive lines must NOT contribute to the path list.
        cmd = (
            "git reset --hard " + ("c" * 40) + "\n"
            "git checkout " + ("d" * 40) + " -- ignored.go\n"
            "git checkout " + VALID_SHA + " -- only_this.go"
        )
        commit, files = parse_test_files_from_before_cmd(cmd)
        assert commit == VALID_SHA
        assert files == ["only_this.go"]


class TestSeedBaselineTestFiles:
    """End-to-end seeding via a real temporary git repo."""

    def _init_repo(self, repo_dir):
        repo_dir.mkdir(parents=True, exist_ok=True)
        for cmd in (
            ["git", "init", "-q", "--initial-branch=main", str(repo_dir)],
            ["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"],
            ["git", "-C", str(repo_dir), "config", "user.name", "test"],
        ):
            subprocess.run(cmd, check=True, capture_output=True)
        # parent commit (no test files yet)
        (repo_dir / "src.go").write_text("package main\n")
        subprocess.run(["git", "-C", str(repo_dir), "add", "src.go"], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-q", "-m", "parent"],
            check=True,
            capture_output=True,
        )
        parent = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # test_commit (adds the test file)
        (repo_dir / "src_test.go").write_text("package main\n// test\n")
        subprocess.run(["git", "-C", str(repo_dir), "add", "src_test.go"], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-q", "-m", "test"],
            check=True,
            capture_output=True,
        )
        test_commit = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # Reset back to parent so the cache prep mirrors the SWE-bench Pro
        # contract (start from the broken state).
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "-f", "--detach", parent],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "clean", "-fdx"],
            check=True,
            capture_output=True,
        )
        return parent, test_commit

    def test_cherry_pick_and_baseline_commit(self, tmp_path):
        from ksi.tasks.repo_cache import _seed_baseline_test_files

        repo = tmp_path / "repo"
        parent, test_commit = self._init_repo(repo)
        cmd = f"git checkout {test_commit} -- src_test.go"

        _seed_baseline_test_files(repo, cmd)

        # The test file is now committed on a baseline commit; agent's
        # ``git diff HEAD --`` would not see it.
        assert (repo / "src_test.go").exists()
        head = subprocess.run(
            ["git", "-C", str(repo), "log", "--oneline", "-1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "swarms-swebench-pro-baseline" in head
        diff = subprocess.run(
            ["git", "-C", str(repo), "diff", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert diff == ""

    def test_no_op_when_before_cmd_missing(self, tmp_path):
        from ksi.tasks.repo_cache import _seed_baseline_test_files

        repo = tmp_path / "repo"
        parent, _ = self._init_repo(repo)
        _seed_baseline_test_files(repo, "")
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert head == parent

    def test_skips_when_test_commit_unreachable(self, tmp_path):
        from ksi.tasks.repo_cache import _seed_baseline_test_files

        repo = tmp_path / "repo"
        parent, _ = self._init_repo(repo)
        unreachable = "f" * 40
        cmd = f"git checkout {unreachable} -- src_test.go"
        _seed_baseline_test_files(repo, cmd)
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert head == parent  # No baseline commit added.

    def test_rejects_path_traversal(self, tmp_path):
        from ksi.tasks.repo_cache import _seed_baseline_test_files

        repo = tmp_path / "repo"
        parent, test_commit = self._init_repo(repo)
        # ``..`` and absolute paths must be filtered out before reaching git.
        cmd = f"git checkout {test_commit} -- ../escape.go /etc/passwd"
        _seed_baseline_test_files(repo, cmd)
        # No file matched the safe filter; nothing should be staged or committed.
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert head == parent


# ---------------------------------------------------------------------------
# Narrow-fetch behaviour (leak-fix C) and metadata propagation (leak-fix A/B)
# ---------------------------------------------------------------------------
class TestNarrowFetch:
    """In upstream-strict mode the fetch must be keyed on base_commit only.

    A `--all --tags` fetch makes test_commit reachable inside the agent repo,
    exposing evaluation test files via ``git show <test_commit>:<path>``.
    The default fetch must request only the base_commit SHA.
    """

    @patch("ksi.tasks.repo_cache._commit_reachable", return_value=True)
    @patch("ksi.tasks.repo_cache._run")
    def test_existing_repo_uses_narrow_fetch_by_default(self, mock_run, _mock_reach, tmp_path, monkeypatch):
        monkeypatch.delenv("KSI_SWEBENCH_FETCH_FULL", raising=False)
        target = tmp_path / "repo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        fetch_calls = [c.args[0] for c in mock_run.call_args_list if "fetch" in c.args[0]]
        assert len(fetch_calls) >= 1, "expected at least one fetch call"
        first_fetch = fetch_calls[0]
        # Narrow fetch: should contain the SHA, NOT --all/--tags
        assert VALID_SHA in first_fetch
        assert "--all" not in first_fetch
        assert "--tags" not in first_fetch

    @patch("ksi.tasks.repo_cache._commit_reachable", return_value=True)
    @patch("ksi.tasks.repo_cache._run")
    def test_wide_fetch_when_env_override_set(self, mock_run, _mock_reach, tmp_path, monkeypatch):
        monkeypatch.setenv("KSI_SWEBENCH_FETCH_FULL", "1")
        target = tmp_path / "repo"
        (target / ".git").mkdir(parents=True)

        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        fetch_calls = [c.args[0] for c in mock_run.call_args_list if "fetch" in c.args[0]]
        assert any("--all" in fc for fc in fetch_calls), "expected wide --all fetch with override"

    @patch("ksi.tasks.repo_cache._commit_reachable", return_value=True)
    @patch("ksi.tasks.repo_cache._run")
    def test_narrow_fetch_failure_falls_back_to_wide(self, mock_run, _mock_reach, tmp_path, monkeypatch):
        """If the server rejects direct-SHA fetch, fall back to wide fetch gracefully."""
        monkeypatch.delenv("KSI_SWEBENCH_FETCH_FULL", raising=False)
        target = tmp_path / "repo"
        (target / ".git").mkdir(parents=True)

        call_count = [0]

        def side_effect(cmd):
            if "fetch" in cmd and VALID_SHA in cmd and "--all" not in cmd:
                # Simulate server rejecting SHA fetch
                raise subprocess.CalledProcessError(1, cmd, stderr="upload-pack: not allowed")
            return None

        mock_run.side_effect = side_effect

        # Should not raise — fallback wide fetch should succeed
        _prepare_one_repo(target=target, repo="owner/repo", base_commit=VALID_SHA)

        fetch_calls = [c.args[0] for c in mock_run.call_args_list if "fetch" in c.args[0]]
        # The fallback wide fetch must have been issued
        assert any("--all" in fc for fc in fetch_calls), "expected fallback wide fetch"


class TestSeedTestsMetadataPropagation:
    """prepare_swebench_repo_snapshots must stamp swebench_pro_seed_tests into
    task.metadata so downstream consumers (prompt builder, workspace_task_files)
    can gate on the flag without a separate config channel."""

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_seed_false_stamped_in_metadata_by_default(self, mock_prep, tmp_path):
        task = TaskSpec(
            id="t1",
            repo="demo/repo",
            prompt="fix",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
        )
        prepare_swebench_repo_snapshots(tasks=[task], repos_cache_dir=tmp_path)
        assert task.metadata.get("swebench_pro_seed_tests") is False

    @patch("ksi.tasks.repo_cache._prepare_one_repo")
    def test_seed_true_stamped_in_metadata_when_opted_in(self, mock_prep, tmp_path):
        task = TaskSpec(
            id="t1",
            repo="demo/repo",
            prompt="fix",
            metadata={"task_source": "swebench_pro", "base_commit": VALID_SHA},
        )
        prepare_swebench_repo_snapshots(tasks=[task], repos_cache_dir=tmp_path, seed_test_files=True)
        assert task.metadata.get("swebench_pro_seed_tests") is True
