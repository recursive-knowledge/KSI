from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

from ..models import TaskSpec
from .registry import resolve_source

log = logging.getLogger(__name__)

_SAFE_REPO_CACHE_TASK_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_FULL_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_BASELINE_COMMIT_MESSAGE = "swarms-swebench-pro-baseline"


def validate_repo_cache_task_id(task_id: str) -> str:
    """Validate a task ID before using it as one repo-cache path segment."""
    if not isinstance(task_id, str):
        raise ValueError("Repo-cache task ID must be a string")
    value = task_id
    if not value or value in {".", ".."} or not _SAFE_REPO_CACHE_TASK_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid repo-cache task ID: {task_id!r}")
    return value


def validate_swebench_base_commit(
    base_commit: object,
    *,
    task_id: str | None = None,
    repo: str | None = None,
) -> str:
    value = str(base_commit or "").strip()
    if _FULL_COMMIT_SHA_RE.fullmatch(value):
        return value.lower()
    subject = f" task {task_id}" if task_id else ""
    repo_part = f" for {repo}" if repo else ""
    raise ValueError(
        f"SWE-bench Pro{subject}{repo_part} base_commit must be a full 40-character hexadecimal commit SHA"
    )


def _repo_cache_target(*, repos_cache_dir: Path, task_id: str) -> Path:
    cache_root = repos_cache_dir.resolve()
    safe_task_id = validate_repo_cache_task_id(task_id)
    target = cache_root / safe_task_id
    if target.exists() and not target.resolve().is_relative_to(cache_root):
        raise ValueError(f"Repo-cache target escapes cache directory: {target}")
    return target


def prepare_swebench_repo_snapshots(
    *,
    tasks: list[TaskSpec],
    repos_cache_dir: Path,
    seed_test_files: bool = False,
) -> None:
    """Materialize per-instance repo snapshots for SWE-bench tasks.

    For each task with metadata:
      - repo: "owner/name"
      - base_commit: commit hash
    this prepares:
      repos_cache_dir/<instance_id>/
    and sets task.metadata["repo_path"] to that directory.

    ``seed_test_files`` controls whether the grader's test files are
    cherry-picked into the agent's repo as a baseline commit on top of
    ``base_commit`` (see ``_seed_baseline_test_files``). Default is
    ``False`` (upstream-strict): the agent works against ``base_commit``
    alone and must infer test APIs from the issue/problem statement,
    matching the upstream SWE-bench Pro reference protocol where
    ``before_repo_set_cmd`` runs only inside the grader, after the
    agent's patch is applied. Set ``True`` for DGM-equivalent runs
    where seeded tests are required for cross-runner comparability.
    """
    seen_task_ids: set[str] = set()
    snapshots: list[tuple[TaskSpec, Path, str, str, str]] = []
    for task in tasks:
        meta = task.metadata or {}
        repo_spec = resolve_source(meta.get("task_source"))
        if repo_spec is None or not repo_spec.uses_repo_snapshots:
            continue
        repo = str(task.repo or "").strip()
        if not repo:
            continue
        base_commit = validate_swebench_base_commit(
            meta.get("base_commit"),
            task_id=task.id,
            repo=repo,
        )
        safe_task_id = validate_repo_cache_task_id(task.id)
        if safe_task_id in seen_task_ids:
            raise ValueError(f"Duplicate SWE-bench Pro task ID: {safe_task_id}")
        seen_task_ids.add(safe_task_id)
        target = _repo_cache_target(repos_cache_dir=repos_cache_dir, task_id=safe_task_id)
        before_repo_set_cmd = str(meta.get("before_repo_set_cmd") or "")
        snapshots.append((task, target, repo, base_commit, before_repo_set_cmd))

    if snapshots:
        repos_cache_dir.mkdir(parents=True, exist_ok=True)
    for task, target, repo, base_commit, before_repo_set_cmd in snapshots:
        _prepare_one_repo(
            target=target,
            repo=repo,
            base_commit=base_commit,
            before_repo_set_cmd=before_repo_set_cmd,
            seed_test_files=seed_test_files,
        )
        task.metadata["repo_path"] = str(target.resolve())
        # Propagate the flag into task metadata so that downstream consumers
        # (prompt builder, workspace_task_files) can gate evaluation-signal
        # leakage without needing a separate configuration channel.
        task.metadata["swebench_pro_seed_tests"] = seed_test_files


def parse_test_files_from_before_cmd(before_repo_set_cmd: str) -> tuple[str, list[str]]:
    """Extract the cherry-pick (commit, files) from the last line of before_repo_set_cmd.

    SWE-bench Pro entries put the test-file checkout as the LAST line, e.g.:
        git checkout <test_commit> -- path/to/test_a path/to/test_b
    The upstream evaluator (`swe_bench_pro_eval.py:create_entryscript`) only
    runs that last line — earlier lines are destructive resets we must not
    mirror. Returns ("", []) if the command does not match the expected shape.
    """
    text = (before_repo_set_cmd or "").strip()
    if not text:
        return "", []
    last_line = text.splitlines()[-1].strip()
    if not last_line:
        return "", []
    try:
        tokens = shlex.split(last_line)
    except ValueError:
        return "", []
    # Expected shape: git checkout <commit> -- <file>...
    if len(tokens) < 5 or tokens[0] != "git" or tokens[1] != "checkout" or "--" not in tokens:
        return "", []
    sep_idx = tokens.index("--")
    commit_token = tokens[sep_idx - 1]
    if not _FULL_COMMIT_SHA_RE.fullmatch(commit_token):
        return "", []
    files = [tok for tok in tokens[sep_idx + 1 :] if tok and not tok.startswith("-")]
    return commit_token.lower(), files


def _seed_baseline_test_files(target: Path, before_repo_set_cmd: str) -> None:
    """Cherry-pick the grader's test files into the agent's repo and commit them.

    Without this, the agent works against a tree that lacks the upstream test
    files; it has to guess the test API and frequently produces a structurally-
    incompatible source patch. After seeding, `git diff HEAD --` excludes the
    test files (already committed to baseline), so the agent's submitted patch
    contains only its source-only changes — which the grader applies on the
    raw base_commit before re-cherry-picking the same test files. Net result
    matches DGM's harness: agent + upstream tests are co-present at runtime
    AND the agent's patch is portable to a non-seeded base_commit checkout.
    """
    test_commit, files = parse_test_files_from_before_cmd(before_repo_set_cmd)
    if not test_commit or not files:
        return
    safe_files: list[str] = []
    for rel in files:
        # Defensive path validation: reject ``..`` traversal and absolute
        # paths so a malformed dataset row cannot make the cherry-pick
        # touch arbitrary host paths via git's pathspec parser.
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            continue
        safe_files.append(rel)
    if not safe_files:
        return
    if not _commit_reachable(target, test_commit):
        try:
            _run(["git", "-C", str(target), "fetch", "--quiet", "--no-tags", "origin", test_commit])
        except subprocess.CalledProcessError:
            pass
    if not _commit_reachable(target, test_commit):
        log.warning(
            "Skipping baseline cherry-pick for %s: test_commit %s not reachable",
            target.name,
            test_commit[:12],
        )
        return
    _run(["git", "-C", str(target), "checkout", test_commit, "--", *safe_files])
    _run(["git", "-C", str(target), "add", "--", *safe_files])
    # Commit the cherry-picked test files. ``--allow-empty`` covers the rare
    # case where the test file already matches the parent (no-op cherry).
    _run(
        [
            "git",
            "-c",
            "user.name=kcsi",
            "-c",
            "user.email=kcsi@example.com",
            "-C",
            str(target),
            "commit",
            "--allow-empty",
            "-m",
            _BASELINE_COMMIT_MESSAGE,
        ]
    )


def _prepare_one_repo(
    *,
    target: Path,
    repo: str,
    base_commit: str,
    before_repo_set_cmd: str = "",
    seed_test_files: bool = False,
) -> None:
    if not re.match(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$", repo):
        raise ValueError(f"Invalid repo name: {repo}")
    base_commit = validate_swebench_base_commit(base_commit, repo=repo)
    remote = f"https://github.com/{repo}.git"
    offline = os.environ.get("KCSI_REPO_CACHE_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
    if (target / ".git").exists():
        if not offline:
            # Narrow *update* fetch: only pull objects reachable from base_commit,
            # to keep the cache lean and avoid widening it with extra tags/commits
            # (a smaller object graph also makes the per-task `git gc` cheaper).
            #
            # NOTE: this is NOT the leak-isolation gate. The initial `git clone`
            # below populates refs/remotes/origin/* with the full upstream history
            # (incl. the future fix commit) at clone time, which this update-only
            # narrow fetch never unwinds. The actual guarantee that the solver
            # cannot read the graded answer is sanitizeRepoHistory() in
            # runtime_runner/src/workspace.ts, which strips remotes/refs and prunes
            # future objects from the disposable per-task workspace copy.
            # So a wide fetch here (KCSI_SWEBENCH_FETCH_FULL=1) is still safe.
            if os.environ.get("KCSI_SWEBENCH_FETCH_FULL", "").strip().lower() in {"1", "true", "yes"}:
                # Legacy wide fetch — tags may clobber.
                _run(["git", "-C", str(target), "fetch", "--all", "--tags", "--prune", "--force"])
            else:
                # Narrow fetch: only pull the specific base_commit.  If the server
                # does not support direct SHA fetches
                # (uploadpack.allowReachableSHA1InWant) the fetch will fail; in
                # that case we fall through to the PR-ref fallback below.
                try:
                    _run(["git", "-C", str(target), "fetch", "origin", base_commit])
                except subprocess.CalledProcessError:
                    # Server doesn't support sha fetch — fall back to wide fetch.
                    # This is the legacy path; log a warning so operators can see it.
                    log.warning(
                        "Narrow fetch of %s failed for %s; falling back to full fetch "
                        "(set KCSI_SWEBENCH_FETCH_FULL=1 to silence this warning)",
                        base_commit[:12],
                        repo,
                    )
                    _run(["git", "-C", str(target), "fetch", "--all", "--tags", "--prune", "--force"])
    else:
        if offline:
            raise RuntimeError(f"Repo cache offline mode is enabled, but no local clone exists for: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        local_source = _find_local_source_clone(repos_cache_dir=target.parent, remote=remote)
        if local_source is not None:
            _run(["git", "clone", "--shared", "--no-checkout", str(local_source), str(target)])
            _run(["git", "-C", str(target), "remote", "set-url", "origin", remote])
        else:
            _run(["git", "clone", "--no-checkout", remote, str(target)])

    # Some SWE-bench Pro tasks have base_commits that live only on PR refs
    # (closed or force-pushed PRs). Default `git fetch` does not retrieve
    # refs/pull/*, so fall back to a PR-ref fetch before giving up.
    if not offline and not _commit_reachable(target, base_commit):
        log.warning(
            "base_commit %s not reachable via branches/tags for %s; fetching PR refs",
            base_commit[:12],
            repo,
        )
        try:
            _run(
                [
                    "git",
                    "-C",
                    str(target),
                    "fetch",
                    "origin",
                    "+refs/pull/*/head:refs/remotes/origin/pr/*",
                ]
            )
        except subprocess.CalledProcessError as exc:
            log.warning(
                "PR-ref fetch failed for %s (continuing to checkout): %s",
                repo,
                exc.stderr,
            )
    _run(["git", "-C", str(target), "checkout", "-f", "--detach", base_commit])
    _run(["git", "-C", str(target), "clean", "-fdx"])
    # Populate git submodules pinned by this commit. Some SWE-bench Pro repos
    # vendor an import-required dependency as a submodule (openlibrary exposes
    # `infogami` via a top-level `infogami -> vendor/infogami/infogami`
    # symlink). A plain clone + checkout leaves the submodule empty, so the
    # symlink dangles and the repo's own conftest.py (`from infogami import
    # ...`) fails to import -- EVERY pytest the agent runs then dies at
    # collection, and the agent silently ships pass-to-pass regressions it
    # could not self-verify. The grader runs in the official image's populated
    # /app, so the asymmetry is invisible to the score. Runs AFTER checkout (so
    # .gitmodules is resolved at the pinned tree) and AFTER clean (so clean
    # doesn't wipe it). Best-effort: a submodule-server hiccup must not abort
    # the whole campaign, and repos without submodules make this a fast no-op.
    # `_run` enforces a git timeout and raises `TimeoutExpired` (a sibling of
    # `CalledProcessError`, not a subclass) on a hung fetch, plus `OSError` if
    # git is missing -- catch all three, else a single slow submodule clone
    # aborts the whole batch's repo prep. `OSError` has no `.stderr`.
    # In offline mode we cannot fetch submodules; instead of silently leaving
    # them empty (the exact failure this guards against) we warn loudly if the
    # existing cache still has uninitialized submodules, so the operator knows
    # to re-prep online.
    if not offline:
        # Some pinned .gitmodules still use the unauthenticated ``git://``
        # protocol, which GitHub sunset in 2022 -> the clone hangs until the git
        # timeout (openlibrary's `vendor/infogami`). The setting is command
        # scoped because fresh submodule clone subprocesses do not inherit the
        # parent repository's local config.
        # ``.gitmodules`` URLs are task-controlled and this fetch runs ON THE
        # HOST (outside container egress isolation), so restrict transports to
        # https: ``protocol.allow=never`` blocks every transport (file://,
        # ext::, ssh, git://, ...) that is not explicitly re-allowed, and
        # ``protocol.https.allow=always`` re-allows https only. The insteadOf
        # rewrite still works because the rewritten URL is https. A specific
        # per-protocol allow from another config scope (e.g. a test-only
        # GIT_CONFIG protocol.file.allow=always) still overrides the general
        # ``protocol.allow`` default -- verified against git 2.34.1.
        try:
            _run(
                [
                    "git",
                    "-c",
                    "url.https://github.com/.insteadOf=git://github.com/",
                    "-c",
                    "protocol.allow=never",
                    "-c",
                    "protocol.https.allow=always",
                    "-C",
                    str(target),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                ]
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            log.warning(
                "Submodule init failed for %s (agent may be unable to run repo tests): %s",
                repo,
                getattr(exc, "stderr", exc),
            )
            # A partially-cloned submodule leaves a broken gitlink (a ``.git``
            # file pointing at a missing gitdir). ``git diff HEAD`` then exits
            # 128 (masked by ``--ignore-submodules=all`` on our diff capture,
            # but the agent's OWN git commands still choke on it). Deinit any
            # failed submodules so the tree stays clean while preserving any
            # successfully initialized siblings.
            try:
                status = subprocess.run(
                    ["git", "-C", str(target), "submodule", "status", "--recursive"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=int(os.environ.get("KCSI_GIT_TIMEOUT_SECONDS", "300")),
                )
                paths = [
                    line.split(maxsplit=1)[1]
                    for line in status.stdout.splitlines()
                    if line.startswith("-") and len(line.split(maxsplit=1)) == 2
                ]
                if paths:
                    _run(["git", "-C", str(target), "submodule", "deinit", "-f", "--", *paths])
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                pass
    else:
        _warn_if_uninitialized_submodules(target, repo)
    if seed_test_files:
        _seed_baseline_test_files(target, before_repo_set_cmd)


def _commit_reachable(target: Path, commit: str) -> bool:
    timeout_s = int(os.environ.get("KCSI_GIT_TIMEOUT_SECONDS", "300"))
    try:
        proc = subprocess.run(
            ["git", "-C", str(target), "cat-file", "-t", commit],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return (proc.stdout or "").strip() == "commit"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _find_local_source_clone(*, repos_cache_dir: Path, remote: str) -> Path | None:
    for candidate in repos_cache_dir.iterdir():
        if not candidate.is_dir():
            continue
        git_dir = candidate / ".git"
        if not git_dir.exists():
            continue
        try:
            proc = subprocess.run(
                ["git", "-C", str(candidate), "config", "--get", "remote.origin.url"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            continue
        candidate_remote = (proc.stdout or "").strip()
        if candidate_remote == remote:
            return candidate
    return None


def _warn_if_uninitialized_submodules(target: Path, repo: str) -> None:
    """Warn when an offline cache still has uninitialized submodules.

    `git submodule status` prefixes uninitialized entries with ``-``. Offline
    mode cannot populate them (submodule fetch needs network), so a cache built
    before submodule support -- or with a submodule-less prep -- leaves the
    agent facing the silent import failure this module otherwise guards against.
    Best-effort: any git error (incl. no submodules) is a quiet no-op.
    """
    timeout_s = int(os.environ.get("KCSI_GIT_TIMEOUT_SECONDS", "300"))
    try:
        proc = subprocess.run(
            ["git", "-C", str(target), "submodule", "status"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return
    uninitialized = [
        line.split()[1] for line in proc.stdout.splitlines() if line.startswith("-") and len(line.split()) >= 2
    ]
    if uninitialized:
        log.warning(
            "Offline repo prep for %s has %d uninitialized submodule(s) (%s); the "
            "agent may be unable to run repo tests. Re-prep this cache online to "
            "populate submodules.",
            repo,
            len(uninitialized),
            ", ".join(uninitialized),
        )


def _run(cmd: list[str]) -> None:
    timeout_s = int(os.environ.get("KCSI_GIT_TIMEOUT_SECONDS", "300"))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout_s, output=stdout, stderr=stderr) from exc
    if proc.returncode != 0:
        exc = subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
        # Log both streams — some git versions emit the rejected-tag diagnostic
        # to stdout rather than stderr, and we need both to diagnose
        # intermittent fetch failures in the cache-prep path.
        log.error(
            "Git command failed: %s\nstdout: %s\nstderr: %s",
            cmd,
            exc.stdout,
            exc.stderr,
        )
        raise exc
