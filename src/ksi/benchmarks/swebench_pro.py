from __future__ import annotations

import ast
import csv
import json
import logging
import os
import posixpath
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..eval.patch_extract import extract_patch, normalize_patch
from ..models import TaskSpec
from ..tasks.repo_cache import parse_test_files_from_before_cmd
from .swebench_pro_external import EVALUATOR_REVISION, REVISION_MARKER, SETUP_COMMAND

log = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_swebench_pro_repo_root() -> Path:
    return _repo_root() / "benchmarks" / "swebench_pro" / "evaluator"


def _swebench_pro_setup_hint() -> str:
    return f"run: {SETUP_COMMAND}"


def _workspace_repo_dir(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if (path / "repo").is_dir():
        path = path / "repo"
    if path.is_dir():
        return path
    return None


def _git_workspace_diff(repo_dir: Path) -> tuple[str, str | None]:
    if not (repo_dir / ".git").exists():
        return "", "workspace is not a git repository"

    # ``--ignore-submodules=all`` mirrors the container-side capture
    # (runtime_runner/src/main.ts): a broken submodule gitlink makes a plain
    # ``git diff HEAD`` exit 128, which this code would treat as an empty diff
    # and silently drop the agent's tracked-file edits (scored ``no_patch``).
    tracked = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary", "--ignore-submodules=all", "HEAD", "--"],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode != 0:
        return "", "git diff failed"
    tracked_diff = tracked.stdout

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if untracked.returncode != 0:
        return tracked_diff, "git ls-files failed"

    untracked_diffs: list[str] = []
    repo_root = repo_dir.resolve()
    for rel_path in [line.strip() for line in untracked.stdout.splitlines() if line.strip()][:50]:
        full_path = (repo_dir / rel_path).resolve()
        if not full_path.is_relative_to(repo_root):
            continue
        try:
            if not full_path.is_file() or full_path.stat().st_size > 1_000_000:
                continue
            content = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if content == "":
            lines = []
        else:
            lines = content[:-1].split("\n") if content.endswith("\n") else content.split("\n")
        untracked_diffs.append(
            "\n".join(
                [
                    f"diff --git a/{rel_path} b/{rel_path}",
                    "new file mode 100644",
                    "--- /dev/null",
                    f"+++ b/{rel_path}",
                    f"@@ -0,0 +1,{len(lines)} @@",
                    *[f"+{line}" for line in lines],
                    "",
                ]
            )
        )

    return "\n".join(part for part in [tracked_diff, *untracked_diffs] if part), None


def _patch_from_workspace(kwargs: dict[str, Any]) -> str | None:
    runtime_meta = kwargs.get("runtime_meta") if isinstance(kwargs.get("runtime_meta"), dict) else {}
    workspace_diff = runtime_meta.get("workspace_diff") if isinstance(runtime_meta, dict) else None
    if isinstance(workspace_diff, str) and workspace_diff.strip():
        return normalize_patch(workspace_diff)

    repo_dir = _workspace_repo_dir(
        kwargs.get("workspace_dir")
        or kwargs.get("host_workspace_repo_dir")
        or runtime_meta.get("host_workspace_repo_dir")
    )
    if repo_dir is None:
        return None
    diff, _ = _git_workspace_diff(repo_dir)
    return normalize_patch(diff)


_EDIT_TOOL_NAMES = {"apply_patch", "edit", "write", "multiedit", "str_replace", "notebookedit"}


def _tool_trace_items(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    trace = kwargs.get("tool_trace")
    if not isinstance(trace, list):
        runtime_meta = kwargs.get("runtime_meta")
        trace = runtime_meta.get("tool_trace") if isinstance(runtime_meta, dict) else None
    if not isinstance(trace, list):
        return []
    return [it for it in trace if isinstance(it, dict)]


def _count_edit_tool_calls(items: list[dict[str, Any]]) -> int:
    count = 0
    for it in items:
        name = str(it.get("tool_name") or it.get("name") or "").lower()
        # Exact names avoid treating TodoWrite, write_stdin, or unrelated
        # command names as source edits. Failed edit calls did not mutate the
        # workspace, so they must not turn a genuine no-submission into an
        # infrastructure failure.
        failed = bool(it.get("tool_is_error")) or str(it.get("status") or "").lower() in {"failed", "error"}
        if name in _EDIT_TOOL_NAMES and not failed:
            count += 1
    return count


def _patch_from_tool_trace(kwargs: dict[str, Any]) -> tuple[str | None, int]:
    """Return mutation evidence, never an unverified patch from tool output.

    Tool output can contain stale diffs, failed-command output, truncation, or
    arbitrary text that merely resembles a patch. The canonical workspace
    capture is the only trustworthy patch source.
    """
    items = _tool_trace_items(kwargs)
    return None, _count_edit_tool_calls(items)


def _normalize_diff_path(value: str) -> str:
    text = (value or "").strip()
    if not text or text == "/dev/null":
        return ""
    if "\t" in text:
        text = text.split("\t", 1)[0]
    if text.startswith("a/") or text.startswith("b/"):
        text = text[2:]
    while text.startswith("./"):
        text = text[2:]
    text = text.lstrip("/")
    normalized = posixpath.normpath(text)
    if normalized in {".", ".."} or normalized.startswith("../"):
        return ""
    return normalized


def _diff_block_paths(block: list[str]) -> set[str]:
    paths: set[str] = set()
    for line in block:
        if line.startswith("diff --git "):
            try:
                parts = shlex.split(line.strip())
            except ValueError:
                parts = line.strip().split()
            if len(parts) >= 4:
                for value in parts[2:4]:
                    normalized = _normalize_diff_path(value)
                    if normalized:
                        paths.add(normalized)
        elif line.startswith("--- ") or line.startswith("+++ "):
            normalized = _normalize_diff_path(line[4:])
            if normalized:
                paths.add(normalized)
    return paths


def _grader_test_file_paths(raw_sample: dict[str, Any] | None) -> set[str]:
    """Collect file paths the grader will cherry-pick onto the agent's patch.

    Two sources, in order of reliability:
    1. The last line of ``before_repo_set_cmd``: ``git checkout <commit> -- <files>``
       — these are exactly the files the upstream entryscript overwrites after
       applying the agent's patch, so any agent edits to them are nullified.
    2. ``selected_test_files_to_run`` and FAIL_TO_PASS / PASS_TO_PASS entries
       that look like file paths (``test/posts.js | name``) — DGM's strategy.
       Pure test names like ``TestCriteria`` are filtered out by the
       ``_normalize_diff_path`` guard since they never match a diff path.
    """
    paths: set[str] = set()
    if not isinstance(raw_sample, dict):
        return paths
    _, files = parse_test_files_from_before_cmd(str(raw_sample.get("before_repo_set_cmd") or ""))
    for rel in files:
        normalized = _normalize_diff_path(rel)
        if normalized:
            paths.add(normalized)

    candidate_lists: list[Any] = [
        raw_sample.get("selected_test_files_to_run"),
        raw_sample.get("fail_to_pass") or raw_sample.get("FAIL_TO_PASS"),
        raw_sample.get("pass_to_pass") or raw_sample.get("PASS_TO_PASS"),
    ]
    for raw in candidate_lists:
        for value in _parse_test_name_list_loose(raw):
            for candidate in (value, value.split(" | ", 1)[0], value.split("::", 1)[0]):
                normalized = _normalize_diff_path(candidate)
                # Require a path-like shape (contains a slash OR a known test
                # file extension) to avoid stripping unrelated source hunks
                # when the upstream supplies bare test-NAMES like "TestHosts".
                if normalized and (
                    "/" in normalized
                    or any(
                        normalized.endswith(ext)
                        for ext in (".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".rb")
                    )
                ):
                    paths.add(normalized)
    return paths


def _parse_test_name_list_loose(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [text]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def filter_grader_test_hunks(patch: str, raw_sample: dict[str, Any] | None) -> str:
    """Strip diff blocks that touch files the grader will overwrite.

    The grader's entryscript applies the agent's patch and then
    ``git checkout <test_commit> -- <files>``, which clobbers any agent edits
    to those test files. Without this filter the agent's test-file changes
    inflate the patch with no effect (and can cause spurious ``git apply``
    conflicts on edge cases). Mirrors DGM ``filter_private_test_patch_hunks``
    but uses ``before_repo_set_cmd`` as the authoritative source of truth.
    """
    grader_paths = _grader_test_file_paths(raw_sample)
    if not grader_paths or not patch.strip():
        return patch

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    kept: list[str] = []
    for block in blocks:
        if _diff_block_paths(block) & grader_paths:
            continue
        kept.extend(block)
    return "".join(kept)


EXPECTED_EVAL_REVISION = EVALUATOR_REVISION
# Prefix the external evaluator uses to name per-instance output files
# (f"{prefix}_output.json"); MUST match the readback below.
_SWE_OUTPUT_PREFIX = "ksi"


def _json_artifact_parse_error(path: Path, exc: BaseException) -> str:
    try:
        prefix = path.read_text(encoding="utf-8")[:200]
    except Exception:
        return str(exc)
    return f"{exc}; artifact_prefix={prefix!r}"


def _read_oom_marker(output_dir: Path, instance_id: str) -> dict[str, Any] | None:
    """Read the OOM-kill diagnostic the patched evaluator writes (if any).

    The vendored ``eval_with_docker`` (patched by
    ``per_instance_resource_limits.patch``) writes
    ``{prefix}_oom.json`` next to the instance's other output files when
    Docker reports the container as OOM-killed. The caller gates an OOM kill on
    a non-resolved instance to ``oom_killed`` (scored None/unscored) so a
    memory-cap kill is not recorded as a genuine agent test failure.
    """
    marker_path = output_dir / instance_id / f"{_SWE_OUTPUT_PREFIX}_oom.json"
    if not marker_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict) or not marker.get("oom_killed"):
        return None
    log.warning(
        "swebench_pro instance %s: container OOM-killed (mem_limit=%s, status_code=%s)",
        instance_id,
        marker.get("mem_limit"),
        marker.get("status_code"),
    )
    return {"oom_killed": True}


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _revision_marker(path: Path) -> str | None:
    marker = path / REVISION_MARKER
    if not marker.is_file():
        return None
    value = marker.read_text(encoding="utf-8").strip()
    return value or None


def _swebench_pro_revision_error(repo_root: Path) -> str | None:
    current = _git_head(repo_root)
    if current is not None:
        if current != EXPECTED_EVAL_REVISION:
            return (
                f"SWE-bench Pro evaluator at {repo_root} is pinned to {current}, "
                f"expected {EXPECTED_EVAL_REVISION} ({_swebench_pro_setup_hint()})"
            )
        return None

    marker = _revision_marker(repo_root)
    if marker != EXPECTED_EVAL_REVISION:
        marker_note = f" found marker {marker}" if marker else " no revision marker found"
        return (
            f"Cannot verify SWE-bench Pro evaluator revision at {repo_root}:"
            f"{marker_note}; expected {EXPECTED_EVAL_REVISION} ({_swebench_pro_setup_hint()})"
        )
    return None


def _terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            return


def _force_close_pipes(proc: subprocess.Popen[Any]) -> None:
    for pipe in (proc.stdout, proc.stderr):
        if pipe:
            try:
                pipe.close()
            except Exception:
                pass


def _run_eval_command(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: int,
) -> tuple[subprocess.CompletedProcess[str] | None, dict[str, Any]]:
    cleanup: dict[str, Any] = {}
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return (
            subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            ),
            cleanup,
        )
    except subprocess.TimeoutExpired:
        cleanup = {
            "swebench_process_cleanup_attempted": True,
            "swebench_process_cleanup_method": "process_group" if os.name == "posix" else "process",
            "swebench_container_cleanup_attempted": False,
            "swebench_container_cleanup_status": "not_attempted",
            "swebench_container_cleanup_note": (
                "SWE-bench Pro harness container names are owned by the upstream evaluator; "
                "timeout cleanup only reaped the evaluator process tree."
            ),
        }
        _terminate_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
            cleanup["swebench_process_cleanup_status"] = "terminated"
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=30)
                cleanup["swebench_process_cleanup_status"] = "killed"
            except subprocess.TimeoutExpired:
                _force_close_pipes(proc)
                stdout, stderr = "", ""
                cleanup["swebench_process_cleanup_status"] = "stuck"
        cleanup["swebench_stdout_tail"] = (stdout or "")[-2000:]
        cleanup["swebench_stderr_tail"] = (stderr or "")[-2000:]
        return None, cleanup


def _read_raw_sample_row(raw_sample_path: Path, instance_id: str) -> dict[str, Any] | None:
    suffix = raw_sample_path.suffix.lower()
    if suffix == ".csv":
        with raw_sample_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("instance_id") or "").strip() == instance_id:
                    return dict(row)
        return None
    if suffix == ".jsonl":
        with raw_sample_path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if isinstance(row, dict) and str(row.get("instance_id") or "").strip() == instance_id:
                    return row
        return None
    raise ValueError(f"Unsupported SWE-bench Pro raw sample file type: {raw_sample_path}")


def _parse_test_name_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [text]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _build_tests_status(output: dict[str, Any], raw_sample: dict[str, Any] | None) -> dict[str, Any]:
    # ``or []`` (not just a ``get`` default): the grader may emit an explicit
    # ``{"tests": null}`` on a parse/timeout signal, and ``dict.get("tests", [])``
    # returns ``None`` in that case — iterating it would raise TypeError and the
    # whole eval result would be lost instead of recorded as a harness failure.
    tests = (output.get("tests") or []) if isinstance(output, dict) else []
    observed_tests: dict[str, str] = {}
    skipped_statuses = {"SKIPPED", "XFAIL", "ERROR"}
    for item in tests:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        status = str(item.get("status") or "").strip().upper()
        if name and status:
            observed_tests[name] = status

    fail_to_pass = _parse_test_name_list(
        (raw_sample or {}).get("fail_to_pass") or (raw_sample or {}).get("FAIL_TO_PASS")
    )
    pass_to_pass = _parse_test_name_list(
        (raw_sample or {}).get("pass_to_pass") or (raw_sample or {}).get("PASS_TO_PASS")
    )

    def classify(expected: list[str]) -> dict[str, Any]:
        return {
            "success": [name for name in expected if observed_tests.get(name) == "PASSED"],
            "failure": [name for name in expected if observed_tests.get(name) == "FAILED"],
            "skipped": [name for name in expected if observed_tests.get(name) in skipped_statuses],
            "unknown": [name for name in expected if name not in observed_tests],
        }

    f2p_class = classify(fail_to_pass)
    p2p_class = classify(pass_to_pass)
    return {
        "observed_count": len(observed_tests),
        # Count of expected test names absent from the grader's JSON output.
        # These force the binary scorer to 0 even when the grader resolved the
        # instance (research-integrity HIGH-1); surface so campaigns can measure
        # the downward bias.
        "unknown_count": len(f2p_class["unknown"]) + len(p2p_class["unknown"]),
        "FAIL_TO_PASS": f2p_class,
        "PASS_TO_PASS": p2p_class,
    }


@dataclass
class SwebenchProEvaluator:
    raw_sample_path: str
    repo_root: str = ""
    dockerhub_username: str = "jefzda"
    scripts_dir: str = ""
    timeout_sec: int = 3600
    use_local_docker: bool = True
    docker_platform: str | None = None
    block_network: bool = False
    harness_grace_sec: int = 0
    # Total attempts for the harness run. A returncode-0 run that produces no
    # output.json for an instance with expected tests is an interrupted/cut-off
    # container test-run — transient, not a real
    # failure — so re-run before giving up rather than silently scoring it 0.
    max_eval_attempts: int = 2

    def __post_init__(self) -> None:
        self.timeout_sec = int(self.timeout_sec)
        self.harness_grace_sec = int(self.harness_grace_sec)
        self.max_eval_attempts = int(self.max_eval_attempts)
        if self.timeout_sec < 1:
            raise ValueError("timeout_sec must be >= 1")
        if self.harness_grace_sec < 0:
            raise ValueError("harness_grace_sec must be >= 0")
        if self.max_eval_attempts < 1:
            raise ValueError("max_eval_attempts must be >= 1")

    def evaluate(self, *, task: TaskSpec, model_output: str, **kwargs: Any) -> dict[str, Any]:
        patch = _patch_from_workspace(kwargs)
        patch_source = "workspace_diff" if patch else "model_output"
        if not patch:
            runtime_meta = kwargs.get("runtime_meta") if isinstance(kwargs.get("runtime_meta"), dict) else {}
            capture_error = runtime_meta.get("workspace_diff_capture_error") if isinstance(runtime_meta, dict) else None
            patch = extract_patch(model_output or "")
        if not patch:
            # Workspace-diff capture came back empty and the model output carried
            # no patch. Tool traces supply mutation evidence only: their output
            # is not a trustworthy patch source.
            _trace_patch, trace_edits = _patch_from_tool_trace(kwargs)
            if capture_error or trace_edits:
                log.warning(
                    "swebench_pro %s: capture_failed — agent made %d edit tool "
                    "call(s) but workspace_diff was empty and no patch was "
                    "recoverable from the trace (check submodule init / diff "
                    "capture)",
                    task.id,
                    trace_edits,
                )
                return {
                    "swebench_status": "capture_failed",
                    "instance_id": task.id,
                    "swebench_capture_failed_edit_calls": trace_edits,
                    "swebench_capture_error": str(capture_error or "workspace diff missing after successful edit"),
                }
            else:
                return {
                    "swebench_status": "no_patch",
                    "instance_id": task.id,
                }

        raw_sample_path = Path(self.raw_sample_path).expanduser().resolve()
        # Filter out hunks targeting files the grader cherry-picks back from
        # the test commit. Without this the agent's edits to those files are
        # silently overwritten — wasted patch bytes plus occasional ``git
        # apply`` conflicts. We do this BEFORE reading raw_sample for
        # eval_results so the same filter applies regardless of how the
        # raw_sample is later parsed.
        pre_filter_raw = _read_raw_sample_row(raw_sample_path, task.id) if raw_sample_path.exists() else None
        original_patch_len = len(patch)
        if pre_filter_raw is not None:
            patch = filter_grader_test_hunks(patch, pre_filter_raw)
        if not patch.strip():
            return {
                "swebench_status": "no_patch",
                "instance_id": task.id,
                "swebench_filter_note": (
                    "agent_patch was entirely test-file modifications; the grader "
                    "cherry-picks those files from the test commit so nothing would apply"
                ),
                "swebench_filter_original_len": original_patch_len,
            }
        filter_dropped_bytes = original_patch_len - len(patch)
        repo_root = Path(self.repo_root).expanduser().resolve() if self.repo_root else _default_swebench_pro_repo_root()
        scripts_dir = Path(self.scripts_dir).expanduser().resolve() if self.scripts_dir else repo_root / "run_scripts"
        script_path = repo_root / "swe_bench_pro_eval.py"

        if not raw_sample_path.exists():
            return {
                "swebench_status": "harness_failed",
                "instance_id": task.id,
                "error": f"missing raw sample path: {raw_sample_path}",
            }
        if not script_path.exists():
            return {
                "swebench_status": "harness_failed",
                "instance_id": task.id,
                "error": f"missing evaluator script: {script_path} ({_swebench_pro_setup_hint()})",
            }
        revision_error = _swebench_pro_revision_error(repo_root)
        if revision_error:
            return {
                "swebench_status": "harness_failed",
                "instance_id": task.id,
                "error": revision_error,
            }
        if not scripts_dir.exists():
            return {
                "swebench_status": "harness_failed",
                "instance_id": task.id,
                "error": f"missing scripts dir: {scripts_dir} ({_swebench_pro_setup_hint()})",
            }

        # Expected-test names for this instance. Used to decide whether a
        # missing output.json is a transient interrupted run (retry) vs an
        # instance that legitimately has nothing to observe.
        raw_sample = _read_raw_sample_row(raw_sample_path, task.id)
        expected_nonempty = bool(
            _parse_test_name_list((raw_sample or {}).get("fail_to_pass") or (raw_sample or {}).get("FAIL_TO_PASS"))
            or _parse_test_name_list((raw_sample or {}).get("pass_to_pass") or (raw_sample or {}).get("PASS_TO_PASS"))
        )

        # Retry loop: a returncode-0 run that yields no output.json for an
        # instance with expected tests is an interrupted/cut-off container
        # test-run — transient. Re-run rather than silently scoring
        # it 0; if every attempt is missing output.json, surface a retryable
        # ``harness_failed`` instead of a false resolved=False.
        last_missing_output: dict[str, Any] | None = None
        for attempt in range(1, self.max_eval_attempts + 1):
            tmpdir = Path(tempfile.mkdtemp(prefix="swebench-pro-"))
            patch_path = tmpdir / "patches.json"
            output_dir = tmpdir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            patch_payload = [{"instance_id": task.id, "patch": patch, "prefix": _SWE_OUTPUT_PREFIX}]
            patch_path.write_text(json.dumps(patch_payload), encoding="utf-8")

            cmd = [
                sys.executable,
                str(script_path),
                "--raw_sample_path",
                str(raw_sample_path),
                "--patch_path",
                str(patch_path),
                "--output_dir",
                str(output_dir),
                "--dockerhub_username",
                self.dockerhub_username,
                "--scripts_dir",
                str(scripts_dir),
                "--num_workers",
                "1",
            ]
            if self.use_local_docker:
                cmd.append("--use_local_docker")
            if self.block_network:
                cmd.append("--block_network")
            if self.docker_platform:
                cmd.extend(["--docker_platform", self.docker_platform])

            # Minimal env to avoid leaking API keys to evaluator subprocesses.
            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            repo_root_str = str(repo_root)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "PYTHONPATH": f"{repo_root_str}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else repo_root_str,
                "PYTHONUNBUFFERED": "1",
            }

            try:
                proc, timeout_cleanup = _run_eval_command(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    timeout=self.timeout_sec + self.harness_grace_sec,
                )
                if proc is None:
                    return {
                        "swebench_status": "harness_timeout",
                        "instance_id": task.id,
                        **timeout_cleanup,
                    }
                if proc.returncode != 0:
                    return {
                        "swebench_status": "harness_failed",
                        "instance_id": task.id,
                        "swebench_returncode": proc.returncode,
                        "swebench_stdout_tail": (proc.stdout or "")[-2000:],
                        "swebench_stderr_tail": (proc.stderr or "")[-2000:],
                    }

                # The patched evaluator (per_instance_resource_limits.patch) writes a
                # marker when Docker reports the container was OOM-killed by the
                # mem_limit cap. An OOM kill on a non-resolved instance is gated to
                # None (unscored) below, not treated as a genuine test failure.
                oom_marker = _read_oom_marker(output_dir, task.id)

                eval_results_path = output_dir / "eval_results.json"
                if not eval_results_path.exists():
                    return {
                        "swebench_status": "missing_report",
                        "instance_id": task.id,
                        "swebench_report_path": str(eval_results_path),
                        "swebench_stdout_tail": (proc.stdout or "")[-2000:],
                        "swebench_stderr_tail": (proc.stderr or "")[-2000:],
                    }

                try:
                    eval_results = json.loads(eval_results_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                    return {
                        "swebench_status": "harness_failed",
                        "instance_id": task.id,
                        "swebench_report_path": str(eval_results_path),
                        "swebench_malformed_report": True,
                        "swebench_parse_error": _json_artifact_parse_error(eval_results_path, exc),
                        "swebench_stdout_tail": (proc.stdout or "")[-2000:],
                        "swebench_stderr_tail": (proc.stderr or "")[-2000:],
                    }
                if not isinstance(eval_results, dict) or task.id not in eval_results:
                    return {
                        "swebench_status": "missing_report",
                        "instance_id": task.id,
                        "swebench_report_path": str(eval_results_path),
                        "swebench_stdout_tail": (proc.stdout or "")[-2000:],
                        "swebench_stderr_tail": (proc.stderr or "")[-2000:],
                    }

                output_json_path = output_dir / task.id / f"{_SWE_OUTPUT_PREFIX}_output.json"
                if not output_json_path.exists() and expected_nonempty:
                    # Interrupted run: returncode 0 and eval_results present, but
                    # no output.json for an instance that has expected tests —
                    # the container test-run was cut off before parser.py wrote
                    # results. Transient; retry instead of scoring 0.
                    last_missing_output = {
                        "swebench_status": "harness_failed",
                        "instance_id": task.id,
                        "swebench_missing_output_json": True,
                        "swebench_eval_attempts": attempt,
                        "swebench_stdout_tail": (proc.stdout or "")[-2000:],
                        "swebench_stderr_tail": (proc.stderr or "")[-2000:],
                        **(oom_marker or {}),
                    }
                    continue

                instance_output: dict[str, Any] = {}
                if output_json_path.exists():
                    try:
                        loaded = json.loads(output_json_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                        return {
                            "swebench_status": "harness_failed",
                            "instance_id": task.id,
                            "swebench_output_json_path": str(output_json_path),
                            "swebench_malformed_output_json": True,
                            "swebench_parse_error": _json_artifact_parse_error(output_json_path, exc),
                            "swebench_stdout_tail": (proc.stdout or "")[-2000:],
                            "swebench_stderr_tail": (proc.stderr or "")[-2000:],
                        }
                    if isinstance(loaded, dict):
                        instance_output = loaded

                tests_status = _build_tests_status(instance_output, raw_sample)
                resolved = bool(eval_results.get(task.id))

                if oom_marker and not resolved:
                    # The container was OOM-killed by the mem_limit cap and the
                    # instance did NOT resolve. The kill — not the agent's patch —
                    # may be why the tests didn't pass, so the 0.0 verdict is
                    # untrustworthy. Score as an infra failure (None/unscored,
                    # retryable) rather than a fabricated 0.0 that would feed
                    # _best_scores/distillation/forum as a genuine agent failure.
                    # (A resolved-despite-OOM instance is kept below: all required
                    # tests passed and were recorded, so the verdict stands.)
                    return {
                        "swebench_status": "oom_killed",
                        "instance_id": task.id,
                        "swebench_stdout_tail": (proc.stdout or "")[-2000:],
                        "swebench_stderr_tail": (proc.stderr or "")[-2000:],
                        **oom_marker,
                    }

                run_summary = {
                    "total_instances": 1,
                    "submitted_instances": 1,
                    "completed_instances": 1,
                    "resolved_instances": 1 if resolved else 0,
                    "unresolved_instances": 0 if resolved else 1,
                    "completed_ids": [task.id],
                    "submitted_ids": [task.id],
                    "resolved_ids": [task.id] if resolved else [],
                    "unresolved_ids": [] if resolved else [task.id],
                    "schema_version": "swebench_pro_v1",
                }
                instance_report = {
                    "status": "ok",
                    "resolved": resolved,
                    "tests": instance_output.get("tests", []) if isinstance(instance_output, dict) else [],
                    "tests_status": tests_status,
                }
                result: dict[str, Any] = {
                    "swebench_status": "ok",
                    "resolved": resolved,
                    "native_score": 1.0 if resolved else 0.0,
                    "instance_id": task.id,
                    "patch_source": patch_source,
                    "run_summary": run_summary,
                    "instance_report": instance_report,
                    "swebench_filter_dropped_bytes": filter_dropped_bytes,
                    **(oom_marker or {}),
                }
                if attempt > 1:
                    result["swebench_eval_attempts"] = attempt
                # If no PASS/FAIL markers were observed and the run wasn't
                # resolved, surface the harness's stderr/stdout tails. Most
                # commonly this means the patched code failed to compile against
                # the cherry-picked tests — without these tails the failure mode
                # is invisible from the eval JSON alone.
                if not resolved and tests_status.get("observed_count", 0) == 0:
                    result["swebench_stdout_tail"] = (proc.stdout or "")[-2000:]
                    result["swebench_stderr_tail"] = (proc.stderr or "")[-2000:]
                    result["swebench_observed_count_zero"] = True
                # Diagnostic for the unknown-test bias (research-integrity HIGH-1):
                # the grader resolved this instance, but some expected test names
                # didn't match its JSON output exactly, so the binary scorer will
                # mark it 0. Surface it so the downward bias is measurable.
                if resolved and tests_status.get("unknown_count", 0) > 0:
                    result["swebench_resolved_but_unknown_tests"] = True
                return result
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Every attempt produced no output.json (interrupted runs): surface a
        # retryable harness failure, never a silent scored 0.
        return last_missing_output or {
            "swebench_status": "harness_failed",
            "instance_id": task.id,
            "swebench_missing_output_json": True,
            "swebench_eval_attempts": self.max_eval_attempts,
        }
