from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ksi.benchmarks.terminal_bench_2 import resolve_terminal_bench_2_task_contract
from ksi.benchmarks.terminal_bench_2_runtime import (
    _TB2_CONTAINER_CTRF_PATH,
    _TB2_CONTAINER_REWARD_PATH,
    _TB2_REWARD_READOUT_SENTINEL,
    _acquire_tb2_image,
    _agent_phase_copies,
    _docker_build_with_retry,
    _docker_pull_with_retry,
    _docker_run_command,
    _enforce_tb2_image_digest_manifest,
    _environment_dir_hash,
    _extract_json_object,
    _keep_tb2_images_default,
    _looks_like_transient_docker_registry_failure,
    _normalize_tb2_repo_digest,
    _parse_reward,
    _parse_reward_text,
    _resolve_tb2_max_steps,
    _run_ksi_agent_in_tb2_container,
    _stable_image_tag,
    _tb2_trim_oldest_history,
    _verifier_phase_copies,
    _verifier_sanitize_paths,
    default_agent_command,
    materialize_terminal_bench_2_workspace_seed,
    run_terminal_bench_2_trial,
)
from ksi.errors import ContainerRegistryError
from ksi.models import TaskSpec
from ksi.runtime import RuntimeResult, TerminalBench2Executor
from ksi.tokens import LLMResponse, TokenUsage


def test_default_agent_command_oracle() -> None:
    assert default_agent_command(agent_mode="oracle") == "/bin/bash /solution/solve.sh"


def test_default_agent_command_noop() -> None:
    assert default_agent_command(agent_mode="noop") == "true"


def test_default_agent_command_ksi() -> None:
    assert default_agent_command(agent_mode="ksi") == ""


def test_default_agent_command_requires_explicit_command() -> None:
    with pytest.raises(ValueError, match="requires --agent-command"):
        default_agent_command(agent_mode="command")


def test_extract_json_object_supports_fenced_payload() -> None:
    payload = _extract_json_object('```json\n{"action":"final","summary":"done"}\n```')
    assert payload["action"] == "final"
    assert payload["summary"] == "done"


def test_extract_json_object_supports_wrapped_json_payload() -> None:
    raw = """I'll start by reading the instruction first.
<function_calls>
<invoke name="send_message">
<parameter name="content">{"action":"shell","command":"cat /workspace/task/workspace/tb2/instruction.md","timeout_sec":10,"summary":"Read instruction"}</parameter>
</invoke>
</function_calls>
"""
    payload = _extract_json_object(raw)
    assert payload["action"] == "shell"
    assert payload["command"] == "cat /workspace/task/workspace/tb2/instruction.md"
    assert payload["timeout_sec"] == 10


def test_extract_json_object_supports_wrapped_command_payload() -> None:
    raw = """<function_calls>
<invoke name="execute">
<parameter name="command">ls -la /workspace/task/workspace/</parameter>
<parameter name="summary">Inspect the mounted workspace</parameter>
<parameter name="timeout_sec">15</parameter>
</invoke>
</function_calls>
"""
    payload = _extract_json_object(raw)
    assert payload["action"] == "shell"
    assert payload["command"] == "ls -la /workspace/task/workspace/"
    assert payload["summary"] == "Inspect the mounted workspace"
    assert payload["timeout_sec"] == 15


def test_extract_json_object_normalizes_command_only_payload() -> None:
    payload = _extract_json_object('{"command":"pwd","summary":"check cwd"}')
    assert payload["action"] == "shell"
    assert payload["command"] == "pwd"
    assert payload["summary"] == "check cwd"
    assert payload["timeout_sec"] == 60


def test_extract_json_object_parses_native_read_action() -> None:
    payload = _extract_json_object('{"action":"read","path":"/etc/hosts","offset":1,"limit":50,"summary":"check"}')
    assert payload["action"] == "read"
    assert payload["path"] == "/etc/hosts"
    assert payload["offset"] == 1
    assert payload["limit"] == 50


def test_extract_json_object_parses_native_write_action() -> None:
    payload = _extract_json_object('{"action":"write","path":"/tmp/x","content":"hello\\nworld","summary":"new file"}')
    assert payload["action"] == "write"
    assert payload["path"] == "/tmp/x"
    assert payload["content"] == "hello\nworld"


def test_extract_json_object_recovers_write_with_literal_newlines() -> None:
    # Models frequently emit a write action whose content is a real multi-line
    # code block instead of a \n-escaped string. Strict json.loads rejects the
    # literal newlines, and the brace-scan salvage used to latch onto the inner
    # "{}" fragment in the code body (which has no "action" key), surfacing the
    # misleading "Unsupported TB2 bridge action: (missing)" error and silently
    # dropping the file write. The lenient decoder must recover the real action.
    raw = (
        '{"action":"write","path":"/app/maze_explorer.py","content":"'
        "import subprocess\n"
        "def explore(maze_id):\n"
        "    d = {}\n"
        "    return d\n"
        '","summary":"write explorer"}'
    )
    payload = _extract_json_object(raw)
    assert payload["action"] == "write"
    assert payload["path"] == "/app/maze_explorer.py"
    assert payload["content"] == "import subprocess\ndef explore(maze_id):\n    d = {}\n    return d\n"


def test_extract_json_object_recovers_shell_heredoc_with_literal_newlines() -> None:
    raw = '{"action":"shell","command":"cd /app && python3 <<EOF\nprint(1)\nEOF","summary":"run"}'
    payload = _extract_json_object(raw)
    assert payload["action"] == "shell"
    assert payload["command"] == "cd /app && python3 <<EOF\nprint(1)\nEOF"


def test_extract_json_object_parses_native_edit_action() -> None:
    payload = _extract_json_object(
        '{"action":"edit","path":"/tmp/x","old_string":"foo","new_string":"bar","replace_all":true,"summary":"swap"}'
    )
    assert payload["action"] == "edit"
    assert payload["replace_all"] is True
    assert payload["old_string"] == "foo"
    assert payload["new_string"] == "bar"


def test_extract_json_object_parses_native_glob_action() -> None:
    payload = _extract_json_object('{"action":"glob","pattern":"*.py","path":"/repo","summary":"find py"}')
    assert payload["action"] == "glob"
    assert payload["pattern"] == "*.py"
    assert payload["path"] == "/repo"


def test_extract_json_object_parses_native_grep_action() -> None:
    payload = _extract_json_object(
        '{"action":"grep","pattern":"TODO","path":"/repo","output_mode":"content","summary":"todos"}'
    )
    assert payload["action"] == "grep"
    assert payload["pattern"] == "TODO"
    assert payload["output_mode"] == "content"


def test_extract_json_object_supports_malformed_invoke_command_wrapper() -> None:
    raw = """I'll inspect the workspace first.
<function_calls>
<invoke name="bash">
<invoke name="shell">
<invoke name="command">cat /workspace/task/workspace/MEMORY.md 2>/dev/null || echo "No MEMORY.md found"</parameter>
</invoke>
</function_calls>
"""
    payload = _extract_json_object(raw)
    assert payload["action"] == "shell"
    assert payload["command"] == 'cat /workspace/task/workspace/MEMORY.md 2>/dev/null || echo "No MEMORY.md found"'
    assert payload["timeout_sec"] == 60


def test_parse_reward_reads_float(tmp_path: Path) -> None:
    path = tmp_path / "reward.txt"
    path.write_text("1\n", encoding="utf-8")
    assert _parse_reward(path) == 1.0


def test_parse_reward_rejects_invalid_number(tmp_path: Path) -> None:
    path = tmp_path / "reward.txt"
    path.write_text("not-a-number\n", encoding="utf-8")
    assert _parse_reward(path) is None


@pytest.mark.parametrize("raw", ["inf", "-inf", "nan", "Infinity", "+inf", "INF", "NaN"])
def test_parse_reward_text_rejects_non_finite(raw: str) -> None:
    # The reward file is AGENT-CONTROLLED and ``float`` accepts nan/inf literals
    # without raising, so a non-finite value must be treated as "no genuine
    # reward" (None), same as an unparseable file -- otherwise ``inf`` would
    # score as a solve.
    assert _parse_reward_text(raw) is None


@pytest.mark.parametrize(("raw", "expected"), [("1.0", 1.0), ("0.5", 0.5), ("0", 0.0), ("1", 1.0)])
def test_parse_reward_text_accepts_finite(raw: str, expected: float) -> None:
    assert _parse_reward_text(raw) == expected


def test_parse_reward_rejects_non_finite_file(tmp_path: Path) -> None:
    path = tmp_path / "reward.txt"
    path.write_text("inf\n", encoding="utf-8")
    assert _parse_reward(path) is None


def test_transient_docker_registry_failure_detection() -> None:
    proc = subprocess.CompletedProcess(
        args=["docker", "build"],
        returncode=1,
        stdout="",
        stderr="failed to fetch oauth token: Post https://auth.docker.io/token: net/http: TLS handshake timeout",
    )
    assert _looks_like_transient_docker_registry_failure(proc) is True


def test_docker_build_with_retry_retries_transient_registry_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_run(cmd: list[str], *, timeout_sec: float | None = None):
        calls["count"] += 1
        if calls["count"] < 3:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="failed to authorize: tls handshake timeout",
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.time.sleep", lambda _: None)

    proc = _docker_build_with_retry(["docker", "build"], timeout_sec=30, attempts=3)

    assert proc.returncode == 0
    assert calls["count"] == 3


def test_docker_build_with_retry_returns_124_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], *, timeout_sec: float | None = None):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_sec or 0)

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.time.sleep", lambda _: None)

    proc = _docker_build_with_retry(["docker", "build"], timeout_sec=30, attempts=1)
    assert proc.returncode == 124
    assert "timed out" in (proc.stderr or "")


def test_docker_exec_uses_absolute_container_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _TB2_CONTAINER_BASH, _docker_exec

    recorded: list[list[str]] = []

    def fake_run(cmd: list[str], *, timeout_sec: float | None = None):
        recorded.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)

    _docker_exec(container_name="tb2", command="true", timeout_sec=1)

    assert recorded == [["docker", "exec", "tb2", _TB2_CONTAINER_BASH, "-c", "true"]]
    assert recorded[0][3] != "bash"


def test_environment_dir_hash_stable_for_identical_content(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
        (d / "deps").mkdir()
        (d / "deps" / "patch.txt").write_text("hello\n", encoding="utf-8")
    assert _environment_dir_hash(a) == _environment_dir_hash(b)


def test_environment_dir_hash_changes_when_file_changes(tmp_path: Path) -> None:
    d = tmp_path / "env"
    d.mkdir()
    (d / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    h1 = _environment_dir_hash(d)
    (d / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN apt-get update\n", encoding="utf-8")
    h2 = _environment_dir_hash(d)
    assert h1 != h2


def test_stable_image_tag_deterministic(tmp_path: Path) -> None:
    d = tmp_path / "env"
    d.mkdir()
    (d / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    tag_one = _stable_image_tag(environment_dir=d, safe_task="demo-task")
    tag_two = _stable_image_tag(environment_dir=d, safe_task="demo-task")
    assert tag_one == tag_two
    assert tag_one.startswith("ksi-tb2-demo-task:")


def test_keep_tb2_images_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KSI_TB2_KEEP_IMAGES", raising=False)
    assert _keep_tb2_images_default() is True
    monkeypatch.setenv("KSI_TB2_KEEP_IMAGES", "0")
    assert _keep_tb2_images_default() is False
    monkeypatch.setenv("KSI_TB2_KEEP_IMAGES", "false")
    assert _keep_tb2_images_default() is False
    monkeypatch.setenv("KSI_TB2_KEEP_IMAGES", "1")
    assert _keep_tb2_images_default() is True


def test_docker_pull_with_retry_returns_success_on_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_run(cmd: list[str], *, timeout_sec: float | None = None):
        calls["count"] += 1
        assert cmd[:2] == ["docker", "pull"]
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    proc = _docker_pull_with_retry("alexgshaw/example:tag", timeout_sec=30)
    assert proc.returncode == 0
    assert calls["count"] == 1


def test_docker_pull_with_retry_retries_transient_registry_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_run(cmd: list[str], *, timeout_sec: float | None = None):
        calls["count"] += 1
        if calls["count"] < 3:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="failed to fetch oauth token: tls handshake timeout",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.time.sleep", lambda _: None)

    proc = _docker_pull_with_retry("alexgshaw/example:tag", timeout_sec=30, attempts=3)
    assert proc.returncode == 0
    assert calls["count"] == 3


@pytest.mark.parametrize(
    "stderr",
    [
        "unauthorized: authentication required",
        "HTTP 401 Unauthorized",
        "received unexpected HTTP status: 500 Internal Server Error",
    ],
)
def test_docker_pull_with_retry_retries_ambiguous_registry_failures(
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
) -> None:
    calls = {"count": 0}

    def fake_run(cmd: list[str], *, timeout_sec: float | None = None):
        calls["count"] += 1
        if calls["count"] < 3:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr=stderr)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.time.sleep", lambda _: None)

    proc = _docker_pull_with_retry("alexgshaw/example:tag", timeout_sec=30, attempts=3)
    assert proc.returncode == 0
    assert calls["count"] == 3


def test_docker_pull_with_retry_normalizes_and_retries_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_run(cmd: list[str], *, timeout_sec: float | None = None):
        calls["count"] += 1
        if calls["count"] < 3:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_sec or 0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.time.sleep", lambda _: None)

    proc = _docker_pull_with_retry("alexgshaw/example:tag", timeout_sec=30, attempts=3)
    assert proc.returncode == 0
    assert calls["count"] == 3


def test_docker_pull_with_retry_does_not_retry_non_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_run(cmd: list[str], *, timeout_sec: float | None = None):
        calls["count"] += 1
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="manifest unknown: image not found in registry",
        )

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.time.sleep", lambda _: None)

    proc = _docker_pull_with_retry("alexgshaw/missing:tag", timeout_sec=30, attempts=3)
    assert proc.returncode == 1
    assert calls["count"] == 1


def test_normalize_tb2_repo_digest_accepts_full_or_bare_digest() -> None:
    digest = "sha256:" + ("a" * 64)
    assert _normalize_tb2_repo_digest(f"alexgshaw/demo@{digest}") == digest
    assert _normalize_tb2_repo_digest(digest.upper()) == digest
    assert _normalize_tb2_repo_digest("not-a-digest") == ""


def test_tb2_image_digest_manifest_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    manifest = tmp_path / "image_digests.json"
    manifest.write_text(
        '{"tasks":{"demo-task":"example/demo-task@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}',
        encoding="utf-8",
    )

    monkeypatch.setenv("KSI_TB2_IMAGE_DIGEST_MANIFEST", str(manifest))

    with pytest.raises(RuntimeError, match="image digest mismatch"):
        _enforce_tb2_image_digest_manifest(
            task=task,
            contract=contract,
            image_acquired_via="pull",
            image_acquired_digest="example/demo-task@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            meta_dir=tmp_path / "meta",
        )


def test_tb2_image_digest_manifest_requires_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    manifest = tmp_path / "image_digests.json"
    manifest.write_text('{"tasks":{}}', encoding="utf-8")

    monkeypatch.setenv("KSI_TB2_IMAGE_DIGEST_MANIFEST", str(manifest))

    with pytest.raises(RuntimeError, match="has no digest entry"):
        _enforce_tb2_image_digest_manifest(
            task=task,
            contract=contract,
            image_acquired_via="pull",
            image_acquired_digest="example/demo-task@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            meta_dir=tmp_path / "meta",
        )


def test_acquire_tb2_image_enforces_digest_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    digest = "sha256:" + ("a" * 64)
    manifest = tmp_path / "image_digests.json"
    manifest.write_text(
        json.dumps({"images": {contract.docker_image: f"{contract.docker_image}@{digest}"}}),
        encoding="utf-8",
    )

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        if cmd[:2] == ["docker", "tag"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["docker", "inspect"] and cmd[-1] == contract.docker_image:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps([{"RepoDigests": [f"{contract.docker_image}@{digest}"]}]),
                stderr="",
            )
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps([{"Id": "sha256:" + ("b" * 64)}]),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_IMAGE_DIGEST_MANIFEST", str(manifest))
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    acquired = _acquire_tb2_image(
        task=task,
        contract=contract,
        image_tag="ksi-tb2-demo:latest",
        meta_dir=meta_dir,
    )

    assert acquired.image_acquired_via == "pull"
    assert acquired.image_acquired_digest == f"{contract.docker_image}@{digest}"
    assert acquired.image_digest_manifest_check["matched"] is True
    assert (meta_dir / "image_digest_manifest_check.json").is_file()


def test_run_terminal_bench_2_trial_requires_real_task_root(tmp_path: Path) -> None:
    task = TaskSpec(
        id="missing-task",
        metadata={
            "task_source": "terminal_bench_2",
            "task_root": str(tmp_path / "does-not-exist"),
        },
    )
    with pytest.raises(ValueError, match="root does not exist"):
        run_terminal_bench_2_trial(task=task, agent_mode="noop")


def _write_tb2_task(tmp_path: Path, task_id: str = "demo-task") -> Path:
    task_root = tmp_path / task_id
    (task_root / "environment").mkdir(parents=True)
    (task_root / "solution").mkdir()
    (task_root / "tests").mkdir()
    (task_root / "instruction.md").write_text("Native task statement.\n", encoding="utf-8")
    (task_root / "task.toml").write_text(
        """\
version = "1.0"

[metadata]
author_name = "Example"
author_email = "example@example.com"
difficulty = "medium"
category = "software-engineering"

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 1200.0

[environment]
docker_image = "example/demo-task:latest"
cpus = 1
memory = "2G"
""",
        encoding="utf-8",
    )
    (task_root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    (task_root / "solution" / "solve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task_root / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    return task_root


def _tb2_task_spec(task_root: Path, task_id: str = "demo-task") -> TaskSpec:
    return TaskSpec(
        id=task_id,
        metadata={
            "task_source": "terminal_bench_2",
            "task_root": str(task_root),
            "task_files": {
                "tb2/instruction.md": "Native task statement.\n",
                "tb2/task.toml": 'version = "1.0"\n',
            },
        },
    )


def test_materialize_terminal_bench_2_workspace_seed_writes_native_files(tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    workspace_root = materialize_terminal_bench_2_workspace_seed(
        task=_tb2_task_spec(task_root),
        output_dir=tmp_path / "out",
    )

    assert not (workspace_root / "INSTRUCTION.md").exists()
    assert not (workspace_root / "TASK.md").exists()
    assert (workspace_root / "TOOLS.md").is_file()
    assert not (workspace_root / "MEMORY.md").exists()
    assert (workspace_root / "tb2" / "instruction.md").read_text(encoding="utf-8") == "Native task statement.\n"
    assert (workspace_root / "tb2" / "task.toml").read_text(encoding="utf-8") == 'version = "1.0"\n'


def test_materialize_terminal_bench_2_workspace_seed_writes_memory_from_seed_package(tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    seed_package = {
        "assigned_task_id": "demo-task",
        "per_task_bundle": {
            "format": "bundle",
            "transferable_insights": [
                {"text": "Back up the disk image before the first boot attempt.", "confidence": "high"}
            ],
        },
    }

    workspace_root = materialize_terminal_bench_2_workspace_seed(
        task=_tb2_task_spec(task_root),
        output_dir=tmp_path / "out",
        seed_package=seed_package,
    )

    memory_path = workspace_root / "MEMORY.md"
    assert memory_path.is_file()
    assert "Back up the disk image before the first boot attempt." in memory_path.read_text(encoding="utf-8")


def test_materialize_terminal_bench_2_workspace_seed_raw_mode_hides_distilled_bundle(tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    seed_package = {
        "assigned_task_id": "demo-task",
        "per_task_bundle": {
            "format": "bundle",
            "transferable_insights": [
                {"text": "Back up the disk image before the first boot attempt.", "confidence": "high"}
            ],
        },
        "prior_attempts": [
            {
                "generation": 1,
                "native_score": 0.0,
                "resolved": False,
                "model_output": "Raw prior attempt: tried mounting /dev/sdb1 directly.",
            }
        ],
    }

    workspace_root = materialize_terminal_bench_2_workspace_seed(
        task=_tb2_task_spec(task_root),
        output_dir=tmp_path / "out",
        seed_package=seed_package,
        raw_mode=True,
    )

    memory_text = (workspace_root / "MEMORY.md").read_text(encoding="utf-8")
    assert "Back up the disk image before the first boot attempt." not in memory_text
    assert "Raw prior attempt: tried mounting /dev/sdb1 directly." in memory_text


def test_materialize_terminal_bench_2_workspace_seed_rejects_path_traversal(tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = TaskSpec(
        id="demo-task",
        metadata={
            "task_source": "terminal_bench_2",
            "task_root": str(task_root),
            "task_files": {"../escape.txt": "nope"},
        },
    )

    with pytest.raises(ValueError, match="unsafe TB2 task_files path"):
        materialize_terminal_bench_2_workspace_seed(task=task, output_dir=tmp_path / "out")


def test_materialize_terminal_bench_2_workspace_seed_rejects_absolute_path(tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = TaskSpec(
        id="demo-task",
        metadata={
            "task_source": "terminal_bench_2",
            "task_root": str(task_root),
            "task_files": {"/etc/passwd": "nope"},
        },
    )

    with pytest.raises(ValueError, match="unsafe TB2 task_files path"):
        materialize_terminal_bench_2_workspace_seed(task=task, output_dir=tmp_path / "out")


def test_phase_copies_keep_solution_out_of_non_oracle_agent_phase(tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    contract = resolve_terminal_bench_2_task_contract(_tb2_task_spec(task_root))

    assert _agent_phase_copies(contract=contract, agent_mode="noop") == []
    assert _agent_phase_copies(contract=contract, agent_mode="oracle") == [(task_root / "solution", "/solution")]
    assert _verifier_phase_copies(contract=contract) == [(task_root / "tests", "/tests")]


def test_verifier_sanitize_paths_covers_copies_and_reward(tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    contract = resolve_terminal_bench_2_task_contract(_tb2_task_spec(task_root))

    paths = _verifier_sanitize_paths(contract=contract)
    # Every verifier copy destination is wiped (so docker-cp can never nest the
    # official tests under an agent-created dir), plus the reward and CTRF
    # output files (both agent-writable under /logs/verifier, #1143).
    for _, dst in _verifier_phase_copies(contract=contract):
        assert dst in paths
    assert _TB2_CONTAINER_REWARD_PATH in paths
    assert _TB2_CONTAINER_CTRF_PATH in paths
    assert "/tests" in paths


def _sequence_index(recorded: list[list[str]], predicate) -> int:
    for idx, cmd in enumerate(recorded):
        if predicate(cmd):
            return idx
    raise AssertionError(f"no recorded command matched predicate; commands={recorded}")


def _is_reward_readout(cmd: list[str]) -> bool:
    """True for the #1186 pre-removal `docker exec` reward-readout command."""
    return cmd[:2] == ["docker", "exec"] and _TB2_REWARD_READOUT_SENTINEL in " ".join(cmd)


def _reward_readout_stdout(reward_host_path: Path, *, active: bool = False) -> str:
    """Build the stdout that `_read_tb2_reward_before_removal` parses.

    Simulates a `docker exec` read of the container reward file (which, on the
    real path, IS the host `/logs` bind mount `reward_host_path`). A finished
    verifier yields identical (mtime, content) samples; an ``active`` background
    writer advances the integer-second mtime between the two samples so the read
    is flagged untrusted.
    """
    sep = "\x1e"
    if not reward_host_path.is_file():
        return "ABSENT"
    content = reward_host_path.read_text(encoding="utf-8").strip()
    m1, m2 = ("100", "200") if active else ("100", "100")
    return f"PRESENT{sep}{m1}{sep}{m2}{sep}{content}{sep}{content}"


def _reward_readout_process(cmd: list[str], reward_host_path: Path, *, active: bool = False):
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=0,
        stdout=_reward_readout_stdout(reward_host_path, active=active),
        stderr="",
    )


def test_verifier_sanitize_runs_before_test_copy_and_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression: an agent that pre-creates /tests/test.sh and pre-writes
    /logs/verifier/reward.txt=1.0 must NOT be able to force resolved=True.

    The trial must issue `rm -rf /tests /logs/verifier/reward.txt` (via
    docker exec) BEFORE it copies the official tests in and BEFORE it runs
    `bash /tests/test.sh`, so the official test.sh lands authoritatively and a
    pre-planted reward cannot survive.

    A full docker-in-test is too heavy for CI, so this drives the trial with a
    recording fake `_run` and asserts the command ordering. See the module
    docstring on `_verifier_sanitize_paths` for the manual docker repro.
    """
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    recorded: list[list[str]] = []

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        recorded.append(list(cmd))
        # Simulate the agent pre-planting a winning reward on /logs (a
        # read-write bind mount) the moment the container starts.
        if cmd[:2] == ["docker", "run"]:
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        # The sanitize step wipes container paths; /logs is host-visible, so
        # honor the reward removal against the host mount.
        if cmd[:2] == ["docker", "exec"] and any("reward.txt" in part for part in cmd) and "rm -rf" in " ".join(cmd):
            reward_host_path.unlink(missing_ok=True)
        # The official verifier writes the REAL reward. Here the honest test
        # fails, so it writes 0.0 (not the pre-planted 1.0).
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("0.0\n", encoding="utf-8")
        # #1186: the trial reads the reward from inside the container BEFORE rm.
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    def is_sanitize(cmd: list[str]) -> bool:
        return (
            cmd[:2] == ["docker", "exec"]
            and "rm -rf" in " ".join(cmd)
            and any(part == "/tests" or "/tests " in part or part.endswith("/tests") for part in cmd)
            and any("reward.txt" in part for part in cmd)
        )

    def is_verifier_copy(cmd: list[str]) -> bool:
        return cmd[:2] == ["docker", "cp"] and cmd[-1].endswith(":/tests")

    def is_verifier_run(cmd: list[str]) -> bool:
        return cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd)

    sanitize_idx = _sequence_index(recorded, is_sanitize)
    copy_idx = _sequence_index(recorded, is_verifier_copy)
    run_idx = _sequence_index(recorded, is_verifier_run)
    sanitize_cmd = recorded[sanitize_idx]

    # Ordering: rm -rf sanitize -> copy official tests -> run official test.sh.
    assert sanitize_idx < copy_idx, "sanitize must run before the official tests are copied in"
    assert sanitize_idx < run_idx, "sanitize must run before bash /tests/test.sh"
    assert sanitize_cmd[3] == "/bin/bash"
    assert sanitize_cmd[-1].startswith("/bin/rm -rf ")

    # The pre-planted 1.0 was wiped; the trial scores the official 0.0 only.
    assert result.reward == 0.0
    assert result.resolved is False


def test_cleanup_timeout_does_not_mask_completed_trial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """#1223: a `docker rm` cleanup that stalls (raising `TimeoutExpired`) runs
    in the trial's `finally`; it must NOT propagate out and mask the completed
    verifier outcome or prevent final trial metadata from being written.

    The trial must still return the scored verifier result, record the cleanup
    failure in `runtime_meta["cleanup_error"]`, and not raise.
    """
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        # The `finally` cleanup `docker rm -f` of the trial container hangs and
        # times out. (The trusted-bash extraction also removes a short-lived
        # `...-trustedbash-...` temp container -- that one must still succeed.)
        if cmd[:2] == ["docker", "rm"] and "trustedbash" not in cmd[-1]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_sec or 0)
        # The sanitize step wipes any pre-planted reward on the host mount.
        if cmd[:2] == ["docker", "exec"] and any("reward.txt" in part for part in cmd) and "rm -rf" in " ".join(cmd):
            reward_host_path.unlink(missing_ok=True)
        # The official verifier writes a genuine solve (1.0).
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        # #1186: the trial reads the reward from inside the container before rm.
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    # No exception must propagate out of the trial's `finally` cleanup.
    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root))

    # The completed verifier outcome is preserved despite the cleanup timeout.
    assert result.reward == 1.0
    assert result.resolved is True
    assert result.runtime_meta["trial_status"] == "completed"
    # The cleanup failure is recorded, not swallowed silently.
    assert result.runtime_meta["cleanup_error"], "docker rm timeout must be recorded in cleanup_error"
    # Final trial metadata was written to disk.
    assert (output_root / "meta" / "trial_result.json").is_file()


def test_verifier_invoked_via_trusted_bash_not_planted_shim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """#1114: a root agent that plants /usr/local/bin/bash (which precedes /bin
    on PATH) must NOT hijack the verifier entrypoint.

    The trial launches the official test.sh through an ABSOLUTE verifier-owned
    bash extracted from the pristine image, with that verifier dir prepended to
    PATH -- so a planted `bash` shim is neither the entrypoint interpreter nor
    the first `bash` on PATH. This drives the trial with a recording fake `_run`
    (no real Docker) and asserts the invocation shape.
    """
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    recorded: list[list[str]] = []

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        recorded.append(list(cmd))
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    # The pristine bash is extracted from the image (docker create + docker cp of
    # :/bin/bash) and injected into a fresh verifier-owned dir before the run.
    def is_bash_extraction(cmd: list[str]) -> bool:
        return cmd[:2] == ["docker", "cp"] and cmd[2].endswith(":/bin/bash")

    assert _sequence_index(recorded, is_bash_extraction) >= 0
    setup_cmd = recorded[
        _sequence_index(
            recorded,
            lambda cmd: cmd[:2] == ["docker", "exec"] and "/bin/mkdir -p" in " ".join(cmd),
        )
    ]
    assert setup_cmd[3] == "/bin/bash"
    assert "/bin/rm -rf" in setup_cmd[-1]
    assert "/bin/mkdir -p" in setup_cmd[-1]

    def is_verifier_run(cmd: list[str]) -> bool:
        return cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd)

    verifier_cmd = recorded[_sequence_index(recorded, is_verifier_run)]
    entrypoint = verifier_cmd[3]
    # The entrypoint interpreter is an ABSOLUTE trusted bash, never bare "bash"
    # (which PATH would resolve to a planted /usr/local/bin/bash shim).
    assert entrypoint != "bash", f"verifier entrypoint must not be PATH-resolved bash: {verifier_cmd}"
    assert entrypoint.startswith("/") and entrypoint.endswith("/bash"), verifier_cmd
    trusted_dir = entrypoint.rsplit("/", 1)[0]
    joined = " ".join(verifier_cmd)
    # test.sh is run through that same absolute interpreter, with the trusted dir
    # prepended to PATH so a planted bare `bash` inside test.sh also loses.
    assert f"exec {entrypoint} /tests/test.sh" in joined, verifier_cmd
    assert f"export PATH={trusted_dir}:" in joined, verifier_cmd
    assert "/usr/local/bin" not in trusted_dir  # the trusted dir is verifier-owned, not agent-writable

    assert result.runtime_meta["verifier_trusted_toolchain"] is True
    assert result.reward == 1.0
    assert result.resolved is True


def _is_verifier_run(cmd: list[str]) -> bool:
    """True for the entrypoint that runs the official /tests/test.sh."""
    return cmd[:2] == ["docker", "exec"] and "/tests/test.sh" in " ".join(cmd)


def test_strict_mode_fails_closed_when_trusted_toolchain_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#1206: trusted-verifier default makes an untrusted toolchain UNSCORED.

    When the trusted-bash injection cannot take effect (here: the pristine-image
    ``docker create`` used to extract bash fails), the default path must REFUSE
    the legacy PATH-resolved ``bash -c "bash /tests/test.sh"`` invocation
    entirely rather than run it and silently record an equivalent-to-main
    verdict. The trial is left unscored (reward ``None``) and flagged via
    ``verifier_fail_closed``. A reward the attacker planted in the container is
    never read.
    """
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    recorded: list[list[str]] = []

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        recorded.append(list(cmd))
        # Simulate trusted-bash extraction failure.
        if cmd[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="no trusted bash")
        # If the legacy verifier ran (it must NOT under strict mode), it would
        # write a planted winning reward -- the assertions below catch that.
        if _is_verifier_run(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")
    monkeypatch.delenv("KSI_TB2_REQUIRE_TRUSTED_VERIFIER", raising=False)

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    # The verifier must never have run -- neither the trusted nor the legacy form.
    assert not any(_is_verifier_run(cmd) for cmd in recorded), (
        f"strict mode ran the verifier despite an untrusted toolchain: {recorded}"
    )
    assert result.reward is None
    assert result.resolved is False
    assert result.runtime_meta["verifier_trusted_toolchain"] is False
    assert result.runtime_meta["require_trusted_verifier"] is True
    assert result.runtime_meta["verifier_fail_closed"] is True
    assert result.runtime_meta["trial_status"] == "verifier_fail_closed_untrusted_toolchain"


def test_strict_mode_fails_closed_when_trusted_toolchain_extraction_times_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    recorded: list[list[str]] = []

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        recorded.append(list(cmd))
        if cmd[:2] == ["docker", "create"]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_sec or 0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")
    monkeypatch.delenv("KSI_TB2_REQUIRE_TRUSTED_VERIFIER", raising=False)

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    assert not any(_is_verifier_run(cmd) for cmd in recorded)
    assert result.reward is None
    assert result.resolved is False
    assert result.runtime_meta["verifier_fail_closed"] is True
    assert result.runtime_meta["trial_status"] == "verifier_fail_closed_untrusted_toolchain"
    assert "timed out" in result.runtime_meta["verifier_trusted_bash_detail"].lower()


def test_strict_mode_still_scores_when_toolchain_is_trusted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """#1206: default strict mode is a no-op on the trusted path -- a legit solve still
    scores when the trusted-bash injection succeeds (default fake `_run`)."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        if _is_verifier_run(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")
    monkeypatch.delenv("KSI_TB2_REQUIRE_TRUSTED_VERIFIER", raising=False)

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    assert result.runtime_meta["verifier_trusted_toolchain"] is True
    assert result.runtime_meta["verifier_fail_closed"] is False
    assert result.reward == 1.0
    assert result.resolved is True


def test_forced_fallback_runs_legacy_when_strict_mode_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#1206: explicit legacy mode preserves the never-worse-than-main fallback.

    An untrusted toolchain still runs the legacy verifier and scores only when
    KSI_TB2_REQUIRE_TRUSTED_VERIFIER is deliberately set false.
    """
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    recorded: list[list[str]] = []

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        recorded.append(list(cmd))
        if cmd[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="no trusted bash")
        if _is_verifier_run(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("0.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")
    monkeypatch.setenv("KSI_TB2_REQUIRE_TRUSTED_VERIFIER", "0")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    # The legacy PATH-resolved invocation still runs in explicit legacy mode.
    legacy_idx = _sequence_index(
        recorded,
        lambda cmd: cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd),
    )
    assert legacy_idx >= 0
    assert result.runtime_meta["verifier_trusted_toolchain"] is False
    assert result.runtime_meta["require_trusted_verifier"] is False
    assert result.runtime_meta["verifier_fail_closed"] is False
    assert result.reward == 0.0
    assert result.resolved is False


def test_legitimate_solve_still_scores_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Normal-path guard: when the official verifier writes reward.txt=1.0, the
    sanitize step (a no-op on the honest path) does not break scoring."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    assert result.reward == 1.0
    assert result.resolved is True


def test_verifier_never_kills_container_processes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The trial must NEVER issue a process-kill/sweep before the verifier.

    An unconditional pre-verifier ``kill -9`` of container processes destroys
    agent-STARTED services that legitimate task verifiers probe while running
    (~6/89 tasks: kv-store-grpc's gRPC server, nginx-request-logging's nginx,
    hf-model-inference's Flask API, install-windows-3.11 / qemu-startup /
    qemu-alpine-ssh's VMs), turning honest solves into ``resolved=False``. The
    resident-process reward-replant is a KNOWN, ACCEPTED residual (deferred to
    the pristine verifier in #1114; threat model in #1174). This guard fails if
    any process-killing command is ever re-introduced.
    """
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    recorded: list[list[str]] = []

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        recorded.append(list(cmd))
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    for cmd in recorded:
        joined = " ".join(cmd)
        assert "kill -9" not in joined, f"trial must not kill container processes: {joined}"
        assert not (cmd[:2] == ["docker", "exec"] and "/proc/[0-9]" in joined and "kill" in joined), (
            f"trial must not sweep container processes: {joined}"
        )

    # The agent-started service survives (no sweep), so the honest solve scores.
    assert result.reward == 1.0
    assert result.resolved is True


def test_nonzero_verifier_with_reward_reports_verifier_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        if _is_verifier_run(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("0.0\n", encoding="utf-8")
            return subprocess.CompletedProcess(args=cmd, returncode=7, stdout="", stderr="tests failed")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    assert result.runtime_meta["trial_status"] == "verifier_failed"
    assert result.runtime_meta["verifier_exit_code"] == 7
    assert result.reward == 0.0
    assert result.resolved is False


def test_bridge_failure_with_passing_verifier_is_not_completed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A ksi-bridge exception is caught (agent_exit_code=1) and is NOT terminal:
    the verifier still runs. Such a crashed agent phase must report
    ``trial_status='agent_failed_but_verifier_ran'`` -- never 'completed' -- so
    it can't be silently counted as a clean solve. Conservative: ``resolved``
    still reflects the real reward, so a genuine solve then a late crash scores.
    """
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def boom(**_kwargs):
        raise RuntimeError("tb2 bridge exploded mid-run")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        # The official verifier passes (reward 1.0) despite the agent crash.
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run_ksi_agent_in_tb2_container", boom)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(
        task=task,
        agent_mode="ksi",
        output_dir=str(output_root),
        keep_container=True,
        provider_env={"MODEL": "haiku"},
    )

    assert result.runtime_meta["trial_status"] == "agent_failed_but_verifier_ran"
    assert result.runtime_meta["agent_exit_code"] == 1
    # Conservative: real passing reward is preserved (late-crash solves survive).
    assert result.reward == 1.0
    assert result.resolved is True


def test_clean_run_reads_completed_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A clean agent phase (exit 0) with a passing verifier still reads
    ``trial_status='completed'`` -- the new agent-failed branch must not fire
    when the agent succeeded."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    assert result.runtime_meta["trial_status"] == "completed"
    assert result.runtime_meta["agent_exit_code"] == 0
    assert result.reward == 1.0
    assert result.resolved is True


def test_trial_fails_closed_when_purge_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If the pre-verifier purge (sanitize `rm -rf`) fails, the whole trial must
    fail closed -- raise, never return a passing/0.0 result off a possibly
    pre-planted reward. The raise propagates to the engine as an unscored (None)
    infra failure. Previously only the sanitize helper's nonzero exit -> raise
    was implicit; this pins it at the whole-trial level."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        joined = " ".join(cmd)
        # An agent pre-planted a winning reward; if the purge fails to wipe it,
        # the trial must NOT go on to score it.
        if cmd[:2] == ["docker", "run"]:
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        # The sanitize/purge step fails.
        if cmd[:2] == ["docker", "exec"] and "rm -rf" in joined and "reward.txt" in joined:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="rm: cannot remove")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    with pytest.raises(RuntimeError, match="sanitize"):
        run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)


def test_neither_reward_file_present_scores_none_without_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Empty verifier logs (the verifier ran but wrote neither reward.txt nor
    ctrf.json) must not raise and must score None (unscored) -- never a
    fabricated 0.0/pass. The pre-removal readout reports the reward absent."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        # The verifier runs but produces NO reward file at all.
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, reward_host_path)  # absent -> "ABSENT"
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    assert result.reward is None
    assert result.resolved is False
    assert result.runtime_meta["trial_status"] == "verifier_did_not_produce_reward"
    assert result.runtime_meta["reward_active_writer_detected"] is False
    assert result.runtime_meta["ctrf_path"] == ""


def test_reward_scored_from_pre_removal_read_not_post_read_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#1186 regression: the recorded reward comes from the pre-removal
    `docker exec` read, so a value written to /logs AFTER that read (a racing
    writer's late write, or any post-verifier host-side mutation) does not
    change the recorded score.

    The honest verifier writes 0.0 (fail); the readout snapshots 0.0; then a
    racing writer overwrites the host mount to 1.0. Under the OLD behavior the
    host was read only AFTER `docker rm -f`, so the score would have been the
    racing 1.0 (a false pass). The fix scores the snapshotted 0.0."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("0.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            # Snapshot the honest 0.0 the grader just wrote (stable read)...
            proc = _reward_readout_process(cmd, reward_host_path)
            # ...then simulate the backgrounded writer overwriting the host mount
            # right after our snapshot (the exact write that used to win).
            reward_host_path.write_text("1.0\n", encoding="utf-8")
            return proc
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    # The post-read host write to 1.0 is ignored; the pre-removal snapshot wins.
    assert reward_host_path.read_text(encoding="utf-8").strip() == "1.0"
    assert result.reward == 0.0
    assert result.resolved is False
    assert result.runtime_meta["reward_source"] == "container_exec_pre_removal"


def test_active_writer_during_verify_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """#1186 fail-closed: if a background writer is STILL racing the grader when
    the reward is snapshotted (the two in-container samples disagree on mtime or
    content), the reward is untrusted and the trial records no score -- never a
    pass -- even though a '1.0' is sitting in the file."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    output_root = tmp_path / "out"
    reward_host_path = output_root / "logs" / "verifier" / "reward.txt"

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_host_path.parent.mkdir(parents=True, exist_ok=True)
            reward_host_path.write_text("1.0\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            # active=True -> the two samples' mtimes differ -> writer still racing.
            return _reward_readout_process(cmd, reward_host_path, active=True)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root), keep_container=True)

    assert result.reward is None
    assert result.resolved is False
    assert result.runtime_meta["reward_active_writer_detected"] is True
    assert result.runtime_meta["trial_status"] == "verifier_did_not_produce_reward"


def _raw_readout(m1: str, m2: str, v1: str, v2: str) -> str:
    sep = "\x1e"
    return f"PRESENT{sep}{m1}{sep}{m2}{sep}{v1}{sep}{v2}"


def test_read_reward_before_removal_parses_stable_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _read_tb2_reward_before_removal

    fake = _fake_docker_exec_factory(returncode=0, stdout=_raw_readout("100", "100", "1.0", "1.0"))
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    readout = _read_tb2_reward_before_removal(container_name="dummy")
    assert readout.present is True
    assert readout.active_writer is False
    assert readout.reward == 1.0
    # The command carries the sentinel and reads the container reward path.
    assert _TB2_REWARD_READOUT_SENTINEL in fake.calls[0]["command"]
    assert _TB2_CONTAINER_REWARD_PATH in fake.calls[0]["command"]


def test_read_reward_before_removal_flags_mtime_advance(monkeypatch: pytest.MonkeyPatch) -> None:
    """A background writer that keeps re-writing the SAME value is still caught:
    the integer-second mtime advances between the two samples."""
    from ksi.benchmarks.terminal_bench_2_runtime import _read_tb2_reward_before_removal

    fake = _fake_docker_exec_factory(returncode=0, stdout=_raw_readout("100", "101", "1.0", "1.0"))
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    readout = _read_tb2_reward_before_removal(container_name="dummy")
    assert readout.active_writer is True
    assert readout.reward is None


def test_read_reward_before_removal_flags_content_change(monkeypatch: pytest.MonkeyPatch) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _read_tb2_reward_before_removal

    fake = _fake_docker_exec_factory(returncode=0, stdout=_raw_readout("100", "100", "0.0", "1.0"))
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    readout = _read_tb2_reward_before_removal(container_name="dummy")
    assert readout.active_writer is True
    assert readout.reward is None


def test_read_reward_before_removal_absent_is_not_active_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _read_tb2_reward_before_removal

    fake = _fake_docker_exec_factory(returncode=0, stdout="ABSENT")
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    readout = _read_tb2_reward_before_removal(container_name="dummy")
    assert readout.present is False
    assert readout.active_writer is False
    assert readout.reward is None


def test_read_reward_before_removal_exec_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _read_tb2_reward_before_removal

    fake = _fake_docker_exec_factory(returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    readout = _read_tb2_reward_before_removal(container_name="dummy")
    assert readout.active_writer is True
    assert readout.reward is None


def test_resolve_tb2_max_steps_defaults_to_unlimited(tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    contract = resolve_terminal_bench_2_task_contract(_tb2_task_spec(task_root))

    # Default is unlimited: the canonical Harbor harness applies no step cap,
    # and 44% of TB2 tasks declare agent.timeout_sec > 900s. A swarms-side
    # 150-step cap can bind before the wall-time and bias scores downward.
    from ksi.benchmarks.terminal_bench_2_runtime import _TB2_STEP_CAP_UNLIMITED

    assert _resolve_tb2_max_steps(contract=contract, env={}) == _TB2_STEP_CAP_UNLIMITED
    # Explicit positive override is honored.
    assert _resolve_tb2_max_steps(contract=contract, env={"KSI_TB2_MAX_STEPS": "7"}) == 7
    # Zero (and any non-positive) also means unlimited.
    assert _resolve_tb2_max_steps(contract=contract, env={"KSI_TB2_MAX_STEPS": "0"}) == _TB2_STEP_CAP_UNLIMITED
    assert _resolve_tb2_max_steps(contract=contract, env={"KSI_TB2_MAX_STEPS": "-1"}) == _TB2_STEP_CAP_UNLIMITED
    # Invalid or empty falls back to unlimited (the new default).
    assert _resolve_tb2_max_steps(contract=contract, env={"KSI_TB2_MAX_STEPS": "abc"}) == _TB2_STEP_CAP_UNLIMITED
    assert _resolve_tb2_max_steps(contract=contract, env={"KSI_TB2_MAX_STEPS": ""}) == _TB2_STEP_CAP_UNLIMITED


def test_resolve_tb2_max_steps_falls_back_to_os_environ(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When the provider env omits the knob, a process-level export is honored
    (matching every other KSI_TB2_* knob)."""
    task_root = _write_tb2_task(tmp_path)
    contract = resolve_terminal_bench_2_task_contract(_tb2_task_spec(task_root))

    monkeypatch.setenv("KSI_TB2_MAX_STEPS", "50")
    assert _resolve_tb2_max_steps(contract=contract, env={}) == 50


def test_resolve_tb2_max_steps_provider_env_wins_over_os_environ(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The provider profile takes precedence over the process-level export."""
    task_root = _write_tb2_task(tmp_path)
    contract = resolve_terminal_bench_2_task_contract(_tb2_task_spec(task_root))

    monkeypatch.setenv("KSI_TB2_MAX_STEPS", "50")
    assert _resolve_tb2_max_steps(contract=contract, env={"KSI_TB2_MAX_STEPS": "7"}) == 7


def test_inspect_image_identity_extracts_digest_and_id() -> None:
    import json as _json

    from ksi.benchmarks.terminal_bench_2_runtime import _inspect_image_identity

    def fake_run(cmd, *, timeout_sec):
        target = cmd[-1]
        if target == "alexgshaw/foo:tag":
            payload = [{"RepoDigests": ["alexgshaw/foo@sha256:abc123"], "Id": "sha256:def456"}]
        elif target == "ksi-tb2-foo:local":
            payload = [{"Id": "sha256:def456", "RepoDigests": []}]
        else:
            payload = []
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=_json.dumps(payload), stderr="")

    import ksi.benchmarks.terminal_bench_2_runtime as runtime_mod

    original_run = runtime_mod._run
    runtime_mod._run = fake_run
    try:
        digest, image_id = _inspect_image_identity(pull_target="alexgshaw/foo:tag", image_tag="ksi-tb2-foo:local")
    finally:
        runtime_mod._run = original_run

    assert digest == "alexgshaw/foo@sha256:abc123"
    assert image_id == "sha256:def456"


def test_inspect_image_identity_build_path_has_empty_digest() -> None:
    import json as _json

    from ksi.benchmarks.terminal_bench_2_runtime import _inspect_image_identity

    def fake_run(cmd, *, timeout_sec):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_json.dumps([{"Id": "sha256:locallybuilt", "RepoDigests": []}]),
            stderr="",
        )

    import ksi.benchmarks.terminal_bench_2_runtime as runtime_mod

    original_run = runtime_mod._run
    runtime_mod._run = fake_run
    try:
        digest, image_id = _inspect_image_identity(pull_target="", image_tag="ksi-tb2-foo:local")
    finally:
        runtime_mod._run = original_run

    assert digest == ""
    assert image_id == "sha256:locallybuilt"


def test_inspect_image_identity_swallows_failures() -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _inspect_image_identity

    def fake_run(cmd, *, timeout_sec):
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="error: no such image")

    import ksi.benchmarks.terminal_bench_2_runtime as runtime_mod

    original_run = runtime_mod._run
    runtime_mod._run = fake_run
    try:
        digest, image_id = _inspect_image_identity(pull_target="x", image_tag="y")
    finally:
        runtime_mod._run = original_run

    assert digest == ""
    assert image_id == ""


def test_require_pull_rejects_when_disable_pull_also_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")
    monkeypatch.setenv("KSI_TB2_DISABLE_PULL", "1")

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(tmp_path / "out"))


def test_image_acquisition_log_goes_to_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)

    # Pretend the image is already present so the trial proceeds past the
    # docker pull/build branch and emits the "[tb2] image ..." line.
    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec: float, **_kwargs):
        # Tag, run, exec, rm — everything else after the pull is fine to no-op.
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    # The trial will fail downstream (no real container), but the image-log
    # line should have been emitted to stderr before that. Catch broadly and
    # assert only on capture output.
    try:
        run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(tmp_path / "out"))
    except Exception:
        pass

    captured = capsys.readouterr()
    assert "[tb2] image task=demo-task via=pull" in captured.err
    assert "[tb2] image task=" not in captured.out


def test_require_pull_failed_pull_refuses_local_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Required-pull failure uses the typed policy and never builds locally."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    expected_image = resolve_terminal_bench_2_task_contract(task).docker_image
    secret = "Bearer secret-token-value-1234567890"
    pull_attempts: list[int] = []

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        pull_attempts.append(attempts)
        return subprocess.CompletedProcess(
            args=["docker", "pull", target],
            returncode=1,
            stdout="",
            stderr=f"manifest unknown\nAuthorization: {secret}\x1b[31m",
        )

    recorded: list[list[str]] = []

    def fake_run(cmd, *, timeout_sec: float, **_kwargs):
        recorded.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    output_root = tmp_path / "out"
    with pytest.raises(ContainerRegistryError, match="refusing to fall back") as caught:
        run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(output_root))

    error = caught.value
    assert error.retryable is False
    assert error.reason == "non_transient"
    assert error.image == expected_image
    assert pull_attempts == [1]
    assert secret not in str(error)
    assert "[REDACTED]" in str(error)
    assert "\n" not in str(error)
    assert "\x1b" not in str(error)
    assert secret in (output_root / "meta" / "docker_pull.stderr.txt").read_text(encoding="utf-8")
    assert not any(cmd[:2] == ["docker", "build"] for cmd in recorded)


@pytest.mark.parametrize(
    "stderr",
    [
        "unauthorized: authentication required",
        "HTTP 401 Unauthorized",
        "received unexpected HTTP status: 500 Internal Server Error",
    ],
)
def test_require_pull_ambiguous_registry_failure_is_retryable_at_real_raise_site(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stderr: str,
) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    expected_image = resolve_terminal_bench_2_task_contract(task).docker_image
    pull_attempts: list[int] = []

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        pull_attempts.append(attempts)
        return subprocess.CompletedProcess(
            args=["docker", "pull", target],
            returncode=1,
            stdout="",
            stderr=stderr,
        )

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setenv("KSI_TB2_REQUIRE_PULL", "1")

    with pytest.raises(ContainerRegistryError) as caught:
        run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(tmp_path / "out"))

    assert caught.value.retryable is True
    assert caught.value.reason == "transient"
    assert caught.value.image == expected_image
    assert pull_attempts == [1]


def test_run_terminal_bench_2_trial_records_task_toml_timeout_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    logs_root_seen: dict[str, Path] = {}

    def fake_pull(target: str, *, timeout_sec: float, attempts: int = 3) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker", "pull", target], returncode=0, stdout="", stderr="")

    def fake_run_command(**kwargs):
        logs_root_seen["path"] = kwargs["logs_root"]
        return _docker_run_command(**kwargs)

    def fake_run(cmd, *, timeout_sec: float | None = None, **_kwargs):
        # #1114: test.sh is now launched via a trusted absolute bash, so match
        # the substring rather than the exact legacy `bash /tests/test.sh` argv.
        if cmd[:2] == ["docker", "exec"] and "bash /tests/test.sh" in " ".join(cmd):
            reward_dir = logs_root_seen["path"] / "verifier"
            reward_dir.mkdir(parents=True, exist_ok=True)
            (reward_dir / "reward.txt").write_text("1\n", encoding="utf-8")
        if _is_reward_readout(cmd):
            return _reward_readout_process(cmd, logs_root_seen["path"] / "verifier" / "reward.txt")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_pull_with_retry", fake_pull)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_run_command", fake_run_command)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)

    result = run_terminal_bench_2_trial(task=task, agent_mode="noop", output_dir=str(tmp_path / "out"))

    assert result.runtime_meta["timeout_source"] == "task.toml"
    assert result.runtime_meta["agent_timeout_sec"] == 1200.0
    assert result.runtime_meta["verifier_timeout_sec"] == 900.0
    assert result.reward == 1.0


def test_docker_run_command_mounts_only_logs_and_workspace() -> None:
    cmd = _docker_run_command(
        image_tag="demo:image",
        container_name="demo-container",
        logs_root=Path("/tmp/logs"),
        workspace_root=Path("/tmp/workspace"),
        cpus=1.0,
        memory="2G",
    )
    rendered = " ".join(str(part) for part in cmd)
    assert "/tmp/logs:/logs" in rendered
    assert "/tmp/workspace:/workspace/task/workspace" in rendered
    assert "/solution" not in rendered
    assert "/tests" not in rendered


def test_terminal_bench_2_executor_wraps_trial_result(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskSpec(id="demo-task", metadata={"task_source": "terminal_bench_2", "task_root": "/tmp/demo-task"})

    class _FakeResult:
        reward = 1.0
        agent_exit_code = 0
        verifier_exit_code = 0
        runtime_meta = {"trial_status": "completed"}
        tool_trace = [{"type": "tool_call", "tool_name": "tb2_shell"}]
        token_usage = TokenUsage(input_tokens=11, output_tokens=7)

    def fake_run_terminal_bench_2_trial(**kwargs):
        assert kwargs["task"] is task
        assert kwargs["agent_mode"] == "noop"
        assert kwargs["generation"] == 3
        assert kwargs["agent_id"] == "agent-7"
        return _FakeResult()

    monkeypatch.setattr(
        "ksi.runtime.terminal_bench_2.run_terminal_bench_2_trial",
        fake_run_terminal_bench_2_trial,
    )

    executor = TerminalBench2Executor(agent_mode="noop")
    result = executor.run_task(generation=3, agent_id="agent-7", task=task)

    assert "reward=1.0" in result.output
    assert result.runtime_meta["generation"] == 3
    assert result.runtime_meta["agent_id"] == "agent-7"
    assert result.runtime_meta["task_id"] == "demo-task"
    assert result.runtime_meta["runner"] == "terminal_bench_2_executor"
    assert result.tool_trace == [{"type": "tool_call", "tool_name": "tb2_shell"}]
    assert result.token_usage.total == 18


def test_terminal_bench_2_executor_forwards_agent_seed_package(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskSpec(id="demo-task", metadata={"task_source": "terminal_bench_2", "task_root": "/tmp/demo-task"})
    seed_package = {"per_task_bundle": {"format": "bundle", "transferable_insights": []}}

    class _FakeResult:
        reward = 1.0
        agent_exit_code = 0
        verifier_exit_code = 0
        runtime_meta = {"trial_status": "completed"}
        tool_trace = []
        token_usage = TokenUsage(input_tokens=1, output_tokens=1)

    captured: dict[str, object] = {}

    def fake_run_terminal_bench_2_trial(**kwargs):
        captured.update(kwargs)
        return _FakeResult()

    monkeypatch.setattr(
        "ksi.runtime.terminal_bench_2.run_terminal_bench_2_trial",
        fake_run_terminal_bench_2_trial,
    )

    executor = TerminalBench2Executor(agent_mode="noop")
    executor.run_task(generation=1, agent_id="agent-0", task=task, agent_seed_package=seed_package)

    assert captured["seed_package"] is seed_package


def test_terminal_bench_2_executor_forwards_memory_seed_raw_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskSpec(id="demo-task", metadata={"task_source": "terminal_bench_2", "task_root": "/tmp/demo-task"})

    class _FakeResult:
        reward = 1.0
        agent_exit_code = 0
        verifier_exit_code = 0
        runtime_meta = {"trial_status": "completed"}
        tool_trace = []
        token_usage = TokenUsage(input_tokens=1, output_tokens=1)

    captured: dict[str, object] = {}

    def fake_run_terminal_bench_2_trial(**kwargs):
        captured.update(kwargs)
        return _FakeResult()

    monkeypatch.setattr(
        "ksi.runtime.terminal_bench_2.run_terminal_bench_2_trial",
        fake_run_terminal_bench_2_trial,
    )

    executor = TerminalBench2Executor(agent_mode="noop", memory_seed_raw_attempts=True)
    executor.run_task(generation=1, agent_id="agent-0", task=task)

    assert captured["raw_mode"] is True


def test_terminal_bench_2_executor_delegates_non_tb2_tasks() -> None:
    forum_task = TaskSpec(id="__forum__g1_r0_agent-0", metadata={"task_source": "per_task_forum"})
    captured: dict[str, object] = {}

    class _FallbackRuntime:
        def run_task(self, **kwargs):
            captured.update(kwargs)
            return RuntimeResult(output="forum ok", runtime_meta={"runner": "fallback"})

        def close(self) -> None:
            captured["closed"] = True

    fallback = _FallbackRuntime()
    executor = TerminalBench2Executor(agent_mode="ksi", fallback_runtime=fallback)

    result = executor.run_task(generation=4, agent_id="agent-9", task=forum_task)

    assert result.output == "forum ok"
    assert captured["generation"] == 4
    assert captured["agent_id"] == "agent-9"
    assert captured["task"] is forum_task

    executor.close()
    assert captured["closed"] is True


def test_run_ksi_agent_in_tb2_container_records_shell_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    workspace_root = materialize_terminal_bench_2_workspace_seed(task=task, output_dir=tmp_path / "out")

    responses = iter(
        [
            (
                '{"action":"shell","command":"cat /workspace/task/workspace/tb2/instruction.md","timeout_sec":5,"summary":"read task"}',
                TokenUsage(input_tokens=3, output_tokens=2),
            ),
            ('{"action":"final","summary":"Read the task and stopped."}', TokenUsage(input_tokens=4, output_tokens=1)),
        ]
    )

    class _FakeCaller:
        def call(self, system: str, user: str, **kwargs):
            assert "Respond with exactly one JSON object" in system
            assert "Treat /workspace/task/workspace as the task specification overlay" in system
            assert "Prefer short mutate-then-check cycles" in system
            assert "Avoid giant here-doc scripts" in system
            assert "make that fix persistent so a fresh shell and the verifier will also pass" in system
            # The step counter + latest observation ride in the varying tail;
            # the stable header/guidance now rides in cache_blocks (issue #1252
            # item 1). Assert against the full rendered prompt the model sees.
            assert "Bridge step" in user
            # Concatenate with NO separator — that is exactly how the LLM APIs
            # join adjacent text blocks, so this is the byte-for-byte prompt the
            # model sees. Reconstructing with a "\n" separator here would paper
            # over any missing inter-block newline (issue #1252 item 1).
            full = "".join(kwargs.get("cache_blocks") or []) + user
            assert "task-spec overlay" in full
            assert "your next command should mutate state or run a verifier-aligned check" in full
            assert "make that behavior persist for a fresh shell and the verifier" in full
            text, usage = next(responses)
            return LLMResponse(text=text, usage=usage)

    def fake_build_llm_caller(**kwargs):
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "gpt-5.4-mini"
        return _FakeCaller()

    def fake_docker_exec(*, container_name: str, command: str, timeout_sec: float):
        assert container_name == "tb2-container"
        assert "instruction.md" in command
        return subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=0,
            stdout="Native task statement.\n",
            stderr="",
        )

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.build_llm_caller", fake_build_llm_caller)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake_docker_exec)

    result = _run_ksi_agent_in_tb2_container(
        task=task,
        contract=contract,
        container_name="tb2-container",
        workspace_root=workspace_root,
        provider_env={"MODEL_PROVIDER": "openai", "MODEL": "gpt-5.4-mini", "KSI_TB2_MAX_STEPS": "3"},
        generation=2,
        agent_id="agent-1",
    )

    assert result.model_output == "Read the task and stopped."
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["tool_name"] == "tb2_shell"
    assert result.token_usage.input_tokens == 7
    assert result.token_usage.output_tokens == 3
    assert "instruction.md" in result.transcript


def test_run_ksi_agent_in_tb2_container_recovers_from_invalid_model_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    workspace_root = materialize_terminal_bench_2_workspace_seed(task=task, output_dir=tmp_path / "out")

    responses = iter(
        [
            ("not valid json", TokenUsage(input_tokens=2, output_tokens=1)),
            ('{"command":"pwd","summary":"check cwd"}', TokenUsage(input_tokens=3, output_tokens=2)),
            ('{"action":"final","summary":"done"}', TokenUsage(input_tokens=1, output_tokens=1)),
        ]
    )

    class _FakeCaller:
        def call(self, system: str, user: str, **kwargs):
            text, usage = next(responses)
            return LLMResponse(text=text, usage=usage)

    def fake_build_llm_caller(**kwargs):
        return _FakeCaller()

    def fake_docker_exec(*, container_name: str, command: str, timeout_sec: float):
        assert command == "pwd"
        return subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=0,
            stdout="/app\n",
            stderr="",
        )

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.build_llm_caller", fake_build_llm_caller)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake_docker_exec)

    result = _run_ksi_agent_in_tb2_container(
        task=task,
        contract=contract,
        container_name="tb2-container",
        workspace_root=workspace_root,
        provider_env={"MODEL_PROVIDER": "openai", "MODEL": "gpt-5.4-mini", "KSI_TB2_MAX_STEPS": "4"},
        generation=2,
        agent_id="agent-1",
    )

    assert result.error_text == ""
    assert result.model_output == "done"
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["tool_input"]["command"] == "pwd"


def test_tb2_trim_oldest_history_drops_oldest_third() -> None:
    history = [{"i": i} for i in range(9)]
    dropped = _tb2_trim_oldest_history(history)
    # Drops ~30% (at least one) from the FRONT, keeping the most-recent steps.
    assert dropped == 3
    assert history == [{"i": i} for i in range(3, 9)]
    # Never drops zero, even for a tiny list.
    tiny = [{"i": 0}, {"i": 1}]
    assert _tb2_trim_oldest_history(tiny) == 1
    assert tiny == [{"i": 1}]


def test_run_ksi_agent_in_tb2_container_trims_history_on_prompt_too_long(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A provider 'prompt is too long' error mid-run must trim history and
    retry the turn, NOT collapse the whole trial and forfeit prior progress."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    workspace_root = materialize_terminal_bench_2_workspace_seed(task=task, output_dir=tmp_path / "out")

    shell = (
        '{"action":"shell","command":"echo step","timeout_sec":5,"summary":"s"}',
        TokenUsage(input_tokens=1, output_tokens=1),
    )
    # 5 shell steps build history (>_TB2_PROMPT_TRIM_MIN_HISTORY_KEEP), then the
    # 6th turn overflows once, then the trim-retry succeeds with a final action.
    responses: list = [shell, shell, shell, shell, shell]
    responses.append(RuntimeError("prompt is too long: 250000 tokens > 200000 maximum"))
    responses.append(('{"action":"final","summary":"done"}', TokenUsage(input_tokens=1, output_tokens=1)))
    it = iter(responses)
    prompt_history_sizes: list[int] = []

    class _FakeCaller:
        def call(self, system: str, user: str, **kwargs):
            prompt_history_sizes.append(
                sum(1 for block in kwargs.get("cache_blocks") or [] if block.startswith("Step "))
            )
            item = next(it)
            if isinstance(item, Exception):
                raise item
            text, usage = item
            return LLMResponse(text=text, usage=usage)

    def fake_docker_exec(*, container_name: str, command: str, timeout_sec: float):
        return subprocess.CompletedProcess(args=["docker", "exec"], returncode=0, stdout="step\n", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.build_llm_caller", lambda **kw: _FakeCaller())
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake_docker_exec)

    result = _run_ksi_agent_in_tb2_container(
        task=task,
        contract=contract,
        container_name="tb2-container",
        workspace_root=workspace_root,
        provider_env={"MODEL_PROVIDER": "openai", "MODEL": "gpt-5.4-mini", "KSI_TB2_MAX_STEPS": "20"},
        generation=1,
        agent_id="agent-1",
    )

    # Recovered rather than collapsing: the overflow was absorbed and the run
    # reached its final answer instead of forfeiting the whole trial. The trim
    # dropped the single oldest step (5 // 3 == 1) from the prompt window only;
    # the persisted audit trace still contains every executed shell step.
    assert result.error_text == ""
    assert result.model_output == "done"
    assert len(result.tool_trace) == 5
    assert prompt_history_sizes[-2:] == [5, 4]


def test_run_ksi_agent_in_tb2_container_prompt_too_long_untrimmable_is_graceful(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When history is too small to trim, an overflow degrades to a graceful
    error result (with progress so far), never an uncaught exception."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    workspace_root = materialize_terminal_bench_2_workspace_seed(task=task, output_dir=tmp_path / "out")

    class _FakeCaller:
        def call(self, system: str, user: str, **kwargs):
            # Overflow on the very first turn (empty history — nothing to trim).
            raise RuntimeError("prompt is too long: 250000 tokens > 200000 maximum")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.build_llm_caller", lambda **kw: _FakeCaller())

    result = _run_ksi_agent_in_tb2_container(
        task=task,
        contract=contract,
        container_name="tb2-container",
        workspace_root=workspace_root,
        provider_env={"MODEL_PROVIDER": "openai", "MODEL": "gpt-5.4-mini", "KSI_TB2_MAX_STEPS": "20"},
        generation=1,
        agent_id="agent-1",
    )

    assert "context" in result.error_text.lower()
    assert result.model_output == result.error_text


def test_run_ksi_agent_in_tb2_container_surfaces_cap_hit_when_no_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    workspace_root = materialize_terminal_bench_2_workspace_seed(task=task, output_dir=tmp_path / "out")

    shell_response = LLMResponse(
        text='{"action":"shell","command":"echo step","timeout_sec":5,"summary":"keep going"}',
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )

    class _FakeCaller:
        def call(self, system: str, user: str, **kwargs):
            return shell_response

    def fake_build_llm_caller(**kwargs):
        return _FakeCaller()

    def fake_docker_exec(*, container_name: str, command: str, timeout_sec: float):
        return subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=0,
            stdout="step\n",
            stderr="",
        )

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.build_llm_caller", fake_build_llm_caller)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake_docker_exec)

    result = _run_ksi_agent_in_tb2_container(
        task=task,
        contract=contract,
        container_name="tb2-container",
        workspace_root=workspace_root,
        provider_env={"MODEL_PROVIDER": "openai", "MODEL": "gpt-5.4-mini", "KSI_TB2_MAX_STEPS": "3"},
        generation=1,
        agent_id="agent-1",
    )

    assert "step cap" in result.error_text
    assert "3" in result.error_text
    assert result.model_output == result.error_text
    assert len(result.tool_trace) == 3


def test_run_ksi_agent_in_tb2_container_surfaces_deadline_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When the wall-clock deadline runs out before the agent emits 'final',
    error_text records the deadline exit (distinct from cap-hit and from
    voluntary termination)."""
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    workspace_root = materialize_terminal_bench_2_workspace_seed(task=task, output_dir=tmp_path / "out")

    # Step the fake clock by an enormous amount on each call, blowing past
    # `deadline = monotonic() + agent_timeout_sec` immediately on the second call.
    clock = iter([0.0, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9])
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.time.monotonic", lambda: next(clock))

    class _FakeCaller:
        def call(self, system: str, user: str, **kwargs):
            return LLMResponse(
                text='{"action":"shell","command":"echo x","timeout_sec":5,"summary":"step"}', usage=TokenUsage()
            )

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.build_llm_caller", lambda **kw: _FakeCaller())

    result = _run_ksi_agent_in_tb2_container(
        task=task,
        contract=contract,
        container_name="tb2-container",
        workspace_root=workspace_root,
        provider_env={"MODEL_PROVIDER": "openai", "MODEL": "gpt-5.4-mini"},
        generation=1,
        agent_id="agent-1",
    )
    assert "agent timeout" in result.error_text
    assert result.model_output == result.error_text


def test_run_ksi_agent_in_tb2_container_preserves_timed_out_command_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_root = _write_tb2_task(tmp_path)
    task = _tb2_task_spec(task_root)
    contract = resolve_terminal_bench_2_task_contract(task)
    workspace_root = materialize_terminal_bench_2_workspace_seed(task=task, output_dir=tmp_path / "out")

    responses = iter(
        [
            (
                '{"action":"shell","command":"python3 /tmp/test_interrupt.py","timeout_sec":5,"summary":"run interrupt test"}',
                TokenUsage(input_tokens=3, output_tokens=2),
            ),
            ('{"action":"final","summary":"stop after timeout"}', TokenUsage(input_tokens=1, output_tokens=1)),
        ]
    )

    class _FakeCaller:
        def call(self, system: str, user: str, **kwargs):
            text, usage = next(responses)
            return LLMResponse(text=text, usage=usage)

    def fake_build_llm_caller(**kwargs):
        return _FakeCaller()

    def fake_docker_exec(*, container_name: str, command: str, timeout_sec: float):
        raise subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=timeout_sec)

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime.build_llm_caller", fake_build_llm_caller)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake_docker_exec)

    result = _run_ksi_agent_in_tb2_container(
        task=task,
        contract=contract,
        container_name="tb2-container",
        workspace_root=workspace_root,
        provider_env={"MODEL_PROVIDER": "openai", "MODEL": "gpt-5.4-mini", "KSI_TB2_MAX_STEPS": "3"},
        generation=2,
        agent_id="agent-1",
    )

    assert result.model_output == "stop after timeout"
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["tool_output"]["exit_code"] == 124
    assert "timed out" in result.tool_trace[0]["tool_output"]["combined_output"]


def test_handle_tb2_edit_rejects_non_unique_old_string(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_edit

    file_content = "x = 1\ny = 1\nz = 1\n"

    def fake_cp_from(*, container_name: str, src: str, dst: Path, timeout_sec: float = 30):
        dst.write_text(file_content, encoding="utf-8")
        return subprocess.CompletedProcess(args=["docker", "cp"], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec=None):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_cp_from_container", fake_cp_from)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)

    history, _ = _handle_tb2_edit(
        action={
            "path": "/work/file.py",
            "old_string": "= 1",
            "new_string": "= 2",
        },
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 24
    assert "occurs 3 times" in history["tool_output"]["combined_output"]


def test_handle_tb2_edit_allows_non_unique_with_replace_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_edit

    captured_dst_content: dict[str, str] = {}
    file_content = "x = 1\ny = 1\nz = 1\n"

    def fake_cp_from(*, container_name: str, src: str, dst: Path, timeout_sec: float = 30):
        dst.write_text(file_content, encoding="utf-8")
        return subprocess.CompletedProcess(args=["docker", "cp"], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec=None):
        # `docker cp <local> container:<path>` — capture local-side content
        if len(cmd) >= 4 and cmd[0] == "docker" and cmd[1] == "cp":
            local = Path(cmd[2])
            if local.is_file():
                captured_dst_content["written"] = local.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_cp_from_container", fake_cp_from)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)

    history, _ = _handle_tb2_edit(
        action={
            "path": "/work/file.py",
            "old_string": "= 1",
            "new_string": "= 2",
            "replace_all": True,
        },
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 0
    assert history["tool_input"]["replacements"] == 3
    assert captured_dst_content["written"] == "x = 2\ny = 2\nz = 2\n"


def test_handle_tb2_edit_preserves_crlf_line_endings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Editing a CRLF file must not silently convert untouched lines to LF.
    read_text/write_text apply universal-newline translation; the handler reads
    and writes bytes to preserve "\\r\\n" exactly."""
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_edit

    captured: dict[str, bytes] = {}
    original_bytes = b"line1\r\nline2\r\n"

    def fake_cp_from(*, container_name: str, src: str, dst: Path, timeout_sec: float = 30):
        dst.write_bytes(original_bytes)
        return subprocess.CompletedProcess(args=["docker", "cp"], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec=None):
        if len(cmd) >= 4 and cmd[0] == "docker" and cmd[1] == "cp":
            local = Path(cmd[2])
            if local.is_file():
                captured["written"] = local.read_bytes()
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_cp_from_container", fake_cp_from)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)

    history, _ = _handle_tb2_edit(
        action={"path": "/work/file.txt", "old_string": "line2", "new_string": "LINE2"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 0
    # line1's CRLF is preserved exactly; only "line2" -> "LINE2" changed.
    assert captured["written"] == b"line1\r\nLINE2\r\n"


def test_handle_tb2_edit_rejects_non_utf8_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-UTF-8 file cannot be substring-edited as text; the handler must
    reject it with exit 22 and a hint to use the shell action."""
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_edit

    written: dict[str, bytes] = {}

    def fake_cp_from(*, container_name: str, src: str, dst: Path, timeout_sec: float = 30):
        dst.write_bytes(b"\xff\xfe invalid")
        return subprocess.CompletedProcess(args=["docker", "cp"], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec=None):
        if len(cmd) >= 4 and cmd[0] == "docker" and cmd[1] == "cp":
            local = Path(cmd[2])
            if local.is_file():
                written["written"] = local.read_bytes()
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_cp_from_container", fake_cp_from)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)

    history, _ = _handle_tb2_edit(
        action={"path": "/work/bin.dat", "old_string": "x", "new_string": "y"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 22
    assert "not valid UTF-8" in history["tool_output"]["combined_output"]
    # The file must NOT be written back when the decode fails.
    assert "written" not in written


def test_handle_tb2_edit_missing_old_string_does_not_write_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """old_string absent from the file -> exit 23 and the file is left untouched
    (no docker cp back into the container)."""
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_edit

    written: dict[str, bytes] = {}

    def fake_cp_from(*, container_name: str, src: str, dst: Path, timeout_sec: float = 30):
        dst.write_text("alpha\nbeta\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=["docker", "cp"], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec=None):
        if len(cmd) >= 4 and cmd[0] == "docker" and cmd[1] == "cp":
            local = Path(cmd[2])
            if local.is_file():
                written["written"] = local.read_bytes()
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_cp_from_container", fake_cp_from)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)

    history, _ = _handle_tb2_edit(
        action={"path": "/work/file.txt", "old_string": "gamma", "new_string": "delta"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 23
    assert "old_string not found" in history["tool_output"]["combined_output"]
    # No write-back: the cp-into-container path is never reached on a miss.
    assert "written" not in written


def test_handle_tb2_edit_rejects_identical_old_new() -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_edit

    history, _ = _handle_tb2_edit(
        action={
            "path": "/work/file.py",
            "old_string": "x = 1",
            "new_string": "x = 1",
        },
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 2
    assert "identical" in history["tool_output"]["combined_output"]


def test_resolve_tb2_max_steps_explicit_zero_is_unlimited_sentinel(tmp_path: Path) -> None:
    """0 is the documented sentinel for unlimited; ensure it stays distinct from negatives."""
    from ksi.benchmarks.terminal_bench_2_runtime import _TB2_STEP_CAP_UNLIMITED

    task_root = _write_tb2_task(tmp_path)
    contract = resolve_terminal_bench_2_task_contract(_tb2_task_spec(task_root))

    assert _resolve_tb2_max_steps(contract=contract, env={"KSI_TB2_MAX_STEPS": "0"}) == _TB2_STEP_CAP_UNLIMITED
    # Positive small value is honored verbatim.
    assert _resolve_tb2_max_steps(contract=contract, env={"KSI_TB2_MAX_STEPS": "3"}) == 3
    # Negative values are rejected and fall through to the default (unlimited),
    # rather than being silently aliased to the 0-sentinel.
    assert _resolve_tb2_max_steps(contract=contract, env={"KSI_TB2_MAX_STEPS": "-5"}) == _TB2_STEP_CAP_UNLIMITED


def time_monotonic() -> float:
    import time as _t

    return _t.monotonic()


# ---------------------------------------------------------------------------
# Native-action handler tests (read / write / glob / grep)
# ---------------------------------------------------------------------------


def _fake_docker_exec_factory(*, returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Return a fake `_docker_exec` recording call args and producing a canned result."""
    calls: list[dict[str, object]] = []

    def fake(*, container_name: str, command: str, timeout_sec: float):
        calls.append({"container_name": container_name, "command": command, "timeout_sec": timeout_sec})
        return subprocess.CompletedProcess(
            args=["docker", "exec", container_name, "sh", "-lc", command],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def test_handle_tb2_read_rejects_missing_path() -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_read

    history, _ = _handle_tb2_read(
        action={"path": ""},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 2
    assert "requires 'path'" in history["tool_output"]["combined_output"]


def test_handle_tb2_read_passes_offset_limit_to_awk(monkeypatch: pytest.MonkeyPatch) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_read

    fake = _fake_docker_exec_factory(returncode=0, stdout="hello\nworld\n")
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    history, _ = _handle_tb2_read(
        action={"path": "/work/file.py", "offset": 10, "limit": 5},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 0
    assert history["tool_input"]["offset"] == 10
    assert history["tool_input"]["limit"] == 5
    cmd = fake.calls[0]["command"]
    # awk gets `-v s=<offset> -v e=<offset+limit-1>`
    assert "-v s=10" in cmd
    assert "-v e=14" in cmd


def test_handle_tb2_read_clamps_limit_to_2000(monkeypatch: pytest.MonkeyPatch) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_read

    fake = _fake_docker_exec_factory(returncode=0)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    history, _ = _handle_tb2_read(
        action={"path": "/work/file.py", "limit": 999_999},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_input"]["limit"] == 2000


def test_handle_tb2_write_rejects_missing_path() -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_write

    history, _ = _handle_tb2_write(
        action={"content": "hello"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 2
    assert "requires 'path'" in history["tool_output"]["combined_output"]


def test_handle_tb2_write_mkdir_parent_then_cp(monkeypatch: pytest.MonkeyPatch) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_write

    captured_dst_content: dict[str, str] = {}
    fake_exec = _fake_docker_exec_factory(returncode=0, stdout="")
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake_exec)

    def fake_run(cmd, *, timeout_sec=None):
        if len(cmd) >= 4 and cmd[0] == "docker" and cmd[1] == "cp":
            local = Path(cmd[2])
            if local.is_file():
                captured_dst_content["written"] = local.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)

    history, _ = _handle_tb2_write(
        action={"path": "/work/sub/file.py", "content": "x = 1\n"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 0
    assert captured_dst_content["written"] == "x = 1\n"
    # mkdir -p must run before cp; only the mkdir touches _docker_exec
    assert fake_exec.calls, "expected _docker_exec to be called for mkdir -p parent"
    mkdir_cmd = fake_exec.calls[0]["command"]
    assert mkdir_cmd.startswith("mkdir -p ")


def test_handle_tb2_glob_requires_pattern() -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_glob

    history, _ = _handle_tb2_glob(
        action={"path": "/work"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 2
    assert "requires 'pattern'" in history["tool_output"]["combined_output"]


def test_handle_tb2_glob_uses_find_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_glob

    fake = _fake_docker_exec_factory(returncode=0, stdout="/work/a.py\n/work/b.py\n")
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    history, _ = _handle_tb2_glob(
        action={"path": "/work", "pattern": "*.py"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 0
    cmd = fake.calls[0]["command"]
    # The command leads with an existence guard, then runs find.
    assert "find /work " in cmd
    assert " -type f -name " in cmd
    # basename-only semantics: `*.py` is passed to `find -name`, NOT `**/*.py`
    assert "'*.py'" in cmd


def test_handle_tb2_glob_guards_nonexistent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonexistent path must surface as a real error (exit 2 via the
    `__tb2_glob_error__` existence guard), not be silently indistinguishable
    from "no matches"."""
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_glob

    fake = _fake_docker_exec_factory(returncode=0, stdout="")
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    _ = _handle_tb2_glob(
        action={"path": "/nope", "pattern": "*.py"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    cmd = fake.calls[0]["command"]
    assert "__tb2_glob_error__" in cmd
    assert "[ ! -e " in cmd


def test_handle_tb2_grep_requires_pattern() -> None:
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_grep

    history, _ = _handle_tb2_grep(
        action={"path": "/work"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 2
    assert "requires 'pattern'" in history["tool_output"]["combined_output"]


def test_handle_tb2_grep_no_matches_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """grep exit 1 with empty stdout/stderr is the documented "no matches" case;
    surface it as exit 0 with a helpful stdout so the agent doesn't misread it
    as a tool error."""
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_grep

    fake = _fake_docker_exec_factory(returncode=1, stdout="", stderr="")
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    history, _ = _handle_tb2_grep(
        action={"pattern": "nonexistent", "path": "/work"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 0
    assert "(no matches)" in history["tool_output"]["combined_output"]
    # The command must forward grep's own exit code past the `head` pipeline,
    # otherwise returncode would always be the final head's (0) and the exit-1
    # remap above would be dead code.
    assert "exit ${PIPESTATUS[0]}" in fake.calls[0]["command"]


def test_handle_tb2_grep_real_error_preserves_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """grep exit 2 (real error like invalid regex) must NOT be collapsed to
    "(no matches)"; the agent needs the stderr message to recover."""
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_grep

    fake = _fake_docker_exec_factory(returncode=2, stdout="", stderr="grep: Invalid regular expression")
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    history, _ = _handle_tb2_grep(
        action={"pattern": "[invalid", "path": "/work"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 2
    assert "Invalid regular expression" in history["tool_output"]["combined_output"]
    assert "(no matches)" not in history["tool_output"]["combined_output"]
    # Same PIPESTATUS forwarding requirement: without it, exit 2 (this case)
    # would never reach the handler.
    assert "exit ${PIPESTATUS[0]}" in fake.calls[0]["command"]


def test_handle_tb2_grep_exit2_with_output_is_partial_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recursive grep returns exit 2 when SOME files/dirs are unreadable even
    though it found matches (common when path defaults to "/"). Non-empty
    stdout means partial results, which must NOT look like a hard failure."""
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_grep

    fake = _fake_docker_exec_factory(
        returncode=2,
        stdout="/etc/hosts:1:127.0.0.1 localhost\n",
        stderr="grep: /proc/1/root: Permission denied",
    )
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    history, _ = _handle_tb2_grep(
        action={"pattern": "localhost", "path": "/"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 0
    assert "localhost" in history["tool_output"]["combined_output"]


def test_handle_tb2_grep_uses_s_flag_for_permission_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """grep `-s` suppresses per-file permission-denied noise without silencing
    real errors. Verify the flag is present in the constructed command."""
    from ksi.benchmarks.terminal_bench_2_runtime import _handle_tb2_grep

    fake = _fake_docker_exec_factory(returncode=0, stdout="/work/file.py:1:hit\n")
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_exec", fake)

    _ = _handle_tb2_grep(
        action={"pattern": "hit", "path": "/work", "output_mode": "content"},
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    cmd = fake.calls[0]["command"]
    # All three output modes should carry `s` in the grep flag bundle.
    assert "-rnEs" in cmd or "-rlEs" in cmd or "-rcEs" in cmd
    # And the old shell-level `2>/dev/null` must be GONE; otherwise real
    # errors would be discarded again.
    assert "2>/dev/null" not in cmd


# ---------------------------------------------------------------------------
# Bug-fix tests: edit size cap + grep stderr propagation + timeout scaling
# ---------------------------------------------------------------------------


def test_handle_tb2_edit_rejects_oversized_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """edit reads the whole file into memory for substring substitution; a
    multi-GB file would blow Python's heap. Files above `_TB2_EDIT_MAX_BYTES`
    must be rejected with exit code 25 and a hint to use shell+sed."""
    from ksi.benchmarks.terminal_bench_2_runtime import _TB2_EDIT_MAX_BYTES, _handle_tb2_edit

    # Simulate a file just over the cap.
    oversize_bytes = _TB2_EDIT_MAX_BYTES + 1

    def fake_cp_from(*, container_name: str, src: str, dst: Path, timeout_sec: float = 30):
        # Write a sparse-ish file at exactly _TB2_EDIT_MAX_BYTES+1 bytes
        # without actually allocating that much heap (truncate to size).
        with dst.open("wb") as f:
            f.truncate(oversize_bytes)
        return subprocess.CompletedProcess(args=["docker", "cp"], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, timeout_sec=None):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._docker_cp_from_container", fake_cp_from)
    monkeypatch.setattr("ksi.benchmarks.terminal_bench_2_runtime._run", fake_run)

    history, _ = _handle_tb2_edit(
        action={
            "path": "/work/huge.bin",
            "old_string": "x",
            "new_string": "y",
        },
        container_name="dummy",
        deadline=time_monotonic() + 60.0,
        agent_id="agent-1",
    )
    assert history["tool_output"]["exit_code"] == 25
    out = history["tool_output"]["combined_output"]
    assert "exceeding" in out or "cap" in out
    assert "shell" in out.lower(), "rejection message should hint at shell+sed alternative"


def test_native_timeout_scale_env_widens_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """KSI_TB2_NATIVE_TIMEOUT_SCALE must scale (and floor at 1.0) the
    per-handler ceiling. Below-1.0 values must not narrow it."""
    from ksi.benchmarks.terminal_bench_2_runtime import (
        _TB2_NATIVE_TIMEOUT_CEILING_SEC,
        _tb2_native_timeout_ceiling,
    )

    monkeypatch.delenv("KSI_TB2_NATIVE_TIMEOUT_SCALE", raising=False)
    assert _tb2_native_timeout_ceiling() == _TB2_NATIVE_TIMEOUT_CEILING_SEC

    monkeypatch.setenv("KSI_TB2_NATIVE_TIMEOUT_SCALE", "2.5")
    assert _tb2_native_timeout_ceiling() == _TB2_NATIVE_TIMEOUT_CEILING_SEC * 2.5

    # Below 1.0 must not narrow the ceiling.
    monkeypatch.setenv("KSI_TB2_NATIVE_TIMEOUT_SCALE", "0.1")
    assert _tb2_native_timeout_ceiling() == _TB2_NATIVE_TIMEOUT_CEILING_SEC

    # Bogus value falls through to 1.0.
    monkeypatch.setenv("KSI_TB2_NATIVE_TIMEOUT_SCALE", "abc")
    assert _tb2_native_timeout_ceiling() == _TB2_NATIVE_TIMEOUT_CEILING_SEC


# --- TB2 prompt-cache block stability (issue #1252 item 1) ----------------


def _tb2_step(idx: int) -> dict:
    return {
        "tool_name": "tb2_shell",
        "tool_input": {"command": f"echo {idx}"},
        "tool_output": {"exit_code": 0, "combined_output": f"out {idx}"},
    }


def test_tb2_cache_blocks_are_append_only_across_turns():
    """Each turn appends exactly one history block and leaves every earlier
    block byte-identical, so the previous turn's cached prefix is a prefix of
    this turn's — the invariant that makes the moving cache breakpoint read."""
    from ksi.runtime.terminal_bench_2_trial import _tb2_bridge_cache_blocks

    task = TaskSpec(id="tb2-t1", repo="", prompt="do the thing")
    common = dict(
        task=task,
        generation=1,
        max_steps=10**9,  # unlimited (TB2 default)
        container_name="c",
        workspace_root=Path("/ws"),
        execution_prompt="EXEC GUIDANCE",
    )
    history: list[dict] = []
    prev: list[str] = []
    for n in range(1, 5):
        history.append(_tb2_step(n))
        blocks = _tb2_bridge_cache_blocks(history=history, **common)
        # One header block + one block per committed step.
        assert len(blocks) == n + 1
        # Every block from the previous turn is unchanged (append-only).
        assert blocks[: len(prev)] == prev
        prev = blocks
    # The header (block 0) carries the stable guidance; steps carry their data.
    assert "EXEC GUIDANCE" in prev[0]
    assert "Recent shell history:" in prev[0]
    assert "Step 4 (shell)" in prev[-1]


def test_tb2_cache_blocks_concatenate_with_clean_separators():
    """The LLM APIs join adjacent text blocks with NO separator, so each block
    must already end with whitespace or the last line of one step runs into the
    next block's first line ("...out 1Step 2 (shell):"). Reconstruct the exact
    model-visible string with an empty-string join and assert every boundary is
    cleanly separated (issue #1252 item 1)."""
    from ksi.runtime.terminal_bench_2_trial import _tb2_bridge_cache_blocks, _tb2_bridge_tail

    task = TaskSpec(id="tb2-t1", repo="", prompt="do the thing")
    history = [_tb2_step(1), _tb2_step(2)]
    blocks = _tb2_bridge_cache_blocks(
        task=task,
        generation=1,
        max_steps=10**9,
        container_name="c",
        workspace_root=Path("/ws"),
        execution_prompt="EXEC GUIDANCE",
        history=history,
    )
    tail = _tb2_bridge_tail(step_index=3, max_steps=10**9, last_observation="LATEST OBS")
    full = "".join(blocks) + tail  # exactly how the API concatenates text blocks
    # No boundary mashing: the header, each step, and the tail start on a fresh line.
    assert "Recent shell history:\nStep 1 (shell):" in full
    assert "out 1\n\nStep 2 (shell):" in full
    assert "out 2\n\nBridge step:" in full


def test_tb2_tail_carries_varying_fields_only():
    """The tail holds the per-turn step counter + latest observation and the
    call-to-action — and nothing that belongs in the cached prefix."""
    from ksi.runtime.terminal_bench_2_trial import _tb2_bridge_tail

    tail = _tb2_bridge_tail(step_index=7, max_steps=0, last_observation="LATEST OBS")
    assert "Bridge step: 7/0" in tail
    assert "LATEST OBS" in tail
    assert "Return the next JSON action now." in tail
    # Stable guidance must NOT leak into the varying tail.
    assert "Recent shell history:" not in tail
    assert "KSI execution guidance:" not in tail


def test_tb2_empty_history_is_single_header_block():
    from ksi.runtime.terminal_bench_2_trial import _tb2_bridge_cache_blocks

    blocks = _tb2_bridge_cache_blocks(
        task=TaskSpec(id="t", repo="", prompt="p"),
        generation=1,
        max_steps=10**9,
        container_name="c",
        workspace_root=Path("/ws"),
        execution_prompt="g",
        history=[],
    )
    assert len(blocks) == 1
    assert "(no prior shell steps yet)" in blocks[0]
