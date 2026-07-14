from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace


def test_cli_parser_accepts_container_runtime():
    from ksi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            "/tmp/tasks.json",
            "--knowledge-db-path",
            "/tmp/knowledge.sqlite",
            "--runtime",
            "container",
        ]
    )
    assert args.runtime == "container"


def test_choose_runtime_returns_shared_container_executor(monkeypatch, tmp_path):
    from ksi.cli import _choose_runtime
    from ksi.runtime import KsiContainerExecutor

    monkeypatch.setattr("ksi.runtime.registry._ensure_runtime_runner_deps", lambda _project_root: None)

    args = Namespace(
        runtime="container",
        runtime_timeout_sec=600,
        knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
        disable_memory_mcp=False,
        container_command="",
        session_scope="task",
        wipe_workspace_per_task="true",
        forum_timeout_sec=900,
    )
    provider_env = {
        "MODEL_PROVIDER": "anthropic",
        "MODEL": "claude-sonnet-4-6",
        "MODEL_AUTH_MODE": "api",
        "ANTHROPIC_API_KEY": "test-key",
    }

    runtime = _choose_runtime(args, provider_env)
    assert isinstance(runtime, KsiContainerExecutor)
    assert runtime.command[-1] == "runtime_runner/src/main.ts"


def test_choose_runtime_returns_tb2_executor_for_terminal_bench_2(monkeypatch, tmp_path):
    from ksi.cli import _choose_runtime
    from ksi.runtime import KsiContainerExecutor, TerminalBench2Executor

    monkeypatch.setattr("ksi.runtime.registry._ensure_runtime_runner_deps", lambda _project_root: None)

    args = Namespace(
        runtime="container",
        task_source="terminal_bench_2",
        runtime_timeout_sec=600,
        knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
        disable_memory_mcp=False,
        container_command="",
        session_scope="task",
        wipe_workspace_per_task="true",
        forum_timeout_sec=900,
    )
    provider_env = {
        "MODEL_PROVIDER": "openai",
        "MODEL": "gpt-5.4-mini",
    }

    runtime = _choose_runtime(args, provider_env)
    assert isinstance(runtime, TerminalBench2Executor)
    assert runtime.agent_mode == "ksi"
    assert runtime.env["MODEL"] == "gpt-5.4-mini"
    assert isinstance(runtime.fallback_runtime, KsiContainerExecutor)


def test_choose_runtime_tb2_keeps_output_when_env_flag_set(monkeypatch, tmp_path):
    from ksi.cli import _choose_runtime
    from ksi.runtime import TerminalBench2Executor

    monkeypatch.setattr("ksi.runtime.registry._ensure_runtime_runner_deps", lambda _project_root: None)
    monkeypatch.setenv("KSI_TB2_KEEP_OUTPUT", "1")

    args = Namespace(
        runtime="container",
        task_source="terminal_bench_2",
        runtime_timeout_sec=600,
        knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
        disable_memory_mcp=False,
        container_command="",
        session_scope="task",
        wipe_workspace_per_task="true",
        forum_timeout_sec=900,
    )
    provider_env = {"MODEL_PROVIDER": "anthropic", "MODEL": "claude-haiku-4-5-20251001"}

    runtime = _choose_runtime(args, provider_env)
    assert isinstance(runtime, TerminalBench2Executor)
    assert runtime.keep_container is True


def test_choose_runtime_tb2_defaults_to_cleaning_up_output(monkeypatch, tmp_path):
    from ksi.cli import _choose_runtime
    from ksi.runtime import TerminalBench2Executor

    monkeypatch.setattr("ksi.runtime.registry._ensure_runtime_runner_deps", lambda _project_root: None)
    monkeypatch.delenv("KSI_TB2_KEEP_OUTPUT", raising=False)

    args = Namespace(
        runtime="container",
        task_source="terminal_bench_2",
        runtime_timeout_sec=600,
        knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
        disable_memory_mcp=False,
        container_command="",
        session_scope="task",
        wipe_workspace_per_task="true",
        forum_timeout_sec=900,
    )
    provider_env = {"MODEL_PROVIDER": "anthropic", "MODEL": "claude-haiku-4-5-20251001"}

    runtime = _choose_runtime(args, provider_env)
    assert isinstance(runtime, TerminalBench2Executor)
    assert runtime.keep_container is False


def test_choose_runtime_tb2_threads_memory_seed_raw_attempts(monkeypatch, tmp_path):
    from ksi.cli import _choose_runtime
    from ksi.runtime import TerminalBench2Executor

    monkeypatch.setattr("ksi.runtime.registry._ensure_runtime_runner_deps", lambda _project_root: None)

    args = Namespace(
        runtime="container",
        task_source="terminal_bench_2",
        runtime_timeout_sec=600,
        knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
        disable_memory_mcp=False,
        container_command="",
        session_scope="task",
        wipe_workspace_per_task="true",
        forum_timeout_sec=900,
        memory_seed_raw_attempts=True,
    )
    provider_env = {"MODEL_PROVIDER": "anthropic", "MODEL": "claude-haiku-4-5-20251001"}

    runtime = _choose_runtime(args, provider_env)
    assert isinstance(runtime, TerminalBench2Executor)
    assert runtime.memory_seed_raw_attempts is True


def test_choose_runtime_tb2_defaults_memory_seed_raw_attempts_to_false_when_absent(monkeypatch, tmp_path):
    from ksi.cli import _choose_runtime
    from ksi.runtime import TerminalBench2Executor

    monkeypatch.setattr("ksi.runtime.registry._ensure_runtime_runner_deps", lambda _project_root: None)

    args = Namespace(
        runtime="container",
        task_source="terminal_bench_2",
        runtime_timeout_sec=600,
        knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
        disable_memory_mcp=False,
        container_command="",
        session_scope="task",
        wipe_workspace_per_task="true",
        forum_timeout_sec=900,
    )
    provider_env = {"MODEL_PROVIDER": "anthropic", "MODEL": "claude-haiku-4-5-20251001"}

    runtime = _choose_runtime(args, provider_env)
    assert isinstance(runtime, TerminalBench2Executor)
    assert runtime.memory_seed_raw_attempts is False


def test_ensure_runtime_runner_deps_installs_when_missing(monkeypatch, tmp_path):
    from ksi.runtime.registry import _ensure_runtime_runner_deps

    runtime_runner_dir = tmp_path / "runtime_runner"
    runtime_runner_dir.mkdir()
    (runtime_runner_dir / "package.json").write_text('{"name":"runtime-runner"}\n', encoding="utf-8")
    (runtime_runner_dir / "package-lock.json").write_text('{"name":"runtime-runner"}\n', encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run(cmd, cwd, capture_output, text, timeout):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ksi.runtime.registry.subprocess.run", fake_run)

    _ensure_runtime_runner_deps(str(tmp_path))

    assert captured["cmd"] == [
        "npm",
        "--prefix",
        str(runtime_runner_dir),
        "ci",
        "--silent",
        "--no-audit",
        "--no-fund",
    ]
    assert captured["cwd"] == str(tmp_path)
    assert captured["timeout"] == 600


def test_ensure_runtime_runner_deps_skips_when_present(monkeypatch, tmp_path):
    from ksi.runtime.registry import _ensure_runtime_runner_deps

    runtime_runner_dir = tmp_path / "runtime_runner"
    tsx_bin = runtime_runner_dir / "node_modules" / ".bin" / "tsx"
    pino_pkg = runtime_runner_dir / "node_modules" / "pino" / "package.json"
    tsx_bin.parent.mkdir(parents=True)
    pino_pkg.parent.mkdir(parents=True)
    (runtime_runner_dir / "package.json").write_text('{"name":"runtime-runner"}\n', encoding="utf-8")
    tsx_bin.write_text("", encoding="utf-8")
    pino_pkg.write_text('{"name":"pino"}\n', encoding="utf-8")

    def fail_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("npm bootstrap should not run when runtime_runner deps are already present")

    monkeypatch.setattr("ksi.runtime.registry.subprocess.run", fail_run)

    _ensure_runtime_runner_deps(str(tmp_path))


def test_task_workspace_key_truncates_long_swebench_pro_ids():
    from ksi.layout import task_workspace_key

    task_id = (
        "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
    )
    workspace_key = task_workspace_key(task_id, "swebench_pro_openai_canary_nodebb_g1_fix1")

    assert workspace_key.startswith("task__")
    assert len(workspace_key) <= 128
    assert "NodeBB" in workspace_key


def test_task_workspace_key_hashes_untruncated_identity():
    from ksi.layout import task_workspace_key

    common_task_prefix = "task-" + ("a" * 90)
    first = task_workspace_key(common_task_prefix + "-first", "experiment-" + ("b" * 40))
    second = task_workspace_key(common_task_prefix + "-second", "experiment-" + ("b" * 40))

    assert first != second
    assert first.rsplit("__", 1)[0] == second.rsplit("__", 1)[0]
