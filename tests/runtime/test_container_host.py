import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import REPO_ROOT

from ksi.benchmarks.polyglot_harness import DEFAULT_POLYGLOT_TIMEOUT_SEC
from ksi.errors import AuthenticationFailure
from ksi.models import TaskSpec
from ksi.runtime.container_host import (
    _ARC_ATTEMPT_PRESTUB,
    _ARC_VALIDATE_PREDICTION_SCRIPT,
    KsiContainerExecutor,
    _arc_attempt_stub_files,
    _build_runner_env,
    _build_tools_md,
    _scrub_credentials,
    _swebench_agent_overlay_dockerfile,
    _swebench_official_base_image,
    _tail,
    _validate_provider_auth,
)
from ksi.runtime.seeding import seed_package_to_memory_md
from ksi.runtime.swebench_images import _swebench_runner_image


def _runner_stdout(
    *,
    result: str | None = "ok",
    generation: int = 1,
    agent_id: str = "agent-0",
    task_id: str = "t1",
    status: str = "success",
    tool_trace: list[dict] | None = None,
    meta: dict | None = None,
) -> str:
    runtime_meta = {
        "generation": generation,
        "agent_id": agent_id,
        "task_id": task_id,
        "status": status,
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    if meta:
        runtime_meta.update(meta)
    return json.dumps(
        {
            "result": result,
            "tool_trace": tool_trace if tool_trace is not None else [],
            "meta": runtime_meta,
        }
    )


def test_build_tools_md_for_forum_round_2_mentions_commenting_tools():
    task = TaskSpec(
        id="__forum__g1_r2_agent-0",
        repo="",
        prompt="",
        metadata={
            "task_source": "per_task_forum",
            "forum_round": 2,
        },
    )
    tools_md = _build_tools_md(
        task=task,
        provider="openai",
        has_memory_mcp=True,
    )
    assert "forum_read" in tools_md
    assert "query" in tools_md
    assert "semantic related hits" in tools_md
    assert "knowledge" in tools_md
    assert "forum_post" in tools_md
    assert "forum_signal_done" in tools_md
    assert "shell" in tools_md
    assert "apply_patch" in tools_md
    assert "shell_call" not in tools_md


def test_swebench_runner_image_prefers_container_image_over_legacy_alias():
    assert (
        _swebench_runner_image(
            {
                "CONTAINER_IMAGE": "ksi-agent:from-container",
                "KSI_CONTAINER_IMAGE": "ksi-agent:from-legacy",
            }
        )
        == "ksi-agent:from-container"
    )


def test_build_tools_md_for_cross_task_forum_mentions_openai_forum_tools():
    task = TaskSpec(
        id="__cross_task_forum__g1_r0_agent-0",
        repo="",
        prompt="",
        metadata={"task_source": "cross_task_forum"},
    )
    tools_md = _build_tools_md(
        task=task,
        provider="openai",
        has_memory_mcp=True,
    )
    assert "query" in tools_md
    assert "semantic related hits" in tools_md
    assert "knowledge" in tools_md
    assert "forum_post" in tools_md
    assert "forum_signal_done" in tools_md
    assert "Task memory is pre-injected" not in tools_md


def test_build_tools_md_for_task_execution_does_not_advertise_live_memory_tools():
    task = TaskSpec(
        id="swe-001",
        repo="repo",
        prompt="fix",
        metadata={"task_source": "swebench_pro"},
    )
    tools_md = _build_tools_md(
        task=task,
        provider="openai",
        has_memory_mcp=True,
    )
    assert "Task memory is pre-injected into MEMORY.md" in tools_md
    assert "query:" not in tools_md
    assert "knowledge:" not in tools_md
    assert "forum_post" not in tools_md


def test_build_tools_md_arc_advertises_native_file_tools_and_offline_web():
    """ARC is native (attempt-file) now: the agent reads payload.json and writes
    its attempt files with the standard file/shell tools, while web tools stay
    disabled for the offline benchmark. The legacy ARC MCP tools are gone."""
    task = TaskSpec(
        id="arc-001",
        repo="",
        prompt="",
        metadata={"task_source": "arc"},
    )
    tools_md = _build_tools_md(
        task=task,
        provider="anthropic",
        has_memory_mcp=True,
    )
    # Native file/shell tools ARE advertised for the attempt-file workflow.
    assert "Bash, Read, Write, Edit, Glob, Grep" in tools_md
    # Web tools stay disabled for the offline benchmark.
    assert "WebSearch, WebFetch," not in tools_md
    assert "disabled for this offline benchmark" in tools_md
    # No legacy ARC MCP tool references survive.
    assert "arc_load_task" not in tools_md
    assert "arc_submit_trial" not in tools_md


def test_build_tools_md_non_arc_default_off_does_not_advertise_web_tools():
    """Issue #666: with web tools default-OFF, the non-ARC TOOLS.md must NOT
    advertise WebSearch/WebFetch as enabled (the runtime denies them), or the
    agent is told it has tools the SDK blocks. TodoWrite/NotebookEdit stay."""
    task = TaskSpec(
        id="swe-001",
        repo="",
        prompt="",
        metadata={"task_source": "swebench_pro"},
    )
    tools_md = _build_tools_md(
        task=task,
        provider="anthropic",
        has_memory_mcp=True,
        web_tools_enabled=False,
    )
    # The old always-on advertisement must be gone.
    assert "WebSearch, WebFetch, TodoWrite, NotebookEdit" not in tools_md
    # Web tools are surfaced as disabled-by-default with the opt-in flag.
    assert "WebSearch and WebFetch are disabled by default" in tools_md
    assert "KSI_ALLOW_WEB_TOOLS=1" in tools_md
    # Utility tools that are genuinely always available are still advertised.
    assert "TodoWrite, NotebookEdit" in tools_md
    # Not the ARC-offline wording either.
    assert "disabled for this offline benchmark" not in tools_md


def test_build_tools_md_non_arc_advertises_web_tools_when_enabled():
    """When KSI_ALLOW_WEB_TOOLS is set (web_tools_enabled=True), the non-ARC
    TOOLS.md advertises WebSearch/WebFetch as enabled for the run."""
    task = TaskSpec(
        id="swe-001",
        repo="",
        prompt="",
        metadata={"task_source": "swebench_pro"},
    )
    tools_md = _build_tools_md(
        task=task,
        provider="anthropic",
        has_memory_mcp=True,
        web_tools_enabled=True,
    )
    assert "WebSearch, WebFetch: enabled for this run" in tools_md
    assert "disabled by default" not in tools_md


def test_agent_runner_index_ts_guards_web_tools_for_arc():
    """Regression guard: the TS agent-runner must strip web tools for ARC runs
    AND, post-#666, default-OFF web tools for every benchmark task.

    The authoritative defense is the SDK's `disallowedTools` option (the
    claude_code preset re-adds web tools to context regardless of
    `allowedTools`; only `disallowedTools` removes them — see the SDK's
    `disallowedTools` JSDoc; runtime_runner/agent-runner pins the SDK at ^0.1.0,
    where it is in entrypoints/sdk/runtimeTypes.d.ts, exact path varies by
    version).
    Web tools are enabled only when `KSI_ALLOW_WEB_TOOLS` is truthy AND the
    task is not ARC; otherwise they are disallowed.
    """
    runner_src = REPO_ROOT / "runtime_runner" / "agent-runner" / "src"
    # The web-tool gating wiring moved out of the (former god-file) index.ts:
    # the allow/disallow lists live in query_config.ts and the query() call that
    # passes disallowedTools lives in query_runner.ts. Concatenate both.
    src = "\n".join((runner_src / name).read_text(encoding="utf-8") for name in ("query_config.ts", "query_runner.ts"))
    web = (runner_src / "web_tools.ts").read_text(encoding="utf-8")
    # ARC offline guard still keys on 'arc'.
    assert "taskSource === 'arc'" in src
    # The gating decision lives in the SDK-free web_tools.ts (behaviorally
    # tested in tests/js/web_tools_gating.test.mjs); index.ts imports it.
    assert "from './web_tools.js'" in src
    assert "const webToolGating = buildWebToolGating(sdkEnv, isOffline)" in src
    # Web tools are default-OFF: gated behind KSI_ALLOW_WEB_TOOLS, AND-gated
    # with !isOffline so ARC is always denied (issue #666).
    assert "KSI_ALLOW_WEB_TOOLS" in web
    assert "const webToolsEnabled = webToolsAllowedByFlag && !isOffline" in web
    # Allowlist adds web tools ONLY when enabled; the old always-on-for-non-ARC
    # pattern must be gone from both files.
    assert "...webToolGating.allowlistWebTools" in src
    assert "isOffline ? [] : ['WebSearch', 'WebFetch']" not in src
    # The disallowedTools field must be passed to query().
    assert "disallowedTools:" in src
    # CRITICAL: the denial covers both web tools whenever they are not enabled
    # (ARC always + every benchmark unless the flag is set), not just ARC.
    assert "...webToolGating.disallowedWebTools" in src
    assert "disallowedWebTools: webToolsEnabled ? [] : [...WEB_TOOLS]" in web
    assert "const disallowedToolsList: string[] = isOffline" not in src
    # Scheduled MCP-protocol tasks (ARC with MCP and forum tasks) must also
    # deny native Claude file/shell tools via disallowedTools; allowedTools=[]
    # alone is not sufficient under the claude_code preset.
    assert "NATIVE_FILE_SHELL_TOOLS" in src
    assert "const protocolNativeToolDenials = isMcpProtocolOnlyTask ? [...NATIVE_FILE_SHELL_TOOLS] : []" in src


# ── KNOWLEDGE_DB_PATH in container payload ──────────────────────────────────


class TestKnowledgeDbPathInPayload:
    """Verify knowledge_db_path is passed to containers via the payload."""

    def _make_executor(self, tmp_path, **kw):
        defaults = dict(
            command=["echo", "dummy"],
            working_dir=str(tmp_path),
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )
        defaults.update(kw)
        return KsiContainerExecutor(**defaults)

    def test_knowledge_db_path_present_when_memory_configured(self, tmp_path):
        """When knowledge_db_path is set, payload.knowledge should include knowledge_db_path."""
        captured = []
        knowledge_db_path = tmp_path / "foo_knowledge.sqlite"

        def fake_run(cmd, **kw):
            with open(cmd[-1]) as f:
                captured.append(json.load(f))
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(task_id="t1"),
                stderr="",
            )

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            ex = self._make_executor(tmp_path, knowledge_db_path=str(knowledge_db_path))
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(id="t1", prompt="solve"),
                agent_seed_package={},
                experiment_name="test_exp",
            )

        assert len(captured) == 1
        payload = captured[0]
        assert "knowledge" in payload
        mem = payload["knowledge"]
        assert "db_path" in mem
        # Per-experiment knowledge sibling: foo_knowledge.sqlite -> foo_knowledge.sqlite
        expected = str(knowledge_db_path.parent / "foo_knowledge.sqlite")
        assert mem["db_path"] == expected

    def test_no_knowledge_db_path_when_memory_not_configured(self, tmp_path):
        """When knowledge_db_path is empty, payload should not have a memory section."""
        captured = []

        def fake_run(cmd, **kw):
            with open(cmd[-1]) as f:
                captured.append(json.load(f))
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(task_id="t1"),
                stderr="",
            )

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            ex = self._make_executor(tmp_path, knowledge_db_path="")
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(id="t1", prompt="solve"),
                agent_seed_package={},
                experiment_name="test_exp",
            )

        assert len(captured) == 1
        payload = captured[0]
        assert "knowledge" not in payload

    def test_seeded_memory_sets_execution_prompt_without_memory_mcp(self, tmp_path):
        """Seeded MEMORY.md is real memory even when no live memory MCP is mounted."""
        captured = []

        def fake_run(cmd, **kw):
            with open(cmd[-1]) as f:
                captured.append(json.load(f))
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(generation=2, task_id="swe-001"),
                stderr="",
            )

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            ex = self._make_executor(
                tmp_path,
                knowledge_db_path="",
                disable_memory_mcp=True,
            )
            ex.run_task(
                generation=2,
                agent_id="agent-0",
                task=TaskSpec(
                    id="swe-001",
                    prompt="solve",
                    metadata={"task_source": "swebench_pro"},
                ),
                agent_seed_package={
                    "workstream_name": "solver",
                    "per_task_bundle": {
                        "transferable_insights": ["reuse the cached parse"],
                        "pitfalls": [],
                        "checks": [],
                        "evidence_post_ids": [12],
                    },
                },
                experiment_name="test_exp",
            )

        assert len(captured) == 1
        payload = captured[0]
        assert "knowledge" not in payload
        assert "reuse the cached parse" in payload["workspace_seed"]["memory_md"]
        assert "Review prior attempts" in payload["execution_prompt"]
        assert "No prior memory is provided" not in payload["execution_prompt"]
        assert "Task memory is pre-injected into MEMORY.md" in payload["workspace_seed"]["tools_md"]

    def test_knowledge_db_path_uses_absolute_path(self, tmp_path):
        """knowledge_db_path should be an absolute path alongside the knowledge DB."""
        captured = []
        knowledge_db_path = tmp_path / "subdir" / "exp_knowledge.sqlite"
        knowledge_db_path.parent.mkdir(parents=True, exist_ok=True)

        def fake_run(cmd, **kw):
            with open(cmd[-1]) as f:
                captured.append(json.load(f))
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(task_id="t1"),
                stderr="",
            )

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            ex = self._make_executor(tmp_path, knowledge_db_path=str(knowledge_db_path))
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(id="t1", prompt="solve"),
                agent_seed_package={},
                experiment_name="test_exp",
            )

        knowledge_path = captured[0]["knowledge"]["db_path"]
        assert Path(knowledge_path).is_absolute()
        # Per-experiment sibling: exp_knowledge.sqlite -> exp_knowledge.sqlite
        assert knowledge_path.endswith("exp_knowledge.sqlite")
        assert str(knowledge_db_path.parent) in knowledge_path

    def test_run_task_returns_injected_memory_md_in_runtime_meta(self, tmp_path):
        """The exact MEMORY.md injected into the workspace must be auditable.

        The engine persists ``runtime_meta["injected_memory_md"]`` into
        task_memory_records. The runner does not know this value, so the
        container host must attach it after constructing the workspace payload.
        """
        captured = []

        def fake_run(cmd, **kw):
            with open(cmd[-1]) as f:
                captured.append(json.load(f))
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(
                    generation=2,
                    task_id="t1",
                    meta={"native_session_memory": "runtime transcript"},
                ),
                stderr="",
            )

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            ex = self._make_executor(
                tmp_path,
                knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
            )
            result = ex.run_task(
                generation=2,
                agent_id="agent-0",
                task=TaskSpec(id="t1", prompt="solve"),
                agent_seed_package={
                    "workstream_name": "solver",
                    "per_task_bundle": {
                        "transferable_insights": ["reuse the cached parse"],
                        "pitfalls": [],
                        "checks": ["run the parser smoke test"],
                    },
                },
                experiment_name="test_exp",
            )

        injected = result.runtime_meta.get("injected_memory_md")
        assert injected == captured[0]["workspace_seed"]["memory_md"]
        assert "reuse the cached parse" in injected
        assert result.runtime_meta["native_session_memory"] == "runtime transcript"

    def test_setup_phase_raise_does_not_leak_memory_snapshot(self, tmp_path):
        """A setup-phase raise after the per-task snapshot write must not leak
        the persistent ``memory_snapshot_*.json`` (#849).

        The snapshot is written into ``knowledge_db.parent`` (a persistent dir,
        not the per-task ``TemporaryDirectory``) before provider-auth
        validation. Forcing ``_validate_provider_auth`` to raise (``MODEL=""``)
        exercises a reachable raiser between the write and the runner block; the
        cleanup must still fire.
        """
        knowledge_db_path = tmp_path / "exp_knowledge.sqlite"
        ex = self._make_executor(
            tmp_path,
            knowledge_db_path=str(knowledge_db_path),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "",  # forces _validate_provider_auth AuthenticationFailure post-write
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )

        with pytest.raises(AuthenticationFailure):
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(id="t1", prompt="solve"),
                agent_seed_package={"memory_snapshot": {"entries": []}},
                experiment_name="test_exp",
            )

        leaked = list(knowledge_db_path.parent.glob("memory_snapshot_*.json"))
        assert leaked == [], f"snapshot leaked after setup-phase raise: {leaked}"


class TestKnowledgeSnapshotFailureFlag:
    """`_materialize_payload_side_files` signals a failed snapshot write (#981).

    A snapshot write failure makes the container start cold (memory-less); the
    second return value lets the caller stamp `runtime_meta` so the cold-start
    is not silently counted as a like-for-like data point in gen-over-gen
    comparisons.
    """

    def _make_executor(self, tmp_path, **kw):
        defaults = dict(
            command=["echo", "dummy"],
            working_dir=str(tmp_path),
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            knowledge_db_path=str(tmp_path / "exp_knowledge.sqlite"),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )
        defaults.update(kw)
        return KsiContainerExecutor(**defaults)

    def _materialize(self, executor, tmp_path, snapshot):
        payload: dict = {}
        return executor._materialize_payload_side_files(
            payload=payload,
            td=str(tmp_path),
            task=TaskSpec(id="t1", prompt="solve"),
            metadata={},
            agent_id="agent-0",
            generation=0,
            source="polyglot",
            seed_package={"memory_snapshot": snapshot},
            swebench_container_images={},
            experiment_name="exp",
        ), payload

    def test_healthy_snapshot_write_reports_no_failure(self, tmp_path):
        executor = self._make_executor(tmp_path)
        (snapshot_path, failed), payload = self._materialize(executor, tmp_path, {"entries": []})
        assert failed is False
        assert snapshot_path is not None
        assert payload["knowledge"].get("snapshot_path") == str(snapshot_path)

    def test_failed_snapshot_write_sets_flag_and_omits_path(self, tmp_path):
        executor = self._make_executor(tmp_path)
        # A non-JSON-serializable snapshot makes the host-side write_text raise,
        # so the container would start cold. The method must swallow it (the
        # container still runs) AND signal the failure.
        (snapshot_path, failed), payload = self._materialize(executor, tmp_path, {"bad": object()})
        assert failed is True
        # No snapshot_path is threaded into the payload, so the container starts cold.
        assert "snapshot_path" not in payload.get("knowledge", {})


class TestRunnerEnvelopeIdentity:
    def _make_executor(self, tmp_path):
        return KsiContainerExecutor(
            command=["echo", "dummy"],
            working_dir=str(tmp_path),
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )

    @pytest.mark.parametrize(
        "meta_override",
        [
            {"generation": 2},
            {"agent_id": "agent-1"},
            {"task_id": "other-task"},
        ],
    )
    def test_run_task_rejects_envelope_identity_mismatch(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
        meta_override: dict,
    ):
        trace_dir = tmp_path / "traces"
        monkeypatch.setenv("KSI_TRACE_DIR", str(trace_dir))
        stdout = _runner_stdout(task_id="t1", meta=meta_override)
        ex = self._make_executor(tmp_path)

        def fake_run(cmd, **kw):
            return MagicMock(returncode=0, stdout=stdout, stderr="")

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            with pytest.raises(RuntimeError, match="Container stdout protocol error"):
                ex.run_task(
                    generation=1,
                    agent_id="agent-0",
                    task=TaskSpec(id="t1", prompt="solve"),
                    agent_seed_package={},
                    experiment_name="test_exp",
                )

        events_path = trace_dir / "runtime_events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        assert any(event.get("event") == "runtime.parse_error" for event in events)

    def test_run_task_salvages_coding_max_turn_workspace_artifacts(self, tmp_path):
        stdout = _runner_stdout(
            result="MaxTurnsExceededError",
            task_id="t1",
            status="error",
            meta={
                "error": "maxTurns exceeded",
                "workspace_diff": "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-x\n+y\n",
            },
        )
        ex = self._make_executor(tmp_path)

        def fake_run(cmd, **kw):
            return MagicMock(returncode=0, stdout=stdout, stderr="")

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            result = ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(id="t1", prompt="solve", metadata={"task_source": "swebench_pro"}),
                agent_seed_package={},
                experiment_name="test_exp",
            )

        assert result.output == "MaxTurnsExceededError"
        assert "workspace_diff" in result.runtime_meta
        assert result.runtime_meta["salvaged_error_status"] == "error"
        assert result.runtime_meta["salvaged_error"] == "maxTurns exceeded"


# ── arc_tools payload block (independent of memory gate) ────────────────────


class TestArcToolsPayloadBlock:
    """Verify the `arc_tools` payload block is always emitted with `enable` False.

    The legacy ARC MCP toolset has been removed: ARC is always native (the agent
    reads payload.json and writes attempt files). The `arc_tools` block is
    retained for TS-side stability but never enables an MCP server.
    """

    def _make_executor(self, tmp_path, **kw):
        defaults = dict(
            command=["echo", "dummy"],
            working_dir=str(tmp_path),
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )
        defaults.update(kw)
        return KsiContainerExecutor(**defaults)

    def _capture_payload(self, ex, task):
        captured = []

        def fake_run(cmd, **kw):
            with open(cmd[-1]) as f:
                captured.append(json.load(f))
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(task_id=task.id),
                stderr="",
            )

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=task,
                agent_seed_package={},
                experiment_name="test_exp",
            )
        assert len(captured) == 1
        return captured[0]

    def test_arc_tools_emitted_disabled_for_arc_task_with_no_memory(self, tmp_path):
        """ARC is native now: arc_tools is still emitted but ``enable`` is hard
        False (no MCP server registered), independent of knowledge_db_path."""
        ex = self._make_executor(tmp_path, knowledge_db_path="")
        task = TaskSpec(
            id="arc-001",
            prompt="solve grid",
            metadata={"task_source": "arc"},
        )
        payload = self._capture_payload(ex, task)

        # Knowledge block is absent (no knowledge_db_path set).
        assert "knowledge" not in payload
        # arc_tools is present but disabled (native path, no MCP).
        assert "arc_tools" in payload
        assert payload["arc_tools"]["enable"] is False
        # task_source is echoed so the TS side can derive the ARC guard.
        assert payload["arc_tools"]["task_source"] == "arc"
        assert payload["arc_tools"]["task_id"] == "arc-001"

    def test_arc_tools_emitted_disabled_for_arc_task_with_memory(self, tmp_path):
        """ARC task + configured knowledge_db_path: knowledge present, arc_tools
        emitted with enable False (native)."""
        knowledge_db_path = tmp_path / "exp_knowledge.sqlite"
        ex = self._make_executor(tmp_path, knowledge_db_path=str(knowledge_db_path))
        task = TaskSpec(
            id="arc-002",
            prompt="solve grid",
            metadata={"task_source": "arc"},
        )
        payload = self._capture_payload(ex, task)

        assert "knowledge" in payload
        assert "arc_tools" in payload
        assert payload["arc_tools"]["enable"] is False
        assert payload["arc_tools"]["task_id"] == "arc-002"

    def test_arc_tools_disabled_for_non_arc_task(self, tmp_path):
        """Non-ARC task: arc_tools.enable=False regardless of memory state."""
        ex = self._make_executor(tmp_path, knowledge_db_path="")
        task = TaskSpec(
            id="swe-001",
            prompt="patch bug",
            metadata={"task_source": "swebench_pro"},
        )
        payload = self._capture_payload(ex, task)

        # arc_tools is always emitted, but enable=False for non-ARC sources.
        assert "arc_tools" in payload
        assert payload["arc_tools"]["enable"] is False
        # And task_source reflects the actual source.
        assert payload["arc_tools"]["task_source"] == "swebench_pro"
        assert payload["arc_tools"]["task_id"] == "swe-001"

    def test_unparseable_stdout_records_runtime_parse_error(self, tmp_path, monkeypatch):
        """When the container returns an unparseable envelope, container_host raises
        RuntimeError and records a runtime.parse_error event."""
        trace_dir = tmp_path / "traces"
        monkeypatch.setenv("KSI_TRACE_DIR", str(trace_dir))

        ex = self._make_executor(
            tmp_path,
            knowledge_db_path="",
        )
        task = TaskSpec(id="bad-stdout", prompt="solve", metadata={"task_source": "swebench_pro"})

        def fake_run(cmd, *, cwd, env, timeout):
            return MagicMock(returncode=0, stdout="not-json", stderr="")

        with patch.object(
            KsiContainerExecutor,
            "_run_runner_command",
            side_effect=fake_run,
        ):
            with pytest.raises(RuntimeError, match="Container stdout protocol error"):
                ex.run_task(
                    generation=1,
                    agent_id="agent-0",
                    task=task,
                    agent_seed_package={},
                    experiment_name="test_exp",
                )

        events = (trace_dir / "runtime_events.jsonl").read_text().splitlines()
        assert any("runtime.parse_error" in line for line in events), events


# ── Scoped distillation bundles in seed_package_to_memory_md ────────────────


class TestScopedBundlesInMemoryMd:
    """Plan Task 16: per_task_bundle + cross_task_bundle rendering.

    The legacy ``distilled_knowledge`` key was removed in Plan Task 15;
    scoped per-task and cross-task bundles now flow through the seeder and
    are rendered in dedicated sections.
    """

    def test_per_task_bundle_rendered(self):
        pkg = {
            "workstream_name": "solver",
            "per_task_bundle": {
                "transferable_insights": [
                    "Use pattern matching for grid transforms",
                ],
                "pitfalls": ["Check edge cases first"],
                "checks": [],
                "evidence_post_ids": [1],
            },
        }
        md = seed_package_to_memory_md(pkg, current_task_id="t1")
        assert "Task-specific guidance" in md
        assert "pattern matching" in md
        assert "edge cases" in md
        assert "Pitfalls" in md
        assert "Evidence posts: #1" in md

    def test_cross_task_bundle_rendered(self):
        pkg = {
            "workstream_name": "solver",
            "cross_task_bundle": {
                "transferable_insights": ["Always verify output format"],
                "pitfalls": ["Try brute force before optimization"],
                "checks": [],
                "evidence_post_ids": [8],
            },
        }
        md = seed_package_to_memory_md(pkg, current_task_id="t1")
        assert "Cross-task patterns" in md
        assert "verify output format" in md
        assert "brute force" in md
        assert "Evidence posts: #8" in md

    def test_no_bundles_no_section(self):
        pkg = {"workstream_name": "solver"}
        md = seed_package_to_memory_md(pkg, current_task_id="t1")
        assert "Task-specific guidance" not in md
        assert "Cross-task patterns" not in md

    def test_empty_bundle_dict_no_section(self):
        pkg = {
            "workstream_name": "solver",
            "per_task_bundle": {},
            "cross_task_bundle": {},
        }
        md = seed_package_to_memory_md(pkg, current_task_id="t1")
        assert "Task-specific guidance" not in md
        assert "Cross-task patterns" not in md

    def test_per_task_bundle_capped_at_10_items(self):
        pkg = {
            "workstream_name": "solver",
            "per_task_bundle": {
                "transferable_insights": [f"Insight number {i}" for i in range(15)],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [],
            },
        }
        md = seed_package_to_memory_md(pkg, current_task_id="t1")
        assert "Insight number 9" in md
        assert "Insight number 10" not in md

    def test_per_task_bundle_sanitized(self):
        pkg = {
            "workstream_name": "solver",
            "per_task_bundle": {
                "transferable_insights": [
                    "Container exited with code 1: 400 Invalid prompt: your prompt was flagged as bad",
                ],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [],
            },
        }
        md = seed_package_to_memory_md(pkg, current_task_id="t1")
        assert "Task-specific guidance" in md
        assert "flagged as bad" not in md
        assert "Previous run was blocked before model execution" in md
        assert "avoid repeating the flagged wording verbatim" in md


# ── Cross-task forum timeout selection ──────────────────────────────────────


class TestCrossTaskForumTimeout:
    """Verify ``cross_task_forum`` tasks use ``cross_task_forum_timeout_sec``
    instead of the default runtime timeout (or the per-task forum timeout).
    """

    def _make_executor(self, tmp_path, **kw):
        defaults = dict(
            command=["echo", "dummy"],
            working_dir=str(tmp_path),
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )
        defaults.update(kw)
        return KsiContainerExecutor(**defaults)

    def _run_and_capture_timeout_env(
        self,
        executor: KsiContainerExecutor,
        task: TaskSpec,
    ) -> dict:
        seen = {}

        def fake_run(cmd, *, cwd, env, timeout):
            seen["env"] = dict(env)
            seen["timeout"] = timeout
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(task_id=task.id),
                stderr="",
            )

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            executor.run_task(
                generation=1,
                agent_id="agent-0",
                task=task,
                agent_seed_package={},
                experiment_name="test_exp",
            )
        return seen

    def test_cross_task_forum_uses_cross_task_timeout(self, tmp_path):
        ex = self._make_executor(
            tmp_path,
            timeout_sec=1800,
            forum_timeout_sec=900,
            cross_task_forum_timeout_sec=1234,
        )
        task = TaskSpec(
            id="__cross_task_forum__g1_r0_agent-0",
            prompt="cross task",
            metadata={"task_source": "cross_task_forum"},
        )
        seen = self._run_and_capture_timeout_env(ex, task)
        # Backstop is effective_timeout + 120 when effective_timeout > 0.
        assert seen["timeout"] == 1234 + 120
        # CONTAINER_TIMEOUT is in ms and reduced by 15s safety margin.
        expected_ms = max(1234 - 15, 300) * 1000
        assert seen["env"].get("CONTAINER_TIMEOUT") == str(expected_ms)

    def test_per_task_forum_still_uses_forum_timeout(self, tmp_path):
        ex = self._make_executor(
            tmp_path,
            timeout_sec=1800,
            forum_timeout_sec=777,
            cross_task_forum_timeout_sec=1234,
        )
        task = TaskSpec(
            id="__forum__g1_r0_agent-0",
            prompt="debate",
            metadata={"task_source": "per_task_forum"},
        )
        seen = self._run_and_capture_timeout_env(ex, task)
        assert seen["timeout"] == 777 + 120

    def test_regular_task_uses_default_timeout(self, tmp_path):
        ex = self._make_executor(
            tmp_path,
            timeout_sec=1800,
            forum_timeout_sec=777,
            cross_task_forum_timeout_sec=1234,
        )
        task = TaskSpec(
            id="t1",
            prompt="solve",
            metadata={"task_source": "polyglot"},
        )
        seen = self._run_and_capture_timeout_env(ex, task)
        assert seen["timeout"] == 1800 + 120


# ── SWE-bench Pro official container selection ───────────────────────────────


class TestSwebenchProOfficialContainers:
    def _make_executor(self, tmp_path, **kw):
        defaults = dict(
            command=["echo", "dummy"],
            working_dir=str(tmp_path),
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
                "SWEBENCH_PRO_DOCKERHUB_USERNAME": "customhub",
            },
        )
        defaults.update(kw)
        return KsiContainerExecutor(**defaults)

    def test_official_base_image_prefers_image_name(self):
        task = TaskSpec(
            id="instance_demo__repo-123",
            repo="demo/repo",
            metadata={
                "task_source": "swebench_pro",
                "image_name": "123456789012.dkr.ecr.us-west-2.amazonaws.com/sweap-images/demo:repo-123",
                "dockerhub_tag": "demo.repo-ignored",
            },
        )

        assert _swebench_official_base_image(task, {}) == (
            "123456789012.dkr.ecr.us-west-2.amazonaws.com/sweap-images/demo:repo-123"
        )

    def test_official_base_image_uses_dockerhub_tag_and_username(self):
        task = TaskSpec(
            id="instance_demo__repo-123",
            repo="demo/repo",
            metadata={
                "task_source": "swebench_pro",
                "dockerhub_tag": "demo.repo-instance_demo__repo-123",
            },
        )

        assert _swebench_official_base_image(task, {"SWEBENCH_PRO_DOCKERHUB_USERNAME": "otherhub"}) == (
            "otherhub/sweap-images:demo.repo-instance_demo__repo-123"
        )

    def test_run_task_uses_derived_official_container_image(self, tmp_path):
        captured: dict[str, object] = {}

        def fake_run(cmd, *, cwd, env, timeout):
            with open(cmd[-1]) as f:
                captured["payload"] = json.load(f)
            captured["env"] = dict(env)
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(task_id="instance_demo__repo-123"),
                stderr="",
            )

        with (
            patch(
                "ksi.runtime.swebench_images._ensure_swebench_agent_image",
                return_value=("ksi-swebench-pro-agent:abc123", "ksi-agent:bench"),
            ) as ensure_image,
            # Overlay is now built only for a positively-detected glibc base
            # (fail-closed). Pin glibc so this overlay-wiring test exercises the
            # build path without a real docker libc probe / pre-pull.
            patch(
                "ksi.runtime.swebench_images._detect_base_image_libc",
                return_value="glibc",
            ),
            patch(
                "ksi.runtime.container_host._run_command_with_backstop",
                side_effect=fake_run,
            ),
        ):
            ex = self._make_executor(tmp_path, knowledge_db_path=str(tmp_path / "knowledge.sqlite"))
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(
                    id="instance_demo__repo-123",
                    repo="demo/repo",
                    prompt="Fix the issue.",
                    metadata={
                        "task_source": "swebench_pro",
                        "instance_id": "instance_demo__repo-123",
                        "dockerhub_tag": "demo.repo-instance_demo__repo-123",
                        "selected_test_files_to_run": ["tests/test_widget.py"],
                    },
                ),
                agent_seed_package={},
                experiment_name="test_exp",
            )

        ensure_image.assert_called_once()
        assert ensure_image.call_args.args[0] == "customhub/sweap-images:demo.repo-instance_demo__repo-123"
        payload = captured["payload"]
        env = captured["env"]
        assert isinstance(payload, dict)
        assert isinstance(env, dict)
        assert payload["runtime"]["official_container_image"] == (
            "customhub/sweap-images:demo.repo-instance_demo__repo-123"
        )
        assert payload["runtime"]["container_image"] == "ksi-swebench-pro-agent:abc123"
        assert payload["runtime"]["runner_image"] == "ksi-agent:bench"
        assert payload["runtime"]["repo_container_path"] == "/workspace/task/workspace/repo"
        assert payload["runtime"]["official_repo_container_path"] == "/app"
        assert payload["runtime"]["runner_root"] == "/ksi-runner"
        assert payload["task"]["metadata"]["repo_container_path"] == "/workspace/task/workspace/repo"
        assert payload["task"]["metadata"]["official_repo_container_path"] == "/app"
        # In upstream-strict mode (no swebench_pro_seed_tests flag) the run_script
        # command uses the bare form without leaking the selected-test-files argument.
        assert "cd '/app' && bash /workspace/task/workspace/run_script.sh" in (payload["workspace_seed"]["task_md"])
        # The selected_test_files_to_run argument must NOT appear in the command
        # (test names withheld in upstream-strict mode).
        assert "'tests/test_widget.py'" not in payload["workspace_seed"]["task_md"]
        assert payload["knowledge"]["disable_memory_tools"] is True
        assert env["KSI_CONTAINER_IMAGE"] == "ksi-swebench-pro-agent:abc123"
        assert env["CONTAINER_IMAGE"] == "ksi-swebench-pro-agent:abc123"
        assert env["KSI_TASK_REPO_CONTAINER_PATH"] == "/app"
        assert env["KSI_RUNNER_ROOT"] == "/ksi-runner"


# ── Credential scrubbing ──────────────────────────────────────────────────────


class TestScrubCredentials:
    """API keys in stderr must be scrubbed before trace output."""

    def test_scrub_anthropic_api_key(self):
        text = "Error: Invalid API key sk-ant-api03-abcdef1234567890 in request"
        scrubbed = _scrub_credentials(text)
        assert "sk-ant-api03" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_scrub_bearer_token(self):
        text2 = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.sig"
        scrubbed2 = _scrub_credentials(text2)
        assert "eyJhbGciOi" not in scrubbed2

    def test_passthrough_without_credentials(self):
        text3 = "Normal error message without secrets"
        assert _scrub_credentials(text3) == text3


# ── _build_runner_env ─────────────────────────────────────────────────────────


class TestBuildRunnerEnv:
    """Tests for _build_runner_env(base_env, timeout_sec)."""

    def test_sets_container_timeout_from_timeout_param(self):
        env = _build_runner_env({"FOO": "bar"}, timeout_sec=600)
        # (600 - 15) * 1000 = 585000
        assert env["CONTAINER_TIMEOUT"] == str((600 - 15) * 1000)

    def test_passes_through_existing_keys(self):
        base = {"ANTHROPIC_API_KEY": "sk-test-123", "MODEL": "haiku"}
        env = _build_runner_env(base, timeout_sec=300)
        assert env["ANTHROPIC_API_KEY"] == "sk-test-123"
        assert env["MODEL"] == "haiku"

    def test_does_not_mutate_input_dict(self):
        base = {"KEY": "val"}
        _build_runner_env(base, timeout_sec=100)
        assert "CONTAINER_TIMEOUT" not in base
        assert "LOG_LEVEL" not in base

    def test_sets_parent_process_egress_run_id_by_default(self, monkeypatch):
        monkeypatch.delenv("KSI_RUN_ID", raising=False)
        env = _build_runner_env({}, timeout_sec=120)
        assert env["KSI_RUN_ID"] == f"ksi-{os.getpid()}"

    def test_threads_host_egress_run_id_when_present(self, monkeypatch):
        monkeypatch.setenv("KSI_RUN_ID", "campaign-abc")
        env = _build_runner_env({}, timeout_sec=120)
        assert env["KSI_RUN_ID"] == "campaign-abc"

    def test_base_env_egress_run_id_wins_over_host_default(self, monkeypatch):
        monkeypatch.setenv("KSI_RUN_ID", "host-campaign")
        env = _build_runner_env({"KSI_RUN_ID": "profile-campaign"}, timeout_sec=120)
        assert env["KSI_RUN_ID"] == "profile-campaign"

    def test_sets_default_log_level_and_idle_timeout(self):
        env = _build_runner_env({}, timeout_sec=120)
        assert env["LOG_LEVEL"] == "silent"
        assert env["IDLE_TIMEOUT"] == "60000"

    def test_preserves_existing_log_level(self):
        env = _build_runner_env({"LOG_LEVEL": "debug"}, timeout_sec=120)
        assert env["LOG_LEVEL"] == "debug"

    def test_preserves_existing_idle_timeout(self):
        env = _build_runner_env({"IDLE_TIMEOUT": "5000"}, timeout_sec=120)
        assert env["IDLE_TIMEOUT"] == "5000"

    def test_positive_timeout_scales_and_subtracts_grace(self):
        # timeout=600 -> max(600-15, 300) = 585
        env = _build_runner_env({}, timeout_sec=600)
        assert env["CONTAINER_TIMEOUT"] == str(585 * 1000)

    def test_zero_timeout_uses_default_1800(self):
        # 0 / absent (the CLI default) preserves the historical 1800s hard cap.
        env = _build_runner_env({}, timeout_sec=0)
        assert env["CONTAINER_TIMEOUT"] == str(1800 * 1000)

    def test_negative_timeout_disables_container_timeout(self):
        # A negative timeout is an EXPLICIT opt-in to no hard container cap
        # (TB2/Harbor fairness): CONTAINER_TIMEOUT == "0" tells the TS runner to
        # skip its hard-kill timer.
        env = _build_runner_env({}, timeout_sec=-10)
        assert env["CONTAINER_TIMEOUT"] == "0"

    def test_small_timeout_clamps_to_300(self):
        # timeout=10 -> max(10-15, 300) = max(-5, 300) = 300
        env = _build_runner_env({}, timeout_sec=10)
        assert env["CONTAINER_TIMEOUT"] == str(300 * 1000)

    def test_timeout_just_above_315_gives_exact(self):
        # timeout=320 -> max(320-15, 300) = max(305, 300) = 305
        env = _build_runner_env({}, timeout_sec=320)
        assert env["CONTAINER_TIMEOUT"] == str(305 * 1000)

    def test_threads_semantic_search_flag_from_host_environ(self, monkeypatch):
        # The engine records the authoritative --require-vector decision on
        # os.environ AFTER the provider profile is snapshotted into base_env, so
        # _build_runner_env must thread it from the host env or it never reaches
        # the in-container ``query`` tool (which defaults to semantic ON).
        monkeypatch.setenv("MEMORY_ENABLE_SEMANTIC_SEARCH", "0")
        env = _build_runner_env({}, timeout_sec=600)
        assert env["MEMORY_ENABLE_SEMANTIC_SEARCH"] == "0"

    def test_host_semantic_flag_overrides_stale_base_env_value(self, monkeypatch):
        # Direct assignment, NOT setdefault: the host decision is authoritative
        # and must win over any stale value carried in a provider profile.
        monkeypatch.setenv("MEMORY_ENABLE_SEMANTIC_SEARCH", "0")
        env = _build_runner_env({"MEMORY_ENABLE_SEMANTIC_SEARCH": "1"}, timeout_sec=600)
        assert env["MEMORY_ENABLE_SEMANTIC_SEARCH"] == "0"

    def test_absent_semantic_flag_not_injected(self, monkeypatch):
        monkeypatch.delenv("MEMORY_ENABLE_SEMANTIC_SEARCH", raising=False)
        env = _build_runner_env({}, timeout_sec=600)
        assert "MEMORY_ENABLE_SEMANTIC_SEARCH" not in env


# ── _validate_provider_auth ───────────────────────────────────────────────────


class TestValidateProviderAuth:
    """Tests for _validate_provider_auth(env).

    Every misconfiguration branch raises :class:`AuthenticationFailure` (a
    non-retryable, run-aborting error) rather than a bare ``ValueError`` so a
    missing/invalid credential fast-aborts the run at attempt 1 instead of
    silently failing every task in every generation to 0% solved.
    """

    def test_auth_failure_is_not_retryable_valueerror(self):
        # AuthenticationFailure must NOT be a ValueError (so it is not caught
        # by generic ValueError handlers) but MUST be a RuntimeError so the
        # existing fast-abort paths keyed on RuntimeError still see it.
        assert not issubclass(AuthenticationFailure, ValueError)
        assert issubclass(AuthenticationFailure, RuntimeError)

    def test_valid_anthropic_api_key_passes(self):
        env = {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "api",
            "MODEL": "claude-sonnet-4-6",
            "ANTHROPIC_API_KEY": "sk-ant-real-key",
        }
        # Should not raise
        _validate_provider_auth(env)

    def test_valid_anthropic_subscription_passes(self):
        env = {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "subscription",
            "MODEL": "claude-sonnet-4-6",
            "CLAUDE_CODE_OAUTH_TOKEN": "oauth-tok-123",
        }
        _validate_provider_auth(env)

    def test_valid_openai_passes(self):
        env = {
            "MODEL_PROVIDER": "openai",
            "MODEL": "gpt-4o",
            "OPENAI_API_KEY": "sk-openai-key",
        }
        _validate_provider_auth(env)

    def test_unsupported_provider_raises(self):
        env = {"MODEL_PROVIDER": "google", "MODEL": "gemini"}
        with pytest.raises(AuthenticationFailure, match="Unsupported MODEL_PROVIDER"):
            _validate_provider_auth(env)

    def test_missing_provider_raises(self):
        env = {"MODEL": "some-model"}
        with pytest.raises(AuthenticationFailure, match="Unsupported MODEL_PROVIDER"):
            _validate_provider_auth(env)

    def test_missing_model_raises(self):
        env = {"MODEL_PROVIDER": "anthropic", "MODEL_AUTH_MODE": "api"}
        with pytest.raises(AuthenticationFailure, match="Model is missing"):
            _validate_provider_auth(env)

    def test_empty_model_raises(self):
        env = {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "api",
            "MODEL": "  ",
        }
        with pytest.raises(AuthenticationFailure, match="Model is missing"):
            _validate_provider_auth(env)

    def test_anthropic_missing_api_key_raises(self):
        env = {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "api",
            "MODEL": "claude-sonnet-4-6",
        }
        with pytest.raises(AuthenticationFailure, match="ANTHROPIC_API_KEY"):
            _validate_provider_auth(env)

    def test_anthropic_missing_oauth_token_raises(self):
        env = {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "subscription",
            "MODEL": "claude-sonnet-4-6",
        }
        with pytest.raises(AuthenticationFailure, match="CLAUDE_CODE_OAUTH_TOKEN"):
            _validate_provider_auth(env)

    def test_anthropic_invalid_auth_mode_raises(self):
        env = {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "oauth",
            "MODEL": "claude-sonnet-4-6",
            "ANTHROPIC_API_KEY": "sk-ant-key",
        }
        with pytest.raises(AuthenticationFailure, match="MODEL_AUTH_MODE must be"):
            _validate_provider_auth(env)

    def test_openai_missing_key_raises(self):
        env = {"MODEL_PROVIDER": "openai", "MODEL": "gpt-4o"}
        with pytest.raises(AuthenticationFailure, match="OPENAI_API_KEY"):
            _validate_provider_auth(env)

    def test_provider_is_case_insensitive(self):
        env = {
            "MODEL_PROVIDER": "  Anthropic  ",
            "MODEL_AUTH_MODE": "api",
            "MODEL": "claude-sonnet-4-6",
            "ANTHROPIC_API_KEY": "sk-ant-key",
        }
        _validate_provider_auth(env)


# ── _tail ─────────────────────────────────────────────────────────────────────


class TestTail:
    """Tests for _tail(text, max_chars)."""

    def test_short_text_returns_as_is(self):
        assert _tail("hello", 100) == "hello"

    def test_exact_length_returns_as_is(self):
        assert _tail("abcde", 5) == "abcde"

    def test_long_text_truncated_from_start(self):
        result = _tail("abcdefghij", 5)
        assert result == "fghij"
        assert len(result) == 5

    def test_empty_string_returns_empty(self):
        assert _tail("", 10) == ""

    def test_none_input_returns_empty(self):
        assert _tail(None, 10) == ""

    def test_max_chars_zero_returns_full_string(self):
        # value[-0:] == value, so max_chars=0 returns the full string
        assert _tail("abc", 0) == "abc"

    def test_single_char_boundary(self):
        assert _tail("abcdef", 1) == "f"


# ── Forum MEMORY.md uses real task id, not synthetic forum id ────────────────


class TestForumMemoryMdTaskId:
    """Verify run_task prefers assigned_task_id over task.id for memory rendering."""

    def _make_executor(self, tmp_path, **kw):
        defaults = dict(
            command=["echo", "dummy"],
            working_dir=str(tmp_path),
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )
        defaults.update(kw)
        return KsiContainerExecutor(**defaults)

    def test_forum_task_uses_assigned_task_id_in_memory_md(self, tmp_path):
        """Forum path: assigned_task_id from seed_package is used, not synthetic task.id."""
        captured = []

        def fake_run(cmd, **kw):
            with open(cmd[-1]) as f:
                captured.append(json.load(f))
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(task_id="__forum__g1_r0_agent-0"),
                stderr="",
            )

        seed_package = {
            "assigned_task_id": "real-task-001",
            "workstream_name": "solver",
            "prior_attempts": [
                {
                    "task_id": "real-task-001",
                    "score": 0.5,
                    "summary": "Partial solution found",
                }
            ],
        }

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            ex = self._make_executor(tmp_path)
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(id="__forum__g1_r0_agent-0", prompt="discuss"),
                agent_seed_package=seed_package,
                experiment_name="test_exp",
            )

        assert len(captured) == 1
        payload = captured[0]
        memory_md = payload["workspace_seed"]["memory_md"]
        assert "real-task-001" in memory_md
        assert "__forum__g1_r0_agent-0" not in memory_md

    def test_task_path_assigned_task_id_matches_task_id(self, tmp_path):
        """Task-execution path: assigned_task_id == task.id so behavior is identical."""
        captured = []

        def fake_run(cmd, **kw):
            with open(cmd[-1]) as f:
                captured.append(json.load(f))
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(task_id="real-task-002"),
                stderr="",
            )

        seed_package = {
            "assigned_task_id": "real-task-002",
            "workstream_name": "solver",
        }

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            ex = self._make_executor(tmp_path)
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(id="real-task-002", prompt="solve"),
                agent_seed_package=seed_package,
                experiment_name="test_exp",
            )

        assert len(captured) == 1
        payload = captured[0]
        memory_md = payload["workspace_seed"]["memory_md"]
        assert "real-task-002" in memory_md

    def test_no_assigned_task_id_falls_back_to_task_id(self, tmp_path):
        """Fallback: seed_package without assigned_task_id uses task.id."""
        captured = []

        def fake_run(cmd, **kw):
            with open(cmd[-1]) as f:
                captured.append(json.load(f))
            return MagicMock(
                returncode=0,
                stdout=_runner_stdout(task_id="fallback-task-003"),
                stderr="",
            )

        seed_package = {
            "workstream_name": "solver",
        }

        with patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ):
            ex = self._make_executor(tmp_path)
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(id="fallback-task-003", prompt="solve"),
                agent_seed_package=seed_package,
                experiment_name="test_exp",
            )

        assert len(captured) == 1
        payload = captured[0]
        memory_md = payload["workspace_seed"]["memory_md"]
        assert "fallback-task-003" in memory_md


# ── run_task timeout path ────────────────────────────────────────────────────


class TestRunTaskTimeout:
    """Tests for the timeout handling in KsiContainerExecutor.run_task()."""

    def _make_executor(self, tmp_path, **kw):
        defaults = dict(
            command=["echo", "dummy"],
            working_dir=str(tmp_path),
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            timeout_sec=60,
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )
        defaults.update(kw)
        return KsiContainerExecutor(**defaults)

    def test_timeout_expired_raises_runtime_error(self, tmp_path):
        """When _run_runner_command raises TimeoutExpired, run_task wraps it in RuntimeError."""
        ex = self._make_executor(tmp_path)

        with patch.object(
            ex,
            "_run_runner_command",
            side_effect=subprocess.TimeoutExpired(cmd="test", timeout=60),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                ex.run_task(
                    generation=1,
                    agent_id="agent-0",
                    task=TaskSpec(id="task-timeout", prompt="solve this"),
                    agent_seed_package={},
                )

    def test_timeout_error_includes_task_id(self, tmp_path):
        """The RuntimeError message should include the task ID."""
        ex = self._make_executor(tmp_path)

        with patch.object(
            ex,
            "_run_runner_command",
            side_effect=subprocess.TimeoutExpired(cmd="test", timeout=60),
        ):
            with pytest.raises(RuntimeError, match="task-xyz-123"):
                ex.run_task(
                    generation=1,
                    agent_id="agent-0",
                    task=TaskSpec(id="task-xyz-123", prompt="solve"),
                    agent_seed_package={},
                )

    def test_nonzero_exit_raises_runtime_error(self, tmp_path):
        """When runner returns non-zero exit code, run_task raises RuntimeError."""
        ex = self._make_executor(tmp_path)

        mock_result = MagicMock(
            returncode=1,
            stdout="",
            stderr="some error occurred",
        )
        with patch.object(ex, "_run_runner_command", return_value=mock_result):
            with pytest.raises(RuntimeError, match="failed.*exit=1"):
                ex.run_task(
                    generation=1,
                    agent_id="agent-0",
                    task=TaskSpec(id="task-fail", prompt="solve"),
                    agent_seed_package={},
                )

    def test_empty_command_raises(self, tmp_path):
        """When command is empty, run_task raises RuntimeError before subprocess."""
        ex = self._make_executor(tmp_path, command=[])
        with pytest.raises(RuntimeError, match="command must be configured"):
            ex.run_task(
                generation=1,
                agent_id="agent-0",
                task=TaskSpec(id="task-empty", prompt="solve"),
                agent_seed_package={},
            )


def test_swebench_overlay_dockerfile_does_not_copy_npm_npx_binaries() -> None:
    """The overlay must not COPY /usr/local/bin/npm or /usr/local/bin/npx
    from the runner image.

    In the runner image those paths are SYMLINKS to
    ../lib/node_modules/npm/bin/{npm,npx}-cli.js. Docker's COPY follows
    symlinks and writes the dereferenced content into /usr/local/bin/.
    The relocated *-cli.js scripts then do `require('../lib/cli.js')`
    relative to /usr/local/bin/ — which lands on /usr/lib/cli.js (missing).
    Result: every overlay container exits at startup with
    `Cannot find module '../lib/cli.js'` the moment the entrypoint runs
    `npx tsc` to recompile mounted TypeScript. We instead recreate the
    symlinks at the destination so npm/npx resolve their main module via
    the relative path that npm itself bakes in.
    """
    dockerfile = _swebench_agent_overlay_dockerfile(
        base_image="jefzda/sweap-images:demo",
        runner_image="ksi-agent:bench",
    )

    assert "COPY --from=ksi_runner /usr/local/bin/npm" not in dockerfile, (
        "Direct COPY of /usr/local/bin/npm dereferences the symlink and "
        "breaks npm-cli.js's relative `require('../lib/cli.js')` resolution."
    )
    assert "COPY --from=ksi_runner /usr/local/bin/npx" not in dockerfile, (
        "Direct COPY of /usr/local/bin/npx dereferences the symlink and "
        "breaks npx-cli.js's relative `require('../lib/cli.js')` resolution."
    )
    # The recovery path must be a symlink from /usr/local/bin/ into
    # ../lib/node_modules/npm/bin/, identical to the layout npm itself ships.
    assert "ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm" in dockerfile
    assert "ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx" in dockerfile
    # node_modules and the node binary must still come over from the runner.
    assert "COPY --from=ksi_runner /usr/local/bin/node /usr/local/bin/node" in dockerfile
    assert "COPY --from=ksi_runner /usr/local/lib/node_modules /usr/local/lib/node_modules" in dockerfile


def _swebench_pro_task() -> TaskSpec:
    return TaskSpec(
        id="instance_demo__repo-1",
        repo="demo/repo",
        metadata={"task_source": "swebench_pro", "dockerhub_tag": "demo.repo-1"},
    )


def test_swebench_overlay_skips_on_inconclusive_libc(monkeypatch) -> None:
    """Fail CLOSED: an inconclusive libc probe ('') must NOT build the glibc-node
    overlay.

    The overlay COPYs a glibc `node` into the task's base image. On a musl
    (Alpine) base that node exit-127s at container start (`fcntl64: symbol not
    found`). The libc probe returns '' when it times out on a cold ~5GB pull, so
    a musl base whose probe timed out was silently given the overlay and every
    task on it crashed non-retryably. Skipping unless positively glibc routes
    the task to the working legacy shared-runner instead.
    """
    from ksi.runtime import swebench_images as si

    monkeypatch.setattr(si, "_swebench_official_base_image", lambda t, e: "jefzda/sweap-images:demo.repo-1")
    monkeypatch.setattr(si, "_detect_base_image_libc", lambda b, e=None: "")

    def _must_not_build(*a, **k):
        raise AssertionError("overlay built on inconclusive libc — must fail closed to the shared runner")

    monkeypatch.setattr(si, "_ensure_swebench_agent_image", _must_not_build)
    assert si._swebench_pro_container_images(_swebench_pro_task(), {}) == {}


def test_swebench_overlay_skips_on_musl_libc(monkeypatch) -> None:
    """Regression guard: a positively-detected musl base skips the overlay."""
    from ksi.runtime import swebench_images as si

    monkeypatch.setattr(si, "_swebench_official_base_image", lambda t, e: "jefzda/sweap-images:demo.repo-1")
    monkeypatch.setattr(si, "_detect_base_image_libc", lambda b, e=None: "musl")
    monkeypatch.setattr(
        si,
        "_ensure_swebench_agent_image",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("overlay built on musl base")),
    )
    assert si._swebench_pro_container_images(_swebench_pro_task(), {}) == {}


def test_swebench_overlay_builds_on_glibc_libc(monkeypatch) -> None:
    """Positive path preserved: a positively-detected glibc base builds the overlay."""
    from ksi.runtime import swebench_images as si

    monkeypatch.setattr(si, "_swebench_official_base_image", lambda t, e: "jefzda/sweap-images:demo.repo-1")
    monkeypatch.setattr(si, "_detect_base_image_libc", lambda b, e=None: "glibc")
    monkeypatch.setattr(si, "_ensure_swebench_agent_image", lambda base, env: ("agent:tag", "runner:tag"))
    out = si._swebench_pro_container_images(_swebench_pro_task(), {})
    assert out.get("official_container_image") == "jefzda/sweap-images:demo.repo-1"
    assert out.get("container_image") == "agent:tag"


def test_detect_base_image_libc_does_not_cache_inconclusive(monkeypatch) -> None:
    """A '' (inconclusive) probe result must not be cached — a later warm re-probe
    (after the image finishes pulling) must be able to return the real libc."""
    from ksi.runtime import swebench_images as si

    si._BASE_IMAGE_LIBC_CACHE.pop("some/base:img", None)
    calls = {"n": 0}

    class _Proc:
        def __init__(self, out):
            self.stdout = out

    def fake_run(*a, **k):
        calls["n"] += 1
        # first probe inconclusive (empty), second probe succeeds
        return _Proc("" if calls["n"] == 1 else "musl\n")

    monkeypatch.setattr(si.subprocess, "run", fake_run)
    monkeypatch.setattr(si, "_ensure_base_image_present", lambda b, e=None: None)
    assert si._detect_base_image_libc("some/base:img") == ""
    # not cached → re-probes and now gets the real answer
    assert si._detect_base_image_libc("some/base:img") == "musl"
    assert calls["n"] == 2


def _arc_task(test_inputs):
    return TaskSpec(
        id="arc-multitest",
        repo="",
        prompt="arc",
        metadata={
            "task_source": "arc",
            "arc_train_pairs": [{"input": [[0]], "output": [[1]]}],
            "arc_test_inputs": [{"input": grid} for grid in test_inputs],
        },
    )


def test_arc_attempt_stub_files_single_test_only_legacy():
    """A 1-test ARC no-MCP task seeds ONLY the legacy attempt_1/attempt_2 files
    (byte-identical to the pre-multi-test behavior — no per-test files)."""
    files = _arc_attempt_stub_files(_arc_task([[[0]]]))
    assert set(files) == {"attempt_1.txt", "attempt_2.txt"}
    assert all(v == _ARC_ATTEMPT_PRESTUB for v in files.values())


def test_arc_attempt_stub_files_multi_test_per_test_files():
    """A 2-test ARC no-MCP task seeds per-test attempt_<k>_<t>.txt files
    (k in 0..1, t in 1..2) plus the legacy safety-net files, all sentinel."""
    files = _arc_attempt_stub_files(_arc_task([[[0]], [[0]]]))
    assert set(files) == {
        "attempt_1.txt",
        "attempt_2.txt",
        "attempt_0_1.txt",
        "attempt_0_2.txt",
        "attempt_1_1.txt",
        "attempt_1_2.txt",
    }
    assert all(v == _ARC_ATTEMPT_PRESTUB for v in files.values())


def test_arc_attempt_prestub_is_non_parseable_sentinel(tmp_path):
    """The pre-populated `attempt_{1,2}.txt` files must NOT parse as a valid
    grid. Otherwise the TS synthesizer's `parseAsciiGrid` succeeds and a
    `arc_submit_trial` tool call is synthesized with that grid -- which
    would silently credit a free submission to any task whose expected
    output happens to match (e.g. the previous `"0\\n"` stub matched
    `[[0]]` outputs).

    Both the embedded `_parse_grid_text` (validate_prediction.py shipped
    into every ARC workspace) and the TS `parseAsciiGrid` accept only
    integer cells in 0..9; underscores and letters are rejected. We
    exercise the validate script directly here since it shares the parser
    logic.
    """
    assert _ARC_ATTEMPT_PRESTUB.strip(), "prestub must not be empty"
    assert any(not c.isdigit() and not c.isspace() for c in _ARC_ATTEMPT_PRESTUB), (
        "prestub must contain a non-digit char so parseAsciiGrid rejects it"
    )

    script = tmp_path / "validate_prediction.py"
    script.write_text(_ARC_VALIDATE_PREDICTION_SCRIPT, encoding="utf-8")
    sentinel_path = tmp_path / "attempt_1.txt"
    sentinel_path.write_text(_ARC_ATTEMPT_PRESTUB, encoding="utf-8")
    result = subprocess.run(
        ["python3", str(script), str(sentinel_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0, (
        f"validate_prediction must reject the sentinel prestub, "
        f"got exit {result.returncode}: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "non-integer" in result.stderr.lower() or "validate_prediction" in result.stderr, (
        f"expected validation-error message, got stderr={result.stderr!r}"
    )

    # And a real ASCII grid must still pass through, so the overwrite path
    # the agent uses to actually submit stays unbroken.
    real_grid = tmp_path / "attempt_2.txt"
    real_grid.write_text("0 1 2\n3 4 5\n", encoding="utf-8")
    ok = subprocess.run(
        ["python3", str(script), str(real_grid)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert ok.returncode == 0, (
        f"validate_prediction must accept a real grid, got exit {ok.returncode}: stderr={ok.stderr!r}"
    )


class TestSnapshotCleanupOnFallbackTimeout:
    """Regression for issue #843: the per-task memory snapshot lives in the
    persistent knowledge-DB directory (not a TemporaryDirectory) and must be
    unlinked on every exit path — including when the tsx → npx fallback times
    out. Otherwise stale ``memory_snapshot_*.json`` files accumulate forever.
    """

    def _make_executor(self, tmp_path, **kw):
        defaults = dict(
            command=["echo", "dummy"],
            working_dir=str(tmp_path),
            instruction_path=str(tmp_path / "INSTRUCTION.md"),
            agent_workspace_root=str(tmp_path / "workspaces"),
            env={
                "MODEL_PROVIDER": "anthropic",
                "MODEL_AUTH_MODE": "api",
                "MODEL": "claude-sonnet-4-6",
                "ANTHROPIC_API_KEY": "sk-test",
            },
        )
        defaults.update(kw)
        return KsiContainerExecutor(**defaults)

    def test_snapshot_unlinked_when_npx_fallback_times_out(self, tmp_path):
        knowledge_db_path = tmp_path / "foo_knowledge.sqlite"
        ex = self._make_executor(tmp_path, knowledge_db_path=str(knowledge_db_path))

        calls = {"n": 0}
        snapshot_visible_during = {}

        def fake_run(cmd, *, cwd, env, timeout):
            calls["n"] += 1
            snapshot_visible_during[calls["n"]] = bool(list(knowledge_db_path.parent.glob("memory_snapshot_*.json")))
            if calls["n"] == 1:
                # Primary tsx command: non-zero exit with the sentinel that
                # triggers the npx fallback path.
                return MagicMock(returncode=2, stdout="", stderr="tsx: not found")
            # npx fallback: time out.
            raise subprocess.TimeoutExpired(cmd, timeout)

        with patch.object(
            KsiContainerExecutor,
            "_run_runner_command",
            side_effect=fake_run,
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                ex.run_task(
                    generation=1,
                    agent_id="agent-0",
                    task=TaskSpec(id="t1", prompt="solve"),
                    agent_seed_package={"memory_snapshot": {"items": []}},
                    experiment_name="test_exp",
                )

        # Both the primary and the fallback must have run.
        assert calls["n"] == 2
        # The fallback re-reads the same payload.json (which references the
        # snapshot via payload.knowledge.snapshot_path), so the snapshot must
        # still exist while the fallback runs -- it must NOT be cleaned by the
        # primary command's finally.
        assert snapshot_visible_during[1] is True
        assert snapshot_visible_during[2] is True, "snapshot was deleted before the npx fallback could read it"
        # ...and it must be cleaned up on the fallback-timeout exit path.
        leaked = list(knowledge_db_path.parent.glob("memory_snapshot_*.json"))
        assert leaked == [], f"snapshot not cleaned up after fallback timeout: {leaked}"

    def test_watcher_stays_alive_through_npx_fallback(self, tmp_path):
        """Finding #7 (fixed): barrier watchers (phase1/cross_task/polyglot_tf)
        must stay alive through the npx-tsx fallback, not just the primary
        invocation -- otherwise a fallback-run container's barrier sentinels
        get no host response. Verify ``.stop()`` fires exactly once, AFTER
        both the primary and fallback commands have run (not between them)."""
        knowledge_db_path = tmp_path / "foo_knowledge.sqlite"
        ex = self._make_executor(tmp_path, knowledge_db_path=str(knowledge_db_path))

        call_order = []

        def fake_run(cmd, *, cwd, env, timeout):
            call_order.append("primary" if len(call_order) == 0 else "fallback")
            if len(call_order) == 1:
                return MagicMock(returncode=2, stdout="", stderr="tsx: not found")
            return MagicMock(returncode=0, stdout='{"result":"ok"}', stderr="")

        fake_watcher = MagicMock()
        fake_watcher.stop.side_effect = lambda: call_order.append("watcher.stop")

        payload_path = tmp_path / "payload.json"
        payload_path.write_text("{}", encoding="utf-8")

        with patch.object(KsiContainerExecutor, "_run_runner_command", side_effect=fake_run):
            ex._execute_runner_with_fallback(
                payload_path=payload_path,
                runner_env={},
                backstop=None,
                effective_timeout=60,
                task=TaskSpec(id="t1", prompt="solve"),
                phase1_state={"watcher": fake_watcher, "workspace_file": None},
                cross_task_state={"watcher": None, "workspace_file": None},
                polyglot_tf_state={"watcher": None, "workspace_file": None},
                snapshot_path=None,
            )

        fake_watcher.stop.assert_called_once()
        assert call_order == ["primary", "fallback", "watcher.stop"], (
            f"watcher.stop() must fire after BOTH commands, not between them: {call_order}"
        )

    def test_snapshot_unlinked_after_successful_npx_fallback(self, tmp_path):
        # Companion to the timeout case: when the npx fallback SUCCEEDS, the
        # snapshot must still survive into the fallback runner (which re-reads
        # payload.json) and then be cleaned up by the outer finally on the
        # normal exit path -- not only on the timeout exit path.
        knowledge_db_path = tmp_path / "foo_knowledge.sqlite"
        ex = self._make_executor(tmp_path, knowledge_db_path=str(knowledge_db_path))

        calls = {"n": 0}
        snapshot_visible_during = {}

        def fake_run(cmd, *, cwd, env, timeout):
            calls["n"] += 1
            snapshot_visible_during[calls["n"]] = bool(list(knowledge_db_path.parent.glob("memory_snapshot_*.json")))
            if calls["n"] == 1:
                # Primary tsx command: non-zero exit with the sentinel that
                # triggers the npx fallback path.
                return MagicMock(returncode=2, stdout="", stderr="tsx: not found")
            # npx fallback: succeeds (returncode 0). Empty stdout then trips the
            # downstream protocol parse -- but the snapshot lifecycle under test
            # has already completed in the outer finally before that point.
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(
            KsiContainerExecutor,
            "_run_runner_command",
            side_effect=fake_run,
        ):
            with pytest.raises(RuntimeError, match="protocol error"):
                ex.run_task(
                    generation=1,
                    agent_id="agent-0",
                    task=TaskSpec(id="t1", prompt="solve"),
                    agent_seed_package={"memory_snapshot": {"items": []}},
                    experiment_name="test_exp",
                )

        # Both the primary and the (successful) fallback must have run.
        assert calls["n"] == 2
        assert snapshot_visible_during[1] is True
        assert snapshot_visible_during[2] is True, "snapshot was deleted before the npx fallback could read it"
        # ...and it must be cleaned up on the normal (non-timeout) exit path too.
        leaked = list(knowledge_db_path.parent.glob("memory_snapshot_*.json"))
        assert leaked == [], f"snapshot not cleaned up after successful fallback: {leaked}"


def test_polyglot_workspace_runtime_meta_points_at_seeded_repo_dir(tmp_path):
    """The sentinel lives at the workspace ROOT (the host dir the container
    mounts as ``/workspace/task``; TS side passes ``workspaceDir:
    CONTAINER_WORKSPACE_ROOT``), but ``seedWorkspace`` (workspace.ts) copies
    the agent's exercise repo to ``<root>/workspace/repo`` — the same dir
    ``captureWorkspaceArtifacts`` (main.ts) uses for the post-session
    ``host_workspace_repo_dir``. The helper must return that repo dir, not
    the sentinel's parent: ``_workspace_repo_dir`` only normalizes one
    ``repo/`` level, so handing it the root silently resolved to a dir with
    no solution files and mid-loop evals fell back to fenced-block
    extraction (92/12-run forensics: real workspace code scored
    ``no_solution`` without any test run).
    """
    from ksi.runtime.barrier import BarrierEvent
    from ksi.runtime.container_host import _polyglot_workspace_runtime_meta

    workspace_root = tmp_path / "ws-root"
    repo_dir = workspace_root / "workspace" / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "solution.py").write_text("def add(a, b): return a + b\n")
    event = BarrierEvent(
        sentinel_path=workspace_root / ".barrier.polyglot_test_feedback.agent.sentinel",
        response_path=workspace_root / ".barrier.polyglot_test_feedback.agent.response",
        payload={},
    )

    runtime_meta = _polyglot_workspace_runtime_meta(event.sentinel_path)

    assert runtime_meta == {"host_workspace_repo_dir": str(repo_dir)}


def test_polyglot_feedback_barrier_eval_scores_production_workspace_layout(tmp_path):
    """End-to-end regression for the mid-loop no_solution bug: with the REAL
    ``PolyglotHarnessEvaluator`` (skip_docker) and the PRODUCTION workspace
    layout (sentinel at the root, solution files under
    ``<root>/workspace/repo/``), a barrier-round model_output with NO fenced
    code block must still resolve ``solution_source=workspace_files`` — not
    fall through to ``status=no_solution`` with zero test runs.
    """
    from ksi.benchmarks.polyglot_harness import PolyglotHarnessEvaluator
    from ksi.runtime.barrier import BarrierEvent
    from ksi.runtime.container_host import _polyglot_workspace_runtime_meta

    workspace_root = tmp_path / "ws-root"
    repo_dir = workspace_root / "workspace" / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "solution.py").write_text("def add(a, b): return a + b\n")

    task = TaskSpec(
        id="python__adder",
        repo="",
        prompt="",
        metadata={"task_source": "polyglot", "language": "python", "starter_code": {"solution.py": ""}},
    )
    event = BarrierEvent(
        sentinel_path=workspace_root / ".barrier.polyglot_test_feedback.agent.sentinel",
        response_path=workspace_root / ".barrier.polyglot_test_feedback.agent.response",
        payload={"model_output": "I edited solution.py in place; tests should pass now."},
    )
    evaluator = PolyglotHarnessEvaluator(skip_docker=True)

    result = evaluator.evaluate(
        task=task,
        model_output=str(event.payload["model_output"]),
        runtime_meta=_polyglot_workspace_runtime_meta(event.sentinel_path),
        tool_trace=[],
    )

    assert result["status"] != "no_solution"
    assert result["solution_source"] == "workspace_files"
    assert result["extracted_files"] == ["solution.py"]


def test_launch_polyglot_test_feedback_watcher_scores_live_workspace_files(tmp_path):
    """The polyglot test-feedback watcher must (a) pass a real
    ``host_workspace_repo_dir`` (via ``_polyglot_workspace_runtime_meta``) to
    ``evaluator.evaluate`` so mid-loop scoring reflects the agent's live
    on-disk edits, (b) return the FULL eval_result (including raw
    test_stdout_tail/test_stderr_tail) across the barrier response for the
    TS-side retry loop, and (c) cache the FULL raw eval_result (Finding #5
    fix: matches phase1's caching so ``_postprocess_runner_output`` can
    reuse it when safe -- this holder is never itself persisted; the
    separate, still-sanitized research-bookkeeping summary is built
    independently on the TS side).
    """
    from ksi.runtime.barrier import response_filename, sentinel_filename

    class _FakeEvaluator:
        def __init__(self):
            self.received_runtime_meta = None

        def evaluate(self, *, task, model_output, runtime_meta, tool_trace):
            self.received_runtime_meta = runtime_meta
            return {
                "native_score": 0.0,
                "resolved": False,
                "status": "ok",
                "test_exit_code": 1,
                "test_stdout_tail": "AssertionError: expected 5 got 6",
                "test_stderr_tail": "",
            }

    evaluator = _FakeEvaluator()
    executor = KsiContainerExecutor(
        command=["fake"],
        working_dir=str(tmp_path),
        timeout_sec=60,
        knowledge_db_path="",
        evaluator=evaluator,
    )
    task = TaskSpec(id="python__bowling", repo="", prompt="", metadata={"task_source": "polyglot"})

    agent_id = "agent-tf"
    workspace_file = tmp_path / "workspace_path.txt"
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    workspace_file.write_text(str(workspace_dir))

    watcher = executor._launch_polyglot_test_feedback_watcher(
        workspace_file=workspace_file,
        agent_id=agent_id,
        task=task,
        poll_timeout_sec=5.0,
        max_rounds=1,
    )
    assert watcher is not None
    try:
        sentinel_path = workspace_dir / sentinel_filename("polyglot_test_feedback", agent_id)
        response_path = workspace_dir / response_filename("polyglot_test_feedback", agent_id)
        sentinel_path.write_text(json.dumps({"model_output": "attempt 1 solution"}), encoding="utf-8")

        deadline = time.monotonic() + 5.0
        response_payload = None
        while time.monotonic() < deadline:
            if response_path.exists():
                try:
                    response_payload = json.loads(response_path.read_text(encoding="utf-8"))
                except Exception:
                    response_payload = None
                break
            time.sleep(0.05)

        assert response_payload is not None, "barrier response never appeared"
        # Full eval_result (including raw tails) crosses the barrier.
        assert response_payload["native_score"] == 0.0
        assert response_payload["test_stdout_tail"] == "AssertionError: expected 5 got 6"
        assert response_payload["test_stderr_tail"] == ""

        # The evaluator saw a real live-workspace runtime_meta pointing at
        # the SEEDED repo dir (<workspace root>/workspace/repo — the same
        # path main.ts's captureWorkspaceArtifacts uses), not {} and not the
        # sentinel's parent (the workspace root has no solution files).
        assert evaluator.received_runtime_meta == {"host_workspace_repo_dir": str(workspace_dir / "workspace" / "repo")}

        # The FULL raw eval_result is cached (including raw stdout/stderr
        # tails) — matches phase1's caching, gated reuse happens downstream.
        cached = getattr(watcher, "_cached_eval_holder", None)
        assert cached is not None
        assert cached["value"] == {
            "native_score": 0.0,
            "resolved": False,
            "status": "ok",
            "test_exit_code": 1,
            "test_stdout_tail": "AssertionError: expected 5 got 6",
            "test_stderr_tail": "",
        }
    finally:
        watcher.stop()
        watcher.join(timeout=2.0)


def test_launch_phase1_watcher_scores_live_workspace_files(tmp_path):
    """Regression for the phase1_reflection barrier scoring from model_output
    only. With the REAL ``PolyglotHarnessEvaluator`` (skip_docker) and the
    PRODUCTION workspace layout (sentinel at the root, solution files under
    ``<root>/workspace/repo/``), a phase1 barrier round whose model_output
    has NO fenced code block must still resolve
    ``solution_source=workspace_files`` — not fall through to
    ``status=no_solution``. Before the fix, phase1's callback passed
    ``runtime_meta={}`` and this exact scenario scored ``no_solution`` with
    no test run, and (because the reflection turn cannot edit the workspace)
    that wrong result was reused verbatim as the FINAL score.
    """
    from ksi.benchmarks.polyglot_harness import PolyglotHarnessEvaluator
    from ksi.runtime.barrier import response_filename, sentinel_filename

    task = TaskSpec(
        id="python__adder",
        repo="",
        prompt="",
        metadata={"task_source": "polyglot", "language": "python", "starter_code": {"solution.py": ""}},
    )
    evaluator = PolyglotHarnessEvaluator(skip_docker=True)
    executor = KsiContainerExecutor(
        command=["fake"],
        working_dir=str(tmp_path),
        timeout_sec=60,
        knowledge_db_path="",
        evaluator=evaluator,
        phase1_reflection_enabled=True,
    )

    agent_id = "agent-p1"
    workspace_file = tmp_path / "workspace_path.txt"
    workspace_root = tmp_path / "ws"
    repo_dir = workspace_root / "workspace" / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "solution.py").write_text("def add(a, b): return a + b\n")
    workspace_file.write_text(str(workspace_root))

    watcher = executor._launch_phase1_watcher(
        workspace_file=workspace_file,
        agent_id=agent_id,
        task=task,
        poll_timeout_sec=5.0,
    )
    assert watcher is not None
    try:
        sentinel_path = workspace_root / sentinel_filename("phase1_reflection", agent_id)
        response_path = workspace_root / response_filename("phase1_reflection", agent_id)
        sentinel_path.write_text(
            json.dumps({"model_output": "I edited solution.py in place; tests should pass now."}),
            encoding="utf-8",
        )

        deadline = time.monotonic() + 5.0
        response_payload = None
        while time.monotonic() < deadline:
            if response_path.exists():
                response_payload = json.loads(response_path.read_text(encoding="utf-8"))
                break
            time.sleep(0.05)

        assert response_payload is not None, "phase1 barrier response never appeared"
        assert response_payload["status"] != "no_solution"
        assert response_payload["solution_source"] == "workspace_files"
        assert response_payload["extracted_files"] == ["solution.py"]

        cached = getattr(watcher, "_cached_eval_holder", None)
        assert isinstance(cached, dict)
        assert cached["value"]["solution_source"] == "workspace_files"
    finally:
        watcher.stop()
        watcher.join(timeout=2.0)


def test_launch_polyglot_test_feedback_watcher_answers_multiple_rounds(tmp_path):
    """``--polyglot-test-feedback-tries`` > 2 means the TS-side retry loop
    (``runPolyglotTestFeedback``) writes MORE than one sentinel over the
    task's lifetime — one per retry round — but the watcher's ``stop()``
    is only called once, after the whole container subprocess exits (see
    ``_execute_runner_with_fallback``). The watcher must therefore be
    ``persistent`` and answer a second (and third) round, not just the
    first — this is the production wiring a single-shot watcher would
    silently break for any ``--polyglot-test-feedback-tries`` above the
    default of 2."""
    from ksi.runtime.barrier import response_filename, sentinel_filename

    class _FakeEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, *, task, model_output, runtime_meta, tool_trace):
            self.calls += 1
            return {
                "native_score": 0.0,
                "resolved": False,
                "status": "ok",
                "test_exit_code": 1,
                "test_stdout_tail": f"failure round {self.calls}",
                "test_stderr_tail": "",
            }

    evaluator = _FakeEvaluator()
    executor = KsiContainerExecutor(
        command=["fake"],
        working_dir=str(tmp_path),
        timeout_sec=60,
        knowledge_db_path="",
        evaluator=evaluator,
    )
    task = TaskSpec(id="python__bowling", repo="", prompt="", metadata={"task_source": "polyglot"})

    agent_id = "agent-tf-multi"
    workspace_file = tmp_path / "workspace_path.txt"
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    workspace_file.write_text(str(workspace_dir))

    watcher = executor._launch_polyglot_test_feedback_watcher(
        workspace_file=workspace_file,
        agent_id=agent_id,
        task=task,
        poll_timeout_sec=5.0,
        max_rounds=2,
    )
    assert watcher is not None
    try:
        sentinel_path = workspace_dir / sentinel_filename("polyglot_test_feedback", agent_id)
        response_path = workspace_dir / response_filename("polyglot_test_feedback", agent_id)

        for round_num in range(1, 3):
            sentinel_path.write_text(json.dumps({"model_output": f"attempt {round_num}"}), encoding="utf-8")
            deadline = time.monotonic() + 5.0
            response_payload = None
            while time.monotonic() < deadline:
                if response_path.exists():
                    try:
                        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
                    except Exception:
                        response_payload = None
                    if response_payload is not None:
                        break
                time.sleep(0.05)
            assert response_payload is not None, f"barrier response never appeared for round {round_num}"
            assert response_payload["test_stdout_tail"] == f"failure round {round_num}"
            response_path.unlink()

        assert evaluator.calls == 2
    finally:
        watcher.stop()
        watcher.join(timeout=2.0)


def test_launch_polyglot_test_feedback_watcher_refuses_rounds_beyond_max_rounds(tmp_path):
    """Security regression guard (PR #1032 deep review, security.md Finding
    1): the host must cap barrier rounds at ``max_rounds`` server-side,
    independent of what the container/TS-side loop claims via
    ``triesRemaining`` -- a Bash-capable agent could otherwise write
    sentinel files directly in a tight loop and force unbounded real Docker
    evaluations."""
    from ksi.runtime.barrier import response_filename, sentinel_filename

    class _FakeEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, *, task, model_output, runtime_meta, tool_trace):
            self.calls += 1
            return {"native_score": 0.0, "resolved": False, "status": "ok", "test_exit_code": 1}

    evaluator = _FakeEvaluator()
    executor = KsiContainerExecutor(
        command=["fake"],
        working_dir=str(tmp_path),
        timeout_sec=60,
        knowledge_db_path="",
        evaluator=evaluator,
    )
    task = TaskSpec(id="python__bowling", repo="", prompt="", metadata={"task_source": "polyglot"})

    agent_id = "agent-tf-capped"
    workspace_file = tmp_path / "workspace_path.txt"
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    workspace_file.write_text(str(workspace_dir))

    watcher = executor._launch_polyglot_test_feedback_watcher(
        workspace_file=workspace_file,
        agent_id=agent_id,
        task=task,
        poll_timeout_sec=5.0,
        max_rounds=1,
    )
    assert watcher is not None
    try:
        sentinel_path = workspace_dir / sentinel_filename("polyglot_test_feedback", agent_id)
        response_path = workspace_dir / response_filename("polyglot_test_feedback", agent_id)

        def _write_and_wait(round_num):
            sentinel_path.write_text(json.dumps({"model_output": f"attempt {round_num}"}), encoding="utf-8")
            deadline = time.monotonic() + 5.0
            payload = None
            while time.monotonic() < deadline:
                if response_path.exists():
                    try:
                        payload = json.loads(response_path.read_text(encoding="utf-8"))
                    except Exception:
                        payload = None
                    if payload is not None:
                        break
                time.sleep(0.05)
            assert payload is not None, f"barrier response never appeared for round {round_num}"
            response_path.unlink()
            return payload

        first = _write_and_wait(1)
        assert first["resolved"] is False
        second = _write_and_wait(2)
        assert second.get("error"), "a round beyond max_rounds must be refused, not evaluated"

        assert evaluator.calls == 1
    finally:
        watcher.stop()
        watcher.join(timeout=2.0)


def test_execute_runner_with_fallback_waits_for_inflight_polyglot_watcher(tmp_path, monkeypatch):
    """Concurrency regression guard (PR #1032 deep review, concurrency-ipc.md
    Finding 1): if the container gives up waiting for a barrier round, the
    finally block must actually wait (bounded, proportional to the
    evaluator's own timeout) for the watcher's in-flight Docker evaluation
    to finish before returning -- otherwise execution_phase.py's fallback
    evaluate() call can run concurrently with an orphaned watcher thread
    still mid-Docker-eval on the same task."""
    import subprocess
    import threading as pythreading

    from ksi.runtime.barrier import sentinel_filename

    class _SlowEvaluator:
        timeout_sec = 1

        def __init__(self):
            self.started = pythreading.Event()
            self.finished = pythreading.Event()

        def evaluate(self, *, task, model_output, runtime_meta, tool_trace):
            self.started.set()
            time.sleep(0.3)
            self.finished.set()
            return {"native_score": 0.0, "resolved": False, "status": "ok", "test_exit_code": 1}

    evaluator = _SlowEvaluator()
    executor = KsiContainerExecutor(
        command=["fake"],
        working_dir=str(tmp_path),
        timeout_sec=60,
        knowledge_db_path="",
        evaluator=evaluator,
    )
    task = TaskSpec(id="python__bowling", repo="", prompt="", metadata={"task_source": "polyglot"})

    agent_id = "agent-tf-wait"
    workspace_file = tmp_path / "workspace_path.txt"
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    workspace_file.write_text(str(workspace_dir))

    watcher = executor._launch_polyglot_test_feedback_watcher(
        workspace_file=workspace_file,
        agent_id=agent_id,
        task=task,
        poll_timeout_sec=5.0,
        max_rounds=1,
    )
    assert watcher is not None

    sentinel_path = workspace_dir / sentinel_filename("polyglot_test_feedback", agent_id)
    sentinel_path.write_text(json.dumps({"model_output": "attempt 1"}), encoding="utf-8")

    # Wait for the callback to actually start running so we're genuinely
    # racing an in-flight evaluate() call, not a not-yet-started one.
    assert evaluator.started.wait(timeout=2.0), "evaluator.evaluate() never started"

    def _fake_run_runner_command(cmd, *, cwd, env, timeout):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(executor, "_run_runner_command", _fake_run_runner_command)

    started_at = time.monotonic()
    executor._execute_runner_with_fallback(
        payload_path=tmp_path / "payload.json",
        runner_env={},
        backstop=None,
        effective_timeout=60,
        task=task,
        phase1_state={"watcher": None, "workspace_file": None},
        cross_task_state={"watcher": None, "workspace_file": None},
        polyglot_tf_state={"watcher": watcher, "workspace_file": workspace_file},
        snapshot_path=None,
    )
    elapsed = time.monotonic() - started_at

    assert evaluator.finished.is_set(), "the finally block must wait for the in-flight evaluate() call to finish"
    assert elapsed >= 0.25, f"returned too quickly ({elapsed:.2f}s) to have genuinely waited for the callback"


def test_maybe_setup_polyglot_test_feedback_scales_min_effective_timeout_with_tries(tmp_path):
    """concurrency-ipc.md Finding 2: raising --polyglot-test-feedback-tries
    above the default must scale the session timeout/backstop, or a
    multi-round retry can hit the flat per-source backstop and hard-fail
    the task instead of gracefully degrading via the barrier protocol's
    own poll timeout."""

    class _FakePolyglotEvaluator:
        timeout_sec = 120

    executor = _make_polyglot_executor(tmp_path, evaluator=_FakePolyglotEvaluator(), timeout_sec=60)
    task = TaskSpec(
        id="rust__wordy",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 5,
            "polyglot_test_feedback_max_lines": 50,
        },
    )
    runner_env, effective_timeout, backstop = executor._build_task_runner_env(
        source="polyglot",
        arc_no_mcp_active=False,
        swebench_container_images={},
        min_effective_timeout=5 * (120 + 60) + 300,
    )
    # The original self.timeout_sec (60) is far too small for 5 tries at a
    # 120s-budget evaluator; the scaled minimum must win.
    assert effective_timeout == 5 * (120 + 60) + 300
    assert backstop == effective_timeout + 120
    # CONTAINER_TIMEOUT (derived from effective_timeout) must reflect the
    # SAME scaled value, not the stale pre-scaling one, or the container's
    # own internal watchdog would still fire early.
    assert runner_env["CONTAINER_TIMEOUT"] == str(max(effective_timeout - 15, 300) * 1000)


def test_launch_polyglot_test_feedback_watcher_returns_none_without_evaluator(tmp_path):
    executor = KsiContainerExecutor(
        command=["fake"],
        working_dir=str(tmp_path),
        timeout_sec=60,
        knowledge_db_path="",
        evaluator=None,
    )
    task = TaskSpec(id="t-noeval", repo="", prompt="", metadata={"task_source": "polyglot"})
    watcher = executor._launch_polyglot_test_feedback_watcher(
        workspace_file=tmp_path / "never_written.txt",
        agent_id="agent-noeval",
        task=task,
        poll_timeout_sec=5.0,
        max_rounds=2,
    )
    assert watcher is None


def _make_polyglot_executor(tmp_path, **kw):
    defaults = dict(
        command=["fake"],
        working_dir=str(tmp_path),
        instruction_path=str(tmp_path / "INSTRUCTION.md"),
        agent_workspace_root=str(tmp_path / "workspaces"),
        knowledge_db_path="",
        env={
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "api",
            "MODEL": "claude-sonnet-4-6",
            "ANTHROPIC_API_KEY": "sk-test",
        },
    )
    defaults.update(kw)
    return KsiContainerExecutor(**defaults)


def test_cross_task_r1_response_poll_timeout_outlives_forum_timeout(tmp_path):
    from ksi.orchestrator.forum_phase import _cross_task_coordinator_timeout_sec

    executor = _make_polyglot_executor(tmp_path)
    payload_path = tmp_path / "payload.json"
    payload: dict = {}
    runner_env: dict[str, str] = {}

    with patch.object(executor, "_launch_cross_task_r1_watcher", return_value=object()) as launch:
        state = executor._maybe_setup_cross_task_r1(
            source="cross_task_forum",
            cross_task_shared_container=True,
            cross_task_r1_callback=lambda event: {"r1_prompt_text": "reply"},
            payload=payload,
            payload_path=payload_path,
            runner_env=runner_env,
            effective_timeout=900,
            td=str(tmp_path),
            agent_id="agent-0",
            phase1_eligible=False,
            phase1_state={},
        )

    cfg = json.loads(payload_path.read_text(encoding="utf-8"))["cross_task_shared_container"]
    assert cfg["enabled"] is True
    assert cfg["barrier_name"] == "cross_task_r1"

    # The REAL invariant: the in-container poll must give up GRACEFULLY (ship
    # an R0-only envelope) BEFORE the container's hard external kill deadline
    # (CONTAINER_TIMEOUT), and only AFTER the coordinator's own bounded wait.
    # A poll timeout >= CONTAINER_TIMEOUT (the old effective_timeout+30 bug)
    # makes the graceful fallback dead code -- the external kill always wins.
    effective_timeout = 900
    container_timeout_ms = max(effective_timeout - 15, 300) * 1000  # _build_runner_env CONTAINER_TIMEOUT
    coord_timeout_ms = int(_cross_task_coordinator_timeout_sec(float(effective_timeout)) * 1000)
    poll_ms = cfg["response_poll_timeout_ms"]
    assert poll_ms == max(30, max(effective_timeout - 15, 300) - 5) * 1000
    assert poll_ms == 880_000
    assert poll_ms < container_timeout_ms, (poll_ms, container_timeout_ms)
    assert poll_ms > coord_timeout_ms, (poll_ms, coord_timeout_ms)
    assert runner_env["KSI_BARRIER_WORKSPACE_FILE"] == str(state["workspace_file"])
    launch.assert_called_once()


def test_maybe_setup_polyglot_test_feedback_enables_for_polyglot_when_tries_gt_1(tmp_path):
    executor = _make_polyglot_executor(tmp_path, evaluator=object())
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 50,
        },
    )
    cfg = executor._maybe_setup_polyglot_test_feedback(task=task, agent_id="agent1")
    assert cfg is not None
    assert cfg["enabled"] is True
    assert cfg["triesRemaining"] == 2
    assert cfg["maxLines"] == 50
    # Evaluator has no ``timeout_sec`` attribute (a plain ``object()``), so the
    # shared ``DEFAULT_POLYGLOT_TIMEOUT_SEC`` fallback applies:
    # max(30_000, (DEFAULT_POLYGLOT_TIMEOUT_SEC+60)*1000).
    assert cfg["evalResultPollTimeoutMs"] == max(30_000, (DEFAULT_POLYGLOT_TIMEOUT_SEC + 60) * 1000)


def test_maybe_setup_polyglot_test_feedback_eval_result_poll_timeout_ms_scales_with_evaluator_timeout(tmp_path):
    """Finding #1 (fixed): the container-side barrier wait must always be
    LONGER than the polyglot evaluator's own Docker-run timeout budget
    (``--polyglot-timeout-sec``), added on top of it (not subtracted from
    it, unlike ``_maybe_setup_phase1_reflection``'s margin, which shrinks a
    much larger session budget) -- otherwise the barrier can give up before
    the host's own Docker run legitimately finishes.
    """

    class _FakePolyglotEvaluator:
        timeout_sec = 180

    executor = _make_polyglot_executor(tmp_path, evaluator=_FakePolyglotEvaluator())
    task = TaskSpec(
        id="rust__wordy",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 50,
        },
    )
    cfg = executor._maybe_setup_polyglot_test_feedback(task=task, agent_id="agent1")
    assert cfg is not None
    # max(30_000, (DEFAULT_POLYGLOT_TIMEOUT_SEC+60)*1000), strictly greater than
    # the evaluator's own 180_000ms worst case.
    assert cfg["evalResultPollTimeoutMs"] == max(30_000, (DEFAULT_POLYGLOT_TIMEOUT_SEC + 60) * 1000)
    assert cfg["evalResultPollTimeoutMs"] > _FakePolyglotEvaluator.timeout_sec * 1000


def test_maybe_setup_polyglot_test_feedback_eval_result_poll_timeout_ms_floors_for_degenerate_timeout(tmp_path):
    """The floor guards a degenerate/misconfigured ``timeout_sec`` (e.g. a
    fake/broken evaluator reporting a non-positive timeout); at any
    realistic positive ``timeout_sec`` the additive margin already exceeds
    the floor."""

    class _FakePolyglotEvaluator:
        timeout_sec = -500

    executor = _make_polyglot_executor(tmp_path, evaluator=_FakePolyglotEvaluator())
    task = TaskSpec(
        id="rust__wordy",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 50,
        },
    )
    cfg = executor._maybe_setup_polyglot_test_feedback(task=task, agent_id="agent1")
    assert cfg is not None
    assert cfg["evalResultPollTimeoutMs"] == 30_000


def test_maybe_setup_polyglot_test_feedback_returns_none_without_evaluator(tmp_path):
    """Finding #2: mirror ``_maybe_setup_phase1_reflection``'s
    ``self.evaluator is not None`` eligibility gate. Without it, a polyglot
    run with no evaluator wired still builds a barrier config the container
    will wait on forever (nothing will ever answer the sentinel)."""
    executor = _make_polyglot_executor(tmp_path, evaluator=None)
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 50,
        },
    )
    assert executor._maybe_setup_polyglot_test_feedback(task=task, agent_id="agent1") is None


def test_maybe_setup_polyglot_test_feedback_disabled_when_tries_is_1(tmp_path):
    """Also the byte-identical backward-compat pin: tries=1 must be a true
    no-op (no barrier config at all), reproducing the old strict single-shot
    protocol exactly."""
    executor = _make_polyglot_executor(tmp_path, evaluator=object())
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 1,
            "polyglot_test_feedback_max_lines": 50,
        },
    )
    assert executor._maybe_setup_polyglot_test_feedback(task=task, agent_id="agent1") is None


def test_maybe_setup_polyglot_test_feedback_disabled_when_tries_is_0(tmp_path):
    """Finding #8 (fixed): an explicit ``polyglot_test_feedback_tries: 0``
    must disable the feature like ``tries=1`` does, NOT get silently
    coerced back to the default (2) by a falsy-``or`` fallback."""
    executor = _make_polyglot_executor(tmp_path, evaluator=object())
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 0,
            "polyglot_test_feedback_max_lines": 50,
        },
    )
    assert executor._maybe_setup_polyglot_test_feedback(task=task, agent_id="agent1") is None


def test_maybe_setup_polyglot_test_feedback_respects_explicit_max_lines_0(tmp_path):
    """Finding #8 (fixed): an explicit ``polyglot_test_feedback_max_lines: 0``
    must be respected, NOT silently coerced back to the default (50)."""
    executor = _make_polyglot_executor(tmp_path, evaluator=object())
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 0,
        },
    )
    cfg = executor._maybe_setup_polyglot_test_feedback(task=task, agent_id="agent1")
    assert cfg is not None
    assert cfg["maxLines"] == 0


def test_phase1_reflection_and_polyglot_test_feedback_can_both_be_eligible(tmp_path):
    """Finding #9: phase1_reflection and polyglot_test_feedback have
    independent eligibility gates with no mutual exclusion -- a polyglot
    task run with ``--phase1-reflection-enabled`` genuinely reaches BOTH
    ``_maybe_setup_phase1_reflection`` and
    ``_maybe_setup_polyglot_test_feedback`` returning non-None. This test
    locks in that both CAN be simultaneously eligible (the combination is
    reachable, not just theoretical) so a future change to either gate
    doesn't silently make them mutually exclusive without a decision to do
    so, or vice versa without test coverage."""
    executor = _make_polyglot_executor(tmp_path, evaluator=object(), phase1_reflection_enabled=True)
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 50,
        },
    )
    polyglot_cfg = executor._maybe_setup_polyglot_test_feedback(task=task, agent_id="agent1")
    assert polyglot_cfg is not None
    phase1_eligible, _phase1_state = executor._maybe_setup_phase1_reflection(
        source="polyglot",
        payload={},
        payload_path=tmp_path / "payload.json",
        runner_env={},
        effective_timeout=60,
        td=str(tmp_path),
        agent_id="agent1",
        task=task,
    )
    assert phase1_eligible is True


def test_maybe_setup_polyglot_test_feedback_disabled_for_non_polyglot(tmp_path):
    executor = _make_polyglot_executor(tmp_path, evaluator=object())
    task = TaskSpec(id="t1", repo="", prompt="", metadata={"task_source": "swebench_pro"})
    assert executor._maybe_setup_polyglot_test_feedback(task=task, agent_id="agent1") is None


def test_run_task_injects_polyglot_test_feedback_into_payload_and_launches_watcher(tmp_path):
    """Task 6 wiring: a polyglot task with tries>1 must get a
    ``polyglot_test_feedback`` block written into payload.json (the same
    JSON main.ts reads to build ``containerInput``), and the host must
    launch the Task-5 barrier watcher and stop it once the runner
    subprocess completes -- mirroring how ``_maybe_setup_phase1_reflection``
    wires ``phase1_reflection`` at its call site.
    """
    fake_watcher = MagicMock()
    ex = _make_polyglot_executor(tmp_path, evaluator=object())
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="solve",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 50,
            "starter_code": {"bowling.py": "..."},
        },
    )

    captured_payload = []
    captured_env = []

    def fake_run(cmd, **kw):
        with open(cmd[-1]) as f:
            captured_payload.append(json.load(f))
        captured_env.append(kw.get("env"))
        return MagicMock(
            returncode=0,
            stdout=_runner_stdout(task_id=task.id, agent_id="agent-tf"),
            stderr="",
        )

    with (
        patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ),
        patch.object(
            KsiContainerExecutor,
            "_launch_polyglot_test_feedback_watcher",
            return_value=fake_watcher,
        ) as mock_launch,
    ):
        ex.run_task(
            generation=1,
            agent_id="agent-tf",
            task=task,
            agent_seed_package={},
            experiment_name="test_exp",
        )

    assert len(captured_payload) == 1
    payload = captured_payload[0]
    cfg = payload.get("polyglot_test_feedback")
    assert cfg is not None
    assert cfg["enabled"] is True
    assert cfg["agentId"] == "agent-tf"
    assert cfg["triesRemaining"] == 2
    assert cfg["maxLines"] == 50

    # The watcher launcher must have been invoked (Task 5's watcher wired in).
    mock_launch.assert_called_once()
    _, launch_kwargs = mock_launch.call_args
    assert launch_kwargs["agent_id"] == "agent-tf"
    assert launch_kwargs["task"] is task

    # runner_env carries the barrier workspace file for the watcher to poll.
    assert captured_env[0] is not None
    assert "KSI_BARRIER_WORKSPACE_FILE" in captured_env[0]

    # The watcher must be stopped once the runner subprocess completes.
    fake_watcher.stop.assert_called_once()


def test_run_task_wires_both_phase1_reflection_and_polyglot_test_feedback(tmp_path):
    """Finding #9: locks in that ``run_task`` handles BOTH features being
    eligible for the same polyglot task without crashing or clobbering each
    other's payload/watcher wiring -- both configs must land in
    payload.json and both watchers must be launched and stopped cleanly."""
    fake_phase1_watcher = MagicMock()
    fake_polyglot_watcher = MagicMock()
    ex = _make_polyglot_executor(tmp_path, evaluator=object(), phase1_reflection_enabled=True)
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="solve",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 50,
            "starter_code": {"bowling.py": "..."},
        },
    )

    captured_payload = []

    def fake_run(cmd, **kw):
        with open(cmd[-1]) as f:
            captured_payload.append(json.load(f))
        return MagicMock(
            returncode=0,
            stdout=_runner_stdout(task_id=task.id, agent_id="agent-tf"),
            stderr="",
        )

    with (
        patch("ksi.runtime.container_host._run_command_with_backstop", side_effect=fake_run),
        patch.object(KsiContainerExecutor, "_launch_phase1_watcher", return_value=fake_phase1_watcher),
        patch.object(
            KsiContainerExecutor,
            "_launch_polyglot_test_feedback_watcher",
            return_value=fake_polyglot_watcher,
        ),
    ):
        ex.run_task(
            generation=1,
            agent_id="agent-tf",
            task=task,
            agent_seed_package={},
            experiment_name="test_exp",
        )

    assert len(captured_payload) == 1
    payload = captured_payload[0]
    assert payload.get("phase1_reflection") is not None
    assert payload.get("polyglot_test_feedback") is not None

    fake_phase1_watcher.stop.assert_called_once()
    fake_polyglot_watcher.stop.assert_called_once()


def test_run_task_reuses_cached_polyglot_eval_when_final_eval_matches_output(tmp_path):
    """Finding #5 (fixed): when the TS side reports
    ``final_eval_matches_output: true`` (the last barrier round's eval is
    provably the final graded state -- no agent turn ran after it),
    ``_postprocess_runner_output`` must surface the watcher's cached
    eval_result via ``runtime_meta.polyglot_test_feedback_reuse_eligible`` /
    ``polyglot_test_feedback_eval_result`` so execution_phase.py can skip a
    second, redundant Docker evaluation."""
    fake_watcher = MagicMock()
    fake_watcher._cached_eval_holder = {
        "value": {"native_score": 1.0, "resolved": True, "status": "ok", "test_exit_code": 0},
        "error": None,
    }
    ex = _make_polyglot_executor(tmp_path, evaluator=object())
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="solve",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 50,
        },
    )

    def fake_run(cmd, **kw):
        return MagicMock(
            returncode=0,
            stdout=_runner_stdout(
                task_id=task.id,
                agent_id="agent-tf",
                meta={
                    "polyglot_test_feedback_meta": {
                        "enabled": True,
                        "rounds_used": 0,
                        "attempt_1_eval_summary": {"resolved": True},
                        "captured": True,
                        "final_eval_matches_output": True,
                    }
                },
            ),
            stderr="",
        )

    with (
        patch("ksi.runtime.container_host._run_command_with_backstop", side_effect=fake_run),
        patch.object(
            KsiContainerExecutor,
            "_launch_polyglot_test_feedback_watcher",
            return_value=fake_watcher,
        ),
    ):
        result = ex.run_task(
            generation=1,
            agent_id="agent-tf",
            task=task,
            agent_seed_package={},
            experiment_name="test_exp",
        )

    assert result.runtime_meta.get("polyglot_test_feedback_reuse_eligible") is True
    assert result.runtime_meta.get("polyglot_test_feedback_eval_result") == fake_watcher._cached_eval_holder["value"]


def test_run_task_does_not_reuse_cached_polyglot_eval_when_final_eval_does_not_match(tmp_path):
    """The unsafe case: the retry loop exhausted its tries after an edit
    turn, so the last cached eval scored the PRE-turn state -- it must NOT
    be surfaced as reuse-eligible."""
    fake_watcher = MagicMock()
    fake_watcher._cached_eval_holder = {
        "value": {"native_score": 0.0, "resolved": False, "status": "ok", "test_exit_code": 1},
        "error": None,
    }
    ex = _make_polyglot_executor(tmp_path, evaluator=object())
    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="solve",
        metadata={
            "task_source": "polyglot",
            "polyglot_test_feedback_tries": 2,
            "polyglot_test_feedback_max_lines": 50,
        },
    )

    def fake_run(cmd, **kw):
        return MagicMock(
            returncode=0,
            stdout=_runner_stdout(
                task_id=task.id,
                agent_id="agent-tf",
                meta={
                    "polyglot_test_feedback_meta": {
                        "enabled": True,
                        "rounds_used": 1,
                        "attempt_1_eval_summary": {"resolved": False},
                        "captured": True,
                        "final_eval_matches_output": False,
                    }
                },
            ),
            stderr="",
        )

    with (
        patch("ksi.runtime.container_host._run_command_with_backstop", side_effect=fake_run),
        patch.object(
            KsiContainerExecutor,
            "_launch_polyglot_test_feedback_watcher",
            return_value=fake_watcher,
        ),
    ):
        result = ex.run_task(
            generation=1,
            agent_id="agent-tf",
            task=task,
            agent_seed_package={},
            experiment_name="test_exp",
        )

    assert "polyglot_test_feedback_reuse_eligible" not in result.runtime_meta
    assert "polyglot_test_feedback_eval_result" not in result.runtime_meta


def test_run_task_does_not_inject_polyglot_test_feedback_for_non_polyglot(tmp_path):
    ex = _make_polyglot_executor(tmp_path, evaluator=object())
    task = TaskSpec(id="swe-1", repo="", prompt="solve", metadata={"task_source": "swebench_pro"})

    captured_payload = []

    def fake_run(cmd, **kw):
        with open(cmd[-1]) as f:
            captured_payload.append(json.load(f))
        return MagicMock(returncode=0, stdout=_runner_stdout(task_id=task.id), stderr="")

    with (
        patch(
            "ksi.runtime.container_host._run_command_with_backstop",
            side_effect=fake_run,
        ),
        patch.object(
            KsiContainerExecutor,
            "_launch_polyglot_test_feedback_watcher",
        ) as mock_launch,
    ):
        ex.run_task(
            generation=1,
            agent_id="agent-0",
            task=task,
            agent_seed_package={},
            experiment_name="test_exp",
        )

    assert len(captured_payload) == 1
    assert "polyglot_test_feedback" not in captured_payload[0]
    mock_launch.assert_not_called()
