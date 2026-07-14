"""Regression tests for silent agent-runner failures.

Reproduces the bug observed on ARC-AGI-1 task ``55783887`` during the live
Haiku baseline sweep (2026-04-20 snapshot at
``results/baseline_sweep_haiku/artifacts_partial_20260420_102117/memory/
baseline_haiku_arc1_memory.sqlite``):

Across 4 generations (3, 5, 6, 7), every attempt on this single task returned
``runtime_meta.status='success'`` with ``input_tokens=output_tokens=0``,
empty ``tool_trace``, and NULL ``model_output``/``error_text``. The
``arc_session`` evaluator then reported ``parse_error:empty_model_output``
with no diagnostic trail, and the engine counted it as a zero-score success
attempt (not a failure), hiding the bug in aggregate metrics.

Root cause: the claude-agent-sdk ``query(...)`` async iterator inside the
container drained without yielding any messages. The agent-runner's
scheduled-task fallback (``index.ts`` line ~802) only emits output when
``lastAssistantFallback`` is truthy; when it is null, no ``writeOutput`` is
ever called. The container exits code 0 with no ``OUTPUT_START_MARKER`` in
stdout, and ``container_runner.ts`` previously resolved this as
``{status: 'success', result: null}`` (line ~888). The empty envelope then
propagated through ``parse_runner_stdout`` → engine → attempts table with
``status='success'``.

Fix surface (tested here):
  * ``kcsi.runtime.normalize.is_silent_agent_failure`` -- detects the
    "success + 0 tokens + empty trace + empty output" fingerprint.
  * ``kcsi.runtime.normalize.parse_runner_stdout`` -- reclassifies the
    runtime_meta.status to ``silent_failure`` and adds an ``error`` message.
  * ``kcsi.runtime.container_host.KcsiContainerExecutor.run_task`` --
    raises ``RuntimeError`` when the reclassified status is observed, so
    the engine's ``_eval_stage`` catches it and records ``error_text`` on
    the attempt row (engine path exercised indirectly here).
"""

from __future__ import annotations

import json

import pytest

from kcsi.runtime.normalize import (
    SILENT_FAILURE_MESSAGE,
    SILENT_FAILURE_STATUS,
    is_silent_agent_failure,
    mark_silent_failure,
    parse_runner_stdout,
)
from kcsi.tokens import TokenUsage


def _silent_stdout_like_real_sweep() -> str:
    """Return stdout matching the shape the container emits on silent failure.

    Mirrors the meta fields observed on attempt_id=201 (task_id=55783887,
    gen=3) in the snapshot DB: all token counters at 0, empty tool_trace,
    null result, but status='success' because the outer envelope was
    synthesized by ``runtime_runner/src/main.ts`` from an empty
    ContainerOutput.
    """
    return json.dumps(
        {
            "result": "",
            "tool_trace": [],
            "meta": {
                "generation": 3,
                "agent_id": "agent-7",
                "task_id": "55783887",
                "status": "success",
                "session_scope": "task",
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "workspace_key": ("task__baseline_haiku_arc1__55783887__af5783676b"),
                "session_id": "863fd737-e10a-4bce-8a2f-13fccbcf4cbe",
                "tool_call_counts": {},
                "memory_tool_call_counts": {},
                "arc_tool_call_counts": {},
                "forum_tool_call_counts": {},
                "arc_submit_trial_results": [],
            },
        }
    )


class TestIsSilentAgentFailure:
    def test_detects_exact_sweep_fingerprint(self):
        """Must detect the exact shape we saw on ARC1 task 55783887."""
        result = {
            "output": "",
            "tool_trace": [],
            "runtime_meta": {
                "status": "success",
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            "token_usage": TokenUsage(),
        }
        assert is_silent_agent_failure(result) is True

    def test_rejects_nonzero_input_tokens(self):
        result = {
            "output": "",
            "tool_trace": [],
            "runtime_meta": {"status": "success"},
            "token_usage": TokenUsage(input_tokens=10),
        }
        assert is_silent_agent_failure(result) is False

    def test_rejects_nonzero_cache_read_tokens(self):
        # Prompt-cached rerun that returned empty output is still suspicious,
        # but if cache_read > 0 the SDK did exchange something with the API
        # and we should not hijack the flow.
        result = {
            "output": "",
            "tool_trace": [],
            "runtime_meta": {"status": "success"},
            "token_usage": TokenUsage(cache_read_input_tokens=500),
        }
        assert is_silent_agent_failure(result) is False

    def test_rejects_nonempty_output(self):
        result = {
            "output": "a partial answer",
            "tool_trace": [],
            "runtime_meta": {"status": "success"},
            "token_usage": TokenUsage(),
        }
        assert is_silent_agent_failure(result) is False

    def test_rejects_when_tool_trace_has_entries(self):
        result = {
            "output": "",
            "tool_trace": [{"type": "tool_call", "tool_name": "Read"}],
            "runtime_meta": {"status": "success"},
            "token_usage": TokenUsage(),
        }
        assert is_silent_agent_failure(result) is False

    def test_rejects_when_status_is_already_error(self):
        # If the runner already flagged an error, don't touch it.
        result = {
            "output": "",
            "tool_trace": [],
            "runtime_meta": {"status": "error", "error": "something"},
            "token_usage": TokenUsage(),
        }
        assert is_silent_agent_failure(result) is False

    def test_accepts_missing_status_key(self):
        # Some legacy paths don't populate runtime_meta.status at all.
        result = {
            "output": "",
            "tool_trace": [],
            "runtime_meta": {},
            "token_usage": TokenUsage(),
        }
        assert is_silent_agent_failure(result) is True

    def test_whitespace_only_output_is_silent(self):
        result = {
            "output": "   \n\t  ",
            "tool_trace": [],
            "runtime_meta": {"status": "success"},
            "token_usage": TokenUsage(),
        }
        assert is_silent_agent_failure(result) is True


class TestMarkSilentFailure:
    def test_rewrites_status_and_preserves_other_meta(self):
        original = {
            "output": "",
            "tool_trace": [],
            "runtime_meta": {
                "status": "success",
                "task_id": "55783887",
                "session_id": "abc-123",
                "native_session_memory": "<partial journal>",
            },
            "token_usage": TokenUsage(),
        }
        out = mark_silent_failure(original)
        assert out["runtime_meta"]["status"] == SILENT_FAILURE_STATUS
        assert out["runtime_meta"]["error"] == SILENT_FAILURE_MESSAGE
        # Preserve diagnostic info for post-mortems
        assert out["runtime_meta"]["task_id"] == "55783887"
        assert out["runtime_meta"]["session_id"] == "abc-123"
        assert out["runtime_meta"]["native_session_memory"] == "<partial journal>"

    def test_does_not_mutate_input(self):
        original = {
            "output": "",
            "tool_trace": [],
            "runtime_meta": {"status": "success"},
            "token_usage": TokenUsage(),
        }
        _ = mark_silent_failure(original)
        assert original["runtime_meta"]["status"] == "success"


class TestParseRunnerStdoutSilentFailure:
    def test_reclassifies_sweep_shaped_payload(self):
        parsed = parse_runner_stdout(_silent_stdout_like_real_sweep(), key="result")
        assert parsed["runtime_meta"]["status"] == SILENT_FAILURE_STATUS
        assert "error" in parsed["runtime_meta"]
        # Task_id and other diagnostic fields must survive
        assert parsed["runtime_meta"]["task_id"] == "55783887"
        # Tokens remain 0 so aggregate counters don't lie
        assert parsed["token_usage"].total == 0

    def test_healthy_payload_is_unchanged(self):
        stdout = json.dumps(
            {
                "result": "the grid is [[1,2,3]]",
                "tool_trace": [{"type": "tool_call", "tool_name": "Read"}],
                "meta": {
                    "generation": 1,
                    "agent_id": "agent-0",
                    "task_id": "healthy_task",
                    "status": "success",
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        )
        parsed = parse_runner_stdout(stdout, key="result")
        assert parsed["runtime_meta"]["status"] == "success"
        assert parsed["runtime_meta"].get("error") is None
        assert parsed["token_usage"].total == 150

    def test_existing_status_error_is_preserved(self):
        # When the runner already flagged status='error', don't rewrite it.
        stdout = json.dumps(
            {
                "result": None,
                "tool_trace": [],
                "meta": {
                    "status": "error",
                    "error": "container oom",
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            }
        )
        parsed = parse_runner_stdout(stdout, key="result")
        assert parsed["runtime_meta"]["status"] == "error"
        assert parsed["runtime_meta"]["error"] == "container oom"

    def test_empty_stdout_does_not_raise(self):
        # Guard against the early-return path — should not invoke the detector
        # (no outer envelope to classify).
        parsed = parse_runner_stdout("", key="result")
        assert parsed["runtime_meta"] == {}
        assert parsed["token_usage"].total == 0


class TestContainerHostRaisesOnSilentFailure:
    """End-to-end: container_host.KcsiContainerExecutor.run_task must
    raise RuntimeError when the outer envelope matches the silent-failure
    fingerprint, so the engine can catch it and record error_text."""

    def _build_executor(self, tmp_path, *, stdout: str):
        import subprocess
        import types

        from kcsi.runtime import container_host as host_mod
        from kcsi.runtime.container_host import KcsiContainerExecutor

        fake_env = {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_AUTH_MODE": "api",
            "MODEL": "claude-haiku-4-5-20251001",
            "ANTHROPIC_API_KEY": "sk-ant-test-placeholder",
        }
        executor = KcsiContainerExecutor(
            command=["/bin/true"],
            working_dir=str(tmp_path),
            timeout_sec=60,
            env=fake_env,
            knowledge_db_path="",
            disable_memory_mcp=True,
        )

        def fake_runner(self, cmd, *, cwd, env, timeout):
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=stdout,
                stderr="",
            )

        executor._run_runner_command = types.MethodType(fake_runner, executor)
        return executor, host_mod

    def test_run_task_raises_on_silent_failure(self, tmp_path, monkeypatch):
        from kcsi.models import TaskSpec

        stdout = _silent_stdout_like_real_sweep()
        executor, _host_mod = self._build_executor(tmp_path, stdout=stdout)

        # run_task builds workspace artifacts from the task spec; point the
        # repo_source at a non-existent dir so seeding stays a no-op.
        task = TaskSpec(
            id="55783887",
            repo=None,
            prompt="solve this",
            metadata={"task_source": "arc"},
        )

        with pytest.raises(RuntimeError) as excinfo:
            executor.run_task(
                generation=3,
                agent_id="agent-7",
                task=task,
            )
        msg = str(excinfo.value).lower()
        assert "silent" in msg or "no output" in msg
        assert "55783887" in str(excinfo.value)

    def test_run_task_succeeds_on_healthy_output(self, tmp_path):
        from kcsi.models import TaskSpec

        stdout = json.dumps(
            {
                "result": "patch ok",
                "tool_trace": [{"type": "tool_call", "tool_name": "Read"}],
                "meta": {
                    "status": "success",
                    "generation": 1,
                    "agent_id": "agent-0",
                    "task_id": "healthy_task",
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        )
        executor, _host_mod = self._build_executor(tmp_path, stdout=stdout)

        task = TaskSpec(
            id="healthy_task",
            repo=None,
            prompt="solve this",
            metadata={"task_source": "arc"},
        )
        result = executor.run_task(
            generation=1,
            agent_id="agent-0",
            task=task,
        )
        assert result.output == "patch ok"
        assert result.token_usage.input_tokens == 100
