"""Regression tests for status='error' envelopes emitted with exit-code 0.

Context
-------
The TypeScript ``agent-runner`` emits ``{status: 'error', result: null, ...}``
envelopes on several code paths (``runtime_runner/agent-runner/src/index.ts``):

  * ``emitTerminalDiagnostic`` (unhandledRejection / uncaughtException
    handlers around line 1541-1603).
  * The iterator-threw branch after partial output (around line 1572-1603).
  * The OpenAI adapter's throw path (``openai.ts`` caught at
    ``index.ts:1519-1528``).

Most of these call ``process.exit(1)`` *after* writing the envelope, so the
existing ``container_host`` non-zero-exit raise handles them. But some paths
(e.g. ``emitTerminalDiagnostic`` under a SIGTERM reaper, or any future adapter
that exits cleanly after writing an error envelope) land on stdout with
exit-code 0. When that happens, ``container_host.run_task`` currently returns
a ``RuntimeResult`` with ``runtime_meta.status='error'`` and empty output. The
engine's ``_eval_stage`` then runs the evaluator on empty output -- ARC reports
``parse_error:empty_model_output`` and the attempt row lands with
``trace.error=None``, looking like a successful 0-score run rather than a
runtime error.

This test suite pins the fix: whenever the parsed ``runtime_meta.status`` is
``'error'``, ``run_task`` must raise a ``SilentAgentRuntimeError`` (which is a
``RuntimeError`` subclass) carrying the meta across the raise/except boundary
so the engine's existing exception handlers preserve the forensics.
"""

from __future__ import annotations

import json
import subprocess
import types

import pytest

from kcsi.models import TaskSpec
from kcsi.runtime.container_host import KcsiContainerExecutor, _error_envelope_event_name
from kcsi.runtime.normalize import SilentAgentRuntimeError


def _error_envelope_stdout(
    *,
    error_message: str = "OpenAI run produced no observable output: adapter threw",
    native_session_memory: str | None = None,
) -> str:
    """Stdout shape for a status='error' envelope with exit-code 0.

    Mirrors the envelope written by the ``emitTerminalDiagnostic`` /
    iterator-threw paths in ``index.ts``. Token counters are zero because the
    error envelope fires on the drain path where nothing was accumulated.
    """
    meta: dict[str, object] = {
        "generation": 1,
        "agent_id": "agent-0",
        "task_id": "error-envelope-task",
        "status": "error",
        "error": error_message,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    if native_session_memory is not None:
        meta["native_session_memory"] = native_session_memory
        meta["raw_native_session_memory"] = native_session_memory
    return json.dumps({"result": None, "tool_trace": [], "meta": meta})


def _build_executor(tmp_path, *, stdout: str):
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
    return executor


class TestContainerHostRaisesOnStatusErrorEnvelope:
    """End-to-end: when the container exits code 0 with an envelope whose
    ``meta.status == 'error'``, ``run_task`` must raise ``RuntimeError``."""

    def test_run_task_raises_on_error_envelope(self, tmp_path):
        stdout = _error_envelope_stdout(
            error_message="OpenAI run produced no observable output: adapter threw",
        )
        executor = _build_executor(tmp_path, stdout=stdout)

        task = TaskSpec(
            id="error-envelope-task",
            repo=None,
            prompt="solve this",
            metadata={"task_source": "arc"},
        )

        with pytest.raises(RuntimeError) as excinfo:
            executor.run_task(
                generation=1,
                agent_id="agent-0",
                task=task,
            )

        msg = str(excinfo.value)
        assert "error-envelope-task" in msg
        # Must surface the adapter-side error message so the DB's
        # error_text column is diagnosable, not just a generic label.
        assert "OpenAI run produced no observable output" in msg

    def test_raised_error_preserves_runtime_meta(self, tmp_path):
        """The raised exception must carry ``runtime_meta`` so the engine's
        ``_cap_native_memory_fields`` step can preserve forensics evidence
        (native_session_memory). Uses ``SilentAgentRuntimeError`` — the same
        carrier already used for status='silent_failure'.
        """
        native_memory = "# file: projects/error-task/abc-def.jsonl\n{'type':'assistant', ...} × 20 turns × 8192 tokens"
        stdout = _error_envelope_stdout(
            error_message="OpenAI adapter threw: fetch aborted",
            native_session_memory=native_memory,
        )
        executor = _build_executor(tmp_path, stdout=stdout)

        task = TaskSpec(
            id="error-envelope-task",
            repo=None,
            prompt="solve this",
            metadata={"task_source": "arc"},
        )

        with pytest.raises(SilentAgentRuntimeError) as excinfo:
            executor.run_task(
                generation=1,
                agent_id="agent-0",
                task=task,
            )

        # The carrier must preserve meta so the engine can restore
        # native_session_memory on the trace row.
        assert excinfo.value.runtime_meta.get("status") == "error"
        assert excinfo.value.runtime_meta.get("native_session_memory") == native_memory
        assert excinfo.value.runtime_meta.get("raw_native_session_memory") == native_memory

    def test_healthy_status_success_does_not_raise(self, tmp_path):
        """Sanity: status='success' envelopes still return normally."""
        stdout = json.dumps(
            {
                "result": "ok",
                "tool_trace": [{"type": "tool_call", "tool_name": "Read"}],
                "meta": {
                    "generation": 1,
                    "agent_id": "agent-0",
                    "status": "success",
                    "task_id": "healthy_task",
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        )
        executor = _build_executor(tmp_path, stdout=stdout)

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
        assert result.output == "ok"


def test_error_envelope_event_name_uses_forum_specific_label() -> None:
    assert _error_envelope_event_name("cross_task_forum") == "runtime.forum_error_envelope"
    assert _error_envelope_event_name("per_task_forum") == "runtime.forum_error_envelope"
    assert _error_envelope_event_name("terminal_bench_2") == "runtime.error_envelope"
