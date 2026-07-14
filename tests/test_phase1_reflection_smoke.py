"""End-to-end-ish smoke test for Phase-1 reflection (Path a).

This test substitutes the TS runtime command with a tiny Python script that
plays the container's role: writes ``WORKSPACE_PATH=`` to stderr, drops the
sentinel, reads back the response, and emits a stdout envelope shaped like
the real container output. The host-side path under test is the full
``KsiContainerExecutor.run_task`` integration:

  * payload contains ``phase1_reflection: { enabled: true }``
  * a BarrierWatcher is launched
  * the watcher's callback runs the evaluator
  * the response is observed by the (fake) container
  * the resulting RuntimeResult.runtime_meta carries ``phase1_reflection``

A real docker-backed smoke is out of scope here because every dev box
without ``ksi-agent:bench`` would skip it; this test instead exercises
every Python and host-TS piece of the wiring (BarrierWatcher launch,
sentinel discovery, evaluator callback, response delivery, envelope
parsing). The TS runtime is exercised separately by the agent-runner
typecheck and tests/js/barrier_protocol.test.mjs.
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

import pytest

from ksi.models import TaskSpec
from ksi.runtime.container_host import KsiContainerExecutor

# A small Python script that emulates ``runtime_runner/src/main.ts``:
# writes the workspace path file, then waits for the agent's barrier
# sentinel (we synthesize this inline since we don't run agent-runner),
# reads the response, and emits an envelope to stdout. The script
# accepts a single payload.json argument exactly like main.ts.
_FAKE_RUNNER_TEMPLATE = r"""
import json, os, sys, time
from pathlib import Path

payload_path = sys.argv[1]
payload = json.loads(Path(payload_path).read_text(encoding='utf-8'))

# Pretend the workspace lives under a tempdir we are told about.
ws = Path(os.environ['FAKE_WORKSPACE_DIR'])
ws.mkdir(parents=True, exist_ok=True)

barrier_file = os.environ.get('KSI_BARRIER_WORKSPACE_FILE')
if barrier_file:
    Path(barrier_file).write_text(str(ws), encoding='utf-8')

agent_id = payload.get('agent_id')
phase1 = payload.get('phase1_reflection') or {}

base_meta = {
    'generation': payload['generation'],
    'agent_id': agent_id,
    'task_id': payload['task']['id'],
    'status': 'success',
    'session_scope': 'task',
    'input_tokens': 0,
    'output_tokens': 0,
    'cache_creation_input_tokens': 0,
    'cache_read_input_tokens': 0,
    'tokens_source': 'unavailable',
    'workspace_key': '',
    'session_id': '',
    'active_task_dir': '',
    'knowledge_db_path': '',
    'container_image': '',
    'official_container_image': '',
    'runner_image': '',
    'repo_container_path': '',
    'official_repo_container_path': '',
    'runner_root': '',
    'model_requested': '',
    'raw_native_session_memory': '',
    'native_session_memory': '',
    'conversation_archives': '',
    'tool_call_counts': {},
    'memory_tool_call_counts': {},
    'arc_tool_call_counts': {},
    'forum_tool_call_counts': {},
    'arc_submit_trial_results': [],
    'arc_last_submit_result': None,
    'recovery_note': None,
    'error': None,
}

if phase1.get('enabled'):
    sentinel = ws / ('.barrier.phase1_reflection.' + str(agent_id) + '.ready')
    sentinel.write_text(json.dumps({
        'schema': 'phase1_reflection.v1',
        'agent_id': agent_id,
        'task_id': payload['task']['id'],
        'model_output': 'simulated model output',
    }), encoding='utf-8')
    response = ws / ('.barrier.phase1_reflection.' + str(agent_id) + '.response')
    deadline = time.time() + 30.0
    response_payload = None
    while time.time() < deadline:
        if response.exists():
            try:
                response_payload = json.loads(response.read_text(encoding='utf-8'))
            except Exception:
                response_payload = None
            break
        time.sleep(0.05)
    reflection_text = ''
    if response_payload is not None:
        reflection_text = (
            'Assumption: outputs were already valid. '
            'Change: tighten the schema check before submission. '
            'Predicted: 5 percent fewer rejections.'
        )
    base_meta.update({
        'native_session_memory': 'simulated transcript',
        'phase1_reflection': reflection_text,
        'phase1_reflection_meta': {
            'enabled': True,
            'captured': bool(reflection_text),
            'note': None,
            'elapsed_ms': 100,
        },
        # Simulated reflection-turn usage so the host's engine records
        # a `phase1_reflection` row in token_phases.
        'phase1_reflection_token_usage': {
            'input_tokens': 250,
            'output_tokens': 80,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 200,
        },
    })

print(json.dumps({
    'result': 'simulated model output',
    'tool_trace': [],
    'meta': base_meta,
}))
"""


def _provider_env() -> dict:
    return {
        "MODEL_PROVIDER": "anthropic",
        "MODEL_AUTH_MODE": "api",
        "MODEL": "claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY": "fake-key-for-test",
    }


def test_phase1_reflection_end_to_end_with_fake_runner(tmp_path):
    """Drive ``KsiContainerExecutor.run_task`` with a fake runner that
    plays the container side of the barrier protocol; assert the resulting
    runtime_meta carries the phase1_reflection text the host fed back."""

    fake_runner = tmp_path / "fake_main.py"
    fake_runner.write_text(_FAKE_RUNNER_TEMPLATE, encoding="utf-8")

    workspace_dir = tmp_path / "ws"

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"native_score": 0.5, "resolved": False, "status": "scored"}

    executor = KsiContainerExecutor(
        command=[sys.executable, str(fake_runner)],
        working_dir=str(tmp_path),
        timeout_sec=60,
        env=_provider_env(),
        knowledge_db_path="",  # avoids snapshot path / KnowledgeStore deps
        phase1_reflection_enabled=True,
        evaluator=evaluator,
    )

    # Pass the fake workspace dir through env. ``_build_runner_env`` only
    # adds debug flags + container timeout; FAKE_WORKSPACE_DIR survives
    # through ``self.env`` below because it pre-seeds via the executor.
    executor.env = {**executor.env, "FAKE_WORKSPACE_DIR": str(workspace_dir)}

    task = TaskSpec(
        id="t-smoke",
        repo="dummy",
        prompt="solve nothing",
        metadata={"task_source": "polyglot"},
    )

    result = executor.run_task(generation=1, agent_id="a-smoke", task=task)
    assert result.runtime_meta.get("status") == "success"
    reflection = result.runtime_meta.get("phase1_reflection") or ""
    assert "Assumption" in reflection and "Predicted" in reflection
    assert evaluator.evaluate.call_count == 1
    # Verify the watcher used the model_output we shipped.
    call_kwargs = evaluator.evaluate.call_args.kwargs
    assert call_kwargs["model_output"] == "simulated model output"
    assert call_kwargs["task"].id == "t-smoke"
    # Important-3 wiring: the reflection-turn token usage must propagate
    # through runtime_meta so the engine can record it as its own
    # ``phase1_reflection`` token-phase row. (The engine-side persistence
    # is exercised by the dedicated test below.)
    p1_usage = result.runtime_meta.get("phase1_reflection_token_usage")
    assert isinstance(p1_usage, dict), (
        f"phase1_reflection_token_usage missing from runtime_meta; got keys {sorted(result.runtime_meta.keys())}"
    )
    assert p1_usage.get("input_tokens") == 250
    assert p1_usage.get("output_tokens") == 80
    assert p1_usage.get("cache_read_input_tokens") == 200
    # Critical-2 wiring: the watcher's eval_result must also be cached on
    # runtime_meta so the engine can skip its own evaluate() call.
    assert result.runtime_meta.get("phase1_reflection_enabled") is True
    cached_eval = result.runtime_meta.get("phase1_eval_result")
    assert isinstance(cached_eval, dict), (
        f"phase1_eval_result missing; got {result.runtime_meta.get('phase1_eval_result')!r}"
    )
    assert cached_eval.get("native_score") == 0.5


def test_phase1_reflection_smoke_disabled_path(tmp_path):
    """When the feature flag is OFF the fake runner emits no reflection,
    no BarrierWatcher is launched, and the evaluator is not called from
    the host's barrier path (any later eval is the engine's own call)."""
    fake_runner = tmp_path / "fake_main.py"
    fake_runner.write_text(_FAKE_RUNNER_TEMPLATE, encoding="utf-8")

    workspace_dir = tmp_path / "ws"

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"native_score": 0.0, "resolved": False, "status": "scored"}

    executor = KsiContainerExecutor(
        command=[sys.executable, str(fake_runner)],
        working_dir=str(tmp_path),
        timeout_sec=60,
        env=_provider_env(),
        knowledge_db_path="",
        phase1_reflection_enabled=False,
        evaluator=evaluator,
    )
    executor.env = {**executor.env, "FAKE_WORKSPACE_DIR": str(workspace_dir)}

    task = TaskSpec(
        id="t-off",
        repo="dummy",
        prompt="solve nothing",
        metadata={"task_source": "polyglot"},
    )
    result = executor.run_task(generation=1, agent_id="a-off", task=task)
    assert result.runtime_meta.get("status") == "success"
    assert (result.runtime_meta.get("phase1_reflection") or "") == ""
    # Host's barrier path never invoked the evaluator (the engine's own
    # post-run eval is a separate code path; the executor doesn't call it).
    assert evaluator.evaluate.call_count == 0


def test_deferred_watcher_exits_promptly_on_stop_before_workspace_written(tmp_path):
    """Critical-1 regression guard: the deferred watcher's
    _resolve_workspace_dir poll must observe the stop_event between
    iterations so a short subprocess that calls watcher.stop() before
    main.ts ever wrote the workspace path file doesn't leave a zombie
    daemon thread polling a deleted path for the full 60s deadline.
    """

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {"native_score": 0.0}

    executor = KsiContainerExecutor(
        command=["fake"],
        working_dir=str(tmp_path),
        timeout_sec=60,
        env=_provider_env(),
        knowledge_db_path="",
        phase1_reflection_enabled=True,
        evaluator=evaluator,
    )

    workspace_file = tmp_path / "never_written.txt"
    task = TaskSpec(id="t-stop", repo="", prompt="", metadata={"task_source": "polyglot"})

    started = time.monotonic()
    watcher = executor._launch_phase1_watcher(
        workspace_file=workspace_file,
        agent_id="a-stop",
        task=task,
        poll_timeout_sec=5.0,
    )
    assert watcher is not None
    # Brief delay so the deferred thread is genuinely inside the
    # _resolve_workspace_dir poll loop when we call stop().
    time.sleep(0.1)
    watcher.stop()
    watcher.join(timeout=2.0)
    elapsed = time.monotonic() - started
    assert not watcher.is_alive(), "deferred watcher should exit on stop_event"
    assert elapsed < 1.5, (
        f"deferred watcher took {elapsed:.2f}s to exit after stop() — "
        "must be well under the 60s _resolve_workspace_dir deadline"
    )
    # Evaluator was never called because the resolve loop bailed out.
    assert evaluator.evaluate.call_count == 0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
