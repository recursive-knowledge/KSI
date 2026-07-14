"""Unit tests for the host-side BarrierWatcher protocol."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from conftest import REPO_ROOT

from kcsi.runtime.barrier import (
    MAX_SENTINEL_BYTES,
    BarrierEvent,
    BarrierWatcher,
    response_filename,
    sentinel_filename,
)


def test_sentinel_and_response_filename_are_namespaced():
    assert sentinel_filename("phase1_reflection", "agent-7") == ".barrier.phase1_reflection.agent-7.ready"
    assert response_filename("phase1_reflection", "agent-7") == ".barrier.phase1_reflection.agent-7.response"


def test_filename_sanitizes_unsafe_characters():
    # Slashes / dots in inputs must be neutered so the basename stays
    # within the workspace dir.
    assert "/" not in sentinel_filename("evil/name", "../escape")
    assert ".." not in sentinel_filename("name", "../escape").split(".barrier.")[1]


def test_watcher_invokes_callback_and_writes_response(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: list[BarrierEvent] = []

    def cb(event: BarrierEvent) -> dict:
        captured.append(event)
        return {"score": 0.9, "echo": event.payload.get("hint")}

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="phase1_reflection",
        agent_id="agent-1",
        callback=cb,
        poll_interval_sec=0.05,
        timeout_sec=5.0,
    )
    watcher.start()

    # Simulate the container writing the sentinel.
    sentinel = workspace / sentinel_filename("phase1_reflection", "agent-1")
    sentinel.write_text(json.dumps({"hint": "foo"}), encoding="utf-8")

    watcher.join(timeout=5.0)
    assert not watcher.is_alive()
    assert watcher.fired() is True

    response_path = workspace / response_filename("phase1_reflection", "agent-1")
    assert response_path.exists()
    body = json.loads(response_path.read_text(encoding="utf-8"))
    assert body == {"score": 0.9, "echo": "foo"}
    # The watcher consumed the sentinel.
    assert not sentinel.exists()
    assert len(captured) == 1


def test_watcher_times_out_when_no_sentinel(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def cb(event: BarrierEvent) -> dict:  # pragma: no cover - never called
        raise AssertionError("callback should not run")

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="phase1_reflection",
        agent_id="agent-1",
        callback=cb,
        poll_interval_sec=0.05,
        timeout_sec=0.25,
    )
    started = time.monotonic()
    watcher.start()
    watcher.join(timeout=2.0)
    assert not watcher.is_alive()
    assert watcher.fired() is False
    elapsed = time.monotonic() - started
    assert elapsed >= 0.20
    assert not (workspace / response_filename("phase1_reflection", "agent-1")).exists()


def test_watcher_stop_event_unblocks(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="phase1_reflection",
        agent_id="agent-1",
        callback=lambda evt: {"ok": True},
        poll_interval_sec=0.05,
    )
    watcher.start()
    time.sleep(0.1)
    watcher.stop()
    watcher.join(timeout=2.0)
    assert not watcher.is_alive()
    assert watcher.fired() is False


def test_watcher_callback_exception_writes_error_response(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def cb(event: BarrierEvent) -> dict:
        raise RuntimeError("evaluator died")

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="phase1_reflection",
        agent_id="a",
        callback=cb,
        poll_interval_sec=0.05,
        timeout_sec=5.0,
    )
    watcher.start()
    (workspace / sentinel_filename("phase1_reflection", "a")).write_text("{}", encoding="utf-8")
    watcher.join(timeout=5.0)
    response_path = workspace / response_filename("phase1_reflection", "a")
    assert response_path.exists()
    body = json.loads(response_path.read_text(encoding="utf-8"))
    assert "error" in body
    assert "evaluator died" in body["error"]
    assert isinstance(watcher.error(), RuntimeError)


def test_watcher_handles_invalid_sentinel_json_as_empty_payload(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    seen: list[dict] = []

    def cb(event: BarrierEvent) -> dict:
        seen.append(event.payload)
        return {"ok": True}

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="b",
        agent_id="x",
        callback=cb,
        poll_interval_sec=0.05,
        timeout_sec=5.0,
    )
    watcher.start()
    (workspace / sentinel_filename("b", "x")).write_text("{not valid json", encoding="utf-8")
    watcher.join(timeout=5.0)
    assert seen == [{}]
    assert (workspace / response_filename("b", "x")).exists()


def test_watcher_starts_before_workspace_dir_exists(tmp_path: Path):
    """The host can launch the watcher before the container creates the workspace."""

    workspace = tmp_path / "later"

    fired = threading.Event()

    def cb(event: BarrierEvent) -> dict:
        fired.set()
        return {"ok": True}

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="b",
        agent_id="x",
        callback=cb,
        poll_interval_sec=0.05,
        timeout_sec=5.0,
    )
    watcher.start()
    time.sleep(0.15)  # let it poll a few times against a missing dir
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / sentinel_filename("b", "x")).write_text("{}", encoding="utf-8")
    watcher.join(timeout=5.0)
    assert fired.is_set()


def test_watcher_response_file_overwrites_stale_response(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    stale = workspace / response_filename("b", "x")
    stale.write_text(json.dumps({"old": True}), encoding="utf-8")

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="b",
        agent_id="x",
        callback=lambda evt: {"new": True},
        poll_interval_sec=0.05,
        timeout_sec=5.0,
    )
    watcher.start()
    (workspace / sentinel_filename("b", "x")).write_text("{}", encoding="utf-8")
    watcher.join(timeout=5.0)
    body = json.loads(stale.read_text(encoding="utf-8"))
    assert body == {"new": True}


def test_persistent_watcher_answers_multiple_sequential_rounds(tmp_path: Path):
    """``polyglot_test_feedback`` with ``--polyglot-test-feedback-tries`` > 2
    needs MORE than one sentinel/response round-trip over the watcher's
    lifetime (one per retry round), and the caller only calls ``stop()``
    once after the whole container subprocess exits. A single-shot watcher
    would answer round 0 then exit, leaving round 1+ unanswered until the
    caller's own poll timeout expires. ``persistent=True`` must keep
    serving fresh sentinels, each written only after the previous round's
    response was consumed — exactly matching ``runPolyglotTestFeedback``'s
    sequential (not concurrent) round structure."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    seen_hints: list[str] = []

    def cb(event: BarrierEvent) -> dict:
        seen_hints.append(str(event.payload.get("hint")))
        return {"echo": event.payload.get("hint")}

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="polyglot_test_feedback",
        agent_id="agent-1",
        callback=cb,
        poll_interval_sec=0.05,
        timeout_sec=5.0,
        persistent=True,
    )
    watcher.start()

    sentinel = workspace / sentinel_filename("polyglot_test_feedback", "agent-1")
    response_path = workspace / response_filename("polyglot_test_feedback", "agent-1")

    for round_num in range(3):
        sentinel.write_text(json.dumps({"hint": f"round-{round_num}"}), encoding="utf-8")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not response_path.exists():
            time.sleep(0.02)
        assert response_path.exists(), f"no response for round {round_num}"
        body = json.loads(response_path.read_text(encoding="utf-8"))
        assert body == {"echo": f"round-{round_num}"}
        # Consume the response the way the TS side does, so the next
        # sentinel write is unambiguous.
        response_path.unlink()
        # The watcher must have re-armed: the sentinel it just consumed is gone.
        assert not sentinel.exists()

    assert seen_hints == ["round-0", "round-1", "round-2"]
    assert watcher.is_alive()

    watcher.stop()
    watcher.join(timeout=3.0)
    assert not watcher.is_alive()


def test_non_persistent_watcher_ignores_a_second_sentinel(tmp_path: Path):
    """Baseline contrast for the persistent-watcher test above: the default
    (``persistent=False``) single-shot watchers used by phase1_reflection /
    cross_task_r1 must NOT answer a second sentinel — they exit after the
    first round, exactly as before this change."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    call_count = 0

    def cb(event: BarrierEvent) -> dict:
        nonlocal call_count
        call_count += 1
        return {"ok": True}

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="phase1_reflection",
        agent_id="agent-1",
        callback=cb,
        poll_interval_sec=0.05,
        timeout_sec=5.0,
    )
    watcher.start()

    sentinel = workspace / sentinel_filename("phase1_reflection", "agent-1")
    sentinel.write_text(json.dumps({"hint": "first"}), encoding="utf-8")
    watcher.join(timeout=5.0)
    assert not watcher.is_alive()
    assert call_count == 1

    # A second sentinel arrives after the watcher has already exited.
    sentinel.write_text(json.dumps({"hint": "second"}), encoding="utf-8")
    time.sleep(0.2)
    assert call_count == 1


def test_deferred_watcher_full_indirection_through_workspace_path_file(tmp_path: Path):
    """Important-4: exercise the full _DeferredWatcher production path.

    Previously the only ``BarrierWatcher`` test inserted the watcher
    directly against a known workspace dir — bypassing the production
    indirection where:
      1. the host writes ``KCSI_BARRIER_WORKSPACE_FILE=<path>`` to the
         runner's env,
      2. ``main.ts`` writes the resolved workspace dir into that file,
      3. ``_DeferredWatcher`` polls for the file, reads it, and only
         then constructs the inner BarrierWatcher,
      4. the agent drops a sentinel inside that workspace,
      5. the watcher's callback runs and the response file is written.

    Step (3) is exactly where the Critical-1 zombie-thread bug lived,
    and where future races between the workspace-path-file and stop()
    could regress. This test goes through the actual code path
    end-to-end — ``KcsiContainerExecutor._launch_phase1_watcher`` —
    so a regression on _resolve_workspace_dir's stop_event handling
    fails this test directly.
    """

    from unittest.mock import MagicMock

    from kcsi.models import TaskSpec
    from kcsi.runtime.container_host import KcsiContainerExecutor

    workspace_dir = tmp_path / "task_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    workspace_path_file = tmp_path / "workspace_path.txt"

    captured_payloads: list[dict] = []

    evaluator = MagicMock()

    def fake_evaluate(*, task, model_output, runtime_meta, tool_trace):
        captured_payloads.append(
            {
                "task_id": task.id,
                "model_output": model_output,
                "runtime_meta": runtime_meta,
            }
        )
        return {"native_score": 0.875, "resolved": True, "status": "scored"}

    evaluator.evaluate.side_effect = fake_evaluate

    executor = KcsiContainerExecutor(
        command=["fake"],
        working_dir=str(tmp_path),
        evaluator=evaluator,
        phase1_reflection_enabled=True,
    )

    task = TaskSpec(id="t-indirection", repo="r", prompt="p", metadata={})

    # Step 1: launch deferred watcher — workspace_path_file does not yet
    # exist, so the watcher is polling for it.
    watcher = executor._launch_phase1_watcher(
        workspace_file=workspace_path_file,
        agent_id="a-ind",
        task=task,
        poll_timeout_sec=10.0,
    )
    assert watcher is not None

    # Step 2: simulate ``main.ts`` thread writing the resolved workspace
    # dir into the path file.
    def main_ts_writer():
        time.sleep(0.1)  # let the deferred watcher poll once or twice
        workspace_path_file.write_text(str(workspace_dir), encoding="utf-8")

    threading.Thread(target=main_ts_writer, daemon=True).start()

    # Step 3: simulate the agent dropping a sentinel inside the
    # workspace dir AFTER the workspace path file exists.
    def agent_writer():
        time.sleep(0.4)  # _DeferredWatcher needs a moment to construct inner
        sentinel = workspace_dir / sentinel_filename("phase1_reflection", "a-ind")
        sentinel.write_text(
            json.dumps({"agent_id": "a-ind", "model_output": "agent solved task"}),
            encoding="utf-8",
        )

    threading.Thread(target=agent_writer, daemon=True).start()

    # Step 4: poll for the response file the watcher writes.
    response_path = workspace_dir / response_filename("phase1_reflection", "a-ind")
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and not response_path.exists():
        time.sleep(0.05)
    assert response_path.exists(), (
        f"BarrierWatcher did not write the response file within deadline; "
        f"workspace contents: {sorted(p.name for p in workspace_dir.iterdir())}"
    )

    body = json.loads(response_path.read_text(encoding="utf-8"))
    assert body.get("native_score") == 0.875
    assert body.get("resolved") is True

    # Callback fired exactly once, with the model_output we sent.
    assert evaluator.evaluate.call_count == 1
    assert captured_payloads[0]["task_id"] == "t-indirection"
    assert captured_payloads[0]["model_output"] == "agent solved task"
    # Characterization: phase1_reflection's barrier callback supplies the
    # agent's live on-disk workspace repo (``<sentinel parent>/workspace/
    # repo``, same as the polyglot test-feedback watcher), NOT an empty
    # runtime_meta — so workspace-scored evaluators grade the real submitted
    # files rather than falling back to model_output extraction.
    assert captured_payloads[0]["runtime_meta"] == {
        "host_workspace_repo_dir": str(workspace_dir / "workspace" / "repo")
    }

    # The cached eval result must be on the deferred watcher so the
    # executor would inject it into runtime_meta (Critical-2 wiring).
    cached = getattr(watcher, "_cached_eval_holder", None)
    assert isinstance(cached, dict)
    assert cached["value"] is not None
    assert cached["value"]["native_score"] == 0.875

    # Stop and join — should complete promptly.
    started = time.monotonic()
    watcher.stop()
    watcher.join(timeout=3.0)
    assert not watcher.is_alive()
    assert time.monotonic() - started < 2.5


def test_main_ts_writes_workspace_root_not_inner_active_dir(tmp_path: Path):
    """Regression: host/container barrier path agreement.

    Production failure (commit 9e1fa515, polyglot smoke): container side
    wrote ``.barrier.phase1_reflection.<agent>.ready`` into
    ``CONTAINER_WORKSPACE_ROOT='/workspace/task'`` (the bind-mount root),
    but the host's ``main.ts`` previously wrote the WORKSPACE_PATH file
    pointing at ``workspaceRootPath/workspace/task`` — the inner active
    working dir CONTAINER_ACTIVE_WORKSPACE_DIR. The two host paths differ
    by ``/workspace/task`` so the BarrierWatcher polled an empty
    directory and never observed the sentinel; container then timed out
    after ``effective_timeout`` (~840s for polyglot smoke) with
    ``phase1_reflection_meta.captured=False``.

    The host-side mount in container_runner.ts is::

        hostPath:      <workspaceRootPath>
        containerPath: /workspace/task     # = CONTAINER_WORKSPACE_ROOT

    Therefore inside the container ``/workspace/task/X`` is host-equivalent
    to ``<workspaceRootPath>/X``. The host's WORKSPACE_PATH file MUST emit
    ``<workspaceRootPath>`` (not ``<workspaceRootPath>/workspace/task``)
    so host BarrierWatcher and the container-side writer reference the
    same physical directory.

    This test parses ``runtime_runner/src/main.ts`` and asserts the
    one-liner that computes ``workspaceTaskDir`` does NOT join the
    ``'workspace', 'task'`` segments. A subsequent edit that re-introduces
    the bug fails this test.
    """

    main_ts = REPO_ROOT / "runtime_runner" / "src" / "main.ts"
    if not main_ts.exists():  # pragma: no cover - guards repo-layout drift
        pytest.skip(f"main.ts not found at {main_ts}")

    text = main_ts.read_text(encoding="utf-8")

    # The exact assignment we care about is the value passed to the
    # BarrierWatcher path file write. Find it.
    import re

    m = re.search(
        r"const\s+workspaceTaskDir\s*=\s*([^;]+);",
        text,
    )
    assert m is not None, "main.ts must define `const workspaceTaskDir = ...;`"
    rhs = m.group(1).strip()

    # The bug pattern: path.join(workspaceRootPath, 'workspace', 'task').
    # We accept the bare identifier ``workspaceRootPath`` (the fix) or
    # any other identifier whose name doesn't include ``workspace`` and
    # ``task`` joined together. The strict check below is the simplest
    # that catches the historical regression.
    assert "'workspace'" not in rhs and '"workspace"' not in rhs and "'task'" not in rhs and '"task"' not in rhs, (
        f"main.ts workspaceTaskDir RHS includes the inner active dir "
        f"path segments — this re-introduces the host/container barrier "
        f"path mismatch (PR #573 production failure on commit 9e1fa515). "
        f"workspaceTaskDir must equal the bind-mount source path "
        f"workspaceRootPath, not the inner active dir. Got: {rhs!r}"
    )


def test_read_sentinel_rejects_oversized_payload_without_reading_it(tmp_path: Path):
    """Security hardening (PR #1032 deep review, security.md Finding 2): an
    oversized sentinel must be rejected by a cheap stat() check BEFORE the
    watcher reads its full content into memory -- otherwise a Bash-capable
    agent (or a bug) writing an arbitrarily large sentinel could pressure
    host memory once per poll iteration."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: list[BarrierEvent] = []

    def cb(event: BarrierEvent) -> dict:
        captured.append(event)
        return {"ok": True}

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="polyglot_test_feedback",
        agent_id="agent-1",
        callback=cb,
        poll_interval_sec=0.05,
        timeout_sec=3.0,
    )
    watcher.start()

    sentinel = workspace / sentinel_filename("polyglot_test_feedback", "agent-1")
    # Write a file just over the cap without holding the whole oversized
    # string in memory ourselves.
    with open(sentinel, "w", encoding="utf-8") as fh:
        fh.write('{"model_output": "')
        fh.write("x" * (MAX_SENTINEL_BYTES + 1000))
        fh.write('"}')

    watcher.join(timeout=4.0)
    assert not watcher.is_alive()
    # Treated as an empty payload (same degrade path as invalid JSON) --
    # the callback still fires so the retry loop doesn't hang forever, but
    # with no model_output field.
    assert len(captured) == 1
    assert captured[0].payload == {}


def test_persistent_watcher_logs_when_sentinel_unlink_fails(tmp_path: Path, monkeypatch, caplog):
    """Observability hardening (PR #1032 deep review, errors-timeouts.md
    Finding 2): a failed sentinel unlink must be logged, not silently
    swallowed -- for a persistent watcher, a leftover sentinel gets
    reprocessed on the very next poll with no diagnostic trail otherwise."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def cb(event: BarrierEvent) -> dict:
        return {"ok": True}

    watcher = BarrierWatcher(
        workspace_dir=workspace,
        name="polyglot_test_feedback",
        agent_id="agent-1",
        callback=cb,
        poll_interval_sec=0.05,
        timeout_sec=2.0,
        persistent=True,
    )

    original_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self.name.endswith(".ready"):
            raise OSError("simulated unlink failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with caplog.at_level("WARNING"):
        watcher.start()
        sentinel = workspace / sentinel_filename("polyglot_test_feedback", "agent-1")
        sentinel.write_text(json.dumps({}), encoding="utf-8")
        watcher.join(timeout=4.0)

    assert any("failed to unlink consumed sentinel" in rec.message for rec in caplog.records)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
