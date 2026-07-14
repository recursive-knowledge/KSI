from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..benchmarks.polyglot_harness import DEFAULT_POLYGLOT_TIMEOUT_SEC
from ..errors import AuthenticationFailure
from ..models import TaskSpec
from ..prompts import build_execution_prompt, build_task_markdown
from ..tasks.registry import resolve_source

# Optional ARC no-MCP prompt builders. Some test harnesses stub
# ``..prompts`` with only ``build_execution_prompt`` / ``build_task_markdown``;
# fall back to no-op shims so module import still succeeds in those harnesses.
try:
    from ..prompts import (  # type: ignore[attr-defined]
        _build_arc_no_mcp_execution_prompt,
        _build_arc_no_mcp_task_markdown,
    )
except ImportError:  # pragma: no cover — exercised only by stubbed test harnesses

    def _build_arc_no_mcp_execution_prompt(*, has_memory: bool, generation: int, test_count: int = 1) -> str:
        return build_execution_prompt(None)  # type: ignore[arg-type]

    def _build_arc_no_mcp_task_markdown(task):  # type: ignore[no-untyped-def]
        return build_task_markdown(task)


from ..tokens import TokenUsage
from ..trace_events import append_trace_event, get_trace_dir
from .barrier import (
    BarrierEvent,
    BarrierWatcher,
)
from .normalize import (
    SILENT_FAILURE_MESSAGE,
    SILENT_FAILURE_STATUS,
    SilentAgentRuntimeError,
    extract_token_usage,
    extract_tsc_compile_error,
    parse_runner_stdout,
)
from .seeding import (
    repo_source_path,
    safe_read_text,
    seed_package_to_memory_md,
    workspace_task_files,
)

# SWE-bench Pro Docker image build/tag/ensure helpers live in ``swebench_images``.
# Re-imported here so the ``container_host.<name>`` access path (used by callers
# and tests) keeps resolving, along with the shared low-level helpers
# (``_scrub_credentials`` / ``_tail`` / ``_is_enabled_env``) co-located there to
# avoid an import cycle.
from .swebench_images import (
    _DOCKER_IMAGE_ID_CACHE,  # noqa: F401  re-exported for tests
    _docker_image_id,  # noqa: F401  re-exported for tests
    _is_enabled_env,
    _scrub_credentials,
    _swebench_agent_overlay_dockerfile,  # noqa: F401  re-exported for tests
    _swebench_official_base_image,  # noqa: F401  re-exported for tests
    _swebench_pro_container_images,
    _tail,
)
from .types import RuntimeResult

log = logging.getLogger(__name__)

_parse_runner_stdout = parse_runner_stdout


def _arc_payload_for_workspace(task: TaskSpec) -> dict[str, Any]:
    """HA-compatible payload.json: train pairs + test inputs, no test outputs.

    Reads from ``task.metadata`` (the ARC loaders normalize source data into
    ``arc_train_pairs`` / ``arc_test_inputs``). Test outputs are deliberately
    omitted so the agent cannot copy the answer.
    """
    metadata = task.metadata or {}
    train = metadata.get("arc_train_pairs") or []
    test_inputs = metadata.get("arc_test_inputs") or []
    if not isinstance(test_inputs, list) or not test_inputs:
        legacy = metadata.get("arc_test_pairs") or []
        if isinstance(legacy, list):
            test_inputs = [{"input": p.get("input")} for p in legacy if isinstance(p, dict)]
    return {
        "task_id": task.id,
        "train": [
            {"input": pair.get("input"), "output": pair.get("output")} for pair in train if isinstance(pair, dict)
        ],
        "test": [{"input": pair.get("input")} for pair in test_inputs if isinstance(pair, dict)],
    }


def _arc_grid_summary_md(task: TaskSpec) -> str:
    """Readable rendering of train pairs + test inputs."""
    metadata = task.metadata or {}
    train = metadata.get("arc_train_pairs") or []
    test_inputs = metadata.get("arc_test_inputs") or []
    if not isinstance(test_inputs, list) or not test_inputs:
        legacy = metadata.get("arc_test_pairs") or []
        if isinstance(legacy, list):
            test_inputs = [{"input": p.get("input")} for p in legacy if isinstance(p, dict)]
    lines = ["# Grid Summary", "", f"Task: {task.id}", ""]
    for i, pair in enumerate(train):
        if not isinstance(pair, dict):
            continue
        lines.append(f"## Train pair {i}")
        lines.append("")
        lines.append("Input:")
        lines.append("```")
        for row in pair.get("input") or []:
            lines.append(" ".join(str(c) for c in row))
        lines.append("```")
        lines.append("")
        lines.append("Output:")
        lines.append("```")
        for row in pair.get("output") or []:
            lines.append(" ".join(str(c) for c in row))
        lines.append("```")
        lines.append("")
    for i, pair in enumerate(test_inputs):
        if not isinstance(pair, dict):
            continue
        lines.append(f"## Test input {i}")
        lines.append("")
        lines.append("```")
        for row in pair.get("input") or []:
            lines.append(" ".join(str(c) for c in row))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


_ARC_VALIDATE_PREDICTION_SCRIPT = '''\
#!/usr/bin/env python3
"""Format-only validator for ARC attempt_*.txt files. Does NOT score correctness.

Each input file must be a plain ASCII grid: rows of space-separated integers
0-9, one row per line, rectangular shape, side <=30. Exits 0 on valid, non-zero
with a human-readable message on invalid. Pass one or more file paths as args.
"""
import sys


def _parse_grid_text(text, label):
    rows = []
    for line_num, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s:
            continue
        cells = s.split()
        try:
            row = [int(c) for c in cells]
        except ValueError as e:
            raise ValueError(f"{label}: line {line_num}: non-integer cell ({e})")
        for v in row:
            if v < 0 or v > 9:
                raise ValueError(f"{label}: cell value out of range 0..9: {v}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label}: empty grid")
    width = len(rows[0])
    if width == 0 or width > 30 or len(rows) > 30:
        raise ValueError(f"{label}: dimensions must be 1..30 (got {len(rows)}x{width})")
    if not all(len(r) == width for r in rows):
        raise ValueError(f"{label}: rows have inconsistent widths")
    return rows


def main():
    paths = sys.argv[1:] or ["attempt_1.txt", "attempt_2.txt"]
    failed = False
    for path in paths:
        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            print(f"validate_prediction: {path} not found", file=sys.stderr); failed = True; continue
        try:
            grid = _parse_grid_text(text, label=path)
        except ValueError as e:
            print(f"validate_prediction: {e}", file=sys.stderr); failed = True; continue
        print(f"validate_prediction: {path} OK ({len(grid)}x{len(grid[0])})")
    sys.exit(2 if failed else 0)


if __name__ == "__main__":
    main()
'''
# Sentinel pre-populated into ARC ``attempt_{1,2}.txt`` before the agent runs.
# Must NOT parse as a valid ASCII grid in either ``parseAsciiGrid``
# (runtime_runner/src/main.ts) or ``_parse_grid_text`` above -- both reject any
# token that is not an integer in 0..9, so this string is safely rejected by
# both, and the synthesizer's "no parsed grid -> no submission" branch fires
# when the agent never overwrites. See container_host.py near the
# ``ws_task_files["attempt_1.txt"] = ...`` assignment for context.
_ARC_ATTEMPT_PRESTUB = "__NOT_SUBMITTED__\n"


def _arc_attempt_stub_files(task: TaskSpec) -> dict[str, str]:
    """Return the ``attempt_*.txt`` -> sentinel map seeded for an ARC no-MCP task.

    Single-test tasks (``test_count <= 1``) get only the legacy
    ``attempt_1.txt`` / ``attempt_2.txt`` files (byte-identical to the
    pre-multi-test behavior). Multi-test tasks additionally get per-test
    ``attempt_<k>_<t>.txt`` files (k = 0-based test index, t = trial 1/2) that
    the runtime synthesizer globs to emit a multi-test trace. The legacy
    files are always written too as a cheap safety net (they map to test 0).
    """
    files: dict[str, str] = {
        "attempt_1.txt": _ARC_ATTEMPT_PRESTUB,
        "attempt_2.txt": _ARC_ATTEMPT_PRESTUB,
    }
    test_count = len(_arc_payload_for_workspace(task).get("test") or [])
    if test_count > 1:
        for k in range(test_count):
            files[f"attempt_{k}_1.txt"] = _ARC_ATTEMPT_PRESTUB
            files[f"attempt_{k}_2.txt"] = _ARC_ATTEMPT_PRESTUB
    return files


_extract_token_usage = extract_token_usage
_FORUM_TASK_SOURCES = frozenset({"per_task_forum", "cross_task_forum"})


def _instruction_markdown_for_task(task: TaskSpec, default_instruction_path: str) -> str:
    metadata = task.metadata or {}
    override = metadata.get("instruction_md_override")
    if isinstance(override, str) and override.strip():
        return override.strip() + "\n"
    return safe_read_text(default_instruction_path)


def _error_envelope_event_name(task_source: str) -> str:
    source = str(task_source or "").strip().lower()
    if source in _FORUM_TASK_SOURCES:
        return "runtime.forum_error_envelope"
    return "runtime.error_envelope"


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


def _run_command_with_backstop(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: int | None,
) -> subprocess.CompletedProcess[str]:
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
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                # Force-close pipes — process is truly stuck
                for pipe in (proc.stdout, proc.stderr):
                    if pipe:
                        try:
                            pipe.close()
                        except Exception:
                            pass
                stdout, stderr = "", ""
        raise subprocess.TimeoutExpired(
            cmd=exc.cmd,
            timeout=exc.timeout,
            output=stdout,
            stderr=stderr,
        ) from None
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _unlink_snapshot(path: Path) -> None:
    """Best-effort unlink of a per-task memory snapshot file.

    The snapshot is written into the persistent knowledge-DB directory, so it is
    not auto-reaped with the per-task ``TemporaryDirectory``. Swallow every error
    (missing file, permissions) so cleanup never masks the original failure.
    """
    with contextlib.suppress(Exception):
        path.unlink(missing_ok=True)


def _validate_provider_auth(env: dict[str, str]) -> None:
    """Pre-flight provider/credential validation for the runner env.

    Every failure here is a deterministic misconfiguration (unsupported
    provider, missing model, missing/invalid credential) that can never
    succeed on retry. Raising :class:`AuthenticationFailure` — rather than a
    bare ``ValueError`` — lets the per-task retry loop and the engine's
    phase wrappers fast-abort the whole run at attempt 1 instead of failing
    every task in every generation individually and silently running the
    campaign to 0% solved (the ``_is_auth_error`` substring matcher does not
    recognize these local config messages, so a bare ``ValueError`` was NOT
    converted to a fast-fail). ``AuthenticationFailure`` subclasses
    ``RuntimeError`` and is treated as non-retryable by
    ``_is_retryable_task_error``, so it propagates through
    ``run_task`` -> ``_eval_one_attempt`` -> the collection loop and aborts
    loudly.
    """
    provider = (env.get("MODEL_PROVIDER") or "").strip().lower()
    auth_mode = (env.get("MODEL_AUTH_MODE") or "").strip().lower()
    model = (env.get("MODEL") or "").strip()

    if provider not in {"anthropic", "openai"}:
        raise AuthenticationFailure(f"Unsupported MODEL_PROVIDER={provider!r}; expected 'anthropic' or 'openai'.")
    if not model:
        raise AuthenticationFailure("Model is missing. Set MODEL in your provider profile.")
    if provider == "openai":
        if not env.get("OPENAI_API_KEY"):
            raise AuthenticationFailure("openai provider requires OPENAI_API_KEY.")
    elif provider == "anthropic":
        if auth_mode not in {"api", "subscription"}:
            raise AuthenticationFailure("MODEL_AUTH_MODE must be 'api' or 'subscription'.")
        if auth_mode == "api" and not env.get("ANTHROPIC_API_KEY"):
            raise AuthenticationFailure("anthropic/api mode requires ANTHROPIC_API_KEY.")
        if auth_mode == "subscription" and not env.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise AuthenticationFailure("anthropic/subscription mode requires CLAUDE_CODE_OAUTH_TOKEN.")


def _build_runner_env(base_env: dict[str, str], timeout_sec: int) -> dict[str, str]:
    env = dict(base_env)
    # Egress isolation is managed by the TypeScript host runner, but the Python
    # orchestrator owns the campaign fanout. Stamp a parent-process-scoped run id
    # so all per-task runner subprocesses share one egress proxy/network lease by
    # default instead of each falling back to its own TypeScript PID.
    host_run_id = os.environ.get("KSI_RUN_ID")
    env.setdefault("KSI_RUN_ID", host_run_id.strip() if host_run_id and host_run_id.strip() else f"ksi-{os.getpid()}")
    # Honor operator-provided LOG_LEVEL / ANTHROPIC_LOG from the host env so
    # `LOG_LEVEL=debug ...` surfaces SDK subprocess stderr during silent-fail
    # repros. Without this thread-through, the default below forces "silent"
    # and the container stderr buffer stays empty even when the operator
    # explicitly asked for verbose output.
    for debug_key in ("LOG_LEVEL", "ANTHROPIC_LOG"):
        host_val = os.environ.get(debug_key)
        if host_val and host_val.strip():
            env.setdefault(debug_key, host_val.strip())
    # Web-tool opt-in. Default OFF: WebSearch/WebFetch are a
    # benchmark-solution leak vector and a Claude-vs-GPT scaffold asymmetry, so
    # the agent-runner offers them only when ``KSI_ALLOW_WEB_TOOLS`` is set
    # truthy. ARC stays strictly offline regardless. Thread the flag from the
    # host process env (or provider profile, already merged into ``base_env``)
    # into the runner env so ``container_runner.ts`` forwards it into the
    # container. ``env`` already carries any value provided via the provider
    # profile; this block additionally honors the value when set only on the
    # host ``os.environ``. Same threading pattern as ``OPENAI_PARITY_TOOLS``.
    for web_key in ("KSI_ALLOW_WEB_TOOLS",):
        host_val = os.environ.get(web_key)
        if host_val and host_val.strip():
            env.setdefault(web_key, host_val.strip())
    # Flag-gated OpenAI scaffold parity. Default OFF. Thread the
    # flag from the host process env (or provider profile, already merged into
    # ``base_env``) into the runner env so ``container_runner.ts`` forwards it
    # into the container, where the OpenAI agent-runner enables native
    # read/write/edit/glob/grep tools and the OpenAI direct-ARC path. ``env``
    # already carries any value provided via the provider profile; this block
    # additionally honors the value when set only on the host ``os.environ``.
    for parity_key in ("OPENAI_PARITY_TOOLS",):
        host_val = os.environ.get(parity_key)
        if host_val and host_val.strip():
            env.setdefault(parity_key, host_val.strip())
    # Shared per-turn output-token cap for the direct-ARC paths.
    # Thread the knob from the host env (or provider profile, already merged into
    # ``base_env``) so ``container_runner.ts`` forwards it into the container,
    # where both direct-ARC adapters read it (default 4096). Same threading
    # pattern as ``OPENAI_PARITY_TOOLS``.
    for cap_key in ("KSI_DIRECT_ARC_MAX_TOKENS",):
        host_val = os.environ.get(cap_key)
        if host_val and host_val.strip():
            env.setdefault(cap_key, host_val.strip())
    # Retrieval mode: FTS5 is the default; semantic vector search is opt-in via
    # ``--require-vector``. The engine records the authoritative decision on
    # ``os.environ["MEMORY_ENABLE_SEMANTIC_SEARCH"]`` in its ``__init__`` — but
    # that runs AFTER the provider profile is snapshotted into ``base_env``, so
    # the value is not in ``base_env`` and would otherwise never reach the
    # container. Thread it from the host ``os.environ`` so ``container_args.ts``
    # forwards it and the in-container ``query`` tool stays on FTS too (else a
    # resumed DB carrying a stale ``knowledge_vec`` table from a prior
    # ``--require-vector`` run silently queries stale embeddings that miss every
    # new FTS-only row). Direct assignment, NOT ``setdefault``: the host
    # decision is authoritative and must win over any stale profile value.
    host_semantic = os.environ.get("MEMORY_ENABLE_SEMANTIC_SEARCH")
    if host_semantic and host_semantic.strip():
        env["MEMORY_ENABLE_SEMANTIC_SEARCH"] = host_semantic.strip()
    env.setdefault("LOG_LEVEL", "silent")
    env.setdefault("IDLE_TIMEOUT", "60000")
    if timeout_sec > 0:
        container_timeout_ms = str(max(timeout_sec - 15, 300) * 1000)
    elif timeout_sec == 0:
        # 0 / absent (the CLI default) preserves the historical 1800s hard
        # container safety cap for every task source. This is the default for
        # ARC, polyglot, swebench, etc. — do NOT remove it.
        container_timeout_ms = str(1800 * 1000)
    else:
        # A negative timeout_sec is an EXPLICIT opt-in to disable the hard
        # container deadline entirely (CONTAINER_TIMEOUT <= 0 tells the TS
        # runner to skip its hard-kill timer). Used by TB2 fairness runs so the
        # per-task task.toml timeout is the sole wall-time bound (Harbor parity).
        #
        # SAFE ONLY where a host-side wall-clock backstop already exists. TB2
        # qualifies because its native trial loop
        # (ksi.runtime.terminal_bench_2_trial) enforces
        # ``deadline = now + task.toml [agent].timeout_sec`` itself: every step
        # and every shell exec is capped against that deadline, independent of
        # this TS timer. Task sources that run through the generic
        # container_runner.ts agent-runner path (ARC, polyglot, swebench) have
        # NO such backstop — for them a negative value would leave the container
        # unbounded (a hung/idle run never reaped). Do NOT pass a negative
        # --runtime-timeout-sec for those sources; keep 0/absent (1800s cap).
        container_timeout_ms = "0"
    env["CONTAINER_TIMEOUT"] = container_timeout_ms
    return env


def _seed_package_has_real_memory(seed_package: Any) -> bool:
    """True iff ``seed_package`` carries real prior-attempt or distillation content.

    Gates on actual content keys (``prior_attempts`` / ``per_task_bundle`` /
    ``cross_task_bundle``), not on whether ``seed_package_to_memory_md`` happens
    to emit a non-empty string. At gen 1 the renderer emits a skeleton
    (workstream name + Task ID Reference) even when no real content exists; if
    we used the rendered string as the truthiness signal,
    ``_memory_block(has_memory=True)`` would tell the agent "Prior attempt
    summaries are already in MEMORY.md" when none exist — which Haiku-class
    models then waste tool budget reconciling with an empty file.
    """
    if not isinstance(seed_package, dict):
        return False
    return (
        bool(seed_package.get("prior_attempts"))
        or bool(seed_package.get("per_task_bundle"))
        or bool(seed_package.get("cross_task_bundle"))
    )


def _build_tools_md(
    *,
    task: TaskSpec,
    provider: str,
    has_memory_mcp: bool,
    has_injected_memory: bool = False,
    web_tools_enabled: bool = False,
) -> str:
    metadata = task.metadata or {}
    task_source = str(metadata.get("task_source") or "").strip().lower()
    forum_phases = set(_FORUM_TASK_SOURCES)
    is_forum_phase = task_source in forum_phases
    forum_round = int(metadata.get("forum_round") or 0) if is_forum_phase else 0

    lines = [
        "# TOOLS",
        "",
        "This run exposes KSI-owned tools plus provider-native tools.",
        "Use KSI tools for protocol-critical work. Use provider-native tools as execution aids.",
        "",
        "## KSI Tools",
    ]

    ksi_tools: list[str] = []
    if is_forum_phase:
        if has_memory_mcp:
            if provider == "openai":
                ksi_tools.append(
                    "query: retrieve compact exact task memory and optional semantic related hits with query text."
                )
                ksi_tools.append("knowledge: read the full knowledge page for a task.")
                ksi_tools.append("forum_post: post a task-specific or cross-task forum message.")
                ksi_tools.append("forum_signal_done: signal that your forum contribution is complete.")
                ksi_tools.append("forum_read: read current-generation forum messages when needed.")
            else:
                ksi_tools.append("query / mcp__memory__query: inspect task memory when exact evidence is needed.")
                ksi_tools.append("forum_post / mcp__memory__forum_post: post a forum message.")
                ksi_tools.append("forum_signal_done / mcp__memory__forum_signal_done: signal completion.")
            if forum_round >= 2:
                ksi_tools.append(
                    "forum_read / mcp__memory__forum_read: read the current forum board before commenting."
                )
        else:
            ksi_tools.append("No KSI memory/forum MCP tools are mounted for this run.")
    elif has_memory_mcp or has_injected_memory:
        ksi_tools.append("Task memory is pre-injected into MEMORY.md for normal task execution.")
    else:
        ksi_tools.append("No KSI MCP tools are required for this run.")

    lines.extend(f"- {item}" for item in ksi_tools)
    lines.extend(["", "## Provider-Native Tools"])

    if provider == "openai":
        native_tools = [
            "shell: run shell commands inside the task container.",
            "apply_patch: edit files using patch application.",
            "OpenAI native tool semantics drive execution; KSI MCP tools are mounted alongside them.",
        ]
    else:
        # ARC is a sealed offline puzzle benchmark. The native ARC path has the
        # agent read payload.json and write its attempt files with the standard
        # file/shell tools; only web tools are withheld for this offline source.
        _native_spec = resolve_source(task_source)
        is_offline = bool(_native_spec and _native_spec.is_offline)
        if is_offline:
            native_tools = [
                (
                    "Bash, Read, Write, Edit, Glob, Grep: native Claude tools for reading "
                    "payload.json and writing your attempt files."
                ),
                "WebSearch and WebFetch are disabled for this offline benchmark.",
            ]
        else:
            native_tools = [
                "Bash, Read, Write, Edit, Glob, Grep: native Claude coding tools.",
                "TodoWrite, NotebookEdit: Claude utility tools enabled for scheduled benchmark runs.",
            ]
            # WebSearch/WebFetch are OFF by default for benchmark tasks
            # (solution-leak vector). The runtime DENIES them via
            # the SDK `disallowedTools` field unless KSI_ALLOW_WEB_TOOLS is
            # truthy, so this line must reflect the effective state or the agent
            # is told it has tools the runtime blocks. See benchmarks/docs/web_tools_policy.md.
            if web_tools_enabled:
                native_tools.append("WebSearch, WebFetch: enabled for this run (KSI_ALLOW_WEB_TOOLS is set).")
            else:
                native_tools.append(
                    "WebSearch and WebFetch are disabled by default for benchmark tasks "
                    "(set KSI_ALLOW_WEB_TOOLS=1 to enable)."
                )
            native_tools.extend(
                [
                    "ToolSearch and Skill are intentionally disabled for scheduled benchmark runs to avoid sidechain/helper-agent drift.",
                    "Claude native tool semantics drive execution; KSI MCP tools are mounted alongside them.",
                ]
            )
    lines.extend(f"- {item}" for item in native_tools)
    return "\n".join(lines).strip() + "\n"


def _resolve_deferred_workspace_dir(workspace_file: Path, stop_event: threading.Event) -> Path | None:
    """Poll ``workspace_file`` for the host-visible workspace path written by
    ``main.ts``, returning it as a :class:`Path` once present.

    Threading the deferred watcher's ``stop_event`` into the resolve loop is
    critical: ``with tempfile.TemporaryDirectory(...)`` in ``run_task`` cleans
    up the host-side scratch dir BEFORE the ``finally`` calls
    ``watcher.stop()``. Without this check, the deferred thread keeps polling a
    deleted ``workspace_file`` for up to 60s on every short successful run — at
    ``--max-concurrent-tasks=25`` that piles up 25 zombie daemon threads per
    generation. Polling the stop_event between iterations lets the thread exit
    within one poll interval (~200ms) of ``stop()``.
    """
    deadline = time.monotonic() + 60.0  # main.ts writes this very early
    while time.monotonic() < deadline:
        if stop_event.is_set():
            return None
        try:
            raw = workspace_file.read_text(encoding="utf-8").strip()
            if raw:
                return Path(raw)
        except FileNotFoundError:
            pass
        except Exception:
            pass
        # Use stop_event.wait() rather than time.sleep so the thread also
        # unblocks immediately when stop() fires mid-sleep.
        if stop_event.wait(timeout=0.2):
            return None
    return None


class _DeferredBarrierWatcher(threading.Thread):
    """Wrapper thread that defers constructing a real :class:`BarrierWatcher`
    until ``main.ts`` writes the host-visible workspace path to
    ``workspace_file``, then promotes itself into the inner watcher.

    Shared by :meth:`KsiContainerExecutor._launch_phase1_watcher` and
    :meth:`KsiContainerExecutor._launch_cross_task_r1_watcher`. The host
    doesn't know the workspace dir at launch time, so this thread polls for the
    path file (via :func:`_resolve_deferred_workspace_dir`) and only constructs
    the inner watcher once the path appears.

    Failure modes (both fine):
      * ``workspace_file`` never appears within 60s — the thread exits without
        an inner watcher (the attempt itself is unaffected; no reflection).
      * ``stop()`` fires before the inner watcher starts — the host subprocess
        already finished, so a now-useless inner watcher is never constructed.

    ``cached_eval_holder`` is exposed as ``_cached_eval_holder`` so the phase-1
    caller can ``getattr(watcher, "_cached_eval_holder")`` after
    ``proc.communicate()`` returns; cross-task callers pass ``None``.
    """

    def __init__(
        self,
        *,
        thread_name: str,
        log_tag: str,
        workspace_file: Path,
        barrier_name: str,
        agent_id: str,
        callback: Callable[[BarrierEvent], dict[str, Any]],
        poll_timeout_sec: float,
        cached_eval_holder: dict[str, Any] | None = None,
        persistent: bool = False,
        stop_join_timeout_sec: float = 2.0,
    ) -> None:
        super().__init__(daemon=True, name=thread_name)
        self._log_tag = log_tag
        self._workspace_file = workspace_file
        self._barrier_name = barrier_name
        self._agent_id = agent_id
        self._callback = callback
        self._poll_timeout_sec = poll_timeout_sec
        self._persistent = persistent
        self._stop_join_timeout_sec = float(stop_join_timeout_sec)
        self._stop_event = threading.Event()
        self._inner: BarrierWatcher | None = None
        self._cached_eval_holder = cached_eval_holder

    def run(self) -> None:
        ws = _resolve_deferred_workspace_dir(self._workspace_file, self._stop_event)
        if ws is None:
            # Two cases land here, both fine:
            #  (a) workspace_file never appeared within 60s — the container
            #      never wrote it (no reflection, but the attempt itself is
            #      unaffected).
            #  (b) stop() fired while we were polling — the host subprocess
            #      already finished and we shouldn't construct a now-useless
            #      inner watcher.
            log.info(
                "[%s] deferred BarrierWatcher exiting without inner watcher: "
                "workspace path file %s never appeared or stop() fired",
                self._log_tag,
                self._workspace_file,
            )
            return
        if self._stop_event.is_set():
            return
        inner = BarrierWatcher(
            workspace_dir=ws,
            name=self._barrier_name,
            agent_id=self._agent_id,
            callback=self._callback,
            timeout_sec=self._poll_timeout_sec,
            persistent=self._persistent,
        )
        self._inner = inner
        inner.start()
        # Block this deferred thread until inner exits or stop().
        while inner.is_alive():
            if self._stop_event.wait(timeout=0.5):
                inner.stop()
                break
        inner.join(timeout=self._stop_join_timeout_sec)

    def stop(self) -> None:
        # Set the stop event FIRST so any in-progress
        # _resolve_deferred_workspace_dir() poll exits on its next iteration
        # (without ever constructing an inner watcher). The inner watcher
        # branch below is a no-op when the deferred thread never reached the
        # BarrierWatcher start (the common case for short successful runs).
        self._stop_event.set()
        if self._inner is not None:
            try:
                self._inner.stop()
            except Exception:
                pass


def _raise_runner_failure(
    proc: subprocess.CompletedProcess[str],
    *,
    trace_dir: str,
    generation: int,
    agent_id: str,
    task: TaskSpec,
) -> None:
    """Emit a ``runtime.error`` trace event and raise for a non-zero runner exit.

    Surfaces the ``container/entrypoint.sh`` ``TSC_COMPILE_FAILED`` excerpt (when
    present) prominently so the raised ``RuntimeError`` — which the engine stores
    as ``trace.error`` / ``error_text`` — is diagnosable instead of an opaque
    "Container exited with code 2: ". Always raises.
    """
    stderr_tail = _scrub_credentials(_tail(proc.stderr, 3000))
    tsc_excerpt = extract_tsc_compile_error(proc.stderr)
    if tsc_excerpt:
        tsc_excerpt = _scrub_credentials(tsc_excerpt)
    append_trace_event(
        trace_dir,
        "runtime_events.jsonl",
        {
            "event": "runtime.error",
            "generation": generation,
            "agent_id": agent_id,
            "task_id": task.id,
            "returncode": proc.returncode,
            "stderr_tail": stderr_tail,
            "tsc_compile_error": tsc_excerpt or "",
        },
    )
    if tsc_excerpt:
        raise RuntimeError(
            f"Shared container runner failed (exit={proc.returncode}) — "
            f"tsc compile error:\n{tsc_excerpt}\n"
            f"(stderr tail: {stderr_tail})"
        )
    raise RuntimeError(f"Shared container runner failed (exit={proc.returncode}): {stderr_tail}")


def _validate_runner_protocol(
    runtime_meta_parsed: dict[str, Any],
    *,
    generation: int,
    agent_id: str,
    task: TaskSpec,
    stdout: str | None,
    trace_dir: str,
) -> None:
    """Assert the runner echoed back the expected generation/agent/task identity.

    A mismatch means the container stdout protocol was violated (e.g. a stale or
    cross-wired envelope); emit a ``runtime.parse_error`` trace event and raise.
    Returns ``None`` when the identity matches.
    """
    protocol_errors: list[str] = []
    try:
        parsed_generation = int(runtime_meta_parsed.get("generation"))
    except (TypeError, ValueError):
        parsed_generation = None
    if parsed_generation != int(generation):
        protocol_errors.append(
            f"meta.generation={runtime_meta_parsed.get('generation')!r} does not match expected {generation!r}"
        )
    if str(runtime_meta_parsed.get("agent_id") or "") != str(agent_id):
        protocol_errors.append(
            f"meta.agent_id={runtime_meta_parsed.get('agent_id')!r} does not match expected {agent_id!r}"
        )
    if str(runtime_meta_parsed.get("task_id") or "") != str(task.id):
        protocol_errors.append(
            f"meta.task_id={runtime_meta_parsed.get('task_id')!r} does not match expected {task.id!r}"
        )
    if protocol_errors:
        append_trace_event(
            trace_dir,
            "runtime_events.jsonl",
            {
                "event": "runtime.parse_error",
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task.id,
                "stdout": (stdout or "")[:2000],
                "error": "; ".join(protocol_errors),
            },
        )
        raise RuntimeError(
            f"Container stdout protocol error for task {task.id} (agent={agent_id}, generation={generation})"
        )


def _finalize_runner_result(
    parsed: dict[str, Any],
    *,
    generation: int,
    agent_id: str,
    task: TaskSpec,
    trace_dir: str,
) -> RuntimeResult:
    """Apply terminal-status handling to a parsed runner envelope and return the
    :class:`RuntimeResult` — or raise when the run must be treated as a failure.

    Three terminal cases:

    * ``status == silent-failure`` — the container exited cleanly but the
      agent-runner emitted no model_output / tool calls / tokens. Raise a
      :class:`SilentAgentRuntimeError` carrying ``runtime_meta`` so the engine
      records ``error_text`` and preserves ``native_session_memory`` instead of
      counting a 0-score success.
    * ``status == "error"`` — an adapter-side error envelope with exit-code 0.
      For ``swebench_pro`` / ``polyglot`` max-turn errors that left a workspace
      artifact, salvage the partial solution (mark ``salvaged_error*`` and return
      it); otherwise raise ``SilentAgentRuntimeError``.
    * otherwise — return the normal ``RuntimeResult``.

    ``source`` is read raw (not lowercased) from ``task.metadata`` to match the
    historical salvage gate.
    """
    metadata = task.metadata or {}
    meta = parsed.get("runtime_meta") or {}
    status_lower = str(meta.get("status") or "").lower()
    if status_lower == SILENT_FAILURE_STATUS:
        error_msg = str(meta.get("error") or SILENT_FAILURE_MESSAGE)
        append_trace_event(
            trace_dir,
            "runtime_events.jsonl",
            {
                "event": "runtime.silent_failure",
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task.id,
                "task_source": str(metadata.get("task_source") or ""),
                "message": error_msg,
            },
        )
        raise SilentAgentRuntimeError(
            f"Silent agent-runner failure for task {task.id} (agent={agent_id}, generation={generation}): {error_msg}",
            runtime_meta=meta,
        )
    if status_lower == "error":
        error_msg = str(meta.get("error") or "agent-runner returned status=error")
        source = str(metadata.get("task_source") or "")
        has_workspace_artifact = bool(
            str(meta.get("workspace_diff") or "").strip()
            or (isinstance(meta.get("workspace_solution_files"), dict) and bool(meta.get("workspace_solution_files")))
        )
        max_turns_error = "maxturn" in error_msg.replace(" ", "").lower()
        if source in {"swebench_pro", "polyglot"} and has_workspace_artifact and max_turns_error:
            meta["salvaged_error_status"] = status_lower
            meta["salvaged_error"] = error_msg
            parsed["runtime_meta"] = meta
            return RuntimeResult(
                output=parsed["output"],
                tool_trace=parsed["tool_trace"],
                runtime_meta=parsed["runtime_meta"],
                token_usage=parsed["token_usage"],
            )
        append_trace_event(
            trace_dir,
            "runtime_events.jsonl",
            {
                "event": _error_envelope_event_name(source),
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task.id,
                "task_source": source,
                "message": error_msg,
            },
        )
        raise SilentAgentRuntimeError(
            f"agent-runner error envelope for task {task.id} (agent={agent_id}, generation={generation}): {error_msg}",
            runtime_meta=meta,
        )
    return RuntimeResult(
        output=parsed["output"],
        tool_trace=parsed["tool_trace"],
        runtime_meta=parsed["runtime_meta"],
        token_usage=parsed["token_usage"],
    )


# ---------------------------------------------------------------------------
# Barrier-watcher container-side poll-timeout formulas.
#
# The three barrier features share an identical host-side setup shape (see
# ``KsiContainerExecutor._setup_barrier``), but each computes the deadline the
# *container* waits for its sentinel response DIFFERENTLY. Keeping the three
# formulas as adjacent, individually-named pure functions makes them visually
# diffable -- CLAUDE.md records two prior production bugs where one feature
# silently inherited another's hardcoded poll timeout. NOTE the differing signs:
# phase1 SUBTRACTS a margin from a large session budget; cross-task NESTS three
# deadlines below the container hard-kill; polyglot ADDS a margin on top of the
# evaluator's own Docker run (``polyglot_timeout`` IS the callback's worst case,
# so subtracting would let the barrier give up before the host legitimately
# finishes). Do not "unify" these into one expression.


def _phase1_poll_timeout_ms(effective_timeout: int) -> int:
    """Phase-1 reflection container-side ``eval_result_poll_timeout_ms``.

    Allow the host's ``effective_timeout`` minus a 60s margin so the
    in-container poll never outruns the subprocess backstop; floor at 30s and
    only trust the derived value once the session budget clears 90s.
    """
    return max(30_000, (effective_timeout - 60) * 1000) if effective_timeout > 90 else 30_000


def _cross_task_timeouts(effective_timeout: int) -> tuple[int, int]:
    """Cross-task R0->R1 nested ``(container_timeout_sec, poll_timeout_sec)``.

    Nest the deadlines so the in-container poll's own graceful R0-only fallback
    stays reachable: ``coord_timeout < poll_timeout_sec <= container_timeout_sec``.
    ``container_timeout_sec`` mirrors ``_build_runner_env``'s CONTAINER_TIMEOUT
    (the container's hard external kill deadline); ``poll_timeout_sec`` sits just
    below it so the container gives up gracefully BEFORE the external kill, and
    above the coordinator's own wait so an agent never abandons the barrier while
    the host is still legitimately draining R0 posts and building R1 prompts.
    Timeline for the default 900s forum timeout: coordinator 855s < poll 880s
    <= container hard-kill 885s.
    """
    container_timeout_sec = max(effective_timeout - 15, 300)
    poll_timeout_sec = max(30, container_timeout_sec - 5)
    return container_timeout_sec, poll_timeout_sec


def _polyglot_tf_poll_timeout_ms(polyglot_timeout: int) -> int:
    """Polyglot test-feedback container-side ``evalResultPollTimeoutMs``.

    The host callback runs a full ``PolyglotHarnessEvaluator`` Docker evaluation
    bounded by the evaluator's own ``--polyglot-timeout-sec`` (default 180s) --
    not the agent session's ``effective_timeout``. Unlike phase1's margin
    (subtracted from a much larger session budget), ``polyglot_timeout`` here IS
    the callback's own worst-case duration, so the container's wait must be ADDED
    on top of it (not subtracted) or the barrier can give up before the host's
    own Docker run legitimately finishes.
    """
    return max(30_000, (polyglot_timeout + 60) * 1000)


@dataclass(frozen=True)
class _RunTaskContext:
    """Resolved per-attempt context produced by
    :meth:`KsiContainerExecutor._prepare_run_context`: the task (possibly
    rebuilt with swebench container-image metadata), the source-derived ARC
    tool flags, and the fully built runner ``payload`` plus its resolved
    ``memory_md`` surface. Carried into the temp-dir execution block so
    ``run_task`` stays an orchestration skeleton.

    ``frozen=True`` is SHALLOW: it only blocks rebinding the fields. The
    ``payload`` (and ``metadata``) dicts are deliberately mutated in place by the
    ``_maybe_setup_*`` helpers (e.g. injecting ``phase1_reflection`` /
    ``cross_task_shared_container`` keys before the payload is re-written to
    disk) — matching the pre-refactor local-dict semantics. Do not read
    ``frozen`` as a deep-immutability guarantee.
    """

    trace_dir: Path
    task: TaskSpec
    metadata: dict[str, Any]
    source: str
    swebench_container_images: dict[str, str]
    arc_no_mcp_active: bool
    seed_package: Any
    experiment_name: str
    payload: dict[str, Any]  # mutated in place by _maybe_setup_* (see class docstring)
    memory_md: str


def _polyglot_workspace_runtime_meta(sentinel_path: Path) -> dict[str, Any]:
    """Build the ``runtime_meta`` kwarg for a mid-session polyglot evaluate() call.

    ``sentinel_path`` is a ``BarrierEvent.sentinel_path`` — always
    ``<workspace root>/<sentinel filename>`` (see ``barrier.py``'s
    ``sentinel_path`` property; the TS side writes the sentinel at
    ``workspaceDir: CONTAINER_WORKSPACE_ROOT``, the host dir mounted as
    ``/workspace/task``). The agent's exercise repo is NOT that root:
    ``seedWorkspace`` (workspace.ts) copies it to ``<root>/workspace/repo``,
    the same dir ``captureWorkspaceArtifacts`` (main.ts) publishes as the
    post-session ``host_workspace_repo_dir``. Point the evaluator at that
    repo dir so ``_solution_files_from_workspace`` reads the agent's live
    on-disk edits instead of falling back to fenced-code-block extraction
    from ``model_output`` text. (Handing it the root scored ``no_solution``
    whenever the round's message had no fenced block:
    ``_workspace_repo_dir`` only normalizes ONE ``repo/`` level, so the
    root resolved to a dir with no candidate files.) If the repo dir does
    not exist, ``_workspace_repo_dir`` returns ``None`` and the evaluator
    degrades to the fenced-block fallback exactly as before.
    """
    return {"host_workspace_repo_dir": str(sentinel_path.parent / "workspace" / "repo")}


@dataclass
class KsiContainerExecutor:
    """Shared host-side container runtime for provider-backed task execution."""

    command: list[str]
    working_dir: str = "."
    timeout_sec: int = 1800
    env: dict[str, str] = field(default_factory=dict)
    output_json_key: str = "result"
    session_scope: str = "task"
    wipe_workspace_per_task: bool = True
    instruction_path: str = "templates/INSTRUCTION.md"
    agent_workspace_root: str = "workspaces"
    knowledge_db_path: str = ""
    runtime_db_path: str = ""
    disable_memory_mcp: bool = False
    forum_timeout_sec: int = 900
    cross_task_forum_timeout_sec: int = 900
    # Phase-1 self-reflection (Path a). When ``True`` AND ``evaluator`` is
    # set, the host launches a BarrierWatcher thread that polls the
    # workspace for a sentinel written by the in-container agent after the
    # task completes; the watcher runs the evaluator and writes the result
    # back so the agent can produce a 3-5 sentence structured reflection.
    # Default off — flip on via ``--phase1-reflection-enabled`` after
    # smoke-testing on a small subset.
    phase1_reflection_enabled: bool = False
    # Raw-attempts memory mode (ablation). When True, MEMORY.md is built from
    # ONLY the raw prior-attempt model_output + eval detail — no distilled
    # bundles, no insights, no condensed-approach reflection. Defaults False
    # (preserves existing behavior).
    memory_seed_raw_attempts: bool = False
    # ARC native mode. The legacy ARC MCP toolset has been removed, so ARC is
    # always native: registry passes ``arc_no_mcp=True``. For ARC tasks the
    # host forces the SDK adapter (KSI_ANTHROPIC_ARC_ADAPTER=claude-code),
    # seeds payload.json / grid_summary.md / validate_prediction.py into the
    # workspace, and emits the native ARC system prompt and TASK.md. Retained
    # as a field so the registry wiring and public arg namespace stay stable.
    arc_no_mcp: bool = True
    # Optional handle to the orchestrator's evaluator. The engine assigns
    # this after constructing both the executor and the evaluator (see
    # ``GenerationalOrchestrator.__init__``). Left as ``Any`` to avoid a
    # circular import on ``ksi.eval``.
    evaluator: Any | None = None

    def _run_runner_command(
        self,
        cmd: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int | None,
    ) -> subprocess.CompletedProcess[str]:
        return _run_command_with_backstop(cmd, cwd=cwd, env=env, timeout=timeout)

    def _launch_phase1_watcher(
        self,
        *,
        workspace_file: Path,
        agent_id: str,
        task: TaskSpec,
        poll_timeout_sec: float,
    ) -> "BarrierWatcher | None":
        """Spawn a BarrierWatcher that runs ``evaluator.evaluate`` on the
        agent's submission and writes the result back across the barrier.

        The watcher waits up to ~5 seconds for ``main.ts`` to write the
        host-visible workspace path to ``workspace_file``; if that never
        happens, it gives up so the container run still completes (no
        reflection, but no failure either).
        """
        if self.evaluator is None:
            return None

        # Wait briefly (in a background thread) for main.ts to populate
        # workspace_file. Once it's available, start the BarrierWatcher
        # against ``<workspace>/workspace/task``.
        evaluator = self.evaluator

        # Shared slot the callback fills with the evaluator's raw return
        # so the executor can re-read it AFTER proc.communicate() and
        # cache it into the returned runtime_meta. The engine then sees
        # the cached result in _eval_stage and skips re-running the
        # evaluator (avoiding double Docker invocations on
        # polyglot / swebench_pro and avoiding score disagreement when
        # the watcher and engine see different inputs).
        cached_eval_holder: dict[str, Any] = {"value": None, "error": None}

        # The watcher's callback closure
        def _on_barrier(event: BarrierEvent) -> dict[str, Any]:
            payload = event.payload or {}
            model_output = payload.get("model_output") or ""
            # Supply the agent's live on-disk workspace repo — exactly what
            # the polyglot test-feedback watcher does — so workspace-scored
            # evaluators (polyglot, swebench_pro) grade the real submitted
            # files instead of falling back to fenced-code-block / patch
            # extraction from ``model_output`` text. Passing ``{}`` here made
            # the cached phase1 result score from ``model_output`` only, and
            # because the phase1 reflection turn runs with NO filesystem
            # access (query_runner.ts: ``allowedTools: []``/``mcpServers:
            # {}``) the workspace is frozen at barrier time — so that
            # wrong-input score was reused verbatim as the FINAL score (see
            # execution_phase.py's ``phase1_eval_result`` reuse), producing
            # spurious ``no_solution``/``no_patch`` on runs with real work.
            # The helper is polyglot-named but its ``host_workspace_repo_dir``
            # payload is generic: swebench_pro's ``_patch_from_workspace``
            # reads the same key, and ``captureWorkspaceArtifacts`` (main.ts)
            # publishes ``<root>/workspace/repo`` for both sources. ARC never
            # reaches this callback (gated out in ``_maybe_setup_phase1_
            # reflection``) and has no workspace scorer, so it is unaffected.
            try:
                eval_result = evaluator.evaluate(
                    task=task,
                    model_output=model_output,
                    runtime_meta=_polyglot_workspace_runtime_meta(event.sentinel_path),
                    tool_trace=[],
                )
            except Exception as exc:
                # Record the error so the engine knows the watcher tried
                # but failed; engine will fall back to its own
                # evaluate() call rather than silently ship a 0.
                cached_eval_holder["error"] = f"{type(exc).__name__}: {exc}"
                return {
                    "status": "evaluator_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            # Cache the *raw* eval_result for engine reuse — preserve
            # whatever shape the evaluator returned (dict, EvalResult
            # dataclass, etc.) so the engine sees an identical object to
            # what it would have computed itself.
            cached_eval_holder["value"] = eval_result
            response: dict[str, Any] = {}
            if isinstance(eval_result, dict):
                response.update(eval_result)
                # Best-effort: surface a flat ``score`` key for the prompt.
                if "native_score" not in response:
                    if "score" in response:
                        response["native_score"] = response["score"]
            else:
                response["raw"] = str(eval_result)
            return response

        # Defer constructing the real BarrierWatcher until main.ts writes the
        # workspace path; the shared wrapper polls for it then promotes itself.
        watcher = _DeferredBarrierWatcher(
            thread_name=f"barrier-watcher-deferred-{agent_id}",
            log_tag="phase1",
            workspace_file=workspace_file,
            barrier_name="phase1_reflection",
            agent_id=agent_id,
            callback=_on_barrier,
            poll_timeout_sec=poll_timeout_sec,
            cached_eval_holder=cached_eval_holder,
        )
        watcher.start()
        # Cast for the caller's sake -- the deferred wrapper exposes
        # `.stop()` which is all the caller needs.
        return watcher  # type: ignore[return-value]

    def _launch_polyglot_test_feedback_watcher(
        self,
        *,
        workspace_file: Path,
        agent_id: str,
        task: TaskSpec,
        poll_timeout_sec: float,
        max_rounds: int,
    ) -> "BarrierWatcher | None":
        """Spawn a BarrierWatcher for the polyglot test-feedback retry loop.

        Unlike ``_launch_phase1_watcher``, this callback supplies a real
        ``host_workspace_repo_dir`` (via ``_polyglot_workspace_runtime_meta``)
        so the evaluator scores the agent's LIVE on-disk edits rather than
        falling back to fenced-code-block extraction from ``model_output``.
        ``cached_eval_holder`` caches the FULL raw ``eval_result`` (same
        shape phase1's watcher caches) so ``_postprocess_runner_output`` can
        reuse it and skip a second Docker evaluation -- but ONLY when the
        TS side reports ``final_eval_matches_output: true`` (the last
        barrier round resolved and no further agent turn ran after it); a
        round that exhausts its tries after an edit turn evaluated the
        PRE-turn state, so that cached value must never be reused as the
        final score. When reuse is eligible, this raw value (including raw
        ``test_stdout_tail``/``test_stderr_tail``) DOES get written into
        ``runtime_meta_parsed["polyglot_test_feedback_eval_result"]`` and
        from there into ``runtime_meta_json`` in the non-agent-facing
        runtime audit sidecar DB (see ``src/ksi/memory/parity.py``) --
        mirroring the pre-existing ``phase1_eval_result`` pattern above.
        This is NOT the agent-facing knowledge substrate: no code path
        reads ``runtime_meta_json`` back into MEMORY.md, the forum, or
        distillation (security.md Finding 3 / leakage-parity.md Finding 1).
        The separate, still-sanitized ``attempt_1_eval_result`` research
        bookkeeping field IS what reaches the agent-facing knowledge DB,
        and is built independently on the TS side (``summarizeEvalResult``
        in ``polyglot_test_feedback.ts``); it never reads this holder.

        ``persistent=True``: unlike the single-shot phase1/cross-task
        watchers, the TS-side retry loop (``runPolyglotTestFeedback``) can
        write MULTIPLE sentinels over the task's lifetime — one per retry
        round, up to ``triesRemaining - 1``, plus one final post-loop
        check — and this watcher's only ``stop()`` call happens after the
        whole container subprocess exits (see
        ``_execute_runner_with_fallback``'s ``finally``). A single-shot
        watcher would answer round 0 then exit, leaving round 1+ sentinels
        unanswered until the TS side's own poll timeout expires.

        ``max_rounds`` is a HOST-SIDE cap on the number of barrier rounds
        this watcher will answer, independent of anything the container
        claims. The container already has unrestricted Bash/Write access to
        the same workspace directory the barrier protocol polls, so without
        this cap an adversarial or compromised agent could write sentinel
        files directly in a loop and force unbounded real Docker
        evaluations (security.md Finding 1). Must equal the honest
        protocol's total round count: ``triesRemaining`` (the
        ``triesRemaining - 1`` retry-loop rounds plus 1 final post-loop
        check -- see polyglot_test_feedback.ts).
        """
        if self.evaluator is None:
            return None
        evaluator = self.evaluator
        cached_eval_holder: dict[str, Any] = {"value": None, "error": None}
        rounds_answered = {"count": 0}

        def _on_barrier(event: BarrierEvent) -> dict[str, Any]:
            if rounds_answered["count"] >= max_rounds:
                log.warning(
                    "barrier polyglot_test_feedback/%s: refusing round %d (max_rounds=%d) — "
                    "the container requested more barrier rounds than "
                    "--polyglot-test-feedback-tries allows",
                    agent_id,
                    rounds_answered["count"],
                    max_rounds,
                )
                return {"status": "round_limit_exceeded", "error": "max retry rounds exceeded"}
            rounds_answered["count"] += 1
            payload = event.payload or {}
            model_output = payload.get("model_output") or ""
            runtime_meta = _polyglot_workspace_runtime_meta(event.sentinel_path)
            try:
                eval_result = evaluator.evaluate(
                    task=task,
                    model_output=model_output,
                    runtime_meta=runtime_meta,
                    tool_trace=[],
                )
            except Exception as exc:
                cached_eval_holder["error"] = f"{type(exc).__name__}: {exc}"
                return {"status": "evaluator_error", "error": f"{type(exc).__name__}: {exc}"}
            # Cache the *raw* eval_result for potential engine reuse (gated on
            # final_eval_matches_output, see _postprocess_runner_output) —
            # matches phase1's caching, not a leak: this holder itself is
            # never persisted, and the separate research-bookkeeping summary
            # stays sanitized on the TS side regardless of what's cached here.
            cached_eval_holder["value"] = eval_result
            # The FULL eval_result (including raw tails) IS returned across the
            # barrier — the TS side needs it to build the next retry prompt.
            response: dict[str, Any] = {}
            if isinstance(eval_result, dict):
                response.update(eval_result)
            return response

        # Bound how long a deferred-stop join waits for an in-flight Docker
        # evaluation to finish, so `_execute_runner_with_fallback`'s finally
        # can synchronously wait out (not abandon) a callback that's still
        # running when the container gives up on a barrier round --
        # eliminating the race where execution_phase.py's own fallback
        # evaluate() call could otherwise run concurrently with an orphaned
        # watcher thread still mid-Docker-eval on the same task
        # (concurrency-ipc.md Finding 1).
        stop_join_timeout_sec = float(getattr(evaluator, "timeout_sec", 120) or 120) + 15.0

        watcher = _DeferredBarrierWatcher(
            thread_name=f"barrier-watcher-deferred-polyglot-tf-{agent_id}",
            log_tag="polyglot_test_feedback",
            workspace_file=workspace_file,
            barrier_name="polyglot_test_feedback",
            agent_id=agent_id,
            callback=_on_barrier,
            poll_timeout_sec=poll_timeout_sec,
            cached_eval_holder=cached_eval_holder,
            persistent=True,
            stop_join_timeout_sec=stop_join_timeout_sec,
        )
        watcher.start()
        return watcher  # type: ignore[return-value]

    def _maybe_setup_polyglot_test_feedback(
        self,
        *,
        task: TaskSpec,
        agent_id: str,
    ) -> dict[str, Any] | None:
        """Build the ``polyglotTestFeedback`` containerInput block, or None if disabled.

        Always-on for polyglot tasks whose ``polyglot_test_feedback_tries`` >
        1 — there is no separate CLI enable flag; setting tries to 1 IS the
        off-switch (reproduces the old strict single-shot protocol).

        Mirrors ``_maybe_setup_phase1_reflection``'s eligibility gate: no
        evaluator wired means nothing will ever answer the barrier sentinel
        the container writes, so bail out before building a config at all
        (the container-side watcher launcher already refuses independently,
        but without this gate the container still waits out the full
        ``evalResultPollTimeoutMs`` for a response that will never come).
        """
        if self.evaluator is None:
            return None
        meta = task.metadata or {}
        if str(meta.get("task_source") or "") != "polyglot":
            return None
        tries_raw = meta.get("polyglot_test_feedback_tries")
        tries = int(tries_raw) if tries_raw is not None else 2
        if tries <= 1:
            return None
        max_lines_raw = meta.get("polyglot_test_feedback_max_lines")
        max_lines = int(max_lines_raw) if max_lines_raw is not None else 50
        starter_code = meta.get("starter_code") if isinstance(meta.get("starter_code"), dict) else {}
        file_list = ", ".join(sorted(starter_code.keys())) or "the solution file"
        # DISTINCT container-side poll timeout is _polyglot_tf_poll_timeout_ms
        # (it ADDS a margin to the evaluator's own Docker timeout -- see that
        # function's docstring for why the sign differs from phase1's).
        polyglot_timeout = int(
            getattr(self.evaluator, "timeout_sec", DEFAULT_POLYGLOT_TIMEOUT_SEC) or DEFAULT_POLYGLOT_TIMEOUT_SEC
        )
        eval_result_poll_timeout_ms = _polyglot_tf_poll_timeout_ms(polyglot_timeout)
        return {
            "enabled": True,
            "agentId": agent_id,
            "triesRemaining": tries,
            "maxLines": max_lines,
            "fileList": file_list,
            "allowedTools": ["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
            "mcpServers": {},
            "maxTurnsPerRound": 30,
            "evalResultPollTimeoutMs": eval_result_poll_timeout_ms,
        }

    def _setup_polyglot_test_feedback_watcher(
        self,
        *,
        polyglot_tf_cfg: dict[str, Any] | None,
        task: TaskSpec,
        agent_id: str,
        payload: dict[str, Any],
        payload_path: Path,
        runner_env: dict[str, str],
        effective_timeout: int,
        td: str,
        phase1_state: dict[str, Any],
        cross_task_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Wire the polyglot test-feedback retry-loop watcher, given an
        already-built ``polyglot_tf_cfg`` (see ``_maybe_setup_polyglot_test_feedback``,
        called earlier in ``run_task`` so its ``triesRemaining`` can scale
        the session timeout before ``_build_task_runner_env`` runs).
        Mirrors ``_maybe_setup_cross_task_r1``'s workspace-file-reuse shape
        (complexity.md Finding 1: `run_task` was inlining this instead of
        delegating like its two sibling features). Mutates
        ``payload``/``runner_env`` and re-writes ``payload_path`` when
        eligible. Returns ``polyglot_tf_state``.
        """
        polyglot_tf_state: dict[str, Any] = {"watcher": None, "workspace_file": None}
        if polyglot_tf_cfg is None:
            return polyglot_tf_state
        # The DISTINCT container-side poll timeout was already computed into
        # ``polyglot_tf_cfg`` by _maybe_setup_polyglot_test_feedback (called
        # earlier in run_task). Reuse phase1's or cross-task's workspace file if
        # either already claimed it (passed as prior states, in that order).
        return self._setup_barrier(
            payload=payload,
            payload_path=payload_path,
            payload_key="polyglot_test_feedback",
            payload_block=polyglot_tf_cfg,
            runner_env=runner_env,
            td=td,
            prior_states=(phase1_state, cross_task_state),
            launch_watcher=lambda workspace_file: self._launch_polyglot_test_feedback_watcher(
                workspace_file=workspace_file,
                agent_id=agent_id,
                task=task,
                poll_timeout_sec=max(60, effective_timeout + 90),
                max_rounds=int(polyglot_tf_cfg["triesRemaining"]),
            ),
        )

    def _launch_cross_task_r1_watcher(
        self,
        *,
        workspace_file: Path,
        agent_id: str,
        callback: "Any",
        poll_timeout_sec: float,
        barrier_name: str = "cross_task_r1",
    ) -> "BarrierWatcher | None":
        """Spawn a BarrierWatcher that fires when the in-container forum
        agent finishes round 0 of a shared-container cross-task forum
        dispatch. The ``callback`` is provided by the engine — typically a
        ``_CrossTaskR1Coordinator.on_sentinel`` method that synchronizes
        across all agents in the dispatch and returns the per-agent R1
        prompt response dict.

        Mirrors :meth:`_launch_phase1_watcher`'s deferred-resolution
        pattern: the host doesn't know the workspace dir until ``main.ts``
        writes it to ``workspace_file``, so a wrapper thread polls for
        that file then promotes itself into a real ``BarrierWatcher``.

        Failure modes (both fine):
          * ``workspace_file`` never appears within 60s — wrapper exits;
            container side will hit its own poll timeout and emit an
            R0-only envelope.
          * ``stop()`` fires before the inner watcher starts — the
            subprocess already finished; no response to write.
        """
        watcher = _DeferredBarrierWatcher(
            thread_name=f"barrier-watcher-cross-task-r1-{agent_id}",
            log_tag="cross_task_r1",
            workspace_file=workspace_file,
            barrier_name=barrier_name,
            agent_id=agent_id,
            callback=callback,
            poll_timeout_sec=poll_timeout_sec,
        )
        watcher.start()
        return watcher  # type: ignore[return-value]

    def close(self) -> None:
        """No-op -- executor is stateless after seed enrichment refactor."""
        pass

    def _build_runner_payload(
        self,
        *,
        task: TaskSpec,
        generation: int,
        agent_id: str,
        metadata: dict[str, Any],
        source: str,
        swebench_container_images: dict[str, str],
        arc_no_mcp_active: bool,
        experiment_name: str,
        seed_package: Any,
    ) -> tuple[dict[str, Any], str]:
        """Assemble the full runner payload dict and return ``(payload, memory_md)``.

        Covers prompt/tools/memory resolution, the base payload (task,
        workspace_seed, execution_prompt, runtime), and the native ARC
        workspace-file blocks. ``memory_md`` is returned alongside the
        payload because ``run_task`` threads it back into ``runtime_meta`` so
        downstream consumers can reconstruct the memory surface the agent saw.

        The caller still mounts the temp-dir-scoped blocks (``knowledge``,
        ``runtime_audit``, ``phase1_reflection``, ``cross_task_shared_container``)
        onto the returned dict, since those depend on the
        ``tempfile.TemporaryDirectory`` and watcher wiring.
        """
        provider = (
            str((self.env.get("MODEL_PROVIDER") or os.environ.get("MODEL_PROVIDER") or "anthropic")).strip().lower()
        )

        assigned_task_id = seed_package.get("assigned_task_id") if isinstance(seed_package, dict) else None
        seed_memory_md = seed_package_to_memory_md(
            seed_package,
            current_task_id=assigned_task_id or task.id,
            task_source=source,
            raw_mode=bool(self.memory_seed_raw_attempts),
        )
        memory_md = seed_memory_md
        memory_override = metadata.get("memory_md_override")
        if isinstance(memory_override, str) and memory_override.strip():
            memory_md = memory_override.strip()
            if seed_memory_md:
                memory_md += "\n\n## Seed Context\n" + seed_memory_md.strip()

        has_memory_mcp = bool(self.knowledge_db_path) and not self.disable_memory_mcp
        has_injected_memory = _seed_package_has_real_memory(seed_package)

        # Mirror the in-container web-tool gate so TOOLS.md tells the
        # agent the truth: web tools are OFF for benchmark tasks unless
        # KSI_ALLOW_WEB_TOOLS is truthy, and offline sources (e.g. ARC) are
        # always offline. Resolution matches container_runner.ts/_build_runner_env
        # (provider profile in self.env wins, else host os.environ); truthiness
        # matches the TS isWebToolsAllowed via _is_enabled_env.
        _source_spec = resolve_source(source)
        source_is_offline = bool(_source_spec and _source_spec.is_offline)
        web_tools_enabled = (not source_is_offline) and _is_enabled_env(
            self.env.get("KSI_ALLOW_WEB_TOOLS") or os.environ.get("KSI_ALLOW_WEB_TOOLS"),
            default=False,
        )

        tools_md = _build_tools_md(
            task=task,
            provider=provider,
            has_memory_mcp=has_memory_mcp,
            has_injected_memory=has_injected_memory,
            web_tools_enabled=web_tools_enabled,
        )

        payload: dict[str, Any] = {
            "generation": generation,
            "agent_id": agent_id,
            "experiment_name": experiment_name,
            "task": {
                "id": task.id,
                "repo": task.repo,
                "prompt": task.prompt,
                "metadata": task.metadata,
            },
            "workspace_seed": {
                "instruction_md": _instruction_markdown_for_task(task, self.instruction_path),
                "memory_md": memory_md,
                "task_md": (_build_arc_no_mcp_task_markdown(task) if arc_no_mcp_active else build_task_markdown(task)),
                "tools_md": tools_md,
                "task_files": workspace_task_files(task),
                "repo_source_path": repo_source_path(task),
            },
            "execution_prompt": (
                _build_arc_no_mcp_execution_prompt(
                    has_memory=has_injected_memory,
                    generation=generation,
                    test_count=len(_arc_payload_for_workspace(task).get("test") or []),
                )
                if arc_no_mcp_active
                else build_execution_prompt(
                    task,
                    has_memory=has_injected_memory,
                    generation=generation,
                )
            ),
            "runtime": {
                "session_scope": self.session_scope,
                "wipe_workspace_per_task": self.wipe_workspace_per_task,
            },
        }
        if swebench_container_images:
            payload["runtime"].update(swebench_container_images)

        # The legacy ARC MCP toolset has been removed: ARC is always native
        # (the agent reads payload.json and writes attempt files). This block is
        # retained with ``enable`` hard-wired False so the TS side never
        # registers an ARC MCP server; the native path is driven by
        # ``payload["arc_no_mcp"]`` below.
        payload["arc_tools"] = {
            "enable": False,
            "mcp_server_dir": str(Path(__file__).parent.parent / "memory"),
            "task_source": source,
            "task_id": task.id,
        }
        # ARC no-MCP A/C-test wiring: top-level boolean for runtime_runner,
        # plus workspace files the agent reads/writes instead of MCP tools.
        payload["arc_no_mcp"] = bool(arc_no_mcp_active)
        if arc_no_mcp_active:
            ws_seed = payload["workspace_seed"]
            ws_task_files = ws_seed.setdefault("task_files", {})
            if not isinstance(ws_task_files, dict):
                ws_task_files = {}
                ws_seed["task_files"] = ws_task_files
            ws_task_files["payload.json"] = json.dumps(_arc_payload_for_workspace(task), indent=2)
            ws_task_files["grid_summary.md"] = _arc_grid_summary_md(task)
            ws_task_files["validate_prediction.py"] = _ARC_VALIDATE_PREDICTION_SCRIPT
            # Pre-populate placeholder attempt files with a non-parseable
            # sentinel. The agent overwrites them with a real ASCII grid; if it
            # never overwrites, the runtime synthesizer's ``parseAsciiGrid``
            # rejects the sentinel (no integer cells), no submission is
            # synthesized, and the task scores 0 -- the correct outcome.
            #
            # The previous prestub of ``"0\n"`` parsed as a valid 1x1 zero grid
            # and was synthesized as a real submission, which would have
            # silently credited any task whose expected test output happened to
            # be ``[[0]]``. None of the 50 ARC1+ARC2 training tasks in the
            # headline task maps have that signature, so this changes no
            # historical numbers, but the sentinel removes the structural
            # leakage path for future task selections.
            #
            # Multi-test tasks additionally get per-test stub files
            # ``attempt_<k>_<t>.txt`` (k = 0-based test index, t = trial 1/2).
            # The runtime synthesizer
            # (runtime_runner/src/arc_nomcp_synth.ts) globs these to emit a
            # multi-test trace (with ``arc_next_test_input`` between tests) the
            # scorer understands. Legacy ``attempt_1/2.txt`` are still written
            # as a cheap safety net (they map to test 0); single-test tasks are
            # byte-identical to before.
            ws_task_files.update(_arc_attempt_stub_files(task))
        return payload, memory_md

    def _materialize_payload_side_files(
        self,
        *,
        payload: dict[str, Any],
        td: str,
        task: TaskSpec,
        metadata: dict[str, Any],
        agent_id: str,
        generation: int,
        source: str,
        seed_package: Any,
        swebench_container_images: dict[str, str],
        experiment_name: str,
    ) -> tuple[Path | None, bool]:
        """Materialize knowledge / runtime-audit side files into *td* and
        point *payload* at them.

        Returns ``(snapshot_path, snapshot_failed)``: the memory-snapshot path
        (so the caller can unlink it after the run) or ``None``, and a flag that
        is ``True`` when a memory snapshot WAS available but its ``write_text``
        failed — so the container starts cold (memory-less) and the caller can
        stamp ``runtime_meta`` for later analysis. Mutates *payload*
        in place: it may set ``payload["knowledge"]`` and
        ``payload["runtime_audit"]``.
        """
        snapshot_path: Path | None = None
        snapshot_failed = False
        if self.knowledge_db_path:
            knowledge_db = Path(self.knowledge_db_path)
            if not knowledge_db.is_absolute():
                knowledge_db = (Path.cwd() / knowledge_db).resolve()
            knowledge_db.parent.mkdir(parents=True, exist_ok=True)
            knowledge_db.touch(exist_ok=True)

            snapshot_written = False
            if not self.disable_memory_mcp:
                # Prefer pre-built snapshot from EngineEnrichmentPhaseService.enrich()
                snapshot = seed_package.get("memory_snapshot") if isinstance(seed_package, dict) else None
                if snapshot is not None:
                    snapshot_path = (
                        knowledge_db.parent / f"memory_snapshot_{task.id}_{agent_id}_{uuid.uuid4().hex[:8]}.json"
                    )
                    try:
                        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=True), encoding="utf-8")
                        snapshot_written = True
                    except Exception:
                        log.warning("Failed to write memory snapshot for task %s", task.id, exc_info=True)
                        # A snapshot was available but could not be written, so
                        # the container will start cold (memory-less). Signal the
                        # caller so the attempt's runtime_meta records it instead
                        # of silently contaminating the gen-over-gen comparison.
                        snapshot_failed = True

            disable_memory_tools = bool(self.disable_memory_mcp)
            if swebench_container_images and source == "swebench_pro":
                # The official SWE-bench Pro images provide task deps, not the
                # ksi memory-server Python stack. Task memory is already
                # injected into MEMORY.md and the prompt, so avoid starting
                # MCP sidecars inside those images.
                disable_memory_tools = True

            payload["knowledge"] = {
                "db_path": str(knowledge_db),
                "mcp_server_dir": str(Path(__file__).parent.parent / "memory"),
                "disable_memory_tools": disable_memory_tools,
                "forum_generation": generation,
                "experiment_name": experiment_name,
            }
            if snapshot_written and snapshot_path is not None:
                payload["knowledge"]["snapshot_path"] = str(snapshot_path)
        if self.runtime_db_path:
            runtime_db = Path(self.runtime_db_path)
            if not runtime_db.is_absolute():
                runtime_db = (Path.cwd() / runtime_db).resolve()
            runtime_db.parent.mkdir(parents=True, exist_ok=True)
            runtime_db.touch(exist_ok=True)
            payload["runtime_audit"] = {"db_path": str(runtime_db)}
        return snapshot_path, snapshot_failed

    def _build_task_runner_env(
        self,
        *,
        source: str,
        arc_no_mcp_active: bool,
        swebench_container_images: dict[str, str],
        min_effective_timeout: int = 0,
    ) -> tuple[dict[str, str], int, int | None]:
        """Build the runner subprocess env plus its effective timeout and
        watchdog backstop.

        ``min_effective_timeout`` raises the floor on ``effective_timeout``
        (e.g. to fit a multi-round polyglot test-feedback retry loop) but
        never re-enables a timeout the operator explicitly disabled
        (``effective_timeout <= 0``).

        ``backstop`` is ``None`` when timeouts are disabled (``effective_timeout
        <= 0``), otherwise ``effective_timeout + 120``.
        """
        # Start from provider profile keys only; add minimal system env
        env = {**self.env}
        for sys_key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "USER", "SHELL"):
            if sys_key in os.environ:
                env.setdefault(sys_key, os.environ[sys_key])
        _validate_provider_auth(env)

        if source == "per_task_forum":
            effective_timeout = self.forum_timeout_sec
        elif source == "cross_task_forum":
            effective_timeout = self.cross_task_forum_timeout_sec
        else:
            effective_timeout = self.timeout_sec
        # Scale up for a multi-round polyglot test-feedback retry loop
        # (concurrency-ipc.md Finding 2) -- but never override an
        # explicitly-disabled timeout (effective_timeout <= 0 means
        # "unlimited"; a scaled minimum must not silently re-enable one).
        if effective_timeout > 0 and min_effective_timeout > effective_timeout:
            effective_timeout = min_effective_timeout
        runner_env = _build_runner_env(env, effective_timeout)
        if arc_no_mcp_active:
            # ARC runs the native (attempt-file) path via the Anthropic SDK
            # claude-code adapter. Read by
            # runtime_runner/agent-runner/src/index.ts.
            runner_env["KSI_ANTHROPIC_ARC_ADAPTER"] = "claude-code"
        if swebench_container_images:
            runner_env["KSI_CONTAINER_IMAGE"] = swebench_container_images["container_image"]
            runner_env["CONTAINER_IMAGE"] = swebench_container_images["container_image"]
            runner_env["KSI_TASK_REPO_CONTAINER_PATH"] = swebench_container_images["official_repo_container_path"]
            runner_env["KSI_RUNNER_ROOT"] = swebench_container_images["runner_root"]

        backstop = None if effective_timeout <= 0 else effective_timeout + 120
        return runner_env, effective_timeout, backstop

    def _prepare_run_context(
        self,
        *,
        generation: int,
        agent_id: str,
        task: TaskSpec,
        seed_package: Any,
        experiment_name: str,
    ) -> _RunTaskContext:
        """Resolve task/source metadata, derive ARC tool flags, build the
        runner payload, and emit the ``runtime.payload`` trace event.

        When the task is a swebench_pro task with a derivable official
        container image, ``task`` is rebuilt with the image metadata merged
        into ``metadata``. Returns the immutable :class:`_RunTaskContext`
        consumed by the temp-dir execution block in :meth:`run_task`.
        """
        trace_dir = get_trace_dir()
        if not self.command:
            raise RuntimeError("KsiContainerExecutor.command must be configured")

        metadata = task.metadata or {}
        source = str(metadata.get("task_source") or "").strip().lower()
        container_image_env = {**os.environ, **self.env}
        swebench_container_images = _swebench_pro_container_images(task, container_image_env)
        if swebench_container_images:
            metadata = {
                **metadata,
                "official_container_image": swebench_container_images["official_container_image"],
                "container_image": swebench_container_images["container_image"],
                "repo_container_path": swebench_container_images["repo_container_path"],
                "official_repo_container_path": swebench_container_images["official_repo_container_path"],
                "runner_root": swebench_container_images["runner_root"],
            }
            task = TaskSpec(id=task.id, repo=task.repo, prompt=task.prompt, metadata=metadata)
        # ARC is always native now (the legacy MCP toolset was removed):
        # ``arc_no_mcp_active`` is simply "is this an ARC task".
        _src_spec = resolve_source(source)
        arc_no_mcp_active = bool(_src_spec and _src_spec.supports_mcp_arc)
        payload, memory_md = self._build_runner_payload(
            task=task,
            generation=generation,
            agent_id=agent_id,
            metadata=metadata,
            source=source,
            swebench_container_images=swebench_container_images,
            arc_no_mcp_active=arc_no_mcp_active,
            experiment_name=experiment_name,
            seed_package=seed_package,
        )

        append_trace_event(
            trace_dir,
            "prompt_events.jsonl",
            {
                "event": "runtime.payload",
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task.id,
                "task_source": str(metadata.get("task_source") or ""),
                "execution_prompt": payload.get("execution_prompt", ""),
                "instruction_md": payload["workspace_seed"].get("instruction_md", ""),
                "memory_md": payload["workspace_seed"].get("memory_md", ""),
                "task_md": payload["workspace_seed"].get("task_md", ""),
                "tools_md": payload["workspace_seed"].get("tools_md", ""),
                "task_files": payload["workspace_seed"].get("task_files", {}),
                "runtime": payload.get("runtime", {}),
            },
        )

        return _RunTaskContext(
            trace_dir=trace_dir,
            task=task,
            metadata=metadata,
            source=source,
            swebench_container_images=swebench_container_images,
            arc_no_mcp_active=arc_no_mcp_active,
            seed_package=seed_package,
            experiment_name=experiment_name,
            payload=payload,
            memory_md=memory_md,
        )

    def _postprocess_runner_output(
        self,
        proc: Any,
        *,
        generation: int,
        agent_id: str,
        task: TaskSpec,
        metadata: dict[str, Any],
        memory_md: str,
        trace_dir: Path,
        phase1_eligible: bool,
        phase1_state: dict[str, Any],
        polyglot_tf_eligible: bool = False,
        polyglot_tf_state: dict[str, Any] | None = None,
        knowledge_snapshot_failed: bool = False,
    ) -> RuntimeResult:
        """Parse, validate, enrich, and finalize a completed runner subprocess.

        Raises on a non-zero exit, a stdout protocol error, or a terminal
        error-envelope; otherwise returns the :class:`RuntimeResult`. The
        ``runtime_meta`` carried on both the raises and the result preserves
        ``injected_memory_md`` and (when eligible) the phase-1 reflection
        evaluation cached by the BarrierWatcher.
        """
        if proc.returncode != 0:
            _raise_runner_failure(
                proc,
                trace_dir=trace_dir,
                generation=generation,
                agent_id=agent_id,
                task=task,
            )

        try:
            parsed = parse_runner_stdout(proc.stdout, key=self.output_json_key, strict=True)
        except ValueError as exc:
            append_trace_event(
                trace_dir,
                "runtime_events.jsonl",
                {
                    "event": "runtime.parse_error",
                    "generation": generation,
                    "agent_id": agent_id,
                    "task_id": task.id,
                    "stdout": (proc.stdout or "")[:2000],
                    "error": str(exc),
                },
            )
            raise RuntimeError(
                f"Container stdout protocol error for task {task.id} (agent={agent_id}, generation={generation})"
            ) from exc
        runtime_meta_parsed = dict(parsed.get("runtime_meta") or {})

        _validate_runner_protocol(
            runtime_meta_parsed,
            generation=generation,
            agent_id=agent_id,
            task=task,
            stdout=proc.stdout,
            trace_dir=trace_dir,
        )
        # Defense-in-depth: warn when the runner emitted zero tokens despite a
        # non-trivial tool trace. This indicates the per-turn / result-event
        # token extraction in the agent-runner missed both sources — typically
        # a streaming bug (e.g. SDK nests assistant usage under `.message.usage`
        # but the accumulator only inspected the top-level path). Keeping the
        # zero value in the DB preserves legacy schema, but the warning makes
        # the gap discoverable in logs.
        tokens_source = runtime_meta_parsed.get("tokens_source")
        tok_usage = parsed.get("token_usage") or TokenUsage()
        tool_trace_len = len(parsed.get("tool_trace") or [])
        if tok_usage.total == 0 and tool_trace_len >= 5 and runtime_meta_parsed.get("status") == "success":
            log.warning(
                "[container_host] zero-token report with non-trivial trace: "
                "generation=%d agent=%s task=%s trace_len=%d tokens_source=%s — "
                "likely a streaming reporting gap, real token consumption was non-zero",
                generation,
                agent_id,
                task.id,
                tool_trace_len,
                tokens_source or "absent",
            )
        # Thread the resolved memory_md through runtime_meta so downstream
        # consumers (analysis scripts, distillation) can reconstruct exactly
        # what memory surface the agent saw for this attempt.
        runtime_meta_parsed["injected_memory_md"] = memory_md
        # A memory snapshot was available but failed to write host-side, so this
        # attempt ran cold (memory-less). Stamp it so analysis can exclude the
        # attempt from gen-over-gen comparisons instead of silently treating a
        # cold-start as a like-for-like data point. Set only when
        # True to keep the common (healthy) runtime_meta unchanged.
        if knowledge_snapshot_failed:
            runtime_meta_parsed["_knowledge_snapshot_failed"] = True
        # Phase-1 reflection: when the BarrierWatcher's _on_barrier
        # callback already invoked ``evaluator.evaluate(...)``, surface
        # both the raw result and an enabled-flag so the engine's
        # ``_eval_stage`` can reuse it instead of paying for a second
        # docker subprocess (polyglot/swebench_pro). Engine reads
        # ``phase1_eval_result`` only when ``phase1_reflection_enabled``
        # is True; if the watcher errored, only the flag is set so the
        # engine falls through to its own evaluate() call.
        if phase1_eligible:
            runtime_meta_parsed["phase1_reflection_enabled"] = True
            watcher_obj = phase1_state.get("watcher")
            cached_holder = getattr(watcher_obj, "_cached_eval_holder", None) if watcher_obj is not None else None
            if isinstance(cached_holder, dict):
                cached_value = cached_holder.get("value")
                cached_error = cached_holder.get("error")
                if cached_value is not None:
                    runtime_meta_parsed["phase1_eval_result"] = cached_value
                if cached_error:
                    runtime_meta_parsed["phase1_eval_error"] = cached_error
        # Polyglot test-feedback: reuse the watcher's cached Docker eval
        # ONLY when the TS side reports the last barrier round's evaluation
        # genuinely reflects the final graded state (no agent turn ran
        # after it) — see _launch_polyglot_test_feedback_watcher's
        # docstring. Otherwise the cached value is stale (scored the
        # pre-turn state) and execution_phase.py must fall through to its
        # own evaluate() call.
        if polyglot_tf_eligible and polyglot_tf_state is not None:
            tf_meta = runtime_meta_parsed.get("polyglot_test_feedback_meta")
            watcher_obj = polyglot_tf_state.get("watcher")
            cached_holder = getattr(watcher_obj, "_cached_eval_holder", None) if watcher_obj is not None else None
            if isinstance(cached_holder, dict):
                cached_error = cached_holder.get("error")
                # Mirrors phase1_eval_error above: surface the watcher's
                # last error (if any) so a run where the retry loop's host
                # callback failed is diagnosable rather than indistinguishable
                # from a genuine test failure (errors-timeouts.md Finding 1).
                if cached_error:
                    runtime_meta_parsed["polyglot_test_feedback_eval_error"] = cached_error
                if (
                    isinstance(tf_meta, dict)
                    and tf_meta.get("final_eval_matches_output") is True
                    and cached_holder.get("value") is not None
                ):
                    runtime_meta_parsed["polyglot_test_feedback_reuse_eligible"] = True
                    runtime_meta_parsed["polyglot_test_feedback_eval_result"] = cached_holder["value"]
        parsed["runtime_meta"] = runtime_meta_parsed
        append_trace_event(
            trace_dir,
            "runtime_events.jsonl",
            {
                "event": "runtime.result",
                "generation": generation,
                "agent_id": agent_id,
                "task_id": task.id,
                "task_source": str(metadata.get("task_source") or ""),
                "tool_trace_count": tool_trace_len,
                "runtime_meta": runtime_meta_parsed,
                "token_usage": tok_usage.to_dict(),
                "tokens_source": tokens_source or "absent",
                "output_text": parsed.get("output") or "",
            },
        )
        # Terminal-status handling: silent-failure / error-envelope detection,
        # max-turn workspace salvage, or the normal RuntimeResult. See
        # :func:`_finalize_runner_result` for the full rationale (these raises
        # carry ``runtime_meta`` so the engine preserves ``native_session_memory``
        # rather than counting a 0-score success).
        return _finalize_runner_result(
            parsed,
            generation=generation,
            agent_id=agent_id,
            task=task,
            trace_dir=trace_dir,
        )

    def _setup_barrier(
        self,
        *,
        payload: dict[str, Any],
        payload_path: Path,
        payload_key: str,
        payload_block: dict[str, Any],
        runner_env: dict[str, str],
        td: str,
        prior_states: Sequence[dict[str, Any]],
        launch_watcher: Callable[[Path], "BarrierWatcher | None"],
    ) -> dict[str, Any]:
        """Shared setup shape for the barrier-watcher triad
        (phase1_reflection / cross_task_r1 / polyglot_test_feedback).

        Each feature does the identical dance: inject its ``payload_block``
        under ``payload_key``, re-write the payload JSON, resolve the single
        ``KSI_BARRIER_WORKSPACE_FILE`` (reusing a prior feature's file if one
        already claimed it -- ``main.ts`` writes the resolved workspace dir once
        and any number of watchers can share that path -- else create+register a
        new one), and launch a feature-specific ``BarrierWatcher``. Only the
        block contents (incl. each feature's DISTINCT container-side poll-timeout
        formula, computed by the caller) and the ``launch_watcher`` closure
        differ; this helper owns everything else. Returns
        ``{"watcher": ..., "workspace_file": ...}``.
        """
        state: dict[str, Any] = {"watcher": None, "workspace_file": None}
        payload[payload_key] = payload_block
        payload_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        reused = next(
            (s.get("workspace_file") for s in prior_states if s.get("workspace_file") is not None),
            None,
        )
        if reused is not None:
            state["workspace_file"] = reused
        else:
            workspace_file = Path(td) / f"workspace_path_{uuid.uuid4().hex[:8]}.txt"
            state["workspace_file"] = workspace_file
            runner_env["KSI_BARRIER_WORKSPACE_FILE"] = str(workspace_file)
        state["watcher"] = launch_watcher(state["workspace_file"])
        return state

    def _maybe_setup_phase1_reflection(
        self,
        *,
        source: str,
        payload: dict[str, Any],
        payload_path: Path,
        runner_env: dict[str, str],
        effective_timeout: int,
        td: str,
        agent_id: str,
        task: TaskSpec,
    ) -> tuple[bool, dict[str, Any]]:
        """Phase-1 reflection (Path a): when the feature flag is on AND
        an evaluator is wired AND we are NOT on a strict-protocol
        scheduled task (ARC / forum, which use their own dedicated
        adapters and don't go through the runQuery success branch
        that triggers the barrier), launch a BarrierWatcher thread
        that polls the workspace for the in-container sentinel,
        runs evaluator.evaluate(), and writes the response back so
        the agent can produce a structured 3-5 sentence reflection.

        Mutates ``payload`` and ``runner_env`` in place and re-writes
        ``payload_path`` when eligible. Returns ``(phase1_eligible,
        phase1_state)``.
        """
        forum_phases_for_phase1 = {"per_task_forum", "cross_task_forum"}
        phase1_eligible = (
            self.phase1_reflection_enabled
            and self.evaluator is not None
            and source not in forum_phases_for_phase1
            and source != "arc"
        )
        phase1_state: dict[str, Any] = {"watcher": None, "workspace_file": None}
        if phase1_eligible:
            # The payload block flags the feature into the container
            # (checked in agent-runner/src/index.ts); the DISTINCT
            # container-side poll timeout is _phase1_poll_timeout_ms.
            phase1_state = self._setup_barrier(
                payload=payload,
                payload_path=payload_path,
                payload_key="phase1_reflection",
                payload_block={
                    "enabled": True,
                    "eval_result_poll_timeout_ms": _phase1_poll_timeout_ms(effective_timeout),
                },
                runner_env=runner_env,
                td=td,
                prior_states=(),
                launch_watcher=lambda workspace_file: self._launch_phase1_watcher(
                    workspace_file=workspace_file,
                    agent_id=agent_id,
                    task=task,
                    poll_timeout_sec=max(60, effective_timeout + 90),
                ),
            )
        return phase1_eligible, phase1_state

    def _maybe_setup_cross_task_r1(
        self,
        *,
        source: str,
        cross_task_shared_container: bool,
        cross_task_r1_callback: Callable[..., Any] | None,
        payload: dict[str, Any],
        payload_path: Path,
        runner_env: dict[str, str],
        effective_timeout: int,
        td: str,
        agent_id: str,
        phase1_eligible: bool,
        phase1_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Cross-task R0->R1 shared-container handshake. Engine sets
        ``cross_task_r1_callback`` (and ``cross_task_shared_container=True``)
        for cross-task forum dispatches when the feature flag is on.
        The callback is invoked when this agent's R0 sentinel appears
        in the workspace; it must block until the host has drained
        all R0 sentinels (or the coordinator timed out) and computed
        the per-agent R1 prompt suffix, then return a dict that
        gets written to ``.barrier.cross_task_r1.<agent>.response``.

        Mutates ``payload`` and ``runner_env`` in place and re-writes
        ``payload_path`` when active. Returns ``cross_task_state``.
        """
        cross_task_shared = bool(cross_task_shared_container) and source == "cross_task_forum"
        cross_task_state: dict[str, Any] = {"watcher": None, "workspace_file": None}
        if cross_task_shared and cross_task_r1_callback is not None:
            # DISTINCT container-side poll timeout is _cross_task_timeouts (the
            # nested container/poll deadlines). The payload's
            # ``response_poll_timeout_ms`` is the poll deadline in ms. The
            # barrier reuses phase1's workspace file when phase1 was eligible
            # (its state is passed as the sole prior).
            _container_timeout_sec, poll_timeout_sec = _cross_task_timeouts(effective_timeout)
            cross_task_state = self._setup_barrier(
                payload=payload,
                payload_path=payload_path,
                payload_key="cross_task_shared_container",
                payload_block={
                    "enabled": True,
                    "response_poll_timeout_ms": poll_timeout_sec * 1000,
                    "barrier_name": "cross_task_r1",
                },
                runner_env=runner_env,
                td=td,
                prior_states=(phase1_state,),
                launch_watcher=lambda workspace_file: self._launch_cross_task_r1_watcher(
                    workspace_file=workspace_file,
                    agent_id=agent_id,
                    callback=cross_task_r1_callback,
                    poll_timeout_sec=max(60, effective_timeout + 90),
                ),
            )
        return cross_task_state

    def _execute_runner_with_fallback(
        self,
        *,
        payload_path: Path,
        runner_env: dict[str, str],
        backstop: int | None,
        effective_timeout: int,
        task: TaskSpec,
        phase1_state: dict[str, Any],
        cross_task_state: dict[str, Any],
        polyglot_tf_state: dict[str, Any],
        snapshot_path: Path | None,
    ) -> Any:
        """Run the container runner subprocess (with an npx/tsx fallback)
        and return the completed process.

        The per-task memory snapshot lives in the persistent knowledge-DB
        directory (not the TemporaryDirectory ``td``), so it must be
        unlinked on every exit path of the runner block below. Owning its
        cleanup in an outer ``finally`` that wraps BOTH the primary command
        and the npx fallback (a) guarantees it survives long enough for the
        fallback runner -- which re-reads the same ``payload.json``
        referencing ``snapshot_path`` -- and (b) prevents stale snapshots
        from accumulating when the fallback raises ``TimeoutExpired``.
        (A raise during the setup phase above -- before this block
        -- is a pre-existing narrow leak, out of scope here.)

        The barrier watchers' ``.stop()`` calls live in this SAME outer
        ``finally`` (not a ``finally`` around only the primary attempt) so
        they stay alive through the npx-tsx fallback too: if the primary
        invocation fails with "tsx: not found" and a fresh container starts
        via the fallback command below, it still writes barrier sentinels
        that need a live watcher to answer them. This matters most for the
        polyglot test-feedback watcher (``persistent=True``, expected to
        answer sentinels across the whole task's retry lifecycle), but
        applies equally to phase1/cross-task.
        """
        try:
            try:
                proc = self._run_runner_command(
                    [*self.command, str(payload_path)],
                    cwd=self.working_dir,
                    env=runner_env,
                    timeout=backstop,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Shared container runner timed out after {effective_timeout}s for task {task.id}")

            if proc.returncode != 0:
                combined = ((proc.stderr or "") + "\n" + (proc.stdout or "")).lower()
                if "tsx: not found" in combined or "tsx: command not found" in combined:
                    runtime_runner_dir = Path(self.working_dir) / "runtime_runner"
                    fallback_cmd = [
                        "npx",
                        "--yes",
                        "tsx@4.19.3",
                        str(runtime_runner_dir / "src" / "main.ts"),
                        str(payload_path),
                    ]
                    try:
                        proc = self._run_runner_command(
                            fallback_cmd,
                            cwd=self.working_dir,
                            env=runner_env,
                            timeout=backstop,
                        )
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(
                            f"Shared container runner timed out after {effective_timeout}s for task {task.id}"
                        )
        finally:
            if phase1_state["watcher"] is not None:
                try:
                    phase1_state["watcher"].stop()
                except Exception:
                    pass
            if cross_task_state["watcher"] is not None:
                try:
                    cross_task_state["watcher"].stop()
                except Exception:
                    pass
            if polyglot_tf_state["watcher"] is not None:
                watcher = polyglot_tf_state["watcher"]
                try:
                    watcher.stop()
                except Exception:
                    pass
                # Unlike phase1/cross-task's fire-and-forget stop() above,
                # the polyglot watcher's callback can be a genuine
                # Docker-bound evaluate() call still in flight when the
                # container gave up on a barrier round -- wait it out
                # (bounded) so execution_phase.py's own fallback
                # evaluate() call, which runs immediately after this
                # method returns, never races an orphaned watcher thread
                # scoring the same task concurrently (concurrency-ipc.md
                # Finding 1).
                join_timeout = getattr(watcher, "_stop_join_timeout_sec", 2.0) + 5.0
                try:
                    watcher.join(timeout=join_timeout)
                except Exception:
                    pass
                # The Docker-bound evaluate() runs on the INNER BarrierWatcher,
                # not on this outer _DeferredBarrierWatcher (whose run() returns
                # as soon as its own bounded inner.join() elapses, so
                # watcher.is_alive() here is effectively always False and would
                # never surface a still-running callback). Probe the inner
                # watcher, which is what actually holds the evaluate() call.
                inner_watcher = getattr(watcher, "_inner", None)
                if inner_watcher is not None and inner_watcher.is_alive():
                    log.warning(
                        "[container_host] polyglot_test_feedback watcher for task %s did not stop "
                        "within %.0fs; the host-side evaluator.evaluate() callback may still be "
                        "running in the background, possibly racing a fallback evaluate() call",
                        task.id,
                        join_timeout,
                    )
            if snapshot_path is not None:
                try:
                    snapshot_path.unlink(missing_ok=True)
                except Exception:
                    pass
        return proc

    def run_task(
        self,
        *,
        generation: int,
        agent_id: str,
        task: TaskSpec,
        cross_task_shared_container: bool = False,
        cross_task_r1_callback: Callable[..., Any] | None = None,
        **kwargs,
    ) -> RuntimeResult:
        # Deprecation: these kwargs moved to EngineEnrichmentPhaseService.enrich()
        _DEPRECATED_KWARGS = {"task_store", "forum_store", "best_scores"}
        _deprecated_found = _DEPRECATED_KWARGS & set(kwargs)
        if _deprecated_found:
            import warnings

            warnings.warn(
                f"run_task() kwargs {_deprecated_found} are deprecated; "
                "enrichment now happens in EngineEnrichmentPhaseService.enrich(). "
                "These kwargs will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
        ctx = self._prepare_run_context(
            generation=generation,
            agent_id=agent_id,
            task=task,
            seed_package=kwargs.get("agent_seed_package"),
            experiment_name=str(kwargs.get("experiment_name", "") or ""),
        )
        # ``_prepare_run_context`` may rebuild ``task`` with swebench image
        # metadata; use the resolved one for the rest of the attempt.
        task = ctx.task

        with tempfile.TemporaryDirectory(prefix="ksi-task-") as td, contextlib.ExitStack() as snapshot_guard:
            payload_path = Path(td) / "payload.json"
            snapshot_path, knowledge_snapshot_failed = self._materialize_payload_side_files(
                payload=ctx.payload,
                td=td,
                task=task,
                metadata=ctx.metadata,
                agent_id=agent_id,
                generation=generation,
                source=ctx.source,
                seed_package=ctx.seed_package,
                swebench_container_images=ctx.swebench_container_images,
                experiment_name=ctx.experiment_name,
            )
            if snapshot_path is not None:
                # The snapshot lives in the *persistent* knowledge-DB dir (not
                # ``td``), so it is not auto-reaped. Register the unlink now so a
                # raise anywhere in the setup phase between here and the runner
                # block (e.g. ``_validate_provider_auth`` on ``MODEL=""``, a
                # payload ``write_text`` error, or a watcher ``.start()``) still
                # cleans it up. The runner ``finally`` below unlinks
                # promptly on the normal path; this is the idempotent safety net.
                snapshot_guard.callback(_unlink_snapshot, snapshot_path)
            payload_path.write_text(json.dumps(ctx.payload, ensure_ascii=True), encoding="utf-8")

            # Build the polyglot test-feedback config FIRST (it has no
            # dependency on runner_env/effective_timeout -- see
            # _maybe_setup_polyglot_test_feedback's docstring) so its
            # ``triesRemaining`` can scale the session timeout below
            # (concurrency-ipc.md Finding 2) before it's baked into
            # runner_env's CONTAINER_TIMEOUT and the subprocess backstop.
            polyglot_tf_cfg = self._maybe_setup_polyglot_test_feedback(task=task, agent_id=agent_id)
            polyglot_min_timeout = 0
            if polyglot_tf_cfg is not None:
                polyglot_tries = int(polyglot_tf_cfg["triesRemaining"])
                polyglot_round_budget = (
                    int(
                        getattr(self.evaluator, "timeout_sec", DEFAULT_POLYGLOT_TIMEOUT_SEC)
                        or DEFAULT_POLYGLOT_TIMEOUT_SEC
                    )
                    + 60
                )
                polyglot_min_timeout = polyglot_tries * polyglot_round_budget + 300

            runner_env, effective_timeout, backstop = self._build_task_runner_env(
                source=ctx.source,
                arc_no_mcp_active=ctx.arc_no_mcp_active,
                swebench_container_images=ctx.swebench_container_images,
                min_effective_timeout=polyglot_min_timeout,
            )

            phase1_eligible, phase1_state = self._maybe_setup_phase1_reflection(
                source=ctx.source,
                payload=ctx.payload,
                payload_path=payload_path,
                runner_env=runner_env,
                effective_timeout=effective_timeout,
                td=td,
                agent_id=agent_id,
                task=task,
            )

            cross_task_state = self._maybe_setup_cross_task_r1(
                source=ctx.source,
                cross_task_shared_container=cross_task_shared_container,
                cross_task_r1_callback=cross_task_r1_callback,
                payload=ctx.payload,
                payload_path=payload_path,
                runner_env=runner_env,
                effective_timeout=effective_timeout,
                td=td,
                agent_id=agent_id,
                phase1_eligible=phase1_eligible,
                phase1_state=phase1_state,
            )

            polyglot_tf_state = self._setup_polyglot_test_feedback_watcher(
                polyglot_tf_cfg=polyglot_tf_cfg,
                task=task,
                agent_id=agent_id,
                payload=ctx.payload,
                payload_path=payload_path,
                runner_env=runner_env,
                effective_timeout=effective_timeout,
                td=td,
                phase1_state=phase1_state,
                cross_task_state=cross_task_state,
            )

            proc = self._execute_runner_with_fallback(
                payload_path=payload_path,
                runner_env=runner_env,
                backstop=backstop,
                effective_timeout=effective_timeout,
                task=task,
                phase1_state=phase1_state,
                cross_task_state=cross_task_state,
                polyglot_tf_state=polyglot_tf_state,
                snapshot_path=snapshot_path,
            )

        return self._postprocess_runner_output(
            proc,
            generation=generation,
            agent_id=agent_id,
            task=task,
            metadata=ctx.metadata,
            memory_md=ctx.memory_md,
            trace_dir=ctx.trace_dir,
            phase1_eligible=phase1_eligible,
            phase1_state=phase1_state,
            polyglot_tf_eligible=polyglot_tf_cfg is not None,
            polyglot_tf_state=polyglot_tf_state,
            knowledge_snapshot_failed=knowledge_snapshot_failed,
        )
