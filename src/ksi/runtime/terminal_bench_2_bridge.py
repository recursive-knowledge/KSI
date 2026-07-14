from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import TaskSpec
from .terminal_bench_2_docker import _shorten


def _tb2_bridge_system_prompt() -> str:
    return (
        "You are the KSI Terminal-Bench 2 bridge agent.\n"
        "All task work must happen via tool actions targeting the live TB2 container.\n"
        "Respond with exactly one JSON object and no surrounding prose.\n"
        "Available actions (one per reply):\n"
        '{"action":"shell","command":"<bash>","timeout_sec":60,"summary":"..."}\n'
        '{"action":"read","path":"<abs path>","offset":1,"limit":2000,"summary":"..."}\n'
        '{"action":"write","path":"<abs path>","content":"<full file>","summary":"..."}\n'
        '{"action":"edit","path":"<abs path>","old_string":"...","new_string":"...","replace_all":false,"summary":"..."}\n'
        '{"action":"glob","pattern":"*.py","path":"<abs dir>","summary":"..."}\n'
        '{"action":"grep","pattern":"<regex>","path":"<abs path>","output_mode":"content","summary":"..."}\n'
        '{"action":"final","summary":"what you changed, what remains, and why you are stopping"}\n'
        "Tool semantics (mirror Claude Code):\n"
        "- read: returns lines [offset, offset+limit) of <path>. Defaults: offset=1, limit=2000. Up to 200KB returned.\n"
        "- write: overwrites <path> with <content>. Use for creating new files or full rewrites.\n"
        "- edit: exact string replacement. Fails if old_string is not unique unless replace_all=true.\n"
        "- glob: lists files under <path> whose basename matches <pattern> (e.g. '*.py').\n"
        "- grep: greps for <pattern> recursively under <path>. output_mode='files_with_matches' lists files; 'content' shows matches with line numbers.\n"
        "- shell: arbitrary bash; use for compilation, services, package installs, anything dynamic.\n"
        "Rules:\n"
        "- Read /workspace/task/workspace/tb2/instruction.md first.\n"
        "- Read /workspace/task/workspace/MEMORY.md before repeating failed ideas when it exists.\n"
        "- Read /workspace/task/workspace/tb2/task.toml for metadata and constraints.\n"
        "- Treat /workspace/task/workspace as the task specification overlay, not the main place to do the work.\n"
        "- After reading the spec files, identify the real task surface quickly using glob/grep/shell ls/find.\n"
        "- Prefer small, verifiable steps over large speculative edits.\n"
        "- Prefer short mutate-then-check cycles: make one concrete change, then run one verifier-aligned check.\n"
        "- Use read/edit for surgical file changes; use write for new files; use shell for execution and side effects.\n"
        "- Avoid repeated overlay-only inspection once you know the task requirements.\n"
        "- If MEMORY.md names a concrete failure, missing artifact, file path, service, or command, target that exact clue before broad exploration.\n"
        "- Avoid giant here-doc scripts until you have validated the target path, binary, or service with shorter commands.\n"
        "- If a quick manual check works only from the current directory, shell state, or one-off environment variable, make that fix persistent so a fresh shell and the verifier will also pass.\n"
        "- If the task depends on a service, daemon, port, or deployed artifact, verify it from a fresh command after reload/restart instead of assuming the change stuck.\n"
        "- Before stopping, run a concrete verification command for the task's main artifact, service, build, or output.\n"
        "- Do not stop just because a command exited 0; verify the task-specific outcome explicitly.\n"
        "- Do not rely on /solution or hidden verifier assets.\n"
        "- Do not emit XML, <function_calls>, <invoke>, markdown fences, or multiple actions in one reply.\n"
        "- Stop with action=final when another step would be redundant.\n"
    )


def _tb2_format_history_step(idx: int, entry: dict[str, Any]) -> str:
    """Format one committed shell-history step. The text is deterministic in
    ``idx`` + ``entry`` so a given step renders byte-identically on every turn
    — the property that makes the append-only history a cache-stable prefix."""
    tool_input = entry.get("tool_input", {}) if isinstance(entry.get("tool_input"), dict) else {}
    tool_output = entry.get("tool_output", {}) if isinstance(entry.get("tool_output"), dict) else {}
    tool_name = str(entry.get("tool_name") or "tb2_shell")
    kind = tool_name.removeprefix("tb2_") or "shell"
    lines = [f"Step {idx} ({kind}):"]
    if kind == "shell":
        lines.append(f"Command: {tool_input.get('command', '')}")
    elif kind == "read":
        lines.append(
            f"Read: {tool_input.get('path', '')} "
            f"(offset={tool_input.get('offset', '')}, limit={tool_input.get('limit', '')})"
        )
    elif kind == "write":
        lines.append(f"Write: {tool_input.get('path', '')} ({tool_input.get('content_bytes', 0)} bytes)")
    elif kind == "edit":
        lines.append(
            f"Edit: {tool_input.get('path', '')} "
            f"(replace_all={tool_input.get('replace_all', False)}, "
            f"old_len={tool_input.get('old_string_len', '')}, "
            f"new_len={tool_input.get('new_string_len', '')})"
        )
    elif kind == "glob":
        lines.append(f"Glob: pattern={tool_input.get('pattern', '')} path={tool_input.get('path', '')}")
    elif kind == "grep":
        lines.append(
            f"Grep: pattern={tool_input.get('pattern', '')} path={tool_input.get('path', '')} "
            f"output_mode={tool_input.get('output_mode', '')}"
        )
    else:
        lines.append(f"Input: {json.dumps(tool_input, default=str)[:200]}")
    lines.append(f"Exit: {tool_output.get('exit_code', '')}")
    lines.append(f"Output:\n{_shorten(str(tool_output.get('combined_output', '')), 2400)}")
    # Join with a blank line (and end with one) so that when the blocks are
    # concatenated by the LLM API — which inserts NO separator between adjacent
    # text blocks — the rendering is byte-identical to the pre-caching single
    # string (``"\n\n".join`` over the flat line list). Without the trailing
    # blank line the last line of each step would run into the next step's
    # header ("...file_bStep 2 (shell):"), a silent prompt-quality regression.
    return "\n\n".join(lines) + "\n\n"


def _tb2_bridge_stable_header(
    *,
    task: TaskSpec,
    generation: int,
    container_name: str,
    workspace_root: Path,
    execution_prompt: str,
) -> str:
    """The turn-INVARIANT header (task metadata + execution guidance + planning
    reminders). Kept free of any per-turn field so it is byte-stable across the
    whole trial and can lead the cached prefix."""
    return (
        f"Task id: {task.id}\n"
        f"Generation: {generation}\n"
        f"Target container: {container_name}\n"
        f"Mounted workspace root on host: {workspace_root}\n"
        "Workspace root inside container: /workspace/task/workspace\n"
        "Important: this mounted workspace is the task-spec overlay. The real task files, services, and repos may live elsewhere in the container.\n\n"
        f"KSI execution guidance:\n{execution_prompt.strip()}\n\n"
        "Planning reminder: if the recent history is mostly read-only inspection, your next command should mutate state or run a verifier-aligned check.\n"
        "Planning reminder: if the verifier clue or MEMORY names a concrete path, module, service, port, or command, act on that exact clue next.\n"
        "Planning reminder: if a manual import/build/request passed only from your current shell or working directory, your next command should make that behavior persist for a fresh shell and the verifier.\n\n"
        # Ends with a newline so the first step block (or the empty-history
        # sentinel) starts on its own line once the API concatenates the blocks
        # with no separator.
        "Recent shell history:\n"
    )


def _tb2_bridge_cache_blocks(
    *,
    task: TaskSpec,
    generation: int,
    max_steps: int,
    container_name: str,
    workspace_root: Path,
    execution_prompt: str,
    history: list[dict[str, Any]],
) -> list[str]:
    """Build the APPEND-ONLY stable blocks for the TB2 user message: a header
    block followed by one block per committed history step. Passed to the LLM
    caller's ``cache_blocks`` so the accumulated prefix is cache-READ each turn.
    With TB2's default unlimited ``max_steps`` the window
    never slides, so the block list only ever grows — every earlier block is
    byte-identical to the prior turn's."""
    header = _tb2_bridge_stable_header(
        task=task,
        generation=generation,
        container_name=container_name,
        workspace_root=workspace_root,
        execution_prompt=execution_prompt,
    )
    history_window = min(len(history), max_steps)
    history_start = max(1, len(history) - history_window + 1)
    steps = [
        _tb2_format_history_step(idx, entry) for idx, entry in enumerate(history[-history_window:], start=history_start)
    ]
    if not steps:
        return [f"{header}(no prior shell steps yet)\n\n"]
    return [header, *steps]


def _tb2_trim_oldest_history(history: list[dict[str, Any]]) -> int:
    """Drop the oldest ~30% of history steps in place (at least one) so a
    prompt that overflowed the provider context window shrinks before the
    next retry. Trades the append-only cache prefix for that turn (a cache
    miss) against a hard trial failure. Returns the number of steps dropped."""
    drop = max(1, len(history) // 3)
    del history[:drop]
    return drop


def _tb2_bridge_tail(*, step_index: int, max_steps: int, last_observation: str) -> str:
    """The per-turn VARYING tail (step counter + latest observation + the
    call-to-action). Kept after the cached blocks so it never invalidates the
    cached prefix. Note: the step counter moved here from the header, the one
    intentional reordering of the pre-caching prompt."""
    return (
        f"Bridge step: {step_index}/{max_steps}\n"
        f"Latest observation:\n{_shorten(last_observation, 3000)}\n\n"
        "Return the next JSON action now."
    )


def _build_tb2_bridge_transcript(
    *,
    task: TaskSpec,
    history: list[dict[str, Any]],
    final_output: str,
    error_text: str,
) -> str:
    lines = ["# tb2_bridge_transcript", f"task_id: {task.id}"]
    for idx, entry in enumerate(history, start=1):
        tool_input = entry.get("tool_input", {}) if isinstance(entry.get("tool_input"), dict) else {}
        tool_output = entry.get("tool_output", {}) if isinstance(entry.get("tool_output"), dict) else {}
        tool_name = str(entry.get("tool_name") or "tb2_shell")
        kind = tool_name.removeprefix("tb2_") or "shell"
        lines.append(f"step {idx}: {kind}")
        if kind == "shell":
            lines.append(f"command: {tool_input.get('command', '')}")
        elif kind in {"read", "write", "edit"}:
            descriptor = f"path: {tool_input.get('path', '')}"
            if kind == "read":
                descriptor += f" offset={tool_input.get('offset', '')} limit={tool_input.get('limit', '')}"
            elif kind == "write":
                descriptor += f" content_bytes={tool_input.get('content_bytes', '')}"
            elif kind == "edit":
                descriptor += (
                    f" old_len={tool_input.get('old_string_len', '')}"
                    f" new_len={tool_input.get('new_string_len', '')}"
                    f" replace_all={tool_input.get('replace_all', False)}"
                )
            lines.append(descriptor)
        elif kind in {"glob", "grep"}:
            lines.append(f"pattern: {tool_input.get('pattern', '')} path: {tool_input.get('path', '')}")
        else:
            lines.append(f"input: {json.dumps(tool_input, default=str)[:400]}")
        lines.append(f"summary: {tool_input.get('summary', '')}")
        lines.append(f"exit_code: {tool_output.get('exit_code', '')}")
        combined = str(tool_output.get("combined_output", "") or "").strip()
        if combined:
            lines.append("output:")
            lines.append(_shorten(combined, 4000))
    if final_output:
        lines.append("final_output:")
        lines.append(final_output.strip())
    if error_text:
        lines.append("error:")
        lines.append(error_text.strip())
    return "\n".join(lines).strip()
