"""Prompt builders for task execution."""

from __future__ import annotations

import ast
import json
import logging
from typing import Any

from ..models import TaskSpec
from ..tasks.registry import resolve_source

log = logging.getLogger(__name__)

# Prompt kinds with a dedicated build branch below. A registered source whose
# ``prompt_kind`` is not in this set falls through to the generic prompt — which
# is a silent mis-configuration for a benchmark-specific source, so warn once.
_HANDLED_PROMPT_KINDS = frozenset({"generic", "swebench_pro", "arc", "polyglot", "terminal_bench_2"})
_warned_prompt_kinds: set[str] = set()


def _warn_unhandled_prompt_kind(kind: str, *, spec_present: bool) -> None:
    """Warn once if a registered source declared a prompt_kind with no branch."""
    if not spec_present or kind in _HANDLED_PROMPT_KINDS or kind in _warned_prompt_kinds:
        return
    _warned_prompt_kinds.add(kind)
    log.warning(
        "[PROMPTS] task source declares prompt_kind=%r which has no dedicated "
        "prompt branch; falling back to the generic prompt. Add a branch in "
        "kcsi.prompts (build_execution_prompt / build_task_markdown) or set "
        "prompt_kind='generic' to silence this.",
        kind,
    )


# Pre-0610767d inline-grid body. Rendered at call time via .format(...).
_ARC_TASK_BODY_TEMPLATE = """\
Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
----------------------------------------
{training_data}
----------------------------------------
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data:
----------------------------------------
{test_data}
----------------------------------------
What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information.
"""

# ---------------------------------------------------------------------------
# Execution prompts — use {memory_step} placeholder; step numbers after
# the placeholder are filled dynamically so numbering stays correct
# regardless of whether the memory block is present.
# ---------------------------------------------------------------------------


def _memory_block(*, has_memory: bool) -> str:
    if has_memory:
        # Parity: a neutral knowledge pointer only. The prior
        # strategy-diversification exhortations ("avoid repeating the same
        # approach", "focus on a fundamentally different strategy") were
        # memory-arm-only and never seen by the control arm, confounding the
        # measured memory/KT effect. Removed rather than duplicated into the
        # control branch so no strategy advice is injected into either arm.
        return (
            "**Review prior attempts (pre-loaded in MEMORY.md):**\n"
            "- Prior attempt summaries, scores, and insights are already in your MEMORY.md.\n"
        )
    return "No prior memory is provided for this run.\n"


def _numbered_steps(steps: list[str]) -> str:
    return "\n".join(f"{idx}. {step}" for idx, step in enumerate(steps, start=1))


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        text = str(value).strip()
        return [text] if text else []

    text = value.strip()
    if not text:
        return []

    parsed: Any = None
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except Exception:
                continue
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


# Shared self-verification norm, default-on across the hidden-grader task
# sources: confirm the solution meets the spec (signatures, output format,
# structural constraints) and exercise it against the examples/edge cases in the
# statement before submitting. The load-bearing integrity clause below is shared
# verbatim by swebench, polyglot, and generic so the norm cannot be read as
# licence to fabricate the hidden tests.
_SELF_VERIFY_HIDDEN_TESTS_CLAUSE = (
    "Passing your own checks is necessary but not sufficient; the grader's tests "
    "are hidden — never create, modify, or guess them."
)


def _build_swebench_execution_prompt(*, has_memory: bool, generation: int) -> str:
    steps = [
        "Read `TASK.md` in the active task workspace.",
        _memory_block(has_memory=has_memory).strip(),
        "Write a short checklist of the required behavior changes.",
        "Read the verification section in `TASK.md` and inspect the exact failing tests or suites before editing.",
        "For each named failing target, identify the exact assertion or behavior it is checking before you change code.",
        "Find the code path that must change and the interfaces it affects.",
        "Stop broad searching once the likely fix path is clear.",
        "Make the smallest change that satisfies the full checklist.",
        "Use the regression targets and selected tests from `TASK.md` when available instead of ad hoc checks.",
        "When `TASK.md` provides a benchmark test-script command, run that command after each edit and iterate on its first failure.",
        "If `TASK.md` lists named failing tests without a runnable command, open those tests and align the implementation to them directly.",
        "If a targeted check fails, use the first failing assertion or error to refine the exact behavior in your checklist before editing again.",
        "Edit files in place using the workspace tools. The runtime captures the workspace diff as the canonical patch.",
        (
            "**Self-verify before finishing (required):** reproduce the described failing behavior with your "
            "own check and confirm your change flips it to passing, then verify the workspace diff covers "
            "every checklist item. Keep scratch tests or repro scripts out of the captured diff (the runtime "
            "captures the workspace diff as the canonical patch) — remove them before finishing. "
            f"{_SELF_VERIFY_HIDDEN_TESTS_CLAUSE}"
        ),
    ]
    return (
        f"You are a software engineer fixing a bug in a code repository. "
        f"This is generation {generation}.\n\n"
        "**Steps:**\n"
        f"{_numbered_steps(steps)}\n"
    )


# Generation-independent reference block for the native (attempt-file) ARC
# execution prompt: workspace paths, file inventory, output format, constraints,
# and tools. The shared intro line and the numbered **Steps:** list are assembled
# per call in ``_build_arc_no_mcp_execution_prompt`` so the ARC prompt uses the
# same overall shell as the other task sources while keeping its ARC-specific
# submission mechanics.
_ARC_NO_MCP_REFERENCE = """\
Workspace: /workspace/task/workspace

Public task files (already written for you):
- payload.json — canonical JSON with `train` pairs and `test` (list of {{input}}).
- grid_summary.md — readable rendering of the same grids in plain ASCII.
- TASK.md — the same grids inlined in markdown.
- attempt_1.txt and attempt_2.txt — placeholder files containing only a
  sentinel string; you must overwrite them with real ASCII grids to submit.
  Files left at the sentinel are scored as a non-submission (0).
- The hidden test output is NOT in the workspace.

Goal: infer the train-input -> train-output transformation, apply it to the
test input, and write your answer as plain ASCII grids — NOT JSON.

Output format (overwrite both files):
  /workspace/task/workspace/attempt_1.txt
  /workspace/task/workspace/attempt_2.txt

Each file must contain just rows of space-separated integers (0-9), one row
per line, exactly the same format as the `Output:` blocks in `grid_summary.md`.
Example file content for a 3x3 grid:
  1 0 0
  0 1 0
  0 0 1

Constraints:
- Integer colors 0-9 only. Rectangular grid, side <= 30.
- Two attempts (two separate files). If you have only one credible answer,
  write the same grid into both files.
- Do NOT write JSON — write plain space-separated digits.
- Don't modify payload.json, grid_summary.md, or TASK.md.

Tools: you have Read, Edit, Write, Bash, Glob, Grep. Prefer Read for
inspection. Use Bash for short commands, computing transformations, and
overwriting the attempt files (e.g. `cat > attempt_1.txt <<EOF ... EOF`).
Don't `cat` full grids into the conversation — they're already inlined
in TASK.md."""


def _build_arc_no_mcp_execution_prompt(*, has_memory: bool, generation: int, test_count: int = 1) -> str:
    """ARC execution prompt for the native (attempt-file) ARC path, the sole ARC mode.

    ``has_memory`` is accepted to keep the signature stable for future
    seed-context injection; it does not currently change the body. ``generation``
    is rendered into the shared intro line so the ARC prompt matches the format
    of the other task sources.

    ``test_count`` appends a short multi-test paragraph for tasks with more
    than one test input, instructing per-test ``attempt_<k>_<t>.txt`` files.
    """
    del has_memory  # reserved for future variants
    steps = [
        "Read each training pair from the inlined grids (or `payload.json`). Note dimensions, background color, objects, positions, and what changed input->output.",
        "Write a first-pass guess to `attempt_1.txt` and `attempt_2.txt` EARLY (within your first few turns) — even a copy of the test input is fine as a placeholder. The workspace must always contain a submission; analysing forever without committing scores 0.",
        "Form one transformation rule that explains every training pair.",
        "Verify that rule against every training pair before applying it.",
        "Apply the rule to the test input and OVERWRITE `attempt_1.txt` / `attempt_2.txt` with your refined answer (rows of space-separated digits 0-9, one row per line — same format as `grid_summary.md`).",
        "Sanity-check: dimensions, palette, object counts, symmetry, consistency.",
        "Validate the format before exit: `python3 /workspace/task/workspace/validate_prediction.py /workspace/task/workspace/attempt_1.txt /workspace/task/workspace/attempt_2.txt`.",
        "Write a one-sentence confirmation and stop.",
    ]
    prompt = (
        f"You are solving one ARC visual reasoning task. "
        f"This is generation {generation}.\n\n"
        f"{_ARC_NO_MCP_REFERENCE}\n\n"
        "**Steps:**\n"
        f"{_numbered_steps(steps)}\n"
    )
    if test_count > 1:
        prompt += (
            f"\nMulti-test task ({test_count} test inputs): write a separate "
            "answer for EACH test input using the per-test files. For each "
            f"test i in 0..{test_count - 1}, write `attempt_i_1.txt` and "
            "`attempt_i_2.txt` (e.g. test 0 -> attempt_0_1.txt / "
            "attempt_0_2.txt; test 1 -> attempt_1_1.txt / attempt_1_2.txt). "
            "For multi-test tasks ONLY these per-test `attempt_<i>_<trial>.txt` "
            "files are scored — including test 0, which MUST go in "
            "attempt_0_1.txt / attempt_0_2.txt. Do NOT use the legacy "
            "attempt_1.txt / attempt_2.txt files: they are ignored once any "
            "per-test file exists (and the per-test sentinels always exist on "
            "multi-test tasks), so a test left without its per-test file scores "
            "0. See TASK.md for the full per-test workflow.\n"
        )
    return prompt


def _build_polyglot_execution_prompt(*, has_memory: bool = False, generation: int = 1) -> str:
    steps = [
        "Read `TASK.md` in the active task workspace.",
        _memory_block(has_memory=has_memory).strip(),
        "Read the problem carefully before coding.",
        "Do not expect hidden benchmark tests to be present in the workspace; use the problem statement, starter code, and any examples in `TASK.md` as the specification.",
        "Implement all required functions/types as described.",
        "Handle edge cases mentioned in the problem statement.",
        (
            "**Self-verify before finishing (required):** write and run a small smoke check exercising the "
            "required signatures, output format, and edge cases from the problem statement, then run the "
            "verification command from `TASK.md` whenever it is available. "
            f"{_SELF_VERIFY_HIDDEN_TESTS_CLAUSE}"
        ),
        "If that command reports missing hidden test files or zero collected tests, treat it as an expected hidden-test signal; do not create, stub, or guess those test files.",
        "If the command produces a real compile/type/signature failure in your solution code, fix that exact contract mismatch before editing again.",
        "Do not modify or include test files.",
        "Return the answer in the exact format specified in TASK.md.",
    ]
    return (
        f"You are a skilled programmer solving a coding exercise. "
        f"This is generation {generation}.\n\n"
        "**Steps:**\n"
        f"{_numbered_steps(steps)}\n"
    )


def _build_terminal_bench_2_execution_prompt(*, has_memory: bool = False, generation: int = 1) -> str:
    steps = [
        "Read the native `tb2/instruction.md` first. It is the authoritative task statement.",
        (
            # Parity: a neutral MEMORY.md pointer only. The prior
            # "actively reuse it before planning or repeating exploration"
            # exhortation was memory-arm-only and never seen by the control arm,
            # confounding the measured memory/KT effect.
            "Read `MEMORY.md` next." if has_memory else "No prior memory is provided for this run."
        ),
        "Read `tb2/task.toml` for task metadata, timeouts, and environment hints.",
        "Treat the mounted KCSI workspace as task specification only; the real work happens in the native task container filesystem.",
        "After reading the spec files, identify the real task surface quickly: current directory, repo/app paths, services, ports, config files, and build/runtime entrypoints.",
        "Work directly in the current task container and filesystem; do not assume a synthetic benchmark repo layout.",
        "Do not spend many steps re-reading the overlay once you know the task requirements. Move into state-changing work and concrete validation.",
        "Prefer short mutate-then-check cycles over long speculative shell blocks. Use a small edit or setup step, then run a verifier-aligned check immediately.",
        "If `MEMORY.md` names a concrete failure, command, path, service, port, or missing artifact, attack that exact clue before trying a broad rewrite.",
        "Avoid giant here-doc scripts until you have already validated the target path, service, or command shape with shorter shell commands.",
        "If a manual import, build, request, or binary check works only from the current directory, shell state, or one-off environment variable, make that fix persistent so a fresh shell and the verifier also pass.",
        "If the task depends on a service, daemon, port, or deployed artifact, verify it from a fresh command after reload/restart instead of assuming the change stuck.",
        "Do not rely on `solution/` or hidden verifier assets to decide what to implement.",
        "Make the smallest set of changes needed to satisfy the native task instruction.",
        "If you run checks manually, treat the benchmark verifier as authoritative over ad hoc diagnostics.",
        (
            "**Self-verify before finishing (required):** exercise the concrete artifact, service, build "
            "output, or runtime behavior that the task requires. Passing your own checks is necessary but "
            "not sufficient — the benchmark verifier is authoritative."
        ),
        "Leave a concise final summary of what you changed and any remaining uncertainty.",
    ]
    return (
        f"You are solving a Terminal-Bench 2 task. This is generation {generation}.\n\n"
        "**Steps:**\n"
        f"{_numbered_steps(steps)}\n"
    )


def _build_generic_execution_prompt(*, has_memory: bool, generation: int) -> str:
    steps = [
        "Read `TASK.md` in the active task workspace.",
        _memory_block(has_memory=has_memory).strip(),
        "Solve the task using the files in the workspace.",
        (
            "**Self-verify before finishing (required):** confirm your solution meets the spec — required "
            "signatures, types, and output format/structural constraints — and exercise it with your own "
            "checks against the examples and edge cases in the task statement. Fix any mismatch before "
            f"submitting. {_SELF_VERIFY_HIDDEN_TESTS_CLAUSE}"
        ),
        "Provide a concise final answer in the exact format required by `TASK.md`.",
    ]
    return (
        f"You are solving a task in the current workspace. "
        f"This is generation {generation}.\n\n"
        "**Steps:**\n"
        f"{_numbered_steps(steps)}\n"
    )


def build_execution_prompt(
    task: TaskSpec,
    *,
    has_memory: bool = False,
    generation: int = 1,
) -> str:
    metadata = task.metadata or {}
    spec = resolve_source(metadata.get("task_source"))
    # Spec-attached builder wins; the hardcoded ``prompt_kind`` chain below is
    # the fallback for built-in sources that do not supply one.
    if spec is not None and spec.execution_prompt_builder is not None:
        return spec.execution_prompt_builder(task, has_memory=has_memory, generation=generation)
    kind = spec.prompt_kind if spec is not None else "generic"
    _warn_unhandled_prompt_kind(kind, spec_present=spec is not None)
    if kind == "swebench_pro":
        return _build_swebench_execution_prompt(has_memory=has_memory, generation=generation)
    if kind == "arc":
        return _build_arc_no_mcp_execution_prompt(
            has_memory=has_memory,
            generation=generation,
            test_count=_arc_test_count(task),
        )
    if kind == "polyglot":
        return _build_polyglot_execution_prompt(has_memory=has_memory, generation=generation)
    if kind == "terminal_bench_2":
        return _build_terminal_bench_2_execution_prompt(has_memory=has_memory, generation=generation)
    return _build_generic_execution_prompt(has_memory=has_memory, generation=generation)


def _json_compact(value: Any) -> str:
    # Compact separators (no whitespace) keep the inline ARC grids — which live
    # in the cached TASK.md prefix and are re-paid on every cache-read turn —
    # as small as possible. Pretty-printing (indent=2) was ~78% whitespace
    # The grids remain valid JSON and machine-parseable
    # via payload.json.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _arc_test_count(task: TaskSpec) -> int:
    """Count the ARC test inputs for a task, mirroring the derivation in
    :func:`_build_arc_no_mcp_task_markdown` (arc_test_inputs with a legacy
    arc_test_pairs fallback)."""
    metadata = task.metadata or {}
    test_inputs = metadata.get("arc_test_inputs") or []
    if not isinstance(test_inputs, list) or not test_inputs:
        legacy_test_pairs = metadata.get("arc_test_pairs") or []
        if isinstance(legacy_test_pairs, list):
            test_inputs = [{"input": p.get("input")} for p in legacy_test_pairs if isinstance(p, dict)]
    return len([p for p in test_inputs if isinstance(p, dict)])


def _build_arc_no_mcp_task_markdown(task: TaskSpec) -> str:
    """ARC TASK.md variant for the native (attempt-file) ARC path, the sole ARC mode.

    Inlines the full train/test grids so the cached prefix stays warm, and
    the workflow section points at ASCII per-attempt file submission
    (attempt_1.txt + attempt_2.txt + validate_prediction.py) rather than
    JSON output. Haiku 4.5 is unreliable at JSON
    output for nested grid structures, so we use the same plain ASCII
    format the agent reads for input.
    """
    metadata = task.metadata or {}
    train_pairs = metadata.get("arc_train_pairs") or []
    test_inputs = metadata.get("arc_test_inputs") or []
    if not isinstance(test_inputs, list) or not test_inputs:
        legacy_test_pairs = metadata.get("arc_test_pairs") or []
        if isinstance(legacy_test_pairs, list):
            test_inputs = [{"input": p.get("input")} for p in legacy_test_pairs if isinstance(p, dict)]
    split = metadata.get("arc_split") or "unknown"

    train_count = len([p for p in train_pairs if isinstance(p, dict)])
    test_count = len([p for p in test_inputs if isinstance(p, dict)])

    test_prompt_pairs = [{"input": pair.get("input"), "output": [[]]} for pair in test_inputs if isinstance(pair, dict)]

    body = _ARC_TASK_BODY_TEMPLATE.format(
        training_data=_json_compact(train_pairs),
        test_data=_json_compact(test_prompt_pairs),
    ).strip()

    if test_count > 1:
        workflow = (
            "## Workflow (no-MCP mode, ASCII output)\n"
            f"This task has {test_count} test inputs. You must produce a "
            "separate answer for EACH one.\n"
            "1. Read the inlined grids above (or `payload.json` for "
            "machine-readable form). `payload.json`'s `test` list has "
            f"{test_count} entries, indexed 0..{test_count - 1}.\n"
            f"2. For EACH test i in 0..{test_count - 1}, write your answer to\n"
            "   `attempt_i_1.txt` and `attempt_i_2.txt` (two trials per test).\n"
            f"   e.g. test 0 -> `attempt_0_1.txt` / `attempt_0_2.txt`; test 1 ->\n"
            "   `attempt_1_1.txt` / `attempt_1_2.txt`; and so on.\n"
            "3. EARLY (within your first few turns) overwrite every per-test\n"
            "   file with a first-pass guess — even copying that test's input is\n"
            "   fine as a placeholder. The workspace must always contain a\n"
            "   submission for each test; analysing forever without committing\n"
            "   scores 0.\n"
            "4. Refine your transformation rule against every train pair, then\n"
            "   OVERWRITE each `attempt_i_*.txt` with the refined answer for\n"
            "   test i (rows of space-separated digits 0-9, one row per line —\n"
            "   same format as `grid_summary.md`).\n"
            "5. Run `python3 validate_prediction.py attempt_0_1.txt "
            "attempt_0_2.txt ...` (list every per-test file) to check format.\n"
            "6. Exit with a brief confirmation.\n"
        )
    else:
        workflow = (
            "## Workflow (no-MCP mode, ASCII output)\n"
            "1. Read the inlined grids above (or `payload.json` for machine-readable form).\n"
            "2. EARLY (within your first few turns) overwrite `attempt_1.txt` and\n"
            "   `attempt_2.txt` with a first-pass guess — even copying the test\n"
            "   input is fine as a placeholder. The workspace must always contain a\n"
            "   submission; analysing forever without committing scores 0.\n"
            "3. Refine your transformation rule against every train pair.\n"
            "4. Apply the rule to the test input and OVERWRITE attempt_1.txt /\n"
            "   attempt_2.txt with your refined answer (rows of space-separated\n"
            "   digits 0-9, one row per line — same format as `grid_summary.md`).\n"
            "5. Run `python3 validate_prediction.py attempt_1.txt attempt_2.txt`\n"
            "   to check format.\n"
            "6. Exit with a brief confirmation.\n"
        )

    lines = [
        "# TASK",
        "",
        f"- task_id: {task.id}",
        "- task_source: arc",
        f"- split: {split}",
        "- mode: arc-no-mcp",
        f"- train_examples: {train_count}",
        f"- test_inputs: {test_count}",
        "",
        "## Summary",
        "You are solving an ARC puzzle. The full train/test grids are provided "
        "inline below and as `payload.json` / `grid_summary.md` in your workspace. "
        "Submit by overwriting `attempt_1.txt` and `attempt_2.txt` with plain "
        "ASCII grids (rows of space-separated digits 0-9). Do NOT write JSON.",
        "",
        "## Training Examples and Test Input",
        "The full train/test grids are inlined here as your primary inspection "
        "surface — read them directly from this document on every turn.",
        "",
        body,
        "",
        workflow,
        "",
        "## Validation discipline",
        "Before submitting, ALWAYS: (1) implement your candidate rule as executable "
        "code; (2) for EVERY training pair, construct the full predicted output grid "
        "and compare it cell-by-cell to the expected output; (3) only submit when "
        "every training pair matches exactly — if any cell mismatches, extract the "
        "mismatching cells and revise the rule; (4) build the final answer as a "
        "complete 2D integer grid in exactly the required output format. Verbal "
        "validation ('looks correct') does not count.",
        "",
        "## Output Format (required)",
    ]
    if test_count > 1:
        per_test_files = ", ".join(f"`attempt_{k}_1.txt`/`attempt_{k}_2.txt`" for k in range(test_count))
        lines.extend(
            [
                f"- This task has {test_count} test inputs. Write one answer per "
                "test into its OWN pair of files: " + per_test_files + ".",
                "- Each file must contain rows of space-separated digits 0-9, one row per line. NOT JSON.",
                "- Integer colors 0-9 only; rectangular grids; side <= 30.",
                "- If you only have one credible answer for a test, write the same "
                "grid into both of that test's files.",
                "- Do NOT call any MCP tool; do NOT modify payload.json/grid_summary.md/TASK.md.",
            ]
        )
    else:
        lines.extend(
            [
                "- Overwrite `attempt_1.txt` and `attempt_2.txt`. Each file must "
                "contain rows of space-separated digits 0-9, one row per line. "
                "NOT JSON.",
                "- Integer colors 0-9 only; rectangular grids; side <= 30.",
                "- If you only have one credible answer, write the same grid into both files.",
                "- Do NOT call any MCP tool; do NOT modify payload.json/grid_summary.md/TASK.md.",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _build_swebench_task_markdown(task: TaskSpec) -> str:
    metadata = task.metadata or {}
    task_source = str(metadata.get("task_source") or "swebench_pro").strip() or "swebench_pro"
    instance_id = str(metadata.get("instance_id") or task.id).strip()
    repo = str(task.repo or "").strip() or "(unknown)"
    base_commit = str(metadata.get("base_commit") or "").strip()
    hints_text = str(metadata.get("hints_text") or "").strip()
    # leak_test_names is only True when swebench_pro_seed_tests=True (DGM-equivalent mode).
    # In upstream-strict mode (default) test names are withheld from the prompt to
    # prevent the agent from reading them out of TASK.md / MEMORY.md / the workspace.
    leak_test_names: bool = bool(metadata.get("swebench_pro_seed_tests", False))
    fail_to_pass = _coerce_string_list(metadata.get("fail_to_pass")) if leak_test_names else []
    pass_to_pass = _coerce_string_list(metadata.get("pass_to_pass")) if leak_test_names else []
    selected_tests = _coerce_string_list(metadata.get("selected_test_files_to_run")) if leak_test_names else []
    # Counts are always available (no test-name information).
    fail_to_pass_count = len(_coerce_string_list(metadata.get("fail_to_pass")))
    pass_to_pass_count = len(_coerce_string_list(metadata.get("pass_to_pass")))
    repo_container_path = str(metadata.get("repo_container_path") or "/workspace/task/workspace/repo").strip()
    if not repo_container_path.startswith("/"):
        repo_container_path = "/workspace/task/workspace/repo"
    test_repo_container_path = str(
        metadata.get("official_repo_container_path") or metadata.get("test_repo_container_path") or repo_container_path
    ).strip()
    if not test_repo_container_path.startswith("/"):
        test_repo_container_path = repo_container_path

    lines = [
        "# TASK",
        "",
        f"- task_id: {task.id}",
        f"- task_source: {task_source}",
        f"- instance_id: {instance_id}",
        f"- repo: {repo}",
    ]
    if base_commit:
        lines.append(f"- base_commit: {base_commit}")

    lines.extend(
        [
            "",
            "## Objective",
            "Complete the named behavior or interface described by the issue and requirements; "
            "include any helper code needed to compile and pass the specified checks.",
        ]
    )
    if hints_text:
        lines.extend(["", "## Hints", hints_text])

    lines.extend(["", "## Verification"])
    if leak_test_names and fail_to_pass:
        lines.extend(
            [
                "Inspect these exact failing tests before editing and prioritize them before finalizing:",
                *[f"- `{item}`" for item in fail_to_pass],
            ]
        )
    elif not leak_test_names and fail_to_pass_count:
        lines.append(
            f"There are {fail_to_pass_count} target test(s) that must change from failing to passing. "
            "Infer which tests are relevant from the issue description, repository structure, and "
            "any existing test files — the exact names are not provided in upstream-strict mode."
        )
    if leak_test_names and selected_tests:
        lines.extend(
            [
                "",
                "Selected test files or suites to inspect first:",
                *[f"- `{item}`" for item in selected_tests],
            ]
        )
    selected_arg = ",".join(selected_tests)
    if leak_test_names:
        if selected_arg:
            script_command = (
                f"cd {_shell_single_quote(test_repo_container_path)} && "
                f"bash /workspace/task/workspace/run_script.sh {_shell_single_quote(selected_arg)}"
            )
        else:
            script_command = (
                f"cd {_shell_single_quote(test_repo_container_path)} && bash /workspace/task/workspace/run_script.sh"
            )
        lines.extend(
            [
                "",
                "Benchmark test script command:",
                f"- `{script_command}`",
                (
                    "Run this command ONLY if the file `/workspace/task/workspace/run_script.sh` "
                    "exists. Check with `test -f /workspace/task/workspace/run_script.sh` first. "
                    "If the script is absent, fall back to the named test files above and a "
                    "language-appropriate runner (`pytest`, `go test`, etc.) — do not waste "
                    "turns retrying the missing wrapper."
                ),
                "When present, prefer this wrapper over guessing the test command.",
            ]
        )
    else:
        script_command = (
            f"cd {_shell_single_quote(test_repo_container_path)} && bash /workspace/task/workspace/run_script.sh"
        )
        lines.extend(
            [
                "",
                "Benchmark test script command:",
                f"- `{script_command}`",
                (
                    "Run this command ONLY if the file `/workspace/task/workspace/run_script.sh` "
                    "exists. Check with `test -f /workspace/task/workspace/run_script.sh` first. "
                    "If the script is absent, fall back to language-appropriate runners "
                    "(`pytest`, `go test`, etc.) guided by the issue description — "
                    "do not waste turns retrying the missing wrapper."
                ),
                "When present, prefer this wrapper over guessing the test command.",
            ]
        )
    if (
        fail_to_pass
        or selected_tests
        or pass_to_pass
        or (not leak_test_names and (fail_to_pass_count or pass_to_pass_count))
    ):
        lines.extend(
            [
                "",
                "After each targeted run, use the first failing assertion or error to decide the next edit.",
            ]
        )
        if leak_test_names and pass_to_pass:
            preview = pass_to_pass[:5]
            lines.extend(
                [
                    "",
                    f"Preserve existing passing behavior covered by {len(pass_to_pass)} benchmark checks.",
                ]
            )
            if preview:
                lines.extend(["Examples:", *[f"- `{item}`" for item in preview]])
            if len(pass_to_pass) > len(preview):
                lines.append("Do not regress unrelated behavior while fixing the targeted tests.")
        elif not leak_test_names and pass_to_pass_count:
            lines.extend(
                [
                    "",
                    f"Preserve existing passing behavior covered by {pass_to_pass_count} benchmark checks. "
                    "Do not regress unrelated behavior while fixing the targeted tests.",
                ]
            )

    lines.extend(
        [
            "",
            "## Issue",
            task.prompt.strip() or "(none)",
        ]
    )

    lines.extend(
        [
            "",
            "## Output Format (required)",
            "Edit files in the workspace repo. The runtime captures the workspace diff as the canonical patch.",
            "Do not hand-write a `<patch>` block — it will be ignored unless the workspace contains no edits.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _build_polyglot_task_markdown(task: TaskSpec) -> str:
    metadata = task.metadata or {}
    language = str(metadata.get("language") or "unknown").strip()
    exercise_name = str(metadata.get("exercise_name") or "unknown").strip()
    starter_code: dict[str, str] = metadata.get("starter_code") or {}
    test_command = str(metadata.get("test_command") or "").strip()

    lines = [
        "# TASK",
        "",
        f"- task_id: {task.id}",
        "- task_source: polyglot",
        f"- language: {language}",
        f"- exercise: {exercise_name}",
        "",
        "## Problem",
        task.prompt.strip() or "(none)",
    ]

    if test_command:
        lines.extend(
            [
                "",
                "## Required Verification",
                "The hidden test suite for this exercise is NOT seeded into your workspace "
                "(by design: for most exercises the tests double as the answer key), so no "
                "test file exists on disk here to `Read` or inspect. Before finalizing, write "
                "and run your own ad hoc smoke check that exercises the required "
                "function/type signatures described above, then run the real exercise test "
                "command below as a compile/type-check pass:",
                "",
                "```",
                test_command,
                "```",
                "",
                "The hidden test file is intentionally absent, and what that looks like depends "
                "on the language/build system: an interpreted-language test runner (e.g. "
                "pytest) may report zero tests collected/run, while a compiled or CMake-based "
                "language (e.g. C++) may instead fail at the build/configure step with an error "
                "naming the missing test source file. Either outcome is expected and not itself "
                "a failure signal. Do not use ad hoc snippets in place of this command when it "
                "is available, and do not create, stub, or guess the contents of the missing "
                "test file to make the build succeed or to work around either failure mode — "
                "that file will be supplied by the grader.",
            ]
        )

    lines.extend(
        [
            "",
            # Memory MCP quickstart removed — query results pre-injected into MEMORY.md
        ]
    )

    if starter_code:
        lines.extend(["", "## Starter Code"])
        for filename, content in starter_code.items():
            # Infer language hint from file extension for fenced code blocks.
            ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            lines.extend(["", f"### `{filename}`", f"```{ext}", content.rstrip(), "```"])

    lines.extend(
        [
            "",
            "## Output Format (required)",
            f"Write the complete solution for the {language} exercise.",
            "Return the full source code of the solution file(s) — not a diff.",
            "Wrap each file in a fenced code block with the filename as a comment on the opening line:",
            "",
            "```",
            "// file: <filename>",
            "<complete file contents>",
            "```",
            "",
            "Do not include test files or build files — only the solution.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_task_markdown(task: TaskSpec) -> str:
    metadata = task.metadata or {}
    override = metadata.get("task_md_override")
    if isinstance(override, str) and override.strip():
        return override.strip() + "\n"

    spec = resolve_source(metadata.get("task_source"))
    # Spec-attached builder wins (after the explicit task_md_override above); the
    # hardcoded ``prompt_kind`` chain below is the fallback for built-in sources
    # that do not supply one.
    if spec is not None and spec.task_markdown_builder is not None:
        return spec.task_markdown_builder(task)
    kind = spec.prompt_kind if spec is not None else "generic"
    _warn_unhandled_prompt_kind(kind, spec_present=spec is not None)
    if kind == "arc":
        return _build_arc_no_mcp_task_markdown(task)
    if kind == "swebench_pro":
        return _build_swebench_task_markdown(task)
    if kind == "polyglot":
        return _build_polyglot_task_markdown(task)
    if kind == "terminal_bench_2":
        return ""

    lines = [
        "# TASK",
        "",
        f"- task_id: {task.id}",
        f"- repo: {task.repo or '(unknown)'}",
        "",
    ]

    best_score = metadata.get("best_score")
    if best_score is not None:
        try:
            score_val = float(best_score)
            if score_val > 0:
                lines.append(f"**Current best score: {score_val:.4f}. Try to exceed it.**")
                lines.append("")
        except (TypeError, ValueError):
            pass

    lines += [
        "## Problem",
        task.prompt.strip() or "(none)",
    ]
    return "\n".join(lines).strip() + "\n"


__all__ = [
    "build_execution_prompt",
    "build_task_markdown",
]
