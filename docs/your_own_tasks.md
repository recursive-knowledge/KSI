# Your own tasks

The built-in `custom` task source runs KSI on any set of tasks you define —
no benchmark dataset, no loader code. Point `--task-source custom` (or
`load_tasks_for_source(task_source="custom", ...)`) at a `.json` or `.jsonl`
file, and each record becomes one attempt.

## Record schema

```jsonc
{
  "task_id": "my-task-1",                     // required, unique
  "prompt": "…instruction for the agent…",    // required
  "workspace_dir": "path/to/starting/files",  // optional, dir; relative to the tasks file
  "files": {"relative/path.py": "content"},   // optional, inline alternative to workspace_dir
  "eval": {"command": "python3 tests.py",     // optional; graded by the `command` evaluator
           "timeout_sec": 300}
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `task_id` | yes | Must be unique within the file. |
| `prompt` | yes | The instruction shown to the agent. KSI appends a short note pointing it at `repo/` (see below). |
| `workspace_dir` | no | A directory of starting files, resolved relative to the tasks file. Mutually exclusive with `files`. |
| `files` | no | An inline `{relative/path: content}` map of starting files, materialized into a temp directory at load time. Mutually exclusive with `workspace_dir`. Paths must be relative and may not contain `..`. |
| `eval` | no | `{"command": "...", "timeout_sec": 300}`. Omit entirely for an unscored/manual-inspection task. |

A `.json` file must be a JSON array of records; a `.jsonl` file is one record
per line. Unknown keys on a record are rejected, so a typo fails at load time
instead of being silently ignored.

Neither `workspace_dir` nor `files` is required — a record that sets neither
starts the agent from an empty `repo/`.

## The workspace / `repo/` contract

Whichever way you supply starting files, KSI seeds them into the agent's
workspace under a `repo/` directory before the attempt starts (the same seam
the benchmark task sources use). The agent is told in its prompt that
`repo/` holds the task's starting files and that it should create or edit
files there. After the attempt, the `command` evaluator (below) runs in a
post-attempt copy of that same directory.

**Known limitation — workspace capture is capped at 12 files.** By default
KSI wipes the container workspace after each task
(`--wipe-workspace-per-task true`), so grading runs against a *captured* copy
of `repo/` rather than the live container filesystem. For a generic (non
-benchmark) task source, that capture channel reads back at most 12 files
from the workspace, skipping anything named `score.json` or containing
`test` in its filename (plus `.pyc`/`__pycache__` noise and any single file
over 1 MB); a solution spread across more than 12 files silently
loses the extras from this channel. This is fine for small, self-contained
tasks (a script or two). For a larger multi-file solution, either point
`--wipe-workspace-per-task false` at your run so the evaluator sees the live
on-disk workspace instead, or design the eval command to check for what
matters rather than relying on every generated file surviving capture.

Note that `--wipe-workspace-per-task false` also bypasses the capture-path
anti-tamper filtering: with the live workspace graded directly, an agent
that edited a `test`-named file or wrote its own `score.json` mid-run has
those files graded as-is (a stale `score.json` is still deleted before the
eval command runs, but edited test files are not restored from the seed).
Prefer the default capture path when your eval command relies on its test
assets staying pristine.

## The `command` evaluator contract

`--evaluator command` (`ksi.eval.command.CommandEvaluator`) runs
`eval.command` as a shell command in the workspace described above, **on the
host** — never inside the agent's container.

- **Exit 0** scores `1.0`.
- **Any nonzero exit** scores `0.0`.
- **Partial credit:** if the command writes a `score.json` file with
  `{"score": <0..1>}` in the workspace, that value overrides the exit-code
  score. (A stale `score.json` left over from an earlier attempt or written
  directly by the agent is deleted before the command runs, so only the
  eval command's own output counts.)
- **Timeout:** if the command doesn't finish within `eval.timeout_sec`
  (default 300s), the attempt is **unscored** (`None`), not scored `0.0` —
  it is excluded from best-score tracking, distillation, and forum context,
  the same way any other infra failure is. A missing workspace or a command
  that fails to spawn is unscored for the same reason.

**Security:** the eval command comes straight from the tasks file you point
KSI at, and runs with the privileges of whoever launches the run. **Do not
point KSI at a tasks file you don't trust.**

## JSONL vs. Python `TaskSpec` forms

The JSONL/JSON file form above is the quickest path and is what
`--tasks-path` expects. If you're driving a run programmatically (see
[Programmatic API](programmatic_api.md)), you can either load the same file:

```python
from ksi.tasks.loaders import load_tasks_for_source

tasks = load_tasks_for_source(task_source="custom", tasks_path="tasks.jsonl")
```

or hand-build `TaskSpec` objects directly, bypassing the file entirely — this
is what `load_custom_tasks` produces under the hood:

```python
from ksi.models import TaskSpec

tasks = [
    TaskSpec(
        id="my-task-1",
        repo="",
        prompt="…instruction for the agent…",
        metadata={
            "task_source": "custom",
            "repo_path": "/absolute/path/to/starting/files",  # a real, existing directory
            "eval_command": "python3 tests.py",
            "eval_timeout_sec": 300.0,
        },
    )
]
```

`repo_path` must already exist and hold the starting files (the loader
materializes `files`/`workspace_dir` into a directory for you; building
`TaskSpec` by hand means you own that step).

## When you need more

The `custom` source and `command` evaluator cover single-command,
pass/fail-or-partial-credit grading. If your evaluation needs something the
`command` evaluator can't express — a multi-step harness, a different
scoring contract, or a task source with its own prompt/loader behavior —
register your own:

- [Adding a benchmark / task source](adding_a_benchmark.md)
- [Adding an evaluator](adding_an_evaluator.md)
