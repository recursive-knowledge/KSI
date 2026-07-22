# Custom-tasks demo

Runs KSI on three self-contained tasks (`fizzbuzz`, `reverse-words`,
`anagram-groups`) defined entirely in `tasks.jsonl` — no benchmark dataset
download required. Each eval only needs host `python3`, no pip deps.

## Task record schema

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

`workspace_dir` and `files` are mutually exclusive; a record may omit both
to start the agent from an empty `repo/`. Either way the task's starting
files are seeded into the agent workspace's `repo/` directory, and the
`command` evaluator later runs `eval.command` in a post-attempt copy of that
directory.

## How eval commands run

`eval.command` runs **host-side** (never inside the agent container) with
the launching user's privileges, so grader output stays out of the agent's
reach by construction. Exit 0 scores 1.0, nonzero scores 0.0 (a `score.json`
`{"score": <0..1>}` written by the command overrides this for partial
credit).

**SECURITY:** the eval command comes straight from the tasks file you point
KSI at. Do not run a custom tasks file you don't trust.

## Run it

First create a provider profile from a template (once):

```bash
cp configs/ksi/.env.haiku.template configs/ksi/.env.haiku   # then set your API key
```

CLI form:

```bash
PROVIDER_PROFILE=configs/ksi/.env.haiku bash examples/custom_tasks/run.sh
```

Programmatic form:

```bash
uv run python examples/custom_tasks/run.py
```

Both need Docker running, Node.js (>=22.16.0 <23), the `ksi-agent:bench` image
built, and a provider profile with a real API key (see `configs/ksi/*.template`).

## Expected output

A typical run solves 3/3 tasks; per-task eval results show
`"status": "evaluated"` with `"native_score": 1.0`.
