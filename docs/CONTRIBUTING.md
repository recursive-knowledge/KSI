# Contributing to kcsi

Thanks for your interest in extending the knowledge-centric self-improvement
agent. This guide gets you from clone to a tested change.

## Environment

Python 3.12+ and [uv](https://github.com/astral-sh/uv) are required. All Python
runs through `uv run`.

```bash
git clone https://github.com/recursive-knowledge/KCSI
cd kcsi
bash scripts/setup_all.sh          # installs deps + builds the agent container image
# or, deps only:
uv sync --extra memory
```

Verify your setup:

```bash
uv run kcsi-doctor                 # setup readiness check
bash scripts/quickstart.sh          # end-to-end synthetic ARC demo (no dataset download)
```

## Develop / test / lint

```bash
uv run pytest                       # full test suite (see note below)
uv run ruff check src tests scripts examples --fix   # lint (line length 120) — CI scope
uv run ruff format src tests scripts examples        # format — CI scope
uv run mypy src/kcsi --follow-imports=silent  # type check (CI scope)
```

> Mypy runs over the WHOLE `src/kcsi` package (honoring the shipped `py.typed`
> marker). Modules that still carry pre-existing errors are explicitly
> opted out with per-module `[[tool.mypy.overrides]] ignore_errors = true`
> entries in `pyproject.toml`, so new modules are type-checked by default.
> Ratchet: bring an opted-out module to zero errors, then delete it from that
> override list.

Git hooks live in `.githooks/` and are enabled by `bash scripts/setup_all.sh`
(via `git config core.hooksPath .githooks`). The pre-commit hook runs the CI
ruff lint + format check (`uv run ruff check` / `ruff format --check src tests
scripts examples`) when Python files are staged, and the TypeScript typechecks
when runtime_runner / agent-runner
sources are staged. Bypass with `SKIP_KCSI_LINT=1` /
`SKIP_KCSI_TYPECHECK=1`, or `--no-verify`.

**Run the full suite before pushing — not a file subset.** A change to a shared
dispatch site or a moved symbol can break a test in a file you never opened;
running only the files you touched hides that. (The pre-commit hook runs ruff
and the TypeScript typechecks, but not pytest.)

## Extending kcsi (the seams)

kcsi is built around small, typed extension points so you can add a benchmark,
evaluator, runtime, or improvement strategy **without editing core engine code**.
Each is a `Protocol` + a registry. Start here:

➡️ **[extending.md](./extending.md)** — the index of every seam and its guide.

The pattern is always the same: implement the `Protocol`, then `register_*(...)`
your spec. No `if name == ...` dispatch edits.

## Project layout

[architecture.md](./architecture.md) is the canonical architecture guide: it covers the
runtime and DB design, including the maintained execution path (§1 System Map,
§2 Attempt Runtime Lifecycle, §9 Provider Runtime) and database ownership
(§5 Database Ownership). The primary package is `src/kcsi/`. See
[extending.md](./extending.md) for the index of extension seams.

## Pull requests

- Keep PRs focused. Split multi-part work into small logical commits by intent,
  pairing related code + tests + docs.
- Add or update tests for any behavior change; for pure refactors, add a
  parity/equivalence test that pins the preserved behavior.
- Ensure `uv run pytest`, `ruff check`, `ruff format --check`, and the
  CI-scoped `mypy` (see the commands above) pass.
- PRs merge to `main` (squash). For stacked work, retarget child PRs to `main`
  before merging the base.

## Code style

- Linter/formatter: `ruff` (line length 120). Config in `pyproject.toml`.
- Type checking: `mypy` (`kcsi` ships a PEP 561 `py.typed` marker).
- Match the surrounding code's idiom; touch only what the change requires.
