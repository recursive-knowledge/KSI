"""Setup readiness check: `uv run kcsi-doctor`.

Prints a ✓/✗/⚠ report of everything a run needs (Python, Docker daemon and the
agent image, Node and the host runtime deps, a provider profile with a real
key) plus the exact command to fix each failure. Exits non-zero if any
hard requirement is missing so it can gate `scripts/quickstart.sh` and CI.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from kcsi.layout import PROJECT_ROOT
from kcsi.providers import ProviderConfigError, load_provider_profile
from kcsi.tasks.custom import validate_custom_tasks_path

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"

PROVIDERS_DIR = PROJECT_ROOT / "configs" / "kcsi"
RUNTIME_RUNNER_DIR = PROJECT_ROOT / "runtime_runner"
AGENT_IMAGE = "kcsi-agent:bench"
CUSTOM_TASKS_DEMO = PROJECT_ROOT / "examples" / "custom_tasks" / "tasks.jsonl"


def _run(cmd: list[str], timeout: float = 15.0) -> tuple[int, str]:
    """Run a command, returning (exit_code, combined output). 127 if missing."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""


class Report:
    def __init__(self) -> None:
        self.hard_failures = 0

    def line(self, status: str, label: str, detail: str = "", fix: str = "") -> None:
        msg = f"  {status} {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        if fix:
            print(f"      fix: {fix}")

    def ok(self, label: str, detail: str = "") -> None:
        self.line(OK, label, detail)

    def warn(self, label: str, detail: str = "", fix: str = "") -> None:
        self.line(WARN, label, detail, fix)

    def fail(self, label: str, detail: str = "", fix: str = "") -> None:
        self.hard_failures += 1
        self.line(FAIL, label, detail, fix)


def _check_python(r: Report) -> None:
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 12):
        r.ok(label)
    else:
        r.fail(label, "need >= 3.12", "install Python 3.12+ and re-run `uv sync`")


def _check_uv(r: Report) -> None:
    if shutil.which("uv"):
        r.ok("uv on PATH")
    else:
        r.warn("uv not on PATH", fix="install uv: https://docs.astral.sh/uv/")


def _check_docker(r: Report) -> None:
    if not shutil.which("docker"):
        r.fail("Docker", "not installed", "install Docker and start the daemon")
        return
    code, _ = _run(["docker", "info"])
    if code != 0:
        r.fail("Docker daemon", "not running", "start Docker (e.g. `sudo systemctl start docker`)")
        return
    r.ok("Docker daemon running")

    code, out = _run(["docker", "images", "-q", AGENT_IMAGE])
    if code == 0 and out.strip():
        r.ok(f"image {AGENT_IMAGE} present")
    else:
        r.fail(
            f"image {AGENT_IMAGE} missing",
            fix="bash container/build.sh --bench",
        )


def _check_node(r: Report) -> None:
    if not shutil.which("node"):
        r.fail("Node.js", "not installed", "install Node.js 20+ (the host runtime needs it)")
        return
    code, out = _run(["node", "--version"])
    r.ok("Node.js", out.strip() if code == 0 else "")

    if (RUNTIME_RUNNER_DIR / "node_modules").is_dir():
        r.ok("runtime_runner deps installed")
    else:
        r.fail(
            "runtime_runner deps missing",
            fix="cd runtime_runner && npm install --legacy-peer-deps",
        )


def _check_providers(r: Report) -> None:
    if not PROVIDERS_DIR.is_dir():
        r.fail(
            "provider profiles",
            f"{PROVIDERS_DIR} not found",
            "bash scripts/setup_all.sh --no-test",
        )
        return

    # Only consider concrete profiles, not *.template files.
    profiles = sorted(
        p for p in PROVIDERS_DIR.glob(".env.*") if p.suffix != ".template" and not p.name.endswith(".template")
    )
    if not profiles:
        r.fail(
            "provider profiles",
            "none found",
            "bash scripts/setup_all.sh  (creates .env.haiku / .env.sonnet / .env.openai)",
        )
        return

    usable: list[str] = []
    for prof in profiles:
        try:
            load_provider_profile(str(prof))
            usable.append(prof.name)
        except ProviderConfigError:
            pass

    if usable:
        r.ok("provider profile ready", ", ".join(usable))
    else:
        r.fail(
            "no provider profile has a usable key",
            f"checked {len(profiles)} profile(s) in {PROVIDERS_DIR}",
            "add a real key, e.g. set ANTHROPIC_API_KEY in configs/kcsi/.env.haiku",
        )


def _check_quickstart_demo(r: Report) -> None:
    if not CUSTOM_TASKS_DEMO.is_file():
        r.fail(
            "quickstart demo tasks",
            f"{CUSTOM_TASKS_DEMO} not found",
            "check out a clean copy of the repo (bundled file should not be deleted)",
        )
        return
    error = validate_custom_tasks_path(CUSTOM_TASKS_DEMO)
    if error is None:
        r.ok("quickstart demo tasks", str(CUSTOM_TASKS_DEMO.relative_to(PROJECT_ROOT)))
    else:
        r.fail("quickstart demo tasks", error)


def _check_memory(r: Report) -> None:
    try:
        import sqlite_vec  # noqa: F401

        r.ok("vector memory extra installed", "(optional)")
    except Exception:
        r.warn(
            "vector memory extra not installed",
            "(optional; FTS-only fallback is used)",
            "uv sync --extra memory  and set HF_TOKEN",
        )


def _check_vector_status(r: Report, knowledge_db: str) -> None:
    """Report embedder/vector-index state recorded in a knowledge DB.

    Reads the ``vector_status`` telemetry table (written by the engine at
    init, embedder startup, and backfill) so a run with missing vector search
    is diagnosable after the fact — see the "Semantic vector search not active"
    troubleshooting row.
    """
    path = Path(knowledge_db)
    if not path.is_file():
        r.fail("knowledge DB", f"{path} not found")
        return
    try:
        conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vector_status'").fetchone()
            if row is None:
                r.warn("vector_status table missing", f"{path.name} is not a knowledge DB (or predates telemetry)")
                return
            # Latest row per phase = current state of that phase.
            rows = conn.execute(
                """
                SELECT phase, status, detail, embedding_count, skipped_count
                FROM vector_status
                WHERE id IN (SELECT MAX(id) FROM vector_status GROUP BY phase)
                ORDER BY phase
                """
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        r.fail("knowledge DB unreadable", str(exc))
        return

    if not rows:
        r.warn("vector_status empty", "no embedder/vector telemetry recorded yet")
        return
    embedded = sum(int(row["embedding_count"] or 0) for row in rows)
    skipped = sum(int(row["skipped_count"] or 0) for row in rows)
    for row in rows:
        label = f"vector_status[{row['phase']}]"
        detail = str(row["detail"] or "")
        # "off" (FTS5 default) and "disabled" (KCSI_DISABLE_VECTOR) are
        # intentional states, not faults — only a genuine "degraded" warns.
        if row["status"] in {"enabled", "off", "disabled"}:
            r.ok(label, detail)
        else:
            r.warn(label, f"{row['status']}: {detail}", "uv sync --extra memory  and set HF_TOKEN")
    r.ok("embedding coverage", f"{embedded} embedded, {skipped} skipped")


def doctor_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kcsi-doctor",
        description="Check that this machine is ready to run kcsi experiments.",
    )
    parser.add_argument(
        "--knowledge-db",
        metavar="PATH",
        default=None,
        help="optional: existing <stem>_knowledge.sqlite — report its vector_status (embedder/vector backfill state)",
    )
    args = parser.parse_args(argv)

    print("kcsi setup check\n")
    r = Report()

    print("Core")
    _check_python(r)
    _check_uv(r)
    print("\nRuntime (container execution)")
    _check_docker(r)
    _check_node(r)
    print("\nProviders")
    _check_providers(r)
    print("\nExamples")
    _check_quickstart_demo(r)
    print("\nOptional")
    _check_memory(r)
    if args.knowledge_db:
        print("\nKnowledge DB vector state")
        _check_vector_status(r, args.knowledge_db)

    print()
    if r.hard_failures == 0:
        print(f"{OK} Ready. Try: bash scripts/quickstart.sh")
        return 0
    print(f"{FAIL} {r.hard_failures} blocking issue(s). Fix the items above, then re-run `uv run kcsi-doctor`.")
    return 1


def doctor_cli() -> None:
    raise SystemExit(doctor_main())


if __name__ == "__main__":
    doctor_cli()
