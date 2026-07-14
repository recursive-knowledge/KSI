from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

# Snapshot env keys that existed BEFORE load_dotenv() pollutes os.environ.
# Only these should be allowed to override provider-profile values.
_PRE_DOTENV_PROVIDER_KEYS = frozenset(
    k
    for k in (
        "MODEL",
        "MODEL_PROVIDER",
        "MODEL_AUTH_MODE",
        "REASONING_EFFORT",
        "KSI_OPENAI_MAX_TURNS",
        "OPENAI_AGENTS_DISABLE_TRACING",
    )
    if k in os.environ
)
load_dotenv()

log = logging.getLogger(__name__)

from .benchmarks.polyglot_harness import DEFAULT_POLYGLOT_TIMEOUT_SEC
from .benchmarks.swebench_pro_external import DATASET_REVISION as SWEBENCH_PRO_DATASET_REVISION
from .cli_reporting import (
    _build_tb2_results_summary as _build_tb2_results_summary,  # noqa: F401  re-export for ksi.cli test imports
)
from .cli_reporting import (
    _resolve_arc_split as _resolve_arc_split,  # noqa: F401  re-export for ksi.cli test imports
)
from .cli_reporting import (
    _serialize_trace_with_retry_summary as _serialize_trace_with_retry_summary,  # noqa: F401  re-export
)
from .cli_reporting import (
    write_output_json,
    write_pretask_debug_json,
)
from .distillation._removed_env import assert_no_removed_channel_env
from .eval import (
    get_evaluator_spec,
    supported_evaluators,
)
from .layout import (
    default_knowledge_db_path,
    default_runtime_db_path,
    default_swebench_repo_cache_dir,
    derive_legacy_sibling,
    derive_runtime_sibling,
    legacy_flat_knowledge_db_path,
    sanitize_key,
)
from .logging_config import configure_logging
from .models import GenerationConfig, TaskSpec
from .orchestrator.engine import GenerationalOrchestrator
from .orchestrator.persistence import (
    CollectingPersistence,
    CompositePersistence,
    SqlitePersistence,
)
from .orchestrator.strategy import get_strategy_spec, supported_strategies
from .protocols import Evaluator, RuntimeExecutor
from .providers import ProviderConfigError, apply_provider_env, load_provider_profile
from .runtime import (
    TerminalBench2Executor,
    get_runtime_spec,
    supported_runtimes,
)
from .runtime.llm import build_llm_caller
from .tasks import (
    CLASSIFY_MAX_WORKERS,
    classify_tasks,
    get_spec,
    load_categories_json,
    load_eval_records_for_source,
    load_tasks_for_source,
    resolve_source,
    upstream_strict_task_sources,
)
from .tasks.repo_cache import prepare_swebench_repo_snapshots

# Derived from the registry (upstream_strict flag) so a newly-registered
# published benchmark is covered without editing a parallel hardcoded list.
_PUBLISHED_BENCHMARK_TASK_SOURCES = frozenset(upstream_strict_task_sources())


@contextmanager
def _temporary_env_override(name: str, value: str | None):
    """Temporarily set an environment variable for one in-process CLI run."""
    if not value:
        yield
        return

    had_previous = name in os.environ
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if had_previous and previous is not None:
            os.environ[name] = previous
        else:
            os.environ.pop(name, None)


def _parse_bool_flag(value) -> bool:
    """Strict boolean parser used by CLI toggle flags.

    Accepts Python ``bool`` values (returned verbatim) as well as string
    aliases such as ``true``/``false``, ``yes``/``no``, ``1``/``0``.
    """
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off", ""):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


class _RemovedMemoryDbPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            "--memory-db-path was removed. Use --knowledge-db-path for the persistent knowledge substrate "
            "or --runtime-db-path for optional audit logs."
        )


class _RemovedAgentsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            "--agents was removed. In decentralized task mode the agent count derives from the "
            "filtered task pool; use --max-concurrent-tasks to cap parallelism."
        )


#: Guidance for removed compatibility flags (former deprecated aliases).
_REMOVED_FORUM_FLAG_GUIDANCE = {
    "--forum-rounds": "Use --per-task-forum-rounds and --cross-task-forum-rounds.",
    "--forum-mode": (
        "Forums are on by default; use --per-task-forum-rounds 0 --cross-task-forum-rounds 0 to disable them."
    ),
    "--forum-ablate-r3": "Use --distill-enabled=false to skip the distillation phase.",
}


class _RemovedForumFlagAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        guidance = _REMOVED_FORUM_FLAG_GUIDANCE.get(option_string or "", "")
        parser.error(f"{option_string} was removed. {guidance}".strip())


def _resolve_runtime_timeout_default() -> int:
    """Return the default per-task runtime timeout in seconds.

    Checks ``CROSS_RUNNER_AGENT_TIMEOUT_SEC`` first so cross-runner sweep
    scripts can set a single unified timeout for all three runners (ksi,
    HyperAgents, DGM) without touching the CLI invocation.  Falls back to
    1800 when the env var is absent or invalid.
    """
    raw = os.environ.get("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return 1800


class _KsiArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        parsed = super().parse_args(args=args, namespace=namespace)
        _validate_cross_task_distill_flags(parsed, self)
        return parsed


def _validate_cross_task_distill_flags(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if bool(getattr(args, "cross_task_distill_per_target_selection", False)) and not bool(
        getattr(args, "cross_task_distill_target_conditioning", True)
    ):
        parser.error("--cross-task-distill-per-target-selection requires --cross-task-distill-target-conditioning true")


def build_parser() -> argparse.ArgumentParser:
    p = _KsiArgumentParser(description="Knowledge-centric benchmark runtime")

    # ── Observability ───────────────────────────────────────────────────────
    g_obs = p.add_argument_group("Observability")
    g_obs.add_argument(
        "--log-level",
        type=str.upper,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity. Overrides the KSI_LOG_LEVEL env var; default (when neither is set) is INFO.",
    )
    g_obs.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        action="store_const",
        const="DEBUG",
        help="Shortcut for --log-level DEBUG.",
    )

    # ── Task Input ──────────────────────────────────────────────────────────
    g_task = p.add_argument_group("Task Input")
    g_task.add_argument(
        "--task-source",
        required=True,
        help="Task dataset format. "
        "'arc' accepts a json file or directory; "
        "swebench_pro accepts parquet/csv/jsonl; "
        "polyglot expects a tasks JSON file; "
        "terminal_bench_2 expects a task-map JSON file.",
    )
    g_task.add_argument(
        "--tasks-path",
        required=True,
        help="Path to tasks. csv/jsonl/parquet for swebench_pro; json for polyglot; "
        "json file or directory for arc; json task map for terminal_bench_2.",
    )
    g_task.add_argument("--evals-path", default=None, help="Optional path to eval records (parquet).")
    g_task.add_argument("--task-ids", default=None, help="Comma-separated list of task IDs to include")
    g_task.add_argument(
        "--task-ids-file",
        default=None,
        help="Path to JSON file containing a list of task IDs to include",
    )
    g_task.add_argument(
        "--task-map-path",
        default=None,
        help=(
            "Optional task-map/manifest path used to select this run's tasks. "
            "When --task-ids or --task-ids-file is also provided, it must select "
            "the same task IDs in the same order. When --output-json is set, "
            "compact map metadata is persisted under the top-level task_map field."
        ),
    )
    g_task.add_argument("--max-tasks", type=int, default=0, help="Cap tasks after filtering (0=all)")
    g_task.add_argument(
        "--holdout-task-ids",
        default=None,
        help="Comma-separated hold-out task IDs: attempted every generation with the "
        "current cross-task knowledge but excluded from learning (forums, distillation, "
        "seeding inputs), --drop-solved, early-stop, and headline metrics. Must be "
        "disjoint from the training task ids.",
    )
    g_task.add_argument(
        "--holdout-task-ids-file",
        default=None,
        help="Path to JSON file of hold-out task IDs (same shapes as --task-ids-file)",
    )

    # ── Evaluation ──────────────────────────────────────────────────────────
    g_eval = p.add_argument_group("Evaluation")
    g_eval.add_argument(
        "--arc-max-trials",
        type=int,
        default=2,
        help="ARC trial budget enforced by MCP tools and reflected in TASK.md (default: 2).",
    )
    g_eval.add_argument(
        "--polyglot-test-feedback-tries",
        type=int,
        default=2,
        help=(
            "Polyglot retry-with-test-feedback budget, matching Aider's "
            "--tries default: on a failing attempt, the solver sees its own "
            "capped test-runner output and gets one more try in the same "
            "session. Set to 1 to reproduce the old strict single-shot "
            "protocol (default: 2)."
        ),
    )
    g_eval.add_argument(
        "--polyglot-test-feedback-max-lines",
        type=int,
        default=50,
        help="Max lines of test-runner stdout/stderr shown per retry round (default: 50, matching Aider).",
    )
    # ARC always uses native tools (Bash/Read/Edit/Write/Glob/Grep): the agent
    # reads payload.json and writes attempt_1.txt / attempt_2.txt, or per-test
    # attempt files for multi-test ARC tasks. The legacy ARC MCP toolset has
    # been removed, so there is no flag to select it.
    # default=None is an "omitted" sentinel: when --evaluator is absent, the
    # task source's registered default evaluator is applied downstream (see
    # _normalize_evaluator_for_task_source). A default of "swebench_pro" here
    # would be indistinguishable from an explicit --evaluator swebench_pro,
    # silently rewriting the explicit choice for non-swebench task sources
    # argparse does not validate the default against `choices`.
    g_eval.add_argument("--evaluator", choices=supported_evaluators(include_aliases=True), default=None)
    g_eval.add_argument(
        "--swebench-timeout-sec",
        type=int,
        default=3600,
        help=("SWE-bench Pro harness subprocess timeout in seconds. Default 3600; no hidden grace is added."),
    )
    g_eval.add_argument(
        "--swebench-harness-grace-sec",
        type=int,
        default=0,
        help=(
            "Extra subprocess timeout grace for the SWE-bench Pro harness. "
            "Default 0 keeps the requested --swebench-timeout-sec contract exact."
        ),
    )
    g_eval.add_argument(
        "--swebench-docker-network-mode",
        default="host",
        help=(
            "Deprecated SWE-bench network alias. Use --swebench-pro-block-network "
            "for network=none; any other value is accepted for old configs but "
            "does not change the third-party Pro harness invocation."
        ),
    )
    g_eval.add_argument(
        "--swebench-repos-dir",
        default=None,
        help="Directory with pre-checked-out repos keyed by instance_id "
        "(default: benchmarks/swebench_pro/repo_cache/<instance_id>/).",
    )
    g_eval.add_argument(
        "--swebench-pro-raw-sample-path",
        default="",
        help="Raw SWE-bench Pro dataset file used by the official evaluator (csv/jsonl). "
        "Required for --evaluator swebench_pro unless --tasks-path is already csv/jsonl.",
    )
    g_eval.add_argument(
        "--swebench-pro-repo-root",
        default="",
        help="Path to cloned scaleapi/SWE-bench_Pro-os repo root. "
        "Default: benchmarks/swebench_pro/evaluator under this repo.",
    )
    g_eval.add_argument(
        "--swebench-pro-dockerhub-username",
        default="jefzda",
        help="Docker Hub username hosting SWE-bench Pro images (default: jefzda).",
    )
    g_eval.add_argument(
        "--swebench-pro-use-local-docker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local Docker for SWE-bench Pro evaluation (default: true).",
    )
    g_eval.add_argument(
        "--swebench-pro-docker-platform",
        default=None,
        help="Optional Docker platform override for SWE-bench Pro, e.g. linux/amd64.",
    )
    g_eval.add_argument(
        "--swebench-pro-block-network",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Block network access inside SWE-bench Pro evaluation containers.",
    )
    g_eval.add_argument(
        "--swebench-pro-seed-tests",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Seed grader test files into the agent's repo as a baseline commit "
            "on top of base_commit (DGM-equivalent harness). Default false = "
            "upstream-strict: agent works against base_commit alone, must "
            "infer test APIs from the issue/problem statement, matching the "
            "upstream SWE-bench Pro reference protocol where "
            "before_repo_set_cmd executes only inside the grader, after the "
            "agent's patch is applied. Set true only for DGM-comparable runs; "
            "results are NOT comparable to public SWE-bench Pro leaderboards."
        ),
    )
    g_eval.add_argument(
        "--strict-swebench-dataset-integrity",
        action="store_true",
        default=False,
        help=(
            "Fail closed (SystemExit) when a SWE-bench Pro task map's recorded "
            "source_sha256 does not match the sha256 of the --tasks-path "
            "dataset. Maps with source_revision are strict by default. Legacy "
            "maps without source_revision WARN and continue unless this flag "
            "or KSI_STRICT_SWEBENCH_DATASET_INTEGRITY=1 is set."
        ),
    )
    g_eval.add_argument(
        "--polyglot-timeout-sec",
        type=int,
        default=DEFAULT_POLYGLOT_TIMEOUT_SEC,
        help=f"Per-task timeout for polyglot exercise test execution (default: {DEFAULT_POLYGLOT_TIMEOUT_SEC}).",
    )
    g_eval.add_argument(
        "--polyglot-docker-image",
        default=os.environ.get("POLYGLOT_DOCKER_IMAGE", "ksi-polyglot-eval:latest"),
        help=("Docker image for polyglot evaluation (default: $POLYGLOT_DOCKER_IMAGE or 'ksi-polyglot-eval:latest')."),
    )

    # ── Runtime ─────────────────────────────────────────────────────────────
    g_runtime = p.add_argument_group("Runtime")
    g_runtime.add_argument(
        "--runtime",
        choices=supported_runtimes(include_aliases=True),
        default="container",
        help="Execution runtime. `container` is the shared container runtime.",
    )
    g_runtime.add_argument(
        "--container-command",
        default="",
        help="Command used by the shared container runtime (space-separated).",
    )
    g_runtime.add_argument(
        "--runtime-timeout-sec",
        type=int,
        default=None,
        help=(
            "Per-task runtime hard container cap in seconds. Omit to use the "
            "default (1800s, or CROSS_RUNNER_AGENT_TIMEOUT_SEC); 0 keeps the "
            "1800s cap; a negative value disables the hard cap so the per-task "
            "task.toml timeout is the sole wall-time bound. NOTE: for "
            "--task-source terminal_bench_2 the timeout is NOT configurable — "
            "the per-task task.toml [agent].timeout_sec is authoritative "
            "(Harbor parity), so a non-negative value is rejected and omission "
            "defaults to no hard cap."
        ),
    )
    g_runtime.add_argument(
        "--session-scope",
        choices=("task", "agent"),
        default="task",
        help="Runtime/session memory scope for shared container execution.",
    )
    g_runtime.add_argument(
        "--wipe-workspace-per-task",
        choices=("true", "false"),
        default="true",
        help="Whether to wipe/rebuild task workspace before each task run.",
    )

    # ── Execution Control ───────────────────────────────────────────────────
    g_exec = p.add_argument_group("Execution Control")
    g_exec.add_argument(
        "--agents",
        dest="removed_agents",
        default=argparse.SUPPRESS,
        action=_RemovedAgentsAction,
        help=argparse.SUPPRESS,
    )
    g_exec.add_argument("--generations", type=int, default=10)
    g_exec.add_argument(
        "--max-concurrent-tasks", type=int, default=25, help="Max concurrent task workers (default: 25)"
    )
    g_exec.add_argument(
        "--max-task-retries",
        type=int,
        default=3,
        help="Max retries per task on transient failure (0=no retry, default: 3)",
    )
    g_exec.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "RNG seed. Seeds Python's stdlib random (affects retry-backoff jitter only). "
            "NOTE: no LLM receives a usable seed — Anthropic ignores it, the OpenAI "
            "Responses API does not support it, the in-container task agent is not "
            "seeded, and IDs use uuid4 — so this does NOT make runs reproducible. "
            "Default 0."
        ),
    )
    g_exec.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help=(
            "Temperature passed to direct LLM calls (forum/distill/claiming). "
            "Default 0.0 for maximum determinism. Reasoning-family OpenAI models "
            "ignore this flag because the Responses API rejects temperature for "
            "gpt-5*/o-family backbones."
        ),
    )
    g_exec.add_argument(
        "--drop-solved",
        action="store_true",
        default=True,
        help="Drop solved tasks from the pool in subsequent generations. "
        "A task is 'solved' when its best native_score >= threshold. (default: True)",
    )
    g_exec.add_argument(
        "--no-drop-solved",
        action="store_false",
        dest="drop_solved",
        help="Disable dropping of solved tasks between generations.",
    )
    g_exec.add_argument(
        "--solved-threshold",
        type=float,
        default=1.0,
        help="native_score threshold for considering a task 'solved' (default: 1.0).",
    )

    # ── Memory ──────────────────────────────────────────────────────────────
    g_mem = p.add_argument_group("Memory")
    g_mem.add_argument(
        "--knowledge-db-path",
        dest="knowledge_db_path",
        default="",
        help="Path to authoritative SQLite knowledge DB. "
        "Default: runtime_state/knowledge/<experiment>/<experiment>_knowledge.sqlite.",
    )
    g_mem.add_argument(
        "--runtime-db-path",
        dest="runtime_db_path",
        default="",
        help="Path to optional SQLite runtime audit DB. "
        "Default: sibling <stem>_runtime.sqlite derived from --knowledge-db-path.",
    )
    g_mem.add_argument(
        "--no-runtime-db",
        action="store_true",
        default=False,
        help="Disable the optional runtime audit DB sidecar.",
    )
    g_mem.add_argument(
        "--memory-db-path",
        dest="removed_memory_db_path",
        default=argparse.SUPPRESS,
        action=_RemovedMemoryDbPathAction,
        help=argparse.SUPPRESS,
    )
    g_mem.add_argument(
        "--native-memory-max-chars",
        type=int,
        default=240_000,
        help="Max total characters for native session memory capture (default: 240000, 0=disabled).",
    )
    g_mem.add_argument(
        "--native-memory-max-files",
        type=int,
        default=8,
        help="Max native session memory files to read per agent (default: 8).",
    )
    g_mem.add_argument(
        "--native-memory-max-chars-per-file",
        type=int,
        default=60_000,
        help="Max characters per native session memory file (default: 60000).",
    )
    g_mem.add_argument(
        "--disable-memory-mcp",
        action="store_true",
        default=False,
        help="Ablation: disable MCP knowledge tools in task containers (keep DB for discussion storage).",
    )
    g_mem.add_argument(
        "--no-memory",
        action="store_true",
        default=False,
        help=(
            "Disable agent knowledge tools, discussion phases, distillation, and "
            "seeding for clean solo baseline comparison. ARC MCP tools remain "
            "registered for ARC tasks. The authoritative knowledge DB remains "
            "enabled for attempts/resume state; the optional runtime audit DB "
            "remains enabled unless --no-runtime-db is set."
        ),
    )
    g_mem.add_argument(
        "--migrate-memory",
        metavar="OUTPUT_PATH",
        default=None,
        help="Migrate old 3-file memory layout to unified knowledge DB and exit.",
    )
    g_mem.add_argument(
        "--phase1-reflection-enabled",
        nargs="?",
        const=True,
        type=_parse_bool_flag,
        default=False,
        help=(
            "Enable Phase-1 self-reflection (Path a): after each scheduled "
            "non-strict-protocol task attempt, the host runs evaluator.evaluate "
            "and feeds the score back to the agent inside the same SDK session "
            "via a barrier protocol. The agent then writes a 3-5 sentence "
            "structured reflection (load-bearing assumption + proposed change + "
            "predicted outcome) which is stored at attempt.content.reflection "
            "for downstream per-task distillation. Costs one extra SDK turn "
            "per attempt; off by default. Accepts true/false/0/1/yes/no."
        ),
    )

    # ── Forum / Knowledge Transfer ──────────────────────────────────────────
    g_forum = p.add_argument_group("Forum / Knowledge Transfer")
    g_forum.add_argument(
        "--forum-rounds",
        dest="removed_forum_rounds",
        nargs="?",
        default=argparse.SUPPRESS,
        action=_RemovedForumFlagAction,
        help=argparse.SUPPRESS,
    )
    g_forum.add_argument(
        "--forum-timeout-sec",
        type=int,
        default=900,
        help="Timeout in seconds for the per-task discussion task (default: 900)",
    )
    g_forum.add_argument(
        "--forum-mode",
        dest="removed_forum_mode",
        nargs="?",
        default=argparse.SUPPRESS,
        action=_RemovedForumFlagAction,
        help=argparse.SUPPRESS,
    )
    g_forum.add_argument(
        "--forum-ablate-r3",
        dest="removed_forum_ablate_r3",
        nargs=0,
        default=argparse.SUPPRESS,
        action=_RemovedForumFlagAction,
        help=argparse.SUPPRESS,
    )
    g_forum.add_argument(
        "--improvement-strategy",
        choices=supported_strategies(include_aliases=True),
        default="knowledge",
        help="Self-improvement mechanism (forum/distill/seed phases). "
        "'knowledge' (default) is the standard loop; 'raw_attempts' skips forums + "
        "distillation. Registry-backed — see docs/improvement_strategies.md.",
    )
    # ── Three-phase generation loop flags (per-task forum / cross-task forum / distill) ──
    g_forum.add_argument(
        "--per-task-forum-rounds",
        type=int,
        default=1,
        help="Number of per-task forum dispatches per generation (0 disables phase 2).",
    )
    g_forum.add_argument(
        "--cross-task-forum-rounds",
        type=int,
        default=2,
        help="Number of cross-task forum dispatches per generation (0 disables phase 3).",
    )
    g_forum.add_argument(
        "--cross-task-forum-timeout-sec",
        type=int,
        default=900,
        help="Container timeout for cross-task forum tasks (default: 900).",
    )
    g_forum.add_argument(
        "--cross-task-shared-container",
        nargs="?",
        const=True,
        type=_parse_bool_flag,
        default=False,
        help=(
            "Phase 3 R0->R1 shared-container: keep cross-task forum round 0 and "
            "round 1 in the SAME Anthropic-Messages-API session per agent, "
            "coordinating the round 0 -> round 1 handoff via the host<->container "
            "barrier protocol (see src/ksi/runtime/barrier.py). When off (default), "
            "cross-task rounds dispatch as separate containers per agent. "
            "Tokens for round 1 are recorded under 'cross_task_forum_round_1' "
            "in token_phases. Accepts true/false/0/1/yes/no."
        ),
    )
    g_forum.add_argument(
        "--forum-early-exit",
        nargs="?",
        const=True,
        type=_parse_bool_flag,
        default=False,
        help="End discussion phases early when all expected agents have signalled "
        "done via forum_signal_done (default: false; accepts "
        "true/false/0/1/yes/no). The --forum-timeout-sec / "
        "--cross-task-forum-timeout-sec values remain the hard cap -- "
        "this only ends phases early, never late.",
    )
    g_forum.add_argument(
        "--forum-early-exit-poll-sec",
        type=float,
        default=3.0,
        help="Poll interval (seconds) for the discussion-phase early-exit watcher "
        "(default: 3.0). Lower values react faster but query the "
        "forum JSONL more often.",
    )
    g_forum.add_argument(
        "--forum-early-exit-quorum-pct",
        type=float,
        default=100.0,
        help="Percentage of expected agents that must signal done before the "
        "early-exit watcher fires (default: 100.0, i.e. every agent required). "
        "Lowering this lets the watcher "
        "cut off a straggler minority instead of waiting out the full "
        "--forum-timeout-sec / --cross-task-forum-timeout-sec backstop. Only "
        "takes effect when --forum-early-exit is enabled.",
    )
    g_forum.add_argument(
        "--forum-early-exit-quorum-grace-sec",
        type=float,
        default=0.0,
        help="Extra seconds the early-exit watcher waits after "
        "--forum-early-exit-quorum-pct is first reached, before cutting off "
        "the remaining stragglers (default: 0.0). Ignored when "
        "--forum-early-exit-quorum-pct is 100 (the all-required path never "
        "needs a grace window).",
    )
    g_forum.add_argument(
        "--distill-enabled",
        nargs="?",
        const=True,
        type=_parse_bool_flag,
        default=True,
        help="Toggle distillation phase (default: true; accepts true/false/0/1/yes/no). "
        "Bare --distill-enabled is equivalent to --distill-enabled true.",
    )
    g_forum.add_argument(
        "--distill-per-task-model",
        type=str,
        default=None,
        help="Optional model override for per-task distillation.",
    )
    g_forum.add_argument(
        "--distill-cross-task-model",
        type=str,
        default=None,
        help="Optional model override for cross-task distillation.",
    )
    g_forum.add_argument(
        "--cross-task-distill-target-conditioning",
        nargs="?",
        const=True,
        type=_parse_bool_flag,
        default=True,
        help="Condition cross-task distillation on the downstream task "
        "(default: true; accepts true/false/0/1/yes/no). When true, each "
        "downstream seed target gets its own cross-task bundle distilled "
        "with that task's prompt; false restores the single broadcast bundle. "
        "Bare flag = true.",
    )
    g_forum.add_argument(
        "--cross-task-distill-per-target-selection",
        nargs="?",
        const=True,
        type=_parse_bool_flag,
        default=False,
        help="Select cross-task forum posts PER downstream target by relevance "
        "(default: false; accepts true/false/0/1/yes/no; requires "
        "--cross-task-distill-target-conditioning). When false (default), the "
        "forum history is trimmed ONCE against the largest target so every "
        "target shares a byte-identical cache_prefix. When true, each target "
        "gets its own relevance-ranked post "
        "set, fixing the defect where every target but the "
        "largest is selected against the largest target's vocabulary — at the "
        "COST of defeating the cross-target prompt cache (each target sends a "
        "different post set, so every over-budget target pays a full cache "
        "write). Bare flag = true.",
    )
    g_forum.add_argument(
        "--abort-on-distill-stall",
        type=int,
        default=0,
        help="Abort the run after this many consecutive generations whose distillation "
        "was fully zeroed by failures — e.g. a sustained host->provider outage. "
        "0 = disabled (ERROR-log only, default).",
    )
    g_forum.add_argument(
        "--max-concurrent-forum-tasks",
        type=int,
        default=0,
        help="Max concurrent forum workers; 0 = follow --max-concurrent-tasks (default); explicit values override",
    )
    g_forum.add_argument(
        "--seed-bundle-path",
        default="",
        help="Path to an exported cross-task knowledge-bundle JSON to inject at seeding time. "
        "Injects the bundle into gen-1 agents' seed packages for knowledge transfer.",
    )
    g_forum.add_argument(
        "--seed-per-task-bundles-path",
        default="",
        help="Path to a per-task distilled knowledge snapshot JSON file. "
        "Injects task-specific distilled bundles into gen-1 agents based on assigned task_id.",
    )
    g_forum.add_argument(
        "--require-vector",
        action="store_true",
        default=False,
        help=(
            "Opt in to semantic vector search (sqlite-vec + embedding model). The default "
            "retrieval path is lexical FTS5; this flag enables vector search and fails fast "
            "unless sqlite-vec and the embedding model are available and embeddings are written."
        ),
    )
    g_forum.add_argument(
        "--memory-seed-raw-attempts",
        action="store_true",
        default=False,
        help="Ablation: build MEMORY.md from ONLY a minimal per-prior-attempt block "
        "containing generation, native_score, resolved status, and a 1000-char "
        "approach_excerpt (narration-stripped slice of model_output). Skips every "
        "model-generated reflection layer: distilled per-task / cross-task bundles, "
        "insight_bundle, related_summaries, condensed Insight reasoning, and the "
        "best-attempt / stagnation heuristics. Pair with --distill-enabled false "
        "to fully disable the distillation pipeline.",
    )
    # ── Classification ──────────────────────────────────────────────────────
    g_class = p.add_argument_group("Classification")
    g_class.add_argument(
        "--classify",
        action="store_true",
        default=False,
        help="Run LLM classification to assign categories to tasks before starting (swebench only). "
        "Results are cached to a JSON sidecar next to --tasks-path.",
    )
    g_class.add_argument(
        "--categories-json",
        default=None,
        help="Path to a pre-existing task_id→category JSON file. Overrides auto-computed sidecar path for --classify.",
    )

    # ── Output ──────────────────────────────────────────────────────────────
    g_out = p.add_argument_group("Output")
    g_out.add_argument("--experiment-name", default="ksi", help="Experiment name for logging")
    g_out.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume a previous experiment: reuse the existing run entry and seed "
        "best scores from prior generations. Without this flag, a colliding "
        "experiment name is auto-suffixed (e.g. myexp → myexp_2).",
    )
    g_out.add_argument("--output-json", default=None, help="Optional run artifact JSON path")
    g_out.add_argument(
        "--pretask-debug-json",
        default=None,
        help="Optional path to write pretask claim-phase debug artifacts "
        "(task summaries, per-agent claim responses, and resolved assignments).",
    )
    g_out.add_argument(
        "--provider-profile",
        default="",
        help="Path to provider config file in configs/ksi/*.env format. Required for provider-backed runtimes.",
    )

    return p


# Alias so tests and downstream callers can import the parser factory under
# a private name (the plan's Task 17 refers to this as _build_parser).
_build_parser = build_parser


def default_arg_namespace() -> argparse.Namespace:
    """Return a ``Namespace`` pre-populated with every CLI default.

    This lets non-CLI callers build registry components via
    :func:`ksi.build_evaluator` / :func:`ksi.build_runtime` without parsing
    ``argv`` or supplying the ``required=True`` flags (``--task-source`` /
    ``--tasks-path``) — which the evaluator/runtime factories never read.
    Defaults are read straight off the parser actions, so they cannot drift
    from the CLI.
    """
    parser = build_parser()
    namespace = argparse.Namespace()
    for action in parser._actions:
        if action.dest == argparse.SUPPRESS:
            continue
        # Match argparse's real default-application semantics: only set a
        # default when the attr is unset, so the FIRST-registered action's
        # default wins. Several flags share a dest (e.g. --drop-solved /
        # --no-drop-solved both write drop_solved); an unconditional setattr
        # would let a later action's default clobber an earlier one, diverging
        # from how parse_args() behaves.
        if not hasattr(namespace, action.dest):
            setattr(namespace, action.dest, action.default)
    return namespace


def _choose_runtime(args: argparse.Namespace, provider_env: dict[str, str] | None = None) -> RuntimeExecutor:
    base = get_runtime_spec(args.runtime).factory(args, provider_env)
    runtime_spec = resolve_source(getattr(args, "task_source", ""))
    if runtime_spec is not None and runtime_spec.delegates_runtime:
        keep_output = str(os.environ.get("KSI_TB2_KEEP_OUTPUT") or "").strip().lower() in {"1", "true", "yes"}
        return TerminalBench2Executor(
            agent_mode="ksi",
            env=provider_env or {},
            fallback_runtime=base,
            keep_container=keep_output,
            memory_seed_raw_attempts=bool(getattr(args, "memory_seed_raw_attempts", False)),
        )
    return base


def _choose_evaluator(args: argparse.Namespace) -> Evaluator:
    return get_evaluator_spec(args.evaluator).factory(args)


def _load_ids_file(path: str, *, flag: str = "--task-ids-file") -> list[str]:
    """Load task IDs from a JSON file.

    Accepts a bare list of strings, ``{"task_ids": [...]}``, or a task-map
    shape ``{"tasks": [{"task_id": ...}, ...]}``. Returns stripped, non-empty
    IDs in file order. ``flag`` names the CLI flag in error messages.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{flag} not found: {path}")
    with open(p) as f:
        ids = json.load(f)
    if isinstance(ids, dict):
        if isinstance(ids.get("task_ids"), list):
            ids = ids.get("task_ids")
        elif isinstance(ids.get("tasks"), list):
            ids = [
                str(item.get("task_id") or "").strip()
                for item in ids["tasks"]
                if isinstance(item, dict) and str(item.get("task_id") or "").strip()
            ]
        else:
            ids = None
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError(
            f"{flag} must contain either a JSON array of strings, "
            "a JSON object with a string-array field `task_ids`, or "
            "a JSON object with a `tasks` array of objects containing `task_id`: "
            f"{path}"
        )
    return [i.strip() for i in ids if i.strip()]


def _task_ids_from_csv(task_ids_csv: str | None) -> list[str]:
    return [x.strip() for x in task_ids_csv.split(",") if x.strip()] if task_ids_csv else []


def _normalize_task_map_selection(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Make ``--task-map-path`` the task selector, or reject conflicting selectors."""
    if not getattr(args, "task_map_path", None):
        return

    task_map_path = Path(args.task_map_path)
    if not task_map_path.is_file():
        parser.error(f"--task-map-path does not exist or is not a file: {task_map_path}")
    args.task_map_path = str(task_map_path)

    try:
        task_map_ids = _load_ids_file(str(task_map_path), flag="--task-map-path")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not task_map_ids:
        parser.error(f"--task-map-path did not contain any task IDs: {task_map_path}")

    explicit_ids = _task_ids_from_csv(getattr(args, "task_ids", None))
    if getattr(args, "task_ids_file", None):
        try:
            explicit_ids.extend(_load_ids_file(args.task_ids_file, flag="--task-ids-file"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))

    if explicit_ids:
        if explicit_ids != task_map_ids:
            parser.error(
                "--task-map-path task IDs differ from --task-ids/--task-ids-file; "
                "use the same manifest or omit the conflicting selector"
            )
        return

    args.task_ids_file = str(task_map_path)


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 hex digest of ``path`` (1 MiB chunks)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_UNPINNED_SWEBENCH_REVISION_MARKERS = {"", "none", "null", "<none>", "<null>", "<unpinned>", "unpinned"}


def _pinned_swebench_source_revision(payload: dict[str, object]) -> str | None:
    raw = payload.get("source_revision")
    if not isinstance(raw, str):
        return None
    revision = raw.strip()
    if revision.lower() in _UNPINNED_SWEBENCH_REVISION_MARKERS:
        return None
    return revision


def _verify_swebench_task_map_source(ids_file_path: str, tasks_path: Path, *, strict: bool = False) -> None:
    """Detect drift between a SWE-bench Pro task map's ``source_sha256`` and the
    sha256 of the resolved ``--tasks-path`` dataset file.

    The task map records the sha256 of the exact dataset file used to select the
    task membership (``benchmarks/scripts/dataprep/generate_swebench_pro_task_map.py``). If
    the dataset under ``--tasks-path`` has drifted (for example a re-export
    against a different upstream revision), the selected task ids may no longer
    map to the same rows and any scored result would be against a mismatched
    dataset.

    Enforcement policy:

    - By DEFAULT (``strict=False``), legacy maps with no pinned
      ``source_revision`` log a LOUD ``log.warning`` on mismatch and continue.
      The primary committed seed0 map was exported WITHOUT a pinned revision,
      but its ``source_sha256`` IS reproducible: re-exporting at the default pin
      yields a byte-identical file, so strict mode verifies and passes for it.
    - Maps with a non-empty ``source_revision`` are reproducible maps and are
      fail-closed by default. A mismatch raises ``SystemExit`` even when the
      CLI flag is omitted.
    - Under ``strict=True`` (``--strict-swebench-dataset-integrity`` or
      ``KSI_STRICT_SWEBENCH_DATASET_INTEGRITY=1``) a mismatch raises
      ``SystemExit`` for every map, and also refuses maps that lack
      ``source_sha256`` because there is no digest to verify.

    Task maps with no recorded ``source_sha256`` (older or unpinned exports) are
    not checked unless strict mode is requested or the map records a pinned
    ``source_revision``.
    """
    p = Path(ids_file_path)
    if not p.exists():
        return
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return
    if not isinstance(payload, dict):
        return
    source_revision = _pinned_swebench_source_revision(payload)
    effective_strict = strict or source_revision is not None
    expected = payload.get("source_sha256")
    if not isinstance(expected, str) or not expected.strip():
        if effective_strict:
            revision_note = f" source_revision={source_revision}" if source_revision else ""
            raise SystemExit(
                "SWE-bench Pro dataset integrity check failed: task map "
                f"{ids_file_path}{revision_note} does not record source_sha256, "
                "so the --tasks-path dataset cannot be verified. Regenerate the "
                "map with:\n"
                "  uv run python benchmarks/scripts/dataprep/generate_swebench_pro_task_map.py "
                f"--dataset-path {tasks_path} --source-revision {SWEBENCH_PRO_DATASET_REVISION} "
                "--selection-name <name> --seed <n> --count <n> --output " + ids_file_path
            )
        return
    expected = expected.strip()

    tasks_path = Path(tasks_path)
    if not tasks_path.is_file():
        return
    actual = _sha256_file(tasks_path)
    if actual == expected:
        return
    message = (
        "SWE-bench Pro dataset integrity check failed: task map "
        f"{ids_file_path} records source_sha256={expected} but the dataset at "
        f"--tasks-path {tasks_path} hashes to {actual}. The dataset has drifted "
        "from the file used to select the task membership; running would score "
        "against a mismatched dataset. Re-export the pinned dataset and "
        "regenerate the task map:\n"
        "  uv run python benchmarks/scripts/dataprep/export_swebench_pro_dataset.py "
        f"--split test --format jsonl --revision {source_revision or SWEBENCH_PRO_DATASET_REVISION} "
        f"--output {tasks_path}\n"
        "  uv run python benchmarks/scripts/dataprep/generate_swebench_pro_task_map.py "
        f"--dataset-path {tasks_path} --source-revision "
        f"{source_revision or SWEBENCH_PRO_DATASET_REVISION} --selection-name <name> "
        "--seed <n> --count <n> --output " + ids_file_path
    )
    if effective_strict:
        raise SystemExit(message)
    log.warning(
        "%s\nContinuing anyway because --strict-swebench-dataset-integrity is "
        "not set (default warn-mode). Pass --strict-swebench-dataset-integrity "
        "(or set KSI_STRICT_SWEBENCH_DATASET_INTEGRITY=1) to fail closed on "
        "this mismatch.",
        message,
    )


def _resolve_holdout_ids(
    holdout_csv: str | None,
    holdout_file: str | None,
    *,
    training_ids: set[str],
) -> list[str]:
    """Resolve the hold-out task id list from CLI inputs.

    Merges ``--holdout-task-ids`` (CSV) and ``--holdout-task-ids-file``
    (same JSON shapes as ``--task-ids-file``), deduplicates preserving
    order, and raises ``ValueError`` when any hold-out id is also a
    training task id — the two sets must be disjoint for the transfer
    probe to be meaningful.
    """
    want: list[str] = []
    if holdout_csv:
        want.extend(x.strip() for x in holdout_csv.split(",") if x.strip())
    if holdout_file:
        want.extend(_load_ids_file(holdout_file, flag="--holdout-task-ids-file"))
    seen: set[str] = set()
    out: list[str] = []
    for holdout_id in want:
        if holdout_id in seen:
            continue
        seen.add(holdout_id)
        out.append(holdout_id)
    overlap = [holdout_id for holdout_id in out if holdout_id in training_ids]
    if overlap:
        raise ValueError(
            "hold-out task ids must be disjoint from the training task ids; overlapping id(s): " + ", ".join(overlap)
        )
    return out


def _select_holdout_tasks(all_tasks: list[TaskSpec], holdout_ids: list[str]) -> list[TaskSpec]:
    """Select hold-out TaskSpecs (in ``holdout_ids`` order) from the full
    loaded task list. Raises ``ValueError`` when an id is missing from the
    source."""
    if not holdout_ids:
        return []
    by_id = {t.id: t for t in all_tasks}
    missing_ids = [holdout_id for holdout_id in holdout_ids if holdout_id not in by_id]
    if missing_ids:
        raise ValueError(
            "--holdout-task-ids requested id(s) missing from the loaded task source: " + ", ".join(missing_ids)
        )
    return [by_id[holdout_id] for holdout_id in holdout_ids]


def _filter_tasks(
    tasks: list[TaskSpec],
    task_ids_csv: str | None,
    max_tasks: int,
    task_ids_file: str | None = None,
    *,
    strict: bool = False,
) -> list[TaskSpec]:
    """Filter tasks by explicit IDs while preserving requested order.

    By default, this helper keeps legacy behavior and warns on missing IDs.
    Set ``strict=True`` to fail fast when:

    - a requested task ID is missing
    - the manifest contains duplicate requested IDs
    """
    out = tasks
    # Merge IDs from --task-ids and --task-ids-file preserving request order.
    want_order: list[str] = []
    if task_ids_csv:
        want_order.extend(x.strip() for x in task_ids_csv.split(",") if x.strip())
    if task_ids_file:
        want_order.extend(_load_ids_file(task_ids_file))

    # Normalize requested IDs, preserving order. In strict mode, duplicate IDs
    # in the request are treated as manifest corruption.
    if strict:
        seen_seen: set[str] = set()
        deduped = []
        for requested_id in want_order:
            if requested_id in seen_seen:
                raise ValueError(f"duplicate task id requested in --task-ids / --task-ids-file: {requested_id}")
            seen_seen.add(requested_id)
            deduped.append(requested_id)
        want_order = deduped
    else:
        # Legacy behavior: deduplicate silently but preserve the first occurrence.
        seen_seen = set()
        deduped = []
        for requested_id in want_order:
            if requested_id in seen_seen:
                continue
            seen_seen.add(requested_id)
            deduped.append(requested_id)
        want_order = deduped

    if not want_order:
        if max_tasks and max_tasks > 0:
            return out[:max_tasks]
        return out

    if strict:
        requested = set(want_order)
        counts = Counter(t.id for t in out if t.id in requested)
        duplicate_loaded = sorted(task_id for task_id, count in counts.items() if count > 1)
        if duplicate_loaded:
            raise ValueError(
                "duplicate loaded task id(s) requested by --task-ids / --task-ids-file: " + ", ".join(duplicate_loaded)
            )

    task_by_id = {t.id: t for t in out}
    missing = [task_id for task_id in want_order if task_id not in task_by_id]
    if missing:
        if strict:
            raise ValueError(f"requested task IDs not found in loaded tasks: {', '.join(sorted(missing))}")
        log.warning(
            "--task-ids filter requested %d ID(s) not found in loaded tasks: %s",
            len(missing),
            ", ".join(sorted(missing)),
        )
    if want_order:
        out = [task_by_id[task_id] for task_id in want_order if task_id in task_by_id]
    if max_tasks and max_tasks > 0:
        out = out[:max_tasks]
    return out


def _sanitize_experiment_name(name: str) -> str:
    raw = (name or "").strip() or "ksi"
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in raw)
    safe = safe.strip("._-") or "ksi"
    return safe


def _ensure_trace_dir(experiment_name: str) -> str:
    """Ensure trace-event JSONL output directory exists.

    If KSI_TRACE_DIR is already set, preserve it.
    Otherwise default to analysis/traces/<experiment_name>/.
    """
    configured = str((os.environ.get("KSI_TRACE_DIR", "") or "").strip())
    if configured:
        trace_root = Path(configured).expanduser().resolve()
    else:
        exp = _sanitize_experiment_name(experiment_name)
        trace_root = (Path("analysis") / "traces" / exp).resolve()
        os.environ["KSI_TRACE_DIR"] = str(trace_root)
        log.info("KSI_TRACE_DIR not set; defaulting to %s", trace_root)
    trace_root.mkdir(parents=True, exist_ok=True)
    return str(trace_root)


def _checkpoint_and_move_sqlite(src: Path, dst: Path) -> None:
    """Move a SQLite DB file (and its WAL/SHM sidecars) ``src`` -> ``dst``.

    Folds any pending WAL frames into the main file first (``wal_checkpoint``
    TRUNCATE) so the main file alone is self-consistent — a crash mid-move then
    cannot strand un-checkpointed transactions in an orphaned ``-wal``. The
    ``-wal``/``-shm`` sidecars travel with the DB regardless, so even if the
    checkpoint is skipped (e.g. the file is not in WAL mode or is momentarily
    busy) the trio stays consistent in the destination dir.
    """
    import shutil
    import sqlite3

    try:
        conn = sqlite3.connect(str(src))
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except sqlite3.Error:
        pass  # not WAL / busy / unreadable — move whatever exists as-is
    shutil.move(str(src), str(dst))
    for suffix in ("-wal", "-shm"):
        side = src.with_name(src.name + suffix)
        if side.exists():
            shutil.move(str(side), str(dst.with_name(dst.name + suffix)))


def _migrate_legacy_flat_db_to_subdir(legacy_db: Path, target_db: Path) -> None:
    """Move a legacy flat knowledge DB into its isolated subdir.

    The pre-M1 layout placed the knowledge DB (and its runtime-audit sibling)
    directly under the shared ``runtime_state/knowledge/`` directory. The
    container memory mount is a *directory* mount (load-bearing: WAL mode needs
    host and container to share the ``-wal``/``-shm`` wal-index across the
    bind), so resolving a legacy DB in place would mount that shared flat dir —
    every experiment's DB — read-write during forum phases, re-exposing the
    cross-experiment poisoning per-experiment isolation closes. Moving the DB into a private per-
    experiment subdir restores the directory-mount semantics while isolating
    siblings. The runtime-audit sibling is moved too so the run continues its
    prior audit history (it is best-effort; absence is fine).
    """
    target_dir = target_db.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    _checkpoint_and_move_sqlite(legacy_db, target_db)
    legacy_runtime = Path(derive_runtime_sibling(str(legacy_db)))
    if legacy_runtime.exists():
        _checkpoint_and_move_sqlite(legacy_runtime, target_dir / legacy_runtime.name)


def _resolve_knowledge_db_path(raw_path: str, experiment_name: str) -> str:
    """PURE: compute the knowledge DB path to use — no filesystem mutation.

    An explicit ``--knowledge-db-path`` is honored verbatim; otherwise the
    isolated per-experiment subdir path. This does NOT migrate a
    legacy flat DB and does NOT create directories — callers that use the
    default path must first run :func:`_migrate_legacy_flat_knowledge_db`
    (see :func:`_prepare_knowledge_db_path`, the orchestrator that does both).
    Keeping resolution side-effect-free makes it safe to call in dry runs and
    trivially testable without touching disk.
    """
    configured = str((raw_path or "").strip())
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(default_knowledge_db_path(experiment_name).resolve())


def _migrate_legacy_flat_knowledge_db(experiment_name: str) -> None:
    """Explicit, destructive legacy-flat -> subdir migration step.

    A pre-subdir experiment lives at the flat path
    ``runtime_state/knowledge/<exp>_knowledge.sqlite``, whose parent dir is
    shared by every experiment. If the new per-experiment subdir DB does not
    exist yet but the legacy flat one does, MOVE the flat DB (+ WAL/SHM sidecars
    and the runtime sibling) into the isolated subdir on this first resume.
    Resolving the flat path in place would instead mount the shared flat dir
    read-write during forum phases, re-exposing sibling experiments' DBs (the
    cross-experiment poisoning per-experiment isolation closes).

    Idempotent: a no-op (beyond ensuring the subdir exists) when the subdir DB
    already exists or no legacy flat DB is present. Raises ``RuntimeError``
    (fail closed) if the main DB cannot be moved out of the shared flat dir.
    Only meaningful for the default-path case; an explicit ``--knowledge-db-path``
    bypasses migration entirely.
    """
    default_path = default_knowledge_db_path(experiment_name).resolve()
    if not default_path.exists():
        legacy_path = legacy_flat_knowledge_db_path(experiment_name).resolve()
        if legacy_path.exists():
            try:
                _migrate_legacy_flat_db_to_subdir(legacy_path, default_path)
            except Exception as exc:  # defensive: never crash a resume on migration
                # The main knowledge DB is moved FIRST (the runtime-audit sibling
                # follows, best-effort). If the failure struck AFTER the main DB
                # already landed in the subdir, the flat path no longer holds the
                # data — resuming there would open a fresh empty DB and silently
                # reset the cursor/best-scores. Detect that and resume against the
                # subdir instead (the move is what matters; an orphaned sibling is
                # harmless audit history).
                if default_path.exists() and not legacy_path.exists():
                    log.warning(
                        "legacy flat knowledge DB %s was migrated into "
                        "isolated subdir %s but a later migration step failed (%s); "
                        "resuming against the subdir — the runtime-audit sibling may "
                        "be left at the old flat path",
                        legacy_path,
                        default_path,
                        exc,
                    )
                    default_path.parent.mkdir(parents=True, exist_ok=True)
                    return
                # Fail closed: the main DB is still at the shared flat path, so
                # resolving it in place would bind-mount runtime_state/knowledge
                # read-write during forum phases and re-expose sibling
                # experiments' DBs — exactly the cross-experiment poisoning
                # per-experiment isolation closes. Refuse the resume loudly rather than silently mounting
                # the shared dir. The operator recovers deterministically by
                # moving the DB into the subdir manually, or by passing an
                # explicit --knowledge-db-path (honored verbatim, bypassing
                # migration).
                raise RuntimeError(
                    f"could not migrate legacy flat knowledge DB {legacy_path} "
                    f"into isolated subdir {default_path.parent} ({exc}); refusing to "
                    f"resume against the shared runtime_state/knowledge mount, which "
                    f"would expose sibling experiments' DBs read-write during forum "
                    f"phases. Move the DB into {default_path} manually, or pass an "
                    f"explicit --knowledge-db-path."
                ) from exc
    default_path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_knowledge_db_path(raw_path: str, experiment_name: str) -> str:
    """Resolve the knowledge DB path, migrating a legacy flat DB first when the
    default (per-experiment subdir) layout is in use.

    Thin orchestrator over the pure :func:`_resolve_knowledge_db_path` and the
    explicit :func:`_migrate_legacy_flat_knowledge_db`. An explicit
    ``--knowledge-db-path`` skips migration and is returned verbatim.
    """
    if not str((raw_path or "").strip()):
        _migrate_legacy_flat_knowledge_db(experiment_name)
    return _resolve_knowledge_db_path(raw_path, experiment_name)


def _resolve_runtime_db_path(
    raw_path: str,
    experiment_name: str,
    *,
    knowledge_db_path: str = "",
    disabled: bool = False,
) -> str:
    if disabled:
        return ""
    configured = str((raw_path or "").strip())
    if configured:
        path = Path(configured).expanduser().resolve()
    elif knowledge_db_path:
        path = Path(derive_runtime_sibling(knowledge_db_path))
    else:
        path = default_runtime_db_path(experiment_name).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _validate_db_paths(*, runtime_db_path: str, knowledge_db_path: str) -> None:
    if not runtime_db_path or not knowledge_db_path:
        return
    runtime_resolved = Path(runtime_db_path).resolve()
    knowledge_resolved = Path(knowledge_db_path).resolve()
    if runtime_resolved == knowledge_resolved:
        raise ValueError(
            f"--runtime-db-path and --knowledge-db-path must differ "
            f"(both resolve to {runtime_resolved}). "
            f"The two stores have incompatible schemas and separate "
            f"lock registries; sharing a single file recreates the "
            f"AB-BA lock hazard."
        )
    if runtime_resolved.name == knowledge_resolved.name:
        raise ValueError(
            f"--runtime-db-path and --knowledge-db-path must not share a "
            f"filename ({runtime_resolved.name}), even in different "
            f"directories: both are bind-mounted into the container's "
            f"/app/memory-db by basename, so a collision makes Docker "
            f'refuse to start the container ("Duplicate mount point").'
        )


# Backwards-compatible alias; canonical implementation lives in ksi.layout.
_derive_legacy_sibling = derive_legacy_sibling


def _run_memory_migration(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Run legacy 3-file → unified knowledge DB migration, then exit."""
    from .memory.knowledge_store import KnowledgeStore

    legacy_runtime_db_path = args.runtime_db_path
    if not legacy_runtime_db_path:
        parser.error("--migrate-memory requires --runtime-db-path to derive source files")

    forum_db_path = _derive_legacy_sibling(legacy_runtime_db_path, "forum")
    docs_db_path = _derive_legacy_sibling(legacy_runtime_db_path, "task_docs")
    output_path = args.migrate_memory
    experiment = args.experiment_name or "default"

    log.info(
        "[MIGRATE] Starting migration: memory=%s forum=%s docs=%s → %s",
        legacy_runtime_db_path,
        forum_db_path,
        docs_db_path,
        output_path,
    )

    try:
        count = KnowledgeStore.migrate_from_legacy(
            memory_db_path=legacy_runtime_db_path,
            forum_db_path=forum_db_path,
            docs_db_path=docs_db_path,
            output_path=output_path,
            experiment=experiment,
        )
        print(f"Migration complete: {count} entries migrated to {output_path}")
    except Exception as exc:
        log.error("[MIGRATE] Migration failed: %s", exc, exc_info=True)
        print(f"Migration failed: {exc}", file=sys.stderr)
        sys.exit(1)


def _evaluator_source_map() -> dict[str, str]:
    """Map each source's default evaluator back to its canonical source name.

    Derived from the registry so the M14 warn-map stays in sync with the
    per-source defaults. Built over canonical specs only (aliases share a spec).
    """
    from .tasks.registry import REGISTRY

    seen: dict[str, str] = {}
    for spec in REGISTRY.values():
        # default "none" is the generic no-op evaluator; it is not source-specific.
        if spec.default_evaluator and spec.default_evaluator != "none":
            seen.setdefault(spec.default_evaluator, spec.name)
    return seen


def _normalize_evaluator_for_task_source(
    evaluator: str | None,
    *,
    task_source: str,
) -> str:
    """Resolve the effective evaluator, honoring an explicit --evaluator choice.

    ``evaluator is None`` means the flag was omitted: fall back to the task
    source's registered default evaluator (or the historical ``swebench_pro``
    default when the source is unknown). Any explicit value — including one
    equal to the historical default ``swebench_pro`` — is preserved verbatim,
    so ``--evaluator swebench_pro --task-source arc`` is no longer silently
    rewritten to ``arc_session``.
    """
    if evaluator is not None:
        return evaluator
    spec = resolve_source(task_source)
    if spec is not None and spec.default_evaluator:
        return spec.default_evaluator
    return "swebench_pro"


def _runtime_container_name_prefix(experiment_name: str) -> str:
    """Compute the Docker container name prefix used for this experiment.

    Mirrors runtime_runner/src/container_runner.ts:
        safeName = workspaceRuntime.folder.replace(/[^a-zA-Z0-9-]/g, '-')
        containerName = `ksi-runtime-${safeName}-${Date.now()}`
    and layout.py:task_workspace_key which produces folders like
        task__{experiment_part}__{task_part}__{digest}
    so the shared per-experiment prefix is:
        ksi-runtime-task--{js_safe_experiment_part}--
    """
    experiment_part = sanitize_key(experiment_name, fallback="default", max_len=24)
    js_safe_exp = re.sub(r"[^a-zA-Z0-9-]", "-", experiment_part)
    return f"ksi-runtime-task--{js_safe_exp}--"


# Module-level filter set once args are parsed in main(). atexit/signal handlers
# read it so cleanup is scoped to THIS experiment and never cross-kills containers
# belonging to other runs on the same host.
_container_name_prefix: str | None = None


def _set_container_name_prefix(experiment_name: str) -> None:
    """Update the module-level prefix used by atexit/signal cleanup.

    Called from main() twice: once after CLI arg parsing (initial best guess)
    and again after the orchestrator resolves the final experiment name --
    the engine may auto-suffix on DB name collisions (see engine.py around
    lines 400-408). Without the second call, atexit cleanup targets a stale
    prefix and either misses this run's containers or cross-kills a sibling
    experiment's.
    """
    global _container_name_prefix
    _container_name_prefix = _runtime_container_name_prefix(experiment_name)


def _cleanup_containers() -> None:
    """Stop Docker containers whose names start with the current experiment prefix.

    Until main() sets _container_name_prefix, this is a no-op — better to leave
    containers running than to cross-kill a sibling experiment's workers.
    """
    prefix = _container_name_prefix
    if not prefix:
        return
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name={prefix}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        container_ids = result.stdout.strip().split()
        if container_ids:
            log.info(
                "Stopping %d container(s) matching %s...",
                len(container_ids),
                prefix,
            )
            subprocess.run(
                ["docker", "stop", *container_ids],
                capture_output=True,
                timeout=30,
            )
    except Exception:
        pass  # best-effort cleanup


def _signal_handler(signum: int, frame: object) -> None:
    """Handle SIGINT/SIGTERM by cleaning up containers and exiting."""
    log.warning("Received signal %d, cleaning up containers...", signum)
    _cleanup_containers()
    sys.exit(128 + signum)


def _ksi_code_commit(repo_root: Path) -> str:
    """Resolve KSI's own git commit SHA for provenance stamping.

    Best-effort: returns "unknown" when the repo root has no ``.git`` (e.g. a
    packaged install) or ``git rev-parse`` fails, rather than raising.

    A ``+dirty`` suffix is appended when the working tree has uncommitted
    **tracked** changes, so a run whose code differs from ``HEAD`` isn't recorded
    as an exact, checked-out artifact (checking out the bare SHA later would not
    reproduce it) — and a dirty run forms its own ``code_commit_provenance`` arm
    instead of pooling with the clean commit. Untracked files are ignored to
    avoid false positives from stray local artifacts (mirrors
    ``git describe --dirty``).
    """
    if not (repo_root / ".git").exists():
        return "unknown"
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    sha = proc.stdout.strip()
    if not sha:
        return "unknown"
    # `git diff --quiet HEAD` exits 1 iff tracked content differs from HEAD
    # (staged or unstaged); 0 when clean; other codes on error → treat as clean.
    dirty = (
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
        ).returncode
        == 1
    )
    return f"{sha}+dirty" if dirty else sha


def _enable_unbuffered_stdout() -> None:
    """Make stdout/stderr line-buffered on the CLI path.

    Without this, a process whose stdout is redirected to a file (the common
    campaign case) switches to full buffering and the log stays empty until
    exit — looking identical to a hang. Prefer this over requiring callers to
    export ``PYTHONUNBUFFERED``.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            # Non-TextIOWrapper streams (e.g. captured under pytest) may lack
            # reconfigure; best-effort only.
            pass


def _validate_and_normalize_args(args, parser: argparse.ArgumentParser) -> None:
    """Validate scalar CLI args and apply cross-flag normalization.

    Split out of ``main()``: raises via ``parser.error`` on out-of-range scalar
    bounds, applies the ``--no-memory`` side effects (zeroing forum rounds,
    disabling distill + memory MCP), and logs the mismatched-evaluator /
    unsupported-``--classify`` warnings. Mutates ``args`` in place (the
    ``--no-memory`` normalization) exactly as the inline block it replaces.
    """
    source_spec = resolve_source(getattr(args, "task_source", ""))
    if source_spec is not None:
        args.task_source = source_spec.name

    if args.arc_max_trials < 1:
        parser.error("--arc-max-trials must be >= 1")
    if args.swebench_timeout_sec < 1:
        parser.error("--swebench-timeout-sec must be >= 1")
    if args.swebench_harness_grace_sec < 0:
        parser.error("--swebench-harness-grace-sec must be >= 0")
    if args.polyglot_test_feedback_max_lines < 0:
        parser.error("--polyglot-test-feedback-max-lines must be >= 0")
    # Runtime hard-cap policy for --runtime-timeout-sec (default None = omitted).
    # The value controls the container runner's hard-kill / idle-reaper timer:
    #   * negative -> disable the hard cap (per-task timeout is the sole bound)
    #   * 0        -> keep the 1800s hard container safety cap
    #   * positive -> cap at that many seconds
    #
    # TB2 is a special case: the per-task task.toml [agent].timeout_sec is the
    # AUTHORITATIVE wall-time bound (Harbor parity; e.g. build-pov-ray = 12000s),
    # enforced by the native trial loop's own host-side deadline. A KSI-side
    # hard cap could only truncate a task below its official budget, so the TB2
    # timeout is NOT user-configurable: any non-negative value is rejected and
    # omission defaults to no hard cap.
    if args.task_source == "terminal_bench_2":
        if args.runtime_timeout_sec is None:
            args.runtime_timeout_sec = -1
        elif args.runtime_timeout_sec >= 0:
            parser.error(
                "--runtime-timeout-sec is not configurable for --task-source "
                "terminal_bench_2: the per-task task.toml [agent].timeout_sec is "
                "the authoritative wall-time bound (Harbor parity). Omit the flag "
                "to use no hard container cap (the per-task timeout binds); "
                f"got {args.runtime_timeout_sec}."
            )
        # A negative value is the explicit no-cap opt-in — equivalent to
        # omission — so leave it as-is.
    else:
        if args.runtime_timeout_sec is None:
            args.runtime_timeout_sec = _resolve_runtime_timeout_default()
        elif args.runtime_timeout_sec < 0:
            # A negative value disables the hard-kill / idle-reaper timer. Safe
            # ONLY for TB2, whose native trial loop enforces its own host-side
            # deadline. Every other task source runs through the generic
            # container_runner.ts path with no such backstop, so a negative
            # value there would leave a hung/idle container unbounded. Reject it.
            parser.error(
                "--runtime-timeout-sec may only be negative (no hard container cap) "
                f"for --task-source terminal_bench_2, not {args.task_source!r}. "
                "Use 0 (or omit) to keep the 1800s hard container safety cap."
            )
    # --no-memory disables agent knowledge access/discussion/distillation/seeding, but
    # keeps the authoritative knowledge DB available for attempts/resume state.
    if args.no_memory:
        args.per_task_forum_rounds = 0
        args.cross_task_forum_rounds = 0
        args.distill_enabled = False
        args.disable_memory_mcp = True
        log.info(
            "Agent-facing memory disabled (--no-memory): knowledge DB kept at %s; "
            "per_task_forum_rounds, cross_task_forum_rounds set to 0; "
            "distill_enabled=False; memory MCP disabled",
            args.knowledge_db_path,
        )

    # M14: Warn on mismatched evaluator / task-source combinations. The
    # evaluator->source map is derived from the registry's default_evaluator
    # (one source of truth).
    _normalize_task_map_selection(args, parser)

    expected_source = _evaluator_source_map().get(args.evaluator)
    if expected_source and args.task_source != expected_source:
        log.warning(
            "Evaluator %r is designed for task source %r, but got %r",
            args.evaluator,
            expected_source,
            args.task_source,
        )

    # M18: Warn when --classify is used with a task source that does not support
    # category classification (only swebench_pro does today).
    classify_spec = source_spec
    if getattr(args, "classify", False) and not (classify_spec and classify_spec.supports_classification):
        log.warning("--classify is only supported for swebench_pro task source; ignoring")

    # M19: Warn that --no-drop-solved retains solved tasks across generations so
    # per-task answers can carry forward. Scoped to registry upstream-strict
    # published benchmarks (derived, not a parallel hardcoded list) so a
    # non-benchmark source does not spuriously warn.
    canonical_task_source = classify_spec.name if classify_spec else str(args.task_source)
    if args.drop_solved is False and canonical_task_source in _PUBLISHED_BENCHMARK_TASK_SOURCES:
        log.warning(
            "--no-drop-solved is enabled for benchmark task source %r; solved tasks will keep reappearing in later "
            "generations, so per-task answers can carry forward. Disclose this setting for published benchmark "
            "comparisons or leave --drop-solved enabled.",
            canonical_task_source,
        )


def _build_generation_config(
    args,
    *,
    num_agents: int,
    holdout_ids,
    model: str,
    code_commit: str = "",
    model_provider: str = "",
    scoring_mode: str = "",
    config_json: str = "",
) -> GenerationConfig:
    """Assemble the engine's :class:`GenerationConfig` from parsed CLI args.

    Split out of ``main()`` so the ~35-kwarg constructor (a merge-conflict
    magnet) lives in one focused, testable place. Purely maps validated/
    normalized ``args`` plus a few derived values (agent count, hold-out ids,
    resolved provider model) onto config fields — no side effects.
    """
    return GenerationConfig(
        code_commit=code_commit,
        model_provider=model_provider,
        scoring_mode=scoring_mode,
        config_json=config_json,
        num_generations=args.generations,
        num_agents=num_agents,
        forum_timeout_sec=args.forum_timeout_sec,
        max_concurrent_tasks=args.max_concurrent_tasks,
        max_concurrent_forum_tasks=args.max_concurrent_forum_tasks,
        max_task_retries=args.max_task_retries,
        drop_solved=args.drop_solved,
        solved_threshold=args.solved_threshold,
        knowledge_db_path=args.knowledge_db_path,
        runtime_db_path=args.runtime_db_path,
        experiment_name=args.experiment_name,
        native_memory_max_chars=args.native_memory_max_chars,
        native_memory_max_files=args.native_memory_max_files,
        native_memory_max_chars_per_file=args.native_memory_max_chars_per_file,
        seed_bundle_path=getattr(args, "seed_bundle_path", "") or "",
        seed_per_task_bundles_path=getattr(args, "seed_per_task_bundles_path", "") or "",
        disable_memory_mcp=args.disable_memory_mcp or args.no_memory,
        resume=args.resume,
        model=model,
        no_memory=args.no_memory,
        # Three-phase generation loop fields. These are now declared on
        # GenerationConfig so a directly-constructed config carries
        # the same defaults; the engine reads them as real attributes.
        per_task_forum_rounds=args.per_task_forum_rounds,
        cross_task_forum_rounds=args.cross_task_forum_rounds,
        cross_task_forum_timeout_sec=args.cross_task_forum_timeout_sec,
        cross_task_shared_container=bool(getattr(args, "cross_task_shared_container", False)),
        distill_enabled=args.distill_enabled,
        cross_task_distill_target_conditioning=bool(getattr(args, "cross_task_distill_target_conditioning", True)),
        cross_task_distill_per_target_selection=bool(getattr(args, "cross_task_distill_per_target_selection", False)),
        distill_per_task_model=args.distill_per_task_model,
        distill_cross_task_model=args.distill_cross_task_model,
        abort_on_distill_stall=args.abort_on_distill_stall,
        forum_early_exit=bool(args.forum_early_exit),
        forum_early_exit_poll_sec=float(args.forum_early_exit_poll_sec),
        forum_early_exit_quorum_pct=float(args.forum_early_exit_quorum_pct),
        forum_early_exit_quorum_grace_sec=float(args.forum_early_exit_quorum_grace_sec),
        require_vector=bool(args.require_vector),
        holdout_task_ids=holdout_ids,
    )


def main(argv: list[str] | None = None) -> int:
    _enable_unbuffered_stdout()
    configure_logging()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup_containers)
    parser = build_parser()
    args = parser.parse_args(argv)
    # Apply the --log-level flag now that args are parsed. Precedence:
    # explicit flag > KSI_LOG_LEVEL env > default INFO. When the flag is
    # absent (None), configure_logging falls back to env/default.
    configure_logging(level=args.log_level)
    assert_no_removed_channel_env()
    # Seed Python's `random` so host-side retry-backoff jitter is deterministic
    # for a fixed --seed. This does NOT make runs reproducible — no provider
    # receives a usable seed (see the --seed help text).
    if getattr(args, "seed", None) is not None:
        import random as _stdlib_random

        _stdlib_random.seed(int(args.seed))
    # Scope container cleanup to this experiment before any container can start.
    global _container_name_prefix
    _container_name_prefix = _runtime_container_name_prefix(args.experiment_name)
    evaluator_omitted = args.evaluator is None
    resolved_evaluator = _normalize_evaluator_for_task_source(
        args.evaluator,
        task_source=args.task_source,
    )
    if evaluator_omitted:
        log.info(
            "No evaluator override provided; using %r for task source %r",
            resolved_evaluator,
            args.task_source,
        )
    args.evaluator = resolved_evaluator
    args.knowledge_db_path = _prepare_knowledge_db_path(args.knowledge_db_path, args.experiment_name)
    args.runtime_db_path = _resolve_runtime_db_path(
        args.runtime_db_path,
        args.experiment_name,
        knowledge_db_path=args.knowledge_db_path,
        disabled=args.no_runtime_db,
    )
    _validate_db_paths(
        runtime_db_path=args.runtime_db_path,
        knowledge_db_path=args.knowledge_db_path,
    )
    _ensure_trace_dir(args.experiment_name)

    # --migrate-memory: migrate old 3-file layout and exit early.
    if getattr(args, "migrate_memory", None):
        _run_memory_migration(args, parser)
        return 0

    _validate_and_normalize_args(args, parser)

    tasks_path = Path(args.tasks_path)
    evals_path = Path(args.evals_path) if args.evals_path else None
    # Early registry validation: resolve the source (canonical or alias) and
    # raise a helpful, single-source-of-truth error for anything unknown before
    # any per-source --tasks-path format checks run.
    try:
        source_spec = get_spec(args.task_source)
    except ValueError as exc:
        parser.error(str(exc))
    # Per-source --tasks-path validation is registered on the spec
    # (ksi.tasks.path_validation); a source without a validator is unsupported.
    path_validator = source_spec.validate_tasks_path
    if path_validator is None:
        parser.error(
            f"task source {args.task_source!r} is registered but has no "
            "validate_tasks_path callback; set one when calling "
            "register_task_source() (see docs/adding_a_benchmark.md)"
        )
    else:
        path_error = path_validator(tasks_path, evals_path=evals_path)
        if path_error:
            parser.error(path_error)
    if args.evaluator == "swebench_pro":
        raw_sample_path = args.swebench_pro_raw_sample_path.strip()
        if not raw_sample_path and tasks_path.suffix.lower() in {".csv", ".jsonl"}:
            raw_sample_path = str(tasks_path)
            args.swebench_pro_raw_sample_path = raw_sample_path
        if not raw_sample_path:
            parser.error(
                "--evaluator swebench_pro requires --swebench-pro-raw-sample-path "
                "unless --tasks-path is already a csv/jsonl raw sample file"
            )
        raw_sample = Path(raw_sample_path)
        if not raw_sample.exists():
            parser.error(f"--swebench-pro-raw-sample-path does not exist: {raw_sample}")
        if raw_sample.suffix.lower() not in {".csv", ".jsonl"}:
            parser.error(f"--swebench-pro-raw-sample-path must be .csv or .jsonl: {raw_sample}")
    # SWE-bench Pro dataset-integrity tripwire: when a task map
    # (--task-ids-file) records a source_sha256, refuse to run if the
    # --tasks-path dataset has drifted from the file used to select the task
    # membership. Only enforced for task-source swebench_pro with both flags set.
    if source_spec.name == "swebench_pro" and getattr(args, "task_ids_file", None):
        strict_integrity = (
            bool(getattr(args, "strict_swebench_dataset_integrity", False))
            or os.environ.get("KSI_STRICT_SWEBENCH_DATASET_INTEGRITY") == "1"
        )
        _verify_swebench_task_map_source(args.task_ids_file, tasks_path, strict=strict_integrity)
    # Keep this explicit preflight call for eager eval-path validation and
    # CLI tests that intercept the boundary before task loading proceeds.
    load_eval_records_for_source(task_source=args.task_source, evals_path=evals_path)
    all_loaded_tasks = load_tasks_for_source(
        task_source=args.task_source,
        tasks_path=tasks_path,
        evals_path=evals_path,
        arc_max_trials=getattr(args, "arc_max_trials", 2),
        polyglot_test_feedback_tries=getattr(args, "polyglot_test_feedback_tries", 2),
        polyglot_test_feedback_max_lines=getattr(args, "polyglot_test_feedback_max_lines", 50),
    )
    tasks = _filter_tasks(
        all_loaded_tasks,
        args.task_ids,
        args.max_tasks,
        getattr(args, "task_ids_file", None),
        strict=True,
    )
    # Hold-out transfer probe: resolve hold-out ids against the SELECTED
    # training task ids, then pick the hold-out TaskSpecs from the same full
    # loaded task list. Hold-out tasks ride along in ``tasks`` (so repo prep,
    # classification, and agent sizing see them); the engine distinguishes
    # them by id via ``GenerationConfig.holdout_task_ids``.
    holdout_ids = _resolve_holdout_ids(
        getattr(args, "holdout_task_ids", None),
        getattr(args, "holdout_task_ids_file", None),
        training_ids={t.id for t in tasks},
    )
    if holdout_ids:
        tasks = tasks + _select_holdout_tasks(all_loaded_tasks, holdout_ids)

    provider_env: dict[str, str] = {}
    if args.runtime == "container":
        if not args.provider_profile:
            parser.error(f"--provider-profile is required when --runtime {args.runtime}")
        try:
            provider_env = load_provider_profile(args.provider_profile)
        except ProviderConfigError as exc:
            parser.error(str(exc))
        # Allow explicit process env (for example manifest-level per-experiment
        # overrides) to win over provider-profile defaults, but only for
        # non-secret knobs we intentionally vary per experiment.
        for key in (
            "MODEL",
            "MODEL_PROVIDER",
            "MODEL_AUTH_MODE",
            "REASONING_EFFORT",
            "KSI_OPENAI_MAX_TURNS",
            "OPENAI_AGENTS_DISABLE_TRACING",
        ):
            if key in _PRE_DOTENV_PROVIDER_KEYS:
                override = os.environ.get(key, "").strip()
                if override:
                    provider_env[key] = override
        apply_provider_env(provider_env)

    # Populate repo_path for repo-snapshot sources (SWE-bench Pro): auto-clone
    # if --swebench-repos-dir not set.
    if source_spec.uses_repo_snapshots:
        seed_test_files = bool(getattr(args, "swebench_pro_seed_tests", False))
        repos_dir = (
            Path(args.swebench_repos_dir)
            if args.swebench_repos_dir
            else default_swebench_repo_cache_dir(args.task_source)
        )
        prepare_swebench_repo_snapshots(
            tasks=tasks,
            repos_cache_dir=repos_dir,
            seed_test_files=seed_test_files,
        )

    if not tasks:
        parser.error("No tasks available after loading/filtering.")

    derived_num_agents = len(tasks)

    # Run LLM-based category classification for classification-capable sources
    # (swebench_pro) when requested.
    if getattr(args, "classify", False) and source_spec.supports_classification:
        cache_path: Path | None = None
        if getattr(args, "categories_json", None):
            cache_path = Path(args.categories_json)
        else:
            # Auto-derive sidecar path: e.g. dev.parquet → dev_categories.json
            cache_path = tasks_path.with_name(tasks_path.stem + "_categories.json")
        log.info("[classify] Running LLM classification for %d tasks → %s", len(tasks), cache_path)
        classify_tasks(tasks, args.task_source, cache_path=cache_path)
        log.info("[classify] Done.")
    elif getattr(args, "categories_json", None) and source_spec.supports_classification:
        # Load pre-existing categories JSON without running LLM
        cat_path = Path(args.categories_json)
        if cat_path.exists():
            cats = load_categories_json(cat_path)
            for task in tasks:
                if task.id in cats:
                    task.metadata["category"] = cats[task.id]
            log.info("[classify] Loaded %d categories from %s", len(cats), cat_path)
    runtime = _choose_runtime(args, provider_env)
    evaluator = _choose_evaluator(args)

    # Build direct LLM caller from normalized provider config so forum/claiming
    # use the same provider family as the task runtime.
    llm = build_llm_caller(
        provider=provider_env.get("MODEL_PROVIDER", "anthropic"),
        model=provider_env.get("MODEL", "claude-sonnet-4-20250514"),
        reasoning_effort=provider_env.get("REASONING_EFFORT", "").strip() or None,
        temperature=float(getattr(args, "llm_temperature", 0.0)),
        seed=int(args.seed) if args.seed is not None else None,
    )

    collecting = CollectingPersistence()
    observers: list[object] = [collecting]
    if args.runtime_db_path:
        observers.append(
            SqlitePersistence(
                runtime_db_path=args.runtime_db_path,
                experiment_name=args.experiment_name,
            )
        )
    persistence = CompositePersistence(observers)

    # Resolve project root for absolute-path cleanup of task artifacts.
    project_root = str(Path(__file__).resolve().parents[2])
    # Provenance stamp: resolved once here for this launch. The DB run
    # row is write-once on resume; cli_reporting mirrors that original stamp in
    # --output-json when the row already exists.
    code_commit = _ksi_code_commit(Path(project_root))
    # Full effective launch config, so the authoritative knowledge DB is
    # self-describing (a run's exact config is recoverable from the DB alone, not
    # only the optional gitignored --output-json). Same vars(args) already
    # persisted to the output-json payload — no secrets (API keys live in env /
    # the provider profile, never in args). default=str tolerates Path values.
    config_json = json.dumps(vars(args), default=str, sort_keys=True)

    config = _build_generation_config(
        args,
        num_agents=derived_num_agents,
        holdout_ids=holdout_ids,
        model=provider_env.get("MODEL", ""),
        code_commit=code_commit,
        model_provider=provider_env.get("MODEL_PROVIDER", ""),
        scoring_mode=args.evaluator,
        config_json=config_json,
    )

    orchestrator = GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=persistence,
        working_dir=project_root,
    )
    # Select the self-improvement mechanism from the registry. Default
    # "knowledge" rebuilds the engine's own default (DefaultKnowledgeStrategy),
    # so this is behavior-preserving; "raw_attempts" swaps in the ablation.
    orchestrator.set_improvement_strategy(get_strategy_spec(args.improvement_strategy).factory())
    # Re-scope cleanup filter to the FINAL experiment name. The engine's
    # __init__ may auto-suffix config.experiment_name on DB collisions (e.g.,
    # "arc_audit" -> "arc_audit_2"); the initial capture in main() would then
    # target containers that no longer carry that prefix, either missing the
    # real containers or cross-killing a sibling experiment.
    _set_container_name_prefix(config.experiment_name)
    if args.output_json:
        # Best-effort incremental snapshot: re-write
        # --output-json after every generation completes, not only once at
        # the very end, so a mid-run crash doesn't lose all traces. Failures
        # here are swallowed by CollectingPersistence.on_generation_end and
        # never abort the run; the final write below remains the canonical,
        # authoritative one.
        collecting.on_generation_snapshot = lambda gen, traces_so_far: write_output_json(
            args,
            tasks=tasks,
            traces=traces_so_far,
            collecting=collecting,
            orchestrator=orchestrator,
            code_commit=code_commit,
            resolved_model=f"{provider_env.get('MODEL_PROVIDER', '')}/{provider_env.get('MODEL', '')}",
            scoring_mode=args.evaluator,
            run_complete=False,
        )
    try:
        traces = orchestrator.run(tasks=tasks)
        log.info("completed traces=%d tasks=%d", len(traces), len(tasks))

        if args.output_json:
            write_output_json(
                args,
                tasks=tasks,
                traces=traces,
                collecting=collecting,
                orchestrator=orchestrator,
                code_commit=code_commit,
                resolved_model=f"{provider_env.get('MODEL_PROVIDER', '')}/{provider_env.get('MODEL', '')}",
                scoring_mode=args.evaluator,
                run_complete=True,
            )

        if args.pretask_debug_json:
            write_pretask_debug_json(args, orchestrator=orchestrator)
    finally:
        persistence.close()
    return 0


def cli() -> None:
    raise SystemExit(main())


# ── Standalone classify subcommand ────────────────────────────────────────────


def _build_classify_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pre-classify tasks with LLM and save a task_id→category JSON sidecar. "
            "Supports the swebench_pro task source."
        )
    )
    p.add_argument(
        "--task-source",
        choices=("swebench_pro",),
        required=True,
        help="Task dataset format.",
    )
    p.add_argument(
        "--tasks-path",
        required=True,
        help="Path to tasks parquet/csv/jsonl file.",
    )
    p.add_argument(
        "--evals-path",
        default=None,
        help="Optional eval sidecar parquet (swebench_pro only).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path for category mapping (default: <tasks_stem>_categories.json next to tasks-path).",
    )
    p.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task IDs to include (filters loaded tasks).",
    )
    p.add_argument(
        "--task-ids-file",
        default=None,
        help="Path to JSON file containing a list of task IDs to include.",
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Cap tasks after filtering (0=all).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=CLASSIFY_MAX_WORKERS,
        help=f"Parallel LLM classification workers (default: {CLASSIFY_MAX_WORKERS}).",
    )
    return p


def classify_main(argv: list[str] | None = None) -> int:
    """Standalone entry point for 'python -m ksi.cli classify' or 'ksi-classify'."""
    configure_logging()
    parser = _build_classify_parser()
    args = parser.parse_args(argv)

    tasks_path = Path(args.tasks_path)
    evals_path = Path(args.evals_path) if args.evals_path else None

    if not tasks_path.exists():
        parser.error(f"--tasks-path does not exist: {tasks_path}")
    if tasks_path.suffix.lower() not in {".parquet", ".csv", ".jsonl"}:
        parser.error(f"--tasks-path must be .parquet, .csv, or .jsonl: {tasks_path}")

    out_path = Path(args.output) if args.output else tasks_path.with_name(tasks_path.stem + "_categories.json")

    tasks = load_tasks_for_source(
        task_source=args.task_source,
        tasks_path=tasks_path,
        evals_path=evals_path,
        arc_max_trials=getattr(args, "arc_max_trials", 2),
    )
    tasks = _filter_tasks(
        tasks,
        args.task_ids,
        args.max_tasks,
        getattr(args, "task_ids_file", None),
        strict=True,
    )

    if not tasks:
        parser.error("No tasks available after loading/filtering.")

    log.info("Classifying %d %s tasks -> %s", len(tasks), args.task_source, out_path)
    classify_tasks(tasks, args.task_source, cache_path=out_path, max_workers=args.workers)

    # Log a summary of the category distribution
    from collections import Counter

    dist = Counter(t.metadata.get("category", "Uncat") for t in tasks)
    log.info("Category distribution:")
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        log.info("  %s %4d (%.1f%%)", cat.ljust(12), count, 100 * count / len(tasks))
    log.info("Saved to %s", out_path)
    return 0


def classify_cli() -> None:
    raise SystemExit(classify_main())


if __name__ == "__main__":
    raise SystemExit(main())
