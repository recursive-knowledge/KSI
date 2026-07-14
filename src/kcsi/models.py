from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from .tokens import TokenUsage


class EvalResult(TypedDict, total=False):
    """Structured result returned by ``Evaluator.evaluate`` and stored on
    ``TaskTrace.eval_result``.

    ``total=False`` so every key is optional: evaluators emit partial dicts
    (e.g. an error-only dict before scoring, or a setup-failure dict), and this
    type captures the known keys without forcing any to be present. This keeps
    three-state fields meaningful at the type level — a key being absent is a
    distinct, type-checkable state from ``False``.

    This declares the cross-evaluator keys read by generic consumers (e.g.
    ``orchestrator.scoring``, ``orchestrator.engine``, ``approach_diagnosis``).
    Evaluator-specific scalar keys (terminal_bench_2 ``reward``/exit codes,
    swebench ``patch_source``) are intentionally not
    enumerated here — they are read only inside their own evaluator modules,
    not through this generic type.
    """

    status: str
    instance_id: str
    task_type: str
    resolved: bool
    native_score: float
    error: str
    # Cross-evaluator keys consumed by generic readers (scoring/engine/etc.).
    swebench_status: str
    run_summary: dict[str, Any]
    instance_report: dict[str, Any]


class ArcEvalResult(EvalResult, total=False):
    """ARC-session evaluator result, adding ARC trial-scoring keys.

    ``scored_from_runtime_trials`` is the canonical three-state field: ``True``
    (canonical runtime-trial scoring — only formal ``arc_submit_trial``
    submissions are scored), ``False`` (reference-validation failure:
    missing or invalid expected grids), or absent (legacy traces, or infra
    failures with no tool trace — status ``no_runtime_submission``).
    ``total=False`` preserves that the key may be missing.
    """

    arc_pass_ratio: float
    normalized_output_json: str
    arc_correct_count: int
    arc_total_count: int
    arc_per_test: list[dict[str, Any]]
    scored_from_runtime_trials: bool


@dataclass
class GenerationConfig:
    """Configuration for a generational orchestrator run."""

    num_generations: int
    num_agents: int
    drop_solved: bool = True
    solved_threshold: float = 1.0
    max_concurrent_tasks: int = 25  # max parallel container workers for task execution; matches CLI default (conservative host/provider-safe ceiling)
    max_concurrent_forum_tasks: int = 0  # discussion-phase worker cap; 0 = follow max_concurrent_tasks
    max_task_retries: int = 3  # 0 means no retry; retries on transient failures (timeout, container crash)
    knowledge_db_path: str = ""  # Authoritative knowledge substrate / retrieval SQLite DB
    runtime_db_path: str = ""  # Optional runtime audit SQLite DB sidecar
    experiment_name: str = "kcsi"  # Mirrors the CLI --experiment-name default (single source of truth)
    forum_timeout_sec: int = 900  # timeout for the per-task discussion task
    native_memory_max_chars: int = 240_000
    native_memory_max_files: int = 8
    native_memory_max_chars_per_file: int = 60_000
    seed_bundle_path: str = ""  # Path to external knowledge bundle JSON (empty = disabled)
    seed_per_task_bundles_path: str = ""  # Path to external per-task distilled knowledge JSON (empty = disabled)
    disable_memory_mcp: bool = False  # Disable MCP knowledge tools in task containers (keep DB-backed discussion state)
    resume: bool = False  # Resume prior experiment (seed scores from DB); otherwise auto-suffix on collision
    start_generation: int = 1  # Internal resume cursor; normally 1, set to latest completed generation + 1 on --resume
    model: str = ""  # Model identifier for cost tracking (e.g. "claude-haiku-4-5-20251001")
    no_memory: bool = False  # Disable agent knowledge tools/discussion/seeding; keep KnowledgeStore authoritative
    # Provenance stamp: pins a published number to its exact code/model/scoring-mode.
    code_commit: str = ""  # KCSI's own resolved git commit SHA at launch time
    model_provider: str = ""  # Mirrors the CLI-resolved MODEL_PROVIDER (e.g. "anthropic")
    scoring_mode: str = ""  # Mirrors the CLI's --evaluator choice (e.g. "arc_session")
    # ── Three-phase generation loop (per-task forum / cross-task forum / distill) ──
    # Declared fields so direct programmatic configs (once num_generations and
    # num_agents are supplied) carry the same defaults the CLI applies. Each
    # default mirrors the corresponding src/kcsi/cli.py argparse default; the
    # engine reads them as real attributes.
    per_task_forum_rounds: int = 1  # CLI --per-task-forum-rounds default
    cross_task_forum_rounds: int = 2  # CLI --cross-task-forum-rounds default (single source of truth)
    cross_task_forum_timeout_sec: int = 900  # CLI --cross-task-forum-timeout-sec default
    cross_task_shared_container: bool = False  # CLI --cross-task-shared-container default
    distill_enabled: bool = True  # CLI --distill-enabled default
    # Cross-task distillation conditioned on the downstream task: when True
    # (default), each attempted next-gen task gets its own cross-task bundle
    # distilled with that task's prompt as conditioning; when False, one
    # broadcast bundle under CROSS_TASK_SENTINEL is shared across all agents
    # (the legacy/ablation path).
    cross_task_distill_target_conditioning: bool = True
    # Per-target relevance selection for cross-task distillation (opt-in).
    # Default False keeps the shared-set behavior: the forum
    # history is trimmed ONCE against the largest target so every target shares a
    # byte-identical cache_prefix. True gives EACH target its own
    # relevance-ranked post set (fixing the defect where non-largest targets are
    # selected against the largest target's vocabulary) at the cost of defeating
    # the cross-target prompt cache. Only meaningful under
    # cross_task_distill_target_conditioning. See distiller.distill().
    cross_task_distill_per_target_selection: bool = False  # CLI --cross-task-distill-per-target-selection default
    # abort the run after this many consecutive generations whose
    # distillation was fully zeroed by failures (0 = disabled, ERROR-log only).
    abort_on_distill_stall: int = 0  # CLI --abort-on-distill-stall default
    distill_per_task_model: str | None = None  # CLI --distill-per-task-model default
    distill_cross_task_model: str | None = None  # CLI --distill-cross-task-model default
    forum_early_exit: bool = False  # CLI --forum-early-exit default
    forum_early_exit_poll_sec: float = 3.0  # CLI --forum-early-exit-poll-sec default
    # Quorum-based early exit: requiring 100% of agents to signal
    # done means a straggler minority (observed ~5-30% in some generations,
    # concentrated in generation 1 cold starts) blocks early-exit from firing
    # at all, forcing the full hard-timeout wait. 100.0 (default) preserves
    # the all-required behavior exactly; opting into a lower
    # threshold trades a small chance of cutting off a still-working
    # straggler for not waiting out the full --cross-task-forum-timeout-sec
    # / --forum-timeout-sec backstop.
    forum_early_exit_quorum_pct: float = 100.0  # CLI --forum-early-exit-quorum-pct default
    forum_early_exit_quorum_grace_sec: float = 0.0  # CLI --forum-early-exit-quorum-grace-sec default
    require_vector: bool = False  # CLI --require-vector default
    # Hold-out transfer probe (--holdout-task-ids): these task ids are attempted
    # every generation with the current cross-task knowledge injected but are
    # excluded from learning (forums, distillation, seeding inputs),
    # --drop-solved, early-stop, and headline metrics. Empty = feature off.
    holdout_task_ids: list[str] = field(default_factory=list)
    # Full effective launch config as a JSON string (the CLI stamps json.dumps of
    # vars(args)), so the authoritative knowledge DB is self-describing and a run's
    # exact configuration is recoverable from the DB alone — not only from the
    # optional, gitignored --output-json sidecar. "" when not supplied. Keep this
    # at the end so new provenance storage does not shift older positional callers
    # after scoring_mode.
    config_json: str = ""


@dataclass
class TaskSpec:
    """A single task to be solved by an agent."""

    id: str
    repo: str = ""
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Mutable state for an agent across generations."""

    id: str
    generation: int = 0
    alive: bool = True
    workstream: str = ""
    workstream_description: str = ""
    seed_package: dict[str, Any] = field(default_factory=dict)
    tasks_completed: int = 0
    token_usage: int = 0


@dataclass
class Assignment:
    """A task assigned to an agent for a generation."""

    generation: int
    agent_id: str
    task_id: str


@dataclass
class TaskTrace:
    """Result of an agent executing a single task."""

    generation: int
    agent_id: str
    task_id: str
    model_output: str | None = None
    eval_result: EvalResult = field(default_factory=EvalResult)
    native_score: float | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    runtime_meta: dict[str, Any] = field(default_factory=dict)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    error: str | None = None
    repo: str = ""
    """Upstream repo (e.g. ``owner/name``) for the task, propagated from
    ``TaskSpec.repo``. Persistence writers use this to populate ``tasks.repo``
    even on silent-failure traces where the downstream
    ``insert_task_summary`` path would otherwise be skipped."""


@dataclass
class Insight:
    """A knowledge nugget published by an agent to a discussion channel."""

    id: str
    text: str
    author_agent_id: str
    generation: int
    workstream: str
    source_task_id: str | None = None
    confidence: str = "medium"
    evidence_refs: list[str] = field(default_factory=list)
