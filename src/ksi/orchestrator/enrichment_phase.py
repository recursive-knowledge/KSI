"""Seed-package enrichment phase-service boundary for the orchestrator."""

from __future__ import annotations

import copy
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from ..memory.parity import redact_solver_hidden_eval_fields, redact_solver_hidden_text
from ..models import TaskSpec
from ..tasks.registry import resolve_source

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import GenerationalOrchestrator

log = logging.getLogger(__name__)


class ArcAnswerSanitizationError(RuntimeError):
    """Raised when an answer-bearing ARC ``arc_task_refs`` row cannot be stripped
    on disk. The runtime DB is bind-mounted read-only *whole*
    into the solver container, so a single unsanitized row would leak the answer;
    we fail closed (abort the run) rather than mount the leaking DB."""


@dataclass(frozen=True)
class EnrichmentCollaborators:
    """Explicit dependencies for the seed-package enrichment phase body."""

    config: Any  # read: no_memory, experiment_name
    knowledge: Any  # authoritative KnowledgeStore (or None)
    memory_store: Any  # optional runtime DB sidecar (or None)
    agents: list[Any]  # live list ref (seed_package mutated in place)
    best_scores: dict[str, float]  # live dict ref (read for per-task best score)
    holdout_ids: frozenset[str] | Any  # hold-out probe id set
    is_holdout: Callable[[str], bool]  # engine._is_holdout bound helper
    external_per_task_bundles: dict[str, Any]  # externally injected per-task bundles
    improvement_strategy: Any  # ImprovementStrategy — gates should_enrich()


@runtime_checkable
class EnrichmentPhaseService(Protocol):
    """Capability for running the seed-package enrichment phase."""

    def enrich(
        self,
        generation: int,
        assigned_map: dict[str, list[str]],
        tasks: list[TaskSpec],
    ) -> None: ...


def _knowledge_attempts_to_seed_records(page: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Convert KnowledgeStore query_task attempts into seed snapshot rows."""
    if not isinstance(page, dict):
        return []
    page = redact_solver_hidden_eval_fields(copy.deepcopy(page))
    attempts = page.get("attempts")
    if not isinstance(attempts, list):
        return []
    task_id = str(page.get("task_id") or "")
    records: list[dict[str, Any]] = []
    records_by_key: dict[tuple[Any, str], dict[str, Any]] = {}
    for attempt in reversed(attempts):
        if not isinstance(attempt, dict):
            continue
        content = attempt.get("content") if isinstance(attempt.get("content"), dict) else {}
        eval_results = content.get("eval_results") if isinstance(content.get("eval_results"), dict) else {}
        eval_results = dict(eval_results)
        score = attempt.get("score")
        if score is not None and eval_results.get("native_score") is None:
            eval_results["native_score"] = score
        task_specific_insights = list(content.get("insights")) if isinstance(content.get("insights"), list) else []
        records.append(
            {
                "gen": attempt.get("gen"),
                "generation": attempt.get("gen"),
                "agent_id": attempt.get("agent_id"),
                "task_id": task_id,
                "eval_results": eval_results,
                "final_model_output": str(content.get("model_output") or ""),
                "model_output": str(content.get("model_output") or ""),
                "full_memory_trace_condensed": redact_solver_hidden_text(content.get("trace_condensed")),
                "task_specific_insights": task_specific_insights,
                "attempt_history": [],
                "updated_at": "",
            }
        )
        records_by_key[(attempt.get("gen"), str(attempt.get("agent_id") or ""))] = records[-1]
    insights = page.get("insights")
    if isinstance(insights, list):
        for insight in insights:
            if not isinstance(insight, dict):
                continue
            text = str(insight.get("text") or "").strip()
            if not text:
                continue
            gen = insight.get("gen", insight.get("generation"))
            agent_id = str(insight.get("agent_id") or "")
            record = records_by_key.get((gen, agent_id))
            if record is None:
                record = {
                    "gen": gen,
                    "generation": gen,
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "eval_results": {},
                    "final_model_output": "",
                    "model_output": "",
                    "full_memory_trace_condensed": "",
                    "task_specific_insights": [],
                    "attempt_history": [],
                    "updated_at": "",
                    # Synthesized from a standalone R0 insight row with no
                    # matching attempt — it carries insight text only and
                    # must not be rendered or ranked as a prior attempt.
                    "insight_only": True,
                }
                records.append(record)
                records_by_key[(gen, agent_id)] = record
            task_insights = record.setdefault("task_specific_insights", [])
            if isinstance(task_insights, list) and text not in task_insights:
                task_insights.append(text)
    records.sort(key=_seed_record_generation_order, reverse=True)
    return records


def _seed_record_generation_order(record: dict[str, Any]) -> int:
    gen = record.get("gen", record.get("generation"))
    try:
        return int(gen)
    except (TypeError, ValueError):
        return -1


def _related_summary_rank_key(row: dict[str, Any]) -> float:
    """Rank key for related-summary candidates: highest ``native_score`` first.

    ``list_task_summaries`` orders rows by recency (latest attempt id DESC).
    Under ``--drop-solved`` a solved task's id freezes at whenever it was
    solved, while an unsolved task's id keeps climbing every generation, so
    recency alone systematically favors still-struggling siblings over
    solved ones. Missing/non-numeric scores rank lowest; ties keep
    their original (recency) order via Python's stable sort.
    """
    score = row.get("score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("-inf")


# Related-summary routing (H5b): the primary routed knowledge channel — the
# ``related_summaries`` rendered into pre-injected MEMORY.md — used to select
# candidates purely by ``task_id`` prefix ∪ repo, then re-rank by native_score,
# with NO relevance to the current task statement. For polyglot,
# ``task_id.split("__")[0]`` is the programming LANGUAGE, so it surfaced
# content-unrelated same-language exercises (0:5 routed-subset harm); for ARC
# (no ``__``, no repo) the candidate set was empty, so the channel was dead on
# the one benchmark where knowledge-transfer works. We instead rank ALL
# candidate summaries by lexical similarity of their task statement to the
# current task's statement, keep only those above a relevance floor, and fall
# back to the old prefix/repo candidate set only when nothing clears the floor
# (so behavior is unchanged when there is no statement-relevant sibling).
#
# Lexical only (deterministic, no embedder/network dependency). Legitimacy: uses
# the current task's own agent-visible statement plus other tasks'
# solver-visible approaches/lessons (already redacted at the seeding boundary) —
# no hidden tests or answers.
_RELATED_SUMMARY_MIN_SIMILARITY = 0.05
_RELATED_SUMMARY_MIN_TOKEN_OVERLAP = 2

# Common English + task-instruction boilerplate dropped before similarity so
# shared framing ("implement the following function") doesn't masquerade as
# topical relevance.
_SIMILARITY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "as",
        "is",
        "are",
        "be",
        "was",
        "were",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "from",
        "into",
        "out",
        "up",
        "down",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "should",
        "must",
        "can",
        "will",
        "would",
        "may",
        "each",
        "any",
        "all",
        "some",
        "such",
        "given",
        "using",
        "use",
        "used",
        "return",
        "returns",
        "function",
        "implement",
        "write",
        "task",
        "following",
        "value",
        "values",
        "input",
        "output",
        "test",
        "tests",
        "code",
        "please",
        "when",
        "where",
        "which",
        "what",
    }
)

_SIMILARITY_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _similarity_tokens(text: Any) -> frozenset[str]:
    """Lowercase word/number tokens (len ≥ 2) with stopwords removed."""
    return frozenset(
        tok
        for tok in _SIMILARITY_TOKEN_RE.findall(str(text or "").lower())
        if len(tok) >= 2 and tok not in _SIMILARITY_STOPWORDS
    )


def _lexical_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard overlap of two token sets (0.0 when either is empty)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _lexical_overlap_count(a: frozenset[str], b: frozenset[str]) -> int:
    return len(a & b) if a and b else 0


def _redact_summary_row_in_place(row: dict[str, Any]) -> None:
    """Scrub hidden verifier/test-output fragments from a task-summary row in place.

    ``KnowledgeStore.list_task_summaries`` copies raw ``trace_condensed`` into
    ``approach`` and dumps raw insights into the ``lessons`` JSON string. Those
    rows reach task agents through the mounted snapshot (``search_rows`` /
    ``related_summaries``), so — like the prior-attempts path — they must pass
    through the same read-time redactor (:func:`redact_solver_hidden_text`) that
    strips stale hidden-output fragments. ``lessons`` is JSON, so redact each
    insight string then re-dump rather than scrubbing the raw JSON text (which
    would mangle the structure). Idempotent: already-clean rows are unchanged.
    """
    approach = row.get("approach")
    if isinstance(approach, str) and approach:
        row["approach"] = redact_solver_hidden_text(approach)
    lessons = row.get("lessons")
    if isinstance(lessons, str) and lessons:
        try:
            insights = json.loads(lessons)
        except (ValueError, TypeError):
            insights = None
        if isinstance(insights, list):
            row["lessons"] = json.dumps(
                [redact_solver_hidden_text(item) if isinstance(item, str) else item for item in insights]
            )
        else:
            # Unexpected non-list shape: scrub the raw string defensively.
            row["lessons"] = redact_solver_hidden_text(lessons)


def _assert_single_task_per_agent(assigned_map: dict[str, list[str]], *, generation: int) -> None:
    """Fail loud if enrichment is ever handed more than one task for an agent.

    ``enrich`` reads a single ``task_ids[0]`` per agent (one seed_package per
    agent, one assigned task). Today the claim phase guarantees that (see
    ``engine._assert_single_task_per_agent``), but that is a behavioral property
    of one implementation, not a type-level one — ``assigned_map`` is typed
    ``dict[str, list[str]]``. Without this guard a future multi-task-per-agent
    change would silently enrich only the first task and drop the rest. Assert
    the invariant here so it fails loudly at the source instead.
    """
    violations = {agent_id: task_ids for agent_id, task_ids in assigned_map.items() if len(task_ids) > 1}
    if violations:
        raise RuntimeError(
            f"[gen {generation}] enrichment received multiple tasks for a single agent, violating the "
            f"1-task-per-agent invariant that enrich() relies on (it reads task_ids[0] per agent): {violations}"
        )


@dataclass(frozen=True)
class _EnrichmentContext:
    """Prebuilt inputs shared across every agent in one ``enrich`` call.

    Built ONCE by ``_build_enrichment_context`` (the expensive batch queries and
    prefix/repo groupings), then read per agent by ``_enrich_single_agent_seed``.
    """

    generation: int
    experiment: str
    task_by_id: dict[str, TaskSpec]
    external_per_task_bundles: dict[str, Any]
    all_summaries: list[dict]
    summaries_by_prefix: dict[str, list[dict]]
    summaries_by_repo: dict[str, list[dict]]
    knowledge_pages_by_task: dict[str, dict]


@dataclass
class EngineEnrichmentPhaseService:
    """Engine-backed seed-package enrichment phase adapter.

    The per-task KnowledgeStore enrichment body lives here behind an explicit
    service boundary used by the generation loop and tests.
    """

    engine: "GenerationalOrchestrator"

    def _collaborators(self) -> EnrichmentCollaborators:
        engine = self.engine
        return EnrichmentCollaborators(
            config=engine.config,
            knowledge=engine._knowledge,
            memory_store=engine._memory_store,
            agents=engine.agents,
            best_scores=engine._best_scores,
            holdout_ids=engine._holdout_ids,
            is_holdout=engine._is_holdout,
            external_per_task_bundles=getattr(engine, "_external_per_task_bundles", {}) or {},
            improvement_strategy=engine._improvement_strategy,
        )

    def enrich(
        self,
        generation: int,
        assigned_map: dict[str, list[str]],
        tasks: list[TaskSpec],
    ) -> None:
        """Enrich each agent's seed_package with KnowledgeStore data.

        Attaches prior_attempts and related_summaries (plus best_score /
        memory_snapshot) per assigned task. Does NOT populate
        ``distilled_knowledge``; scoped distillation bundles are attached by
        the seeding phase service via the seeder.
        """
        collab = self._collaborators()
        if not collab.improvement_strategy.should_enrich():
            return
        if collab.config.no_memory:
            return
        if collab.memory_store is None and collab.knowledge is None:
            return

        # Risk 1 (H6): enrich reads task_ids[0] per agent — fail loud rather
        # than silently enrich only the first task if that invariant is broken.
        _assert_single_task_per_agent(assigned_map, generation=generation)

        ctx = self._build_enrichment_context(collab, generation, assigned_map, tasks)

        external_bundles_attached = 0
        for agent in collab.agents:
            task_ids = assigned_map.get(agent.id)
            if not task_ids:
                continue
            # One task per agent: take the first (and only) assignment.
            task_id = task_ids[0]
            task = ctx.task_by_id.get(task_id)
            if task is None:
                log.warning("[ENGINE] enrich: task_id %s not found in tasks list", task_id)
                continue
            external_bundles_attached += self._enrich_single_agent_seed(agent, task, task_id, collab, ctx)

        # Observability for externally injected per-task KT bundles: the
        # load-time "Loaded N" log only proves the file parsed. Attach happens
        # here, keyed on task_id — a mis-keyed donor/recipient id map silently
        # attaches zero without this signal (biases KT toward a false null).
        external_per_task_bundles = ctx.external_per_task_bundles
        if external_per_task_bundles:
            # Numerator = agents that newly received a bundle this call;
            # denominator = distinct donor bundles loaded. With multiple agents
            # per task_id the numerator can exceed the denominator, so the two
            # counts are named explicitly rather than shown as a bare ratio.
            log.info(
                "[ENGINE] external per-task bundles: newly attached to %d agent(s) out of %d loaded bundle(s)",
                external_bundles_attached,
                len(external_per_task_bundles),
            )
            if external_bundles_attached == 0:
                log.warning(
                    "[ENGINE] %d external per-task seed bundle(s) loaded but NONE "
                    "attached — no assigned task_id matched a donor key. Check for "
                    "mis-keyed donor/recipient task ids.",
                    len(external_per_task_bundles),
                )

    def _build_enrichment_context(
        self,
        collab: EnrichmentCollaborators,
        generation: int,
        assigned_map: dict[str, list[str]],
        tasks: list[TaskSpec],
    ) -> _EnrichmentContext:
        """Build the once-per-call shared inputs: the (expensive) summaries
        fetch + hold-out drop + redaction + prefix/repo grouping, and the
        batched per-task knowledge query. Grouped ONCE here, not per agent."""
        experiment = collab.config.experiment_name or ""

        # Fetch summaries once for all agents (expensive query).
        all_summaries: list[dict] = []
        if collab.knowledge is not None:
            try:
                list_task_summaries = getattr(collab.knowledge, "list_task_summaries", None)
                if callable(list_task_summaries):
                    all_summaries = list(list_task_summaries(experiment=experiment or None, limit=200) or [])
            except Exception:
                log.warning("[ENGINE] Failed to fetch knowledge summaries for enrichment", exc_info=True)
        elif collab.memory_store is not None:
            try:
                all_summaries = list(
                    collab.memory_store.list_task_summaries(experiment=experiment or None, limit=200) or []
                )
            except Exception:
                log.warning("[ENGINE] Failed to fetch task summaries for enrichment", exc_info=True)

        # Hold-out probe: hold-out attempt summaries must never enrich other
        # agents (reverse leak via prefix/repo-matched related summaries or
        # the snapshot's search_rows) — drop them at the source.
        if collab.holdout_ids:
            all_summaries = [row for row in all_summaries if not collab.is_holdout(str(row.get("task_id") or ""))]

        # list_task_summaries copies raw trace_condensed into
        # ``approach`` and dumps raw insights into ``lessons``; both reach task
        # agents through the mounted snapshot (search_rows / related_summaries)
        # WITHOUT the read-time redaction the prior_attempts path already applies.
        # Scrub stale hidden-output fragments here, once, in place — every
        # downstream consumer (search_rows, the prefix/repo groupings, and the
        # deduped related_summaries) shares these row objects, so this covers
        # them all. Idempotent, so already-clean rows are untouched.
        for row in all_summaries:
            _redact_summary_row_in_place(row)

        # Pre-group summaries for O(1) lookup instead of O(M) per task
        summaries_by_prefix: dict[str, list[dict]] = defaultdict(list)
        summaries_by_repo: dict[str, list[dict]] = defaultdict(list)
        for row in all_summaries:
            row_task_id = str(row.get("task_id") or "")
            row_prefix = row_task_id.split("__")[0] if "__" in row_task_id else ""
            if row_prefix:
                summaries_by_prefix[row_prefix].append(row)
            row_repo = str(row.get("repo") or "").strip().lower()
            if row_repo:
                summaries_by_repo[row_repo].append(row)

        # Batch the per-task knowledge query (was an N+1: one query_task per
        # agent, each issuing a sub-query per entry-type serialized through the
        # store lock). One WHERE task_id IN (...) pass returns identical pages;
        # the loop below reads them from this map. Hold-out agents never query
        # (they take the `if holdout_agent` branch), so they are excluded here.
        knowledge_pages_by_task: dict[str, dict] = {}
        if collab.knowledge is not None:
            batch_task_ids: list[str] = []
            seen_batch_ids: set[str] = set()
            for agent in collab.agents:
                tids = assigned_map.get(agent.id)
                if not tids:
                    continue
                tid = tids[0]
                if not tid or tid in seen_batch_ids or collab.is_holdout(tid):
                    continue
                seen_batch_ids.add(tid)
                batch_task_ids.append(tid)
            if batch_task_ids:
                try:
                    knowledge_pages_by_task = collab.knowledge.query_tasks(
                        batch_task_ids,
                        entry_types=["attempt", "insight"],
                        experiment=experiment or None,
                        limit=8,
                    )
                except Exception:
                    log.warning("[ENGINE] Failed to batch-query knowledge attempts", exc_info=True)
                    knowledge_pages_by_task = {}

        return _EnrichmentContext(
            generation=generation,
            experiment=experiment,
            task_by_id={t.id: t for t in tasks},
            external_per_task_bundles=collab.external_per_task_bundles or {},
            all_summaries=all_summaries,
            summaries_by_prefix=summaries_by_prefix,
            summaries_by_repo=summaries_by_repo,
            knowledge_pages_by_task=knowledge_pages_by_task,
        )

    def _enrich_single_agent_seed(
        self,
        agent: Any,
        task: TaskSpec,
        task_id: str,
        collab: EnrichmentCollaborators,
        ctx: _EnrichmentContext,
    ) -> int:
        """Enrich one agent's seed_package for its assigned task. Returns 1 if a
        NEW external per-task bundle was attached this call, else 0.

        Security ordering (H6): the ARC-answer sanitization
        (``_build_arc_reference_payload``, which may raise
        ``ArcAnswerSanitizationError`` and fail the whole run closed) runs BEFORE
        any ``agent.seed_package`` mutation below — so an un-sanitizable answer
        aborts before the seed (and thus the mounted snapshot) is ever attached.
        """
        experiment = ctx.experiment
        metadata = task.metadata or {}
        task_source = str(metadata.get("task_source") or "").strip().lower()
        _src_spec = resolve_source(task_source)

        # Hold-out probe: a hold-out agent must receive exactly what a
        # brand-new task would — the cross-task channel only. Skip the
        # own-task prior-attempt history and the related/search summary
        # channels entirely (their seed_package keeps empty slots).
        holdout_agent = collab.is_holdout(task_id)

        # --- Prior attempts from authoritative knowledge store ---
        prior_attempts: list[dict] = []
        if holdout_agent:
            pass
        elif collab.knowledge is not None:
            # Read the page from the batched query above. Fall back to a
            # per-task query only on a batch miss (e.g. the batch failed),
            # preserving the original behavior exactly.
            page = ctx.knowledge_pages_by_task.get(task_id)
            if page is None:
                try:
                    page = collab.knowledge.query_task(
                        task_id,
                        entry_types=["attempt", "insight"],
                        experiment=experiment or None,
                        limit=8,
                    )
                except Exception:
                    log.warning("[ENGINE] Failed to query knowledge attempts for %s", task_id, exc_info=True)
                    page = None
            if page is not None:
                prior_attempts = _knowledge_attempts_to_seed_records(page)
        elif collab.memory_store is not None:
            try:
                prior_attempts = collab.memory_store.query_task_memory(
                    task_id=task_id,
                    experiment=experiment or None,
                    limit=8,
                )
            except Exception:
                log.warning("[ENGINE] Failed to query task memory for %s", task_id, exc_info=True)

        # --- Related summaries (statement-relevance ranking) ---
        related_summaries: list[dict] = []
        if not holdout_agent:
            related_summaries = self._rank_related_summaries(task, task_id, ctx)

        # --- Best score ---
        best_score = collab.best_scores.get(task_id)

        # --- Query records for snapshot (reuse prior_attempts to avoid duplicate query) ---
        relevant_task_ids = [task_id] if task_id else []
        query_records_by_task: dict[str, list[dict]] = {}
        if prior_attempts and task_id:
            query_records_by_task[task_id] = prior_attempts

        # --- ARC task reference (security fail-closed; MUST precede seed mutation) ---
        arc_payload_by_task = self._build_arc_reference_payload(task_id, metadata, _src_spec, collab, experiment)

        # --- Build memory snapshot ---
        memory_snapshot: dict = {
            "version": 2,
            "experiment": experiment,
            "generation": int(ctx.generation),
            "task_source": task_source,
            "task_id": task_id,
            "relevant_task_ids": relevant_task_ids,
            "query_records_by_task": query_records_by_task,
            # search_rows is a belt-and-suspenders snapshot field; the LIVE
            # same-context gate is exclude_task_ids (set below), which the MCP
            # server actually applies to vec/FTS retrieval. Filtered to empty
            # for hold-out agents so their rows never reach retrieval.
            "search_rows": [] if holdout_agent else ctx.all_summaries,
            "related_summaries": related_summaries,
            "arc_payload_by_task": arc_payload_by_task,
        }
        # exclude_task_ids is the LIVE retrieval gate: the MCP server reads it
        # from the snapshot and drops these rows from vec/FTS query retrieval
        # (forum + task agents). Hold-out ids are always excluded.
        exclude_ids: set[str] = set(collab.holdout_ids or ())
        if exclude_ids:
            # Omitted when empty so the snapshot stays byte-identical for
            # sources without hold-out exclusions.
            memory_snapshot["exclude_task_ids"] = sorted(exclude_ids)

        # --- Mutate seed_package in-place ---
        # Hold-out agents never receive a per-task bundle — not even an
        # externally injected one keyed by their task id.
        external_bundles_attached = 0
        if not holdout_agent and "per_task_bundle" not in agent.seed_package:
            external_bundle = ctx.external_per_task_bundles.get(task_id)
            if isinstance(external_bundle, dict) and external_bundle:
                agent.seed_package["per_task_bundle"] = dict(external_bundle)
                # Counts NEWLY attached agents only (the guard above skips
                # agents that already carry a per_task_bundle), not the
                # total number of agents holding a bundle.
                external_bundles_attached = 1
        agent.seed_package["assigned_task_id"] = task_id
        agent.seed_package["prior_attempts"] = prior_attempts
        agent.seed_package["related_summaries"] = related_summaries
        agent.seed_package["best_score"] = best_score
        agent.seed_package["memory_snapshot"] = memory_snapshot
        return external_bundles_attached

    def _rank_related_summaries(
        self,
        task: TaskSpec,
        task_id: str,
        ctx: _EnrichmentContext,
    ) -> list[dict]:
        """Rank related-summary candidates for one task (H5b).

        Primary: rank ALL candidate summaries by lexical similarity of the
        candidate task's statement to the current task's statement, keep those
        above a relevance floor, break ties by native_score. Fallback: the
        historical prefix/repo candidate set (native_score ranked) when nothing
        clears the floor — preserving old behavior (never regressing to an empty
        list where one existed before). Caller guards on hold-out.
        """
        all_summaries = ctx.all_summaries
        task_by_id = ctx.task_by_id
        related_summaries: list[dict] = []
        task_prefix = task_id.split("__")[0] if "__" in task_id else ""
        if all_summaries and task_id:
            prefix = task_prefix
            # ``task.repo`` (the TaskSpec field) is authoritative — no
            # task loader (ARC, SWE-bench Pro, polyglot, TB2) ever puts a
            # "repo" key into ``metadata``, so reading it from there was
            # a permanent no-op.
            task_repo = str(task.repo or "").strip().lower()

            current_tokens = _similarity_tokens(task.prompt)
            # Best (highest similarity) row per candidate task_id.
            relevant_by_tid: dict[str, tuple[float, dict]] = {}
            if current_tokens:
                for row in all_summaries:
                    row_tid = str(row.get("task_id") or "")
                    if not row_tid or row_tid == task_id:
                        continue
                    # Prefer the candidate's own agent-visible statement;
                    # fall back to its stored (redacted) approach/lessons
                    # text when the task isn't in the current pool.
                    cand_task = task_by_id.get(row_tid)
                    if cand_task is not None and cand_task.prompt:
                        cand_text: Any = cand_task.prompt
                    else:
                        cand_text = f"{row.get('approach') or ''} {row.get('lessons') or ''}"
                    cand_tokens = _similarity_tokens(cand_text)
                    sim = _lexical_similarity(current_tokens, cand_tokens)
                    if (
                        sim < _RELATED_SUMMARY_MIN_SIMILARITY
                        or _lexical_overlap_count(current_tokens, cand_tokens) < _RELATED_SUMMARY_MIN_TOKEN_OVERLAP
                    ):
                        continue
                    prev = relevant_by_tid.get(row_tid)
                    if prev is None or sim > prev[0]:
                        relevant_by_tid[row_tid] = (sim, row)

            if relevant_by_tid:
                ranked = sorted(
                    relevant_by_tid.values(),
                    key=lambda pair: (pair[0], _related_summary_rank_key(pair[1])),
                    reverse=True,
                )
                related_summaries = [row for _sim, row in ranked[:5]]
                log.debug(
                    "[ENGINE] related_summaries for %s: %d candidate(s) cleared "
                    "similarity floor %.2f (statement-relevance routing)",
                    task_id,
                    len(relevant_by_tid),
                    _RELATED_SUMMARY_MIN_SIMILARITY,
                )
            else:
                # Fallback: historical prefix/repo candidate set.
                candidates: list[dict] = []
                if prefix:
                    candidates.extend(ctx.summaries_by_prefix.get(prefix, []))
                if task_repo:
                    candidates.extend(ctx.summaries_by_repo.get(task_repo, []))
                seen: set[str] = set()
                deduped: list[dict] = []
                for row in candidates:
                    row_tid = str(row.get("task_id") or "")
                    if row_tid == task_id or row_tid in seen:
                        continue
                    seen.add(row_tid)
                    deduped.append(row)
                # Re-rank by native_score before truncating to the top 5 —
                # see _related_summary_rank_key.
                deduped.sort(key=_related_summary_rank_key, reverse=True)
                related_summaries = deduped[:5]
                log.debug(
                    "[ENGINE] related_summaries for %s: no candidate cleared "
                    "similarity floor %.2f; fell back to prefix/repo routing "
                    "(%d candidate(s))",
                    task_id,
                    _RELATED_SUMMARY_MIN_SIMILARITY,
                    len(deduped),
                )
        return related_summaries

    def _build_arc_reference_payload(
        self,
        task_id: str,
        metadata: dict,
        src_spec: Any,
        collab: EnrichmentCollaborators,
        experiment: str,
    ) -> dict[str, dict]:
        """Build the hidden ARC reference payload for the mounted snapshot,
        stripping expected test OUTPUTS (the answer) fail-closed.

        Raises :class:`ArcAnswerSanitizationError` when an answer-bearing on-disk
        row cannot be re-stripped — the caller runs this BEFORE mutating the
        seed_package, so a leak aborts the run before any snapshot is attached.
        """
        arc_payload_by_task: dict[str, dict] = {}
        if src_spec is not None and src_spec.arc_task_reference:
            train_pairs = metadata.get("arc_train_pairs")
            eval_pairs = metadata.get("arc_eval_test_pairs")
            if not isinstance(eval_pairs, list):
                eval_pairs = metadata.get("arc_test_pairs")
            if isinstance(train_pairs, list) and isinstance(eval_pairs, list):
                # The expected test OUTPUTS (the answer) must never
                # enter solver-mounted material. Both the snapshot
                # (arc_payload_by_task) and the arc_task_refs DB row derive
                # from hidden_payload and are bind-mounted RO into the
                # container, so strip test outputs here — keep test INPUTS
                # only. Train pairs keep outputs (the agent legitimately
                # learns from them). Authoritative scoring is host-side
                # (benchmarks/arc_session.py) from the tool trace against
                # task.metadata, so nothing in-container needs the answer.
                test_inputs_only = [{"input": p.get("input")} for p in eval_pairs if isinstance(p, dict)]
                hidden_payload = {
                    "task_id": task_id,
                    "train": train_pairs,
                    "test": test_inputs_only,
                    "max_trials": int(metadata.get("arc_max_trials") or 2),
                }
                arc_payload_by_task[task_id] = hidden_payload
                if collab.memory_store is not None:
                    try:
                        collab.memory_store.upsert_arc_task_reference(
                            task_id=task_id,
                            payload=hidden_payload,
                            experiment=experiment or None,
                        )
                    except Exception:
                        log.warning("[ENGINE] Failed to upsert ARC audit ref for %s", task_id, exc_info=True)
            elif collab.memory_store is not None:
                try:
                    ref = collab.memory_store.get_arc_task_reference(
                        task_id=task_id,
                        experiment=experiment or None,
                    )
                    if isinstance(ref, dict):
                        # A row written by a pre-fix run may carry
                        # unstripped test outputs; sanitize on load so the
                        # mounted snapshot never exposes the answer.
                        ref_test = ref.get("test")
                        if isinstance(ref_test, list):
                            # Did the on-disk row actually carry an answer?
                            # Only then is a failed strip a real leak.
                            had_answer = any(isinstance(p, dict) and p.get("output") is not None for p in ref_test)
                            ref = {
                                **ref,
                                "test": [{"input": p.get("input")} for p in ref_test if isinstance(p, dict)],
                            }
                            # The runtime sqlite is bind-mounted RO into the
                            # container (/app/memory-db), so sanitizing only
                            # the in-memory snapshot is not enough: a pre-fix
                            # arc_task_refs row still exposes the answer via
                            # `sqlite3 ... SELECT payload_json`. Overwrite the
                            # on-disk row with the stripped payload too. The
                            # upsert is idempotent (already-clean rows rewrite
                            # to the same value).
                            try:
                                collab.memory_store.upsert_arc_task_reference(
                                    task_id=task_id,
                                    payload=ref,
                                    experiment=experiment or None,
                                )
                            except Exception as exc:
                                # Fail closed when the row actually held
                                # an answer. The DB is mounted RO wholesale, so
                                # we cannot exclude one unstripped row — if we
                                # cannot strip it on disk, abort rather than
                                # leak. A failed rewrite of an already-clean
                                # row (no output) is harmless, so only warn.
                                if had_answer:
                                    raise ArcAnswerSanitizationError(
                                        f"could not strip answer-bearing on-disk ARC ref for {task_id}; "
                                        f"refusing to mount the runtime DB (would leak the answer)"
                                    ) from exc
                                log.warning(
                                    "[ENGINE] Failed to rewrite already-clean ARC ref for %s",
                                    task_id,
                                    exc_info=True,
                                )
                        arc_payload_by_task[task_id] = ref
                except ArcAnswerSanitizationError:
                    # Security fail-closed: must not be swallowed by the
                    # best-effort get_arc_task_reference handler below.
                    raise
                except Exception:
                    log.warning("[ENGINE] Failed to get ARC ref for %s", task_id, exc_info=True)
        return arc_payload_by_task
