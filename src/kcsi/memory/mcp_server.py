"""MCP stdio server for memory/forum tools.

Memory/forum tools:
  - query
  - forum_read
"""

from __future__ import annotations

try:
    from ..utils import to_int
except ImportError:
    # Standalone mode (MCP server in container) — inline the helper.
    def to_int(value, default=0):  # type: ignore[misc]
        try:
            return default if value is None else int(value)
        except Exception:
            return default


import copy
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from .forum_bus import ForumBus
    from .knowledge_store import KnowledgeStore
    from .store import MemoryStore
except ImportError:
    # isort: off
    from forum_bus import ForumBus
    from knowledge_store import KnowledgeStore
    from store import MemoryStore
    # isort: on

log = logging.getLogger(__name__)
_EMBEDDER: Any | None = None
_EMBEDDER_INIT_DONE = False


def _knowledge_vec_enabled(knowledge_store: Any | None) -> bool:
    """Return whether semantic vector search is actually available."""
    return bool(knowledge_store is not None and getattr(knowledge_store, "_vec_enabled", False))


class ForumProtocolState:
    """Track retrieval steps within one forum MCP session."""

    def __init__(self) -> None:
        self.knowledge_task_ids: set[str] = set()
        self.semantic_query_task_ids: set[str] = set()

    def mark_knowledge(self, task_id: str) -> None:
        normalized = str(task_id or "").strip()
        if normalized:
            self.knowledge_task_ids.add(normalized)

    def mark_query(self, task_id: str, query: str) -> None:
        normalized = str(task_id or "").strip()
        if normalized and str(query or "").strip():
            self.semantic_query_task_ids.add(normalized)

    def missing_for_post(self, task_id: str) -> list[str]:
        normalized = str(task_id or "").strip()
        if not normalized:
            return ["task_id"]
        missing: list[str] = []
        if normalized == "__cross_task__":
            if normalized not in self.semantic_query_task_ids:
                missing.append("query(task_id='__cross_task__', query='...')")
            return missing
        if normalized not in self.knowledge_task_ids:
            missing.append(f"knowledge(task_id='{normalized}')")
        if normalized not in self.semantic_query_task_ids:
            missing.append(f"query(task_id='{normalized}', query='...')")
        return missing


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _respond(msg_id: Any, result: dict[str, Any]) -> None:
    response = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _respond_error(msg_id: Any, code: int, message: str) -> None:
    response = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _get_embedder() -> Any | None:
    """Lazy embedder loader for semantic MCP search."""
    global _EMBEDDER, _EMBEDDER_INIT_DONE
    if _EMBEDDER_INIT_DONE:
        return _EMBEDDER
    _EMBEDDER_INIT_DONE = True
    try:
        try:
            from .embeddings import Embedder
        except ImportError:
            from embeddings import Embedder
        _EMBEDDER = Embedder()
        return _EMBEDDER
    except Exception as exc:
        log.warning("Semantic embedder unavailable; falling back to lexical search: %s", exc)
        _EMBEDDER = None
        return None


def _resolve_embedder(candidate: Any | None) -> Any | None:
    """Resolve a possibly-lazy embedder factory."""
    if candidate is None:
        return None
    if callable(candidate) and not hasattr(candidate, "embed"):
        return candidate()
    return candidate


def _resolve_exclude_task_ids(snapshot: dict[str, Any] | None) -> frozenset[str]:
    """Task ids to exclude from ``query`` retrieval (hold-out probe).

    Merges the ``MEMORY_EXCLUDE_TASK_IDS`` env CSV with the snapshot's
    ``exclude_task_ids`` list (written by the engine's seed enrichment).
    Empty when the hold-out probe is unused.
    """
    exclude = {tid.strip() for tid in os.environ.get("MEMORY_EXCLUDE_TASK_IDS", "").split(",") if tid.strip()}
    if isinstance(snapshot, dict):
        snap_ids = snapshot.get("exclude_task_ids")
        if isinstance(snap_ids, list):
            exclude |= {str(tid).strip() for tid in snap_ids if str(tid).strip()}
    return frozenset(exclude)


def _drop_excluded_rows(
    rows: list[dict[str, Any]],
    exclude_task_ids: frozenset[str],
) -> list[dict[str, Any]]:
    """Drop retrieval rows whose ``task_id`` is excluded (hold-out probe)."""
    if not exclude_task_ids:
        return rows
    return [row for row in rows if str(row.get("task_id") or "") not in exclude_task_ids]


def _excluded_query_result(task_id: str, *, semantic_query_text: str = "") -> dict[str, Any]:
    """Empty ``query`` payload for task ids excluded from live retrieval."""
    return {
        "task_id": task_id,
        "records": [],
        "insights": [],
        "related": [],
        "semantic_enabled": False,
        "retrieval_mode": "excluded",
        "semantic_query": semantic_query_text,
        "semantic_result_count": 0,
        "semantic_error": "",
    }


# FTS5 boolean operators, filtered out of tokenized fallback queries so they
# are never emitted as bare terms in the OR-joined MATCH expression.
_FTS_OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})


def _fts_fallback_related(
    knowledge_store: Any | None,
    query_text: str,
    *,
    max_results: int = 5,
    experiment: str | None = None,
    exclude_task_ids: frozenset[str] = frozenset(),
    task_id_fallback: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Lexical (FTS5) retrieval fallback for the ``related`` field.

    Used when semantic vector search is unavailable (no working embedder /
    vec index) or when a query embedding fails at runtime. A ``distance`` key
    is added (set to ``None``) so the FTS items carry a superset of the keys a
    semantic ``vec_search`` row carries — every ``vec_search`` key is present,
    plus FTS5's own ``created_at``. Agents that parse ``related`` items by
    key-presence therefore see the same fields regardless of retrieval mode;
    none branch on ``distance`` being a float, so ``None`` is safe.

    Returns ``(related, error)``. ``error`` is non-empty only when FTS itself
    raised; an empty FTS result with no error is a normal "no lexical match".
    """
    text = str(query_text or "").strip()
    if not text or knowledge_store is None:
        return [], ""
    fts_fn = getattr(knowledge_store, "fts_search", None)
    if not callable(fts_fn):
        return [], ""
    # Build an OR-joined MATCH expression from user query tokens. The default
    # FTS5 sanitizer space-joins terms, which FTS5 reads as implicit AND — a
    # multi-word ``related`` query then matches almost nothing. We tokenize on
    # word chars (dropping punctuation and FTS5 operators, so no injection
    # surface) and OR-join, then call the store's ``raw_match=True`` path so the
    # tokens reach MATCH as a real OR-query.
    terms = [t for t in re.findall(r"\w+", text) if t.upper() not in _FTS_OPERATORS]
    if task_id_fallback:
        # A task id is not a natural-language query. Numeric/short suffixes such
        # as the ``1`` in ``django__django-1`` are too broad by themselves, so
        # prefer meaningful identity tokens and require all remaining tokens.
        narrowed = [t for t in terms if len(t) >= 3 and not t.isdigit()]
        # ...but never emit an EMPTY query. A short or digit-only task id (e.g.
        # ``t1`` or a digit-only ARC id) has no >=3-char non-numeric token, so
        # narrowing would drop everything and silently disable the fallback —
        # returning no related rows at all. When narrowing empties the list,
        # fall back to the raw (operator-filtered) tokens so such ids still
        # produce a usable MATCH query and retrieve related rows.
        terms = narrowed or terms
    if not terms:
        return [], ""
    match_query = " ".join(terms) if task_id_fallback else " OR ".join(terms)
    try:
        rows = (
            fts_fn(
                match_query,
                max_results=max_results,
                experiment=experiment,
                raw_match=True,
            )
            or []
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("FTS fallback query failed: %s", exc)
        return [], str(exc)
    related: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("task_id") or "") in exclude_task_ids:
            continue
        item = dict(row)
        item = _redact_related_row(item)
        # Shape-match semantic vec_search rows: they carry a ``distance`` key.
        item.setdefault("distance", None)
        related.append(item)
    return related, ""


def _is_redundant(candidate: str, accepted: list[str], threshold: float = 0.6) -> bool:
    """Check if candidate insight is redundant with any accepted insight using word overlap."""
    if not candidate.strip():
        return True
    cand_words = set(candidate.lower().split())
    if not cand_words:
        return True
    for existing in accepted:
        exist_words = set(existing.lower().split())
        if not exist_words:
            continue
        overlap = len(cand_words & exist_words)
        smaller = min(len(cand_words), len(exist_words))
        if smaller > 0 and overlap / smaller >= threshold:
            return True
    return False


def _load_snapshot(path: str | None) -> dict[str, Any] | None:
    snapshot_path = str(path or "").strip()
    if not snapshot_path:
        return None
    p = Path(snapshot_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to load memory snapshot %s: %s", snapshot_path, exc)
        return None
    return data if isinstance(data, dict) else None


def _query_from_snapshot(
    *,
    snapshot: dict[str, Any],
    task_id: str,
    max_records: int = 8,
    experiment: str | None = None,
    knowledge_store: KnowledgeStore | None = None,
    semantic_embedder: Any | None = None,
    semantic_query: str = "",
    exclude_task_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    task_id = str(task_id or "").strip()
    if not task_id:
        raise ValueError("task_id is required")
    if task_id in exclude_task_ids:
        return _excluded_query_result(task_id, semantic_query_text=str(semantic_query or "").strip())
    by_task = snapshot.get("query_records_by_task") or {}
    rows = []
    if isinstance(by_task, dict):
        raw_rows = by_task.get(task_id) or []
        if isinstance(raw_rows, list):
            rows = raw_rows[: max(1, min(int(max_records), 20))]
    compact_records: list[dict[str, Any]] = []
    insights: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        eval_results = row.get("eval_results") or {}
        tagged_meta = {
            "gen": row.get("gen"),
            "agent_id": row.get("agent_id"),
            "task_id": row.get("task_id"),
            "status": (eval_results.get("status") or eval_results.get("swebench_status") or ""),
            "resolved": bool(eval_results.get("resolved")) if isinstance(eval_results, dict) else False,
            "native_score": eval_results.get("native_score") if isinstance(eval_results, dict) else None,
        }
        attempt_history = _redact_attempt_history_for_query(row.get("attempt_history"))
        compact_records.append(
            {
                "gen": row.get("gen"),
                "agent_id": row.get("agent_id"),
                "task_id": row.get("task_id"),
                "status": tagged_meta["status"],
                "resolved": tagged_meta["resolved"],
                "native_score": tagged_meta["native_score"],
                "full_memory_trace_condensed": _redact_solver_hidden_text(row.get("full_memory_trace_condensed") or ""),
                "task_specific_insights": row.get("task_specific_insights") or [],
                "attempt_count": len(attempt_history),
                "attempt_history": attempt_history,
                "updated_at": row.get("updated_at", ""),
            }
        )
        for text in row.get("task_specific_insights") or []:
            if isinstance(text, str) and text.strip():
                insights.append({"text": text.strip(), "meta": tagged_meta})
    # Deduplicate insights across generations
    deduped_insights: list[dict[str, Any]] = []
    accepted_texts: list[str] = []
    for ins in insights:
        text = ins.get("text", "")
        if not _is_redundant(text, accepted_texts):
            accepted_texts.append(text)
            deduped_insights.append(ins)
    insights = deduped_insights
    related = _redact_related_rows(snapshot.get("related_summaries") or [])
    if related:
        related = related[:5]
    semantic_query_text = str(semantic_query or "").strip()
    resolved_embedder = (
        _resolve_embedder(semantic_embedder) if knowledge_store is not None and semantic_query_text else None
    )
    semantic_available = _knowledge_vec_enabled(knowledge_store) and resolved_embedder is not None
    semantic_error = ""
    retrieval_mode = "none"
    if semantic_query_text:
        if semantic_available:
            try:
                related = _redact_related_rows(
                    _drop_excluded_rows(
                        knowledge_store.vec_search(
                            resolved_embedder.embed(semantic_query_text),
                            max_results=5,
                            experiment=experiment,
                        )
                        or [],
                        exclude_task_ids,
                    )
                )
                retrieval_mode = "semantic"
            except Exception as exc:
                log.debug("snapshot semantic query failed for task_id=%s: %s", task_id, exc)
                semantic_error = str(exc)
                # Per-call fallback: a single embedding failure must not leave
                # agents with zero retrieval — degrade to lexical FTS.
                related, _ = _fts_fallback_related(
                    knowledge_store,
                    semantic_query_text,
                    max_results=5,
                    experiment=experiment,
                    exclude_task_ids=exclude_task_ids,
                )
                retrieval_mode = "fts"
        else:
            # Embedder/vec unavailable (e.g. no HF_TOKEN): use the promised
            # FTS-only fallback rather than returning an empty related list.
            # Semantic was never attempted here, so leave ``semantic_error``
            # empty (it must mean "semantic failed", not "FTS failed"); the
            # ``retrieval_mode="fts"`` field already signals the degraded mode.
            related, _ = _fts_fallback_related(
                knowledge_store,
                semantic_query_text,
                max_results=5,
                experiment=experiment,
                exclude_task_ids=exclude_task_ids,
            )
            retrieval_mode = "fts"
    return {
        "task_id": task_id,
        "records": compact_records,
        "insights": insights,
        "related": related,
        "semantic_enabled": semantic_available,
        "retrieval_mode": retrieval_mode,
        "semantic_query": semantic_query_text,
        "semantic_result_count": len(related),
        "semantic_error": semantic_error,
    }


def _knowledge_attempts_to_query_rows(page: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(page, dict):
        return []
    attempts = page.get("attempts")
    if not isinstance(attempts, list):
        return []
    task_id = str(page.get("task_id") or "")
    rows: list[dict[str, Any]] = []
    for attempt in reversed(attempts):
        if not isinstance(attempt, dict):
            continue
        content = attempt.get("content") if isinstance(attempt.get("content"), dict) else {}
        eval_results = content.get("eval_results") if isinstance(content.get("eval_results"), dict) else {}
        eval_results = dict(eval_results)
        score = attempt.get("score")
        if score is not None and eval_results.get("native_score") is None:
            eval_results["native_score"] = score
        rows.append(
            {
                "gen": attempt.get("gen"),
                "agent_id": attempt.get("agent_id"),
                "task_id": task_id,
                "eval_results": eval_results,
                "full_memory_trace_condensed": str(content.get("trace_condensed") or ""),
                "task_specific_insights": content.get("insights") if isinstance(content.get("insights"), list) else [],
                "attempt_history": [],
                "updated_at": "",
            }
        )
    return rows


def handle_query(
    *,
    store: MemoryStore | None,
    task_id: str,
    max_records: int = 8,
    experiment: str | None = None,
    knowledge_store: KnowledgeStore | None = None,
    semantic_embedder: Any | None = None,
    semantic_query: str = "",
    exclude_task_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    task_id = str(task_id or "").strip()
    if not task_id:
        raise ValueError("task_id is required")
    if task_id in exclude_task_ids:
        return _excluded_query_result(task_id, semantic_query_text=str(semantic_query or task_id).strip())
    capped_limit = max(1, min(int(max_records), 20))
    query_task_fn = getattr(knowledge_store, "query_task", None) if knowledge_store is not None else None
    if callable(query_task_fn):
        page = query_task_fn(
            task_id,
            entry_types=["attempt"],
            experiment=experiment,
            limit=capped_limit,
        )
        # Defense-in-depth: the `query` tool is a second forum-facing exit of the
        # same attempt page as `handle_knowledge`. Today `compact_records` below
        # projects to scalars and never echoes the ARC gold answer, but route the
        # page through the same redactor so a future field added to
        # `_knowledge_attempts_to_query_rows`/`compact_records` can't silently
        # reopen the leak. Idempotent and scalar-preserving.
        _redact_solver_hidden_eval_fields(page)
        rows = _knowledge_attempts_to_query_rows(page)
    elif store is not None:
        rows = store.query_task_memory(task_id=task_id, experiment=experiment, limit=capped_limit)
    else:
        raise ValueError("task memory store unavailable")
    compact_records: list[dict[str, Any]] = []
    insights: list[dict[str, Any]] = []
    for row in rows:
        eval_results = row.get("eval_results") or {}
        tagged_meta = {
            "gen": row.get("gen"),
            "agent_id": row.get("agent_id"),
            "task_id": row.get("task_id"),
            "status": (eval_results.get("status") or eval_results.get("swebench_status") or ""),
            "resolved": bool(eval_results.get("resolved")) if eval_results else False,
            "native_score": eval_results.get("native_score"),
        }
        attempt_history = _redact_attempt_history_for_query(row.get("attempt_history"))
        compact_records.append(
            {
                "gen": row.get("gen"),
                "agent_id": row.get("agent_id"),
                "task_id": row.get("task_id"),
                "status": tagged_meta["status"],
                "resolved": tagged_meta["resolved"],
                "native_score": tagged_meta["native_score"],
                "full_memory_trace_condensed": _redact_solver_hidden_text(row.get("full_memory_trace_condensed") or ""),
                "task_specific_insights": row.get("task_specific_insights") or [],
                "attempt_count": len(attempt_history),
                "attempt_history": attempt_history,
                "updated_at": row.get("updated_at", ""),
            }
        )
        for text in row.get("task_specific_insights") or []:
            if isinstance(text, str) and text.strip():
                insights.append({"text": text.strip(), "meta": tagged_meta})
    # Deduplicate insights across generations
    deduped_insights: list[dict[str, Any]] = []
    accepted_texts: list[str] = []
    for ins in insights:
        text = ins.get("text", "")
        if not _is_redundant(text, accepted_texts):
            accepted_texts.append(text)
            deduped_insights.append(ins)
    insights = deduped_insights
    related: list[dict[str, Any]] = []
    semantic_query_text = ""
    semantic_error = ""
    retrieval_mode = "none"
    resolved_embedder = None
    if knowledge_store is not None:
        # Fall back to the task_id as retrieval text even without an embedder
        # so the FTS path still has something to match on.
        semantic_query_text = str(semantic_query or task_id).strip()
        task_id_fallback = not str(semantic_query or "").strip()
        if semantic_query_text and semantic_embedder is not None:
            resolved_embedder = _resolve_embedder(semantic_embedder)
    else:
        task_id_fallback = False
    semantic_available = _knowledge_vec_enabled(knowledge_store) and resolved_embedder is not None
    if semantic_query_text:
        if semantic_available:
            try:
                related = _redact_related_rows(
                    _drop_excluded_rows(
                        knowledge_store.vec_search(
                            resolved_embedder.embed(semantic_query_text),
                            max_results=5,
                            experiment=experiment,
                        )
                        or [],
                        exclude_task_ids,
                    )
                )
                retrieval_mode = "semantic"
            except Exception as exc:
                log.debug("semantic query failed for task_id=%s: %s", task_id, exc)
                semantic_error = str(exc)
                # Per-call fallback: a single embedding failure must not leave
                # agents with zero retrieval — degrade to lexical FTS.
                related, _ = _fts_fallback_related(
                    knowledge_store,
                    semantic_query_text,
                    max_results=5,
                    experiment=experiment,
                    exclude_task_ids=exclude_task_ids,
                    task_id_fallback=task_id_fallback,
                )
                retrieval_mode = "fts"
        else:
            # Embedder/vec unavailable (e.g. no HF_TOKEN): use the promised
            # FTS-only fallback rather than returning an empty related list.
            # Semantic was never attempted here, so leave ``semantic_error``
            # empty (it must mean "semantic failed", not "FTS failed"); the
            # ``retrieval_mode="fts"`` field already signals the degraded mode.
            related, _ = _fts_fallback_related(
                knowledge_store,
                semantic_query_text,
                max_results=5,
                experiment=experiment,
                exclude_task_ids=exclude_task_ids,
                task_id_fallback=task_id_fallback,
            )
            retrieval_mode = "fts"

    return {
        "task_id": task_id,
        "records": compact_records,
        "insights": insights,
        "related": related,
        "semantic_enabled": semantic_available,
        "retrieval_mode": retrieval_mode,
        "semantic_query": semantic_query_text,
        "semantic_result_count": len(related),
        "semantic_error": semantic_error,
    }


def handle_forum_read(
    *,
    forum_bus: ForumBus | None = None,
    round_num: int | None = None,
    up_to_round: bool = False,
    include_round0_from_store: bool = False,
    forum_store: Any = None,  # deprecated, ignored
    generation: int | None = None,
    experiment: str | None = None,
    exclude_task_ids: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    if forum_bus is None:
        messages = []
    else:
        messages = forum_bus.read_messages(round_num=round_num, up_to_round=up_to_round)
    if exclude_task_ids:
        messages = [message for message in messages if not (_forum_message_task_ids(message) & exclude_task_ids)]

    # Apply newest-first sort to avoid positional bias toward
    # earliest-posted messages.
    def _bus_sort_key(item: dict[str, Any]) -> tuple[int, int]:
        try:
            rn = int(item.get("round_num") or 0)
        except Exception:
            rn = 0
        try:
            mid = int(item.get("id") or 0)
        except Exception:
            mid = 0
        return rn, mid

    messages.sort(key=_bus_sort_key, reverse=True)
    return messages


# The hidden/safe field policy lives in kcsi.memory.parity (single source of
# truth, shared with the distillation prompts + condensed trace). These names are
# re-exported here under their historical ``_``-prefixed aliases so existing call
# sites and tests keep working. The rule: adaptive surfaces may contain only
# information from the declared phase/split
# feedback channel. For default upstream-strict no-feedback benchmarks (polyglot /
# SWE-bench / terminal_bench_2), hidden grader/test-runner/verifier material is
# outside that channel. The ARC ``arc_per_test`` allow-list projection drops the
# gold ``detail`` answer (and any future nested answer key) while keeping
# ``{test_index, correct}``.
try:
    from . import parity
except ImportError:
    import parity  # type: ignore[no-redef]

# Historical ``_``-prefixed aliases so existing call sites + contract tests keep
# importing these names from this module.
_HIDDEN_TEST_RUNNER_TAIL_KEYS = parity.HIDDEN_TEST_RUNNER_TAIL_KEYS
_HIDDEN_ATTEMPT_META_KEYS = parity.HIDDEN_ATTEMPT_META_KEYS
_redact_solver_hidden_eval_fields = parity.redact_solver_hidden_eval_fields
_redact_solver_hidden_text = parity.redact_solver_hidden_text

_RELATED_TEXT_FIELDS = (
    "approach",
    "outcome",
    "lessons",
    "text",
    "summary",
    "trace_condensed",
    "full_memory_trace_condensed",
    "output_summary",
)


def _redact_related_row(row: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(row)
    for key in _RELATED_TEXT_FIELDS:
        if isinstance(item.get(key), str):
            item[key] = _redact_solver_hidden_text(item[key])
    content = item.get("content")
    if isinstance(content, dict):
        _redact_solver_hidden_eval_fields({"attempts": [{"content": content}]})
    return item


def _redact_related_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [_redact_related_row(row) for row in rows if isinstance(row, dict)]


def _redact_attempt_history_for_query(history: Any) -> list[Any]:
    """Strip hidden grader-answer fields from legacy attempt_history payloads."""
    if not isinstance(history, list):
        return []
    redacted: list[Any] = []
    for item in history:
        if not isinstance(item, dict):
            redacted.append(item)
            continue
        event = copy.deepcopy(item)
        _redact_solver_hidden_eval_fields({"attempts": [{"content": {"eval_results": event}}]})
        nested_eval = event.get("eval_results")
        if isinstance(nested_eval, dict):
            _redact_solver_hidden_eval_fields({"attempts": [{"content": {"eval_results": nested_eval}}]})
        nested_meta = event.get("attempt_meta")
        if isinstance(nested_meta, dict):
            _redact_solver_hidden_eval_fields({"attempts": [{"content": {"attempt_meta": nested_meta}}]})
        redacted.append(event)
    return redacted


def _forum_message_task_ids(message: dict[str, Any]) -> set[str]:
    content = message.get("content") if isinstance(message, dict) else {}
    if not isinstance(content, dict):
        return set()
    ids: set[str] = set()
    task_id = str(content.get("task_id") or "").strip()
    if task_id:
        ids.add(task_id)
    task_ids = content.get("task_ids")
    if isinstance(task_ids, str):
        task_ids = [task_ids]
    if isinstance(task_ids, list):
        ids.update(str(tid).strip() for tid in task_ids if str(tid).strip())
    return ids


def handle_knowledge(
    *,
    knowledge_store: KnowledgeStore | None,
    task_id: str,
    include: str = "all",
    experiment: str | None = None,
    exclude_task_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Query unified knowledge page for a task."""
    if task_id in exclude_task_ids:
        return {"task_id": task_id, "attempts": [], "discussion": [], "insights": [], "distilled": []}

    if knowledge_store is None:
        return {"task_id": task_id, "attempts": [], "discussion": [], "insights": [], "distilled": []}

    entry_types = None
    if include == "attempts":
        entry_types = ["attempt"]
    elif include == "discussion":
        entry_types = ["post"]
    elif include == "insights":
        entry_types = ["insight"]
    elif include == "distilled":
        entry_types = ["distillation"]

    page = knowledge_store.query_task(
        task_id,
        entry_types=entry_types,
        experiment=experiment,
    )
    return _redact_solver_hidden_eval_fields(page)


def handle_forum_post(
    *,
    knowledge_store: KnowledgeStore | None,
    forum_bus: ForumBus | None,
    task_id: str,
    text: str,
    parent_post_id: int | None = None,
    agent_id: str,
    generation: int,
    experiment: str | None = None,
    round_num: int = 0,
    allowed_task_ids: set[str] | None = None,
    enforce_evidence: bool = True,
    exclude_task_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Post a message to a task discussion page.

    Writes to ForumBus only. The orchestrator drains ForumBus -> KnowledgeStore
    after the discussion round completes. When ``parent_post_id`` is supplied
    we also emit ``reply_to`` in the event content so the drain path can thread
    the reply under the parent post in ``knowledge.reply_to``.

    Grounding enforcement (§2.2 of the paper). When ``enforce_evidence`` is
    True and the post targets the cross-task page in round 1, the JSON body
    must declare a non-empty ``evidence_task_ids`` list, and every cited id
    must be in ``allowed_task_ids`` (the current generation's task set). This
    matches the protocol-level contract: candidate cross-task insights cannot
    enter downstream curation without verifiable per-task provenance.
    """
    normalized_task_id = str(task_id or "").strip()
    normalized_allowed_task_ids = (
        {str(tid).strip() for tid in allowed_task_ids if str(tid).strip()} if allowed_task_ids is not None else None
    )
    if normalized_task_id in exclude_task_ids:
        raise ValueError(f"forum_post target task_id is excluded from this forum session: {normalized_task_id}")
    if (
        normalized_allowed_task_ids is not None
        and normalized_task_id != "__cross_task__"
        and normalized_task_id not in normalized_allowed_task_ids
    ):
        raise ValueError(f"forum_post target task_id is not assigned to this forum session: {normalized_task_id}")

    # One post per agent per task per round (§ forum protocol: "Exactly one
    # post-mortem per task" / "Exactly ONE post per agent per round"). The
    # prompt instructs this but nothing enforced it server-side, so a
    # duplicate/near-duplicate tool call from the same agent for the same
    # task+round silently inflated that agent's weight in the distillation
    # input. Scoped to (agent_id, task_id, round_num) — a
    # different round or a different task from the same agent is unaffected.
    if forum_bus is not None:
        normalized_agent_id = str(agent_id or "")
        # Events from a failed forum-task attempt are marked stale by the
        # retry helper and must not count toward this guard: a
        # retry deliberately re-posts for the same (agent_id, task_id,
        # round_num) after its earlier attempt's post was marked stale, and
        # that legitimate re-post must not be rejected as a duplicate.
        try:
            stale_event_ids = forum_bus.read_stale_event_ids()
        except Exception:
            stale_event_ids = set()
            log.warning("forum_bus.read_stale_event_ids failed; skipping stale-skip in dedup guard", exc_info=True)
        for ev in forum_bus.read_events(round_num=round_num, message_types={"post"}):
            if ev.event_id in stale_event_ids:
                continue
            if (
                ev.agent_id == normalized_agent_id
                and str(ev.content.get("task_id") or "").strip() == normalized_task_id
            ):
                raise ValueError(
                    f"forum_post rejected: agent {normalized_agent_id!r} already posted to "
                    f"task_id={normalized_task_id!r} in round {round_num}. Exactly one post per "
                    "agent per task per round is allowed."
                )

    if enforce_evidence and round_num == 1 and normalized_task_id == "__cross_task__":
        body_text = str(text or "").strip()
        parsed: Any = None
        if body_text.startswith("{") and body_text.endswith("}"):
            try:
                parsed = json.loads(body_text)
            except (json.JSONDecodeError, ValueError):
                parsed = None
        if not isinstance(parsed, dict):
            raise ValueError(
                "Cross-task round-1 forum_post must carry a JSON object with "
                "non-empty evidence_task_ids; received non-JSON or non-object body."
            )
        evidence_ids_raw = parsed.get("evidence_task_ids")
        if isinstance(evidence_ids_raw, str):
            evidence_ids_raw = [evidence_ids_raw]
        if not isinstance(evidence_ids_raw, list):
            evidence_ids_raw = []
        normalized_evidence = [str(t).strip() for t in evidence_ids_raw if str(t).strip()]
        if not normalized_evidence:
            raise ValueError(
                "Cross-task round-1 forum_post requires non-empty evidence_task_ids "
                "in the JSON body (§2.2 grounding constraint)."
            )
        if normalized_allowed_task_ids:
            bad_ids = [tid for tid in normalized_evidence if tid not in normalized_allowed_task_ids]
            if bad_ids:
                raise ValueError(
                    f"Unknown evidence_task_ids: {bad_ids}. "
                    "Cite only tasks present in the current generation's evidence map."
                )
    # Coerce parent_post_id to int-or-None. Tool callers (LLMs) occasionally
    # emit the literal string "null"/"none" or a quoted integer ("5") despite
    # the integer schema; without coercion these land unchanged in the
    # ForumBus event and then in the INTEGER parent_id/reply_to columns,
    # orphaning the thread because no join can resolve a TEXT 'null' to an id.
    coerced_parent: int | None
    if parent_post_id is None:
        coerced_parent = None
    elif isinstance(parent_post_id, bool):  # bool subclasses int — reject
        coerced_parent = None
    elif isinstance(parent_post_id, int):
        coerced_parent = parent_post_id
    elif isinstance(parent_post_id, str):
        stripped = parent_post_id.strip()
        if not stripped or stripped.lower() in {"null", "none", "nil", "undefined"}:
            coerced_parent = None
        else:
            try:
                coerced_parent = int(stripped)
            except ValueError:
                coerced_parent = None
    else:
        coerced_parent = None

    entry_id = None
    if forum_bus is not None:
        content: dict[str, Any] = {
            "task_id": normalized_task_id,
            "text": text,
            "parent_post_id": coerced_parent,
        }
        if coerced_parent is not None:
            content["reply_to"] = coerced_parent
        result = forum_bus.append(
            round_num=round_num,
            agent_id=agent_id,
            message_type="post",
            content=content,
        )
        entry_id = result.get("event_id") if isinstance(result, dict) else None
        if getattr(forum_bus, "_read_only", False) or not entry_id:
            return {
                "status": "dropped_readonly",
                "entry_id": None,
                "task_id": normalized_task_id,
                "error": "forum_bus is read-only or write failed — post was not persisted",
            }
    return {"status": "ok", "entry_id": entry_id, "task_id": normalized_task_id}


def handle_forum_signal_done(
    *,
    knowledge_store: KnowledgeStore | None,
    forum_bus: ForumBus | None,
    agent_id: str,
    generation: int,
    task_ids: set[str],
    experiment: str | None = None,
    round_num: int = 0,
) -> dict[str, Any]:
    """Signal agent is done with all assigned discussion pages.

    Writes a ``done`` event to ForumBus (carrying task_ids so the
    orchestrator drain can persist them to ``discussion_done``) and, if
    the knowledge_store is writable, also writes directly. Containers
    typically hold a read-only KnowledgeStore, so the ForumBus path is the
    primary persistence route — the orchestrator drain is what actually
    lands the row.
    """
    task_id_list = sorted({str(t).strip() for t in (task_ids or set()) if str(t).strip()})
    log.info(
        "[MCP] forum_signal_done invoked agent=%s generation=%s round=%s task_count=%d experiment=%s",
        agent_id,
        generation,
        round_num,
        len(task_id_list),
        experiment or "",
    )

    ks_persisted = False
    if knowledge_store is not None and not getattr(knowledge_store, "_read_only", False):
        try:
            for tid in task_id_list:
                knowledge_store.signal_done(
                    task_id=tid,
                    agent_id=agent_id,
                    generation=generation,
                    experiment=experiment,
                )
            ks_persisted = True
        except Exception as exc:
            log.warning(
                "[MCP] forum_signal_done KnowledgeStore.signal_done failed agent=%s gen=%s: %s",
                agent_id,
                generation,
                exc,
            )
    elif knowledge_store is not None:
        log.info(
            "[MCP] forum_signal_done KnowledgeStore is read-only — deferring persistence "
            "to orchestrator drain agent=%s gen=%s",
            agent_id,
            generation,
        )

    bus_wrote = False
    if forum_bus is not None:
        try:
            # Include task_ids in the event content so the orchestrator's
            # _drain_forum_bus can resolve which discussion pages to mark
            # done even when the container's KnowledgeStore is read-only.
            forum_bus.append(
                round_num=round_num,
                agent_id=agent_id,
                message_type="done",
                content={"task_ids": task_id_list},
            )
            bus_wrote = not getattr(forum_bus, "_read_only", False)
        except Exception as exc:
            log.warning(
                "[MCP] forum_signal_done ForumBus.append failed agent=%s gen=%s: %s",
                agent_id,
                generation,
                exc,
            )
    log.info(
        "[MCP] forum_signal_done result agent=%s ks_persisted=%s bus_wrote=%s",
        agent_id,
        ks_persisted,
        bus_wrote,
    )
    return {
        "status": "done",
        "agent_id": agent_id,
        "ks_persisted": ks_persisted,
        "bus_wrote": bus_wrote,
        "task_ids": task_id_list,
    }


# Backward-compatible Python symbol for direct imports during the v2 -> canonical rename.
handle_forum_signal_done_v2 = handle_forum_signal_done


def _build_tools(toolset: str = "all") -> list[dict[str, Any]]:
    query_tool: dict[str, Any] = {
        "name": "query",
        "description": (
            "Query compact task-centric memory by task_id. Optionally provide "
            "query text for semantic vector search across related knowledge; "
            "semantic hits are returned in related."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Exact task memory page to retrieve."},
                "max_records": {"type": "integer"},
                "query": {
                    "type": "string",
                    "description": (
                        "Optional semantic search text. Use this to find related "
                        "attempts, insights, posts, or distilled knowledge even "
                        "when wording differs from the current task."
                    ),
                },
            },
            "required": ["task_id"],
        },
    }
    # NOTE: search tool removed — related task summaries are pre-injected
    # into the snapshot and surfaced via query()'s "related" field.
    knowledge_tool: dict[str, Any] = {
        "name": "knowledge",
        "description": "Query the knowledge page for a task. Returns prior attempts, discussion posts, insights, and distilled assets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task to look up"},
                "include": {
                    "type": "string",
                    "description": "Filter: all|attempts|discussion|insights|distilled (default: all)",
                },
            },
            "required": ["task_id"],
        },
    }
    forum_post_tool: dict[str, Any] = {
        "name": "forum_post",
        "description": (
            "Post a message to a task discussion page. Per-task posts require "
            "prior knowledge(task_id) and semantic query(task_id, query=...). "
            "Cross-task posts require prior query(task_id='__cross_task__', query=...). "
            "For cross-task round-1 posts, the text MUST be a JSON object containing "
            "a non-empty evidence_task_ids list whose entries are task ids in the "
            "current evidence map; ungrounded insights are rejected by the protocol."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task page to post on"},
                "text": {
                    "type": "string",
                    "description": (
                        "Your message. For cross-task round-1 posts, this MUST be a "
                        "JSON object with a non-empty evidence_task_ids list citing "
                        "tasks in the current evidence map."
                    ),
                },
                "parent_post_id": {
                    "type": "integer",
                    "description": "Optional: reply to a specific post ID. Threads the reply under that post.",
                },
            },
            "required": ["task_id", "text"],
        },
    }
    forum_signal_done_tool: dict[str, Any] = {
        "name": "forum_signal_done",
        "description": "Signal that you have finished contributing to the discussion.",
        "inputSchema": {"type": "object", "properties": {}},
    }
    forum_tools: list[dict[str, Any]] = [
        {
            "name": "forum_read",
            "description": "Read forum messages for current generation.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]
    knowledge_tools: list[dict[str, Any]] = [knowledge_tool, forum_post_tool, forum_signal_done_tool]
    if toolset == "task":
        # query() removed from task toolset — results pre-injected into memory_md
        return []
    if toolset == "memory":
        return []
    if toolset == "forum":
        return [query_tool] + forum_tools + knowledge_tools
    # "all" toolset: query (for forum) + forum tools + knowledge tools
    return [query_tool] + forum_tools + knowledge_tools


def _run_server(
    *,
    store: MemoryStore | None,
    docs_store: MemoryStore | None = None,  # deprecated, ignored
    forum_store: MemoryStore | None = None,  # deprecated, ignored
    snapshot: dict[str, Any] | None,
    forum_bus: ForumBus | None,
    forum_generation: int,
    forum_round: int,
    forum_agent_id: str,
    forum_expected_agents: int,
    memory_experiment: str,
    toolset: str,
    forum_task_ids: set[str],
    semantic_embedder: Any | None,
    knowledge_store: KnowledgeStore | None = None,
    enforce_forum_protocol: bool = True,
    exclude_task_ids: frozenset[str] = frozenset(),
) -> None:
    tools = _build_tools(toolset=toolset)
    allowed_tool_names = {str(t.get("name", "")) for t in tools if isinstance(t, dict)}
    forum_protocol = ForumProtocolState()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(msg, dict):
            _respond_error(None, -32600, "Invalid Request: frame must be a JSON object")
            continue

        method = msg.get("method")
        msg_id = msg.get("id")
        if method == "initialize":
            _respond(
                msg_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "memory", "version": "0.2.0"},
                },
            )
            continue
        if method == "notifications/initialized":
            continue
        if method == "tools/list":
            _respond(msg_id, {"tools": tools})
            continue
        if method != "tools/call":
            if msg_id is not None:
                _respond_error(msg_id, -32601, f"Unknown method: {method}")
            continue

        params = msg.get("params", {})
        if not isinstance(params, dict):
            _respond_error(msg_id, -32602, "Invalid params: 'params' must be a JSON object")
            continue
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if not isinstance(args, dict):
            _respond_error(msg_id, -32602, "Invalid params: 'arguments' must be a JSON object")
            continue
        try:
            if tool_name not in allowed_tool_names:
                _respond_error(msg_id, -32601, f"Unknown tool: {tool_name}")
                continue
            if tool_name == "query":
                query_task_id = str(args.get("task_id", ""))
                query_text = str(args.get("query", ""))
                if snapshot is not None:
                    result = _query_from_snapshot(
                        snapshot=snapshot,
                        task_id=query_task_id,
                        max_records=int(args.get("max_records", 8)),
                        experiment=memory_experiment or None,
                        knowledge_store=knowledge_store,
                        semantic_embedder=semantic_embedder,
                        semantic_query=query_text,
                        exclude_task_ids=exclude_task_ids,
                    )
                else:
                    result = handle_query(
                        store=store,
                        task_id=query_task_id,
                        max_records=int(args.get("max_records", 8)),
                        experiment=memory_experiment or None,
                        knowledge_store=knowledge_store,
                        semantic_embedder=semantic_embedder,
                        semantic_query=query_text,
                        exclude_task_ids=exclude_task_ids,
                    )
                forum_protocol.mark_query(query_task_id, query_text)
                _respond(msg_id, {"content": [{"type": "text", "text": json.dumps(result)}]})
            # NOTE: search handler removed — search tool not in any toolset.
            elif tool_name == "forum_read":
                result = handle_forum_read(
                    forum_bus=forum_bus,
                    round_num=forum_round or None,
                    up_to_round=(forum_round >= 2),
                    generation=forum_generation,
                    experiment=memory_experiment or None,
                    exclude_task_ids=exclude_task_ids,
                )
                _respond(msg_id, {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})
            elif tool_name == "knowledge":
                knowledge_task_id = str(args.get("task_id", ""))
                result = handle_knowledge(
                    knowledge_store=knowledge_store,
                    task_id=knowledge_task_id,
                    include=str(args.get("include", "all")),
                    experiment=memory_experiment or None,
                    exclude_task_ids=exclude_task_ids,
                )
                forum_protocol.mark_knowledge(knowledge_task_id)
                _respond(msg_id, {"content": [{"type": "text", "text": json.dumps(result)}]})
            elif tool_name == "forum_post":
                post_task_id = str(args.get("task_id", ""))
                if enforce_forum_protocol:
                    missing = forum_protocol.missing_for_post(post_task_id)
                    if missing:
                        raise ValueError(
                            "Forum protocol violation: before forum_post, call "
                            + " and ".join(missing)
                            + ". This retrieval-first sequence is task-agnostic and audited."
                        )
                result = handle_forum_post(
                    knowledge_store=knowledge_store,
                    forum_bus=forum_bus,
                    task_id=post_task_id,
                    text=str(args.get("text", "")),
                    parent_post_id=args.get("parent_post_id"),
                    agent_id=forum_agent_id,
                    generation=forum_generation,
                    experiment=memory_experiment or None,
                    round_num=forum_round,
                    allowed_task_ids=forum_task_ids,
                    enforce_evidence=enforce_forum_protocol,
                    exclude_task_ids=exclude_task_ids,
                )
                _respond(msg_id, {"content": [{"type": "text", "text": json.dumps(result)}]})
            elif tool_name == "forum_signal_done":
                result = handle_forum_signal_done(
                    knowledge_store=knowledge_store,
                    forum_bus=forum_bus,
                    agent_id=forum_agent_id,
                    generation=forum_generation,
                    task_ids=forum_task_ids,
                    experiment=memory_experiment or None,
                    round_num=forum_round,
                )
                _respond(msg_id, {"content": [{"type": "text", "text": json.dumps(result)}]})
            else:
                _respond_error(msg_id, -32601, f"Unknown tool: {tool_name}")
        except Exception as exc:  # noqa: BLE001
            _respond_error(msg_id, -32603, str(exc))


_VALID_TOOLSETS = {"all", "memory", "task", "forum"}


def _resolve_toolset() -> str:
    """Resolve ``MCP_TOOLSET`` from the environment, failing closed.

    An unrecognized value used to silently coerce to ``"all"`` -- granting the
    full tool surface on a typo. Unset/empty keeps the ``"all"``
    default; any other value must name a known toolset or the server refuses
    to start.
    """
    raw = os.environ.get("MCP_TOOLSET", "").strip().lower()
    if not raw:
        return "all"
    if raw not in _VALID_TOOLSETS:
        raise SystemExit(f"Invalid MCP_TOOLSET {raw!r}: must be one of {sorted(_VALID_TOOLSETS)}")
    return raw


def main() -> None:
    knowledge_db_path = os.environ.get("KNOWLEDGE_DB_PATH", "").strip()
    db_path = os.environ.get("RUNTIME_DB_PATH", "").strip()
    snapshot_path = os.environ.get("MEMORY_SNAPSHOT_PATH", "").strip()
    forum_generation = int(os.environ.get("FORUM_GENERATION", "0"))
    forum_round = int(os.environ.get("FORUM_ROUND", "0"))
    forum_agent_id = os.environ.get("FORUM_AGENT_ID", "")
    forum_expected_agents = int(os.environ.get("FORUM_EXPECTED_AGENTS", "0"))
    memory_experiment = os.environ.get("MEMORY_EXPERIMENT", "").strip()
    enable_semantic = str(os.environ.get("MEMORY_ENABLE_SEMANTIC_SEARCH", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    toolset = _resolve_toolset()
    forum_task_ids = {tid.strip() for tid in os.environ.get("FORUM_TASK_IDS", "").split(",") if tid.strip()}
    snapshot = _load_snapshot(snapshot_path)
    # Hold-out probe: ids whose knowledge rows must never surface in query
    # retrieval. Carried in the engine-written memory snapshot (the per-agent
    # config channel that already reaches this server) or, if ever needed,
    # the MEMORY_EXCLUDE_TASK_IDS env CSV.
    exclude_task_ids = _resolve_exclude_task_ids(snapshot)

    # MCP server runs inside task containers: prefer read-only SQLite mode so
    # containers never mutate schema/data and cannot corrupt host DB files.
    # If DB does not exist yet (tests/local bootstrap), fall back to rw init.
    store: MemoryStore | None = None
    if snapshot is None and db_path:
        db_exists = Path(db_path).exists()
        store = MemoryStore(
            db_path,
            default_experiment=memory_experiment or "__mcp__",
            read_only=db_exists,
        )
    semantic_embedder = _get_embedder if enable_semantic else None
    experiment_name = os.environ.get("EXPERIMENT_NAME", "").strip() or memory_experiment or "__mcp__"
    # Create ForumBus for forum toolset (always) or for any toolset when
    # forum_generation > 0 (enables container -> orchestrator drain path).
    _want_forum_bus = toolset in {"all", "forum"} or forum_generation > 0
    forum_bus: ForumBus | None = None
    if _want_forum_bus and knowledge_db_path:
        try:
            forum_bus = ForumBus(
                db_path=knowledge_db_path,
                experiment=experiment_name,
                generation=forum_generation,
            )
        except Exception:
            pass

    # Initialize KnowledgeStore (authoritative swarm memory/state access)
    knowledge_store: KnowledgeStore | None = None
    if knowledge_db_path and Path(knowledge_db_path).exists():
        try:
            knowledge_store = KnowledgeStore(
                knowledge_db_path,
                read_only=True,
                default_experiment=memory_experiment or "__mcp__",
                enable_vec=enable_semantic,
            )
        except Exception as exc:
            log.warning("KnowledgeStore unavailable: %s", exc)

    # Report which retrieval mode the agent-facing ``query`` tool will actually
    # use, so a run without a working embedder is visibly degraded rather than
    # silently empty. Semantic requires both the env flag AND a live vec index;
    # otherwise we fall back to lexical FTS5 (always available on the knowledge
    # DB via content-sync triggers).
    if not enable_semantic:
        retrieval_mode_active = "fts (semantic disabled via MEMORY_ENABLE_SEMANTIC_SEARCH)"
    elif _knowledge_vec_enabled(knowledge_store):
        retrieval_mode_active = "semantic (sqlite-vec) with FTS5 fallback per query"
    elif knowledge_store is not None:
        retrieval_mode_active = "fts (FTS5 only — vector index unavailable, e.g. no HF_TOKEN)"
    else:
        retrieval_mode_active = "none (no knowledge store)"
    log.info("[MCP] agent retrieval mode: %s", retrieval_mode_active)

    try:
        _run_server(
            store=store,
            snapshot=snapshot,
            forum_bus=forum_bus,
            forum_generation=forum_generation,
            forum_round=forum_round,
            forum_agent_id=forum_agent_id,
            forum_expected_agents=forum_expected_agents,
            memory_experiment=memory_experiment,
            toolset=toolset,
            forum_task_ids=forum_task_ids,
            semantic_embedder=semantic_embedder,
            knowledge_store=knowledge_store,
            enforce_forum_protocol=_env_flag("MEMORY_ENFORCE_FORUM_PROTOCOL", True),
            exclude_task_ids=exclude_task_ids,
        )
    finally:
        if store is not None:
            store.close()
        if knowledge_store is not None:
            knowledge_store.close()


if __name__ == "__main__":
    main()
