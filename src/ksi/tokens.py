"""Token usage accounting.

Semantic note (2026-04 cache-inclusive fix)
-------------------------------------------
``TokenUsage.total`` (and every downstream surface that exposes it —
``to_dict()["total_tokens"]``, ``TokenAccumulator.total()``, the
``agent.token_usage`` counter in ``src/ksi/models.py``, and the
``output_json`` artifact emitted by ``ksi.cli``) includes **all four**
token buckets:

    total = input + output + cache_read + cache_creation

Before this fix, cache tokens were silently excluded from the "total",
which could undercount billed volume by >95 % on cache-heavy runs.
The field name ``total_tokens`` is project-defined; Anthropic itself
does not emit a single "total_tokens" on its ``usage`` blocks — it
returns the four bucket counts independently.

Callers who want the *legacy* input+output figure can add
``input_tokens + output_tokens`` (or subtract
``cache_read_input_tokens + cache_creation_input_tokens`` from
``total_tokens`` in a serialized dict).  Analysis scripts that used to
compute ``total - (input + output)`` expecting 0 will now get
``cache_read + cache_creation``.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, TypedDict

log = logging.getLogger(__name__)


class TokenUsageDict(TypedDict):
    """Serialized shape returned by :meth:`TokenUsage.to_dict`.

    Every key is always present (``to_dict`` emits the cache buckets even when
    zero, see its docstring), so this is a total TypedDict. ``total_tokens`` is
    cache-inclusive; consumers wanting the legacy input+output figure must read
    ``input_tokens`` and ``output_tokens`` directly.
    """

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    uncached_input_tokens: int
    total_tokens: int


# ── Pricing (per million tokens) ─────────────────────────────────────────────
# Audited against current official pricing (see PRICES_AS_OF). The rates below
# are EXACT published rates, not approximations, and match the providers'
# documented cache economics.
#
# Sources (retrieved 2026-06-05):
#   Anthropic — https://platform.claude.com/docs/en/about-claude/models/overview
#               and https://claude.com/pricing#api
#   OpenAI   — https://openai.com/api/pricing/ (Standard tier) and
#              https://developers.openai.com/api/docs/pricing
#
# Anthropic cache economics (documented, not estimated):
#   `cache_create` = 5-minute cache-WRITE rate = 1.25x base input.
#       1-hour writes (2x base input) are a different TTL we do not request, so
#       they are intentionally not modeled — every cache_control breakpoint in
#       this repo uses the default 5-minute ephemeral TTL.
#   `cache_read` = cache-HIT rate = 0.1x base input (the standard 90% discount).
#       Each entry stores the literal product (e.g. opus 0.1*5.00 = 0.50) so the
#       cost function stays a flat lookup.
#
# Bare family keys (haiku/sonnet/opus) represent CURRENT-generation pricing
# and are used as fallbacks for unversioned strings. Version-specific keys
# below override via longest-match-first when generations diverge — Anthropic
# split the Opus tier between 4.1 ($15/$75) and 4.5+ ($5/$25, held through
# 4.6/4.7/4.8), and Haiku diverged between 3.5 ($0.80/$4) and 4.5 ($1/$5).
# Sonnet has held $3/$15 across 3.5/4/4.5/4.6, so no version-specific Sonnet
# keys are needed.
#
# OpenAI: `cache_read` is the documented cached-input rate (0.1x input for the
# gpt-5 / gpt-5.x families; 0.5x input for the older gpt-4o / gpt-4.1 / o-series
# generations). For models with a long-context tier (>272K), the short-context
# rate is canonical here — long-context calls would underprice but we don't
# currently route those. `cache_create` is 0 across OpenAI because the platform
# auto-caches server-side without a separate write charge. Pro tiers
# (5-pro / 5.2-pro / 5.4-pro / 5.5-pro / o1-pro / o3-pro) don't offer prompt
# caching, so cache_read=0 there is correct (any usage payload carries 0 cached
# tokens).
#
# Entries tagged `# UNVERIFIED (as of 2026-06-05)` could not be confirmed
# against a current published table (OpenAI has removed these retired/legacy
# models from its active pricing pages); their pre-audit values are retained.
# Reproducible models actually run by this repo are all verified — see
# configs/ksi/*, benchmarks/*.sh, scripts/run_ksi.sh.
PRICES_AS_OF = "2026-06-05"

PRICING: dict[str, dict[str, float]] = {
    # Anthropic — bare family keys = current-generation pricing.
    "haiku": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_create": 1.25},  # 4.5
    "sonnet": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_create": 3.75},  # 3.5/4/4.5/4.6 same
    "opus": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_create": 6.25},  # 4.5/4.6/4.7
    # Pre-4.x Haiku versions (legacy pricing; both naming conventions covered).
    "haiku-3-5": {"input": 0.80, "output": 4.00, "cache_read": 0.08, "cache_create": 1.00},
    "3-5-haiku": {"input": 0.80, "output": 4.00, "cache_read": 0.08, "cache_create": 1.00},
    "claude-haiku-3": {"input": 0.25, "output": 1.25, "cache_read": 0.03, "cache_create": 0.30},
    "3-haiku": {"input": 0.25, "output": 1.25, "cache_read": 0.03, "cache_create": 0.30},
    # Pre-4.5 Opus versions (the $15/$75 tier — Opus 3, 4, 4.1).
    "opus-4-1": {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_create": 18.75},
    "opus-4": {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_create": 18.75},
    "claude-opus-3": {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_create": 18.75},
    "3-opus": {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_create": 18.75},
    # gpt-5.5 family
    "gpt-5.5-pro": {"input": 30.00, "output": 180.00, "cache_read": 0.0, "cache_create": 0.0},
    "gpt-5.5": {"input": 5.00, "output": 30.00, "cache_read": 0.50, "cache_create": 0.0},
    # gpt-5.4 family (project default for OpenAI runs)
    "gpt-5.4-pro": {"input": 30.00, "output": 180.00, "cache_read": 0.0, "cache_create": 0.0},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25, "cache_read": 0.02, "cache_create": 0.0},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cache_read": 0.075, "cache_create": 0.0},
    "gpt-5.4": {"input": 2.50, "output": 15.00, "cache_read": 0.25, "cache_create": 0.0},
    # gpt-5.2 family
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00, "cache_read": 0.0, "cache_create": 0.0},
    "gpt-5.2": {"input": 1.75, "output": 14.00, "cache_read": 0.175, "cache_create": 0.0},
    # gpt-5.1 / gpt-5 family
    "gpt-5.1-codex-mini": {"input": 0.25, "output": 2.00, "cache_read": 0.025, "cache_create": 0.0},
    "gpt-5.1-codex": {"input": 1.25, "output": 10.00, "cache_read": 0.125, "cache_create": 0.0},
    "gpt-5.1": {"input": 1.25, "output": 10.00, "cache_read": 0.125, "cache_create": 0.0},
    "gpt-5-pro": {"input": 15.00, "output": 120.00, "cache_read": 0.0, "cache_create": 0.0},
    "gpt-5-nano": {"input": 0.05, "output": 0.40, "cache_read": 0.005, "cache_create": 0.0},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "cache_read": 0.025, "cache_create": 0.0},
    "gpt-5": {"input": 1.25, "output": 10.00, "cache_read": 0.125, "cache_create": 0.0},
    # gpt-4.1 family
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cache_read": 0.025, "cache_create": 0.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cache_read": 0.10, "cache_create": 0.0},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cache_read": 0.50, "cache_create": 0.0},
    # gpt-4o family
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read": 0.075, "cache_create": 0.0},
    "gpt-4o": {"input": 2.50, "output": 10.00, "cache_read": 1.25, "cache_create": 0.0},
    # o-series reasoning models. o3 ($2/$8) and o4-mini ($1.10/$4.40) base rates
    # verified 2026-06-05; o3 cache_read = 0.25x input is the current o-series
    # cached-input rate. The retired models below (o3-mini, o3-pro, o1*) have
    # been removed from OpenAI's active pricing pages and could not be
    # re-confirmed, so their pre-audit values are retained as-is.
    "o4-mini": {"input": 1.10, "output": 4.40, "cache_read": 0.275, "cache_create": 0.0},
    "o3-pro": {
        "input": 20.00,
        "output": 80.00,
        "cache_read": 0.0,
        "cache_create": 0.0,
    },  # UNVERIFIED (as of 2026-06-05)
    "o3-mini": {
        "input": 1.10,
        "output": 4.40,
        "cache_read": 0.55,
        "cache_create": 0.0,
    },  # UNVERIFIED (as of 2026-06-05)
    "o3": {"input": 2.00, "output": 8.00, "cache_read": 0.50, "cache_create": 0.0},
    "o1-pro": {
        "input": 150.00,
        "output": 600.00,
        "cache_read": 0.0,
        "cache_create": 0.0,
    },  # UNVERIFIED (as of 2026-06-05)
    "o1-mini": {
        "input": 1.10,
        "output": 4.40,
        "cache_read": 0.55,
        "cache_create": 0.0,
    },  # UNVERIFIED (as of 2026-06-05)
    "o1": {"input": 15.00, "output": 60.00, "cache_read": 7.50, "cache_create": 0.0},  # UNVERIFIED (as of 2026-06-05)
}

_PROVIDER_PREFIXES: tuple[str, ...] = (
    "anthropic/",
    "openai/",
    "azure/global/",
    "azure/eu/",
    "azure/",
    "bedrock/",
    "vertex_ai/",
)

# Matches a "-N" or "-NN" sub-version suffix that is NOT followed by another
# digit (which would make it a date like "-2024-..." rather than a version).
# Used to reject `opus-4` matching inside `opus-4-7` while still allowing
# `gpt-5` to match inside `gpt-5-2025-08-07`.
_SUBVERSION_TAIL_RE = re.compile(r"^-\d{1,2}(?!\d)")


def pricing_for_model(model: str) -> dict[str, float] | None:
    """Match a model name against PRICING, handling provider prefixes.

    Strips litellm/Bedrock provider prefixes (``openai/``, ``azure/global/``,
    ``bedrock/``, ``anthropic/``, …) and matches the longest PRICING key that
    appears as a substring. Two rules guard against substring leakage:

    * The character immediately after the match cannot be a digit or ``.``,
      so ``gpt-5`` does not silently match inside ``gpt-5.4-mini``.
    * A version-bearing key (one that contains a digit, like ``opus-4`` or
      ``haiku-3-5``) is also rejected if the tail starts with a sub-version
      suffix ``-N``/``-NN``. This prevents ``opus-4`` from matching
      ``claude-opus-4-7`` (whose ``-7`` is a sub-version, not a date).
      Bare family keys (``haiku``/``sonnet``/``opus``) skip this check —
      they intentionally match unversioned and unknown-version strings,
      treating them as current-generation pricing.

    Returns ``None`` when no key matches; ``cost_usd`` callers convert that
    to ``0.0`` to preserve "unknown model → no cost" semantics.
    """
    if not model:
        return None
    name = model.lower()
    for prefix in _PROVIDER_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for key in sorted(PRICING.keys(), key=len, reverse=True):
        idx = name.find(key)
        if idx < 0:
            continue
        tail = name[idx + len(key) :]
        if tail and (tail[0].isdigit() or tail[0] == "."):
            continue
        if any(c.isdigit() for c in key) and _SUBVERSION_TAIL_RE.match(tail):
            continue
        return PRICING[key]
    return None


@dataclass
class TokenUsage:
    """Token consumption counters for a single LLM call or aggregated span."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        """Total billed tokens: input + output + cache_read + cache_creation.

        Anthropic bills all four token buckets independently, so a faithful
        "total tokens consumed" must sum all of them. Prior to the cache-aware
        fix, this excluded both cache buckets, which undercounted real volume
        by >95% on cache-heavy runs (hundreds of millions of cache_read /
        cache_creation tokens can be invisible in summaries).

        For the legacy ``input + output`` (pre-cache) figure, add
        ``input_tokens + output_tokens`` directly.
        """
        return self.input_tokens + self.output_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def uncached_input_tokens(self) -> int:
        """Prompt/input tokens not served from prompt cache."""
        return self.input_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    def cost_usd(self, model: str) -> float:
        """Compute cost in USD using cache-aware pricing.

        *model* is matched against PRICING keys via :func:`pricing_for_model`,
        which strips provider prefixes (``openai/``, ``bedrock/`` …) and uses
        longest-match-first to avoid substring leakage. Unknown model families
        return 0.0 rather than being mispriced with another vendor's rate.
        """
        p = pricing_for_model(model)
        if p is None:
            return 0.0
        return (
            self.uncached_input_tokens * p["input"]
            + self.cache_read_input_tokens * p["cache_read"]
            + self.cache_creation_input_tokens * p["cache_create"]
            + self.output_tokens * p["output"]
        ) / 1_000_000

    def to_dict(self) -> TokenUsageDict:
        """Serialize to a plain dict suitable for JSON artifacts.

        The returned dict always contains the raw usage buckets plus
        ``uncached_input_tokens`` and ``total_tokens``. The cache fields
        are emitted even when zero so downstream consumers can rely on
        their presence.

        .. warning::
            ``total_tokens`` reflects the **cache-inclusive** total
            (input + output + cache_read + cache_creation), not the
            legacy input+output figure.  Consumers that previously
            asserted ``total == input + output`` must read
            ``input_tokens`` and ``output_tokens`` directly.  See the
            module docstring for the full semantic history.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "total_tokens": self.total,
        }


@dataclass(frozen=True)
class LLMResponse:
    """Uniform return type for :meth:`ksi.protocols.LLMCaller.call`.

    Replaces the former variable-arity ``tuple[str, TokenUsage]`` /
    ``tuple[str, TokenUsage, dict | None]`` return so every caller has a single,
    statically-typed shape.

    - ``text`` is always the model's textual output (guaranteed-valid JSON when
      a ``json_schema`` was requested and the provider returned a tool/structured
      object).
    - ``usage`` is the token accounting for the call.
    - ``parsed`` carries the provider-validated structured dict when
      ``json_schema=`` was requested and the provider returned a parseable
      object, else ``None`` (always ``None`` on non-structured calls).
    """

    text: str
    usage: TokenUsage
    parsed: dict[str, Any] | None = None


@dataclass
class TokenAccumulator:
    """Accumulates token usage across generations, agents, and tasks."""

    # Unified store keyed by (generation, agent_id, source_id).
    # Task entries use the task_id as source_id.
    # Lifecycle entries use "__lc:<call_type>" as source_id.
    _entries: dict[tuple[int, str, str], TokenUsage] = field(default_factory=dict)
    _flushed_keys: set[tuple[int, str, str]] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_task(self, generation: int, agent_id: str, task_id: str, usage: TokenUsage) -> None:
        with self._lock:
            key = (generation, agent_id, task_id)
            existing = self._entries.get(key, TokenUsage())
            self._entries[key] = existing + usage

    def record_lifecycle(self, generation: int, agent_id: str, call_type: str, usage: TokenUsage) -> None:
        with self._lock:
            key = (generation, agent_id, f"__lc:{call_type}")
            existing = self._entries.get(key, TokenUsage())
            self._entries[key] = existing + usage

    def total(self) -> TokenUsage:
        # Snapshot under the lock so a concurrent writer resizing _entries
        # cannot raise "dictionary changed size during iteration"; sum the
        # cheap snapshot outside the critical section.
        with self._lock:
            values = list(self._entries.values())
        result = TokenUsage()
        for u in values:
            result = result + u
        return result

    def by_agent(self) -> dict[str, TokenUsage]:
        with self._lock:
            items = list(self._entries.items())
        result: dict[str, TokenUsage] = {}
        for (_, agent_id, _), u in items:
            result[agent_id] = result.get(agent_id, TokenUsage()) + u
        return result

    def by_generation(self) -> dict[int, TokenUsage]:
        with self._lock:
            items = list(self._entries.items())
        result: dict[int, TokenUsage] = {}
        for (gen, _, _), u in items:
            result[gen] = result.get(gen, TokenUsage()) + u
        return result

    def flush_to_store(self, store: Any, run_id: int, generation: int, model: str) -> None:
        """Persist token entries for *generation* to the DB.

        Each entry is written as a row in the ``token_phases`` table.  The
        *phase* is derived from the source_id: lifecycle keys (``__lc:<type>``)
        map to the call_type; everything else is ``task_execution``.

        Entries remain in ``_entries`` so that ``total()`` and ``to_dict()``
        still reflect the full run after flushing.  Already-flushed keys are
        tracked in ``_flushed_keys`` to avoid duplicate DB writes on
        subsequent calls.
        """
        with self._lock:
            for (gen, agent_id, source_id), usage in self._entries.items():
                if gen != generation:
                    continue
                key = (gen, agent_id, source_id)
                if key in self._flushed_keys:
                    continue
                if source_id.startswith("__lc:"):
                    phase = source_id[len("__lc:") :]
                else:
                    phase = "task_execution"
                try:
                    store.insert_token_phase(
                        run_id=run_id,
                        generation=generation,
                        phase=phase,
                        agent_ref=agent_id,
                        token_usage=usage,
                        cost_usd=usage.cost_usd(model),
                    )
                except Exception:
                    log.warning(
                        "Failed to flush token entry gen=%d agent=%s source=%s",
                        gen,
                        agent_id,
                        source_id,
                        exc_info=True,
                    )
                    continue
                self._flushed_keys.add(key)

    def load_from_store(
        self,
        store: Any,
        *,
        experiment: str | None = None,
        before_generation: int,
    ) -> int:
        """Rehydrate entries from previously-flushed ``token_phases`` rows.

        On ``--resume`` the accumulator starts empty (``run()`` constructs a
        fresh instance), so ``total()`` would undercount every generation
        completed before the resume cursor.  This replays the
        persisted rows for generations ``< before_generation`` back into
        ``_entries`` so ``token_usage_total`` reflects the full run.

        Rows are keyed back to the same ``(generation, agent_id, source_id)``
        shape :meth:`flush_to_store` wrote them from: lifecycle phases become
        ``__lc:<phase>`` source_ids; everything else collapses under
        ``task_execution`` (the per-task source_id is not persisted and is not
        needed for any ``total``/``by_generation``/``by_agent`` view).
        Rehydrated keys are marked flushed so a later flush never re-writes
        them.  Returns the number of rows replayed.

        Rows are parsed into a local staging dict *before* the lock is taken, so
        a malformed row raises before any mutation — the accumulator is never
        left half-populated (it stays empty rather than partially rehydrated).
        """
        try:
            rows = store.get_token_phases(experiment=experiment, before_generation=before_generation)
        except Exception:
            log.warning("Failed to read token_phases for resume rehydration", exc_info=True)
            return 0
        # Parse first (may raise on a corrupt row); only mutate shared state once
        # the whole batch is known-good.
        staged: dict[tuple[int, str, str], TokenUsage] = {}
        for row in rows:
            gen = int(row.get("generation") or 0)
            agent_id = str(row.get("agent_ref") or "")
            phase = str(row.get("phase") or "task_execution")
            source_id = f"__lc:{phase}" if phase != "task_execution" else "task_execution"
            usage = TokenUsage(
                input_tokens=int(row.get("input_tokens") or 0),
                output_tokens=int(row.get("output_tokens") or 0),
                cache_creation_input_tokens=int(row.get("cache_creation_tokens") or 0),
                cache_read_input_tokens=int(row.get("cache_read_tokens") or 0),
            )
            key = (gen, agent_id, source_id)
            staged[key] = staged.get(key, TokenUsage()) + usage
        with self._lock:
            for key, usage in staged.items():
                self._entries[key] = self._entries.get(key, TokenUsage()) + usage
                self._flushed_keys.add(key)
        return len(rows)

    def to_dict(self) -> dict[str, Any]:
        t = self.total()
        return {
            "total": t.to_dict(),
            "by_agent": {k: v.to_dict() for k, v in sorted(self.by_agent().items())},
            "by_generation": {k: v.to_dict() for k, v in sorted(self.by_generation().items())},
        }
