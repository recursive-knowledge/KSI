"""Shared exception types used across ksi layers.

Kept in a leaf module so orchestration, distillation, and runtime code
can all import the same symbols without creating circular dependencies.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import cast

type _MarkerCategories = dict[str, list[str] | tuple[str, ...]]

# ---------------------------------------------------------------------------
# Retryable-error markers: single source of truth
#
# The substring markers used to classify a task/provider error as transient
# (retry) vs non-retryable (abort) live in a JSON file shared with the
# TypeScript agent-runner: ``runtime_runner/shared/retryable_markers.json``.
# Keeping one categorized list avoids the historical drift where a provider
# error-wording change silently flipped transient<->non-retryable because the
# Python list and the TS emitter fell out of lockstep.
#
# ``load_retryable_markers()`` resolves the JSON relative to the repo root and
# caches it. If the file is missing or malformed it falls back to the vendored
# copy below so importing this module can never crash a run. The vendored copy
# MUST mirror the JSON byte-for-byte (the parity test pins this).
# ---------------------------------------------------------------------------

# Path is resolved relative to this file's repo root (``ksi/`` is one level
# below the root), mirroring ``src/ksi/layout.py::PROJECT_ROOT``. Resolved here
# rather than imported to keep this leaf module dependency-free.
_RETRYABLE_MARKERS_PATH = Path(__file__).resolve().parents[2] / "runtime_runner" / "shared" / "retryable_markers.json"

# Vendored fallback — kept in lockstep with retryable_markers.json by
# tests/test_retryable_markers.py. Do not edit one without the other.
_VENDORED_RETRYABLE_MARKERS: dict[str, object] = {
    "schema_version": 1,
    "categories": {
        "non_retryable": (
            "invalid prompt",
            "usage policy",
            "flagged as potentially violating",
            "no patch",
            "missing report",
            "parse_error",
        ),
        "non_retryable_exit_codes": (
            "exit=137",
            "exit=139",
            "exit=126",
            "exit=127",
        ),
        "upstream_provider_transient": (
            "429",
            "502",
            "503",
            "504",
            "rate limit",
            "too many requests",
            "service unavailable",
            "provider unavailable",
            "internal server error",
            "gateway timeout",
            "bad gateway",
            "upstream connect",
            "connection termination",
            "overloaded",
            "fetch failed",
            "headers timeout",
            "headerstimeouterror",
        ),
        "transient_extra": (
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "broken pipe",
            "temporarily unavailable",
            "temporary failure",
            "network is unreachable",
            "econnreset",
            "eai_again",
            "resource temporarily unavailable",
        ),
        "stream_race": (
            "sdk query loop drained",
            "sdk query iterator threw",
            "sdk emitted an empty result event",
            "silent agent-runner failure",
        ),
    },
}

# The category names every consumer (engine.py's module-level tuples,
# ``transient_markers()``) requires. Derived from the vendored copy so the
# expected set stays single-sourced. A loaded JSON whose shape differs is
# treated as malformed and the vendored fallback is used wholesale — see
# ``load_retryable_markers()``.
_EXPECTED_MARKER_CATEGORIES = frozenset(cast(_MarkerCategories, _VENDORED_RETRYABLE_MARKERS["categories"]))


def _validated_categories(categories: object) -> dict[str, list[str]]:
    """Return ``categories`` if it is a structurally-valid marker mapping.

    Raises ``ValueError`` otherwise so ``load_retryable_markers()`` can fall
    back to the vendored copy. "Valid" means: a dict whose keys are exactly
    ``_EXPECTED_MARKER_CATEGORIES`` and whose every value is a non-empty list
    of non-empty strings. This is intentionally strict — a JSON that parses but
    drops/renames a category, or carries an empty list, would otherwise crash
    import (``KeyError`` in ``transient_markers()``) or silently classify with
    fewer markers. Both failure modes are worse than using the vendored copy.
    """

    if not isinstance(categories, dict) or not categories:
        raise ValueError("retryable_markers.json missing 'categories'")
    if set(categories) != _EXPECTED_MARKER_CATEGORIES:
        raise ValueError(
            f"retryable_markers.json categories {sorted(categories)} != expected {sorted(_EXPECTED_MARKER_CATEGORIES)}"
        )
    for name, markers in categories.items():
        if not isinstance(markers, list) or not markers:
            raise ValueError(f"retryable_markers.json category {name!r} is not a non-empty list")
        if not all(isinstance(m, str) and m for m in markers):
            raise ValueError(f"retryable_markers.json category {name!r} has a non-string/empty marker")
    return categories


@lru_cache(maxsize=1)
def load_retryable_markers() -> dict[str, tuple[str, ...]]:
    """Return retryable-error markers keyed by semantic category.

    Loads ``runtime_runner/shared/retryable_markers.json`` (the single source
    of truth shared with the TypeScript agent-runner) and returns each
    category's markers as a tuple of lowercase substrings. Falls back to the
    vendored copy (``_VENDORED_RETRYABLE_MARKERS``) if the file is missing,
    unparseable, or structurally invalid (wrong category set, empty/non-list
    value, non-string marker) so that import/classification never crashes a run
    and never silently loads a partial marker set.

    The result is cached for the process lifetime.
    """

    raw: dict[str, object]
    try:
        with _RETRYABLE_MARKERS_PATH.open(encoding="utf-8") as fh:
            loaded = json.load(fh)
        raw = {"categories": _validated_categories(loaded.get("categories"))}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raw = _VENDORED_RETRYABLE_MARKERS

    categories = cast(_MarkerCategories, raw["categories"])
    return {name: tuple(str(marker) for marker in markers) for name, markers in categories.items()}


def non_retryable_markers() -> tuple[str, ...]:
    """Deterministic non-retryable error substrings (e.g. usage policy)."""

    return load_retryable_markers()["non_retryable"]


def non_retryable_exit_code_markers() -> tuple[str, ...]:
    """Container exit-code markers that indicate non-transient failures."""

    return load_retryable_markers()["non_retryable_exit_codes"]


def upstream_provider_transient_markers() -> tuple[str, ...]:
    """Upstream LLM-provider transient signatures (5xx, rate limit, etc.).

    These are provider-side and retryable even without execution evidence.
    """

    return load_retryable_markers()["upstream_provider_transient"]


def transient_markers() -> tuple[str, ...]:
    """All transient substrings: provider + network + SDK stream-race phrases.

    Equivalent to the historical ``_TRANSIENT_TASK_ERROR_MARKERS`` tuple:
    ``upstream_provider_transient + transient_extra + stream_race``.
    """

    markers = load_retryable_markers()
    return markers["upstream_provider_transient"] + markers["transient_extra"] + markers["stream_race"]


# Literal substring markers that are safe to match anywhere in the error
# message — they can't appear as incidental substrings of normal identifiers
# (task ids, repo slugs, commit hashes).
_AUTH_ERROR_SUBSTRINGS: tuple[str, ...] = (
    "authenticationerror",
    "authentication_error",
    "invalid_api_key",
    "invalid api key",
    "x-api-key",
    "unauthorized",
)

# HTTP status / numeric markers require word-boundary matching so that
# commit-hash characters like "...83531fe401..." inside a SWE-bench Pro task id
# don't trick us into aborting the whole run as an auth failure. Without it, a
# bare "401"/"403" substring inside a task id could trigger a false
# ``AuthenticationFailure`` that aborts the run mid-sweep.
_AUTH_ERROR_TOKEN_RE = re.compile(r"\b(?:401|403)\b")


class KsiError(Exception):
    """Base class for every exception ksi raises.

    Programmatic callers of :func:`ksi.run` (and the registry/extension API)
    can ``except KsiError`` to catch any ksi-originated failure without
    enumerating the concrete types. Concrete exceptions keep their historical
    second base (``RuntimeError`` / ``ValueError``) so existing
    ``except RuntimeError`` / ``except ValueError`` handlers are unaffected.
    """


class AuthenticationFailure(KsiError, RuntimeError):
    """Raised when the LLM provider reports an auth failure.

    Auth failures are never transient from the provider's perspective — no
    amount of retrying with the same credentials will succeed. Surfacing as
    a dedicated exception lets the orchestrator abort the run instead of
    silently degrading every task or phase to an empty result.
    """


class ContainerRegistryError(KsiError, RuntimeError):
    """Raised when a container image cannot be acquired from its registry.

    The type identifies the failure's origin; it never means LLM-provider
    authentication. Retryability is explicit because registry failures include
    both transient transport/service errors and deterministic image/configuration
    errors. ``image`` and ``reason`` provide stable metadata without requiring
    callers to parse the human-readable message.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        reason: str = "unknown",
        image: str = "",
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.reason = str(reason or "unknown")
        self.image = str(image or "")


class DistillationStalledError(KsiError, RuntimeError):
    """Raised to abort a run after N consecutive fully-zeroed distill generations.

    Opt-in via ``--abort-on-distill-stall N`` (0 disables). A sustained
    host->provider outage that retry cannot ride out zeroes distillation every
    generation while attempts keep spending compute for no learning;
    when the operator has asked for it, surfacing a dedicated exception lets the
    engine abort the run instead of burning the rest of the campaign.
    """


class WriteIndeterminateError(KsiError, RuntimeError):
    """Raised when a queued DB write timed out while already executing.

    The write could not be cancelled and may still be applied after the
    caller gives up, so retrying it can duplicate rows.
    Best-effort callers must drop the write instead of retrying it.
    """


def exception_chain(exc: BaseException, *, max_depth: int = 10) -> tuple[BaseException, ...]:
    """Return ``exc`` plus its explicit cause/context, bounded and cycle-safe."""

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < max(1, max_depth):
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return tuple(chain)


def find_container_registry_error(exc: BaseException) -> ContainerRegistryError | None:
    """Return the nearest typed registry failure in ``exc``'s exception chain."""

    return next((item for item in exception_chain(exc) if isinstance(item, ContainerRegistryError)), None)


def _message_is_auth_error(message: str) -> bool:
    message = message.strip().lower()
    if not message:
        return False
    if any(marker in message for marker in _AUTH_ERROR_SUBSTRINGS):
        return True
    return bool(_AUTH_ERROR_TOKEN_RE.search(message))


def is_auth_error(exc: BaseException) -> bool:
    """Return whether ``exc`` or its cause represents LLM-provider auth failure."""

    chain = exception_chain(exc)
    # Typed registry provenance outranks auth-like wording on wrappers or causes.
    if any(isinstance(item, ContainerRegistryError) for item in chain):
        return False
    if any(isinstance(item, AuthenticationFailure) for item in chain):
        return True
    return any(_message_is_auth_error(str(item)) for item in chain)
