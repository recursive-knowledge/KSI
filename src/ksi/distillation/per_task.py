"""Per-task distillation: one pure function, one LLM call per task."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..errors import AuthenticationFailure, is_auth_error
from .prompts import DISTILL_BUNDLE_JSON_SCHEMA, build_per_task_distill_prompt
from .types import CROSS_TASK_INSIGHT_FIELDS, DistillLLMResult, LLMCallable, PerTaskBundle, truncate_at_boundary
from .types import coerce_positive_int as _coerce_positive_int

log = logging.getLogger(__name__)

_CONCRETE_STRUCTURAL_RE = re.compile(
    r"""
    [A-Za-z_][A-Za-z0-9_]*\(\)             # a call-shaped identifier, e.g. rfind()
    | \b[a-z_]+\.[a-z_]+\b                 # dotted path/attr, e.g. os.path
    | \b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b    # snake_case identifier, e.g. test_index
    | \.[a-z0-9]{1,4}\b                    # file extension, e.g. .py, .rs
    | "[^"]{2,}"|'[^']{2,}'                # a quoted literal
    | \b\d+\s?[x×]\s?\d+\b                 # a grid/shape dimension, e.g. 3x5, 5x5
    | \(\d+,\s*\d+\)                       # a coordinate pair, e.g. (2, 3)
    | \bpost\s+\d+\b                       # an evidence citation, e.g. post 1, (post 1)
    """,
    re.VERBOSE,
)

# Concrete domain vocabulary: nouns/verbs that name a specific structural or
# technical thing in this repo's task domains (ARC grids, SWE-bench/polyglot
# code) rather than generic process advice. Deliberately excludes abstract
# words the prompt-level ANTI_META_BLOCK already calls out as non-concrete
# ("pattern", "structure", "approach", "edge case") so a filler bullet that
# merely echoes those words is not accidentally rescued. Also excludes
# ordinary English words that double as loose SWE terms ("test", "list",
# "class", "method", "argument", "loop", "query", "thread", "post", "index",
# "generation", "pair") -- an LLM writing vague meta-commentary can use one
# of those incidentally without adding any real concreteness.
_CONCRETE_VOCAB = frozenset(
    {
        # grid / ARC-domain nouns
        "row",
        "rows",
        "column",
        "columns",
        "col",
        "grid",
        "grids",
        "cell",
        "cells",
        "block",
        "blocks",
        "box",
        "boxes",
        "shape",
        "shapes",
        "color",
        "colors",
        "colour",
        "colours",
        "border",
        "borders",
        "divider",
        "dividers",
        "mirror",
        "rotation",
        "rotations",
        "rotate",
        "reflect",
        "reflection",
        "symmetry",
        "symmetric",
        "quadrant",
        "quadrants",
        "background",
        "foreground",
        "pixel",
        "pixels",
        "tile",
        "tiles",
        "tiling",
        "region",
        "regions",
        "palette",
        "mask",
        "offset",
        "offsets",
        "fingerprint",
        "fingerprints",
        "bounding",
        "submatrix",
        "subgrid",
        "transformation",
        "translate",
        "translation",
        "overlay",
        "crop",
        "pad",
        "padding",
        # code / SWE-domain nouns
        "function",
        "variable",
        "module",
        "import",
        "exception",
        "traceback",
        "assert",
        "parameter",
        "parameters",
        "indices",
        "array",
        "dict",
        "regex",
        "identifier",
        "attribute",
        "constructor",
        "decorator",
        "docstring",
        "endpoint",
        "schema",
        "migration",
        "socket",
        "fixture",
        "assertion",
        "stdout",
        "stderr",
    }
)

_WORD_RE = re.compile(r"[a-z]+")

# "e.g."/"i.e." (with or without the trailing period) incidentally match the
# dotted-path and file-extension structural patterns below (e.g. the ".g" in
# "e.g." looks like a file extension). Strip them before the structural check
# so citing them doesn't rescue otherwise-generic filler text.
_EG_IE_RE = re.compile(r"\b(?:e\.g\.?|i\.e\.?)\b", re.IGNORECASE)


def _is_concrete(text: str) -> bool:
    """Reject insight text with no concrete grounding.

    Complements the prompt-only ANTI_META_BLOCK: catches generic filler
    ("validate first", "check edge cases") that clears the LLM's own
    self-censoring but carries no verifiable primitive.

    A bullet is concrete if EITHER holds:
      - it contains a structural token that only shows up in genuinely
        concrete text: a call-shaped identifier, a dotted path or snake_case
        identifier, a file extension, a quoted literal, a grid/shape
        dimension (e.g. "3x5"), or a coordinate pair; or
      - it names at least one word from a curated concrete-vocabulary list
        (ARC grid/geometry nouns, code/SWE nouns) drawn from what the
        ANTI_META_BLOCK / prompt "good example" text already treats as
        domain-grounded.

    A single incidental digit is deliberately NOT sufficient on its own
    ("check carefully at step 2", "retry once more (attempt 2)") -- digits
    only count when they form one of the structural shapes above (a
    dimension or coordinate), not when merely present anywhere in the text.
    """
    if _CONCRETE_STRUCTURAL_RE.search(_EG_IE_RE.sub(" ", text)):
        return True
    words = set(_WORD_RE.findall(text.lower()))
    return not _CONCRETE_VOCAB.isdisjoint(words)


def _call_llm(
    llm: LLMCallable,
    sys_prompt: str,
    user_prompt: str,
    *,
    bundle_schema: dict[str, Any] | None = None,
    cache_prefix: str | None = None,
) -> tuple[str, dict | None]:
    """Invoke the distill LLM, requesting provider structured output when the
    callable supports it.

    Returns ``(raw_text, parsed_dict_or_None)``. When the provider returned a
    schema-validated dict, ``parsed_dict`` is that dict and the caller skips the
    lenient free-text parser entirely. When the callable does not accept a
    ``json_schema`` kwarg (unknown providers / legacy adapters / plain test
    stubs), we fall back to a schema-less call and ``parsed_dict`` is ``None``,
    leaving the lenient parser to handle free text.
    """
    schema = bundle_schema or DISTILL_BUNDLE_JSON_SCHEMA
    # ``cache_prefix`` is the cross-call-stable history the
    # target-conditioned cross-task distill re-sends to every target; forwarding
    # it lets the provider caller cache-read it. It is optional: callables that
    # predate it (test stubs, legacy adapters) reject the kwarg, so we retry
    # without it before falling back to the schema-less legacy path.
    call_kwargs: dict[str, Any] = {"json_schema": schema}
    if cache_prefix:
        call_kwargs["cache_prefix"] = cache_prefix
    try:
        result = llm(sys_prompt, user_prompt, **call_kwargs)
    except TypeError as exc:
        # Only treat "callable doesn't accept the kwarg" as the fall-back
        # signal. A TypeError raised *inside* a schema-capable callable is a
        # real bug and must surface — silently retrying would mask it.
        if "unexpected keyword argument" not in str(exc):
            raise
        if cache_prefix and "cache_prefix" in str(exc):
            # Caller accepts json_schema but not cache_prefix. Fold the prefix
            # back into the user message before retrying: the provider caller
            # would have concatenated cache_prefix + user internally, and the
            # target-conditioned caller passes ONLY the suffix as user_prompt
            # (the forum history lives entirely in cache_prefix). Dropping the
            # prefix outright would silently delete the history from the prompt.
            # This preserves the prompt content byte-for-byte; only the caching
            # optimization is lost.
            folded_user = cache_prefix + user_prompt
            try:
                result = llm(sys_prompt, folded_user, json_schema=schema)
            except TypeError as exc2:
                if "unexpected keyword argument" not in str(exc2):
                    raise
                return str(llm(sys_prompt, folded_user) or ""), None
        else:
            return str(llm(sys_prompt, user_prompt) or ""), None

    # Engine adapter returns the shared DistillLLMResult carrier.
    if isinstance(result, DistillLLMResult):
        return str(result.text or ""), result.parsed if isinstance(result.parsed, dict) else None
    # Provider callers return a tuple ``(text, usage[, parsed_dict])``.
    if isinstance(result, tuple):
        raw = str(result[0] or "") if result else ""
        parsed = result[2] if len(result) > 2 and isinstance(result[2], dict) else None
        return raw, parsed
    # Legacy ``(system, user) -> str`` callable.
    return str(result or ""), None


def distill_one_task(
    *,
    task_id: str,
    attempts: list[dict[str, Any]],
    posts: list[dict[str, Any]],
    llm: LLMCallable,
    task_source: str | None = None,
    bundle_schema: dict[str, Any] | None = None,
    win_mode: bool = False,
) -> PerTaskBundle | None:
    """Distill a per-task bundle. Returns None if LLM fails or returns
    unparseable output -- caller proceeds without a bundle for this task.

    V2: input is just the task's attempts (across all gens, with reflection)
    and the task's per-task forum posts (across all gens). No prior bundle
    (consumed at seed time, doesn't feed back). No cross-task history
    (per-task / cross-task layers are structurally independent).

    ``task_source`` is an optional domain hint forwarded to the prompt
    builder so bullets get biased toward benchmark-specific concrete
    primitives.

    ``win_mode`` (KSI_TRANSFER_BRIDGE): the task was solved this generation;
    the bundle prompt gains the win directive.
    """
    schema: dict[str, Any] | None
    sys_prompt, user_prompt = build_per_task_distill_prompt(
        task_id=task_id,
        attempts=attempts,
        posts=posts,
        task_source=task_source,
        win_mode=win_mode,
    )
    schema = bundle_schema
    try:
        raw, structured = _call_llm(llm, sys_prompt, user_prompt, bundle_schema=schema)
    except AuthenticationFailure:
        raise
    except Exception as exc:
        if is_auth_error(exc):
            raise AuthenticationFailure(f"LLM authentication failed for per-task distill ({task_id}): {exc}") from exc
        log.warning("distill_one_task(%s): LLM raised %r", task_id, exc)
        return None

    if not structured and not (raw or "").strip():
        # Empty response with no structured payload — typically a tool-call
        # decline under forced structured output, not malformed JSON. Logged
        # distinctly so it isn't conflated with a parse failure.
        log.warning(
            "distill_one_task(%s): LLM returned an empty response (possible structured-output decline)",
            task_id,
        )
        return None

    if structured:
        # Provider returned a non-empty schema-constrained dict; use it directly
        # and skip the free-text brace-matcher / regex-repair path entirely. An
        # empty ``{}`` is treated as "no usable payload" (falsy) so it falls
        # through to the lenient parser rather than producing a silent all-empty
        # bundle.
        payload: dict | None = structured
    else:
        # Unknown provider / legacy callable: fall back to the lenient parser.
        # (The paid second "repair-LLM" call was removed — structured
        # outputs make it unnecessary, and a second LLM call to fix one bad
        # character was never worth the cost.)
        payload = _parse_json_lenient(raw, label=f"distill_one_task({task_id})")
    if payload is None:
        return None

    valid_post_ids = _post_id_set(posts)

    fields = dedupe_bundle_items(
        {field: _as_insight_list(payload.get(field), allowed_post_ids=valid_post_ids) for field in _BUNDLE_ITEM_FIELDS}
    )

    return PerTaskBundle(
        task_id=task_id,
        **fields,
        # Provenance trust boundary: the ONLY post-id evidence that
        # enters a trusted bundle is membership-filtered here against the posts
        # actually loaded for this task (valid_post_ids). A forum post's own
        # free-text "evidence_post_id" body field (forum/prompt.py) is stored
        # verbatim and never promoted to trusted provenance, so a hostile/
        # malformed citation cannot ride into distillation — an out-of-range id
        # is dropped by _as_int_list rather than trusted.
        evidence_post_ids=_as_int_list(
            payload.get("evidence_post_ids"),
            allowed_values=valid_post_ids,
        ),
    )


_INVALID_ESCAPE_RE = re.compile(r'\\([^"\\/bfnrtu])')
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _parse_json_lenient(raw: str, *, label: str) -> dict | None:
    """Best-effort parse of free-text LLM output as JSON.

    Used only on the fallback path — when the provider/profile does not
    support JSON-schema-constrained output (unknown providers, legacy
    callables, plain test stubs). Providers that DO support structured output
    (Anthropic tool-forcing, OpenAI Responses ``json_schema``) return an
    already-parsed dict upstream and never reach here.

    Uses the existing ``_extract_json_object`` + ``_json_repair_candidates``
    machinery (handles code fences, invalid escapes, trailing commas, raw
    control chars). On failure returns ``None`` — the paid "repair-LLM"
    second call this helper replaced is gone; a second LLM round-trip
    to fix one bad character was never worth the cost, and structured outputs
    make the syntax-slip case rare.
    """
    try:
        return _parse_json(raw)
    except Exception as exc:
        log.warning("%s: could not parse free-text LLM output as JSON (%r)", label, exc)
        return None


def _parse_json(text: str) -> dict:
    # Some models wrap JSON in code fences or extra prose. Find the first
    # balanced {...} JSON object rather than the last brace in the message.
    body = _extract_json_object(text.strip())
    last_error: Exception | None = None
    for candidate in _json_repair_candidates(body):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(payload, dict):
            return payload
        raise ValueError("distillation JSON must be an object")
    assert last_error is not None
    raise last_error


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in output")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    raise ValueError("unterminated JSON object in output")


def _json_repair_candidates(body: str) -> list[str]:
    """Return bounded, deterministic repairs for common model JSON slips."""
    candidates = [body]
    for repair in (
        _repair_invalid_escapes,
        _repair_trailing_commas,
        _repair_string_control_chars,
    ):
        for candidate in list(candidates):
            repaired = repair(candidate)
            if repaired not in candidates:
                candidates.append(repaired)
    return candidates


def _repair_invalid_escapes(body: str) -> str:
    # LLMs occasionally emit invalid backslash escapes inside JSON strings,
    # e.g. regex patterns like ``"\d+"`` or Windows paths.
    return _INVALID_ESCAPE_RE.sub(r"\\\\\1", body)


def _repair_trailing_commas(body: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", body)


def _repair_string_control_chars(body: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in body:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                out.append(ch)
                in_string = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
    return "".join(out)


# Canonical insight-field schema lives in ``ksi.distillation.types``; alias the
# single source of truth so a future drift can't silently desync this module.
_BUNDLE_ITEM_FIELDS = CROSS_TASK_INSIGHT_FIELDS
_DEDUPE_JACCARD = 0.8


def _item_tokens(item: Any) -> set[str]:
    text = item.get("text", "") if isinstance(item, dict) else str(item)
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 2}


def dedupe_bundle_items(bundle: dict[str, Any]) -> dict[str, Any]:
    """Drop items near-duplicating an earlier-kept item ANYWHERE in the bundle
    (token-Jaccard >= 0.8). Bundles self-duplicate
    across fields and within fields; first occurrence wins (field order above)."""
    kept_tokens: list[set[str]] = []
    out = dict(bundle)
    for field in _BUNDLE_ITEM_FIELDS:
        items = bundle.get(field)
        if not isinstance(items, list):
            continue
        kept: list[Any] = []
        for item in items:
            toks = _item_tokens(item)
            dup = False
            for prev in kept_tokens:
                union = toks | prev
                if union and len(toks & prev) / len(union) >= _DEDUPE_JACCARD:
                    dup = True
                    break
            if not dup:
                kept.append(item)
                kept_tokens.append(toks)
        out[field] = kept
    return out


def _as_insight_list(value: Any, *, allowed_post_ids: set[int] | None = None) -> list[Any]:
    """Coerce an LLM-emitted list into structured Insights or legacy strings.

    The V2 distill prompt emits each item as a dict
    ``{text, applies_when, does_not_apply_when, evidence, confidence}``.
    Legacy/older prompts (and v0 tests) emit bare strings. This function
    preserves both shapes so the render path can detect and handle each.
    Dicts missing a non-empty ``text`` field are dropped (corresponds to
    the prompt's "DROP any Insight without text" rule).
    """
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for v in value:
        if v is None:
            continue
        if isinstance(v, dict):
            text = str(v.get("text") or "").strip()
            if not text:
                continue
            if not _is_concrete(text):
                continue
            cleaned: dict[str, Any] = {"text": truncate_at_boundary(text, 480)}
            for k in ("applies_when", "does_not_apply_when"):
                raw = v.get(k)
                if raw:
                    cleaned[k] = truncate_at_boundary(str(raw).strip(), 200)
            confidence = str(v.get("confidence") or "").strip().lower()
            if confidence in ("high", "medium", "low"):
                cleaned["confidence"] = confidence
            evidence = v.get("evidence")
            if isinstance(evidence, list) and evidence:
                cleaned_ev: list[dict[str, Any]] = []
                for ev in evidence[:5]:
                    if not isinstance(ev, dict):
                        continue
                    pid = _coerce_positive_int(ev.get("post_id"))
                    if pid is None:
                        continue
                    if allowed_post_ids is not None and pid not in allowed_post_ids:
                        continue
                    e: dict[str, Any] = {"post_id": pid}
                    tid = ev.get("task_id")
                    if tid:
                        e["task_id"] = str(tid).strip()[:120]
                    quote = ev.get("quote")
                    if quote:
                        e["quote"] = truncate_at_boundary(str(quote).strip(), 200)
                    cleaned_ev.append(e)
                if cleaned_ev:
                    cleaned["evidence"] = cleaned_ev
            out.append(cleaned)
        else:
            text = str(v).strip()
            if text and _is_concrete(text):
                out.append(text)
        if len(out) >= 5:
            break
    return out


def _post_id_set(posts: list[dict[str, Any]]) -> set[int]:
    result: set[int] = set()
    for post in posts:
        if not isinstance(post, dict):
            continue
        value = _coerce_positive_int(post.get("id"))
        if value is not None:
            result.add(value)
    return result


def _as_int_list(value: Any, *, allowed_values: set[int] | None = None) -> list[int]:
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        item = _coerce_positive_int(v)
        if item is None:
            continue
        if allowed_values is not None and item not in allowed_values:
            continue
        out.append(item)
    return out
