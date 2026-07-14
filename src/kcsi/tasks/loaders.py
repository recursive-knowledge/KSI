from __future__ import annotations

import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..models import TaskSpec
from .custom import register_custom_source
from .registry import get_spec, supported_task_sources

log = logging.getLogger(__name__)

# ── SWE-bench category taxonomy ───────────────────────────────────────────────

SWEBENCH_CATEGORIES: tuple[str, ...] = (
    "Logic",  # Wrong conditions, off-by-one, incorrect calculations/algorithms
    "Type",  # Type coercions, null/None checks, missing values, type casting
    "State",  # Initialization order, object lifecycle, side effects, stale state
    "Async",  # Async/await bugs, race conditions, event handling, coroutines
    "API",  # Wrong parameters, return values, method signatures, deprecated usage
    "Parse",  # Parsing/serialization bugs (JSON, HTML, Markdown, regex, etc.)
    "Render",  # Visual rendering, layout, display formatting, output generation
    "Error",  # Missing try/catch, wrong error propagation, error messages
    "Feature",  # New feature or extension of existing functionality
    "Config",  # Build system, dependency, environment setup, config file bugs
    "Perf",  # Performance improvements
    "Security",  # Input validation, authentication, XSS, injection, access control
    "Uncat",  # Doesn't clearly fit any of the above
)

# ── LLM-based task classifier ─────────────────────────────────────────────────

_CLASSIFY_MAX_PROMPT_CHARS = 6_000
_CLASSIFY_MAX_WORKERS = 8
CLASSIFY_MAX_WORKERS = _CLASSIFY_MAX_WORKERS  # public alias for use in CLI

_SWEBENCH_CLASSIFY_PROMPT = """\
You are a software bug classifier. Given a SWE-bench issue/problem statement, classify \
the PRIMARY type of change needed into exactly one of the following categories:

- Logic: Wrong conditions, off-by-one errors, incorrect calculations or algorithms
- Type: Type coercions, null/None checks, missing value handling, type casting bugs
- State: Initialization order, object lifecycle, side effects, stale/shared state
- Async: Async/await bugs, race conditions, event handling, coroutine/callback issues
- API: Wrong parameters, return values, method signatures, deprecated interface usage
- Parse: Parsing or serialization bugs (JSON, HTML, Markdown, CSS, regex, etc.)
- Render: Visual rendering, layout, display formatting, output generation bugs
- Error: Missing try/catch, incorrect error propagation, wrong error messages
- Feature: New feature or extension of existing functionality
- Config: Build system, dependency, environment setup, or config file bugs
- Perf: Performance improvements unrelated to correctness
- Security: Input validation, authentication, XSS, injection, access control fixes
- Uncat: Doesn't clearly fit any of the above

Respond with ONLY a JSON object (no markdown fences):
{"category": "<one of the categories above>", "reasoning": "<one sentence>"}
"""

_CLASSIFY_SYSTEM_PROMPTS: dict[str, str] = {
    "swebench_pro": _SWEBENCH_CLASSIFY_PROMPT,
}

_VALID_CATEGORIES_BY_SOURCE: dict[str, frozenset[str]] = {
    "swebench_pro": frozenset(SWEBENCH_CATEGORIES),
}


def _run_agent_sdk_query(user_prompt: str, model: str | None, *, system_prompt: str | None = None) -> str:
    """Run a single-turn Agent SDK query and return the text response.

    Uses claude_agent_sdk.query() which automatically picks up authentication:
    - CLAUDE_CODE_OAUTH_TOKEN → Claude Code subscription (no per-token cost)
    - ANTHROPIC_API_KEY → falls back to API key billing

    Args:
        user_prompt: User message content.
        model: Model name override, or None for SDK default.
        system_prompt: Optional system prompt content.

    Returns:
        The text content from the assistant's response.
    """
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, query

    options = ClaudeAgentOptions(
        model=model,
        max_turns=1,
        allowed_tools=[],
        permission_mode="bypassPermissions",
        system_prompt=system_prompt or "",
    )

    async def _query() -> str:
        result_text = ""
        async for message in query(prompt=user_prompt, options=options):
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        result_text += block.text
        return result_text

    return asyncio.run(_query())


def classify_task_with_llm(
    task_prompt: str,
    categories: list[str] | tuple[str, ...],
    task_source: str,
) -> str:
    """Classify a single task prompt into one of the provided categories using an LLM.

    Uses the Claude Agent SDK (picks up CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY).
    Default model: claude-haiku-4-5-20251001. Override with MODEL env var.
    Returns one of the provided category strings, or 'Uncat' if classification fails.

    Args:
        task_prompt: The raw task text (e.g. problem statement for SWE-bench).
        categories: Ordered list of valid category names.
        task_source: Task source key (e.g. "swebench") — selects the system prompt.

    Returns:
        A category string from the provided categories list, or "Uncat" on failure.
    """
    system_prompt = _CLASSIFY_SYSTEM_PROMPTS.get(task_source)
    if not system_prompt:
        log.warning("No classifier system prompt for task_source=%r; returning Uncat", task_source)
        return "Uncat"

    valid = frozenset(categories)

    # Truncate overly long prompts
    prompt_text = task_prompt.strip()
    if len(prompt_text) > _CLASSIFY_MAX_PROMPT_CHARS:
        prompt_text = prompt_text[:_CLASSIFY_MAX_PROMPT_CHARS] + "\n\n[TRUNCATED]"

    user_message = f"Problem:\n{prompt_text}"

    try:
        model = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
        text = _run_agent_sdk_query(user_message, model, system_prompt=system_prompt).strip()
    except Exception as exc:
        log.warning("LLM classify call failed: %s — returning Uncat", exc)
        return "Uncat"

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Parse JSON response
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try salvaging truncated JSON
        patched = text.rstrip()
        if not patched.endswith("}"):
            patched += '"}' if patched.count('"') % 2 == 1 else "}"
        try:
            data = json.loads(patched)
        except json.JSONDecodeError:
            log.warning("Failed to parse classifier response (%.120s) — returning Uncat", text)
            return "Uncat"

    raw_cat = (data.get("category", "") if isinstance(data, dict) else "").strip()
    if raw_cat in valid:
        return raw_cat
    # Try case-insensitive match
    for cat in categories:
        if cat.lower() == raw_cat.lower():
            return cat
    log.debug("Unknown category %r from LLM — returning Uncat", raw_cat)
    return "Uncat"


def classify_tasks(
    tasks: list[TaskSpec],
    task_source: str,
    *,
    cache_path: Path | None = None,
    max_workers: int = _CLASSIFY_MAX_WORKERS,
) -> list[TaskSpec]:
    """Classify a list of tasks using the LLM, with JSON sidecar caching.

    Results are cached to ``cache_path`` (a JSON file mapping task_id → category).
    On subsequent calls the cache is loaded and only uncached tasks are classified.
    Tasks that already have a non-empty 'category' in their metadata are skipped.

    Args:
        tasks: List of TaskSpec objects to classify.
        task_source: Task source key (e.g. "swebench") — determines taxonomy and system prompt.
        cache_path: Optional path to a JSON sidecar file for caching results.
        max_workers: Thread pool size for parallel LLM calls.

    Returns:
        The same list of TaskSpec objects, with 'category' set in metadata.
    """
    categories: tuple[str, ...] = _VALID_CATEGORIES_BY_SOURCE.get(task_source, ())  # type: ignore[assignment]
    if not categories:
        log.warning("No categories defined for task_source=%r; skipping classification", task_source)
        return tasks

    # Load existing cache if present
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            log.info("Loaded %d cached categories from %s", len(cache), cache_path)
        except Exception as exc:
            log.warning("Failed to load category cache %s: %s", cache_path, exc)

    # Determine which tasks need classification
    to_classify: list[TaskSpec] = []
    for task in tasks:
        existing = task.metadata.get("category", "")
        if existing and existing != "Uncat":
            # Already has a real category (e.g. pre-assigned label); skip
            continue
        if task.id in cache:
            task.metadata["category"] = cache[task.id]
        else:
            to_classify.append(task)

    if not to_classify:
        log.info("All %d tasks already classified (cache hit or pre-assigned)", len(tasks))
        return tasks

    log.info(
        "Classifying %d tasks (task_source=%r, workers=%d)...",
        len(to_classify),
        task_source,
        max_workers,
    )

    # Run classification in parallel
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                classify_task_with_llm,
                task.prompt,
                categories,
                task_source,
            ): task
            for task in to_classify
        }
        done = 0
        total = len(to_classify)
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                cat = future.result()
            except Exception as exc:
                log.warning("Classification failed for %s: %s — Uncat", task.id, exc)
                cat = "Uncat"
            results[task.id] = cat
            done += 1
            if done % 20 == 0 or done == total:
                log.info("  classified %d/%d tasks", done, total)

    # Apply results to task metadata
    for task in to_classify:
        task.metadata["category"] = results.get(task.id, "Uncat")

    # Update cache and persist
    cache.update(results)
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
            log.info("Saved %d categories to cache %s", len(cache), cache_path)
        except Exception as exc:
            log.warning("Failed to save category cache %s: %s", cache_path, exc)

    return tasks


def load_categories_json(path: Path) -> dict[str, str]:
    """Load task_id → category mapping from a JSON sidecar file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data)}")
    return {str(k): str(v) for k, v in data.items()}


def load_tasks_for_source(
    *,
    task_source: str,
    tasks_path: Path,
    evals_path: Path | None = None,
    arc_max_trials: int = 2,
    polyglot_test_feedback_tries: int = 2,
    polyglot_test_feedback_max_lines: int = 50,
) -> list[TaskSpec]:
    spec = get_spec(task_source)
    if spec.loader is None:
        raise ValueError(
            f"task source {spec.name!r} is registered without a loader; set "
            f"TaskSourceSpec.loader to a callable that returns a list of TaskSpec "
            f"(see src/kcsi/tasks/registry.py)"
        )
    return spec.loader(
        tasks_path,
        task_source=task_source,
        evals_path=evals_path,
        arc_max_trials=arc_max_trials,
        polyglot_test_feedback_tries=polyglot_test_feedback_tries,
        polyglot_test_feedback_max_lines=polyglot_test_feedback_max_lines,
    )


def load_eval_records_for_source(*, task_source: str, evals_path: Path | None) -> dict[str, dict[str, Any]]:
    if evals_path is None:
        return {}
    if get_spec(task_source).needs_eval_records:
        # Local import: `_load_swebench_eval_records` is benchmark-specific and
        # lives in `kcsi.benchmarks.loaders`, which this module's `register_all()`
        # call (below) is what makes importable in the first place.
        from ..benchmarks.loaders import _load_swebench_eval_records

        return _load_swebench_eval_records(evals_path)
    return {}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix != ".parquet":
        raise ValueError(f"Parquet-only loader: expected .parquet file, got {path}")
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        rows = table.to_pylist()
        return [r for r in rows if isinstance(r, dict)]
    except ImportError:
        pass
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path)
        rows = df.to_dict(orient="records")
        return [r for r in rows if isinstance(r, dict)]
    except ImportError as exc:
        raise ImportError(f"Reading parquet requires pyarrow or pandas: {path}") from exc


def _load_tabular_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _load_rows(path)
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if isinstance(row, dict):
                    rows.append(row)
        return rows
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported tabular task file type: {path}")


# ── Registry wiring ───────────────────────────────────────────────────────────
#
# `kcsi.benchmarks.register_all()` is the SINGLE core->benchmarks wiring point:
# it registers the four built-in benchmark TaskSourceSpecs and attaches their
# loaders (both idempotent). Importing `kcsi.tasks.loaders` triggers it here,
# immediately followed by the `custom` source's own registration, so canonical
# registration order stays swebench_pro, arc, polyglot, terminal_bench_2, custom
# (pinned by tests/test_task_registry.py).
from ..benchmarks import register_all as _register_benchmarks

_register_benchmarks()
register_custom_source()

# Canonical task-source names come from the central registry (one source of
# truth), computed AFTER registration above so it reflects the fully-wired
# REGISTRY. Preserves the historical tuple value/order.
SUPPORTED_TASK_SOURCES: tuple[str, ...] = supported_task_sources()
