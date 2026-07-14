"""Active prompt templates/parsers for task claiming and reflection."""

from __future__ import annotations

import json
import re


def _sanitize_agent_output(value: str) -> str:
    """Neutralize forum-protocol keywords in agent-generated text."""
    text = str(value or "")
    text = re.sub(r"(?m)^(\s*)(INSIGHT|COMMENT)(\s*)$", r"\1[\2]\3", text)
    return text


# --- Cache-stable system prompts ------------------------------------------
#
# The system prompt is the cache-stable prefix the prompt cache (Anthropic
# `cache_control: ephemeral`, OpenAI automatic + `prompt_cache_key`)
# reuses across calls. Two properties are required for that to
# work:
#   1. The system text is byte-identical across all calls of the same
#      builder. Any per-call data (task id, eval summary, output excerpt)
#      lives in the user message instead.
#   2. The system text clears the provider's minimum-prefix floor —
#      1024 tokens for OpenAI / Anthropic Sonnet / Anthropic Opus.
# Pre-fix, both phases sent ~10-15-token system strings inline from
# `engine.py`, with the substantive guidance (quality bar, schema)
# concatenated into the per-call user message — the same pathology the
# distill prompt builders had before.

_TASK_INSIGHT_SYSTEM = """\
You write one transferable insight per task — evidence-grounded, reusable
by future agents — and return it as strict JSON.

DOWNSTREAM CONSUMERS
Your insight is loaded verbatim into the next generation's prompt as prior
knowledge. Future agents act on it without being able to ask follow-up
questions, so a vague insight produces a vague action. Insights must be
specific enough that an agent applying them can either succeed or fail
observably. Insights that read like advice from a senior engineer skimming
the problem ("think carefully", "validate first", "watch for edge cases")
add nothing to a future agent's ability to act, and their cost is real:
they push out the limited insights-per-bundle budget and dilute the
signal-to-noise that downstream retrieval ranks against.

CONFIDENCE RUBRIC
- low: weak evidence or first-time pattern with uncertain transfer.
- medium: repeated pattern in this generation but limited validation.
- high: strongly evidenced and reusable pattern with clear outcome
  linkage.

INSIGHT QUALITY BAR
A good insight is a small structured paragraph (roughly 100-400 words,
up to ~2000 characters) that contains, at minimum:
  1. The hypothesis / pattern you inferred — the specific transformation,
     invariant, or failure mode, stated precisely enough to be re-applied.
  2. The evidence — which training example, failed cell, test, or trace
     feature supports the hypothesis. Prefer concrete coordinates, values,
     file paths, or symbols over vague references.
  3. The decision rule — an if/then or procedural check future agents
     can run on a new task to detect and exploit this pattern, or to
     avoid the failure mode.
Cut filler ("always think carefully", "be thorough") — every sentence
must convey either hypothesis, evidence, or decision rule. One dense
paragraph is preferred over bullet fluff; numbered steps are fine if
they are each substantive.

EXAMPLE QUALITY (ARC)
- bad: "Always read errors carefully."
- bad: "When the same failure pattern repeats across similar tasks, check the shared structural constraint."
- good: "Hypothesis: on ARC grids where a 1-cell border of color C surrounds a rectangular region, the transformation preserves C and recolors the interior by the dominant non-C color from the 3x3 training exemplar. Evidence: train pairs 0 and 1 both showed a blue border preserved while interiors shifted red->green; my first attempt recolored the border and failed only at cells (0,0) and (4,4) — the corners where the border runs. Decision rule: before proposing a color-swap rule, compute the outer 1-cell frame of each training output; if frame pixels equal frame pixels of the input, lock the frame and fit the rule on the interior only."

EXAMPLE QUALITY (SWE-bench)
- bad: "Test failures often indicate logic errors."
- bad: "Reading the existing code helps before making changes."
- good: "Hypothesis: django.db.models.QuerySet.iterator() with chunk_size set evicts the prefetch cache between chunks, so .prefetch_related('items') silently degrades to N+1 inside the loop. Evidence: pytest test_qs_iterator_prefetch reported 1 SELECT for items_set on the small fixture and 1024 SELECTs on the chunked fixture; my first patch added prefetch on the queryset but the test still ran the same 1024 queries — the evict happens in iterator(), not in the queryset construction. Decision rule: when a queryset uses .iterator(chunk_size=...), pull the prefetch into a manual fetch (Items.objects.in_bulk(parent_ids)) before the loop; do NOT chain .prefetch_related() on .iterator()."

EXAMPLE QUALITY (polyglot)
- bad: "Rust's borrow checker is strict about lifetimes."
- bad: "Always run the tests after making changes."
- good: "Hypothesis: in Rust, returning a reference from a function that owns a Vec<T> built locally fails the borrow checker because the Vec is dropped at function exit. Evidence: cargo test fizzbuzz reported 'cannot return value referencing local variable v'; my first fix changed the return type to &[T] which produced the same error at the same line. Decision rule: when a function builds an owned collection locally and the caller wants borrowed access, return the owned collection (Vec<T>) and let the caller borrow it via &v[..] at the call site, rather than trying to return &[T] from the function."

WORKSTREAM LABEL
The "workstream" field groups insights for retrieval. A good workstream
label is short (1-5 words), domain-specific, and dash-separated:
"django-orm-prefetch", "arc-symmetry-axes", "rust-lifetimes", "pytest-
parametrize". Avoid generic labels ("general", "debugging", "code-quality")
unless the insight truly is domain-agnostic.

OUTPUT
Return a JSON object only (no markdown fences, no extra text):
{
  "text": "structured, evidence-grounded insight (hypothesis + evidence + decision rule). Aim for 100-400 words; hard cap ~2000 chars.",
  "workstream": "domain-specific workstream label (1-5 words, dash-separated if possible)",
  "confidence": "low|medium|high"
}
"""


_LESSON_EXTRACTION_SYSTEM = """\
You extract 1-3 short reusable lessons from a single completed task
attempt and return them as strict JSON.

DOWNSTREAM CONSUMERS
Each lesson is loaded verbatim into the next generation's prompt as
prior knowledge. Future agents act on lessons without being able to ask
follow-up questions, so a vague lesson produces a vague action. Lessons
that read like generic engineering advice ("validate first", "check
edge cases", "be careful with off-by-one") add nothing to a future
agent's ability to act, and their cost is real — they push out the
limited lessons-per-attempt budget and dilute downstream retrieval.

LESSON QUALITY BAR
A good lesson is one short, dense sentence (under ~120 characters)
that names at least one concrete primitive: a specific operation, API
call, function/class name, import path, file location, language
feature, library function, error message, or code pattern drawn from
the attempt. Abstract nouns alone ("structure", "approach", "pattern")
do not count as concrete.

A good lesson falls into one of three buckets:
  - root cause: a specific named thing that caused failure (e.g.
    "missing torch.no_grad() around the eval loop caused gradient
    accumulation OOM").
  - what worked: a specific named technique that produced progress
    (e.g. "running pytest with -k <name> isolated the regression to
    test_parser_handles_empty_input").
  - reusable pattern: a transferable rule grounded in one of the
    above (e.g. "when SQLAlchemy raises DetachedInstanceError, eager-
    load the relationship in the query rather than in the template").
Anything that doesn't pass this bar should be dropped — empty lessons
arrays are acceptable and preferred over filler.

EVIDENCE GROUNDING
Every lesson must be derivable from concrete content in the attempt
shown to you. If your lesson would still be true without having seen
this attempt's output, it is generic and you should drop it. Useful
evidence anchors include: specific exception names with their stack
frames, test IDs from a pytest/cargo/junit run, file:line references
that appeared in tracebacks, named functions or methods that were
called or modified, named tools the agent invoked from the trace, and
concrete output values (numbers, grids, strings) that distinguished
success from failure.

EXAMPLE QUALITY (across domains)
- bad: "Read errors carefully."
- bad: "Test edge cases."
- bad: "Validate inputs before processing."
- good (ARC, root cause): "rotation-by-180 rule rejected because train pair 2 had non-square 3x5 input but expected square 5x5 output — pure rotation cannot change shape"
- good (SWE-bench, what worked): "narrowing pytest with -k test_serializer_excludes pinned the regression to JSONField default-rendering, not the model change"
- good (polyglot, reusable): "Rust .iter().map(...).collect() needs explicit ::<Vec<_>>() turbofish when the target type is ambiguous from context"

ANTI-META
REJECT lessons that could be lifted into a software-engineering
tutorial unchanged. Examples of phrases that earn auto-rejection:
"validate inputs", "handle errors", "check edge cases", "be careful
with off-by-one", "read the documentation", "write tests",
"think step by step". If a future agent could already produce the
sentence without having seen this attempt's evidence, the lesson is
not earning its place.

LENGTH AND COUNT
Each lesson MUST be under 120 characters. The list MUST contain
between 0 and 3 lessons. If only one lesson clears the quality bar,
return one. If none do, return an empty list — that is a valid and
expected outcome on uneventful attempts.

WHEN TO RETURN EMPTY
Return an empty lessons list when any of the following holds:
  - The attempt did not exercise enough of the task to produce
    grounded evidence (e.g. exited early, no tools called, no test
    output captured).
  - The attempt's output is dominated by trivial mechanics (file
    reads, directory listings, "hello world" probes) with no failure
    pattern or working technique to extract.
  - The only candidate lessons are restatements of the task description
    or generic engineering advice; these fail the anti-meta check.
  - Every candidate lesson exceeds 120 chars when stated specifically
    enough to clear the evidence bar — in that case the right move is
    one tightly-worded lesson, or none, not a vague one.

CONCISENESS DISCIPLINE
The 120-character cap is a hard upper bound, not a target. Aim for
60-100 chars per lesson — the most useful lessons fit on one line.
If a lesson reads like a sentence with three clauses connected by
"and", split it: that is two lessons or one lesson with the weaker
clause dropped. Never pad lessons to look more substantive; padding
costs the limited bullets-per-attempt budget and signals to retrieval
that this attempt produced more signal than it actually did.

OUTPUT
Return a JSON object only (no markdown fences, no extra text):
{
  "lessons": ["lesson 1 (under 120 chars)", "lesson 2", "lesson 3"]
}
The "lessons" list MUST contain between 0 and 3 strings. Each string
MUST be under 120 characters and concretely grounded in the attempt
shown in the user message.
"""


def _strip_output_section(system_text: str) -> str:
    """Drop the trailing ``OUTPUT``/schema block of a single-deliverable system
    prompt so its guidance can be composed under a merged schema."""
    marker = "\nOUTPUT\n"
    idx = system_text.find(marker)
    return system_text[:idx].rstrip() if idx != -1 else system_text.rstrip()


# Merged reflection+lessons system prompt: the insight and
# lesson passes mine the SAME attempt excerpt, so one call produces both — one
# LLM round-trip per attempt instead of two. Both quality bars are preserved
# verbatim (composed from the two single-deliverable systems); only the OUTPUT
# schema is unified. This is a research-behavior change: a single prompt now
# emits both fields, so the exact insight/lesson text shifts vs the two-call
# path — TB2/knowledge numbers relying on the old split should be re-validated.
_TASK_REFLECTION_AND_LESSONS_SYSTEM = (
    _strip_output_section(_TASK_INSIGHT_SYSTEM)
    + "\n\n---\n\nYou ALSO extract 0-3 short reusable lessons from the SAME attempt.\n\n"
    + _strip_output_section(_LESSON_EXTRACTION_SYSTEM)
    + "\n\nOUTPUT\n"
    + "Return a single JSON object only (no markdown fences, no extra text):\n"
    + "{\n"
    + '  "text": "the transferable insight (hypothesis + evidence + decision rule; ~100-400 words, hard cap ~2000 chars)",\n'
    + '  "workstream": "domain-specific label (1-5 words, dash-separated if possible)",\n'
    + '  "confidence": "low|medium|high",\n'
    + '  "lessons": ["lesson 1 (under 120 chars)", "lesson 2", "lesson 3"]\n'
    + "}\n"
    + 'The "lessons" list MUST contain between 0 and 3 strings, each under 120 '
    + "characters. Return an empty list when no lesson clears the quality bar. "
    + "Both the insight and the lessons must be grounded in the attempt shown "
    + "in the user message.\n"
)


def build_task_reflection_and_lessons_prompt(
    *,
    agent_id: str,
    agent_workstream: str,
    task_id: str,
    task_repo: str,
    task_prompt_preview: str,
    eval_summary: str,
    outcome: str,
    score_text: str,
    model_output_excerpt: str,
) -> tuple[str, str]:
    """Return (system, user) for the merged reflection+lessons LLM call.
    Carries the union of the inputs the two former
    single-deliverable builders needed."""
    safe_preview = _sanitize_agent_output(task_prompt_preview)
    safe_excerpt = _sanitize_agent_output(model_output_excerpt)
    repo_line = ("- repo: " + task_repo + "\n") if task_repo else ""
    user = f"""\
You are agent {agent_id}.

## Task
- task_id: {task_id}
{repo_line}- your_current_workstream: {agent_workstream or "general"}
- task_prompt_preview:
<task_description>
{safe_preview}
</task_description>

## Result
{eval_summary}
- outcome: {outcome} (score {score_text})

## Your Output Excerpt
{safe_excerpt}

Write one transferable insight AND 0-3 reusable lessons, following the
rubric, quality bars, and combined output schema in the system prompt."""
    return _TASK_REFLECTION_AND_LESSONS_SYSTEM, user


def parse_task_reflection_and_lessons_response(raw: str) -> dict:
    """Parse the merged reflection+lessons response into
    ``{"insight": {...}|None, "lessons": [...]}``.

    ``insight`` is ``None`` when the ``text`` field is empty; otherwise a
    ``{text, workstream, confidence}`` dict (invalid confidence normalized to
    ``medium``, missing workstream defaulted to ``general``). ``lessons`` is
    0-3 non-empty strings, each capped at 500 chars as a safety net. The two
    are independent: a missing insight does not discard valid lessons."""
    try:
        data = extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {"insight": None, "lessons": []}
    text = str(data.get("text", "")).strip()
    workstream = str(data.get("workstream", "")).strip()
    confidence = str(data.get("confidence", "medium")).strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    insight = {"text": text, "workstream": workstream or "general", "confidence": confidence} if text else None
    lessons_raw = data.get("lessons", [])
    if not isinstance(lessons_raw, list):
        lessons_raw = []
    lessons = [str(item)[:500] for item in lessons_raw[:3] if isinstance(item, str) and item.strip()]
    return {"insight": insight, "lessons": lessons}


def extract_json(raw: str) -> dict:
    text = raw.strip()
    # Try parsing the full text first (handles clean JSON responses).
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract from code fences — use raw_decode so we stop at the first
    # complete JSON object and do not choke on concatenated blocks.
    match = re.search(r"```(?:json)?\s*(\{)", text, re.DOTALL)
    if match:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, match.start(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Last resort: find the start of the first { and decode from there,
    # stopping at the matching close brace (handles concatenated JSON).
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("No JSON object found")
