"""Shared concreteness / anti-meta rules for forum and distill prompts.

The cross-task forum bundle in live baseline-sweep audits was measured
to be ~26% concrete / ~74% process meta-advice ("validate first",
"decompose", "boundary"). The block below names the failure modes,
demands concrete grounding, and caps quantity. It is rendered into:

* the cross-task distill system prompt (so the distiller drops meta
  bullets at compression time), and
* the cross-task forum prompt (so agents see the rules at write time,
  not just after they've already written meta).

Single source of truth — both call sites import ANTI_META_BLOCK and
the test helper from this module.
"""

from __future__ import annotations

ANTI_META_BLOCK = (
    "STRICT ANTI-META RULES (applied before you write anything):\n"
    "- REJECT generic process meta-advice. Insights of the form "
    '"validate first", "decompose before solving", "check edge '
    'cases", "think step by step", "verify boundary conditions", '
    '"test incrementally", "iterate", "reason carefully", '
    '"handle errors", or any wording that could apply to literally '
    "any coding/reasoning task are REJECTED. If your bullet could be "
    "lifted and dropped into a software-engineering tutorial unchanged, "
    "it is not an insight -- drop it.\n"
    "- REQUIRE concrete grounding. Every bullet must name at least one "
    "concrete primitive drawn from the posts: e.g. a specific grid "
    "operation, color index, shape signature, transformation rule "
    "(ARC); a specific API call, function/class name, import, file "
    "path, or code pattern (SWE-bench); a specific language feature, "
    "library function, stdlib module, or test-runner flag (polyglot). "
    'Abstract nouns alone ("structure", "pattern", "approach") do '
    "NOT count as concrete.\n"
    "- REQUIRE evidence grounding. Every bullet must be derivable from "
    "at least one forum post or attempt in the input. Put supporting forum "
    "post IDs in evidence_post_ids when posts support the bullet. Do not "
    "invent post IDs; if the only support is an attempt, leave "
    "evidence_post_ids empty and make the attempt grounding explicit.\n"
    "- PREFER transferable wording. For cross-task insights, describe "
    "the primitive generically enough to apply across multiple tasks, "
    'but keep the primitive itself concrete (e.g. "BFS flood-fill on '
    '8-neighborhood to isolate connected regions of the same color" -- '
    "concrete operation, still task-agnostic).\n"
    "- QUALITY OVER QUANTITY. Return at most 5 insights, 5 pitfalls, "
    "and 5 checks. Pick the best bullets, not the most. Empty lists "
    "are fine when there is no concrete signal.\n"
)


_REQUIRED_PHRASES = (
    "STRICT ANTI-META RULES",
    "validate first",
    "decompose before solving",
    "check edge cases",
    "boundary conditions",
    "REQUIRE concrete grounding",
    "Abstract nouns alone",
    "REQUIRE evidence grounding",
    "evidence_post_ids",
    "at most 5 insights",
    "5 pitfalls",
    "5 checks",
)


def assert_anti_meta_rules_present(text: str) -> None:
    """Test helper: confirms the rendered text exposes the required clauses."""
    for phrase in _REQUIRED_PHRASES:
        assert phrase in text, (
            f"Expected anti-meta phrase {phrase!r} in rendered prompt; missing means the LLM cannot see the blacklist."
        )
