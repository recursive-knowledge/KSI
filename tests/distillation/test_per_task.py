import json

from kcsi.distillation.per_task import (
    _as_insight_list,
    _parse_json,
    dedupe_bundle_items,
    distill_one_task,
    truncate_at_boundary,
)
from kcsi.distillation.types import PerTaskBundle


def test_distill_one_task_builds_bundle_from_llm_json():
    attempts = [{"agent_id": "a1", "native_score": 0.0, "model_output": "tried X"}]
    posts = [{"id": 1, "agent_id": "a1", "text": "X fails"}]

    def fake_llm(sys_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "transferable_insights": ["Use Y instead of X (post 1)"],
                "confirmed_constraints": ["output must remain 3x3"],
                "rejected_hypotheses": ["X fails on the first mismatch (post 1)"],
                "pitfalls": ["X causes cycle"],
                "checks": ["verify input shape"],
                "next_steps": ["try Y with shape check first"],
                "evidence_post_ids": [1],
            }
        )

    b = distill_one_task(
        task_id="t1",
        attempts=attempts,
        posts=posts,
        # V2: prior_bundle removed
        llm=fake_llm,
    )
    assert isinstance(b, PerTaskBundle)
    assert b.task_id == "t1"
    assert b.transferable_insights == ["Use Y instead of X (post 1)"]
    assert b.confirmed_constraints == ["output must remain 3x3"]
    assert b.rejected_hypotheses == ["X fails on the first mismatch (post 1)"]
    assert b.next_steps == ["try Y with shape check first"]
    assert b.evidence_post_ids == [1]


def test_distill_one_task_filters_evidence_ids_to_supplied_posts():
    posts = [{"id": 1, "agent_id": "a1", "text": "grounded"}]

    def fake_llm(sys_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "transferable_insights": ["Use Y instead of X"],
                "confirmed_constraints": [],
                "rejected_hypotheses": [],
                "pitfalls": [],
                "checks": [],
                "next_steps": [],
                "evidence_post_ids": [1, 999, "bad", True],
            }
        )

    b = distill_one_task(
        task_id="t1",
        attempts=[],
        posts=posts,
        # V2: prior_bundle removed
        llm=fake_llm,
    )

    assert isinstance(b, PerTaskBundle)
    assert b.evidence_post_ids == [1]


def test_distill_one_task_returns_none_on_llm_failure():
    def bad_llm(sys_prompt: str, user_prompt: str) -> str:
        return "not json"

    b = distill_one_task(
        task_id="t1",
        attempts=[],
        posts=[],
        # V2: prior_bundle removed
        llm=bad_llm,
    )
    assert b is None


def test_distill_one_task_returns_none_on_llm_exception():
    def raising_llm(s, u):
        raise RuntimeError("api down")

    b = distill_one_task(
        task_id="t1",
        attempts=[],
        posts=[],
        # V2: prior_bundle removed
        llm=raising_llm,
    )
    assert b is None


def test_truncate_at_boundary_short_text_unchanged():
    assert truncate_at_boundary("short text.", 480) == "short text."


def test_truncate_at_boundary_prefers_sentence_end():
    # Sentence boundary lands within the floor..cap window (past 60% of 480
    # == 288), so it must be preferred over a hard cut.
    text = "Long context filler describing constraints in detail. " * 6 + "Final clause ends cleanly here. " + "x" * 500
    out = truncate_at_boundary(text, 480)
    assert out.endswith(".")
    assert len(out) <= 480


def test_truncate_at_boundary_falls_back_to_word_boundary():
    text = "word " * 200
    out = truncate_at_boundary(text, 480)
    assert len(out) <= 480
    assert not out.endswith(" wor")


def test_truncate_at_boundary_degenerate_no_spaces():
    out = truncate_at_boundary("a" * 600, 480)
    assert len(out) == 480


def test_truncate_at_boundary_early_isolated_boundary_hard_cuts_at_cap():
    # Regression for issue #1057 review: a boundary far below the floor
    # (e.g. a short label like "Error:" followed by a long unbroken tail)
    # must NOT collapse the output down to that early boundary -- it should
    # hard-cut at cap instead, matching (or beating) the old plain-slice
    # behavior.
    text = "Error: " + "x" * 5000
    out = truncate_at_boundary(text, 2000)
    assert len(out) == 2000


def test_truncate_at_boundary_early_isolated_boundary_json_blob():
    text = '{"status": "error", "trace": "' + "A" * 5000 + '"}'
    out = truncate_at_boundary(text, 2000)
    assert len(out) == 2000


def test_insight_text_kept_to_480_at_boundary():
    long_text = "For divider-based tasks, identify monochromatic divider rows. " * 12
    items = _as_insight_list([{"text": long_text}])
    assert len(items[0]["text"]) <= 480
    assert items[0]["text"].endswith(".")


def test_as_insight_list_drops_generic_filler():
    items = _as_insight_list([{"text": "Validate inputs before processing and check edge cases carefully."}])
    assert items == []


def test_as_insight_list_keeps_concrete_insight():
    items = _as_insight_list(
        [
            {
                "text": "rotation-by-180 rule rejected because train pair 2 had non-square 3x5 input "
                "but expected square 5x5 output"
            }
        ]
    )
    assert len(items) == 1


def test_as_insight_list_keeps_domain_concrete_examples_without_digits():
    """Precise, domain-grounded insight text must pass even with no digit,
    call()-shape, dotted path, or quoted literal in sight (#992 recalibration)."""
    examples = [
        "identify monochromatic divider rows",
        "extract seed bounding box and compute offset fingerprints for all blocks",
        "hardcoding sampling indices from a single training pair",
        "validate shape before color fill",
    ]
    for text in examples:
        items = _as_insight_list([{"text": text}])
        assert len(items) == 1, f"expected {text!r} to be kept as concrete"


def test_as_insight_list_rejects_vague_filler_with_incidental_digit():
    """A single incidental digit must not rescue generic process filler --
    closes the trivial digit-anywhere gaming hole (#992)."""
    for text in ("check carefully at step 2", "retry once more (attempt 2)"):
        items = _as_insight_list([{"text": text}])
        assert items == [], f"expected {text!r} to be rejected despite containing a digit"


def test_as_insight_list_rejects_filler_using_generic_english_vocab_words():
    """Ordinary English words that double as loose SWE terms (post, index,
    generation, test, list, pair, class, method, argument, loop, query,
    thread) must not rescue generic meta-commentary that merely happens to
    use one of them (#1034)."""
    for text in (
        "post a note about the class of issues found",
        "write a good test for the list class",
        "loop back on the query once the thread finishes",
    ):
        items = _as_insight_list([{"text": text}])
        assert items == [], f"expected {text!r} to be rejected as generic filler"


def test_as_insight_list_keeps_evidence_post_citation():
    """A citation of a specific evidence post (e.g. "post 1") is a genuine
    concrete reference, distinct from the bare word "post" used as generic
    filler -- (#1034)."""
    items = _as_insight_list([{"text": "reuse the fix suggested in post 1"}])
    assert len(items) == 1


def test_as_insight_list_rejects_eg_ie_filler():
    """ "e.g."/"i.e." must not rescue filler text via the dotted-path or
    file-extension structural patterns -- closes the residual regex gaming
    hole (#1034)."""
    for text in (
        "validate inputs first, e.g. before running",
        "do the thing, i.e. run it",
    ):
        items = _as_insight_list([{"text": text}])
        assert items == [], f"expected {text!r} to be rejected despite containing e.g./i.e."


def test_as_insight_list_keeps_real_dotted_path_alongside_eg():
    """Stripping "e.g."/"i.e." must not blind the dotted-path check to a
    genuinely concrete identifier elsewhere in the same text (#1034)."""
    items = _as_insight_list([{"text": "use a helper, e.g. os.path.join, to build the path"}])
    assert len(items) == 1


def test_as_insight_list_keeps_concrete_dict_without_evidence_or_applicability():
    """Deep-review #1264 refutation: the parser is INTENTIONALLY more permissive
    than the prompt, and this is by design, not a defect.

    The V2 distill prompt marks non-empty ``text`` / ``applies_when`` /
    ``does_not_apply_when`` / ``evidence`` as non-negotiable and tells the LLM to
    DROP any Insight lacking them (prompts.py STRUCTURED-INPUT RULES). The parser
    deliberately does NOT re-enforce those: concreteness (``_is_concrete`` +
    non-empty text) is the parser-level quality gate; the prompt's hard rules
    enforce the richer requirements. The design note lives at
    ``prompts.py`` ("items are permissive ... because the prompt's hard rules,
    not the schema, enforce Insight quality"). So a concrete, text-only insight
    dict is KEPT -- and no phantom evidence/applicability is invented for it."""
    items = _as_insight_list(
        [{"text": "for divider-based grids, split on monochromatic divider rows before recoloring"}]
    )
    assert len(items) == 1
    assert items[0]["text"].startswith("for divider-based grids")
    assert "evidence" not in items[0]
    assert "applies_when" not in items[0]


def test_as_insight_list_strips_fabricated_evidence_but_keeps_concrete_text():
    """The trust boundary (#1178) is where "unsupported" is actually enforced --
    NOT by requiring every insight to carry evidence.

    An insight that CLAIMS evidence citing a post outside the loaded set has that
    fabricated provenance dropped (never promoted to trusted), while its concrete
    text survives exactly as a bare text-only insight would. This pins that the
    parser is not more permissive than the downstream *trust model* (which only
    trusts membership-verified post ids) -- only more permissive than the prompt,
    by design. Tightening the parser to drop such dicts wholesale would
    over-reach and contradict the text-only-is-kept contract above."""
    items = _as_insight_list(
        [
            {
                "text": "when a bounding box touches the grid border, clamp the offset to the edge",
                "evidence": [{"post_id": 999, "quote": "fabricated"}],
            }
        ],
        allowed_post_ids={1, 2},
    )
    assert len(items) == 1
    # post 999 is outside the loaded set -> the fabricated citation is stripped,
    # but the concrete rule text is retained.
    assert "evidence" not in items[0]
    assert items[0]["text"].startswith("when a bounding box")


def test_dedupe_drops_cross_field_duplicates():
    bundle = {
        "transferable_insights": [{"text": "extract seed bounding box and compute offset fingerprints for all blocks"}],
        "confirmed_constraints": [
            {"text": "Extract seed bounding box and compute offset fingerprints for all blocks."}
        ],
        "pitfalls": [{"text": "hardcoding sampling indices from a single training pair"}],
    }
    out = dedupe_bundle_items(bundle)
    assert out["transferable_insights"]
    assert not out["confirmed_constraints"]
    assert out["pitfalls"]


def test_dedupe_keeps_distinct_items():
    bundle = {
        "checks": [
            {"text": "verify divider rows are monochromatic before reflecting"},
            {"text": "confirm delta vectors have cardinality one per color"},
        ]
    }
    out = dedupe_bundle_items(bundle)
    assert len(out["checks"]) == 2


def test_distill_one_task_drops_cross_field_duplicates():
    posts = [{"id": 1, "agent_id": "a1", "text": "grounded"}]

    def fake_llm(sys_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "transferable_insights": [
                    {"text": "extract seed bounding box and compute offset fingerprints for all blocks"}
                ],
                "confirmed_constraints": [
                    {"text": "Extract seed bounding box and compute offset fingerprints for all blocks."}
                ],
                "pitfalls": [{"text": "hardcoding sampling indices from a single training pair"}],
                "evidence_post_ids": [1],
            }
        )

    b = distill_one_task(task_id="t1", attempts=[], posts=posts, llm=fake_llm)
    assert isinstance(b, PerTaskBundle)
    assert b.transferable_insights
    assert b.confirmed_constraints == []
    assert b.pitfalls


def test_parse_json_uses_first_balanced_object():
    payload = _parse_json(
        '```json\n{"transferable_insights": ["keep {literal} braces"]}\n```\nextra commentary with {not json}'
    )
    assert payload["transferable_insights"] == ["keep {literal} braces"]


def test_parse_json_repairs_common_model_slips():
    payload = _parse_json(
        '{\n  "checks": ["regex \\d+ still means digits",],\n  "pitfalls": ["line one\nline two"],\n}\n'
    )
    assert payload["checks"] == ["regex \\d+ still means digits"]
    assert payload["pitfalls"] == ["line one\nline two"]
