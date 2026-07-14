"""Pin the public anti-meta block module — single source of truth."""

from kcsi.discussion.concreteness import (
    ANTI_META_BLOCK,
    assert_anti_meta_rules_present,
)


def test_block_contains_strict_header():
    assert "STRICT ANTI-META RULES" in ANTI_META_BLOCK


def test_block_contains_rejected_phrases():
    for phrase in (
        "validate first",
        "decompose before solving",
        "check edge cases",
        "boundary conditions",
        "think step by step",
    ):
        assert phrase in ANTI_META_BLOCK


def test_block_contains_concrete_grounding_clause():
    assert "REQUIRE concrete grounding" in ANTI_META_BLOCK
    assert "Abstract nouns alone" in ANTI_META_BLOCK


def test_block_contains_evidence_grounding_clause():
    assert "REQUIRE evidence grounding" in ANTI_META_BLOCK
    assert "evidence_post_ids" in ANTI_META_BLOCK


def test_block_contains_quality_caps():
    assert "at most 5 insights" in ANTI_META_BLOCK
    assert "5 pitfalls" in ANTI_META_BLOCK
    assert "5 checks" in ANTI_META_BLOCK


def test_assert_helper_passes_on_block_itself():
    assert_anti_meta_rules_present(ANTI_META_BLOCK)


def test_assert_helper_raises_on_empty_text():
    import pytest

    with pytest.raises(AssertionError):
        assert_anti_meta_rules_present("")


def test_distiller_imports_from_shared_module():
    from kcsi.distillation import prompts as distill_prompts

    assert distill_prompts._ANTI_META_BLOCK is ANTI_META_BLOCK
