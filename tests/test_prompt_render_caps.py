"""Pin the prompt-render cap constants that tune the KT-sweep signal chain.

The values were set by the 2026-04-20 audit — changing them changes how much
context reaches the distiller + forum rounds. Keep this test in sync with
the constant docstrings and the distillation+seeding counterparts, not the
other way around.
"""

from __future__ import annotations

from ksi.distillation import prompts as dp
from ksi.forum import prompt as fp


def test_forum_prompt_cap_constants_pinned():
    # V2 bumps:
    # - POST_SNIPPET_CHARS 1200→2000: posts are now structured JSON
    #   post-mortems averaging 1.5-2KB; old cap clipped proposed_change.
    # (The forum-side _FORUM_MODEL_OUTPUT_EXCERPT_CHARS and
    # _PRIOR_CROSS_TASK_BUNDLE_CHARS caps were removed with the legacy R1/R2
    # forum in PR #678; the surviving distillation-side cap is pinned by
    # test_distillation_input_cap_constants_pinned below.)
    assert fp._TASK_DESCRIPTION_PREVIEW_CHARS == 1200
    assert fp._POST_SNIPPET_CHARS == 2000
    assert fp._NATIVE_MEMORY_FORUM_INJECT_CHARS == 32000


def test_distillation_input_cap_constants_pinned():
    # V2 bumps mirror the forum-side bumps for the same reasons.
    assert dp._ATTEMPT_OUTPUT_EXCERPT_CHARS == 2000
    assert dp._POST_TEXT_EXCERPT_CHARS == 2000
    assert dp._EVAL_SUMMARY_CHARS == 1600


def test_native_memory_cap_respects_cli_flag_semantics():
    """--native-memory-max-chars can raise the value; the forum inject cap
    is the later clip. Keep it large enough that the flag is not silently
    eaten by a smaller downstream cap."""
    # The CLI flag's default is 240_000. The forum inject cap must be high
    # enough that users raising the flag see at least a meaningful chunk
    # reach the forum prompt. 32k is ~13% of default; regression floor.
    assert fp._NATIVE_MEMORY_FORUM_INJECT_CHARS >= 16_000
