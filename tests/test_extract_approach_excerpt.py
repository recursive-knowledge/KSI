# tests/test_extract_approach_excerpt.py
"""Tests for _extract_approach_excerpt in src/ksi/orchestrator/engine.py."""

from ksi.orchestrator.engine import _extract_approach_excerpt


class TestExtractApproachExcerpt:
    def test_empty_text_returns_empty(self):
        assert _extract_approach_excerpt("") == ""

    def test_returns_up_to_max_chars(self):
        text = "This is a substantive line that is longer than ten chars."
        result = _extract_approach_excerpt(text, max_chars=10)
        assert len(result) <= 10

    def test_skips_preamble_lines(self):
        text = "I'll start by reading the file.\nLet me check the imports.\nFixed the parser bug in line 42."
        result = _extract_approach_excerpt(text, max_chars=300)
        assert result.startswith("Fixed the parser bug")

    def test_fallback_when_all_lines_are_preamble(self):
        text = "I'll do this.\nLet me think.\nI need to check."
        result = _extract_approach_excerpt(text, max_chars=300)
        # When all lines match preamble, start stays 0 and full text is used as fallback
        assert len(result) > 0
        # The fallback joins all lines
        assert "I'll do this." in result
