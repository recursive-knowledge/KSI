"""Tests for patch extraction and normalize_patch normalization."""

from ksi.eval.patch_extract import extract_patch, normalize_patch

# ---------------------------------------------------------------------------
# normalize_patch: fabricated index lines
# ---------------------------------------------------------------------------


def test_normalize_strips_fabricated_index_non_hex():
    """Index lines with non-hex characters are dropped."""
    patch = "diff --git a/f.py b/f.py\nindex 1234567..abcdefg 100644\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
    result = normalize_patch(patch)
    assert result is not None
    assert "index 1234567..abcdefg" not in result
    assert "--- a/f.py" in result


def test_normalize_strips_fabricated_index_words():
    """Index lines like 'index old..new' or 'index original..modified' are dropped."""
    for bad_index in ("index old..new", "index original..modified", "index abc..def"):
        patch = f"diff --git a/f.py b/f.py\n{bad_index}\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y\n"
        result = normalize_patch(patch)
        assert result is not None
        assert bad_index not in result, f"Expected {bad_index!r} to be dropped"


def test_normalize_preserves_valid_index_line():
    """Real 7+ char all-hex hashes (with or without mode) are preserved."""
    valid_cases = [
        "index a1b2c3d..e4f5a6b",
        "index a1b2c3d..e4f5a6b 100644",
        "index 0000000..1234567 100755",
        "index abcdef0123456789..fedcba9876543210",
    ]
    for valid_index in valid_cases:
        patch = f"diff --git a/f.py b/f.py\n{valid_index}\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
        result = normalize_patch(patch)
        assert result is not None
        assert valid_index in result, f"Expected {valid_index!r} to be preserved"


# ---------------------------------------------------------------------------
# normalize_patch: CRLF → LF
# ---------------------------------------------------------------------------


def test_normalize_crlf_to_lf():
    """CRLF line endings are converted to LF."""
    patch = "diff --git a/f.py b/f.py\r\n--- a/f.py\r\n+++ b/f.py\r\n@@ -1 +1 @@\r\n-old\r\n+new\r\n"
    result = normalize_patch(patch)
    assert result is not None
    assert "\r" not in result


def test_normalize_bare_cr_to_lf():
    """Bare CR line endings are converted to LF."""
    patch = "diff --git a/f.py b/f.py\r--- a/f.py\r+++ b/f.py\r@@ -1 +1 @@\r-old\r+new\r"
    result = normalize_patch(patch)
    assert result is not None
    assert "\r" not in result


# ---------------------------------------------------------------------------
# normalize_patch: trailing whitespace on context lines
# ---------------------------------------------------------------------------


def test_normalize_strips_trailing_whitespace_on_context_lines():
    """Trailing whitespace on context (space-prefixed) and header lines is stripped."""
    patch = "diff --git a/f.py b/f.py   \n--- a/f.py   \n+++ b/f.py   \n@@ -1,2 +1,2 @@\n context line   \n-old\n+new\n"
    result = normalize_patch(patch)
    assert result is not None
    # Header lines trimmed
    assert "diff --git a/f.py b/f.py\n" in result
    assert "--- a/f.py\n" in result
    assert "+++ b/f.py\n" in result
    # Context line trimmed
    assert " context line\n" in result


def test_normalize_preserves_trailing_whitespace_on_diff_lines():
    """Trailing whitespace on +/- lines is significant and must NOT be stripped."""
    patch = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old   \n+new   \n"
    result = normalize_patch(patch)
    assert result is not None
    assert "-old   \n" in result
    assert "+new   \n" in result


def test_normalize_strips_workspace_repo_prefix_from_headers():
    """Diffs captured from the workspace root should apply at repo root."""
    patch = (
        "diff --git a/repo/pkg/f.py b/repo/pkg/f.py\n"
        "--- a/repo/pkg/f.py\n"
        "+++ b/repo/pkg/f.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    result = normalize_patch(patch)
    assert result is not None
    assert "diff --git a/pkg/f.py b/pkg/f.py\n" in result
    assert "--- a/pkg/f.py\n" in result
    assert "+++ b/pkg/f.py\n" in result
    assert "repo/pkg" not in result


def test_normalize_strips_workspace_repo_prefix_from_rename_headers():
    patch = (
        "diff --git a/repo/old.py b/repo/new.py\n"
        "similarity index 100%\n"
        "rename from repo/old.py\n"
        "rename to repo/new.py\n"
    )
    result = normalize_patch(patch)
    assert result is not None
    assert "diff --git a/old.py b/new.py\n" in result
    assert "rename from old.py\n" in result
    assert "rename to new.py\n" in result


# ---------------------------------------------------------------------------
# normalize_patch: trailing newline
# ---------------------------------------------------------------------------


def test_normalize_ensures_trailing_newline():
    """normalize_patch always appends a trailing newline if missing."""
    patch = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new"
    result = normalize_patch(patch)
    assert result is not None
    assert result.endswith("\n")


def test_normalize_does_not_double_trailing_newline():
    """normalize_patch does not add a second newline if one already exists."""
    patch = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
    result = normalize_patch(patch)
    assert result is not None
    assert result.endswith("\n")
    assert not result.endswith("\n\n")


# ---------------------------------------------------------------------------
# normalize_patch: None / empty passthrough
# ---------------------------------------------------------------------------


def test_normalize_none_returns_none():
    assert normalize_patch(None) is None


def test_normalize_empty_string_returns_none():
    assert normalize_patch("") is None


def test_normalize_whitespace_only_returns_none():
    assert normalize_patch("   \n\n  ") is None


# ---------------------------------------------------------------------------
# extract_patch: applies normalization automatically
# ---------------------------------------------------------------------------


def test_extract_patch_from_xml_tags():
    text = (
        "Here is my fix:\n<patch>\ndiff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n</patch>"
    )
    patch = extract_patch(text)
    assert patch is not None
    assert patch.startswith("diff --git")
    assert patch.endswith("\n")


def test_extract_patch_normalizes_index_lines():
    """extract_patch strips fabricated index lines via normalize_patch."""
    text = (
        "<patch>\n"
        "diff --git a/f.py b/f.py\n"
        "index old..new 100644\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "</patch>"
    )
    patch = extract_patch(text)
    assert patch is not None
    assert "index old..new" not in patch
    assert "--- a/f.py" in patch


def test_extract_patch_ensures_trailing_newline():
    """extract_patch (via normalize_patch) ensures trailing newline."""
    text = "<patch>\ndiff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new</patch>"
    patch = extract_patch(text)
    assert patch is not None
    assert patch.endswith("\n")


def test_extract_patch_normalizes_crlf():
    """extract_patch converts CRLF to LF."""
    text = "<patch>\r\ndiff --git a/f.py b/f.py\r\n--- a/f.py\r\n+++ b/f.py\r\n@@ -1 +1 @@\r\n-old\r\n+new\r\n</patch>"
    patch = extract_patch(text)
    assert patch is not None
    assert "\r" not in patch


def test_extract_patch_empty_returns_none():
    assert extract_patch("") is None
    assert extract_patch("<patch></patch>") is None
    assert extract_patch("<patch>   </patch>") is None


def test_extract_patch_marker_delimiters():
    text = (
        "---PATCH_START---\ndiff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n---PATCH_END---"
    )
    patch = extract_patch(text)
    assert patch is not None
    assert patch.startswith("diff --git")
    assert patch.endswith("\n")


def test_extract_patch_fenced_block():
    text = "```diff\ndiff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n```"
    patch = extract_patch(text)
    assert patch is not None
    assert patch.startswith("diff --git")
    assert patch.endswith("\n")


def test_extract_patch_prefers_last_structured_candidate():
    text = (
        "<patch>\n"
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+draft\n"
        "</patch>\n"
        "Final patch:\n"
        "<patch>\n"
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+final\n"
        "</patch>\n"
    )
    patch = extract_patch(text)
    assert patch is not None
    assert "+final\n" in patch
    assert "+draft\n" not in patch


def test_extract_patch_prefers_later_fenced_candidate_over_earlier_xml():
    text = (
        "<patch>\n"
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+xml\n"
        "</patch>\n"
        "```diff\n"
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+fenced\n"
        "```"
    )
    patch = extract_patch(text)
    assert patch is not None
    assert "+fenced\n" in patch
    assert "+xml\n" not in patch


def test_extract_patch_raw_diff():
    text = "Some output\ndiff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
    patch = extract_patch(text)
    assert patch is not None
    assert patch.startswith("diff --git")
    assert patch.endswith("\n")
