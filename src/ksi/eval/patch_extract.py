from __future__ import annotations

import re

# Matches valid git index lines: hex hashes of 7–40 chars, optional mode
_VALID_INDEX_RE = re.compile(r"^index [0-9a-f]{7,40}\.\.[0-9a-f]{7,40}( \d{6})?$")


def _strip_workspace_repo_prefix(line: str) -> str:
    """Strip the runtime workspace checkout prefix from diff headers.

    Agents sometimes run ``git diff`` from ``/workspace/task/workspace`` instead
    of ``/workspace/task/workspace/repo``. That produces paths like
    ``a/repo/src/file.py`` even though the evaluator applies patches at the
    repository root. Normalize only git diff headers, not patch content.
    """
    if line.startswith("diff --git a/repo/") and " b/repo/" in line:
        return line.replace("diff --git a/repo/", "diff --git a/", 1).replace(" b/repo/", " b/", 1)
    if line.startswith("--- a/repo/"):
        return "--- a/" + line[len("--- a/repo/") :]
    if line.startswith("+++ b/repo/"):
        return "+++ b/" + line[len("+++ b/repo/") :]
    if line.startswith("rename from repo/"):
        return "rename from " + line[len("rename from repo/") :]
    if line.startswith("rename to repo/"):
        return "rename to " + line[len("rename to repo/") :]
    return line


def normalize_patch(patch: str | None) -> str | None:
    """Normalize a unified diff patch for reliable application.

    - Converts CRLF and bare CR line endings to LF.
    - Strips fabricated ``index`` lines whose hashes are not valid hex
      (e.g. ``index 1234567..abcdefg`` or ``index old..new``).
    - Strips trailing whitespace from context / header lines only
      (``+``/``-``/``@@`` lines are left untouched to preserve semantics).
    - Ensures the result ends with a newline.
    - Returns ``None`` for empty / whitespace-only input.
    """
    if not patch:
        return None

    # CRLF → LF, bare CR → LF
    patch = patch.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in patch.split("\n"):
        line = _strip_workspace_repo_prefix(line)
        if line.startswith("index "):
            if _VALID_INDEX_RE.match(line):
                lines.append(line)
            # Drop fabricated index lines silently
            continue
        # Strip trailing whitespace on everything except actual diff content lines.
        # Protect +/- lines (added/removed lines) but NOT --- / +++ headers.
        # Context lines (space-prefixed or blank) and all other header lines get trimmed.
        is_diff_content = (line.startswith("+") and not line.startswith("+++")) or (
            line.startswith("-") and not line.startswith("---")
        )
        if not is_diff_content:
            line = line.rstrip()
        lines.append(line)

    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result if result.strip() else None


def extract_patch(text: str) -> str | None:
    """Extract a unified diff patch from raw model output.

    Supports all common formats agents may produce:
    1. ``<patch>...</patch>`` XML tags (SWE-bench convention)
    2. ``---PATCH_START---...---PATCH_END---`` marker delimiters
    3. Markdown fenced blocks (```diff or ```patch)
    4. Raw unified diff starting with ``diff --git`` or ``--- a/``

    The extracted patch is passed through :func:`normalize_patch` before
    being returned, so callers receive a consistently formatted diff.
    """
    if not text:
        return None

    # Prefer the last structured patch candidate. Model outputs commonly
    # include an earlier draft followed by a corrected final patch.
    candidates: list[tuple[int, str]] = []

    # 1. XML-style <patch> tags
    for m in re.finditer(r"<patch>\s*(.*?)\s*</patch>", text, flags=re.DOTALL | re.IGNORECASE):
        patch = m.group(1).strip()
        if patch:
            candidates.append((m.start(), patch))

    # 2. Marker delimiters
    for m in re.finditer(r"---PATCH_START---\s*(.*?)\s*---PATCH_END---", text, flags=re.DOTALL):
        patch = m.group(1).strip()
        if patch:
            candidates.append((m.start(), patch))

    # 3. Fenced diff block (```diff ... ``` or ```patch ... ```)
    for m in re.finditer(r"```(?:diff|patch)\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        patch = m.group(1).strip()
        if patch:
            candidates.append((m.start(), patch))

    if candidates:
        _, patch = max(candidates, key=lambda item: item[0])
        return normalize_patch(patch)

    # 4. Raw unified diff: "diff --git" or "--- a/" followed by "+++ b/"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("diff --git ") or (
            line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")
        ):
            candidate = "\n".join(lines[i:]).strip()
            if candidate:
                return normalize_patch(candidate)

    return None
