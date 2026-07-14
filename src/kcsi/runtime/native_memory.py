from __future__ import annotations

import os
from pathlib import Path


def _sessions_root_candidates() -> list[Path]:
    """Host-side locations where provider session artifacts may exist."""
    return [
        Path("runtime_state/provider_sessions/tasks"),
        Path("runtime_state/provider_sessions/agents"),
    ]


def _latest_existing(paths: list[Path]) -> Path | None:
    """Return the first existing path candidate, else None."""
    for p in paths:
        if p.exists():
            return p
    return None


def _preferred_session_file(root: Path, file_path: Path) -> bool:
    rel_parts = file_path.relative_to(root).parts
    return (
        len(rel_parts) >= 2
        and rel_parts[0] == "projects"
        and file_path.suffix.lower() == ".jsonl"
        and not file_path.name.lower().startswith("agent-")
    )


def _fallback_session_file(root: Path, file_path: Path) -> bool:
    rel_parts = {part.lower() for part in file_path.relative_to(root).parts}
    if file_path.suffix.lower() != ".jsonl":
        return False
    if file_path.name.lower().startswith("agent-"):
        return False
    if rel_parts & {"debug", "todos", "shell-snapshots", "skills"}:
        return False
    return True


def _strip_sidechain_entries(raw: str, file_path: Path) -> str:
    if file_path.suffix.lower() != ".jsonl":
        return raw
    kept: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = __import__("json").loads(stripped)
        except Exception:
            kept.append(line)
            continue
        if isinstance(parsed, dict) and parsed.get("isSidechain") is True:
            continue
        kept.append(line)
    return "\n".join(kept)


def _env_int(name: str, default: int) -> int:
    """Parse integer env var with fallback.

    Non-integer values are ignored and `default` is used.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def collect_native_session_memory(
    group_folder: str,
    *,
    max_chars: int | None = None,
    max_files: int | None = None,
    max_chars_per_file: int | None = None,
) -> str:
    """Collect bounded native session memory text for one workspace key.

    Defaults are read from env vars:
    - KCSI_NATIVE_MEMORY_MAX_CHARS (default 240000)
    - KCSI_NATIVE_MEMORY_MAX_FILES (default 8)
    - KCSI_NATIVE_MEMORY_MAX_CHARS_PER_FILE (default 60000)

    Cap semantics:
    - positive integer => enforced cap
    - 0 or negative => disabled (no capture / no limit for that dimension)
    - ``max_chars <= 0`` ⇒ return empty string (capture disabled)
    - ``max_files <= 0`` ⇒ include all files (no file-count limit)
    - ``max_chars_per_file <= 0`` ⇒ read full file content (no per-file limit)
    """
    if not group_folder:
        return ""
    if max_chars is not None and max_chars <= 0:
        return ""
    if max_chars is None:
        max_chars = _env_int("KCSI_NATIVE_MEMORY_MAX_CHARS", 240_000)
    if max_files is None:
        max_files = _env_int("KCSI_NATIVE_MEMORY_MAX_FILES", 8)
    if max_chars_per_file is None:
        max_chars_per_file = _env_int("KCSI_NATIVE_MEMORY_MAX_CHARS_PER_FILE", 60_000)
    sessions_root = _latest_existing([p / group_folder / ".claude" for p in _sessions_root_candidates()])
    if sessions_root is None:
        return ""

    preferred_files = [p for p in sessions_root.rglob("*") if p.is_file() and _preferred_session_file(sessions_root, p)]
    files = (
        preferred_files
        if preferred_files
        else [p for p in sessions_root.rglob("*") if p.is_file() and _fallback_session_file(sessions_root, p)]
    )
    if not files:
        return ""

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    blocks: list[str] = []
    total = 0
    selected_files = files if max_files is None or max_files <= 0 else files[:max_files]
    for f in selected_files:
        try:
            raw = f.read_text(encoding="utf-8")
        except Exception:
            continue
        raw = _strip_sidechain_entries(raw, f)
        if not raw.strip():
            continue
        if max_chars_per_file is None or max_chars_per_file <= 0:
            chunk = raw
        else:
            chunk = raw[-max_chars_per_file:] if len(raw) > max_chars_per_file else raw
        rel = f.relative_to(sessions_root)
        wrapped = f"# file: {rel}\n{chunk}\n"
        separator_cost = len("\n\n---\n\n") if blocks else 0
        blocks.append(wrapped)
        total += len(wrapped) + separator_cost
        if max_chars is not None and max_chars > 0 and total >= max_chars:
            break

    merged = "\n\n---\n\n".join(blocks).strip()
    if max_chars is not None and max_chars > 0 and len(merged) > max_chars:
        return merged[-max_chars:]
    return merged
