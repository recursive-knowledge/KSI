from __future__ import annotations

import json
import logging
import threading
import time as _time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    cleaned = []
    for ch in value or "":
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("._")
    return out or "default"


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ForumBusEvent:
    seq: int
    event_id: str
    generation: int
    round_num: int | None
    agent_id: str
    message_type: str
    content: dict[str, Any]
    ts: str


class ForumBus:
    """File-backed append-only bus for forum round interaction.

    The bus is intentionally simple:
    - many writers append JSON lines (MCP tool calls from separate processes)
    - orchestrator reads incrementally and persists to SQLite as single writer
    """

    def __init__(self, *, db_path: str, experiment: str, generation: int) -> None:
        db_parent = Path(db_path).resolve().parent
        bus_dir = db_parent / "forum_bus"
        # mkdir may silently succeed even on a read-only bind-mount when the
        # directory already exists (exist_ok=True suppresses EEXIST/EROFS).
        try:
            bus_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # read-only mount — bus_dir exists; reads will still work
        stem = f"{_safe_name(experiment)}__g{int(generation)}"
        self._events_path = bus_dir / f"{stem}.events.jsonl"
        self._lock_path = bus_dir / f"{stem}.lock"
        # Sidecar of event_ids that the orchestrator's drain should skip.
        # The retry helper writes here when a forum-task attempt fails after
        # already appending events to the bus: the failed
        # attempt's events have unique fb-uuid external_ids so the existing
        # bulk_has_external_ids dedup can't catch them, and on retry the bus
        # would
        # otherwise carry both the failed-attempt and success-attempt copies
        # of the same forum_post / insight / comment / done events.
        self._stale_path = bus_dir / f"{stem}.stale_events.jsonl"
        # Detect read-only filesystem: touch() with exist_ok=True first tries
        # os.utime() (update mtime), which raises OSError on a read-only bind-
        # mount even when the file already exists, then falls through to
        # os.open(O_CREAT|O_WRONLY) which also raises.  Catch both cases and
        # mark the bus read-only so that append() / clear() are no-ops.
        self._read_only = False
        try:
            self._events_path.touch(exist_ok=True)
            self._lock_path.touch(exist_ok=True)
            self._stale_path.touch(exist_ok=True)
        except OSError:
            # Running inside a read-only container bind-mount — bus is
            # read-only.  Existing events from the host are still readable
            # via read_events(); writes are silently skipped.
            self._read_only = True
            log.debug(
                "ForumBus: bus directory is read-only (%s); append/clear disabled — reads still work",
                bus_dir,
            )
        self._local_lock = threading.RLock()
        self._generation = int(generation)

    def clear(self) -> None:
        """Clear bus events for this generation."""
        if self._read_only:
            return
        with self._local_lock:
            self._events_path.write_text("", encoding="utf-8")
            try:
                self._stale_path.write_text("", encoding="utf-8")
            except OSError:
                # Sidecar absence is benign — the drain treats a missing
                # stale file as "no stale events".
                pass

    def mark_stale(
        self,
        event_ids: list[str] | tuple[str, ...] | set[str],
        *,
        reason: str = "failed_attempt",
    ) -> None:
        """Mark a set of bus event_ids as stale (skipped at drain time).

        Used by ``_run_retryable_forum_task`` after a forum-task attempt
        fails: any events the failed attempt wrote to the bus before the
        SDK iterator drained must not land in ``KnowledgeStore`` because
        the retry will write fresh (non-deterministic, ``temperature>0``)
        replacements that ``bulk_has_external_ids`` cannot dedup against.

        The sidecar is append-only JSONL alongside ``<stem>.events.jsonl``;
        readers union all entries.  Empty input is a no-op.
        """
        ids = [str(e).strip() for e in event_ids if str(e).strip()]
        if not ids or self._read_only:
            return
        ts = datetime.now(timezone.utc).isoformat()
        lines = [json.dumps({"event_id": ev_id, "reason": str(reason or ""), "ts": ts}) + "\n" for ev_id in ids]
        payload = "".join(lines)
        with self._local_lock:
            with open(self._lock_path, "a+b") as lock_fp:
                if fcntl is not None:
                    _deadline = _time.monotonic() + 60.0
                    while True:
                        try:
                            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except OSError:
                            if _time.monotonic() >= _deadline:
                                log.warning("Forum bus stale-sidecar flock timeout after 60s — proceeding without lock")
                                break
                            _time.sleep(0.05)
                try:
                    with open(self._stale_path, "a", encoding="utf-8") as fp:
                        fp.write(payload)
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def read_stale_event_ids(self) -> set[str]:
        """Return the set of event_ids the drain must skip.

        A missing or unparseable sidecar yields an empty set so the bus
        is forward-compatible with directories created before the
        sidecar was introduced.
        """
        with self._local_lock:
            try:
                text = self._stale_path.read_text(encoding="utf-8")
            except Exception:
                return set()
        out: set[str] = set()
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            ev_id = str(obj.get("event_id") or "").strip()
            if ev_id:
                out.add(ev_id)
        return out

    def append(
        self,
        *,
        round_num: int | None,
        agent_id: str,
        message_type: str,
        content: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._read_only:
            # Silently skip writes — container is on a read-only bind-mount.
            # Forum writes from containers go through text-output parsing on
            # the orchestrator side anyway; the bus is only used by the
            # orchestrator (host) to relay events to the SQLite store.
            return {}
        payload = {
            "event_id": f"fb-{uuid.uuid4().hex}",
            "generation": self._generation,
            "round_num": int(round_num) if round_num is not None else None,
            "agent_id": str(agent_id or ""),
            "message_type": str(message_type or ""),
            "content": content or {},
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(payload, ensure_ascii=True) + "\n"
        with self._local_lock:
            with open(self._lock_path, "a+b") as lock_fp:
                if fcntl is not None:
                    # Non-blocking retry with 60s timeout (avoid indefinite block)
                    _deadline = _time.monotonic() + 60.0
                    while True:
                        try:
                            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except OSError:
                            if _time.monotonic() >= _deadline:
                                log.warning("Forum bus flock timeout after 60s — proceeding without lock")
                                break
                            _time.sleep(0.05)
                try:
                    with open(self._events_path, "a", encoding="utf-8") as fp:
                        fp.write(line)
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        return payload

    def read_events(
        self,
        *,
        after_seq: int = 0,
        round_num: int | None = None,
        message_types: set[str] | None = None,
    ) -> list[ForumBusEvent]:
        events: list[ForumBusEvent] = []
        with self._local_lock:
            try:
                text = self._events_path.read_text(encoding="utf-8")
            except Exception:
                return events
        for idx, raw in enumerate(text.splitlines(), start=1):
            if idx <= int(after_seq):
                continue
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            msg_type = str(item.get("message_type") or "")
            item_round = item.get("round_num")
            parsed_round = None
            if item_round is not None:
                parsed_round = _coerce_int(item_round)
                if parsed_round is None:
                    continue
            if round_num is not None:
                if item_round is not None and parsed_round != int(round_num):
                    continue
                if item_round is None and msg_type != "done":
                    continue
            if message_types is not None and msg_type not in message_types:
                continue
            parsed_generation = _coerce_int(item.get("generation") or self._generation)
            if parsed_generation is None:
                continue
            raw_content = item.get("content")
            content: dict[str, Any] = dict(raw_content) if isinstance(raw_content, dict) else {}
            events.append(
                ForumBusEvent(
                    seq=idx,
                    event_id=str(item.get("event_id") or ""),
                    generation=parsed_generation,
                    round_num=parsed_round,
                    agent_id=str(item.get("agent_id") or ""),
                    message_type=msg_type,
                    content=content,
                    ts=str(item.get("ts") or ""),
                )
            )
        return events

    def read_messages(
        self,
        *,
        round_num: int | None = None,
        up_to_round: bool = False,
    ) -> list[dict[str, Any]]:
        events = self.read_events(
            after_seq=0,
            round_num=None if up_to_round else round_num,
            message_types={"insight", "post", "comment", "cluster"},
        )
        if up_to_round and round_num is not None:
            limit = int(round_num)
            events = [ev for ev in events if ev.round_num is not None and int(ev.round_num) <= limit]
        return [
            {
                "id": ev.seq,
                "generation": ev.generation,
                "agent_id": ev.agent_id,
                "message_type": ev.message_type,
                "content": ev.content,
                "round_num": ev.round_num,
                "created_at": ev.ts,
            }
            for ev in events
        ]
