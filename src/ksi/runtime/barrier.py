"""Reusable file-sentinel barrier protocol for in-flight host<->container coordination.

Motivation
----------
The agent-runner inside the container and the Python orchestrator host
sometimes need a synchronous round-trip *during* a single task run — e.g.
the host evaluates the agent's submission and feeds the score back so the
agent can write a structured reflection in the same SDK session.

The container's stdout is single-shot (fully captured by `proc.communicate`),
and the host doesn't have a streaming MCP channel into the container at the
scheduled-task level. The simplest robust mechanism is a shared workspace
directory polled by both sides.

Protocol
--------
The container writes a *sentinel* file when it has reached a barrier point
and is ready for input from the host. The host's `BarrierWatcher` thread
polls for the sentinel, computes the response, and writes a *response* file.
The container polls for the response file and consumes it.

File names (relative to ``/workspace/task`` inside the container, which maps
to a host directory under ``workspaces/tasks/<key>/workspace/task/``):

  - sentinel:  ``.barrier.<name>.<agent_id>.ready``       (agent -> host)
  - response:  ``.barrier.<name>.<agent_id>.response``    (host  -> agent)

The sentinel file's content is a small JSON payload describing the barrier
context (e.g. what the host should compute). The response file's content is
JSON-encoded data the agent will consume. Both files are best-effort
deleted by their reader after consumption — any leftovers from a previous
run on the same workspace key are cleaned up by ``BarrierWatcher.start()``.

Timeout semantics
-----------------
* TS side ``waitForBarrierFile`` returns ``null`` after its poll budget;
  callers degrade gracefully (never fail the attempt for a missing
  reflection).
* Python ``BarrierWatcher`` runs its callback only when the sentinel
  appears. If the watched workspace never produces a sentinel, the watcher
  simply exits when ``stop()`` is called; no response file is written.

This module is intentionally provider/feature-agnostic so future
host<->container barriers (e.g. Phase 3 R0->R1 forum coordination) can
reuse it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# Default poll cadence; matches the TS ``IPC_POLL_MS`` rhythm so host and
# container observe each other on similar timescales.
DEFAULT_POLL_INTERVAL_SEC = 0.5

# Sentinel payloads must stay well under this to be read into memory. The
# TS side caps `model_output` (the largest field a sentinel carries) at 8MB
# (see phase1_reflection.ts / polyglot_test_feedback.ts); this leaves
# headroom for JSON/schema overhead while still rejecting a pathological or
# adversarial oversized write before it's ever loaded.
MAX_SENTINEL_BYTES = 16_000_000


def sentinel_filename(name: str, agent_id: str) -> str:
    """Return the basename of the sentinel file an agent writes when ready.

    Format: ``.barrier.<name>.<agent_id>.ready``. The leading dot keeps the
    file out of any tooling that filters dotfiles, and the structured name
    makes it easy to grep for in a workspace.
    """
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    safe_agent = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(agent_id))
    return f".barrier.{safe_name}.{safe_agent}.ready"


def response_filename(name: str, agent_id: str) -> str:
    """Return the basename of the response file the host writes back."""
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    safe_agent = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(agent_id))
    return f".barrier.{safe_name}.{safe_agent}.response"


@dataclass
class BarrierEvent:
    """Decoded contents of a sentinel file the host has just observed."""

    sentinel_path: Path
    response_path: Path
    payload: dict[str, Any]


class BarrierWatcher(threading.Thread):
    """Polls a workspace directory for a barrier sentinel and writes a response.

    Parameters
    ----------
    workspace_dir:
        Absolute path to the workspace directory the container has mounted as
        ``/workspace/task``. The watcher polls for sentinel files inside this
        directory directly. May not exist when the watcher is constructed —
        the watcher simply waits for it.
    name:
        Logical barrier name (e.g. ``phase1_reflection``).
    agent_id:
        Agent id token used to namespace the sentinel/response pair.
    callback:
        Function invoked when the sentinel is observed. Receives a
        :class:`BarrierEvent` and must return a JSON-serializable dict that is
        written verbatim into the response file. Exceptions in the callback
        are logged and the watcher writes a ``{"error": ...}`` response.
    poll_interval_sec:
        How often to scan the workspace for the sentinel.
    timeout_sec:
        Optional upper bound on how long the watcher polls before giving up.
        ``None`` (default) means wait indefinitely until ``stop()``.
    persistent:
        When ``False`` (default), the watcher answers exactly one
        sentinel/response round-trip then exits — the right behavior for
        single-shot barriers like ``phase1_reflection``. When ``True``, the
        watcher re-arms after writing each response and keeps polling for a
        follow-up sentinel (resetting its wait budget to a fresh
        ``timeout_sec`` window each round) until ``stop()`` is called or a
        round's wait budget elapses. Needed by multi-round barriers like
        ``polyglot_test_feedback``, whose caller only calls ``stop()`` once
        after the whole container subprocess exits.
    """

    def __init__(
        self,
        *,
        workspace_dir: Path,
        name: str,
        agent_id: str,
        callback: Callable[[BarrierEvent], dict[str, Any]],
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        timeout_sec: Optional[float] = None,
        persistent: bool = False,
    ) -> None:
        super().__init__(daemon=True, name=f"barrier-watcher-{name}-{agent_id}")
        self._workspace_dir = Path(workspace_dir)
        self._name = name
        self._agent_id = str(agent_id)
        self._callback = callback
        self._poll_interval = max(0.05, float(poll_interval_sec))
        self._timeout = float(timeout_sec) if timeout_sec is not None else None
        self._persistent = bool(persistent)
        self._stop_event = threading.Event()
        self._fired = threading.Event()
        self._error: Optional[BaseException] = None

    @property
    def sentinel_path(self) -> Path:
        return self._workspace_dir / sentinel_filename(self._name, self._agent_id)

    @property
    def response_path(self) -> Path:
        return self._workspace_dir / response_filename(self._name, self._agent_id)

    def stop(self) -> None:
        """Signal the watcher to exit on its next poll."""
        self._stop_event.set()

    def fired(self) -> bool:
        """True iff the sentinel was observed and a response was written."""
        return self._fired.is_set()

    def error(self) -> Optional[BaseException]:
        """Last error raised by the callback (if any)."""
        return self._error

    def _read_sentinel(self) -> Optional[dict[str, Any]]:
        try:
            st = self.sentinel_path.stat()
        except FileNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover - extremely defensive
            log.warning("barrier %s: failed to stat sentinel %s: %s", self._name, self.sentinel_path, exc)
            return {}
        if st.st_size > MAX_SENTINEL_BYTES:
            log.warning(
                "barrier %s: sentinel %s exceeds max size (%d > %d bytes), treating as empty payload",
                self._name,
                self.sentinel_path,
                st.st_size,
                MAX_SENTINEL_BYTES,
            )
            return {}
        try:
            raw = self.sentinel_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover - extremely defensive
            log.warning("barrier %s: failed to read sentinel %s: %s", self._name, self.sentinel_path, exc)
            return {}
        raw_size = len(raw.encode("utf-8"))
        if raw_size > MAX_SENTINEL_BYTES:
            log.warning(
                "barrier %s: sentinel %s exceeded max size after read (%d > %d bytes), treating as empty payload",
                self._name,
                self.sentinel_path,
                raw_size,
                MAX_SENTINEL_BYTES,
            )
            return {}
        raw = raw.strip()
        if not raw:
            # File exists but is empty — typically because the writer
            # (`Path.write_text` or similar) created the file but hasn't
            # written content yet. Treat as "not ready"; the watcher
            # keeps polling. A legitimate empty-payload writer must
            # write ``"{}"`` (which strips non-empty and parses to ``{}``).
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("barrier %s: sentinel JSON invalid (%d chars), treating as empty payload", self._name, len(raw))
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    def _write_response(self, response: dict[str, Any]) -> None:
        # Atomic write: tmp file + rename, so the container never reads a
        # partially-written response.
        tmp = self.response_path.with_suffix(self.response_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(response, ensure_ascii=True), encoding="utf-8")
        tmp.replace(self.response_path)

    def run(self) -> None:
        # Clean up only stale RESPONSE leftovers from prior runs. We do NOT
        # delete a pre-existing sentinel here: in the host<->container race
        # where the container (or a fast test stub) writes the sentinel
        # before this thread starts polling, deleting it would silently
        # discard the legitimate trigger. Instead we accept the sentinel
        # as-is on the very first poll iteration.
        try:
            self.response_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

        # Each round gets its own fresh wait budget — reset on entry to the
        # outer loop, not just once at thread start. For a non-persistent
        # watcher the outer loop only ever runs once.
        while not self._stop_event.is_set():
            deadline = (time.monotonic() + self._timeout) if self._timeout is not None else None
            while not self._stop_event.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    log.info("barrier %s/%s: timed out waiting for sentinel", self._name, self._agent_id)
                    return
                payload = self._read_sentinel()
                if payload is None:
                    # Sentinel not yet present; sleep and retry.
                    self._stop_event.wait(timeout=self._poll_interval)
                    continue
                break
            else:
                # stop_event was set while waiting for a sentinel.
                return
            event = BarrierEvent(
                sentinel_path=self.sentinel_path,
                response_path=self.response_path,
                payload=payload,
            )
            try:
                response = self._callback(event)
                if not isinstance(response, dict):
                    response = {"value": response}
            except Exception as exc:
                log.exception("barrier %s/%s: callback raised", self._name, self._agent_id)
                self._error = exc
                response = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                self._write_response(response)
            except Exception as exc:
                log.exception("barrier %s/%s: failed to write response", self._name, self._agent_id)
                self._error = exc
                return
            # Best-effort: remove the consumed sentinel so a follow-up
            # invocation doesn't see the same trigger twice.
            try:
                self.sentinel_path.unlink()
            except Exception as exc:
                log.warning(
                    "barrier %s/%s: failed to unlink consumed sentinel %s: %s — a persistent "
                    "watcher may reprocess the same payload on its next poll iteration",
                    self._name,
                    self._agent_id,
                    self.sentinel_path,
                    exc,
                )
            self._fired.set()
            if not self._persistent:
                return


__all__ = [
    "DEFAULT_POLL_INTERVAL_SEC",
    "MAX_SENTINEL_BYTES",
    "BarrierEvent",
    "BarrierWatcher",
    "sentinel_filename",
    "response_filename",
]
