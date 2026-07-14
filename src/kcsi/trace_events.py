from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

_TRACE_LOCK = threading.RLock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_trace_dir() -> str:
    return str((os.environ.get("KCSI_TRACE_DIR", "") or "").strip())


def append_trace_event(
    trace_dir: str,
    filename: str,
    payload: dict[str, Any],
) -> None:
    if not trace_dir:
        return
    try:
        root = Path(trace_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        out = root / filename
        event = {"ts": _now_iso(), **payload}
        line = json.dumps(event, ensure_ascii=True) + "\n"
        with _TRACE_LOCK:
            with out.open("a", encoding="utf-8") as fh:
                if fcntl is not None:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                    except Exception:
                        pass
                fh.write(line)
                fh.flush()
                if fcntl is not None:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
    except Exception:
        # Tracing must never break the runtime path.
        return
