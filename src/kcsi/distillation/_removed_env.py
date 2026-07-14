"""Hard-error guard for distillation env vars removed when the alternative
channels (fold / ledger / motifs) were deleted.

Call ``assert_no_removed_channel_env()`` at run start (see
``GenerationalOrchestrator.run``) so a removed value fails the run loudly
*before* any work begins — the in-``distill()`` call alone is swallowed by the
distill phase's ``except Exception`` and would silently skip distillation."""

from __future__ import annotations

import os

# var -> surviving default value. Any other non-empty value is rejected; the
# removed values these vars used to accept were fold / ledger / motifs.
_REMOVED = {
    "KCSI_DISTILL_STRATEGY": "window",
    "KCSI_PER_TASK_CHANNEL": "bundle",
    "KCSI_CROSS_TASK_CHANNEL": "bundle",
    "SWARMS_DISTILL_STRATEGY": "window",
    "SWARMS_PER_TASK_CHANNEL": "bundle",
    "SWARMS_CROSS_TASK_CHANNEL": "bundle",
}


def assert_no_removed_channel_env() -> None:
    """Raise if any removed distillation channel/strategy env var is set to a
    non-default value. Unset, or set to the surviving default, is a no-op."""
    for var, default in _REMOVED.items():
        raw = (os.environ.get(var) or "").strip().lower()
        if not raw or raw == default:
            continue
        raise RuntimeError(
            f"{var}={raw!r} is no longer supported: the alternative distillation "
            f"channels (fold / ledger / motifs) were removed. The only supported "
            f"behavior is the window-bundle default; unset {var} (or set it to "
            f"{default!r})."
        )
