"""Advisory-flock timeout behavior in the shared store lock guard (#980).

The 60s flock-acquire timeout used to log a warning and proceed WITHOUT the
only cross-process lock. It now raises ``TimeoutError`` by default, with an
opt-in ``KSI_FLOCK_TIMEOUT_PROCEED=1`` escape hatch that restores the legacy
degrade-over-abort behavior.
"""

import logging
import threading

import pytest

from ksi.memory import _store_common
from ksi.memory._store_common import _flock_timeout_should_proceed, _locked_guard

fcntl = _store_common.fcntl


def _guard_args(lock_fd):
    return dict(
        lock_state=threading.local(),
        process_lock=threading.RLock(),
        thread_lock=threading.RLock(),
        lock_fd=lock_fd,
        logger=logging.getLogger("test.flock"),
    )


def _force_deadline_passed(monkeypatch):
    # First monotonic() call sets the deadline (=+60s); every later call is
    # far past it so the very first failed flock retry hits the timeout branch
    # without a real 60s wait.
    times = iter([0.0] + [1000.0] * 100)
    monkeypatch.setattr(_store_common.time, "monotonic", lambda: next(times))


def test_flock_timeout_should_proceed_env(monkeypatch):
    for val in ("1", "true", "TRUE", "yes", "on", " on "):
        monkeypatch.setenv("KSI_FLOCK_TIMEOUT_PROCEED", val)
        assert _flock_timeout_should_proceed() is True
    for val in ("0", "false", "", "no", "off", "garbage"):
        monkeypatch.setenv("KSI_FLOCK_TIMEOUT_PROCEED", val)
        assert _flock_timeout_should_proceed() is False
    monkeypatch.delenv("KSI_FLOCK_TIMEOUT_PROCEED", raising=False)
    assert _flock_timeout_should_proceed() is False


@pytest.mark.skipif(fcntl is None, reason="flock unavailable on this platform")
def test_flock_timeout_raises_by_default(tmp_path, monkeypatch):
    lock_path = tmp_path / "db.sqlite.lock"
    holder = open(lock_path, "w")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    contender = open(lock_path, "w")
    try:
        monkeypatch.delenv("KSI_FLOCK_TIMEOUT_PROCEED", raising=False)
        _force_deadline_passed(monkeypatch)
        with pytest.raises(TimeoutError, match="advisory flock not acquired"):
            with _locked_guard(**_guard_args(contender)):
                pass  # pragma: no cover — the guard raises before yielding
    finally:
        contender.close()
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


@pytest.mark.skipif(fcntl is None, reason="flock unavailable on this platform")
def test_flock_timeout_proceeds_with_env_optin(tmp_path, monkeypatch):
    lock_path = tmp_path / "db.sqlite.lock"
    holder = open(lock_path, "w")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    contender = open(lock_path, "w")
    try:
        monkeypatch.setenv("KSI_FLOCK_TIMEOUT_PROCEED", "1")
        _force_deadline_passed(monkeypatch)
        entered = False
        with _locked_guard(**_guard_args(contender)):
            entered = True
        assert entered  # opt-in path proceeds without the flock instead of raising
    finally:
        contender.close()
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()
