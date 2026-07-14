"""Tests for host-side distillation LLM retry (issue #1069).

Root cause of #1069: a ~1-hour host-side DNS outage
(``getaddrinfo EAI_AGAIN api.anthropic.com``) made every host-side
``distill_one_task`` LLM call fail once and give up — the forum path retries
(``max_task_retries``) but the distill path had *no* retry at all, so a whole
generation's distillation was zeroed with a single WARNING per task.

Two properties are pinned here:

1. ``is_retryable_distill_error`` classifies the *actual* incident exception
   as retryable. The Anthropic SDK's ``APIConnectionError`` stringifies to just
   ``"Connection error."`` — the transient signature (``getaddrinfo EAI_AGAIN``)
   lives in ``__cause__``, so a classifier that only inspects ``str(exc)``
   (like ``_is_retryable_task_error``) would MISS it. The distill classifier
   walks the exception chain and the exception *type* name.

2. ``run_with_distill_retry`` rides out a transient failure with bounded
   exponential backoff and never retries a deterministic (auth / bad-prompt)
   failure.
"""

from __future__ import annotations

import pytest

from ksi.errors import AuthenticationFailure
from ksi.orchestrator import task_retry

# --- Fakes that mimic the SDK exception shapes we actually see -------------


class APIConnectionError(Exception):
    """Mimics anthropic.APIConnectionError: terse ``str()``, real cause chained."""


class APITimeoutError(Exception):
    pass


def _incident_error() -> APIConnectionError:
    """The exact shape from the #1069 campaign log: terse message, DNS cause."""
    exc = APIConnectionError("Connection error.")
    exc.__cause__ = OSError("[Errno -3] getaddrinfo EAI_AGAIN api.anthropic.com")
    return exc


# --- Predicate: is_retryable_distill_error ---------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        _incident_error(),  # the actual incident
        APIConnectionError("Connection error."),  # terse, no cause — type-name signal
        APITimeoutError("Request timed out."),
        Exception("fetch failed"),
        Exception("overloaded_error: server overloaded"),
        Exception("503 Service Unavailable"),
    ],
)
def test_retryable_distill_errors(exc: Exception) -> None:
    assert task_retry.is_retryable_distill_error(exc) is True, repr(exc)


@pytest.mark.parametrize(
    "exc",
    [
        AuthenticationFailure("LLM authentication failed"),
        Exception("AuthenticationError: invalid_api_key"),
        Exception("invalid prompt: refused by usage policy"),
        Exception("something deterministic and unrelated"),
    ],
)
def test_non_retryable_distill_errors(exc: Exception) -> None:
    assert task_retry.is_retryable_distill_error(exc) is False, repr(exc)


# --- run_with_distill_retry ------------------------------------------------


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_retry, "_sleep", lambda _delay: None)


def test_retry_rides_out_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSI_DISTILL_MAX_RETRIES", "6")
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] <= 3:
            raise _incident_error()
        return "ok"

    result = task_retry.run_with_distill_retry(fn, generation=3, phase="distill_per_task")
    assert result == "ok"
    assert calls["n"] == 4  # 3 transient failures + 1 success


def test_retry_does_not_retry_deterministic_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSI_DISTILL_MAX_RETRIES", "6")
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise ValueError("invalid prompt: refused by usage policy")

    with pytest.raises(ValueError):
        task_retry.run_with_distill_retry(fn, generation=3, phase="distill_per_task")
    assert calls["n"] == 1  # gave up immediately — no wasted retries


def test_retry_exhausts_and_raises_last(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSI_DISTILL_MAX_RETRIES", "2")
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise _incident_error()

    with pytest.raises(APIConnectionError):
        task_retry.run_with_distill_retry(fn, generation=3, phase="distill_per_task")
    assert calls["n"] == 3  # 2 retries + 1 initial attempt


def test_retry_disabled_with_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSI_DISTILL_MAX_RETRIES", "0")
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise _incident_error()

    with pytest.raises(APIConnectionError):
        task_retry.run_with_distill_retry(fn, generation=3, phase="distill_per_task")
    assert calls["n"] == 1  # single attempt, no retry


# --- Engine adapter integration: the distill LLM chokepoint retries --------


def test_make_distill_llm_adapter_retries_transient(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The single ``_make_distill_llm`` chokepoint (covers per-task AND
    cross-task distill) rides out a transient ``self.llm.call`` failure and
    returns the eventual result — instead of failing the whole generation."""
    import json

    from ksi.tokens import LLMResponse, TokenUsage
    from tests.test_distill_phase import _make_orch

    monkeypatch.setenv("KSI_DISTILL_MAX_RETRIES", "6")
    orch = _make_orch(tmp_path)

    good = LLMResponse(
        text=json.dumps({"transferable_insights": [], "pitfalls": []}),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    orch.llm.call.side_effect = [_incident_error(), _incident_error(), good]

    adapter = orch._make_distill_llm(generation=3, phase="distill_per_task")
    result = adapter("sys", "user")

    assert "transferable_insights" in str(result)
    assert orch.llm.call.call_count == 3  # 2 transient failures + 1 success
