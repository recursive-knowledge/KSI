"""Regression coverage for ``ksi.errors.is_auth_error``.

Incident that motivated the numeric-marker tightening (2026-04-21):
during the Haiku baseline sweep, a SWE-bench Pro task id of the form
``instance_internetarchive__openlibrary-bb152d23c004f3d68986877143bb0f83531fe401-...``
tripped the bare ``"401"`` substring check on a Silent-agent-runner error
message that quoted the task id verbatim. The engine re-raised a false
``AuthenticationFailure`` that aborted the whole SWE-bench Pro run before
Polyglot could start. Fix: require word-boundary match for numeric HTTP
status markers (401 / 403) while keeping named auth substrings as before.
"""

from __future__ import annotations

from ksi.errors import AuthenticationFailure, ContainerRegistryError, is_auth_error


# ── Real auth failures the classifier MUST still catch ───────────────────
def test_matches_authenticationerror_word():
    assert is_auth_error(Exception("AuthenticationError: invalid x-api-key"))


def test_matches_unauthorized_phrase():
    assert is_auth_error(Exception("401 Unauthorized: token expired"))


def test_matches_word_boundary_401():
    assert is_auth_error(Exception("HTTP status 401"))
    assert is_auth_error(Exception("request failed: 401"))
    assert is_auth_error(Exception("code=401 message=x"))


def test_matches_word_boundary_403():
    assert is_auth_error(Exception("HTTP 403 Forbidden"))


def test_matches_invalid_api_key():
    assert is_auth_error(Exception("invalid_api_key"))
    assert is_auth_error(Exception("Invalid API Key supplied"))


# ── Task-id substrings that MUST NOT be mistaken for auth failures ───────
def test_does_not_match_401_inside_git_hash():
    # The exact error text that triggered the false positive in the
    # 2026-04-21 Haiku sweep.
    msg = (
        "Silent agent-runner failure for task "
        "instance_internetarchive__openlibrary-"
        "bb152d23c004f3d68986877143bb0f83531fe401-"
        "ve8c8d62a2b60610a3c4631f5f23ed866bada9818 "
        "(agent=agent-42, generation=5): 0 tokens..."
    )
    assert not is_auth_error(Exception(msg))


def test_does_not_match_401_leading_in_hash():
    assert not is_auth_error(Exception("commit 4010abcdef passes"))


def test_does_not_match_401_embedded_in_hash():
    assert not is_auth_error(Exception("commit fe401ab still broken"))


def test_does_not_match_403_embedded_in_hash():
    assert not is_auth_error(Exception("commit 4030fff lives on"))


# ── Defensive: empty / weird inputs ──────────────────────────────────────
def test_empty_message():
    assert not is_auth_error(Exception(""))


def test_none_exc():
    # BaseException subclass with empty str — stripped empty => False
    class _E(Exception):
        def __str__(self) -> str:
            return ""

    assert not is_auth_error(_E())


def test_authentication_failure_is_runtime_error():
    assert issubclass(AuthenticationFailure, RuntimeError)


# --- Registry-vs-provider auth classification -------------------------------


def test_container_registry_error_is_not_a_provider_auth_error() -> None:
    """A registry 401 must never read as an LLM credential failure."""
    exc = ContainerRegistryError(
        "KSI_TB2_REQUIRE_PULL=1 and pull of 'alexgshaw/winning-avg-corewars:20251031' "
        "failed for task 'winning-avg-corewars'; refusing to fall back to local build "
        "for fairness mode. pull stderr: Error response from daemon: "
        "unauthorized: authentication required"
    )
    assert is_auth_error(exc) is False


def test_container_registry_error_rejected_by_type_not_wording() -> None:
    """The guard is type-based: no registry wording can reach the auth path."""
    for message in (
        "unauthorized: authentication required",
        "401 Unauthorized",
        "403 Forbidden",
        "invalid api key",
    ):
        assert is_auth_error(ContainerRegistryError(message)) is False


def test_real_provider_auth_error_still_fatal() -> None:
    """The fix must not blunt genuine provider-credential detection."""
    assert is_auth_error(RuntimeError("unauthorized: invalid x-api-key")) is True
    assert is_auth_error(RuntimeError("AuthenticationError: 401")) is True


def test_container_registry_retry_policy_is_explicit() -> None:
    """Registry provenance alone must not grant an outer task retry."""
    from ksi.orchestrator.task_retry import _is_retryable_task_error

    transient = ContainerRegistryError(
        "unauthorized: authentication required",
        retryable=True,
        reason="transient",
        image="alexgshaw/x:20251031",
    )
    permanent = ContainerRegistryError("manifest unknown", reason="non_transient", image="alexgshaw/missing:tag")

    assert _is_retryable_task_error(transient) is True
    assert _is_retryable_task_error(permanent) is False
    assert transient.reason == "transient"
    assert transient.image == "alexgshaw/x:20251031"


def test_wrapped_container_registry_error_preserves_typed_policy() -> None:
    """Auth-like wrapper text cannot erase registry provenance or retryability."""
    from ksi.orchestrator.task_retry import _is_retryable_task_error
    from ksi.runtime.normalize import build_error_runtime_meta

    registry_error = ContainerRegistryError(
        "unauthorized: authentication required",
        retryable=True,
        reason="transient",
    )
    wrapper = RuntimeError("wrapped unauthorized registry failure")
    wrapper.__cause__ = registry_error

    assert is_auth_error(wrapper) is False
    assert _is_retryable_task_error(wrapper) is True
    assert build_error_runtime_meta(wrapper) == {
        "status": "error",
        "error": "wrapped unauthorized registry failure",
        "error_type": "RuntimeError",
        "error_origin": "container_registry",
        "registry_failure_reason": "transient",
        "registry_failure_retryable": True,
    }


def test_nested_provider_auth_error_still_detected() -> None:
    provider_error = RuntimeError("AuthenticationError: invalid x-api-key")
    wrapper = RuntimeError("provider request failed")
    wrapper.__cause__ = provider_error

    assert is_auth_error(wrapper) is True


def test_exception_chain_cycle_is_bounded() -> None:
    from ksi.orchestrator.task_retry import _is_retryable_task_error

    registry_error = ContainerRegistryError("manifest unknown")
    wrapper = RuntimeError("wrapped unauthorized registry failure")
    wrapper.__cause__ = registry_error
    registry_error.__cause__ = wrapper

    assert is_auth_error(wrapper) is False
    assert _is_retryable_task_error(wrapper) is False
