"""Parity + behaviour tests for the shared retryable-error marker source.

Issue #648: the retryable-error markers used by ``src/ksi/orchestrator/engine.py``
to classify a task error as transient (retry) vs non-retryable (abort) live in a
single source of truth shared with the TypeScript agent-runner:
``runtime_runner/shared/retryable_markers.json``. These tests pin:

  1. the JSON parses and exposes the expected categories;
  2. the in-module vendored fallback in ``src/ksi/errors.py`` stays byte-identical
     to the JSON (so the "never crash" fallback can't silently drift);
  3. the actual classification function ``_is_retryable_task_error`` produces the
     historical outcome for a representative error string from every category —
     this PR is a pure refactor with NO behaviour change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ksi.errors import (
    _RETRYABLE_MARKERS_PATH,
    _VENDORED_RETRYABLE_MARKERS,
    load_retryable_markers,
    non_retryable_exit_code_markers,
    non_retryable_markers,
    transient_markers,
    upstream_provider_transient_markers,
)
from ksi.orchestrator.engine import (
    _NON_RETRYABLE_EXIT_CODES,
    _NON_RETRYABLE_TASK_ERROR_MARKERS,
    _TRANSIENT_TASK_ERROR_MARKERS,
    _UPSTREAM_PROVIDER_TRANSIENT_MARKERS,
    _is_retryable_task_error,
)

EXPECTED_CATEGORIES = {
    "non_retryable",
    "non_retryable_exit_codes",
    "upstream_provider_transient",
    "transient_extra",
    "stream_race",
}

# Golden snapshot of the marker values as they were HARDCODED in engine.py
# before the #648 single-source refactor. This is an independent anchor: it is
# NOT derived from the JSON or the vendored copy, so it catches a value that
# silently drifts during the move (the self-consistency tests above would all
# stay green if the JSON and vendored copy were edited together but wrongly).
# Do not "fix" this to match a reworded JSON without a deliberate decision —
# rewording a marker changes retry classification in production.
_HISTORICAL_MARKERS = {
    "non_retryable": (
        "invalid prompt",
        "usage policy",
        "flagged as potentially violating",
        "no patch",
        "missing report",
        "parse_error",
    ),
    "non_retryable_exit_codes": ("exit=137", "exit=139", "exit=126", "exit=127"),
    "upstream_provider_transient": (
        "429",
        "502",
        "503",
        "504",
        "rate limit",
        "too many requests",
        "service unavailable",
        "provider unavailable",
        "internal server error",
        "gateway timeout",
        "bad gateway",
        "upstream connect",
        "connection termination",
        "overloaded",
        "fetch failed",
        "headers timeout",
        "headerstimeouterror",
    ),
    "transient_extra": (
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "broken pipe",
        "temporarily unavailable",
        "temporary failure",
        "network is unreachable",
        "econnreset",
        "eai_again",
        "resource temporarily unavailable",
    ),
    "stream_race": (
        "sdk query loop drained",
        "sdk query iterator threw",
        "sdk emitted an empty result event",
        "silent agent-runner failure",
    ),
}


def _raw_json() -> dict:
    with _RETRYABLE_MARKERS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_json_file_exists_and_parses():
    assert _RETRYABLE_MARKERS_PATH.exists(), _RETRYABLE_MARKERS_PATH
    data = _raw_json()
    assert data["schema_version"] == 1
    assert set(data["categories"]) == EXPECTED_CATEGORIES
    for name, markers in data["categories"].items():
        assert isinstance(markers, list) and markers, name
        assert all(isinstance(m, str) and m for m in markers), name
        # Markers are matched case-insensitively; the source must be lowercase
        # so the substring contract is unambiguous.
        assert all(m == m.lower() for m in markers), name


def test_loader_matches_raw_json():
    data = _raw_json()
    loaded = load_retryable_markers()
    assert set(loaded) == set(data["categories"])
    for name, markers in data["categories"].items():
        assert loaded[name] == tuple(markers), name


def test_vendored_fallback_matches_json():
    """The vendored fallback in errors.py must mirror the JSON exactly."""
    data = _raw_json()
    vendored = _VENDORED_RETRYABLE_MARKERS["categories"]
    assert set(vendored) == set(data["categories"])
    for name, markers in data["categories"].items():
        assert tuple(vendored[name]) == tuple(markers), (
            f"vendored fallback for '{name}' drifted from retryable_markers.json — update both"
        )


def test_loader_falls_back_when_file_missing(monkeypatch):
    """A missing/garbled JSON file must not crash; vendored copy is used."""
    import ksi.errors as errors_mod

    load_retryable_markers.cache_clear()
    monkeypatch.setattr(
        errors_mod,
        "_RETRYABLE_MARKERS_PATH",
        Path("/nonexistent/retryable_markers.json"),
    )
    try:
        loaded = load_retryable_markers()
        # Falls back to the vendored copy rather than raising.
        assert set(loaded) == EXPECTED_CATEGORIES
        assert loaded["stream_race"] == tuple(_VENDORED_RETRYABLE_MARKERS["categories"]["stream_race"])
    finally:
        load_retryable_markers.cache_clear()


def test_transient_markers_composition():
    """transient = upstream_provider_transient + transient_extra + stream_race."""
    markers = load_retryable_markers()
    assert transient_markers() == (
        markers["upstream_provider_transient"] + markers["transient_extra"] + markers["stream_race"]
    )


# --------------------------------------------------------------------------
# Behaviour parity: drive the real classifier with one sample per category.
# A plain Exception path is exercised here (the SilentAgentRuntimeError path
# has its own dedicated tests elsewhere); these pin the substring contract.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("marker", non_retryable_markers())
def test_non_retryable_markers_block_retry(marker):
    exc = RuntimeError(f"task failed: {marker} in the model output")
    assert _is_retryable_task_error(exc) is False


@pytest.mark.parametrize("marker", non_retryable_exit_code_markers())
def test_non_retryable_exit_codes_block_retry(marker):
    exc = RuntimeError(f"Shared container runner failed ({marker})")
    assert _is_retryable_task_error(exc) is False


@pytest.mark.parametrize("marker", upstream_provider_transient_markers())
def test_upstream_provider_markers_allow_retry(marker):
    exc = RuntimeError(f"provider error: {marker} from upstream")
    assert _is_retryable_task_error(exc) is True


@pytest.mark.parametrize("marker", transient_markers())
def test_all_transient_markers_allow_retry(marker):
    exc = RuntimeError(f"runner blip: {marker} occurred")
    assert _is_retryable_task_error(exc) is True


def test_stream_race_markers_present_in_transient():
    markers = load_retryable_markers()
    for phrase in markers["stream_race"]:
        assert phrase in transient_markers()
        # And the classifier treats a bare exception carrying it as retryable.
        assert _is_retryable_task_error(RuntimeError(phrase.upper())) is True


def test_empty_and_unknown_errors_are_non_retryable():
    assert _is_retryable_task_error(RuntimeError("")) is False
    assert _is_retryable_task_error(RuntimeError("some totally unrelated failure")) is False


def test_non_retryable_wins_over_transient_substring():
    """A message containing BOTH a non-retryable and a transient marker must
    classify as non-retryable (the non-retryable check runs first)."""
    # "no patch" (non-retryable) + "timeout" (transient) in one message.
    exc = RuntimeError("no patch produced after the request timed out")
    assert _is_retryable_task_error(exc) is False


# --------------------------------------------------------------------------
# No-behaviour-change anchors: pin the loaded markers to the pre-refactor
# hardcoded values, and pin engine.py's module-level tuples to the loader.
# --------------------------------------------------------------------------


def test_loaded_markers_match_pre_refactor_hardcoded_values():
    """The single-source markers must equal engine.py's old hardcoded tuples.

    This is the machine-check behind the PR's "pure refactor, no behaviour
    change" claim — independent of the JSON<->vendored self-consistency tests.
    """
    loaded = load_retryable_markers()
    assert loaded == _HISTORICAL_MARKERS


def test_engine_module_tuples_match_loader():
    """engine.py captures the markers at import time; verify it captured the
    loader's values verbatim so a future loader change can't silently diverge
    from what the classifier actually uses."""
    markers = load_retryable_markers()
    assert _NON_RETRYABLE_TASK_ERROR_MARKERS == markers["non_retryable"]
    assert _NON_RETRYABLE_EXIT_CODES == markers["non_retryable_exit_codes"]
    assert _UPSTREAM_PROVIDER_TRANSIENT_MARKERS == markers["upstream_provider_transient"]
    assert _TRANSIENT_TASK_ERROR_MARKERS == transient_markers()


# --------------------------------------------------------------------------
# Robustness: a JSON that parses but is structurally wrong must degrade to the
# vendored copy, never crash at import or silently load a partial marker set.
# (Issue #648 follow-up — the loader's "never crash" guarantee must hold for
# more than just the whole-file-missing case.)
# --------------------------------------------------------------------------


def _point_loader_at(monkeypatch, tmp_path, payload) -> dict[str, tuple[str, ...]]:
    """Write ``payload`` (str or json-serialisable) as the markers file, point
    the loader at it, and return the freshly loaded markers."""
    import ksi.errors as errors_mod

    p = tmp_path / "retryable_markers.json"
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    load_retryable_markers.cache_clear()
    monkeypatch.setattr(errors_mod, "_RETRYABLE_MARKERS_PATH", p)
    return load_retryable_markers()


def _vendored() -> dict[str, tuple[str, ...]]:
    return {name: tuple(markers) for name, markers in _VENDORED_RETRYABLE_MARKERS["categories"].items()}


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("{ not valid json", id="garbled"),
        pytest.param({"schema_version": 1}, id="no-categories-key"),
        pytest.param({"categories": {}}, id="empty-categories"),
        pytest.param({"categories": {"non_retryable": ["invalid prompt"]}}, id="missing-categories"),
        pytest.param(
            {
                "categories": {
                    "non_retryable": ["invalid prompt"],
                    "non_retryable_exit_codes": ["exit=137"],
                    "upstream_provider_transient": ["429"],
                    "transient_extra": [],  # empty list — would silently drop markers
                    "stream_race": ["silent agent-runner failure"],
                }
            },
            id="empty-category-list",
        ),
        pytest.param(
            {
                "categories": {
                    "non_retryable": ["invalid prompt"],
                    "non_retryable_exit_codes": ["exit=137"],
                    "upstream_provider_transient": ["429"],
                    "transient_extra": ["timeout", 123],  # non-string marker
                    "stream_race": ["silent agent-runner failure"],
                }
            },
            id="non-string-marker",
        ),
        pytest.param(
            {
                "categories": {
                    "non_retryable": ["invalid prompt"],
                    "non_retryable_exit_codes": ["exit=137"],
                    "upstream_provider_transient": ["429"],
                    "transient_extra": ["timeout"],
                    "stream_race": ["silent agent-runner failure"],
                    "surprise_extra_category": ["boom"],  # unexpected category
                }
            },
            id="extra-category",
        ),
    ],
)
def test_structurally_invalid_json_falls_back_to_vendored(monkeypatch, tmp_path, payload):
    """Any parseable-but-wrong-shape JSON must yield the vendored copy, not
    crash and not a partial marker set."""
    try:
        loaded = _point_loader_at(monkeypatch, tmp_path, payload)
        assert loaded == _vendored()
    finally:
        load_retryable_markers.cache_clear()


def test_valid_json_override_is_honoured(monkeypatch, tmp_path):
    """Sanity counterpart: a structurally-valid JSON IS loaded (so the
    validator isn't simply ignoring the file)."""
    custom = {name: list(markers) for name, markers in _HISTORICAL_MARKERS.items()}
    custom["non_retryable"] = ["custom refusal"]
    try:
        loaded = _point_loader_at(monkeypatch, tmp_path, {"categories": custom})
        assert loaded["non_retryable"] == ("custom refusal",)
    finally:
        load_retryable_markers.cache_clear()
