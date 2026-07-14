"""Loud escalation when a generation's distillation is fully zeroed (#1069).

The forensic incident behind #1069: a sustained host->provider network outage
made distillation produce ZERO knowledge for 7 consecutive generations, while
attempts (running in containers) kept succeeding — so 7 generations of attempt
compute were spent for no learning, surfaced only as per-task WARNING lines.

Retry (see ``test_distill_retry``) rides out short blips; this escalation is
the backstop for a *sustained* outage retry cannot fix: when a generation's
distill is fully zeroed AND there were sub-failures, the engine logs at ERROR
and tracks how many consecutive generations have been zeroed, so an operator
notices instead of the run silently burning compute.
"""

from __future__ import annotations

import pytest

from kcsi.distillation import DistillOutput
from kcsi.distillation.types import CrossTaskBundle, PerTaskBundle
from kcsi.orchestrator.distillation_phase import (
    DistillationPhaseInput,
    EngineDistillationPhaseService,
)
from tests.test_distill_phase import _make_orch


def _run(orch, generation, task_ids):
    EngineDistillationPhaseService(orch).run(DistillationPhaseInput(generation=generation, task_ids=task_ids))


def _zeroed_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
    # Attempted work, but produced nothing and recorded sub-failures — the
    # signature of a transient/sustained outage, not a healthy empty result.
    return DistillOutput(per_task={}, cross_task=None, failures=3)


def _healthy_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
    return DistillOutput(
        per_task={
            "t1": PerTaskBundle(
                task_id="t1",
                transferable_insights=["x"],
                pitfalls=[],
                checks=[],
                evidence_post_ids=[],
            )
        },
        cross_task=None,
        failures=0,
    )


def _cross_bundle(marker: str = "cross") -> CrossTaskBundle:
    return CrossTaskBundle(
        transferable_insights=[marker],
        pitfalls=[],
        checks=[],
        evidence_post_ids=[],
    )


def test_fully_zeroed_generation_logs_error_and_counts(tmp_path, monkeypatch, caplog):
    orch = _make_orch(tmp_path)
    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", _zeroed_distill)

    with caplog.at_level("ERROR"):
        _run(orch, generation=3, task_ids=["t1"])

    assert orch._consecutive_zeroed_distill_generations == 1
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("zero" in r.message.lower() for r in errors), [r.message for r in errors]


def test_consecutive_zeroed_generations_escalate(tmp_path, monkeypatch, caplog):
    orch = _make_orch(tmp_path)
    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", _zeroed_distill)

    _run(orch, generation=3, task_ids=["t1"])
    _run(orch, generation=4, task_ids=["t1"])
    with caplog.at_level("ERROR"):
        _run(orch, generation=5, task_ids=["t1"])

    assert orch._consecutive_zeroed_distill_generations == 3
    # The escalating message names the consecutive count.
    assert any("3 consecutive" in r.message for r in caplog.records), [r.message for r in caplog.records]


def test_healthy_generation_resets_counter(tmp_path, monkeypatch):
    orch = _make_orch(tmp_path)
    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", _zeroed_distill)
    _run(orch, generation=3, task_ids=["t1"])
    _run(orch, generation=4, task_ids=["t1"])
    assert orch._consecutive_zeroed_distill_generations == 2

    monkeypatch.setattr(dist_pkg, "distill", _healthy_distill)
    _run(orch, generation=5, task_ids=["t1"])
    assert orch._consecutive_zeroed_distill_generations == 0


def test_distill_raising_entirely_counts_as_zeroed(tmp_path, monkeypatch, caplog):
    orch = _make_orch(tmp_path)
    import kcsi.distillation as dist_pkg

    def _boom(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        raise RuntimeError("connection error")

    monkeypatch.setattr(dist_pkg, "distill", _boom)
    with caplog.at_level("ERROR"):
        _run(orch, generation=3, task_ids=["t1"])

    assert orch._consecutive_zeroed_distill_generations == 1
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_distillation_persistence_failure_counts_as_zeroed(tmp_path, monkeypatch, caplog):
    orch = _make_orch(tmp_path)
    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", _healthy_distill)

    def fail_record_distillation(**_kwargs):
        raise RuntimeError("sqlite write failed")

    monkeypatch.setattr(orch._knowledge, "record_distillation", fail_record_distillation)

    with caplog.at_level("ERROR"):
        _run(orch, generation=3, task_ids=["t1"])

    assert orch._consecutive_zeroed_distill_generations == 1
    assert any("1 sub-failure" in r.message for r in caplog.records), [r.message for r in caplog.records]


@pytest.mark.parametrize(
    ("distill_output", "expected_failures"),
    [
        (DistillOutput(per_task={}, cross_task=None, cross_task_by_task={"t1": _cross_bundle("ct-t1")}), 1),
        (DistillOutput(per_task={}, cross_task=_cross_bundle("broadcast")), 1),
        (
            DistillOutput(
                per_task={
                    "t1": PerTaskBundle(
                        task_id="t1",
                        transferable_insights=["x"],
                        pitfalls=[],
                        checks=[],
                        evidence_post_ids=[],
                    )
                },
                cross_task=None,
                cross_task_by_task={"t1": _cross_bundle("ct-t1")},
            ),
            2,
        ),
    ],
    ids=["target-conditioned-cross", "broadcast-cross", "per-task-and-cross"],
)
def test_distillation_persistence_failures_count_all_bundle_paths(
    tmp_path,
    monkeypatch,
    caplog,
    distill_output,
    expected_failures,
):
    orch = _make_orch(tmp_path)
    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", lambda inp, **_kwargs: distill_output)

    def fail_record_distillation(**_kwargs):
        raise RuntimeError("sqlite write failed")

    monkeypatch.setattr(orch._knowledge, "record_distillation", fail_record_distillation)

    with caplog.at_level("ERROR"):
        _run(orch, generation=3, task_ids=["t1"])

    assert orch._consecutive_zeroed_distill_generations == 1
    assert any(f"{expected_failures} sub-failure" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]


def test_empty_task_ids_does_not_count_as_zeroed(tmp_path, monkeypatch):
    orch = _make_orch(tmp_path)
    import kcsi.distillation as dist_pkg

    # Prime a nonzero counter, then a legitimate skip (no task ids) must not
    # increment it (nothing was attempted — not an outage).
    monkeypatch.setattr(dist_pkg, "distill", _zeroed_distill)
    _run(orch, generation=3, task_ids=["t1"])
    assert orch._consecutive_zeroed_distill_generations == 1

    _run(orch, generation=4, task_ids=[])
    assert orch._consecutive_zeroed_distill_generations == 1


# --- Opt-in hard abort after N consecutive zeroed generations --------------


def test_abort_disabled_by_default_never_raises(tmp_path, monkeypatch):
    orch = _make_orch(tmp_path)
    assert orch.config.abort_on_distill_stall == 0  # default: disabled
    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", _zeroed_distill)
    # Many consecutive zeroed generations, but abort is off — no raise.
    for gen in range(1, 6):
        _run(orch, generation=gen, task_ids=["t1"])
    assert orch._consecutive_zeroed_distill_generations == 5


def test_abort_raises_at_threshold(tmp_path, monkeypatch):
    from kcsi.errors import DistillationStalledError

    orch = _make_orch(tmp_path)
    orch.config.abort_on_distill_stall = 3
    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", _zeroed_distill)

    _run(orch, generation=1, task_ids=["t1"])  # streak 1 — below threshold
    _run(orch, generation=2, task_ids=["t1"])  # streak 2 — below threshold
    with pytest.raises(DistillationStalledError):
        _run(orch, generation=3, task_ids=["t1"])  # streak 3 — aborts


def test_abort_streak_must_be_consecutive(tmp_path, monkeypatch):
    from kcsi.errors import DistillationStalledError

    orch = _make_orch(tmp_path)
    orch.config.abort_on_distill_stall = 2
    import kcsi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", _zeroed_distill)
    _run(orch, generation=1, task_ids=["t1"])  # streak 1
    monkeypatch.setattr(dist_pkg, "distill", _healthy_distill)
    _run(orch, generation=2, task_ids=["t1"])  # resets streak to 0
    monkeypatch.setattr(dist_pkg, "distill", _zeroed_distill)
    _run(orch, generation=3, task_ids=["t1"])  # streak 1 again — no abort
    with pytest.raises(DistillationStalledError):
        _run(orch, generation=4, task_ids=["t1"])  # streak 2 — aborts
