"""The distillation phase must re-raise ``AuthenticationFailure``.

The distiller (`ksi.distillation.distiller`) deliberately re-raises
``AuthenticationFailure`` rather than swallowing it. The phase wrapper used to
catch it in its broad ``except Exception`` handler, log a WARNING, and return —
silently disabling knowledge improvement for the rest of a campaign at full
token cost. The phase must let auth failures propagate so the run aborts loudly.
"""

from __future__ import annotations

import pytest

from ksi.errors import AuthenticationFailure
from ksi.orchestrator.distillation_phase import (
    DistillationPhaseInput,
    EngineDistillationPhaseService,
)
from tests.test_distill_phase import _make_orch


def test_distill_auth_failure_propagates(tmp_path, monkeypatch):
    """If ``distill`` raises ``AuthenticationFailure``, the phase re-raises it
    instead of swallowing it into a logged warning."""
    orch = _make_orch(tmp_path)

    def fake_distill(inp, *, unsolved_task_ids=None, newly_solved_task_ids=None):
        raise AuthenticationFailure("401 invalid api key")

    import ksi.distillation as dist_pkg

    monkeypatch.setattr(dist_pkg, "distill", fake_distill)

    with pytest.raises(AuthenticationFailure):
        EngineDistillationPhaseService(orch).run(DistillationPhaseInput(generation=1, task_ids=["t1"]))
