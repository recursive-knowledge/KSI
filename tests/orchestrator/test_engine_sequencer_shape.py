"""Consolidated guard locking in the engine -> phase-service decoupling (#912).

After the claim / enrichment / eval / resume phase extractions, ``engine.py``
is meant to read as a thin sequencer: each phase lives in its own
``Engine*PhaseService`` module and may reference ``engine`` only inside its
single ``_collaborators`` factory. This file consolidates three regression
guards so a future change can't silently re-couple a phase body to the engine,
regrow the god-file, or quietly revert one of the extractions.
"""

from __future__ import annotations

import pytest
from conftest import REPO_ROOT

from kcsi.orchestrator import (
    claim_phase,
    distillation_phase,
    enrichment_phase,
    execution_phase,
    forum_phase,
    resume_phase,
    seeding_phase,
)
from tests.orchestrator_phase_decoupling_guard import functions_referencing_engine

ENGINE = REPO_ROOT / "src" / "kcsi" / "orchestrator" / "engine.py"


@pytest.mark.parametrize(
    "mod",
    [
        claim_phase,
        enrichment_phase,
        resume_phase,
        execution_phase,
        forum_phase,
        distillation_phase,
        seeding_phase,
    ],
)
def test_phase_module_only_touches_engine_in_collaborators(mod):
    """Every phase body must depend on its ``*Collaborators``, never ``engine``.

    Only the ``_collaborators`` factory may reach through to ``engine`` (an
    empty offender set is fine — a module with no engine reference at all still
    satisfies the subset).
    """
    leaked = functions_referencing_engine(mod.__file__) - {"_collaborators"}
    assert not leaked, f"{mod.__name__} references engine outside _collaborators: {leaked}"


def test_guard_catches_self_engine_recoupling(tmp_path):
    """Non-vacuity: the guard must flag a direct ``self.engine._x`` re-coupling
    (no local alias), not just a bare ``engine`` Name — the blind spot the
    hardened guard closes. A body using ``self.collab`` must stay clean."""
    src = (
        "class S:\n"
        "    def _collaborators(self):\n"
        "        engine = self.engine\n"
        "        return engine._best_scores\n"
        "    def leak_via_self(self):\n"
        "        return self.engine._best_scores\n"
        "    def clean(self):\n"
        "        return self.collab.scores\n"
    )
    p = tmp_path / "fake_phase.py"
    p.write_text(src, encoding="utf-8")
    offenders = functions_referencing_engine(str(p))
    assert "leak_via_self" in offenders  # self.engine.* is caught
    assert "_collaborators" in offenders  # the legitimate factory access
    assert "clean" not in offenders  # self.collab.* is not a false positive


def test_engine_stays_a_sequencer():
    """Guard against the god-file regrowing after the phase extractions."""
    loc = ENGINE.read_text().count("\n")
    # Engine is ~2101 LOC (was ~2091; bumped +10 for the #1257 merged
    # reflection+lessons wiring — one merged knowledge-generation call replacing
    # the two-call insight/lesson path). This is a regression guard against the
    # god-file regrowing — NOT a hard target; a method extraction or deliberate
    # feature adds a small, fixed amount of scaffolding, so the ceiling is bumped
    # alongside such changes.
    assert loc <= 2105, f"engine.py grew to {loc} LOC; extract new logic into a phase module"


def test_extracted_methods_removed_from_engine():
    """Prove the big phase methods moved out (catches an accidental revert)."""
    engine_src = ENGINE.read_text()
    for gone in (
        "def _eval_one_attempt",
        "def _enrich_seed_packages",
        "def _deterministic_claim_phase",
        "def _split_carried_forward_assignments",
        "def _persist_carried_forward_trace",
    ):
        assert gone not in engine_src, f"{gone} should have moved to a phase module"
