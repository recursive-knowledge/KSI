"""Seam tests for GenerationalOrchestrator.__init__ extraction (#741, seam 2).

Behavior-preserving extraction of ``_resolve_experiment_name`` and
``_initialize_stores``. The existence test pins the seam; the behavioral test
exercises ``_resolve_experiment_name`` end-to-end via the non-resume name
collision (the ``_2`` suffix), which only fires when the probe store and the
real store cooperate over the same knowledge DB.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import MagicMock, patch

import pytest

from ksi.models import GenerationConfig
from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence
from ksi.tokens import LLMResponse, TokenUsage


def _make_orch(tmp_path, experiment_name: str) -> GenerationalOrchestrator:
    db_path = str(tmp_path / "knowledge.sqlite")
    runtime = MagicMock()
    evaluator = MagicMock()
    llm = MagicMock()
    llm.call.return_value = LLMResponse(
        text=json.dumps(
            {
                "transferable_insights": [],
                "pitfalls": [],
                "checks": [],
                "evidence_post_ids": [],
            }
        ),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=db_path,
        experiment_name=experiment_name,
    )
    return GenerationalOrchestrator(
        config=config,
        runtime=runtime,
        evaluator=evaluator,
        llm=llm,
        persistence=NoopPersistence(),
    )


def test_init_seam_methods_exist():
    assert callable(getattr(GenerationalOrchestrator, "_resolve_experiment_name", None))
    assert callable(getattr(GenerationalOrchestrator, "_initialize_stores", None))


def test_non_resume_collision_resolves_experiment_name(tmp_path):
    """A second orchestrator on the same DB + name (non-resume) gets a ``_2`` suffix."""
    first = _make_orch(tmp_path, "expt")
    assert first.config.experiment_name == "expt"

    second = _make_orch(tmp_path, "expt")
    assert second.config.experiment_name == "expt_2"


def test_claim_experiment_sequential_double_claim_suffixes(tmp_path):
    """Two sequential claims of the same base against one DB get distinct names.

    Proves the atomic-claim path (``KnowledgeStore.claim_experiment``) that the
    non-resume probe now uses: the second claimant is deterministically
    suffixed, so a concurrent same-name launch can never collide on the
    ``runs.experiment`` UNIQUE constraint.
    """
    from ksi.memory.knowledge_store import KnowledgeStore

    store = KnowledgeStore(str(tmp_path / "knowledge.sqlite"), default_experiment="expt")
    try:
        assert store.claim_experiment("expt") == "expt"
        assert store.claim_experiment("expt") == "expt_2"
        assert store.claim_experiment("expt") == "expt_3"
    finally:
        store.close()


def test_claim_experiment_resume_returns_original_name(tmp_path):
    """``resume=True`` returns the requested name without suffixing, even twice."""
    from ksi.memory.knowledge_store import KnowledgeStore

    store = KnowledgeStore(str(tmp_path / "knowledge.sqlite"), default_experiment="expt")
    try:
        assert store.claim_experiment("expt", resume=True) == "expt"
        # A repeat resume claim still reuses the same row/name (no _2 suffix).
        assert store.claim_experiment("expt", resume=True) == "expt"
        assert store.has_experiment("expt")
    finally:
        store.close()


def test_claim_experiment_two_store_contention_suffixes(tmp_path):
    """Two independent store handles racing the same DB claim distinct names."""
    from ksi.memory.knowledge_store import KnowledgeStore

    db_path = str(tmp_path / "knowledge.sqlite")
    barrier = Barrier(2)

    def claim_once() -> str:
        store = KnowledgeStore(db_path, default_experiment="expt")
        try:
            barrier.wait(timeout=5)
            return store.claim_experiment("expt")
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        names = sorted(f.result(timeout=5) for f in [pool.submit(claim_once), pool.submit(claim_once)])

    assert names == ["expt", "expt_2"]


def test_failed_post_claim_initialization_releases_empty_reservation(tmp_path):
    """A later init failure should not leave an empty name claim behind."""
    from ksi.memory.knowledge_store import KnowledgeStore

    db_path = str(tmp_path / "knowledge.sqlite")
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=db_path,
        experiment_name="failed_init",
    )
    with (
        patch.object(
            GenerationalOrchestrator,
            "_initialize_stores",
            side_effect=RuntimeError("later init boom"),
        ),
        pytest.raises(RuntimeError, match="later init boom"),
    ):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
            persistence=NoopPersistence(),
        )

    store = KnowledgeStore(db_path, default_experiment="failed_init")
    try:
        assert not store.has_experiment("failed_init")
        assert store.claim_experiment("failed_init") == "failed_init"
    finally:
        store.close()


def test_real_store_open_failure_is_fatal(tmp_path):
    """A real-store open failure in ``_initialize_stores`` stays fatal (#741 seam 2).

    ``test_knowledge_store_init_failure_is_fatal`` patches ``KnowledgeStore``
    globally and so only exercises the FIRST open (the probe in
    ``_resolve_experiment_name``). The seam split created a SECOND fatal
    ``try/except`` for the real-store open; pin it independently by letting the
    probe resolve normally and failing only the real open.
    """
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        knowledge_db_path=str(tmp_path / "knowledge.sqlite"),
        experiment_name="real_fail",
    )
    # Bypass the probe so the only KnowledgeStore() construction is the real
    # one inside _initialize_stores; make that raise.
    with (
        patch.object(
            GenerationalOrchestrator,
            "_resolve_experiment_name",
            lambda self, config, knowledge_db_path, exp_name: exp_name,
        ),
        patch(
            "ksi.memory.knowledge_store.KnowledgeStore.__init__",
            side_effect=RuntimeError("real boom"),
        ),
        pytest.raises(RuntimeError, match="KnowledgeStore initialization failed"),
    ):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
            persistence=NoopPersistence(),
        )
