"""Tests for seed bundle injection and the new scoped bundle seeding API."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ksi.models import AgentState, GenerationConfig, TaskSpec
from ksi.orchestrator.engine import GenerationalOrchestrator
from ksi.seeding.seeder import PopulationSeeder
from ksi.tokens import LLMResponse, TokenAccumulator, TokenUsage
from tests.orchestrator_phase_helpers import load_cross_task_seed_bundle, prepare_resume_population

# ---------------------------------------------------------------------------
# External seed bundle (--seed-bundle-path) — gen-1 injection remains unchanged
# ---------------------------------------------------------------------------


def _make_bundle_file(tmp_path: Path) -> Path:
    bundle = {
        "meta": {
            "source_experiment": "test-donor",
            "source_db": "test.sqlite",
            "extracted_at": "2026-03-29T00:00:00Z",
            "extraction_mode": "r3_cluster",
            "generation_extracted": 10,
        },
        "assets": [
            {
                "asset_id": "asset-1",
                "text": "Tiling patterns repeat along grid axes.",
                "source_insight_ids": ["ins-10-agent-0-1"],
            },
            {
                "asset_id": "asset-2",
                "text": "Color permutations are often bijective.",
                "source_insight_ids": ["ins-10-agent-1-1"],
            },
        ],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return path


def test_gen1_agents_seeded_with_external_bundle(tmp_path: Path):
    """When seed_bundle_path is set, gen-1 agents get the bundle as seed_package."""
    bundle_file = _make_bundle_file(tmp_path)
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        seed_bundle_path=str(bundle_file),
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
    )
    assert len(orch.agents) == 1
    agent = orch.agents[0]
    seed = agent.seed_package
    assert isinstance(seed, dict)
    bundle = seed.get("insight_bundle") or seed.get("shared_insight_bundle")
    assert isinstance(bundle, list)
    assert len(bundle) == 2
    assert "Tiling patterns" in bundle[0]["text"]
    # Verify asset_id was mapped to id for the normalizer
    assert bundle[0].get("id") == "asset-1"


def _make_cross_task_bundle_file(tmp_path: Path) -> Path:
    bundle = {
        "meta": {
            "source_experiment": "test-donor-crosstask",
            "source_db": "donor.sqlite",
            "extracted_at": "2026-04-19T00:00:00Z",
            "generation_extracted": 10,
            "bundle_kind": "cross_task",
        },
        "cross_task": {
            "transferable_insights": [
                "Grids with bilateral symmetry often use axis-aligned tiles.",
                "Color-count invariants survive most transformations.",
            ],
            "confirmed_constraints": ["Symmetric tasks preserve the tile axis."],
            "rejected_hypotheses": ["Blind recoloring fails when colors encode position."],
            "pitfalls": ["Do not assume the output shape matches the input shape."],
            "checks": ["Verify pixel-count conservation before committing a grid."],
            "next_steps": ["Try axis detection before color mapping."],
            "evidence_post_ids": [42, 97],
        },
    }
    path = tmp_path / "cross_task_bundle.json"
    path.write_text(json.dumps(bundle))
    return path


def test_gen1_agents_seeded_with_cross_task_bundle(tmp_path: Path):
    """Cross-task bundle sets cross_task_bundle on every gen-1 agent's seed_package."""
    bundle_file = _make_cross_task_bundle_file(tmp_path)
    config = GenerationConfig(
        num_generations=1,
        num_agents=2,
        seed_bundle_path=str(bundle_file),
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
    )
    assert len(orch.agents) == 2
    for agent in orch.agents:
        seed = agent.seed_package
        assert isinstance(seed, dict)
        ct = seed.get("cross_task_bundle")
        assert isinstance(ct, dict)
        assert ct["transferable_insights"] == [
            "Grids with bilateral symmetry often use axis-aligned tiles.",
            "Color-count invariants survive most transformations.",
        ]
        assert ct["confirmed_constraints"] == ["Symmetric tasks preserve the tile axis."]
        assert ct["rejected_hypotheses"] == ["Blind recoloring fails when colors encode position."]
        assert ct["pitfalls"] == ["Do not assume the output shape matches the input shape."]
        assert ct["checks"] == ["Verify pixel-count conservation before committing a grid."]
        assert ct["next_steps"] == ["Try axis detection before color mapping."]
        assert ct["evidence_post_ids"] == [42, 97]
        assert seed["workstream_name"] == "kt-cross-task-bundle"
        # After ba4d2fbe (fix(kt): don't leak meta.source_experiment into
        # recipient MEMORY.md), the workstream description fallback no
        # longer interpolates the donor experiment name — agents see the
        # anonymized literal instead. Assert both directions explicitly.
        assert "anonymized donor experiment" in seed["workstream_description"]
        assert "test-donor-crosstask" not in seed["workstream_description"]
        # Legacy insight_bundle should be empty for cross-task-only injection
        assert seed["insight_bundle"] == []


def test_inject_seed_bundle_stamps_external_source_marker(tmp_path: Path):
    """--seed-bundle-path bundles are stamped `_external_seed_source` so the
    seed renderer applies the generous external caps (4000/item + 16k section)
    instead of the internal 480-char cap — mirroring the per-task loader."""
    bundle_file = _make_cross_task_bundle_file(tmp_path)
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        seed_bundle_path=str(bundle_file),
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
    )
    ct = orch.agents[0].seed_package["cross_task_bundle"]
    assert ct["_external_seed_source"] == str(bundle_file)


def test_gen1_agents_without_bundle():
    """Without seed_bundle_path, gen-1 agents have empty seed_package."""
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
    )
    agent = orch.agents[0]
    assert agent.seed_package == {} or not agent.seed_package.get("insight_bundle")


# ---------------------------------------------------------------------------
# Fail-loud on bad --seed-bundle-path (audit M2 follow-up)
# ---------------------------------------------------------------------------


def test_inject_seed_bundle_raises_when_path_missing(tmp_path: Path):
    """If --seed-bundle-path is supplied but the file doesn't exist, raise
    rather than silently running without KT — the user explicitly asked for
    KT and silent skip produces an ablation-noise run with no failure
    signal."""
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        seed_bundle_path=str(tmp_path / "does_not_exist.json"),
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
        )


def test_inject_seed_bundle_raises_on_malformed_json(tmp_path: Path):
    """If --seed-bundle-path points at a file that isn't valid JSON, raise
    with a hint about regenerating via extract_knowledge.py."""
    bad = tmp_path / "bad.json"
    bad.write_text("this is not json {{{")
    config = GenerationConfig(num_generations=1, num_agents=1, seed_bundle_path=str(bad))
    with pytest.raises(ValueError, match="not valid JSON"):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
        )


def test_inject_seed_bundle_raises_on_empty_bundle(tmp_path: Path):
    """A well-formed JSON bundle with neither populated cross_task nor a
    non-empty assets list is empty for our purposes — fail loud."""
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"meta": {}, "cross_task": {}, "assets": []}))
    config = GenerationConfig(num_generations=1, num_agents=1, seed_bundle_path=str(empty))
    with pytest.raises(ValueError, match="no usable content"):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
        )


def test_inject_seed_bundle_raises_on_non_dict_top_level(tmp_path: Path):
    """Bundle file must contain a JSON object at top level (not a list)."""
    listed = tmp_path / "list.json"
    listed.write_text(json.dumps([{"text": "stray list"}]))
    config = GenerationConfig(num_generations=1, num_agents=1, seed_bundle_path=str(listed))
    with pytest.raises(ValueError, match="must contain a JSON object"):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
        )


# ---------------------------------------------------------------------------
# Fail-loud on bad --seed-per-task-bundles-path (mirrors --seed-bundle-path)
# ---------------------------------------------------------------------------


def test_load_per_task_bundles_raises_when_path_missing(tmp_path: Path):
    """--seed-per-task-bundles-path supplied but the file doesn't exist: raise
    rather than silently running without per-task KT."""
    config = GenerationConfig(
        num_generations=1,
        num_agents=1,
        seed_per_task_bundles_path=str(tmp_path / "missing.json"),
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
        )


def test_load_per_task_bundles_raises_on_malformed_json(tmp_path: Path):
    """--seed-per-task-bundles-path file that isn't valid JSON: raise loudly."""
    bad = tmp_path / "bad.json"
    bad.write_text("this is not json {{{")
    config = GenerationConfig(num_generations=1, num_agents=1, seed_per_task_bundles_path=str(bad))
    with pytest.raises(ValueError, match="not valid JSON"):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
        )


def test_load_per_task_bundles_raises_on_empty_bundles_list(tmp_path: Path):
    """A bundle file with no `bundles` key or an empty `bundles` list raises."""
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"bundles": []}))
    config = GenerationConfig(num_generations=1, num_agents=1, seed_per_task_bundles_path=str(empty))
    with pytest.raises(ValueError, match="no usable.*bundles"):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
        )


def test_load_per_task_bundles_raises_when_no_row_has_usable_dict(tmp_path: Path):
    """A non-empty `bundles` list whose rows all lack a usable
    ``distilled_knowledge``/``per_task_bundle`` dict (donor schema drift) must
    raise rather than silently injecting ZERO bundles — otherwise the run
    degrades to a baseline-with-no-KT while looking like a KT run."""
    drifted = tmp_path / "drifted.json"
    drifted.write_text(
        json.dumps(
            {
                "bundles": [
                    {"task_id": "t1"},  # no bundle payload at all
                    {"task_id": "t2", "distilled_knowledge": {}},  # empty dict
                    {"task_id": "t3", "per_task_bundle": "not-a-dict"},  # wrong type
                    {"distilled_knowledge": {"confirmed_constraints": ["x"]}},  # no task_id
                ]
            }
        )
    )
    config = GenerationConfig(num_generations=1, num_agents=1, seed_per_task_bundles_path=str(drifted))
    with pytest.raises(ValueError, match="zero usable"):
        GenerationalOrchestrator(
            config=config,
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
        )


# ---------------------------------------------------------------------------
# Scoped bundle seeding — per-task and cross-task bundles flow through seeder
# ---------------------------------------------------------------------------


@pytest.fixture
def knowledge_store(tmp_path):
    from ksi.memory.knowledge_store import KnowledgeStore

    ks = KnowledgeStore(str(tmp_path / "k.sqlite"), default_experiment="test")
    yield ks
    ks.close()


def test_seeder_accepts_cross_task_and_knowledge_store_kwargs(knowledge_store):
    """PopulationSeeder.seed now accepts cross_task_bundle and knowledge_store."""
    labels = ["t-a", "t-b"]
    cross_task_bundle = {
        "transferable_insights": ["Shared pattern X"],
        "pitfalls": ["Common mistake Y"],
        "checks": ["Run tests"],
        "evidence_post_ids": [1, 2],
    }

    seeder = PopulationSeeder()
    agents = seeder.seed(
        num_agents=2,
        task_labels=labels,
        cross_task_bundle=cross_task_bundle,
        knowledge_store=knowledge_store,
        generation=1,
        experiment="test",
    )

    assert len(agents) == 2
    for agent in agents:
        pkg = agent.seed_package
        assert pkg.get("cross_task_bundle") == cross_task_bundle


def test_seeder_loads_per_task_bundles_from_knowledge_store(knowledge_store):
    """When a per-task bundle exists for an agent's assigned task, it is embedded."""
    knowledge_store.record_distillation(
        task_id="t-a",
        generation=1,
        bundle={
            "task_id": "t-a",
            "transferable_insights": ["Use library X"],
            "pitfalls": [],
            "checks": [],
            "evidence_post_ids": [],
        },
        scope="per_task",
        experiment="test",
    )
    knowledge_store.record_distillation(
        task_id="t-b",
        generation=1,
        bundle={
            "task_id": "t-b",
            "transferable_insights": ["Watch for Y"],
            "pitfalls": [],
            "checks": [],
            "evidence_post_ids": [],
        },
        scope="per_task",
        experiment="test",
    )

    seeder = PopulationSeeder()
    agents = seeder.seed(
        num_agents=2,
        task_labels=["t-a", "t-b"],
        knowledge_store=knowledge_store,
        generation=1,
        experiment="test",
    )

    assert len(agents) == 2
    per_a = agents[0].seed_package.get("per_task_bundle")
    per_b = agents[1].seed_package.get("per_task_bundle")
    assert per_a is not None
    assert per_a.get("transferable_insights") == ["Use library X"]
    assert per_b is not None
    assert per_b.get("transferable_insights") == ["Watch for Y"]


def test_seeder_omits_per_task_bundle_when_store_missing(knowledge_store):
    """If no per-task bundle exists for an agent's task, the key is absent."""
    seeder = PopulationSeeder()
    agents = seeder.seed(
        num_agents=1,
        task_labels=["missing-task"],
        knowledge_store=knowledge_store,
        generation=1,
        experiment="test",
    )
    assert "per_task_bundle" not in agents[0].seed_package


def test_seeder_no_legacy_shared_insight_bundle_param():
    """The legacy shared_insight_bundle/shared_bundle_summary params are gone."""
    import inspect

    sig = inspect.signature(PopulationSeeder.seed)
    assert "shared_insight_bundle" not in sig.parameters
    assert "shared_bundle_summary" not in sig.parameters
    # And the new kwargs are present.
    assert "cross_task_bundle" in sig.parameters
    assert "knowledge_store" in sig.parameters


def test_seeder_without_labels_also_accepts_bundles(knowledge_store):
    """The no-labels fallback path still exposes the cross-task bundle."""
    cross_task_bundle = {
        "transferable_insights": ["X"],
        "pitfalls": [],
        "checks": [],
        "evidence_post_ids": [],
    }
    seeder = PopulationSeeder()
    # Without task_labels, this falls into the fallback branch. cross_task
    # still renders; per_task depends on labels — here there are none.
    agents = seeder.seed(
        num_agents=1,
        cross_task_bundle=cross_task_bundle,
        knowledge_store=knowledge_store,
        generation=1,
        experiment="test",
    )
    assert len(agents) == 1
    assert agents[0].seed_package.get("cross_task_bundle") == cross_task_bundle


def test_resume_seed_preserves_cross_task_bundle_for_haiku_coding_tasks():
    """The Haiku × {polyglot, swebench_pro} cross-task suppression was removed
    so the resume-seed path no longer drops the cross-task bundle for those
    cells. The bundle must reach every recipient agent regardless of model
    family or task source — that's the contract the paper makes."""
    config = GenerationConfig(
        num_generations=2,
        num_agents=1,
        model="claude-haiku-4-5-20251001",
    )
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
    )
    knowledge = MagicMock()
    # Default target-conditioning delivers the cross-task bundle per task via
    # load_distillations_batch(scope="cross_task"), keyed by the recipient's
    # task id — not the single sentinel-keyed load_distillation broadcast.
    knowledge.load_distillations_batch.return_value = {"polyglot-task": {"transferable_insights": ["cross task hint"]}}
    knowledge.count_seed_snapshots.return_value = 1
    orch._knowledge = knowledge

    prepare_resume_population(
        orch,
        source_generation=1,
        next_tasks=[
            TaskSpec(
                id="polyglot-task",
                prompt="solve",
                metadata={"task_source": "polyglot"},
            )
        ],
    )

    assert orch.agents[0].seed_package.get("cross_task_bundle") == {"transferable_insights": ["cross task hint"]}


def test_load_cross_task_seed_bundle_skips_removed_alt_format_rows():
    config = GenerationConfig(num_generations=2, num_agents=1)
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
    )
    knowledge = MagicMock()
    knowledge.load_distillation.return_value = {"format": "motifs", "motifs": [{"name": "legacy"}]}
    orch._knowledge = knowledge

    assert load_cross_task_seed_bundle(orch, generation=1) is None


# ---------------------------------------------------------------------------
# KT adapter LLM-call failure falls back to the deterministic asset memo
# (review finding 609-2)
# ---------------------------------------------------------------------------


def test_kt_adapter_llm_failure_uses_deterministic_fallback():
    """If the adapter LLM call raises a non-auth error, the recipient must get
    the deterministic ranked asset fallback memo, not None (which would silently
    drop the agent back to the raw cross-task rendering)."""
    config = GenerationConfig(num_generations=1, num_agents=1)
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
    )

    def _boom(**kwargs):
        raise ValueError("simulated transient LLM failure")

    orch._llm_call = _boom  # type: ignore[method-assign]

    agent = AgentState(id="agent-0", generation=1)
    task = TaskSpec(id="arc-task-1", prompt="solve", metadata={"task_source": "arc"})
    cross_task = {
        "confirmed_constraints": ["Output preserves the bilateral symmetry axis."],
        "transferable_insights": ["Color-count invariants survive most transforms."],
        "pitfalls": ["Do not assume output shape matches input shape."],
        "checks": ["Verify pixel-count conservation before committing."],
    }

    memo = orch._kt_adapter_service._build_memo(
        generation=1,
        agent=agent,
        task=task,
        cross_task=cross_task,
    )

    assert memo is not None
    assert memo.get("_memo_source") == "asset_fallback"
    assert memo["relevant_constraints"] == ["Output preserves the bilateral symmetry axis."]
    assert memo["relevant_heuristics"] == ["Color-count invariants survive most transforms."]


def test_kt_adapter_polyglot_proceeds_with_stripped_payload():
    """The former polyglot fail-closed guard is retired now that
    adapter_task_payload omits hidden test_files/build_files (parity enforced at
    source). Polyglot KT must therefore proceed to the LLM/fallback path with no
    opt-in env required. Payload-level parity is asserted in
    tests/test_kt_adapter_parity.py."""
    config = GenerationConfig(num_generations=1, num_agents=1)
    orch = GenerationalOrchestrator(config=config, runtime=MagicMock(), evaluator=MagicMock(), llm=MagicMock())

    llm_calls = {"n": 0}

    def _count_then_fail(**kwargs):
        # We reach the LLM (no guard); a transient error then routes to the
        # existing deterministic asset fallback.
        llm_calls["n"] += 1
        raise ValueError("simulated transient LLM failure")

    orch._llm_call = _count_then_fail  # type: ignore[method-assign]

    agent = AgentState(id="agent-0", generation=1)
    task = TaskSpec(id="poly-1", prompt="solve", metadata={"task_source": "polyglot"})
    cross_task = {"transferable_insights": ["x"]}

    memo = orch._kt_adapter_service._build_memo(generation=1, agent=agent, task=task, cross_task=cross_task)
    assert llm_calls["n"] == 1
    assert memo is not None and memo.get("_memo_source") == "asset_fallback"


def test_kt_adapter_records_tokens_into_the_live_run_accumulator():
    """Regression (#982 #1 review): the KtAdapterService must record kt_adapter
    token usage into the engine's *current* accumulator, not the one captured at
    construction. ``run()`` reassigns ``self.accumulator`` each run, so a memo
    built during a run must land in that fresh accumulator (the one flushed and
    totalled), otherwise KT token phases silently vanish. Pre-fix (the service
    captured the ctor-time accumulator) this recorded into the orphaned instance
    and the live one stayed empty."""
    config = GenerationConfig(num_generations=1, num_agents=1)
    orch = GenerationalOrchestrator(
        config=config,
        runtime=MagicMock(),
        evaluator=MagicMock(),
        llm=MagicMock(),
    )
    ctor_accumulator = orch.accumulator  # the instance captured at construction

    # Simulate the start of run(): engine.run() rebinds self.accumulator to a
    # fresh TokenAccumulator (engine.py). The service was built once at __init__.
    orch.accumulator = TokenAccumulator()
    live_accumulator = orch.accumulator
    assert live_accumulator is not ctor_accumulator

    def _fake_llm(**kwargs):
        return LLMResponse(text="{}", usage=TokenUsage(input_tokens=100, output_tokens=50))

    orch._llm_call = _fake_llm  # type: ignore[method-assign]

    agent = AgentState(id="agent-0", generation=2)
    task = TaskSpec(id="arc-task-1", prompt="solve", metadata={"task_source": "arc"})
    cross_task = {"transferable_insights": ["Color-count invariants survive most transforms."]}

    orch._kt_adapter_service._build_memo(generation=2, agent=agent, task=task, cross_task=cross_task)

    kt_key = (2, "agent-0", "__lc:kt_adapter")
    # The kt_adapter usage lands in the LIVE run accumulator...
    assert kt_key in live_accumulator._entries
    assert live_accumulator._entries[kt_key].total == 150
    # ...and NOT in the orphaned ctor-time accumulator.
    assert kt_key not in ctor_accumulator._entries
