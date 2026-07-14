"""Tests for population seeding."""

from kcsi.seeding.seeder import PopulationSeeder


def test_seed_without_labels():
    seeder = PopulationSeeder()
    agents = seeder.seed(num_agents=3, generation=0)
    assert len(agents) == 3
    assert all(a.generation == 1 for a in agents)
    assert all(a.workstream == "" for a in agents)
    assert all(a.seed_package == {} for a in agents)


def test_seed_task_labels_assign_one_task_per_agent():
    labels = ["django-orm", "sqlfluff-linting", "pydicom-parsing"]
    seeder = PopulationSeeder()
    agents = seeder.seed(num_agents=3, generation=0, task_labels=labels)
    assigned = [a.workstream for a in agents]
    assert assigned == labels
    assert all(a.generation == 1 for a in agents)
    assert all(a.seed_package.get("workstream_name") == a.workstream for a in agents)


def test_seed_broadcast_cross_task_bundle_reaches_every_label():
    """The single broadcast bundle is attached to every task label."""
    seeder = PopulationSeeder()
    single = {"transferable_insights": ["broadcast"], "evidence_post_ids": []}
    agents = seeder.seed(
        num_agents=2,
        generation=0,
        task_labels=["t1", "t2"],
        cross_task_bundle=single,
    )
    assert agents[0].seed_package["cross_task_bundle"] == single
    assert agents[1].seed_package["cross_task_bundle"] == single


def test_seed_skips_removed_alt_format_per_task_bundle():
    class Store:
        def load_distillations_batch(self, **kwargs):  # noqa: ANN001, D102
            return {"t1": {"format": "ledger", "task_facts": [{"text": "legacy fact"}]}}

    seeder = PopulationSeeder()
    agents = seeder.seed(num_agents=1, generation=0, task_labels=["t1"], knowledge_store=Store())
    assert "per_task_bundle" not in agents[0].seed_package


def test_seed_attaches_canonical_per_task_bundle_from_batch():
    """A canonical bundle returned by the batched load is attached."""
    bundle = {"transferable_insights": ["insight"], "evidence_post_ids": []}

    class Store:
        def load_distillations_batch(self, **kwargs):  # noqa: ANN001, D102
            return {"t1": bundle}

    seeder = PopulationSeeder()
    agents = seeder.seed(num_agents=1, generation=0, task_labels=["t1"], knowledge_store=Store())
    assert agents[0].seed_package["per_task_bundle"] == bundle


def test_seed_batch_loads_once_and_excludes_holdout_labels():
    """Per-task bundles load in a single batched call; hold-out labels excluded."""
    calls: list[list[str]] = []

    class Store:
        def load_distillations_batch(self, **kwargs):  # noqa: ANN001, D102
            calls.append(list(kwargs["task_ids"]))
            return {tid: {"transferable_insights": [tid]} for tid in kwargs["task_ids"]}

    seeder = PopulationSeeder()
    agents = seeder.seed(
        num_agents=3,
        generation=0,
        task_labels=["t1", "t2", "hold"],
        knowledge_store=Store(),
        skip_per_task_labels={"hold"},
    )
    # One batched query, and the hold-out label is never looked up.
    assert calls == [["t1", "t2"]]
    assert agents[0].seed_package["per_task_bundle"] == {"transferable_insights": ["t1"]}
    assert agents[1].seed_package["per_task_bundle"] == {"transferable_insights": ["t2"]}
    assert "per_task_bundle" not in agents[2].seed_package


def test_seed_without_cross_task_bundle_omits_the_key():
    """No broadcast bundle -> no cross_task_bundle key in the seed package."""
    seeder = PopulationSeeder()
    agents = seeder.seed(
        num_agents=2,
        generation=0,
        task_labels=["t1", "t2"],
        cross_task_bundle=None,
    )
    assert all("cross_task_bundle" not in a.seed_package for a in agents)


def test_seed_per_task_cross_task_bundle_delivers_own_task_bundle():
    """With conditioning on, each label gets ITS OWN cross-task bundle."""

    class Store:
        def load_distillations_batch(self, **kwargs):  # noqa: ANN001
            scope = kwargs["scope"]
            ids = kwargs["task_ids"]
            if scope == "cross_task":
                return {tid: {"transferable_insights": [f"ct-{tid}"], "evidence_post_ids": []} for tid in ids}
            return {}

    seeder = PopulationSeeder()
    agents = seeder.seed(
        num_agents=2,
        generation=0,
        task_labels=["t1", "t2"],
        cross_task_bundle={"transferable_insights": ["BROADCAST"], "evidence_post_ids": []},
        knowledge_store=Store(),
        cross_task_target_conditioning=True,
    )
    assert agents[0].seed_package["cross_task_bundle"]["transferable_insights"] == ["ct-t1"]
    assert agents[1].seed_package["cross_task_bundle"]["transferable_insights"] == ["ct-t2"]


def test_seed_conditioning_on_still_delivers_cross_task_to_holdouts():
    calls: list[tuple[str, list[str]]] = []

    class Store:
        def load_distillations_batch(self, **kwargs):  # noqa: ANN001
            scope = kwargs["scope"]
            ids = list(kwargs["task_ids"])
            calls.append((scope, ids))
            if scope == "cross_task":
                return {tid: {"transferable_insights": [f"ct-{tid}"], "evidence_post_ids": []} for tid in ids}
            return {tid: {"transferable_insights": [f"pt-{tid}"], "evidence_post_ids": []} for tid in ids}

    agents = PopulationSeeder().seed(
        num_agents=2,
        generation=1,
        task_labels=["t1", "h1"],
        knowledge_store=Store(),
        skip_per_task_labels={"h1"},
        cross_task_target_conditioning=True,
    )

    assert calls == [("per_task", ["t1"]), ("cross_task", ["t1", "h1"])]
    assert agents[0].seed_package["per_task_bundle"]["transferable_insights"] == ["pt-t1"]
    assert agents[0].seed_package["cross_task_bundle"]["transferable_insights"] == ["ct-t1"]
    assert "per_task_bundle" not in agents[1].seed_package
    assert agents[1].seed_package["cross_task_bundle"]["transferable_insights"] == ["ct-h1"]


def test_seed_conditioning_off_still_broadcasts():
    """Regression: conditioning off keeps the single broadcast bundle."""
    seeder = PopulationSeeder()
    single = {"transferable_insights": ["broadcast"], "evidence_post_ids": []}
    agents = seeder.seed(
        num_agents=2,
        generation=0,
        task_labels=["t1", "t2"],
        cross_task_bundle=single,
        cross_task_target_conditioning=False,
    )
    assert agents[0].seed_package["cross_task_bundle"] == single
    assert agents[1].seed_package["cross_task_bundle"] == single
