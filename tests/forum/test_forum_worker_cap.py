"""Forum worker cap resolution (2026-07-04 incident regression tests).

A live campaign with ``--max-concurrent-tasks 6`` fanned out ~41 cross-task
forum containers at once because forum concurrency was governed by a separate
knob (``--max-concurrent-forum-tasks``, default 50) that ignored the task cap.
Under default-on egress isolation the burst mass-failed every forum container
("Docker network ... did not become ready"), yielding zero forum posts.

The fix: the forum knob defaults to 0 = "follow --max-concurrent-tasks"; an
explicit positive value overrides. Invariant: with the forum knob unset it
always tracks the task cap (both default to 25 as of #1154), so the fix binds
whenever a user lowered --max-concurrent-tasks — exactly the incident shape.
"""

from pathlib import Path

from kcsi.cli import _build_generation_config, build_parser
from kcsi.models import GenerationConfig
from kcsi.orchestrator.forum_phase import _resolve_forum_worker_cap

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(**overrides) -> GenerationConfig:
    return GenerationConfig(num_generations=1, num_agents=1, **overrides)


def test_default_follows_task_cap():
    """The incident shape: task cap 6, forum knob unset, 41 debate agents."""
    config = _config(max_concurrent_tasks=6, max_concurrent_forum_tasks=0)
    assert _resolve_forum_worker_cap(config, 41) == 6


def test_explicit_forum_value_overrides_task_cap():
    config = _config(max_concurrent_tasks=6, max_concurrent_forum_tasks=12)
    assert _resolve_forum_worker_cap(config, 41) == 12


def test_full_default_behavior_at_25():
    """Both knobs at their defaults resolve to the shared 25-way ceiling.

    max_concurrent_tasks defaults to 25 (cli.py / models.py, #1154), so a
    forum default of 0 = follow-task-cap resolves to 25 for anyone who never
    set --max-concurrent-tasks.
    """
    config = _config()
    assert config.max_concurrent_tasks == 25
    assert _resolve_forum_worker_cap(config, 100) == 25


def test_pool_smaller_than_cap_uses_pool_size():
    config = _config(max_concurrent_tasks=6, max_concurrent_forum_tasks=0)
    assert _resolve_forum_worker_cap(config, 3) == 3
    explicit = _config(max_concurrent_tasks=6, max_concurrent_forum_tasks=12)
    assert _resolve_forum_worker_cap(explicit, 3) == 3


def test_both_caps_zero_falls_back_to_50():
    config = _config(max_concurrent_tasks=0, max_concurrent_forum_tasks=0)
    assert _resolve_forum_worker_cap(config, 100) == 50


def test_empty_pool_still_returns_at_least_one_worker():
    config = _config(max_concurrent_tasks=6, max_concurrent_forum_tasks=0)
    assert _resolve_forum_worker_cap(config, 0) == 1


def test_negative_task_cap_falls_back_to_50():
    """A negative --max-concurrent-tasks (forum knob unset) must not propagate
    as the worker count — ThreadPoolExecutor(max_workers=-1) would ValueError.
    Mirror execution_phase's ``> 0 else 50`` guard."""
    config = _config(max_concurrent_tasks=-1, max_concurrent_forum_tasks=0)
    assert _resolve_forum_worker_cap(config, 100) == 50


def test_negative_explicit_forum_value_follows_task_cap():
    """A negative explicit forum value is not positive, so it falls through to
    the task-cap branch (forum=-3, task=6, pool=41 → 6)."""
    config = _config(max_concurrent_tasks=6, max_concurrent_forum_tasks=-3)
    assert _resolve_forum_worker_cap(config, 41) == 6


def test_model_default_is_follow_task_cap_sentinel():
    assert _config().max_concurrent_forum_tasks == 0


def test_cli_default_is_follow_task_cap_sentinel():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            "/tmp/tasks",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--runtime",
            "container",
            "--provider-profile",
            "/tmp/profile.env",
        ]
    )
    assert args.max_concurrent_forum_tasks == 0


def test_cli_passthrough_into_resolver_incident_shape():
    """End-to-end: parse_args → _build_generation_config → _resolve_forum_worker_cap
    for the incident shape (task cap 6, forum unset, 41 debate agents → 6).

    Guards cli.py's GenerationConfig passthrough of both
    ``max_concurrent_tasks`` / ``max_concurrent_forum_tasks``: dropping either
    kwarg would let the resolver see a default cap and over-fan the forum.
    """
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "arc",
            "--tasks-path",
            "/tmp/tasks",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--runtime",
            "container",
            "--provider-profile",
            "/tmp/profile.env",
            "--max-concurrent-tasks",
            "6",
        ]
    )
    config = _build_generation_config(args, num_agents=1, holdout_ids=[], model="")
    assert config.max_concurrent_tasks == 6
    assert config.max_concurrent_forum_tasks == 0
    assert _resolve_forum_worker_cap(config, 41) == 6


def test_cap_sites_use_shared_resolver():
    """Both forum phases (per-task and cross-task) must resolve their worker
    count via _resolve_forum_worker_cap, not the old raw ``or`` fallback that
    let the forum default silently exceed the task-execution cap."""
    source = (REPO_ROOT / "src" / "kcsi" / "orchestrator" / "forum_phase.py").read_text()
    assert "config.max_concurrent_forum_tasks or len(debate_agents)" not in source
    assert source.count("_resolve_forum_worker_cap(collab.config, len(debate_agents))") == 2
