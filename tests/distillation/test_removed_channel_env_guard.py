import pytest

from ksi.distillation._removed_env import assert_no_removed_channel_env


@pytest.mark.parametrize(
    "var,val",
    [
        ("KSI_DISTILL_STRATEGY", "fold"),
        ("KSI_PER_TASK_CHANNEL", "ledger"),
        ("KSI_CROSS_TASK_CHANNEL", "motifs"),
        ("SWARMS_DISTILL_STRATEGY", "fold"),
        ("SWARMS_PER_TASK_CHANNEL", "ledger"),
        ("SWARMS_CROSS_TASK_CHANNEL", "motifs"),
    ],
)
def test_removed_channel_values_hard_error(monkeypatch, var, val):
    monkeypatch.setenv(var, val)
    with pytest.raises(RuntimeError) as exc:
        assert_no_removed_channel_env()
    assert var in str(exc.value)
    assert "removed" in str(exc.value).lower()


@pytest.mark.parametrize(
    "var,val",
    [
        ("KSI_DISTILL_STRATEGY", "window"),
        ("KSI_PER_TASK_CHANNEL", "bundle"),
        ("KSI_CROSS_TASK_CHANNEL", "bundle"),
        ("SWARMS_DISTILL_STRATEGY", "window"),
        ("SWARMS_PER_TASK_CHANNEL", "bundle"),
        ("SWARMS_CROSS_TASK_CHANNEL", "bundle"),
    ],
)
def test_surviving_default_value_is_accepted(monkeypatch, var, val):
    monkeypatch.setenv(var, val)
    assert_no_removed_channel_env() is None  # no raise


def test_unset_is_noop(monkeypatch):
    for v in (
        "KSI_DISTILL_STRATEGY",
        "KSI_PER_TASK_CHANNEL",
        "KSI_CROSS_TASK_CHANNEL",
        "SWARMS_DISTILL_STRATEGY",
        "SWARMS_PER_TASK_CHANNEL",
        "SWARMS_CROSS_TASK_CHANNEL",
    ):
        monkeypatch.delenv(v, raising=False)
    assert assert_no_removed_channel_env() is None


@pytest.mark.parametrize("val", ["artifact", "garbage", "BUNDLE_TYPO"])
def test_any_non_default_value_hard_errors(monkeypatch, val):
    # Not just the historical removed value (ledger): ANY non-default value is
    # rejected, and the message lists the removed channels generically rather
    # than mis-naming one specific channel.
    monkeypatch.setenv("KSI_PER_TASK_CHANNEL", val)
    with pytest.raises(RuntimeError) as exc:
        assert_no_removed_channel_env()
    msg = str(exc.value)
    assert "fold / ledger / motifs" in msg
    assert "ledger" in msg and "motifs" in msg  # not a single mis-named channel


def test_surviving_default_is_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv("KSI_DISTILL_STRATEGY", "  WINDOW  ")
    assert assert_no_removed_channel_env() is None


def test_run_invokes_the_guard_before_work_begins(monkeypatch, tmp_path):
    # The guard must be called from GenerationalOrchestrator.run itself (outside
    # the distill-phase try/except that would otherwise swallow it), including
    # runs that would otherwise skip distillation.
    from unittest.mock import MagicMock

    from ksi.models import GenerationConfig
    from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence

    runtime = MagicMock()
    orch = GenerationalOrchestrator(
        config=GenerationConfig(
            num_generations=1,
            num_agents=1,
            no_memory=True,
        ),
        runtime=runtime,
        evaluator=MagicMock(),
        llm=MagicMock(),
        persistence=NoopPersistence(),
    )

    monkeypatch.setenv("KSI_DISTILL_STRATEGY", "fold")
    with pytest.raises(RuntimeError, match="KSI_DISTILL_STRATEGY"):
        orch.run([])
    runtime.execute.assert_not_called()


def test_constructor_invokes_guard_before_store_initialization(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from ksi.models import GenerationConfig
    from ksi.orchestrator.engine import GenerationalOrchestrator, NoopPersistence

    db_path = tmp_path / "knowledge.sqlite"
    monkeypatch.setenv("KSI_DISTILL_STRATEGY", "fold")
    with pytest.raises(RuntimeError, match="KSI_DISTILL_STRATEGY"):
        GenerationalOrchestrator(
            config=GenerationConfig(
                num_generations=1,
                num_agents=1,
                knowledge_db_path=str(db_path),
            ),
            runtime=MagicMock(),
            evaluator=MagicMock(),
            llm=MagicMock(),
            persistence=NoopPersistence(),
        )
    assert not db_path.exists()


def test_api_run_invokes_guard_before_orchestrator_construction(monkeypatch):
    import ksi.api as api

    class _ForbiddenOrchestrator:
        def __init__(self, **_kwargs):  # noqa: ANN003
            raise AssertionError("orchestrator should not be constructed")

    monkeypatch.setattr(api, "GenerationalOrchestrator", _ForbiddenOrchestrator)
    monkeypatch.setenv("KSI_PER_TASK_CHANNEL", "ledger")
    with pytest.raises(RuntimeError, match="KSI_PER_TASK_CHANNEL"):
        api.run(object(), [], runtime=object(), evaluator=object(), llm=object())


def test_cli_invokes_guard_before_task_loading(monkeypatch):
    import ksi.cli as cli_module

    def _forbidden(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("task loading should not run")

    monkeypatch.setattr(cli_module, "load_eval_records_for_source", _forbidden)
    monkeypatch.setattr(cli_module, "load_tasks_for_source", _forbidden)
    monkeypatch.setattr(cli_module, "_choose_runtime", _forbidden)
    monkeypatch.setenv("KSI_CROSS_TASK_CHANNEL", "motifs")
    with pytest.raises(RuntimeError, match="KSI_CROSS_TASK_CHANNEL"):
        cli_module.main(["--task-source", "polyglot", "--tasks-path", "missing.json"])
