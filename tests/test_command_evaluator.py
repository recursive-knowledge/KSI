from pathlib import Path

from ksi.eval.command import CommandEvaluator
from ksi.eval.registry import resolve_evaluator
from ksi.models import TaskSpec
from ksi.orchestrator.scoring import score_from_eval_results


def _task(tmp_path, command="true", **md):
    seed = tmp_path / "seed"
    seed.mkdir(exist_ok=True)
    return TaskSpec(
        id="t1",
        prompt="p",
        metadata={
            "task_source": "custom",
            "repo_path": str(seed),
            "eval_command": command,
            "eval_timeout_sec": 30.0,
            **md,
        },
    )


def test_registered():
    assert resolve_evaluator("command") is not None


def test_pass_and_fail_exit_codes(tmp_path):
    ev = CommandEvaluator()
    ws = tmp_path / "ws"
    ws.mkdir()
    ok = ev.evaluate(task=_task(tmp_path, "exit 0"), model_output="", runtime_meta={"host_workspace_repo_dir": str(ws)})
    assert ok["native_score"] == 1.0 and ok["resolved"] is True
    assert score_from_eval_results(ok) == 1.0
    bad = ev.evaluate(
        task=_task(tmp_path, "exit 3"), model_output="", runtime_meta={"host_workspace_repo_dir": str(ws)}
    )
    assert bad["native_score"] == 0.0 and bad["exit_code"] == 3
    assert score_from_eval_results(bad) == 0.0


def test_runs_in_captured_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "flag.txt").write_text("x", encoding="utf-8")
    ev = CommandEvaluator()
    res = ev.evaluate(
        task=_task(tmp_path, "test -f flag.txt"), model_output="", runtime_meta={"host_workspace_repo_dir": str(ws)}
    )
    assert res["native_score"] == 1.0
    assert res["eval_workdir"] == str(ws)


def test_falls_back_to_seed_copy_when_no_capture(tmp_path):
    task = _task(tmp_path, "test -f starter.txt")
    Path(task.metadata["repo_path"], "starter.txt").write_text("s", encoding="utf-8")
    res = CommandEvaluator().evaluate(task=task, model_output="", runtime_meta={})
    assert res["native_score"] == 1.0
    # graded in a COPY, not the seed itself
    assert res["eval_workdir"] != task.metadata["repo_path"]


def test_overlays_workspace_solution_files_when_no_capture(tmp_path):
    # No host_workspace_repo_dir (e.g. the runtime wiped the per-task
    # workspace before this evaluate() call): fall back to a fresh copy of
    # the seed dir, overlaid with runtime_meta["workspace_solution_files"]
    # (the file-content channel main.ts captures in-process, before wipe).
    task = _task(tmp_path, 'python3 -c "import solution; assert solution.value == 42"')
    seed = Path(task.metadata["repo_path"])
    seed.joinpath("starter.txt").write_text("s", encoding="utf-8")
    res = CommandEvaluator().evaluate(
        task=task,
        model_output="",
        runtime_meta={"workspace_solution_files": {"solution.py": "value = 42\n"}},
    )
    assert res["native_score"] == 1.0
    # Graded in a fresh copy, not the seed dir itself.
    assert res["eval_workdir"] != str(seed)
    # The seed's own files are still present alongside the overlay.
    assert (Path(res["eval_workdir"]) / "starter.txt").read_text(encoding="utf-8") == "s"


def test_stale_score_json_in_captured_workspace_is_not_honored(tmp_path):
    # A stale (or agent-forged) score.json already sitting in a captured
    # host_workspace_repo_dir must not be picked up as the eval command's
    # own override -- only the command that just ran may produce it.
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "score.json").write_text('{"score": 1.0}', encoding="utf-8")
    res = CommandEvaluator().evaluate(
        task=_task(tmp_path, "exit 1"), model_output="", runtime_meta={"host_workspace_repo_dir": str(ws)}
    )
    assert res["native_score"] == 0.0


def test_overlay_never_writes_score_json_or_test_files(tmp_path):
    # runtime_meta["workspace_solution_files"] is untrusted content captured
    # from the agent's own workspace: a "score.json" entry must never be
    # written (would forge the override), and a "tests.py" entry must never
    # clobber the seed's real grader file.
    task = _task(tmp_path, "true")
    seed = Path(task.metadata["repo_path"])
    seed.joinpath("tests.py").write_text("REAL TESTS\n", encoding="utf-8")
    res = CommandEvaluator().evaluate(
        task=task,
        model_output="",
        runtime_meta={
            "workspace_solution_files": {
                "score.json": '{"score": 1.0}',
                "tests.py": "FORGED TESTS\n",
            }
        },
    )
    workdir = Path(res["eval_workdir"])
    assert not (workdir / "score.json").exists()
    assert (workdir / "tests.py").read_text(encoding="utf-8") == "REAL TESTS\n"


def test_score_json_partial_credit(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    cmd = "python3 -c \"import json; json.dump({'score': 0.5}, open('score.json','w'))\"; exit 1"
    res = CommandEvaluator().evaluate(
        task=_task(tmp_path, cmd), model_output="", runtime_meta={"host_workspace_repo_dir": str(ws)}
    )
    assert res["native_score"] == 0.5


def test_unscored_statuses_score_none(tmp_path):
    ev = CommandEvaluator()
    ws = tmp_path / "ws"
    ws.mkdir()
    no_cmd = ev.evaluate(task=_task(tmp_path, ""), model_output="", runtime_meta={})
    assert no_cmd["status"] == "no_eval_command"
    assert score_from_eval_results(no_cmd) is None
    timeout_task = _task(tmp_path, "sleep 5")
    timeout_task.metadata["eval_timeout_sec"] = 0.2
    to = ev.evaluate(task=timeout_task, model_output="", runtime_meta={"host_workspace_repo_dir": str(ws)})
    assert to["status"] == "eval_timeout"
    assert score_from_eval_results(to) is None
