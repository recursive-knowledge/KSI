"""Polyglot harness evaluator — runs exercism-style tests in Docker containers.

Supports Python, Rust, Go, JavaScript, Java, and C++.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..models import TaskSpec
from .polyglot_docker import (
    POLYGLOT_RECIPE_BASE_IMAGE_LABEL,
    POLYGLOT_RECIPE_BASE_IMAGE_LABEL_ALIAS,
    POLYGLOT_RECIPE_LABEL,
    POLYGLOT_RECIPE_LABEL_ALIAS,
    POLYGLOT_RECIPE_SOURCE_LABEL,
    POLYGLOT_RECIPE_SOURCE_LABEL_ALIAS,
)

# Load .env so the polyglot eval Docker image name can be overridden without
# editing source. load_dotenv() is idempotent.
load_dotenv()

_DEFAULT_POLYGLOT_DOCKER_IMAGE = os.environ.get("POLYGLOT_DOCKER_IMAGE", "kcsi-polyglot-eval:latest")
DEFAULT_POLYGLOT_TIMEOUT_SEC = 180

log = logging.getLogger(__name__)

_POLYGLOT_EVAL_LABEL = "org.knowledgecentric.kcsi.eval"
# Historical MockingJay-era alias (pre-dates the kcsi rename): external cleanup
# tooling may filter on it, so eval containers keep carrying it.
_POLYGLOT_EVAL_LABEL_ALIAS = "org.mockingjay.swarms.eval"

# Emitted on stderr by the wrapped setup step when it exits nonzero, so the
# harness can distinguish "setup itself failed" (e.g. npm install hitting a
# transient registry error) from a genuine test failure -- both would
# otherwise just be "the chained command exited nonzero".
_SETUP_FAILURE_MARKER = "__kcsi_polyglot_setup_failed__"

# Safety char cap for test_stdout_tail/test_stderr_tail. The test-feedback
# retry loop (--polyglot-test-feedback-max-lines, default 50) caps the
# agent-visible window to the last 50 LINES on the TS side (extractCappedTail
# in runtime_runner/agent-runner/src/polyglot_test_feedback_core.ts), so this
# harness-side cap must comfortably contain 50 long lines. Framework output
# lines (gradle stack frames, jest diffs) run ~100-200 chars; 20_000 chars
# gives 50 lines x 400 chars of headroom. The old 2000-char cap held only
# ~15-25 gradle lines, silently shrinking the documented 50-line window.
# Kept bounded so a pathological run can't bloat eval_result payloads.
_TEST_OUTPUT_TAIL_CHARS = 20_000

# cpp vacuous-pass guard markers: ctest exits 0 when zero tests are registered
# (printing the first marker on stderr); the official Exercism cpp CMakeLists
# runs the Catch2 test binary at build time instead, printing the second
# marker on stdout on success. See the guard in ``_run_in_docker``.
_CTEST_NO_TESTS_MARKER = "No tests were found!!!"
_CATCH2_ALL_PASSED_MARKER = "All tests passed ("

# go/rust/python vacuous-pass guards: like ctest, these runners can exit
# 0 without any test actually executing (e.g. a solution file overwrote the
# official test file). Only markers with a reliable "zero tests ran" signal in
# the runner output are guarded; javascript (jest fails when no tests are found)
# and java lack such a signal and are intentionally not guarded here.
#
# go:   `go test ./...` prints "[no test files]" and exits 0 for a package with
#       no *_test.go; a package that ran tests prints an "ok\t<pkg>\t<time>"
#       line. The no-test marker with no such ok line means nothing was tested.
_GO_NO_TEST_FILES_MARKER = "[no test files]"
_GO_TESTS_RAN_RE = re.compile(r"(?m)^ok\s")
# rust: `cargo test` exits 0 when zero tests run ("test result: ok. 0 passed").
#       A genuine pass has at least one suite reporting a nonzero passed count.
_RUST_TESTS_RAN_RE = re.compile(r"test result: ok\.\s+[1-9]\d*\s+passed")
# python: pytest exits 5 ("no tests collected") — already != 0 — so a vacuous
#       pytest run normally fails on exit code alone. These markers only add
#       belt-and-suspenders coverage for a task that forced exit 0 while
#       collecting nothing.
_PYTEST_NO_TESTS_MARKERS = ("no tests ran", "collected 0 items")

_TOOL_VERSION_SCRIPT = r"""
set +e
python_version="$(python --version 2>&1 | head -n 1)"
pytest_version="$(python -m pytest --version 2>&1 | head -n 1)"
node_version="$(node --version 2>&1 | head -n 1)"
npm_version="$(npm --version 2>&1 | head -n 1)"
go_version="$(go version 2>&1 | head -n 1)"
rustc_version="$(rustc --version 2>&1 | head -n 1)"
cargo_version="$(cargo --version 2>&1 | head -n 1)"
java_version="$(java -version 2>&1 | head -n 1)"
gradle_version="$(gradle --version 2>&1 | sed -n 's/^Gradle /Gradle /p' | head -n 1)"
conda_version="$(conda --version 2>&1 | head -n 1)"
cmake_version="$(cmake --version 2>&1 | head -n 1)"
gcc_version="$(gcc --version 2>&1 | head -n 1)"
printf 'python=%s\n' "$python_version"
printf 'pytest=%s\n' "$pytest_version"
printf 'node=%s\n' "$node_version"
printf 'npm=%s\n' "$npm_version"
printf 'go=%s\n' "$go_version"
printf 'rustc=%s\n' "$rustc_version"
printf 'cargo=%s\n' "$cargo_version"
printf 'java=%s\n' "$java_version"
printf 'gradle=%s\n' "$gradle_version"
printf 'conda=%s\n' "$conda_version"
printf 'cmake=%s\n' "$cmake_version"
printf 'gcc=%s\n' "$gcc_version"
"""


def _inspect_polyglot_image(docker_image: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "docker_image": docker_image,
    }
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", docker_image, "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        metadata["docker_image_error"] = str(exc)
    else:
        if inspect.returncode == 0:
            try:
                image_info = json.loads(inspect.stdout.strip())
            except json.JSONDecodeError:
                metadata["docker_image_error"] = inspect.stdout.strip()
            else:
                metadata["docker_image_id"] = image_info.get("Id", "")
                repo_digests = image_info.get("RepoDigests") or []
                metadata["docker_repo_digests"] = [str(digest) for digest in repo_digests if digest]
                labels = (image_info.get("Config") or {}).get("Labels") or {}
                recipe = labels.get(POLYGLOT_RECIPE_LABEL) or labels.get(POLYGLOT_RECIPE_LABEL_ALIAS)
                recipe_base = labels.get(POLYGLOT_RECIPE_BASE_IMAGE_LABEL) or labels.get(
                    POLYGLOT_RECIPE_BASE_IMAGE_LABEL_ALIAS
                )
                recipe_source = labels.get(POLYGLOT_RECIPE_SOURCE_LABEL) or labels.get(
                    POLYGLOT_RECIPE_SOURCE_LABEL_ALIAS
                )
                if recipe:
                    metadata["recipe"] = recipe
                if recipe_base:
                    metadata["recipe_base_image"] = recipe_base
                if recipe_source:
                    metadata["recipe_source"] = recipe_source
        else:
            metadata["docker_image_error"] = (inspect.stderr or inspect.stdout or "").strip()

    return metadata


@lru_cache(maxsize=16)
def _polyglot_environment_metadata_for_image_id(
    docker_image: str,
    docker_image_id: str,
    recipe: str,
    recipe_base_image: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "runner": "kcsi",
        "docker_image": docker_image,
        "docker_image_id": docker_image_id,
        "tool_versions": {},
    }
    if recipe:
        metadata["recipe"] = recipe
    if recipe_base_image:
        metadata["recipe_base_image"] = recipe_base_image

    try:
        probe = subprocess.run(
            # bash -c (non-login): a login shell runs /etc/profile, which on
            # Debian/Ubuntu resets PATH to the minimal default and strips
            # /usr/local/go/bin, ~/.cargo/bin, etc. — making the probe report
            # missing tools. Match the test-run shell (also bash -c).
            ["docker", "run", "--rm", docker_image, "bash", "-c", _TOOL_VERSION_SCRIPT],
            capture_output=True,
            text=True,
            timeout=45,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        metadata["tool_versions_error"] = str(exc)
    else:
        if probe.returncode == 0:
            versions: dict[str, str] = {}
            for line in (probe.stdout or "").splitlines():
                key, sep, value = line.partition("=")
                if sep:
                    versions[key] = value.strip()
            metadata["tool_versions"] = versions
        else:
            metadata["tool_versions_error"] = (probe.stderr or probe.stdout or "").strip()

    return metadata


def _polyglot_environment_metadata(docker_image: str) -> dict[str, Any]:
    """Return machine-readable Docker/toolchain metadata for result artifacts."""
    image_metadata = _inspect_polyglot_image(docker_image)
    docker_image_id = str(image_metadata.get("docker_image_id") or "")
    if not docker_image_id:
        image_metadata["runner"] = "kcsi"
        image_metadata["tool_versions"] = {}
        return image_metadata

    metadata = _polyglot_environment_metadata_for_image_id(
        docker_image,
        docker_image_id,
        str(image_metadata.get("recipe") or ""),
        str(image_metadata.get("recipe_base_image") or ""),
    )
    return {**image_metadata, **metadata}


_polyglot_environment_metadata.cache_clear = (  # type: ignore[attr-defined]
    _polyglot_environment_metadata_for_image_id.cache_clear
)


def _docker_name_component(value: str, *, max_chars: int = 48) -> str:
    component = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip())
    component = component.strip("-.")
    return (component[:max_chars].strip("-.") or "task").lower()


def _cleanup_docker_container(container_name: str) -> dict[str, Any]:
    cleanup: dict[str, Any] = {
        "cleanup_container": container_name,
        "cleanup_attempted": True,
    }
    try:
        proc = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # pragma: no cover - Docker/runtime dependent
        cleanup["cleanup_status"] = "error"
        cleanup["cleanup_error"] = str(exc)
        return cleanup

    cleanup["cleanup_returncode"] = proc.returncode
    cleanup["cleanup_status"] = "ok" if proc.returncode == 0 else "failed"
    if proc.stdout:
        cleanup["cleanup_stdout_tail"] = proc.stdout[-1000:]
    if proc.stderr:
        cleanup["cleanup_stderr_tail"] = proc.stderr[-1000:]
    return cleanup


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

# P0-6: Allowlist of test command patterns per language.  Commands read from
# dataset JSON are validated against these before being passed to ``bash -c``.
# Multi-step chains joined by ``&&`` / ``||`` / ``;`` are allowed as long as
# every sub-command's prefix matches the language's pattern (or a shared safe
# helper like ``cd``/``ctest``).  This is needed for C++ Exercism tasks whose
# test_command is e.g. ``cmake -B build && cmake --build build && cd build &&
# ctest --output-on-failure``.
_ALLOWED_TEST_CMD_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^python3?\s+-m\s+pytest\b"),
    "rust": re.compile(r"^cargo\s+test\b"),
    "go": re.compile(r"^go\s+test\b"),
    "javascript": re.compile(r"^(npm\s+test|node\s+|jest\b|npx\s+jest\b)"),
    "java": re.compile(r"^(gradle\s+test|mvn\s+test|java\b)"),
    "cpp": re.compile(r"^(make\s+test|cmake\b|ctest\b|cd\s+\S+|g\+\+(?:\s|$)|c\+\+(?:\s|$))"),
}

# Dangerous shell metacharacters that could inject OR exfiltrate commands
# regardless of whether they appear in a chain.  We still reject these even in
# multi-step chains.  Pipes (``|``) are rejected here too because they enable
# curl-piping patterns (``wget x | bash``); command chaining uses ``&&`` etc.
_SHELL_INJECTION_RE = re.compile(r"[`$(){}<>]|(?<![&|])\|(?![|])")

# Splitter for legitimate shell chains.  Splits on ``&&``, ``||``, or ``;``.
_CHAIN_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")


def _validate_test_command(test_command: str, language: str) -> None:
    """Raise ``ValueError`` if *test_command* does not match the allowlist.

    This prevents shell-injection attacks via malicious ``test_command``
    values in dataset JSON.  Three checks are applied:

    1. Dangerous metacharacters (backticks, ``$``, ``()``, ``{}``, ``<>``,
       bare pipes) are rejected outright.
    2. The command may be a chain of sub-commands joined by ``&&`` / ``||``
       / ``;``.  Each sub-command's prefix must match the language allowlist.
    3. Trailing ``&`` (background execution) is rejected.
    """
    cmd = test_command.strip()
    pattern = _ALLOWED_TEST_CMD_PATTERNS.get(language)
    if pattern is None:
        raise ValueError(f"No allowlisted test-command pattern for language {language!r}")
    if _SHELL_INJECTION_RE.search(cmd):
        raise ValueError(f"test_command contains shell metacharacters: {test_command!r}")
    # Reject background execution (``cmd &``) while allowing ``&&`` chains.
    # A lone ``&`` is one that is neither preceded nor followed by ``&``.
    if re.search(r"(?<!&)&(?!&)", cmd):
        raise ValueError(f"test_command contains shell metacharacters: {test_command!r}")
    for step in _CHAIN_SPLIT_RE.split(cmd):
        step = step.strip()
        if not step:
            raise ValueError(f"test_command has empty chain step: {test_command!r}")
        if not pattern.match(step):
            raise ValueError(f"test_command rejected for language {language!r}: {test_command!r}")


def _validate_safe_path(base: Path, name: str) -> Path:
    """Resolve a relative filename against base, rejecting path traversal."""
    base_path = base.resolve()
    p = (base_path / name).resolve()
    try:
        p.relative_to(base_path)
    except ValueError as exc:
        raise ValueError(f"Unsafe file path in task metadata (path traversal): {name!r}") from exc
    return p


def _safe_write(base: Path, name: str, content: str) -> None:
    """Write *content* to ``base / name`` after checking for path traversal.

    Raises ``ValueError`` if the resolved target would escape *base*.
    """
    target = _validate_safe_path(base, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _restore_host_ownership(path: Path, docker_image: str) -> None:
    """Best-effort chown for Docker-created files in the bind-mounted temp tree."""
    if os.name != "posix":
        return
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{path}:/exercise",
                docker_image,
                "bash",
                "-lc",
                f"chown -R {os.getuid()}:{os.getgid()} /exercise",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        log.debug("Failed to restore host ownership for %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Language mappings
# ---------------------------------------------------------------------------

_FENCE_TAGS: dict[str, list[str]] = {
    "python": ["python", "py"],
    "rust": ["rust", "rs"],
    "go": ["go", "golang"],
    "javascript": ["javascript", "js"],
    "java": ["java"],
    "cpp": ["cpp", "c++", "cxx"],
}

_DEFAULT_FILENAMES: dict[str, str] = {
    "python": "solution.py",
    "rust": "src/lib.rs",
    "go": "solution.go",
    "javascript": "solution.js",
    "java": "Solution.java",
    "cpp": "solution.cpp",
}

# ---------------------------------------------------------------------------
# Solution extraction
# ---------------------------------------------------------------------------

# Pattern 1: fenced block with explicit filename comment
_NAMED_FILE_RE = re.compile(
    r"```\w*\s*\n(?://|#)\s*file:\s*(.+?)\n(.*?)\n```",
    re.DOTALL,
)


def extract_solution_files(output: str, *, language: str) -> dict[str, str]:
    """Extract solution file(s) from agent model output.

    Two patterns are tried:
      1. Fenced code blocks with a ``// file: <name>`` or ``# file: <name>``
         comment on the first line.
      2. Fallback: the last fenced code block whose fence tag matches the
         language. A default filename is assigned.

    Returns a mapping of ``{filename: content}``, or an empty dict if nothing
    was found.
    """
    # --- Pattern 1: named files ---
    named: dict[str, str] = {}
    for m in _NAMED_FILE_RE.finditer(output):
        fname = m.group(1).strip()
        content = m.group(2)
        named[fname] = content
    if named:
        return named

    # --- Pattern 2: fallback to last matching fenced block ---
    tags = _FENCE_TAGS.get(language, [language])
    tag_alt = "|".join(re.escape(t) for t in tags)
    pattern = re.compile(
        rf"```(?:{tag_alt})\s*\n(.*?)\n```",
        re.DOTALL,
    )
    matches = pattern.findall(output)
    if not matches:
        return {}

    # Use the last matching block — agents typically refine solutions iteratively,
    # and the final block represents their best attempt
    last = matches[-1]
    default_name = _DEFAULT_FILENAMES.get(language, f"solution.{language}")
    return {default_name: last}


def _drop_protected_solution_files(
    solution_files: dict[str, str],
    *,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Drop solution files whose path collides with an official test or build file.

    Solution files are written to the eval tmpdir AFTER the official
    ``test_files`` / ``build_files``, so an agent-supplied ``CMakeLists.txt``
    (or ``Cargo.toml``, ``package.json``, or a copy of the test file itself)
    would silently replace the official one — e.g. a solution CMakeLists that
    builds only a library target makes ``ctest`` print "No tests were
    found!!!" and exit 0, scoring a vacuous 1.0. Paths are compared after
    ``os.path.normpath`` so ``./CMakeLists.txt`` collides with
    ``CMakeLists.txt``, matching how ``_safe_write`` resolves both to the same
    on-disk target.
    """
    test_files = metadata.get("test_files") if isinstance(metadata.get("test_files"), dict) else {}
    build_files = metadata.get("build_files") if isinstance(metadata.get("build_files"), dict) else {}
    # ``os.path.normpath`` collapses ``./``, ``a/../`` and doubled slashes but
    # does NOT case-fold — it relies on the eval host being case-sensitive
    # (Linux/ext4 Docker), where ``cmakelists.txt`` is a different file than
    # ``CMakeLists.txt`` and so cannot overwrite the official one. On a
    # case-insensitive FS a cased variant could slip past this set.
    protected = {os.path.normpath(str(name)) for name in (*test_files, *build_files)}
    if not protected or not solution_files:
        return solution_files
    kept: dict[str, str] = {}
    dropped: list[str] = []
    for name, content in solution_files.items():
        if os.path.normpath(str(name)) in protected:
            dropped.append(str(name))
        else:
            kept[name] = content
    if dropped:
        log.warning(
            "Dropped solution file(s) colliding with official test/build files: %s",
            sorted(dropped),
        )
    return kept


def _workspace_repo_dir(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if (path / "repo").is_dir():
        path = path / "repo"
    if path.is_dir():
        return path
    return None


def _solution_files_from_workspace(
    *,
    task: TaskSpec,
    language: str,
    runtime_meta: dict[str, Any],
    workspace_dir: Any = None,
) -> dict[str, str]:
    raw_files = runtime_meta.get("workspace_solution_files")
    if isinstance(raw_files, dict):
        files = {
            str(name): content for name, content in raw_files.items() if str(name).strip() and isinstance(content, str)
        }
        if files:
            return files

    repo_dir = _workspace_repo_dir(workspace_dir or runtime_meta.get("host_workspace_repo_dir"))
    if repo_dir is None:
        return {}

    meta = task.metadata or {}
    test_files = meta.get("test_files") if isinstance(meta.get("test_files"), dict) else {}
    build_files = meta.get("build_files") if isinstance(meta.get("build_files"), dict) else {}
    starter_code = meta.get("starter_code") if isinstance(meta.get("starter_code"), dict) else {}
    excluded = set(test_files.keys()) | set(build_files.keys())
    candidates = set(starter_code.keys())
    candidates.add(_DEFAULT_FILENAMES.get(language, f"solution.{language}"))
    files: dict[str, str] = {}
    for name in candidates:
        if not isinstance(name, str) or name in excluded:
            continue
        path = _validate_safe_path(repo_dir, name)
        if path.is_file():
            try:
                files[name] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    return files


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass
class PolyglotHarnessEvaluator:
    """Runs exercism-style tests in a sandboxed Docker container.

    Set ``skip_docker=True`` to unit-test the evaluator without Docker.
    """

    docker_image: str = field(default_factory=lambda: _DEFAULT_POLYGLOT_DOCKER_IMAGE)
    timeout_sec: int = DEFAULT_POLYGLOT_TIMEOUT_SEC
    skip_docker: bool = False

    def _result(self, **fields: Any) -> dict[str, Any]:
        if not self.skip_docker:
            fields["polyglot_environment"] = copy.deepcopy(_polyglot_environment_metadata(self.docker_image))
        return fields

    def evaluate(self, *, task: TaskSpec, model_output: str, **kwargs: Any) -> dict[str, Any]:
        language = (task.metadata or {}).get("language", "python")
        runtime_meta = kwargs.get("runtime_meta") if isinstance(kwargs.get("runtime_meta"), dict) else {}
        # Prefer the workspace files captured by the runtime over fenced blocks
        # in model_output: the agent edits files in place via Edit/Write tools,
        # so the workspace is the source of truth. A stale fenced block from
        # an earlier draft would otherwise score over correctly-edited files.
        solution_files = _solution_files_from_workspace(
            task=task,
            language=language,
            runtime_meta=runtime_meta,
            workspace_dir=kwargs.get("workspace_dir"),
        )
        solution_source = "workspace_files" if solution_files else "model_output"
        if not solution_files:
            solution_files = extract_solution_files(model_output, language=language)

        # Eval-integrity: applied centrally so EVERY solution-file source is
        # covered (raw runtime_meta workspace_solution_files, the repo-dir
        # workspace scan, and model_output extraction) — official test/build
        # files must never be overwritten by agent-supplied content.
        had_files_before_drop = bool(solution_files)
        solution_files = _drop_protected_solution_files(solution_files, metadata=task.metadata or {})
        if had_files_before_drop and not solution_files:
            # Every submitted file collided with an official test/build file
            # and was dropped — report that honestly rather than the pre-drop
            # source, which would misleadingly read as a real submission.
            solution_source = "all_files_protected"

        if not solution_files:
            return self._result(
                status="no_solution",
                instance_id=task.id,
                native_score=0.0,
                resolved=False,
                language=language,
                solution_source=solution_source,
            )

        if self.skip_docker:
            return self._result(
                status="skip_docker",
                instance_id=task.id,
                native_score=0.0,
                resolved=False,
                language=language,
                extracted_files=list(solution_files.keys()),
                solution_source=solution_source,
            )

        result = self._run_in_docker(
            task=task,
            language=language,
            solution_files=solution_files,
        )
        result["solution_source"] = solution_source
        return result

    # ------------------------------------------------------------------
    # Docker execution
    # ------------------------------------------------------------------

    def _run_in_docker(
        self,
        *,
        task: TaskSpec,
        language: str,
        solution_files: dict[str, str],
    ) -> dict[str, Any]:
        tmpdir_root = Path(tempfile.mkdtemp(prefix="polyglot-eval-"))
        try:
            meta = task.metadata or {}
            exercise_name = str(meta.get("exercise_name") or task.id.split("__")[-1])

            # P0-7: Validate exercise_name against path traversal
            _safe_exercise = _validate_safe_path(tmpdir_root, exercise_name)

            # C++ CMakeLists.txt derives project name from the directory name,
            # so we place exercise files in a subdirectory named after the exercise.
            if language == "cpp":
                tmpdir = tmpdir_root / exercise_name
                tmpdir.mkdir(parents=True, exist_ok=True)
            else:
                tmpdir = tmpdir_root

            # Write build / scaffold files provided by the task
            for name, content in meta.get("build_files", {}).items():
                _safe_write(tmpdir, name, content)

            # Write test files
            test_files: dict[str, str] = meta.get("test_files", {})
            for name, content in test_files.items():
                # JavaScript: enable skipped tests (xtest/xit -> test/it)
                if language == "javascript":
                    content = content.replace("xtest(", "test(").replace("xit(", "it(")
                # Java: strip @Disabled annotations so all tests run.
                # Exercism Java marks all but the first test with @Disabled("Remove to run test")
                # which causes Gradle to skip them and still exit 0 — false-positive passes.
                if language == "java":
                    content = re.sub(r"@Disabled(?:\(\"[^\"]*\"\))?\s*\n", "", content)
                _safe_write(tmpdir, name, content)

            # Write solution files (overrides any starter_code with same name)
            for name, content in solution_files.items():
                _safe_write(tmpdir, name, content)

            # Write remaining starter_code not overridden by solution
            for name, content in meta.get("starter_code", {}).items():
                p = _validate_safe_path(tmpdir, name)
                if not p.exists():
                    _safe_write(tmpdir, name, content)

            test_cmd = meta.get("test_command") or self._default_test_command(language)

            # P0-6: Validate dataset-supplied test commands against allowlist
            if meta.get("test_command"):
                _validate_test_command(test_cmd, language)

            # Some languages need a setup step before tests can run
            # (e.g. npm install for JavaScript, gradle wrapper for Java). Wrap
            # the setup step so its OWN exit code is distinguishable from a
            # genuine test failure: a nonzero setup exit (e.g. a transient
            # registry error) short-circuits before the test ever runs, and
            # is chained via `&&` -- so `proc.returncode != 0` alone can't
            # tell which stage failed. Emit a marker on stderr when setup
            # fails so `_run_in_docker` can report a distinct
            # ``status="setup_failed"`` instead of a fabricated ``"ok"``.
            setup_cmd = self._setup_command(language)
            if setup_cmd:
                full_cmd = f"{{ {setup_cmd}; }} || {{ echo '{_SETUP_FAILURE_MARKER}' 1>&2; exit 97; }}; {test_cmd}"
            else:
                full_cmd = test_cmd

            # C++: pass CMake defines for Exercism test infrastructure
            if language == "cpp" and "cmake" in full_cmd:
                full_cmd = full_cmd.replace(
                    "cmake -B build",
                    "cmake -B build -DEXERCISM_TEST_SUITE=1 -DEXERCISM_RUN_ALL_TESTS=1",
                )

            # Network access is needed during setup (npm install, cargo fetch)
            # but disabled would be safer; since exercises are self-contained
            # and the container is ephemeral, we allow network for setup.
            # For C++, mount at a path whose basename matches the exercise name
            # so CMake's get_filename_component derives the correct project name.
            container_workdir = f"/work/{exercise_name}" if language == "cpp" else "/exercise"
            container_name = f"kcsi-polyglot-eval-{_docker_name_component(task.id)}-{uuid.uuid4().hex[:12]}"
            cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--label",
                f"{_POLYGLOT_EVAL_LABEL}=polyglot",
                "--label",
                f"{_POLYGLOT_EVAL_LABEL_ALIAS}=polyglot",
                "-v",
                f"{tmpdir}:{container_workdir}",
                "-w",
                container_workdir,
                self.docker_image,
                "bash",
                "-c",
                full_cmd,
            ]

            log.debug("Polyglot eval command: %s", " ".join(cmd))

            docker_ran = True
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )

            if _SETUP_FAILURE_MARKER in (proc.stderr or ""):
                # Setup itself failed before the test ever ran (e.g. npm
                # install hit a transient registry error) -- not a genuine
                # test verdict. Distinct status so score_polyglot_from_eval
                # gates it to None instead of a fabricated 0.0.
                return self._result(
                    status="setup_failed",
                    instance_id=task.id,
                    native_score=0.0,
                    resolved=False,
                    language=language,
                    test_exit_code=proc.returncode,
                    test_stdout_tail=(proc.stdout or "")[-_TEST_OUTPUT_TAIL_CHARS:],
                    test_stderr_tail=(proc.stderr or "")[-_TEST_OUTPUT_TAIL_CHARS:],
                )

            passed = proc.returncode == 0
            extra: dict[str, Any] = {}
            if passed:
                # Vacuous-pass guards: a runner can exit 0 without any test
                # actually executing (e.g. a solution file overwrote the
                # official test file). Only flip a *pass* to a fail, and only
                # for languages with a reliable "zero tests ran" signal.
                combined = (proc.stdout or "") + (proc.stderr or "")
                if language == "cpp":
                    # Belt-and-suspenders for ctest's exit-0-on-zero-tests
                    # semantics: official Exercism cpp builds run the Catch2
                    # binary at BUILD time (add_custom_target(... COMMAND
                    # ${exercise})), so a legitimate pass prints
                    # "All tests passed (" on stdout while ctest still prints
                    # "No tests were found!!!" on stderr (empirically verified
                    # against every cpp task's build_files in data/*.json). The
                    # ctest marker WITHOUT the Catch2 success marker means no
                    # test ever ran -- a vacuous pass, not a solve.
                    if _CTEST_NO_TESTS_MARKER in combined and _CATCH2_ALL_PASSED_MARKER not in combined:
                        passed = False
                        extra["cpp_vacuous_pass_guard"] = True
                elif language == "go":
                    # "[no test files]" with no "ok\t<pkg>" line means no
                    # package ran any test.
                    if _GO_NO_TEST_FILES_MARKER in combined and not _GO_TESTS_RAN_RE.search(combined):
                        passed = False
                        extra["go_vacuous_pass_guard"] = True
                elif language == "rust":
                    # No suite reported a nonzero passed count -> zero tests
                    # actually ran ("test result: ok. 0 passed").
                    if not _RUST_TESTS_RAN_RE.search(combined):
                        passed = False
                        extra["rust_vacuous_pass_guard"] = True
                elif language == "python":
                    # pytest normally exits 5 on "no tests collected" (caught by
                    # the exit code above); this only fires if the run was forced
                    # to exit 0 while collecting nothing.
                    if any(marker in combined for marker in _PYTEST_NO_TESTS_MARKERS):
                        passed = False
                        extra["python_vacuous_pass_guard"] = True
            return self._result(
                status="ok",
                instance_id=task.id,
                native_score=1.0 if passed else 0.0,
                resolved=passed,
                language=language,
                test_exit_code=proc.returncode,
                test_stdout_tail=(proc.stdout or "")[-_TEST_OUTPUT_TAIL_CHARS:],
                test_stderr_tail=(proc.stderr or "")[-_TEST_OUTPUT_TAIL_CHARS:],
                **extra,
            )

        except subprocess.TimeoutExpired:
            # subprocess.run kills the Docker CLI on timeout, but the container
            # can outlive the client. Reap by deterministic name.
            log.warning("Polyglot eval timed out for %s after %ds", task.id, self.timeout_sec)
            cleanup = _cleanup_docker_container(container_name) if "container_name" in locals() else {}
            return self._result(
                status="timeout",
                instance_id=task.id,
                native_score=0.0,
                resolved=False,
                language=language,
                **cleanup,
            )
        finally:
            if "docker_ran" in locals() and docker_ran:
                _restore_host_ownership(tmpdir_root, self.docker_image)
            shutil.rmtree(tmpdir_root, ignore_errors=True)

    # ------------------------------------------------------------------
    # Default test commands per language
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_command(language: str) -> str:
        """Return a setup command that must run before tests for a given language.

        Handles dependency installation and build scaffolding that exercises
        assume is present but isn't captured in the test command alone.
        """
        setups: dict[str, str] = {
            "javascript": "npm install --silent 2>/dev/null",
            # Gradle needs settings.gradle to avoid scanning for it
            "java": "[ -f settings.gradle ] || echo 'rootProject.name=\"exercise\"' > settings.gradle",
            # C++ Exercism CMake expects a .h file; create an empty one if missing
            "cpp": 'for f in *.cpp; do h="${f%.cpp}.h"; [ -f "$h" ] || touch "$h"; done 2>/dev/null; true',
        }
        return setups.get(language, "")

    @staticmethod
    def _default_test_command(language: str) -> str:
        defaults: dict[str, str] = {
            "python": "python -m pytest -rA --tb=long",
            "rust": "cargo test -- --include-ignored",
            "go": "go test ./...",
            "javascript": "npm test",
            "java": "gradle test",
            "cpp": "cmake -B build && cmake --build build && cd build && ctest --output-on-failure",
        }
        return defaults.get(language, f"echo 'No default test command for {language}'")
