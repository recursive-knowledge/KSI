from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any

from ..benchmarks.terminal_bench_2 import (
    TB2_TIMEOUT_SOURCE,
    TB2_VERIFIER_FAIL_CLOSED_STATUS,
    TerminalBench2TaskContract,
    resolve_terminal_bench_2_task_contract,
)
from ..errors import ContainerRegistryError
from ..models import TaskSpec
from ..prompts import build_execution_prompt
from ..tokens import TokenUsage
from .llm import build_llm_caller
from .seeding import seed_package_to_memory_md
from .swebench_images import _scrub_credentials
from .terminal_bench_2_bridge import (
    _build_tb2_bridge_transcript,
    _tb2_bridge_cache_blocks,
    _tb2_bridge_system_prompt,
    _tb2_bridge_tail,
    _tb2_trim_oldest_history,
)
from .terminal_bench_2_docker import _shorten, _tail


def _docker_name_component(value: str, *, max_chars: int = 48) -> str:
    component = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip())
    component = component.strip("-.")
    return (component[:max_chars].strip("-.") or "task").lower()


def _environment_dir_hash(environment_dir: Path, *, length: int = 12) -> str:
    """Content hash of an environment/ directory tree.

    Same content produces the same digest across runs and processes, so two
    parallel trials on the same task converge on the same image tag and the
    Docker daemon's layer cache can short-circuit the second build.
    """
    h = hashlib.sha256()
    for path in sorted(environment_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(environment_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:length]


def _stable_image_tag(*, environment_dir: Path, safe_task: str) -> str:
    digest = _environment_dir_hash(environment_dir)
    return f"kcsi-tb2-{safe_task}:{digest}"


def _keep_tb2_images_default() -> bool:
    raw = str(os.environ.get("KCSI_TB2_KEEP_IMAGES") or "").strip().lower()
    if raw in {"0", "false", "no"}:
        return False
    return True


def _tb2_require_trusted_verifier() -> bool:
    """Strict-mode gate: fail closed on an untrusted verifier toolchain.

    When enabled, a trial whose trusted-bash injection did NOT take effect
    (``verifier_trusted_toolchain is False``) is left UNSCORED (reward ``None``)
    instead of falling back to the legacy PATH-resolved
    ``bash -c "bash /tests/test.sh"`` invocation. The fallback is "never worse
    than main", but for grader integrity an untrusted verifier toolchain is
    treated as no trustworthy verdict by default.
    An unavailable trusted toolchain surfaces as an unscored trial
    (``runtime_meta.verifier_fail_closed``) rather than a silent legacy
    fallback. Default ON for grader integrity; set
    ``KCSI_TB2_REQUIRE_TRUSTED_VERIFIER=0`` only for legacy comparison runs
    that intentionally preserve the fallback.
    """
    raw = str(os.environ.get("KCSI_TB2_REQUIRE_TRUSTED_VERIFIER") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def default_agent_command(*, agent_mode: str, explicit_command: str | None = None) -> str:
    mode = agent_mode.strip().lower()
    if mode == "oracle":
        return "/bin/bash /solution/solve.sh"
    if mode == "noop":
        return "true"
    if mode == "kcsi":
        return ""
    if mode == "command":
        command = str(explicit_command or "").strip()
        if not command:
            raise ValueError("--agent-mode command requires --agent-command")
        return command
    raise ValueError(f"unsupported agent_mode={agent_mode!r}")


def _run(
    cmd: list[str],
    *,
    timeout_sec: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
    )


def _looks_like_transient_docker_registry_failure(proc: subprocess.CompletedProcess[str]) -> bool:
    text = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
    markers = (
        "tls handshake timeout",
        "failed to fetch oauth token",
        "failed to resolve source metadata",
        "failed to authorize",
        "connection reset by peer",
        "i/o timeout",
        "context deadline exceeded",
        "unexpected eof",
        "temporary failure",
        "command timed out",
        "unauthorized",
        "authentication required",
        "received unexpected http status: 5",
        "internal server error",
        "too many requests",
        "toomanyrequests",
    )
    return any(marker in text for marker in markers)


def _completed_from_timeout(
    *, cmd: list[str], timeout_sec: float, exc: subprocess.TimeoutExpired
) -> subprocess.CompletedProcess[str]:
    stdout = (
        exc.stdout.decode("utf-8", errors="replace")
        if isinstance(exc.stdout, (bytes, bytearray))
        else (exc.stdout or "")
    )
    stderr = (
        exc.stderr.decode("utf-8", errors="replace")
        if isinstance(exc.stderr, (bytes, bytearray))
        else (exc.stderr or "")
    )
    if not stderr:
        stderr = f"command timed out after {timeout_sec:.1f}s"
    return subprocess.CompletedProcess(args=cmd, returncode=124, stdout=stdout, stderr=stderr)


def _docker_build_with_retry(
    cmd: list[str],
    *,
    timeout_sec: float,
    attempts: int = 3,
) -> subprocess.CompletedProcess[str]:
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            proc = _run(cmd, timeout_sec=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            proc = _completed_from_timeout(cmd=cmd, timeout_sec=timeout_sec, exc=exc)
        last = proc
        if proc.returncode == 0:
            return proc
        if attempt >= attempts or not _looks_like_transient_docker_registry_failure(proc):
            return proc
        time.sleep(min(10.0, float(attempt * 2)))
    assert last is not None
    return last


def _inspect_image_identity(
    *,
    pull_target: str,
    image_tag: str,
) -> tuple[str, str]:
    """Return (upstream_digest, local_image_id) for a TB2 image.

    `upstream_digest` is the registry digest (e.g. ``alexgshaw/foo@sha256:abc...``)
    pulled from ``pull_target``'s ``RepoDigests`` — empty string if no upstream
    pull happened (build path) or the registry returned no digest.

    `local_image_id` is the Docker image ID (``Id`` field, e.g.
    ``sha256:def...``) of ``image_tag``. Set whenever we have any image at all;
    used for within-experiment drift detection regardless of provenance.

    Failures during inspect (timeout, malformed JSON, missing image) collapse
    to empty strings rather than raising — digest recording is best-effort
    instrumentation, not a load-bearing trial step.
    """
    upstream_digest = ""
    local_image_id = ""

    if pull_target:
        try:
            proc = _run(["docker", "inspect", pull_target], timeout_sec=10)
            if proc.returncode == 0 and proc.stdout:
                data = json.loads(proc.stdout)
                if isinstance(data, list) and data:
                    repo_digests = data[0].get("RepoDigests") or []
                    if isinstance(repo_digests, list) and repo_digests:
                        upstream_digest = str(repo_digests[0])
        except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
            pass

    if image_tag:
        try:
            proc = _run(["docker", "inspect", image_tag], timeout_sec=10)
            if proc.returncode == 0 and proc.stdout:
                data = json.loads(proc.stdout)
                if isinstance(data, list) and data:
                    local_image_id = str(data[0].get("Id") or "")
        except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
            pass

    return upstream_digest, local_image_id


def _docker_pull_with_retry(
    image: str,
    *,
    timeout_sec: float,
    attempts: int = 3,
) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "pull", image]
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            proc = _run(cmd, timeout_sec=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            proc = _completed_from_timeout(cmd=cmd, timeout_sec=timeout_sec, exc=exc)
        last = proc
        if proc.returncode == 0:
            return proc
        if attempt >= attempts or not _looks_like_transient_docker_registry_failure(proc):
            return proc
        time.sleep(min(10.0, float(attempt * 2)))
    assert last is not None
    return last


def _parse_reward_text(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    # The reward file is written by the AGENT-CONTROLLED container. ``float``
    # accepts the literals ``nan``/``inf``/``-inf`` without raising, so a buggy
    # or adversarial container could write a non-finite reward that would flow
    # through as a solve (``inf >= 1.0``) or corrupt ``native_score`` (``nan``).
    # A non-finite reward is not a genuine score -> treat as unscored (``None``),
    # same as an unparseable file.
    if not math.isfinite(value):
        return None
    return value


def _parse_reward(reward_path: Path) -> float | None:
    if not reward_path.is_file():
        return None
    return _parse_reward_text(reward_path.read_text(encoding="utf-8"))


def _memory_markdown_for_task(task: TaskSpec, seed_package: Any = None, *, raw_mode: bool = False) -> str:
    metadata = task.metadata or {}
    seed_memory_md = seed_package_to_memory_md(
        seed_package,
        current_task_id=task.id,
        task_source="terminal_bench_2",
        raw_mode=raw_mode,
    )
    memory_md = seed_memory_md
    override = metadata.get("memory_md_override")
    if isinstance(override, str) and override.strip():
        memory_md = override.strip()
        if seed_memory_md:
            memory_md += "\n\n## Seed Context\n" + seed_memory_md.strip()
    if memory_md:
        return memory_md.strip() + "\n"
    return ""


def _tools_markdown_for_task(task: TaskSpec) -> str:
    metadata = task.metadata or {}
    override = metadata.get("tools_md_override")
    if isinstance(override, str) and override.strip():
        return override.strip() + "\n"
    return (
        "# TOOLS\n\n"
        "- Work in the current task container and filesystem.\n"
        "- The mounted workspace is for task specification and memory only.\n"
        "- Read the native `tb2/instruction.md` and `tb2/task.toml` files in this workspace.\n"
        "- After that, switch into the real task surface in the container: repo/app paths, config files, services, ports, and build/runtime entrypoints.\n"
        "- Treat verifier-owned assets as hidden unless they are intentionally surfaced for a verification phase.\n"
    )


def _workspace_task_files(task: TaskSpec) -> dict[str, str]:
    metadata = task.metadata or {}
    raw = metadata.get("task_files")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            continue
        name = str(key or "").strip()
        if not name:
            continue
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe TB2 task_files path: {name!r}")
        out[name] = value.strip()
    return out


def materialize_terminal_bench_2_workspace_seed(
    *, task: TaskSpec, output_dir: Path, seed_package: Any = None, raw_mode: bool = False
) -> Path:
    workspace_root = output_dir / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    memory_md = _memory_markdown_for_task(task, seed_package, raw_mode=raw_mode).strip()
    if memory_md:
        (workspace_root / "MEMORY.md").write_text(memory_md + "\n", encoding="utf-8")

    tools_md = _tools_markdown_for_task(task).strip()
    if tools_md:
        (workspace_root / "TOOLS.md").write_text(tools_md + "\n", encoding="utf-8")
    for name, content in _workspace_task_files(task).items():
        path = workspace_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
    return workspace_root


def _agent_phase_copies(*, contract: TerminalBench2TaskContract, agent_mode: str) -> list[tuple[Path, str]]:
    copies: list[tuple[Path, str]] = []
    if agent_mode.strip().lower() == "oracle":
        copies.append((contract.task_root / "solution", "/solution"))
    return copies


def _verifier_phase_copies(*, contract: TerminalBench2TaskContract) -> list[tuple[Path, str]]:
    return [(contract.task_root / "tests", "/tests")]


# The reward the trial scores is read from `/logs/verifier/reward.txt` inside
# the container (host side: `<output_root>/logs/verifier/reward.txt`). `/logs`
# is bind-mounted read-write into the agent's turns, so the agent can write this
# path directly. `ctrf.json` is the CTRF results file the verifier emits beside
# it and which the trial records as an artifact; it is equally agent-writable,
# so it is wiped in the same pre-verifier sanitize.
_TB2_CONTAINER_REWARD_PATH = "/logs/verifier/reward.txt"
_TB2_CONTAINER_CTRF_PATH = "/logs/verifier/ctrf.json"


def _verifier_sanitize_paths(*, contract: TerminalBench2TaskContract) -> list[str]:
    """Container paths that MUST be wiped before the official verifier runs.

    The agent phase and the verifier phase share one long-lived container. Two
    grader-integrity holes let an agent force a vacuous ``resolved=True``:

    1. ``docker cp <host>/tests container:/tests`` does NOT replace an existing
       ``/tests`` directory — if the agent pre-created ``/tests`` during its
       turns, the official tests NEST under ``/tests/tests/`` and the agent's
       own ``/tests/test.sh`` is what ``bash /tests/test.sh`` executes.
    2. The reward is read from ``/logs/verifier/reward.txt`` (and the CTRF
       results recorded as an artifact from ``/logs/verifier/ctrf.json``), and
       ``/logs`` is bind-mounted read-write into the agent's container, so the
       agent can pre-write ``reward.txt=1.0`` / ``ctrf.json``. If the
       real verifier then fails to overwrite them (crash, timeout, or a
       ``test.sh`` that only writes on some paths), the pre-planted value would
       be misread as the verifier's verdict.

    Removing every verifier copy destination plus the reward/CTRF outputs
    immediately before copying/running the official verifier closes both
    *static* holes: the official ``test.sh`` lands authoritatively at
    ``/tests/test.sh`` (a freshly created ``/tests``), and any pre-planted
    reward/CTRF is gone.

    This closes the two *static* holes only. Two dynamic vectors remain KNOWN,
    ACCEPTED residuals for cooperative TB2 (both require running the verifier
    against pristine state — a fresh container / trusted toolchain — currently
    deferred):

    1. *Resident-process replant.* A background daemon the agent detached during
       its turns (e.g. a loop that keeps writing ``reward.txt=1.0``) survives
       ``rm -rf`` and can re-plant the reward after the official ``test.sh``
       writes it. We deliberately do NOT sweep/kill container processes before
       the verifier: an unconditional kill also destroys the agent-STARTED
       services that legitimate task verifiers probe while still running
       (kv-store-grpc's gRPC server, nginx-request-logging's nginx,
       hf-model-inference's Flask API, install-windows-3.11 / qemu-startup /
       qemu-alpine-ssh's VMs — ~6/89 tasks), turning honest solves into
       ``resolved=False`` and diverging from canonical Harbor semantics. The
       false-negative cost to cooperative runs outweighs closing this
       adversarial replant here.
    2. *Trojaned toolchain (ENTRYPOINT hijack hardened, best-effort).*
       The verifier runs in the same container the (potentially root) agent
       controlled, so the agent can trojan interpreters the verifier invokes
       (e.g. shadowing ``bash`` at ``/usr/local/bin/bash`` -- which precedes
       ``/bin`` on PATH -- so ``bash /tests/test.sh`` runs agent code). The
       verifier ENTRYPOINT hijack is now HARDENED (not fully closed): ``test.sh``
       is launched through a verifier-owned bash extracted from the pristine
       image via an ABSOLUTE path with the verifier dir prepended to PATH (see
       ``_extract_trusted_bash_from_image`` / ``_tb2_verifier_command``), so a
       static ``/usr/local/bin/bash`` shim is neither the entrypoint interpreter
       nor the first ``bash`` on PATH -- and any bare ``bash`` inside ``test.sh``
       also resolves to the trusted binary. The verifier sanitize/setup helpers
       use an absolute container shell and absolute ``rm``/``mkdir`` paths, so a
       PATH-shadowed ``/usr/local/bin/bash`` or ``rm`` cannot intercept those
       steps. RESIDUAL that remains open: ``test.sh``'s own internal
       subprocess calls (bare ``python3``/``node``/``go``/``sh``/...) still
       resolve against the agent-controlled filesystem. Fully closing both needs
       a pristine verifier container, deferred because it risks
       breaking tasks whose checks legitimately inspect container/system state.
    """
    paths = [dst for _, dst in _verifier_phase_copies(contract=contract)]
    for extra in (_TB2_CONTAINER_REWARD_PATH, _TB2_CONTAINER_CTRF_PATH):
        if extra not in paths:
            paths.append(extra)
    return paths


# Settle window (seconds) between the two reward samples taken *inside* the
# container in `_read_tb2_reward_before_removal`. Must exceed 1s so that a
# background process rewriting `reward.txt` in a tight loop advances the file's
# integer-second mtime (`stat -c %Y`) across the two reads, letting us detect it
# and fail closed even when it always writes the same value.
_TB2_REWARD_SETTLE_SEC = 1.2
# Sentinel embedded in the reward-readout `docker exec` command so tests (and
# log audits) can recognize it distinctly from the sanitize `rm -rf`.
_TB2_REWARD_READOUT_SENTINEL = "kcsi-tb2-reward-readout"


@dataclass
class _TB2RewardReadout:
    """Reward captured from INSIDE the still-running container via ``docker
    exec``, BEFORE ``docker rm -f``."""

    reward: float | None
    present: bool
    active_writer: bool
    raw: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "present": self.present,
            "active_writer": self.active_writer,
            "raw": self.raw,
            "detail": self.detail,
        }


def _read_tb2_reward_before_removal(
    *,
    container_name: str,
    reward_container_path: str = _TB2_CONTAINER_REWARD_PATH,
    settle_sec: float = _TB2_REWARD_SETTLE_SEC,
    timeout_sec: float = 30.0,
) -> _TB2RewardReadout:
    """Read ``reward.txt`` from inside the container via ``docker exec`` right
    after the official ``test.sh`` returns and BEFORE the container is removed.

    ``/logs`` is bind-mounted read-write for the container's whole life and the
    agent runs as root, so a writer the agent backgrounded during its own phase
    (e.g. ``nohup bash -c 'while :; do echo 1 > /logs/verifier/reward.txt; done'
    &``) survives the one-shot pre-verifier ``rm -rf`` purge and keeps
    overwriting the grader's verdict throughout the entire verifier window. The
    host previously read the mounted reward only AFTER ``docker rm -f``, so that
    racing writer's last write could force a false pass. Reading here
    pins the recorded value to a controlled snapshot the grader itself produced.

    To also catch a writer that is *still* racing at snapshot time, two samples
    of (mtime, content) are taken ``settle_sec`` apart; if either changes, the
    reward is untrusted -> ``active_writer=True`` and the caller records no score
    (fail closed). A legitimate verifier has finished writing by the time
    ``test.sh`` returns, so its reward file is stable across the window and
    scores normally.
    """
    sep = "\x1e"
    quoted = shlex.quote(reward_container_path)
    script = (
        f"# {_TB2_REWARD_READOUT_SENTINEL}\n"
        f"r={quoted}\n"
        'if [ ! -f "$r" ]; then printf ABSENT; exit 0; fi\n'
        'm1=$(stat -c %Y "$r" 2>/dev/null || echo 0)\n'
        'v1=$(tr -d "\\r\\n" < "$r" 2>/dev/null)\n'
        f"sleep {settle_sec}\n"
        'm2=$(stat -c %Y "$r" 2>/dev/null || echo 0)\n'
        'v2=$(tr -d "\\r\\n" < "$r" 2>/dev/null)\n'
        f'printf "PRESENT{sep}%s{sep}%s{sep}%s{sep}%s" "$m1" "$m2" "$v1" "$v2"\n'
    )
    try:
        proc = _docker_exec(container_name=container_name, command=script, timeout_sec=timeout_sec)
    except subprocess.TimeoutExpired:
        return _TB2RewardReadout(reward=None, present=False, active_writer=True, detail="reward readout timed out")
    if proc.returncode != 0:
        return _TB2RewardReadout(
            reward=None,
            present=False,
            active_writer=True,
            detail=f"reward readout exec failed rc={proc.returncode}: {_tail(proc.stderr or '')}",
        )
    stdout = proc.stdout or ""
    if stdout.startswith("ABSENT"):
        return _TB2RewardReadout(reward=None, present=False, active_writer=False, detail="reward file absent")
    if not stdout.startswith("PRESENT"):
        return _TB2RewardReadout(reward=None, present=False, active_writer=True, detail="unrecognized reward readout")
    fields = stdout.split(sep)
    if len(fields) < 5:
        return _TB2RewardReadout(reward=None, present=True, active_writer=True, detail="unparseable reward readout")
    _, m1, m2, v1, v2 = fields[:5]
    if m1 != m2 or v1 != v2:
        return _TB2RewardReadout(
            reward=None,
            present=True,
            active_writer=True,
            raw=v1,
            detail=f"reward mutated during verify window (m1={m1} m2={m2} v1={v1!r} v2={v2!r})",
        )
    return _TB2RewardReadout(reward=_parse_reward_text(v1), present=True, active_writer=False, raw=v1)


@dataclass(frozen=True)
class TerminalBench2TrialResult:
    task_id: str
    task_root: str
    image_tag: str
    container_name: str
    agent_command: str
    agent_exit_code: int | None
    verifier_exit_code: int | None
    reward: float | None
    resolved: bool
    output_dir: str
    runtime_meta: dict[str, Any]
    model_output: str = ""
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_root": self.task_root,
            "image_tag": self.image_tag,
            "container_name": self.container_name,
            "agent_command": self.agent_command,
            "agent_exit_code": self.agent_exit_code,
            "verifier_exit_code": self.verifier_exit_code,
            "reward": self.reward,
            "resolved": self.resolved,
            "output_dir": self.output_dir,
            "runtime_meta": self.runtime_meta,
            "model_output": self.model_output,
            "tool_trace": self.tool_trace,
            "token_usage": self.token_usage.to_dict(),
        }


@dataclass(frozen=True)
class TerminalBench2AgentRunResult:
    model_output: str
    tool_trace: list[dict[str, Any]]
    token_usage: TokenUsage
    error_text: str = ""
    transcript: str = ""


_TB2_VALID_ACTIONS = {"shell", "read", "write", "edit", "glob", "grep", "final"}


def _extract_json_object(text: str) -> dict[str, Any]:
    def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if "action" in payload:
            action_value = str(payload.get("action") or "").strip().lower()
            if action_value in _TB2_VALID_ACTIONS:
                normalized = dict(payload)
                normalized["action"] = action_value
                return normalized
            return payload
        # Back-compat shorthand: a payload with just `command` is a shell action.
        command = str(payload.get("command") or "").strip()
        if command:
            normalized = dict(payload)
            normalized["action"] = "shell"
            normalized["command"] = command
            try:
                normalized["timeout_sec"] = float(normalized.get("timeout_sec") or 60.0)
            except (TypeError, ValueError):
                normalized["timeout_sec"] = 60.0
            normalized["summary"] = (
                str(normalized.get("summary") or "").strip() or "Execute the requested shell command."
            )
            return normalized
        summary = str(payload.get("summary") or payload.get("final") or "").strip()
        if summary and not any(k in payload for k in ("path", "pattern")):
            normalized = dict(payload)
            normalized["action"] = "final"
            normalized["summary"] = summary
            return normalized
        return payload

    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])
    fallback_payload: dict[str, Any] | None = None
    # strict=False tolerates literal control chars (unescaped newlines/tabs)
    # inside string values. Models routinely emit write/edit/shell actions whose
    # content/command field is a real multi-line code block rather than a
    # \n-escaped string; strict json.loads rejects those, and the brace-scan
    # salvage below then latches onto an inner "{...}" fragment (e.g. a Python
    # dict literal or f-string in the code body) that has no "action" key,
    # surfacing the misleading "Unsupported TB2 bridge action: (missing)" error.
    # Retrying with a lenient decoder recovers the real action. strict=False
    # parses all otherwise-valid JSON identically, so existing payloads are
    # unaffected.
    lenient_decoder = json.JSONDecoder(strict=False)

    def _load_candidate(candidate: str) -> Any:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return lenient_decoder.decode(candidate)

    for candidate in candidates:
        try:
            payload = _load_candidate(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payload = _normalize_payload(payload)
            if "action" in payload:
                return payload
            fallback_payload = fallback_payload or payload
    decoder = lenient_decoder
    for match in re.finditer(r"\{", raw):
        try:
            payload, _ = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payload = _normalize_payload(payload)
            if "action" in payload:
                return payload
            fallback_payload = fallback_payload or payload

    def _extract_wrapped_value(name: str) -> str:
        match = re.search(
            rf'<(?:parameter|invoke) name="{re.escape(name)}">(.*?)(?:</parameter>|</invoke>|<parameter name=|<invoke name=|</function_calls>|$)',
            raw,
            flags=re.DOTALL,
        )
        return unescape(match.group(1).strip()) if match else ""

    command = _extract_wrapped_value("command")
    if command:
        timeout_raw = _extract_wrapped_value("timeout_sec")
        summary = _extract_wrapped_value("summary")
        try:
            timeout_sec = float(timeout_raw) if timeout_raw else 60.0
        except ValueError:
            timeout_sec = 60.0
        return {
            "action": "shell",
            "command": command,
            "timeout_sec": timeout_sec,
            "summary": summary or "Execute the extracted shell command from the wrapped model response.",
        }
    if fallback_payload is not None:
        return fallback_payload
    raise ValueError(f"response is not a JSON object: {_shorten(raw, 1200)}")


def _docker_run_command(
    *,
    image_tag: str,
    container_name: str,
    logs_root: Path,
    workspace_root: Path,
    cpus: float | None = None,
    memory: str = "",
) -> list[str]:
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "-v",
        f"{logs_root}:/logs",
        "-v",
        f"{workspace_root}:/workspace/task/workspace",
    ]
    if cpus is not None:
        cmd.extend(["--cpus", str(cpus)])
    if memory:
        cmd.extend(["--memory", memory])
    cmd.extend(
        [
            image_tag,
            "bash",
            "-lc",
            "trap exit TERM INT; while true; do sleep 1; done",
        ]
    )
    return cmd


def _docker_cp_to_container(
    *, src: Path, container_name: str, dst: str, timeout_sec: float = 30
) -> subprocess.CompletedProcess[str]:
    return _run(
        ["docker", "cp", str(src), f"{container_name}:{dst}"],
        timeout_sec=timeout_sec,
    )


_TB2_CONTAINER_BASH = "/bin/bash"


_TB2_CONTAINER_RM = "/bin/rm"
_TB2_CONTAINER_MKDIR = "/bin/mkdir"


def _docker_exec(
    *,
    container_name: str,
    command: str,
    timeout_sec: float,
) -> subprocess.CompletedProcess[str]:
    return _run(
        ["docker", "exec", container_name, _TB2_CONTAINER_BASH, "-c", command],
        timeout_sec=timeout_sec,
    )


# Grader integrity (trojaned toolchain). The official verifier runs
# inside the SAME long-lived container the (root) agent controlled, so the agent
# can shadow the shell that launches the official ``test.sh``. On the default
# PATH ``/usr/local/bin`` precedes ``/usr/bin``/``/bin``, so a planted
# ``/usr/local/bin/bash`` would hijack the confirmed
# ``docker exec ... bash -c "bash /tests/test.sh"`` entrypoint and run agent code
# instead of the official verifier (empirically confirmed shim probe).
#
# Mitigation (trusted-toolchain injection, the
# lower-risk option): extract ``bash`` from the PRISTINE image (built/pulled
# before the agent phase, so it cannot have been trojaned; the same image the
# container runs, so it is libc/arch compatible) and docker-cp it into a fresh,
# randomly-named, verifier-owned directory in the running container, then launch
# ``test.sh`` through that ABSOLUTE trusted interpreter with the verifier
# directory prepended to PATH. A planted ``/usr/local/bin/bash`` is then neither
# the entrypoint interpreter (absolute path) nor the first ``bash`` on PATH.
#
# The verifier dir is prepended to PATH, so a bare ``bash`` inside ``test.sh``
# also resolves to the trusted binary. This HARDENS, but does not fully close,
# the ENTRYPOINT hijack. Verifier sanitize/setup uses an absolute container
# shell plus absolute ``rm``/``mkdir`` paths, so a PATH-shadowed
# ``/usr/local/bin/bash`` or ``rm`` cannot intercept those steps. RESIDUAL
# (documented, NOT closed here): ``test.sh``'s own internal subprocess calls
# (bare ``python3``/``node``/``go``/``sh``/...) still resolve against the in-container
# filesystem the agent controlled, so a trojaned language runtime the verifier
# itself invokes is still reachable. Fully closing both needs a pristine verifier
# container, deliberately deferred because it risks breaking
# tasks whose checks legitimately inspect container/system state.
_TB2_TRUSTED_BIN_ROOT = "/tmp/kcsi-verifier-bin"
_TB2_VERIFIER_TEST_SCRIPT = "/tests/test.sh"


def _extract_trusted_bash_from_image(
    *,
    image_tag: str,
    dest_host_path: Path,
    timeout_sec: float = 60.0,
) -> tuple[bool, str]:
    """Copy ``/bin/bash`` out of the PRISTINE image to ``dest_host_path``.

    Uses a throwaway ``docker create`` (never started, so no agent code runs)
    plus ``docker cp`` from that container's ``/bin/bash``. Because it reads the
    image (not the agent-controlled running container), the extracted binary is
    untampered; because it is the same image the trial runs, it is guaranteed
    libc/arch compatible when injected back.

    Returns ``(ok, detail)``. Best-effort: any failure returns ``(False, ...)``
    so the caller can fall back to the legacy invocation rather than raising.
    """
    if not image_tag:
        return False, "no image tag to extract trusted bash from"
    tmp_name = f"kcsi-tb2-trustedbash-{uuid.uuid4().hex[:10]}"
    try:
        create = _run(["docker", "create", "--name", tmp_name, image_tag, "true"], timeout_sec=timeout_sec)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, _tail(str(exc))
    if create.returncode != 0:
        return False, _tail(create.stderr or create.stdout or "docker create failed")
    try:
        try:
            cp = _run(
                ["docker", "cp", f"{tmp_name}:/bin/bash", str(dest_host_path)],
                timeout_sec=timeout_sec,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, _tail(str(exc))
    finally:
        try:
            _run(["docker", "rm", "-f", tmp_name], timeout_sec=20)
        except (subprocess.TimeoutExpired, OSError):
            pass
    if cp.returncode != 0:
        return False, _tail(cp.stderr or cp.stdout or "docker cp of /bin/bash failed")
    return True, ""


def _tb2_verifier_command(
    *,
    container_name: str,
    trusted_bash: str,
    path_prefix: str,
) -> list[str]:
    """``docker exec`` argv that runs ``/tests/test.sh`` via a trusted bash.

    The container-side entrypoint is ``trusted_bash`` at an ABSOLUTE path (never
    PATH-resolved, so a planted ``/usr/local/bin/bash`` cannot be it), and it
    execs the official ``test.sh`` through the same absolute interpreter with
    ``path_prefix`` prepended to PATH so any bare ``bash`` inside ``test.sh``
    also resolves to the trusted binary. The rest of the container PATH is
    preserved (``:"$PATH"``) so legitimately agent-installed toolchains a task's
    ``test.sh`` relies on still resolve normally.
    """
    inner = (
        f'export PATH={shlex.quote(path_prefix)}:"$PATH"; '
        f"exec {shlex.quote(trusted_bash)} {shlex.quote(_TB2_VERIFIER_TEST_SCRIPT)}"
    )
    return ["docker", "exec", container_name, trusted_bash, "-c", inner]


def _completed_process_from_exec_exception(
    *,
    command: str,
    timeout_sec: float,
    exc: Exception,
) -> subprocess.CompletedProcess[str]:
    if isinstance(exc, subprocess.TimeoutExpired):
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        message = f"[TB2 bridge] command timed out after {timeout_sec:.1f}s"
        combined_stderr = f"{stderr}\n{message}".strip()
        return subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=124,
            stdout=stdout,
            stderr=combined_stderr,
        )
    return subprocess.CompletedProcess(
        args=["docker", "exec"],
        returncode=1,
        stdout="",
        stderr=str(exc),
    )


_TB2_STEP_CAP_UNLIMITED = 1_000_000


# The kcsi-bridge loop re-sends the full accumulated shell/native history on
# every turn (via cache_blocks) and the step cap defaults to unlimited, so a
# long-running or stuck agent can grow the request past the provider's context
# window. When that happens, drop the oldest history steps and retry the turn
# instead of letting the "prompt is too long" 400 propagate and collapse the
# entire trial (mirrors distill_cross_task's trim-and-retry for the identical
# provider error). Keep at least this many of the most-recent steps.
_TB2_PROMPT_TOO_LONG_RE = re.compile(
    r"prompt is too long|maximum context length|context_length_exceeded|too many tokens",
    re.IGNORECASE,
)
_TB2_PROMPT_TRIM_MAX_RETRIES = 4
_TB2_PROMPT_TRIM_MIN_HISTORY_KEEP = 4


_TB2_NATIVE_OUTPUT_BYTES = 200_000
_TB2_GLOB_LIMIT = 200
_TB2_GREP_LIMIT = 500
# Per-handler timeout ceiling for the 5 native actions. Deadline-derived
# remaining wall-clock is clamped to [5, _TB2_NATIVE_TIMEOUT_CEILING_SEC]
# so a single native action can't burn the whole task budget on a runaway
# grep/find. Scale via KCSI_TB2_NATIVE_TIMEOUT_SCALE (>=1.0) for slow
# disks or large trees.
_TB2_NATIVE_TIMEOUT_CEILING_SEC = 120.0
# Hard cap on `edit` file size. Edit reads the file into Python memory to
# perform string substitution; without a cap, a multi-GB file would blow
# memory. Above this size the agent should use shell + sed/awk.
_TB2_EDIT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def _tb2_native_timeout_ceiling() -> float:
    """Per-handler ceiling, scaled by `KCSI_TB2_NATIVE_TIMEOUT_SCALE` env var.

    Scale floors at 1.0 so the env var can only widen the ceiling, never
    narrow it. Useful for slow disks or very large trees where the default
    120s caps a legitimate grep.
    """
    try:
        scale = float(os.environ.get("KCSI_TB2_NATIVE_TIMEOUT_SCALE", "1.0"))
    except (TypeError, ValueError):
        scale = 1.0
    if scale < 1.0:
        scale = 1.0
    return _TB2_NATIVE_TIMEOUT_CEILING_SEC * scale


def _docker_cp_from_container(
    *,
    container_name: str,
    src: str,
    dst: Path,
    timeout_sec: float = 30,
) -> subprocess.CompletedProcess[str]:
    return _run(
        ["docker", "cp", f"{container_name}:{src}", str(dst)],
        timeout_sec=timeout_sec,
    )


def _tb2_action_result(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    agent_id: str = "",
) -> tuple[dict[str, Any], str]:
    """Build a (history_entry, last_observation) pair for any TB2 action."""
    combined = ((stdout or "") + ("\n" if stdout and stderr else "") + (stderr or "")).strip()
    history_entry = {
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_input": dict(tool_input, agent_id=agent_id) if agent_id else dict(tool_input),
        "tool_output": {
            "exit_code": exit_code,
            "stdout": _shorten(stdout or "", 4000),
            "stderr": _shorten(stderr or "", 4000),
            "combined_output": _shorten(combined, 6000),
        },
    }
    observation = f"Tool {tool_name} exit code: {exit_code}\nOutput:\n{_shorten(combined, 3000)}"
    return history_entry, observation


def _handle_tb2_read(
    *,
    action: dict[str, Any],
    container_name: str,
    deadline: float,
    agent_id: str,
) -> tuple[dict[str, Any], str]:
    path = str(action.get("path") or "").strip()
    summary = str(action.get("summary") or "").strip()
    tool_input: dict[str, Any] = {"path": path, "summary": summary}
    if not path:
        return _tb2_action_result(
            tool_name="tb2_read",
            tool_input=tool_input,
            exit_code=2,
            stderr="read action requires 'path' (absolute path inside the container).",
            agent_id=agent_id,
        )
    try:
        offset = max(1, int(action.get("offset") or 1))
    except (TypeError, ValueError):
        offset = 1
    try:
        limit = max(1, min(int(action.get("limit") or 2000), 2000))
    except (TypeError, ValueError):
        limit = 2000
    tool_input["offset"] = offset
    tool_input["limit"] = limit
    end_line = offset + limit - 1
    quoted = shlex.quote(path)
    cmd = (
        f"if [ -d {quoted} ]; then echo '__tb2_read_error__: path is a directory' >&2; exit 21; "
        f"elif [ ! -e {quoted} ]; then echo '__tb2_read_error__: no such file' >&2; exit 2; "
        f"else awk -v s={offset} -v e={end_line} 'NR>=s && NR<=e' {quoted} "
        f"| head -c {_TB2_NATIVE_OUTPUT_BYTES}; fi"
    )
    timeout_sec = max(5.0, min(_tb2_native_timeout_ceiling(), max(1.0, deadline - time.monotonic())))
    try:
        proc = _docker_exec(container_name=container_name, command=cmd, timeout_sec=timeout_sec)
    except Exception as exc:
        proc = _completed_process_from_exec_exception(command=cmd, timeout_sec=timeout_sec, exc=exc)
    return _tb2_action_result(
        tool_name="tb2_read",
        tool_input=tool_input,
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        agent_id=agent_id,
    )


def _handle_tb2_write(
    *,
    action: dict[str, Any],
    container_name: str,
    deadline: float,
    agent_id: str,
) -> tuple[dict[str, Any], str]:
    path = str(action.get("path") or "").strip()
    content = action.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = json.dumps(content)
    summary = str(action.get("summary") or "").strip()
    tool_input = {
        "path": path,
        "summary": summary,
        "content_bytes": len(content.encode("utf-8", errors="replace")),
    }
    if not path:
        return _tb2_action_result(
            tool_name="tb2_write",
            tool_input=tool_input,
            exit_code=2,
            stderr="write action requires 'path' (absolute path inside the container).",
            agent_id=agent_id,
        )
    timeout_sec = max(5.0, min(_tb2_native_timeout_ceiling(), max(1.0, deadline - time.monotonic())))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tb2write", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        # Ensure parent directory exists inside the container.
        parent = os.path.dirname(path) or "/"
        mkdir_proc = _docker_exec(
            container_name=container_name,
            command=f"mkdir -p {shlex.quote(parent)}",
            timeout_sec=min(15.0, timeout_sec),
        )
        if mkdir_proc.returncode != 0:
            return _tb2_action_result(
                tool_name="tb2_write",
                tool_input=tool_input,
                exit_code=mkdir_proc.returncode,
                stdout=mkdir_proc.stdout or "",
                stderr=mkdir_proc.stderr or f"failed to mkdir parent {parent!r}",
                agent_id=agent_id,
            )
        cp_proc = _run(
            ["docker", "cp", str(tmp_path), f"{container_name}:{path}"],
            timeout_sec=timeout_sec,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    if cp_proc.returncode == 0:
        stdout = f"Wrote {tool_input['content_bytes']} bytes to {path}."
    else:
        stdout = cp_proc.stdout or ""
    return _tb2_action_result(
        tool_name="tb2_write",
        tool_input=tool_input,
        exit_code=cp_proc.returncode,
        stdout=stdout,
        stderr=cp_proc.stderr or "",
        agent_id=agent_id,
    )


def _handle_tb2_edit(
    *,
    action: dict[str, Any],
    container_name: str,
    deadline: float,
    agent_id: str,
) -> tuple[dict[str, Any], str]:
    path = str(action.get("path") or "").strip()
    old_string = action.get("old_string")
    new_string = action.get("new_string")
    if not isinstance(old_string, str):
        old_string = "" if old_string is None else json.dumps(old_string)
    if not isinstance(new_string, str):
        new_string = "" if new_string is None else json.dumps(new_string)
    replace_all = bool(action.get("replace_all") or False)
    summary = str(action.get("summary") or "").strip()
    tool_input = {
        "path": path,
        "summary": summary,
        "old_string_len": len(old_string),
        "new_string_len": len(new_string),
        "replace_all": replace_all,
    }
    if not path:
        return _tb2_action_result(
            tool_name="tb2_edit",
            tool_input=tool_input,
            exit_code=2,
            stderr="edit action requires 'path' (absolute path inside the container).",
            agent_id=agent_id,
        )
    if not old_string:
        return _tb2_action_result(
            tool_name="tb2_edit",
            tool_input=tool_input,
            exit_code=2,
            stderr="edit action requires non-empty 'old_string'.",
            agent_id=agent_id,
        )
    if old_string == new_string:
        return _tb2_action_result(
            tool_name="tb2_edit",
            tool_input=tool_input,
            exit_code=2,
            stderr="edit action: old_string and new_string are identical; nothing to do.",
            agent_id=agent_id,
        )
    timeout_sec = max(5.0, min(_tb2_native_timeout_ceiling(), max(1.0, deadline - time.monotonic())))
    with tempfile.TemporaryDirectory(prefix="tb2-edit-") as td:
        local = Path(td) / "file"
        cp_out = _docker_cp_from_container(
            container_name=container_name,
            src=path,
            dst=local,
            timeout_sec=min(60.0, timeout_sec),
        )
        if cp_out.returncode != 0:
            return _tb2_action_result(
                tool_name="tb2_edit",
                tool_input=tool_input,
                exit_code=cp_out.returncode,
                stdout=cp_out.stdout or "",
                stderr=cp_out.stderr or f"failed to read {path!r} from container",
                agent_id=agent_id,
            )
        try:
            file_size = local.stat().st_size
        except OSError as exc:
            return _tb2_action_result(
                tool_name="tb2_edit",
                tool_input=tool_input,
                exit_code=2,
                stderr=f"failed to stat copied file for {path!r}: {exc}",
                agent_id=agent_id,
            )
        if file_size > _TB2_EDIT_MAX_BYTES:
            return _tb2_action_result(
                tool_name="tb2_edit",
                tool_input=tool_input,
                exit_code=25,
                stderr=(
                    f"edit failed: {path} is {file_size} bytes, exceeding the "
                    f"{_TB2_EDIT_MAX_BYTES}-byte cap. Use the shell action with "
                    "`sed -i` / `awk` for files this large; edit loads the whole "
                    "file into memory."
                ),
                agent_id=agent_id,
            )
        try:
            # Decode bytes explicitly rather than read_text(): universal-newline
            # translation would silently rewrite CRLF -> LF on read, making any
            # old_string containing "\r\n" unmatchable and clobbering line
            # endings on write-back. bytes.decode raises the same
            # UnicodeDecodeError, so the non-UTF-8 guard below still fires.
            text = local.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            return _tb2_action_result(
                tool_name="tb2_edit",
                tool_input=tool_input,
                exit_code=22,
                stderr=f"file {path!r} is not valid UTF-8; use shell action for binary edits.",
                agent_id=agent_id,
            )
        occurrences = text.count(old_string)
        if occurrences == 0:
            return _tb2_action_result(
                tool_name="tb2_edit",
                tool_input=tool_input,
                exit_code=23,
                stderr=(
                    f"edit failed: old_string not found in {path}. "
                    "Provide the exact substring to match (whitespace-sensitive)."
                ),
                agent_id=agent_id,
            )
        if occurrences > 1 and not replace_all:
            return _tb2_action_result(
                tool_name="tb2_edit",
                tool_input=tool_input,
                exit_code=24,
                stderr=(
                    f"edit failed: old_string occurs {occurrences} times in {path}. "
                    "Either include more surrounding context to make it unique or set replace_all=true."
                ),
                agent_id=agent_id,
            )
        if replace_all:
            new_text = text.replace(old_string, new_string)
            replacements = occurrences
        else:
            new_text = text.replace(old_string, new_string, 1)
            replacements = 1
        # Write bytes (not write_text) to preserve CRLF / exact line endings.
        local.write_bytes(new_text.encode("utf-8"))
        cp_in = _run(
            ["docker", "cp", str(local), f"{container_name}:{path}"],
            timeout_sec=min(60.0, timeout_sec),
        )
        if cp_in.returncode != 0:
            return _tb2_action_result(
                tool_name="tb2_edit",
                tool_input=tool_input,
                exit_code=cp_in.returncode,
                stdout=cp_in.stdout or "",
                stderr=cp_in.stderr or f"failed to write edited {path!r} back to container",
                agent_id=agent_id,
            )
    tool_input["replacements"] = replacements
    return _tb2_action_result(
        tool_name="tb2_edit",
        tool_input=tool_input,
        exit_code=0,
        stdout=f"Edited {path}: {replacements} replacement(s).",
        agent_id=agent_id,
    )


def _handle_tb2_glob(
    *,
    action: dict[str, Any],
    container_name: str,
    deadline: float,
    agent_id: str,
) -> tuple[dict[str, Any], str]:
    pattern = str(action.get("pattern") or "").strip()
    path = str(action.get("path") or "/").strip() or "/"
    summary = str(action.get("summary") or "").strip()
    tool_input = {"pattern": pattern, "path": path, "summary": summary}
    if not pattern:
        return _tb2_action_result(
            tool_name="tb2_glob",
            tool_input=tool_input,
            exit_code=2,
            stderr="glob action requires 'pattern' (e.g. '*.py').",
            agent_id=agent_id,
        )
    quoted_path = shlex.quote(path)
    # Existence check first (mirrors the read handler): a nonexistent path must
    # surface as a real error (exit 2), not be silently indistinguishable from
    # "no matches". `2>/dev/null` on find itself still swallows per-dir
    # permission noise; the `head` cap bounds the output.
    cmd = (
        f"if [ ! -e {quoted_path} ]; then echo '__tb2_glob_error__: no such path' >&2; exit 2; fi; "
        f"find {quoted_path} -type f -name {shlex.quote(pattern)} "
        f"2>/dev/null | head -n {_TB2_GLOB_LIMIT}"
    )
    timeout_sec = max(5.0, min(_tb2_native_timeout_ceiling(), max(1.0, deadline - time.monotonic())))
    try:
        proc = _docker_exec(container_name=container_name, command=cmd, timeout_sec=timeout_sec)
    except Exception as exc:
        proc = _completed_process_from_exec_exception(command=cmd, timeout_sec=timeout_sec, exc=exc)
    return _tb2_action_result(
        tool_name="tb2_glob",
        tool_input=tool_input,
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        agent_id=agent_id,
    )


def _handle_tb2_grep(
    *,
    action: dict[str, Any],
    container_name: str,
    deadline: float,
    agent_id: str,
) -> tuple[dict[str, Any], str]:
    pattern = str(action.get("pattern") or "").strip()
    path = str(action.get("path") or "/").strip() or "/"
    output_mode = str(action.get("output_mode") or "files_with_matches").strip().lower()
    summary = str(action.get("summary") or "").strip()
    tool_input = {
        "pattern": pattern,
        "path": path,
        "output_mode": output_mode,
        "summary": summary,
    }
    if not pattern:
        return _tb2_action_result(
            tool_name="tb2_grep",
            tool_input=tool_input,
            exit_code=2,
            stderr="grep action requires 'pattern'.",
            agent_id=agent_id,
        )
    if output_mode not in {"content", "files_with_matches", "count"}:
        output_mode = "files_with_matches"
        tool_input["output_mode"] = output_mode
    # `-s` suppresses noisy per-file "permission denied"/"no such file" errors
    # (which are common when grepping at `/`), but DOES NOT silence real grep
    # errors (e.g. invalid regex → exit 2 with a usable stderr message). The
    # previous shell-level `2>/dev/null` discarded both classes uniformly,
    # collapsing real errors into mysterious exit 2 with empty stderr.
    if output_mode == "files_with_matches":
        flags = "-rlEs"
    elif output_mode == "count":
        flags = "-rcEs"
    else:
        flags = "-rnEs"
    # `; exit ${PIPESTATUS[0]}` forwards grep's OWN exit code instead of the
    # final `head`'s (always 0). PIPESTATUS refers to the immediately preceding
    # pipeline in bash, so this captures grep's status before any other command
    # runs. Without it, grep's 1 (no match) / 2 (real error) are masked and the
    # remaps below would be dead code.
    cmd = (
        f"grep {flags} {shlex.quote(pattern)} {shlex.quote(path)} "
        f"| head -n {_TB2_GREP_LIMIT} | head -c {_TB2_NATIVE_OUTPUT_BYTES}"
        f"; exit ${{PIPESTATUS[0]}}"
    )
    timeout_sec = max(5.0, min(_tb2_native_timeout_ceiling(), max(1.0, deadline - time.monotonic())))
    try:
        proc = _docker_exec(container_name=container_name, command=cmd, timeout_sec=timeout_sec)
    except Exception as exc:
        proc = _completed_process_from_exec_exception(command=cmd, timeout_sec=timeout_sec, exc=exc)
    # grep exit codes (now reachable thanks to the PIPESTATUS forwarding above):
    #   0 -> matches found
    #   1 -> no matches (normal, surface as exit 0 + helpful stdout)
    #   2 -> real error (invalid regex, unreadable files, etc.). Usually a hard
    #        failure to preserve. EXCEPTION: recursive grep returns 2 when SOME
    #        files/dirs are unreadable even though it found matches (common when
    #        path defaults to "/"); if stdout is non-empty those are partial
    #        results, not a failure, so remap to 0.
    exit_code = proc.returncode
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if exit_code == 1 and not stdout.strip() and not stderr.strip():
        exit_code = 0
        stdout = "(no matches)"
    elif exit_code == 2 and stdout.strip():
        exit_code = 0
    return _tb2_action_result(
        tool_name="tb2_grep",
        tool_input=tool_input,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        agent_id=agent_id,
    )


def _resolve_tb2_max_steps(
    *,
    contract: TerminalBench2TaskContract,
    env: dict[str, str],
) -> int:
    # Default is unlimited: the canonical Harbor harness has no step cap and
    # actively warns when callers limit `max_turns` below Terminus 2's
    # 1,000,000-episode default. The TB2 contract is wall-time only, declared
    # per task in task.toml [agent].timeout_sec. KCSI_TB2_MAX_STEPS=<N>
    # opts into a step cap for CI smoke tests; KCSI_TB2_MAX_STEPS=0 is the
    # explicit "unlimited" sentinel. Negative values are rejected (fall
    # through to default) rather than silently treated as unlimited.
    # Read from the provider profile first, then fall back to os.environ (like
    # every other KCSI_TB2_* knob), so a process-level export takes effect
    # even when it isn't threaded through the provider env.
    raw = str(env.get("KCSI_TB2_MAX_STEPS") or os.environ.get("KCSI_TB2_MAX_STEPS") or "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            pass
        else:
            if value == 0:
                return _TB2_STEP_CAP_UNLIMITED
            if value > 0:
                return value
    return _TB2_STEP_CAP_UNLIMITED


def _run_kcsi_agent_in_tb2_container(
    *,
    task: TaskSpec,
    contract: TerminalBench2TaskContract,
    container_name: str,
    workspace_root: Path,
    provider_env: dict[str, str] | None,
    generation: int,
    agent_id: str,
    seed_package: Any = None,
    raw_mode: bool = False,
) -> TerminalBench2AgentRunResult:
    env = {k: str(v) for k, v in (provider_env or {}).items() if str(k or "").strip()}
    provider = str(env.get("MODEL_PROVIDER") or os.environ.get("MODEL_PROVIDER") or "anthropic").strip().lower()
    model = str(env.get("MODEL") or os.environ.get("MODEL") or "").strip()
    if not model:
        raise ValueError("TB2 kcsi bridge requires MODEL in the provider environment.")

    max_steps = _resolve_tb2_max_steps(contract=contract, env=env)
    deadline = time.monotonic() + max(30.0, float(contract.agent_timeout_sec))
    history: list[dict[str, Any]] = []
    prompt_history: list[dict[str, Any]] = []
    final_output = ""
    error_text = ""
    last_observation = "No commands have run yet. Start by reading the native TB2 instruction and task metadata in the mounted workspace."
    aggregate_usage = TokenUsage()
    invalid_response_count = 0
    execution_prompt = build_execution_prompt(
        task,
        has_memory=bool(_memory_markdown_for_task(task, seed_package, raw_mode=raw_mode).strip()),
        generation=generation,
    )
    api_key = ""
    if provider == "openai":
        api_key = str(env.get("OPENAI_API_KEY") or "").strip()
    else:
        api_key = str(env.get("ANTHROPIC_API_KEY") or "").strip()
    caller = build_llm_caller(
        provider=provider,
        model=model,
        max_tokens=2048,
        reasoning_effort=env.get("REASONING_EFFORT"),
        api_key=api_key or None,
        temperature=0.0,
    )
    for step_index in range(1, max_steps + 1):
        remaining = int(max(1, deadline - time.monotonic()))
        if remaining <= 1:
            error_text = "TB2 kcsi bridge ran out of agent timeout before requesting another action."
            break
        # Append-only stable blocks (header + per-step history) go through
        # ``cache_blocks`` so the accumulated prefix is cache-READ each turn;
        # the per-turn varying tail (step counter + latest observation) stays
        # after the moving cache breakpoint. This changes
        # the exact prompt TB2's model sees (the step counter moved to the
        # tail), so TB2 solve rate should be re-validated before trusting it.
        # Trim-and-retry guard: if the accumulated prompt history overflows the
        # provider context window, drop the oldest prompt steps and retry THIS turn
        # rather than letting the "prompt is too long" error propagate to the
        # outer catch and collapse the whole trial (forfeiting all prior spend).
        # Keep the persisted audit history separate: trimming prompt_history
        # must not delete tool_trace entries or submission iteration counts.
        # Mirrors distill_cross_task's trim-and-retry for the identical error.
        resp = None
        for _trim_attempt in range(_TB2_PROMPT_TRIM_MAX_RETRIES + 1):
            cache_blocks = _tb2_bridge_cache_blocks(
                task=task,
                generation=generation,
                max_steps=max_steps,
                container_name=container_name,
                workspace_root=workspace_root,
                execution_prompt=execution_prompt,
                history=prompt_history,
            )
            tail = _tb2_bridge_tail(
                step_index=step_index,
                max_steps=max_steps,
                last_observation=last_observation,
            )
            try:
                resp = caller.call(
                    _tb2_bridge_system_prompt(),
                    tail,
                    cache_blocks=cache_blocks,
                    model=model,
                )
                break
            except Exception as exc:  # noqa: BLE001 -- re-raised unless it is a trimmable prompt-too-long
                if not _TB2_PROMPT_TOO_LONG_RE.search(str(exc)):
                    raise
                if len(prompt_history) <= _TB2_PROMPT_TRIM_MIN_HISTORY_KEEP:
                    # Even the minimal prompt overflows; can't trim further.
                    resp = None
                    break
                if _trim_attempt >= _TB2_PROMPT_TRIM_MAX_RETRIES:
                    # No retry remains; do not mutate the prompt window just
                    # before returning an error.
                    resp = None
                    break
                dropped = _tb2_trim_oldest_history(prompt_history)
                print(
                    f"[tb2] prompt too long at step {step_index}; dropped {dropped} "
                    f"oldest prompt history step(s), retrying with {len(prompt_history)} remaining",
                    file=sys.stderr,
                    flush=True,
                )
        if resp is None:
            error_text = (
                "TB2 kcsi bridge could not fit the prompt within the provider context "
                "window even after trimming history; stopping with progress so far."
            )
            break
        raw_response, usage = resp.text, resp.usage
        aggregate_usage = aggregate_usage + usage
        try:
            action = _extract_json_object(raw_response)
        except ValueError as exc:
            invalid_response_count += 1
            error_text = str(exc)
            last_observation = (
                "Your previous reply was unusable because it was not one valid TB2 action JSON object.\n"
                f"Parser error: {_shorten(error_text, 1200)}\n"
                "Reply next with exactly one JSON object in one of the allowed shapes."
            )
            if invalid_response_count >= 3:
                break
            continue
        if invalid_response_count:
            error_text = ""
        kind = str(action.get("action") or "").strip().lower()
        if kind == "final":
            final_output = str(action.get("summary") or "").strip() or "Agent stopped without a final summary."
            break
        if kind not in _TB2_VALID_ACTIONS:
            invalid_response_count += 1
            error_text = f"Unsupported TB2 bridge action: {kind or '(missing)'}"
            last_observation = f"{error_text}\nReply next with one of: shell, read, write, edit, glob, grep, final."
            if invalid_response_count >= 3:
                break
            continue

        if kind == "shell":
            command = str(action.get("command") or "").strip()
            summary = str(action.get("summary") or "").strip()
            if not command:
                invalid_response_count += 1
                error_text = "TB2 bridge produced a shell action without a command."
                last_observation = f"{error_text}\nReply next with a non-empty shell command in the JSON object."
                if invalid_response_count >= 3:
                    break
                continue
            invalid_response_count = 0
            try:
                requested_timeout = float(action.get("timeout_sec") or 60.0)
            except (TypeError, ValueError):
                requested_timeout = 60.0
            shell_timeout = max(1.0, min(requested_timeout, max(1.0, deadline - time.monotonic())))
            try:
                proc = _docker_exec(
                    container_name=container_name,
                    command=command,
                    timeout_sec=shell_timeout,
                )
            except Exception as exc:
                proc = _completed_process_from_exec_exception(
                    command=command,
                    timeout_sec=shell_timeout,
                    exc=exc,
                )
            combined_output = (
                (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
            ).strip()
            history_entry = {
                "type": "tool_call",
                "tool_name": "tb2_shell",
                "tool_input": {
                    "command": command,
                    "summary": summary,
                    "timeout_sec": shell_timeout,
                    "container_name": container_name,
                    "agent_id": agent_id,
                },
                "tool_output": {
                    "exit_code": proc.returncode,
                    "stdout": _shorten(proc.stdout or "", 4000),
                    "stderr": _shorten(proc.stderr or "", 4000),
                    "combined_output": _shorten(combined_output, 6000),
                },
            }
            history.append(history_entry)
            prompt_history.append(history_entry)
            last_observation = (
                f"Command exit code: {proc.returncode}\nCombined output:\n{_shorten(combined_output, 3000)}"
            )
            continue

        # Native SDK-style actions: read / write / edit / glob / grep.
        #
        # DEVIATION FROM CANONICAL HARBOR / META-HARNESS: the canonical TB2
        # contract exposes only `execute_commands` (shell), `task_complete`,
        # and `image_read`. The five native actions below are a kcsi-only
        # ergonomic addition (higher-fidelity, lower-friction file ops) and
        # plausibly inflate TB2 scores vs. shell-only baselines.
        # See `benchmarks/docs/tb2_native_tools.md` for disclosure and labeling policy.
        invalid_response_count = 0
        handler = {
            "read": _handle_tb2_read,
            "write": _handle_tb2_write,
            "edit": _handle_tb2_edit,
            "glob": _handle_tb2_glob,
            "grep": _handle_tb2_grep,
        }[kind]
        history_entry, last_observation = handler(
            action=action,
            container_name=container_name,
            deadline=deadline,
            agent_id=agent_id,
        )
        history.append(history_entry)
        prompt_history.append(history_entry)
    else:
        if not final_output and not error_text:
            error_text = f"TB2 kcsi bridge hit step cap ({max_steps}) without emitting a 'final' action."

    transcript = _build_tb2_bridge_transcript(
        task=task,
        history=history,
        final_output=final_output,
        error_text=error_text,
    )
    if not final_output and error_text:
        final_output = error_text
    return TerminalBench2AgentRunResult(
        model_output=final_output,
        tool_trace=history,
        token_usage=aggregate_usage,
        error_text=error_text,
        transcript=transcript,
    )


@dataclass
class _TB2ImageAcquisition:
    """Outcome of the docker pull-with-retry / build phase."""

    image_built: bool
    image_acquired_via: str
    image_acquired_digest: str
    image_acquired_id: str
    image_digest_manifest_check: dict[str, Any] = field(default_factory=dict)


@dataclass
class _TB2VerifierPhaseResult:
    """Outcome of the fail-closed verifier-trust + verifier + reward readout.

    The security-critical fail-closed invariant (verifier deliberately not run
    in strict mode -> ``reward_readout`` stays ``None`` -> reward ``None`` ->
    unscored) is fully contained in ``_run_tb2_verifier_phase``; this carries the
    resulting state out for the scoring block to read.
    """

    verifier_proc: subprocess.CompletedProcess[str] | None
    reward_readout: _TB2RewardReadout | None
    verifier_trusted_toolchain: bool
    verifier_trusted_bash_detail: str
    require_trusted_verifier: bool
    verifier_fail_closed: bool
    artifact_snapshot_error: str


@dataclass
class _TB2Cleanup:
    """Outcome of the container/image cleanup run in the trial's ``finally``."""

    cleanup_error: str | None
    image_cleanup_error: str | None


_TB2_IMAGE_DIGEST_MANIFEST_ENV = "KCSI_TB2_IMAGE_DIGEST_MANIFEST"


def _normalize_tb2_repo_digest(value: str) -> str:
    """Normalize a Docker repo digest to its sha256 payload for comparison."""
    text = str(value or "").strip()
    marker = "sha256:"
    idx = text.lower().rfind(marker)
    if idx < 0:
        return ""
    return text[idx:].lower()


def _tb2_manifest_entry_digest(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("repo_digest", "image_digest", "digest", "image_acquired_digest"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _tb2_manifest_entry_image(entry: Any) -> str:
    if isinstance(entry, dict):
        value = entry.get("docker_image")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _tb2_manifest_lookup(
    manifest: dict[str, Any],
    *,
    task_id: str,
    docker_image: str,
) -> tuple[str, str, str]:
    for section_name, lookup_key in (("tasks", task_id), ("images", docker_image)):
        section = manifest.get(section_name)
        if not isinstance(section, dict):
            continue
        entry = section.get(lookup_key)
        if entry is None:
            continue
        expected_image = _tb2_manifest_entry_image(entry)
        digest = _tb2_manifest_entry_digest(entry)
        return digest, expected_image, f"{section_name}.{lookup_key}"
    return "", "", ""


def _load_tb2_image_digest_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read TB2 image digest manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"TB2 image digest manifest must be a JSON object: {path}")
    return payload


def _enforce_tb2_image_digest_manifest(
    *,
    task: TaskSpec,
    contract: TerminalBench2TaskContract,
    image_acquired_via: str,
    image_acquired_digest: str,
    meta_dir: Path,
) -> dict[str, Any]:
    raw_path = str(os.environ.get(_TB2_IMAGE_DIGEST_MANIFEST_ENV) or "").strip()
    if not raw_path:
        return {"enabled": False}

    manifest_path = Path(raw_path).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = manifest_path.resolve()
    manifest = _load_tb2_image_digest_manifest(manifest_path)
    expected_digest, expected_image, source = _tb2_manifest_lookup(
        manifest,
        task_id=task.id,
        docker_image=contract.docker_image,
    )
    if expected_image and expected_image != contract.docker_image:
        raise RuntimeError(
            f"TB2 image digest manifest {manifest_path} entry {source} names docker_image={expected_image!r}, "
            f"but task {task.id!r} resolved image {contract.docker_image!r}."
        )
    if not expected_digest:
        raise RuntimeError(
            f"TB2 image digest manifest {manifest_path} has no digest entry for task {task.id!r} "
            f"or image {contract.docker_image!r}."
        )
    expected_normalized = _normalize_tb2_repo_digest(expected_digest)
    actual_normalized = _normalize_tb2_repo_digest(image_acquired_digest)
    if not expected_normalized:
        raise RuntimeError(
            f"TB2 image digest manifest {manifest_path} entry {source} has invalid digest {expected_digest!r}; "
            "expected a value containing sha256:<digest>."
        )
    if image_acquired_via != "pull" or not actual_normalized:
        raise RuntimeError(
            f"TB2 image digest manifest {manifest_path} requires a pulled registry digest for task {task.id!r}; "
            f"image was acquired via {image_acquired_via!r} with digest {image_acquired_digest or '<empty>'!r}. "
            "Set KCSI_TB2_REQUIRE_PULL=1 and leave KCSI_TB2_DISABLE_PULL unset for publishable runs."
        )
    check = {
        "enabled": True,
        "path": str(manifest_path),
        "source": source,
        "task_id": task.id,
        "docker_image": contract.docker_image,
        "pinned_digest": expected_digest,
        "actual_digest": image_acquired_digest,
        "pinned_sha256": expected_normalized,
        "actual_sha256": actual_normalized,
        "matched": actual_normalized == expected_normalized,
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "image_digest_manifest_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    if not check["matched"]:
        raise RuntimeError(
            f"TB2 image digest mismatch for task {task.id!r}: manifest {manifest_path} expected "
            f"{expected_digest!r} from {source}, but Docker reported {image_acquired_digest!r}."
        )
    return check


def _acquire_tb2_image(
    *,
    task: TaskSpec,
    contract: TerminalBench2TaskContract,
    image_tag: str,
    meta_dir: Path,
) -> _TB2ImageAcquisition:
    """Acquire the TB2 image via docker pull-with-retry, falling back to a local
    build (unless ``KCSI_TB2_REQUIRE_PULL`` forbids it), then record image
    identity for fairness audits."""
    image_built = False
    image_acquired_via = ""
    pull_target = (contract.docker_image or "").strip()
    disable_pull = str(os.environ.get("KCSI_TB2_DISABLE_PULL") or "").strip().lower() in {"1", "true", "yes"}
    require_pull = str(os.environ.get("KCSI_TB2_REQUIRE_PULL") or "").strip().lower() in {"1", "true", "yes"}
    if require_pull and disable_pull:
        raise RuntimeError(
            "KCSI_TB2_REQUIRE_PULL=1 and KCSI_TB2_DISABLE_PULL=1 are mutually exclusive; unset one before running."
        )
    if require_pull and not pull_target:
        raise RuntimeError(
            f"KCSI_TB2_REQUIRE_PULL=1 but task {task.id!r} has no environment.docker_image; "
            "the canonical image cannot be pulled."
        )
    pull_failure_tail = ""
    pull_failure_retryable = False
    pull_failure_reason = "unknown"
    if pull_target and not disable_pull:
        pull_timeout = min(max(120.0, contract.build_timeout_sec / 2.0), contract.build_timeout_sec)
        # Required-pull mode uses the task retry loop as its sole retry owner so
        # attempt history is persisted and a 3x inner loop cannot multiply the
        # default four task attempts into twelve registry requests. Optional
        # pull-before-build keeps the local three-attempt tolerance.
        pull = _docker_pull_with_retry(pull_target, timeout_sec=pull_timeout, attempts=1 if require_pull else 3)
        (meta_dir / "docker_pull.stdout.txt").write_text(pull.stdout or "", encoding="utf-8")
        (meta_dir / "docker_pull.stderr.txt").write_text(pull.stderr or "", encoding="utf-8")
        if pull.returncode == 0:
            tag_proc = _run(["docker", "tag", pull_target, image_tag], timeout_sec=30)
            (meta_dir / "docker_tag.stdout.txt").write_text(tag_proc.stdout or "", encoding="utf-8")
            (meta_dir / "docker_tag.stderr.txt").write_text(tag_proc.stderr or "", encoding="utf-8")
            if tag_proc.returncode != 0:
                raise RuntimeError(
                    f"failed to retag pulled TB2 image {pull_target!r} as {image_tag!r}: "
                    f"{_tail(tag_proc.stderr or tag_proc.stdout or '')}"
                )
            image_built = True
            image_acquired_via = "pull"
        else:
            pull_failure_retryable = _looks_like_transient_docker_registry_failure(pull)
            pull_failure_reason = "transient" if pull_failure_retryable else "non_transient"
            redacted_tail = _scrub_credentials(_tail(pull.stderr or pull.stdout or ""))
            printable_tail = "".join(char if char.isprintable() else " " for char in redacted_tail)
            pull_failure_tail = " ".join(printable_tail.split())

    if not image_built:
        if require_pull:
            raise ContainerRegistryError(
                f"KCSI_TB2_REQUIRE_PULL=1 and pull of {pull_target!r} failed for task {task.id!r}; "
                f"refusing to fall back to local build for fairness mode. "
                f"pull stderr: {pull_failure_tail}",
                retryable=pull_failure_retryable,
                reason=pull_failure_reason,
                image=pull_target,
            )
        build_cmd = [
            "docker",
            "build",
            "-t",
            image_tag,
            "-f",
            str(contract.environment_dir / "Dockerfile"),
            str(contract.environment_dir),
        ]
        build = _docker_build_with_retry(build_cmd, timeout_sec=contract.build_timeout_sec)
        (meta_dir / "docker_build.stdout.txt").write_text(build.stdout or "", encoding="utf-8")
        (meta_dir / "docker_build.stderr.txt").write_text(build.stderr or "", encoding="utf-8")
        if build.returncode != 0:
            raise RuntimeError(
                f"failed to acquire TB2 image for {task.id!r}: build exit={build.returncode}\n"
                f"{_tail(build.stderr or build.stdout or '')}"
            )
        image_built = True
        image_acquired_via = "build"

    # Record the bytes we actually run against, for fairness audits.
    # image_acquired_digest is the upstream registry digest (only set
    # when pull succeeded); image_acquired_id is the local Docker image
    # ID (always set). Together they let post-hoc analysis detect
    # within-experiment drift (image ID differs across trials of the
    # same task) or compare against an externally-recorded canonical
    # digest. KCSI_TB2_REQUIRE_PULL only guarantees "pull succeeded
    # at trial start"; the digest record proves which bytes were used.
    image_acquired_digest, image_acquired_id = _inspect_image_identity(
        pull_target=pull_target if image_acquired_via == "pull" else "",
        image_tag=image_tag,
    )
    image_digest_manifest_check = _enforce_tb2_image_digest_manifest(
        task=task,
        contract=contract,
        image_acquired_via=image_acquired_via,
        image_acquired_digest=image_acquired_digest,
        meta_dir=meta_dir,
    )

    print(
        f"[tb2] image task={task.id} via={image_acquired_via} "
        f"pull_target={pull_target or '-'} tag={image_tag} "
        f"digest={image_acquired_digest or '-'} id={image_acquired_id or '-'}",
        file=sys.stderr,
        flush=True,
    )

    return _TB2ImageAcquisition(
        image_built=image_built,
        image_acquired_via=image_acquired_via,
        image_acquired_digest=image_acquired_digest,
        image_acquired_id=image_acquired_id,
        image_digest_manifest_check=image_digest_manifest_check,
    )


def _run_tb2_agent_phase(
    *,
    task: TaskSpec,
    contract: TerminalBench2TaskContract,
    agent_mode: str,
    container_name: str,
    workspace_root: Path,
    meta_dir: Path,
    provider_env: dict[str, str] | None,
    generation: int,
    agent_id: str,
    seed_package: Any,
    raw_mode: bool,
    resolved_agent_command: str,
) -> tuple[subprocess.CompletedProcess[str] | None, TerminalBench2AgentRunResult | None]:
    """Copy agent-phase assets into the container and run the agent (kcsi-bridge
    or oracle/shell). Returns ``(agent_proc, kcsi_result)``."""
    agent_proc: subprocess.CompletedProcess[str] | None = None
    kcsi_result: TerminalBench2AgentRunResult | None = None
    for src, dst in _agent_phase_copies(contract=contract, agent_mode=agent_mode):
        cp_proc = _docker_cp_to_container(src=src, container_name=container_name, dst=dst, timeout_sec=60)
        (meta_dir / f"agent_copy_{src.name}.stdout.txt").write_text(cp_proc.stdout or "", encoding="utf-8")
        (meta_dir / f"agent_copy_{src.name}.stderr.txt").write_text(cp_proc.stderr or "", encoding="utf-8")
        if cp_proc.returncode != 0:
            raise RuntimeError(
                f"failed to copy agent-phase TB2 asset {src} -> {dst}: {_tail(cp_proc.stderr or cp_proc.stdout or '')}"
            )
    if agent_mode.strip().lower() == "kcsi":
        try:
            kcsi_result = _run_kcsi_agent_in_tb2_container(
                task=task,
                contract=contract,
                container_name=container_name,
                workspace_root=workspace_root,
                provider_env=provider_env,
                generation=generation,
                agent_id=agent_id,
                seed_package=seed_package,
                raw_mode=raw_mode,
            )
            agent_proc = subprocess.CompletedProcess(
                args=["tb2_kcsi_bridge"],
                returncode=0 if not kcsi_result.error_text else 1,
                stdout=kcsi_result.model_output,
                stderr=kcsi_result.error_text,
            )
        except Exception as exc:
            kcsi_result = TerminalBench2AgentRunResult(
                model_output="",
                tool_trace=[],
                token_usage=TokenUsage(),
                error_text=str(exc),
                transcript=f"# tb2_bridge_transcript\ntask_id: {task.id}\nerror:\n{exc}",
            )
            agent_proc = subprocess.CompletedProcess(
                args=["tb2_kcsi_bridge"],
                returncode=1,
                stdout="",
                stderr=str(exc),
            )
        (meta_dir / "agent.stdout.txt").write_text(agent_proc.stdout or "", encoding="utf-8")
        (meta_dir / "agent.stderr.txt").write_text(agent_proc.stderr or "", encoding="utf-8")
        (meta_dir / "agent.tool_trace.json").write_text(
            json.dumps(kcsi_result.tool_trace if kcsi_result else [], indent=2),
            encoding="utf-8",
        )
        (meta_dir / "agent.transcript.md").write_text(
            kcsi_result.transcript if kcsi_result else "",
            encoding="utf-8",
        )
    else:
        agent_proc = _run(
            ["docker", "exec", container_name, "bash", "-lc", resolved_agent_command],
            timeout_sec=contract.agent_timeout_sec,
        )
        (meta_dir / "agent.stdout.txt").write_text(agent_proc.stdout or "", encoding="utf-8")
        (meta_dir / "agent.stderr.txt").write_text(agent_proc.stderr or "", encoding="utf-8")
    return agent_proc, kcsi_result


def _run_tb2_verifier_phase(
    *,
    contract: TerminalBench2TaskContract,
    container_name: str,
    image_tag: str,
    suffix: str,
    meta_dir: Path,
    output_root: Path,
) -> _TB2VerifierPhaseResult:
    """Run the fail-closed verifier-trust check, the verifier, and the
    race-guarded reward readout as ONE unit.

    SECURITY-CRITICAL: the fail-closed check MUST stay ordered before the reward
    read, and in strict fail-closed mode the verifier is deliberately NOT run and
    the reward readout + artifact snapshot are SKIPPED (``reward_readout`` stays
    ``None`` -> reward ``None`` -> unscored). Do not reorder or split these.
    """
    verifier_proc: subprocess.CompletedProcess[str] | None = None
    reward_readout: _TB2RewardReadout | None = None
    artifact_snapshot_error: str = ""
    verifier_trusted_toolchain = False
    verifier_trusted_bash_detail = ""
    require_trusted_verifier = _tb2_require_trusted_verifier()
    verifier_fail_closed = False

    # Grader integrity (static pre-plant): wipe every verifier
    # destination and the reward/ctrf outputs in the container before
    # copying/running the official verifier, so an agent that pre-created
    # /tests (shadowing the official tests via docker-cp nesting) or
    # pre-wrote reward.txt=1.0 / ctrf.json cannot force a vacuous
    # resolved=True.
    # `rm -rf` is a no-op on the normal path (these paths don't
    # pre-exist), so legitimate solves are unaffected. This closes the
    # STATIC holes only. We deliberately do NOT kill container processes
    # here: an unconditional sweep also destroys agent-started services
    # that legitimate verifiers probe while running (~6/89 tasks:
    # kv-store-grpc, nginx-request-logging, hf-model-inference,
    # install-windows-3.11, qemu-startup, qemu-alpine-ssh). The
    # resident-process reward-replant vector is a KNOWN, ACCEPTED
    # residual for cooperative TB2. The
    # trojaned-toolchain ENTRYPOINT hijack is HARDENED below by launching
    # test.sh through a trusted image-extracted bash; default
    # trusted-verifier mode fails closed if injection cannot be established.
    # The in-container-subprocess residual remains. See
    # `_verifier_sanitize_paths` and `_tb2_verifier_command`.
    sanitize_paths = _verifier_sanitize_paths(contract=contract)
    sanitize_cmd = f"{_TB2_CONTAINER_RM} -rf " + " ".join(shlex.quote(p) for p in sanitize_paths)
    sanitize_proc = _docker_exec(
        container_name=container_name,
        command=sanitize_cmd,
        timeout_sec=30,
    )
    (meta_dir / "verifier_sanitize.stdout.txt").write_text(sanitize_proc.stdout or "", encoding="utf-8")
    (meta_dir / "verifier_sanitize.stderr.txt").write_text(sanitize_proc.stderr or "", encoding="utf-8")
    if sanitize_proc.returncode != 0:
        raise RuntimeError(
            f"failed to sanitize TB2 verifier targets before scoring: "
            f"{_tail(sanitize_proc.stderr or sanitize_proc.stdout or '')}"
        )

    for src, dst in _verifier_phase_copies(contract=contract):
        cp_proc = _docker_cp_to_container(src=src, container_name=container_name, dst=dst, timeout_sec=60)
        (meta_dir / f"verifier_copy_{src.name}.stdout.txt").write_text(cp_proc.stdout or "", encoding="utf-8")
        (meta_dir / f"verifier_copy_{src.name}.stderr.txt").write_text(cp_proc.stderr or "", encoding="utf-8")
        if cp_proc.returncode != 0:
            raise RuntimeError(
                f"failed to copy verifier-phase TB2 asset {src} -> {dst}: "
                f"{_tail(cp_proc.stderr or cp_proc.stdout or '')}"
            )
    # Grader integrity (trojaned toolchain): a root agent can drop
    # a shim at `/usr/local/bin/bash` (which precedes `/bin` on PATH) to
    # hijack the confirmed `docker exec ... bash -c "bash /tests/test.sh"`
    # entrypoint. Inject a verifier-owned bash extracted from the pristine
    # image into a fresh, randomly-named dir and launch test.sh through
    # that ABSOLUTE trusted interpreter with the dir prepended to PATH, so
    # the planted shim is neither the entrypoint nor the first `bash` on
    # PATH. On extraction/injection failure, default trusted-verifier mode
    # fails closed as unscored; explicit legacy mode can still opt back into the
    # old invocation. This HARDENS the ENTRYPOINT hijack against a static shim;
    # test.sh's own internal subprocess calls still resolve in-container
    # (documented residual). See `_extract_trusted_bash_from_image`
    # / `_tb2_verifier_command`.
    trusted_dir = f"{_TB2_TRUSTED_BIN_ROOT}-{suffix}"
    host_trusted_bash = meta_dir / "trusted_bash"
    extracted, extract_detail = _extract_trusted_bash_from_image(image_tag=image_tag, dest_host_path=host_trusted_bash)
    if extracted:
        # Set the exec bit on the HOST source before docker-cp (which
        # preserves file mode into the container). Doing this instead of
        # an in-container `chmod` keeps verifier setup independent of
        # in-container tool state. `os.chmod` on the host has no container
        # exposure.
        try:
            os.chmod(host_trusted_bash, 0o755)
        except OSError:
            pass
        setup_proc = _docker_exec(
            container_name=container_name,
            command=(
                f"{_TB2_CONTAINER_RM} -rf {shlex.quote(trusted_dir)} "
                f"&& {_TB2_CONTAINER_MKDIR} -p {shlex.quote(trusted_dir)}"
            ),
            timeout_sec=30,
        )
        cp_bash = _docker_cp_to_container(
            src=host_trusted_bash,
            container_name=container_name,
            dst=f"{trusted_dir}/bash",
            timeout_sec=60,
        )
        if setup_proc.returncode == 0 and cp_bash.returncode == 0:
            verifier_trusted_toolchain = True
        else:
            verifier_trusted_bash_detail = "trusted bash injection failed; using legacy invocation: " + _tail(
                cp_bash.stderr or setup_proc.stderr or ""
            )
    else:
        verifier_trusted_bash_detail = f"trusted bash extraction failed; using legacy invocation: {extract_detail}"

    if require_trusted_verifier and not verifier_trusted_toolchain:
        # Strict mode: the trusted-bash injection did not take
        # effect, so the only remaining option is the legacy
        # PATH-resolved invocation. Rather than run it and record a silent
        # equivalent-to-main verdict, fail closed: skip the verifier entirely
        # and leave the trial UNSCORED (reward stays None -> never a fabricated
        # pass). The refused fallback is then detectable as an unscored trial via
        # `verifier_fail_closed` / `verifier_trusted_bash_detail`. Default-on;
        # set KCSI_TB2_REQUIRE_TRUSTED_VERIFIER=0 only for legacy comparisons.
        # See `_tb2_require_trusted_verifier`.
        verifier_fail_closed = True
        downgrade_reason = verifier_trusted_bash_detail or "trusted toolchain unavailable"
        verifier_trusted_bash_detail = (
            "strict mode (KCSI_TB2_REQUIRE_TRUSTED_VERIFIER): refused legacy "
            f"PATH-resolved verifier invocation; trial left unscored ({downgrade_reason})"
        )
        (meta_dir / "verifier.stdout.txt").write_text("", encoding="utf-8")
        (meta_dir / "verifier.stderr.txt").write_text(verifier_trusted_bash_detail + "\n", encoding="utf-8")
        # reward_readout stays None -> reward None -> unscored below.
    else:
        if verifier_trusted_toolchain:
            verifier_cmd = _tb2_verifier_command(
                container_name=container_name,
                trusted_bash=f"{trusted_dir}/bash",
                path_prefix=trusted_dir,
            )
        else:
            # Preserve the exact legacy invocation so an extraction failure is
            # never worse than main (it just forgoes the entrypoint hardening).
            verifier_cmd = ["docker", "exec", container_name, "bash", "-c", "bash /tests/test.sh"]
        verifier_proc = _run(verifier_cmd, timeout_sec=contract.verifier_timeout_sec)
        (meta_dir / "verifier.stdout.txt").write_text(verifier_proc.stdout or "", encoding="utf-8")
        (meta_dir / "verifier.stderr.txt").write_text(verifier_proc.stderr or "", encoding="utf-8")

    # In strict fail-closed mode we deliberately did not run the verifier,
    # so skip the reward readout AND the artifact snapshot entirely: an
    # untrusted-toolchain path also leaves the in-container reward.txt/ctrf.json
    # agent-writable, and reading them here would let a planted reward score.
    # reward_readout stays None -> reward None -> unscored.
    if not verifier_fail_closed:
        # Grader integrity (racing writer): read reward.txt from
        # INSIDE the still-running container via `docker exec`, immediately
        # after the official test.sh returns and BEFORE `docker rm -f`. The
        # score is taken from THIS snapshot, not from the host bind-mount
        # file read after container removal, so a writer the agent
        # backgrounded during its own phase (which survives the one-shot
        # pre-verifier purge) cannot rewrite the verdict after we record it.
        # A two-sample mtime/content check fails the read closed if a writer
        # is still racing at snapshot time. See `_read_tb2_reward_before_removal`.
        reward_readout = _read_tb2_reward_before_removal(container_name=container_name)
        (meta_dir / "reward_readout.json").write_text(json.dumps(reward_readout.to_dict(), indent=2), encoding="utf-8")
        # Snapshot the reward/CTRF artifacts from the container before
        # removal too, so the recorded artifacts match the scored state
        # rather than a post-verifier host-side rewrite.
        (output_root / "artifacts").mkdir(parents=True, exist_ok=True)
        # Best-effort snapshot: a `docker cp` hang/failure here must NOT
        # discard the reward already captured in `reward_readout` above or
        # abort the trial. `_docker_cp_from_container` swallows a
        # nonzero exit, but a `docker cp` timeout raises `TimeoutExpired`;
        # catch it (and any OSError writing the host dst) so a transient
        # Docker slowdown turns a legitimately-scored solve into an unscored
        # infra failure. The scoring below reads `reward_readout`, not these.
        try:
            _docker_cp_from_container(
                container_name=container_name,
                src=_TB2_CONTAINER_REWARD_PATH,
                dst=output_root / "artifacts" / "reward.txt",
                timeout_sec=30,
            )
            _docker_cp_from_container(
                container_name=container_name,
                src=_TB2_CONTAINER_CTRF_PATH,
                dst=output_root / "artifacts" / "ctrf.json",
                timeout_sec=30,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            artifact_snapshot_error = _tail(str(exc))

    return _TB2VerifierPhaseResult(
        verifier_proc=verifier_proc,
        reward_readout=reward_readout,
        verifier_trusted_toolchain=verifier_trusted_toolchain,
        verifier_trusted_bash_detail=verifier_trusted_bash_detail,
        require_trusted_verifier=require_trusted_verifier,
        verifier_fail_closed=verifier_fail_closed,
        artifact_snapshot_error=artifact_snapshot_error,
    )


def _cleanup_tb2(
    *,
    container_name: str,
    keep_container: bool,
    image_built: bool,
    keep_image: bool,
    image_tag: str,
    meta_dir: Path,
) -> _TB2Cleanup:
    """Remove the container and (unless kept) the image. Runs in the trial's
    ``finally`` on every exit path."""
    cleanup_error: str | None = None
    image_cleanup_error: str | None = None
    if not keep_container:
        # Cleanup runs in the trial's `finally`, so a hung/failed `docker rm`
        # or `docker rmi` must NOT raise out and mask the completed
        # agent/verifier outcome or prevent final trial metadata from being
        # written. `_run` raises `TimeoutExpired` on a stalled docker
        # call (and `OSError` if the binary is missing); convert both into a
        # recorded error surfaced via `cleanup_error`/`image_cleanup_error`.
        rm_cmd = ["docker", "rm", "-f", container_name]
        try:
            rm_proc = _run(rm_cmd, timeout_sec=20)
        except subprocess.TimeoutExpired as exc:
            rm_proc = _completed_from_timeout(cmd=rm_cmd, timeout_sec=20, exc=exc)
        except OSError as exc:
            rm_proc = subprocess.CompletedProcess(args=rm_cmd, returncode=1, stdout="", stderr=str(exc))
        if rm_proc.returncode != 0:
            cleanup_error = _tail(rm_proc.stderr or rm_proc.stdout or "")
        (meta_dir / "docker_rm.stdout.txt").write_text(rm_proc.stdout or "", encoding="utf-8")
        (meta_dir / "docker_rm.stderr.txt").write_text(rm_proc.stderr or "", encoding="utf-8")
        if image_built and not keep_image:
            rmi_cmd = ["docker", "rmi", "-f", image_tag]
            try:
                rmi_proc = _run(rmi_cmd, timeout_sec=30)
            except subprocess.TimeoutExpired as exc:
                rmi_proc = _completed_from_timeout(cmd=rmi_cmd, timeout_sec=30, exc=exc)
            except OSError as exc:
                rmi_proc = subprocess.CompletedProcess(args=rmi_cmd, returncode=1, stdout="", stderr=str(exc))
            if rmi_proc.returncode != 0:
                image_cleanup_error = _tail(rmi_proc.stderr or rmi_proc.stdout or "")
            (meta_dir / "docker_rmi.stdout.txt").write_text(rmi_proc.stdout or "", encoding="utf-8")
            (meta_dir / "docker_rmi.stderr.txt").write_text(rmi_proc.stderr or "", encoding="utf-8")
    return _TB2Cleanup(cleanup_error=cleanup_error, image_cleanup_error=image_cleanup_error)


def run_terminal_bench_2_trial(
    *,
    task: TaskSpec,
    agent_mode: str = "oracle",
    agent_command: str | None = None,
    output_dir: str | None = None,
    keep_container: bool = False,
    provider_env: dict[str, str] | None = None,
    generation: int = 1,
    agent_id: str = "",
    seed_package: Any = None,
    raw_mode: bool = False,
) -> TerminalBench2TrialResult:
    contract = resolve_terminal_bench_2_task_contract(task)
    resolved_agent_command = default_agent_command(
        agent_mode=agent_mode,
        explicit_command=agent_command,
    )
    safe_task = _docker_name_component(task.id)
    suffix = uuid.uuid4().hex[:10]
    image_tag = _stable_image_tag(environment_dir=contract.environment_dir, safe_task=safe_task)
    container_name = f"kcsi-tb2-{safe_task}-{suffix}"
    keep_image = _keep_tb2_images_default()

    if output_dir:
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        cleanup_output_root = False
    else:
        output_root = Path(tempfile.mkdtemp(prefix=f"tb2-trial-{safe_task}-")).resolve()
        cleanup_output_root = True

    result: TerminalBench2TrialResult | None = None
    try:
        logs_root = output_root / "logs"
        verifier_logs = logs_root / "verifier"
        verifier_logs.mkdir(parents=True, exist_ok=True)
        meta_dir = output_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        workspace_root = materialize_terminal_bench_2_workspace_seed(
            task=task, output_dir=output_root, seed_package=seed_package, raw_mode=raw_mode
        )

        image = _acquire_tb2_image(
            task=task,
            contract=contract,
            image_tag=image_tag,
            meta_dir=meta_dir,
        )
        image_acquired_via = image.image_acquired_via
        image_acquired_digest = image.image_acquired_digest
        image_acquired_id = image.image_acquired_id
        image_digest_manifest_check = image.image_digest_manifest_check

        run_cmd = _docker_run_command(
            image_tag=image_tag,
            container_name=container_name,
            logs_root=logs_root,
            workspace_root=workspace_root,
            cpus=contract.cpus,
            memory=contract.memory,
        )
        run_proc = _run(run_cmd, timeout_sec=30)
        (meta_dir / "docker_run.stdout.txt").write_text(run_proc.stdout or "", encoding="utf-8")
        (meta_dir / "docker_run.stderr.txt").write_text(run_proc.stderr or "", encoding="utf-8")
        if run_proc.returncode != 0:
            raise RuntimeError(
                f"failed to start TB2 container for {task.id!r}: exit={run_proc.returncode}\n"
                f"{_tail(run_proc.stderr or run_proc.stdout or '')}"
            )

        agent_proc: subprocess.CompletedProcess[str] | None = None
        kcsi_result: TerminalBench2AgentRunResult | None = None
        verifier: _TB2VerifierPhaseResult | None = None
        try:
            agent_proc, kcsi_result = _run_tb2_agent_phase(
                task=task,
                contract=contract,
                agent_mode=agent_mode,
                container_name=container_name,
                workspace_root=workspace_root,
                meta_dir=meta_dir,
                provider_env=provider_env,
                generation=generation,
                agent_id=agent_id,
                seed_package=seed_package,
                raw_mode=raw_mode,
                resolved_agent_command=resolved_agent_command,
            )
            verifier = _run_tb2_verifier_phase(
                contract=contract,
                container_name=container_name,
                image_tag=image_tag,
                suffix=suffix,
                meta_dir=meta_dir,
                output_root=output_root,
            )
        finally:
            cleanup = _cleanup_tb2(
                container_name=container_name,
                keep_container=keep_container,
                image_built=image.image_built,
                keep_image=keep_image,
                image_tag=image_tag,
                meta_dir=meta_dir,
            )

        # Unpack phase results into the local names the scoring block reads
        # below. The security-critical fail-closed ordering (verifier skipped ->
        # reward_readout None -> unscored) lives entirely inside
        # `_run_tb2_verifier_phase`; these assignments are read-only.
        verifier_proc = verifier.verifier_proc
        reward_readout = verifier.reward_readout
        verifier_trusted_toolchain = verifier.verifier_trusted_toolchain
        verifier_trusted_bash_detail = verifier.verifier_trusted_bash_detail
        require_trusted_verifier = verifier.require_trusted_verifier
        verifier_fail_closed = verifier.verifier_fail_closed
        artifact_snapshot_error = verifier.artifact_snapshot_error
        cleanup_error = cleanup.cleanup_error
        image_cleanup_error = cleanup.image_cleanup_error

        reward_path = verifier_logs / "reward.txt"
        ctrf_path = verifier_logs / "ctrf.json"
        # Score from the pre-removal `docker exec` readout, never from the
        # host bind-mount file (which a racing writer could rewrite between the
        # verifier finishing and container removal). Fail closed (reward=None,
        # scored as unscored -- never a fabricated pass) if a writer was still
        # racing the grader when the reward was snapshotted, or if the readout
        # itself failed. The artifacts were already captured from the container
        # before removal above.
        if reward_readout is not None and not reward_readout.active_writer:
            reward = reward_readout.reward
        else:
            reward = None

        # A kcsi-bridge exception is caught above (agent_proc.returncode=1) and
        # is NOT terminal -- the verifier still runs against whatever state the
        # crashed agent left. Surface that distinctly so a failed agent phase can
        # never read "completed" and contaminate solve-rate accounting. `resolved`
        # (and native_score) stay reward-driven on purpose: a genuine solve
        # followed by a late/transient bridge crash still scores. Downstream
        # filters can exclude/inspect `agent_failed_but_verifier_ran`.
        agent_failed = agent_proc is not None and agent_proc.returncode != 0
        if verifier_fail_closed:
            # Strict mode refused the untrusted-toolchain fallback; the
            # verifier never ran, so this is an unscored trial, distinct from a
            # verifier that ran but produced no reward.
            trial_status = TB2_VERIFIER_FAIL_CLOSED_STATUS
        elif reward is None:
            trial_status = "verifier_did_not_produce_reward"
        elif verifier_proc is not None and verifier_proc.returncode != 0:
            trial_status = "verifier_failed"
        elif agent_failed:
            trial_status = "agent_failed_but_verifier_ran"
        else:
            trial_status = "completed"

        runtime_meta: dict[str, Any] = {
            "task_source": "terminal_bench_2",
            "trial_status": trial_status,
            "task_root": str(contract.task_root),
            "docker_image": contract.docker_image,
            "local_image_tag": image_tag,
            "image_acquired_via": image_acquired_via,
            "image_acquired_digest": image_acquired_digest,
            "image_acquired_id": image_acquired_id,
            "image_digest_manifest_check": image_digest_manifest_check,
            "image_digest_manifest_path": str(image_digest_manifest_check.get("path") or ""),
            "image_digest_manifest_pinned_digest": str(image_digest_manifest_check.get("pinned_digest") or ""),
            "image_digest_manifest_matched": bool(image_digest_manifest_check.get("matched")),
            "container_name": container_name,
            "build_timeout_sec": contract.build_timeout_sec,
            "agent_timeout_sec": contract.agent_timeout_sec,
            "verifier_timeout_sec": contract.verifier_timeout_sec,
            "timeout_source": TB2_TIMEOUT_SOURCE,
            "cpus": contract.cpus,
            "memory": contract.memory,
            "storage": contract.storage,
            "agent_mode": agent_mode,
            "agent_command": resolved_agent_command,
            "model_requested": str((provider_env or {}).get("MODEL") or os.environ.get("MODEL") or ""),
            "workspace_root": str(workspace_root),
            "agent_phase_copies": [dst for _, dst in _agent_phase_copies(contract=contract, agent_mode=agent_mode)],
            "verifier_phase_copies": [dst for _, dst in _verifier_phase_copies(contract=contract)],
            "agent_exit_code": None if agent_proc is None else agent_proc.returncode,
            "verifier_exit_code": None if verifier_proc is None else verifier_proc.returncode,
            "verifier_trusted_toolchain": verifier_trusted_toolchain,
            "verifier_trusted_bash_detail": verifier_trusted_bash_detail,
            "require_trusted_verifier": require_trusted_verifier,
            "verifier_fail_closed": verifier_fail_closed,
            "reward": reward,
            "reward_path": str(reward_path),
            "reward_source": "container_exec_pre_removal",
            "reward_active_writer_detected": bool(reward_readout.active_writer) if reward_readout else False,
            "reward_readout_detail": reward_readout.detail if reward_readout else "reward readout missing",
            "artifact_snapshot_error": artifact_snapshot_error,
            "ctrf_path": str(ctrf_path) if ctrf_path.is_file() else "",
            "cleanup_error": cleanup_error or "",
            "image_cleanup_error": image_cleanup_error or "",
            "agent_stdout_tail": _tail((agent_proc.stdout if agent_proc else "") or ""),
            "agent_stderr_tail": _tail((agent_proc.stderr if agent_proc else "") or ""),
            "verifier_stdout_tail": _tail((verifier_proc.stdout if verifier_proc else "") or ""),
            "verifier_stderr_tail": _tail((verifier_proc.stderr if verifier_proc else "") or ""),
            "native_session_memory": kcsi_result.transcript if kcsi_result else "",
            "raw_native_session_memory": kcsi_result.transcript if kcsi_result else "",
            "tool_trace": kcsi_result.tool_trace if kcsi_result else [],
            "token_usage": (kcsi_result.token_usage.to_dict() if kcsi_result else TokenUsage().to_dict()),
        }
        (meta_dir / "trial_result.json").write_text(json.dumps(runtime_meta, indent=2), encoding="utf-8")

        result = TerminalBench2TrialResult(
            task_id=task.id,
            task_root=str(contract.task_root),
            image_tag=image_tag,
            container_name=container_name,
            agent_command=resolved_agent_command,
            agent_exit_code=None if agent_proc is None else agent_proc.returncode,
            verifier_exit_code=None if verifier_proc is None else verifier_proc.returncode,
            reward=reward,
            resolved=bool(reward is not None and reward >= 1.0),
            output_dir=str(output_root),
            runtime_meta=runtime_meta,
            model_output=kcsi_result.model_output if kcsi_result else ((agent_proc.stdout or "") if agent_proc else ""),
            tool_trace=kcsi_result.tool_trace if kcsi_result else [],
            token_usage=kcsi_result.token_usage if kcsi_result else TokenUsage(),
        )
    finally:
        if cleanup_output_root and not keep_container:
            shutil.rmtree(output_root, ignore_errors=True)

    assert result is not None
    return result
