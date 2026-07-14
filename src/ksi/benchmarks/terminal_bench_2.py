from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import KsiError
from ..models import TaskSpec
from ..tasks.registry import resolve_source

_TB2_BUILD_TIMEOUT_FLOOR_SEC = 1800.0

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


TERMINAL_BENCH_2_REQUIRED_TASK_FILES: tuple[str, ...] = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
)

# Final ``status`` emitted by :meth:`TerminalBench2Evaluator.evaluate` when the
# container verifier never ran (crash / OOM / exec timeout before the verifier
# stage), so no trustworthy reward was produced. The TB2 scorer
# (``orchestrator.scoring.score_tb2_from_eval``) treats this as *unscored*
# (``None``), not a genuine ``0.0`` failure.
TB2_VERIFIER_MISSING_STATUS = "verifier_did_not_produce_reward"
# Final ``status`` when strict mode (``KSI_TB2_REQUIRE_TRUSTED_VERIFIER``)
# refused to run the verifier because its trusted-toolchain injection did not
# take effect -- an attacker-selectable forced fallback. The verifier never ran,
# so like the missing-verifier case this is *unscored* (``None``), not ``0.0``.
TB2_VERIFIER_FAIL_CLOSED_STATUS = "verifier_fail_closed_untrusted_toolchain"
# Statuses where no trustworthy reward was produced -> score ``None`` (unscored),
# never a fabricated genuine ``0.0`` that would contaminate _best_scores /
# record_attempt / distillation. Both the evaluator's ``native_score`` and
# ``orchestrator.scoring.score_tb2_from_eval`` gate on this set.
TB2_VERIFIER_UNSCORED_STATUSES = frozenset({TB2_VERIFIER_MISSING_STATUS, TB2_VERIFIER_FAIL_CLOSED_STATUS})
TB2_TIMEOUT_SOURCE = "task.toml"


class TerminalBench2ContractError(KsiError, ValueError):
    """Raised when a TaskSpec cannot be mapped to a valid TB2 task package."""


@dataclass(frozen=True)
class TerminalBench2TaskContract:
    task_id: str
    task_root: Path
    instruction_path: Path
    task_toml_path: Path
    environment_dir: Path
    tests_dir: Path
    docker_image: str
    build_timeout_sec: float
    agent_timeout_sec: float
    verifier_timeout_sec: float
    cpus: float | None
    memory: str
    # NOTE: parsed from task.toml [environment].storage but currently NOT
    # enforced via `docker run --storage-opt`. Most hosts use overlay2 which
    # ignores that flag. Recorded in runtime_meta for observability only.
    storage: str
    category: str
    difficulty: str


def _task_root_from_metadata(task: TaskSpec) -> Path:
    metadata = task.metadata or {}
    raw_value = metadata.get("task_root")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise TerminalBench2ContractError(f"TB2 task {task.id!r} is missing metadata.task_root")
    task_root = Path(raw_value).expanduser().resolve()
    if not task_root.exists() or not task_root.is_dir():
        raise TerminalBench2ContractError(f"TB2 task {task.id!r} root does not exist: {task_root}")
    return task_root


def _resolve_build_timeout_sec(env: dict[str, Any]) -> float:
    """Build timeout = max(task.toml value, env override, floor).

    Harbor pins `build_timeout_sec = 600.0` uniformly across the upstream
    corpus, but several Dockerfiles compile heavy toolchains from source
    (e.g., ``custom-memory-heap-crash`` rebuilds GCC libstdc++ twice) and
    cannot reliably finish in 600s on a cold or contended host. Using the
    max of the three sources keeps task.toml authoritative when it
    raises the value but never lets it sink below the floor.
    """
    candidates: list[float] = [_TB2_BUILD_TIMEOUT_FLOOR_SEC]
    raw_task = env.get("build_timeout_sec")
    if raw_task is not None:
        try:
            candidates.append(float(raw_task))
        except (TypeError, ValueError):
            pass
    raw_env = os.environ.get("KSI_TB2_BUILD_TIMEOUT_SEC")
    if raw_env is not None and raw_env.strip():
        try:
            candidates.append(float(raw_env))
        except ValueError:
            pass
    return max(candidates)


def _read_task_toml(task_toml_path: Path) -> dict[str, Any]:
    try:
        with task_toml_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except Exception as exc:  # pragma: no cover - exercised via contract errors
        raise TerminalBench2ContractError(f"failed to parse task.toml at {task_toml_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TerminalBench2ContractError(f"task.toml at {task_toml_path} is not a TOML table")
    return payload


def resolve_terminal_bench_2_task_contract(task: TaskSpec) -> TerminalBench2TaskContract:
    metadata = task.metadata or {}
    task_source = str(metadata.get("task_source") or "").strip()
    spec = resolve_source(task_source)
    if task_source and (spec is None or not spec.delegates_runtime):
        raise TerminalBench2ContractError(
            f"task {task.id!r} has task_source={task_source!r}, expected 'terminal_bench_2'"
        )

    task_root = _task_root_from_metadata(task)
    missing = [name for name in TERMINAL_BENCH_2_REQUIRED_TASK_FILES if not (task_root / name).is_file()]
    if missing:
        raise TerminalBench2ContractError(
            f"TB2 task {task.id!r} at {task_root} is missing required files: {', '.join(missing)}"
        )

    task_toml_path = task_root / "task.toml"
    payload = _read_task_toml(task_toml_path)
    env = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
    agent = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}
    verifier = payload.get("verifier") if isinstance(payload.get("verifier"), dict) else {}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    docker_image = str(env.get("docker_image") or "").strip()
    if not docker_image:
        raise TerminalBench2ContractError(f"TB2 task {task.id!r} at {task_root} is missing environment.docker_image")

    try:
        agent_timeout_sec = float(agent.get("timeout_sec"))
    except Exception as exc:
        raise TerminalBench2ContractError(f"TB2 task {task.id!r} at {task_root} has invalid agent.timeout_sec") from exc

    try:
        verifier_timeout_sec = float(verifier.get("timeout_sec"))
    except Exception as exc:
        raise TerminalBench2ContractError(
            f"TB2 task {task.id!r} at {task_root} has invalid verifier.timeout_sec"
        ) from exc

    return TerminalBench2TaskContract(
        task_id=task.id,
        task_root=task_root,
        instruction_path=task_root / "instruction.md",
        task_toml_path=task_toml_path,
        environment_dir=task_root / "environment",
        tests_dir=task_root / "tests",
        docker_image=docker_image,
        build_timeout_sec=_resolve_build_timeout_sec(env),
        agent_timeout_sec=agent_timeout_sec,
        verifier_timeout_sec=verifier_timeout_sec,
        cpus=float(env["cpus"]) if env.get("cpus") is not None else None,
        memory=str(env.get("memory") or "").strip(),
        storage=str(env.get("storage") or "").strip(),
        category=str(meta.get("category") or metadata.get("category") or "").strip(),
        difficulty=str(meta.get("difficulty") or metadata.get("difficulty") or "").strip(),
    )


@dataclass
class TerminalBench2Evaluator:
    """Reward-driven evaluator for Harbor-style TB2 task runs."""

    def preflight(self, *, task: TaskSpec) -> dict[str, Any]:
        contract = resolve_terminal_bench_2_task_contract(task)
        return {
            "status": "preflight_ok",
            "instance_id": task.id,
            "task_root": str(contract.task_root),
            "docker_image": contract.docker_image,
            "build_timeout_sec": contract.build_timeout_sec,
            "agent_timeout_sec": contract.agent_timeout_sec,
            "verifier_timeout_sec": contract.verifier_timeout_sec,
            "timeout_source": TB2_TIMEOUT_SOURCE,
            "cpus": contract.cpus,
            "memory": contract.memory,
            "storage": contract.storage,
            "category": contract.category,
            "difficulty": contract.difficulty,
            "required_files": list(TERMINAL_BENCH_2_REQUIRED_TASK_FILES),
        }

    def evaluate(self, *, task: TaskSpec, model_output: str, **kwargs: Any) -> dict[str, Any]:
        preflight = self.preflight(task=task)
        runtime_meta = kwargs.get("runtime_meta") if isinstance(kwargs.get("runtime_meta"), dict) else {}
        reward_raw = runtime_meta.get("reward")
        reward: float | None
        try:
            reward = None if reward_raw is None or reward_raw == "" else float(reward_raw)
        except (TypeError, ValueError):
            reward = None
        # The reward originates from the AGENT-CONTROLLED container's reward
        # file. ``float`` accepts ``nan``/``inf``/``-inf`` without raising, so a
        # buggy or adversarial container could otherwise turn ``inf`` into a
        # solve (``inf >= 1.0``) or leak ``nan`` into ``native_score``. A
        # non-finite reward is not a genuine score -> route it through the
        # existing None/unscored path below.
        reward_was_non_finite = False
        if reward is not None and not math.isfinite(reward):
            reward_was_non_finite = True
            reward = None

        verifier_exit_code = runtime_meta.get("verifier_exit_code")
        agent_exit_code = runtime_meta.get("agent_exit_code")
        resolved = bool(reward is not None and reward >= 1.0)
        trial_status = str(runtime_meta.get("trial_status") or "").strip()
        if trial_status:
            status = trial_status
        elif reward_was_non_finite:
            status = TB2_VERIFIER_MISSING_STATUS
        elif reward is None and verifier_exit_code is None:
            status = TB2_VERIFIER_MISSING_STATUS
        else:
            status = "completed" if verifier_exit_code == 0 or resolved else "verifier_failed"
        return {
            **preflight,
            "status": status,
            # A verifier that never ran yields no genuine score; emit None
            # (unscored) rather than a fabricated 0.0 that would reach
            # _best_scores/record_attempt AND the distillation/seeding prose
            # as a false "genuine failure" learning signal.
            "native_score": (
                reward if reward is not None else (None if status in TB2_VERIFIER_UNSCORED_STATUSES else 0.0)
            ),
            "reward": reward,
            "resolved": resolved,
            "agent_exit_code": agent_exit_code,
            "verifier_exit_code": verifier_exit_code,
            "reward_path": runtime_meta.get("reward_path") or "",
            "ctrf_path": runtime_meta.get("ctrf_path") or "",
            "model_output": model_output,
        }
