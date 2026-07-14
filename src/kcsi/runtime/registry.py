"""Central registry of supported execution runtimes.

Mirrors ``src/kcsi/tasks/registry.py`` / ``src/kcsi/eval/registry.py``. ONE place
lists valid ``--runtime`` values and how to build each. The Terminal-Bench-2
runtime-delegation path is NOT a registry entry: it is selected by
``task_source`` (``spec.delegates_runtime``) and applied in
``cli._choose_runtime`` as a post-construction wrapper over the base runtime.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..protocols import RuntimeExecutor

log = logging.getLogger(__name__)

# ``REGISTRY`` is intentionally NOT part of the public surface:
# direct mutation (``REGISTRY[name] = spec``) bypasses ``register_runtime``'s
# duplicate-name detection. Register via ``register_runtime`` and read via
# ``resolve_runtime`` / ``supported_runtimes``.
__all__ = [
    "RuntimeSpec",
    "register_runtime",
    "resolve_runtime",
    "get_runtime_spec",
    "supported_runtimes",
    "build_runtime",
]


@dataclass(frozen=True)
class RuntimeSpec:
    """How to construct one runtime. ``factory`` is called as
    ``factory(args, provider_env)`` -> RuntimeExecutor, where ``args`` is a
    config object exposing the attributes the factory reads (the CLI passes its
    argparse ``Namespace``). Non-CLI callers should use :func:`build_runtime`."""

    name: str
    factory: Callable[..., "RuntimeExecutor"]
    aliases: tuple[str, ...] = ()
    description: str = ""

    def all_names(self) -> tuple[str, ...]:
        seen: list[str] = [self.name]
        for alias in self.aliases:
            if alias not in seen:
                seen.append(alias)
        return tuple(seen)


REGISTRY: dict[str, RuntimeSpec] = {}


def register_runtime(spec: RuntimeSpec, *, replace: bool = False) -> RuntimeSpec:
    """Register ``spec`` (and aliases). Raises ``ValueError`` on duplicate unless
    ``replace=True``. Returns the spec for convenience."""
    if not replace:
        for key in spec.all_names():
            existing = REGISTRY.get(key)
            if existing is not None:
                raise ValueError(
                    f"runtime name/alias {key!r} already registered to {existing.name!r}; pass replace=True to override"
                )
    for key in spec.all_names():
        REGISTRY[key] = spec
    return spec


def _normalize(name: object) -> str:
    return str(name or "").strip().lower()


def resolve_runtime(name: object) -> "RuntimeSpec | None":
    return REGISTRY.get(_normalize(name))


def get_runtime_spec(name: object) -> RuntimeSpec:
    spec = resolve_runtime(name)
    if spec is None:
        valid = supported_runtimes(include_aliases=True)
        raise ValueError(f"unsupported runtime={name!r}; valid runtimes (incl. aliases): {', '.join(valid)}")
    return spec


def supported_runtimes(*, include_aliases: bool = False) -> tuple[str, ...]:
    canonical: list[str] = []
    for spec in REGISTRY.values():
        if spec.name not in canonical:
            canonical.append(spec.name)
    if not include_aliases:
        return tuple(canonical)
    out = list(canonical)
    for spec in REGISTRY.values():
        for alias in spec.aliases:
            if alias not in out:
                out.append(alias)
    return tuple(out)


def build_runtime(
    name: object,
    /,
    provider_env: "dict[str, str] | None" = None,
    **overrides: object,
) -> "RuntimeExecutor":
    """Construct a registered runtime without an argparse ``Namespace``.

    The programmatic counterpart to the CLI's ``--runtime`` path: starts from the
    full set of CLI defaults and overlays ``overrides``.
    Returns the base runtime; the Terminal-Bench-2 runtime-delegation wrapper is
    a CLI-only concern keyed on ``task_source`` and is not applied here.

    ``overrides`` keys are the CLI argument *dest* names (underscored, e.g.
    ``knowledge_db_path=...``); run ``kcsi --help`` for the full set. An
    unknown/mistyped override key raises ``TypeError`` rather than silently
    falling back to the default and producing an unreproducible config.
    """
    from ..cli import default_arg_namespace

    config = default_arg_namespace()
    known = set(vars(config))
    unknown = sorted(k for k in overrides if k not in known)
    if unknown:
        raise TypeError(
            f"build_runtime() got unknown override(s): {', '.join(unknown)}. "
            "Keys must be CLI argument dest names (run `kcsi --help`)."
        )
    for key, value in overrides.items():
        setattr(config, key, value)
    return get_runtime_spec(name).factory(config, provider_env or {})


# ── helpers copied VERBATIM from cli.py ──
def _ensure_runtime_runner_deps(project_root: str) -> None:
    runtime_runner_dir = Path(project_root) / "runtime_runner"
    package_json = runtime_runner_dir / "package.json"
    if not package_json.exists():
        return

    required_paths = (
        runtime_runner_dir / "node_modules" / ".bin" / "tsx",
        runtime_runner_dir / "node_modules" / "pino" / "package.json",
    )
    if all(path.exists() for path in required_paths):
        return

    install_cmd = [
        (os.environ.get("NPM_BIN") or "npm").strip() or "npm",
        "--prefix",
        str(runtime_runner_dir),
        "ci" if (runtime_runner_dir / "package-lock.json").exists() else "install",
        "--silent",
        "--no-audit",
        "--no-fund",
    ]
    log.info("Bootstrapping runtime_runner npm dependencies in %s", runtime_runner_dir)
    try:
        proc = subprocess.run(
            install_cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "runtime_runner dependencies are missing and npm is unavailable. "
            "Install Node/npm, then run `npm --prefix runtime_runner ci`."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Timed out while bootstrapping runtime_runner dependencies. Retry with `npm --prefix runtime_runner ci`."
        ) from exc

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-2000:]
        stdout_tail = (proc.stdout or "")[-1000:]
        detail = stderr_tail or stdout_tail or "(no npm output)"
        raise RuntimeError(f"Failed to bootstrap runtime_runner dependencies with `{' '.join(install_cmd)}`:\n{detail}")


def _default_container_command(project_root: str) -> list[str]:
    candidates = [
        Path(project_root) / "runtime_runner" / "node_modules" / ".bin" / "tsx",
        Path(project_root) / "node_modules" / ".bin" / "tsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate), "runtime_runner/src/main.ts"]
    return ["npx", "--yes", "--prefix", "runtime_runner", "tsx", "runtime_runner/src/main.ts"]


def _build_container(args, provider_env: "dict[str, str] | None" = None):
    # repo root = parent of kcsi/ pkg dir; from this file that is parents[3].
    project_root = str(Path(__file__).resolve().parents[3])
    provider_env = provider_env or {}
    if not str(args.container_command or "").strip():
        _ensure_runtime_runner_deps(project_root)
    container_command = (
        args.container_command.split()
        if str(args.container_command or "").strip()
        else _default_container_command(project_root)
    )
    from .container_host import KcsiContainerExecutor

    # The CLI normalizes ``runtime_timeout_sec`` to an int in
    # ``_validate_and_normalize_args``, but the programmatic ``build_runtime``
    # path builds its namespace from raw arg defaults (now ``None``) and never
    # runs that validator. A leaked ``None`` would ``TypeError`` in
    # ``_build_runner_env``'s ``if timeout_sec > 0``, so fall back to the
    # standard default here.
    timeout_sec = args.runtime_timeout_sec
    if timeout_sec is None:
        from ..cli import _resolve_runtime_timeout_default

        timeout_sec = _resolve_runtime_timeout_default()

    return KcsiContainerExecutor(
        command=container_command,
        working_dir=project_root,
        timeout_sec=timeout_sec,
        session_scope=args.session_scope,
        # Accept both the CLI string form ("true"/"false") and a real bool from
        # programmatic ``build_runtime(..., wipe_workspace_per_task=True)``
        # overrides. The bare ``== "true"`` compared a Python bool to a string
        # and silently yielded False, leaving workspaces un-wiped with no error.
        wipe_workspace_per_task=(
            args.wipe_workspace_per_task
            if isinstance(args.wipe_workspace_per_task, bool)
            else str(args.wipe_workspace_per_task).strip().lower() == "true"
        ),
        knowledge_db_path=args.knowledge_db_path,
        runtime_db_path=getattr(args, "runtime_db_path", ""),
        disable_memory_mcp=args.disable_memory_mcp or getattr(args, "no_memory", False),
        forum_timeout_sec=args.forum_timeout_sec,
        cross_task_forum_timeout_sec=getattr(args, "cross_task_forum_timeout_sec", args.forum_timeout_sec),
        phase1_reflection_enabled=bool(getattr(args, "phase1_reflection_enabled", False)),
        memory_seed_raw_attempts=bool(getattr(args, "memory_seed_raw_attempts", False)),
        arc_no_mcp=True,
        env=provider_env,
    )


register_runtime(
    RuntimeSpec(
        name="container",
        factory=_build_container,
        description="Shared container runtime.",
    )
)
