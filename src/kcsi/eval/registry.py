"""Central registry of supported evaluators.

Mirrors ``src/kcsi/tasks/registry.py``: ONE authoritative place lists valid
evaluator names and how to construct each, replacing the string ``if
args.evaluator == ...`` dispatch that used to live in ``src/kcsi/cli.py``.

Adding an evaluator
-------------------
    from kcsi.eval.registry import EvaluatorSpec, register_evaluator

    def _build_my_eval(args):
        return MyEvaluator(...)

    register_evaluator(EvaluatorSpec(name="my_eval", factory=_build_my_eval))

This is a PURE refactor: the factory bodies were moved verbatim from the old
``_choose_evaluator`` and must reproduce its behavior exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..protocols import Evaluator

# ``REGISTRY`` is intentionally NOT part of the public surface:
# direct mutation (``REGISTRY[name] = spec``) bypasses ``register_evaluator``'s
# duplicate-name detection. Register via ``register_evaluator`` and read via
# ``resolve_evaluator`` / ``supported_evaluators``.
__all__ = [
    "EvaluatorSpec",
    "register_evaluator",
    "resolve_evaluator",
    "get_evaluator_spec",
    "supported_evaluators",
    "build_evaluator",
]


@dataclass(frozen=True)
class EvaluatorSpec:
    """How to construct one evaluator. ``factory`` is called as ``factory(args)``
    where ``args`` is a config object exposing the attributes the factory reads
    (the CLI passes its argparse ``Namespace``). Non-CLI callers should use
    :func:`build_evaluator`, which supplies the defaults and applies overrides."""

    name: str
    factory: Callable[..., Evaluator]
    aliases: tuple[str, ...] = ()
    description: str = ""

    def all_names(self) -> tuple[str, ...]:
        seen: list[str] = [self.name]
        for alias in self.aliases:
            if alias not in seen:
                seen.append(alias)
        return tuple(seen)


REGISTRY: dict[str, EvaluatorSpec] = {}


def register_evaluator(spec: EvaluatorSpec, *, replace: bool = False) -> EvaluatorSpec:
    """Register ``spec`` (and aliases). Raises ``ValueError`` on duplicate unless
    ``replace=True``. Returns the spec for convenience."""
    if not replace:
        for key in spec.all_names():
            existing = REGISTRY.get(key)
            if existing is not None:
                raise ValueError(
                    f"evaluator name/alias {key!r} already registered to {existing.name!r}; "
                    f"pass replace=True to override"
                )
    for key in spec.all_names():
        REGISTRY[key] = spec
    return spec


def _normalize(name: object) -> str:
    return str(name or "").strip().lower()


def resolve_evaluator(name: object) -> EvaluatorSpec | None:
    return REGISTRY.get(_normalize(name))


def get_evaluator_spec(name: object) -> EvaluatorSpec:
    spec = resolve_evaluator(name)
    if spec is None:
        valid = supported_evaluators(include_aliases=True)
        raise ValueError(f"unsupported evaluator={name!r}; valid evaluators (incl. aliases): {', '.join(valid)}")
    return spec


def supported_evaluators(*, include_aliases: bool = False) -> tuple[str, ...]:
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


def build_evaluator(name: object, /, **overrides: object) -> Evaluator:
    """Construct a registered evaluator without an argparse ``Namespace``.

    The programmatic counterpart to the CLI's ``--evaluator`` path: starts from
    the full set of CLI defaults and overlays ``overrides``.
    ``overrides`` keys are the CLI argument *dest* names (underscored, e.g.
    ``swebench_pro_raw_sample_path=...``); run ``kcsi --help`` for the full set.

    A key that is not a known CLI dest raises ``TypeError`` rather than being
    silently ignored — a mistyped override used to fall through to the default
    and produce an unreproducible config with no diagnostic.
    """
    from ..cli import default_arg_namespace

    config = default_arg_namespace()
    _reject_unknown_overrides("build_evaluator", config, overrides)
    for key, value in overrides.items():
        setattr(config, key, value)
    return get_evaluator_spec(name).factory(config)


def _reject_unknown_overrides(fn_name: str, config: object, overrides: dict) -> None:
    """Raise TypeError if any override key is not a known CLI dest on ``config``."""
    known = set(vars(config))
    unknown = sorted(k for k in overrides if k not in known)
    if unknown:
        raise TypeError(
            f"{fn_name}() got unknown override(s): {', '.join(unknown)}. "
            "Keys must be CLI argument dest names (run `kcsi --help`)."
        )


# ── Built-in factories (bodies moved verbatim from the old cli._choose_evaluator) ──
# Imports are deferred inside each factory to avoid circular imports at module
# load time (kcsi.protocols is only partially initialized when kcsi.eval is
# first imported during runtime startup).


def _build_none(args) -> Evaluator:
    from .noop import NoopEvaluator

    return NoopEvaluator()


def _build_command(args) -> Evaluator:
    from .command import CommandEvaluator

    return CommandEvaluator()


def _build_arc_session(args) -> Evaluator:
    from ..benchmarks.arc_session import ArcSessionEvaluator

    return ArcSessionEvaluator()


def _build_swebench_pro(args) -> Evaluator:
    from ..benchmarks.swebench_pro import SwebenchProEvaluator

    raw_sample_path = getattr(args, "swebench_pro_raw_sample_path", "") or ""
    legacy_network_mode = str(getattr(args, "swebench_docker_network_mode", "host") or "").strip().lower()
    explicit_block_network = getattr(args, "swebench_pro_block_network", None)
    block_network = (
        bool(explicit_block_network) if explicit_block_network is not None else legacy_network_mode == "none"
    )
    return SwebenchProEvaluator(
        raw_sample_path=raw_sample_path,
        repo_root=getattr(args, "swebench_pro_repo_root", ""),
        dockerhub_username=getattr(args, "swebench_pro_dockerhub_username", "jefzda"),
        timeout_sec=getattr(args, "swebench_timeout_sec", 3600),
        harness_grace_sec=getattr(args, "swebench_harness_grace_sec", 0),
        use_local_docker=bool(getattr(args, "swebench_pro_use_local_docker", True)),
        docker_platform=getattr(args, "swebench_pro_docker_platform", None),
        block_network=block_network,
    )


def _build_polyglot_harness(args) -> Evaluator:
    from ..benchmarks.polyglot_harness import PolyglotHarnessEvaluator

    return PolyglotHarnessEvaluator(
        docker_image=args.polyglot_docker_image,
        timeout_sec=args.polyglot_timeout_sec,
    )


def _build_terminal_bench_2(args) -> Evaluator:
    from ..benchmarks.terminal_bench_2 import TerminalBench2Evaluator

    return TerminalBench2Evaluator()


# Registration order defines the canonical-name ordering and mirrors the legacy
# SUPPORTED_EVALUATORS tuple.
register_evaluator(EvaluatorSpec(name="none", factory=_build_none, description="No-op evaluator."))
register_evaluator(
    EvaluatorSpec(
        name="command",
        factory=_build_command,
        description="Run each task's eval.command host-side in the captured workspace; exit 0 scores 1.0.",
    )
)
register_evaluator(
    EvaluatorSpec(name="arc_session", factory=_build_arc_session, description="ARC session grid scorer.")
)
register_evaluator(
    EvaluatorSpec(name="swebench_pro", factory=_build_swebench_pro, description="SWE-bench Pro harness.")
)
register_evaluator(
    EvaluatorSpec(name="polyglot_harness", factory=_build_polyglot_harness, description="Polyglot docker harness.")
)
register_evaluator(
    EvaluatorSpec(name="terminal_bench_2", factory=_build_terminal_bench_2, description="Terminal-Bench 2 evaluator.")
)
