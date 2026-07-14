"""Reference benchmark integrations (ARC, SWE-bench Pro, Polyglot, Terminal-Bench 2).

Core imports this package only via the registry wiring; new deployments don't
need it.

``register_all()`` is the SINGLE core->benchmarks wiring point: it registers
the four built-in ``TaskSourceSpec``s (``ksi.benchmarks.sources``) and
attaches their loader callables (``ksi.benchmarks.loaders.attach_benchmark_loaders``).
``ksi.tasks.loaders`` calls it at import time, immediately followed by the
``custom`` source's own registration, so the canonical registration order
stays ``swebench_pro, arc, polyglot, terminal_bench_2, custom``.

``register_all`` and ``_registered`` are defined here BEFORE the eager
evaluator-class imports below on purpose: some of those evaluator modules
(``swebench_pro``, ``terminal_bench_2``) import from ``ksi.tasks`` at their
own module top level, which — the first time anything touches ``ksi.tasks``
or ``ksi.benchmarks`` — can re-enter this module while it is still mid-import
(via ``ksi.tasks.loaders``' own call to ``register_all()``). Defining the
function first means that re-entrant ``from ..benchmarks import register_all``
always finds a bound name, regardless of which package is imported first.
"""

_registered = False


def register_all() -> None:
    """Register the built-in benchmark task sources and their loaders (idempotent)."""
    global _registered
    if _registered:
        return
    from . import sources  # noqa: F401  (import registers the 4 TaskSourceSpecs)
    from .loaders import attach_benchmark_loaders

    attach_benchmark_loaders()
    _registered = True


from .arc_session import ArcSessionEvaluator
from .polyglot_harness import PolyglotHarnessEvaluator
from .swebench_pro import SwebenchProEvaluator
from .terminal_bench_2 import TerminalBench2Evaluator

__all__ = [
    "ArcSessionEvaluator",
    "PolyglotHarnessEvaluator",
    "SwebenchProEvaluator",
    "TerminalBench2Evaluator",
    "register_all",
]
