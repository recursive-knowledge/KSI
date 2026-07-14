"""Shared AST guard for phase-service / engine decoupling tests (#912).

After the per-phase collaborator decoupling, each ``Engine*PhaseService``
module may reference ``engine._<private>`` attributes only inside its single
``_collaborators`` factory method. Every other function/method body must depend
on the explicit ``*Collaborators`` object instead. These helpers let each
phase's test assert that invariant directly against the module source.
"""

from __future__ import annotations

import ast
from pathlib import Path


def functions_referencing_engine(module_path: str) -> set[str]:
    """Return names of functions/methods whose body references ``engine`` at all.

    A function is flagged if its body reaches the engine in EITHER form:

    1. via the local ``engine`` alias — any ``Name`` node bound to ``engine``,
       covering attribute access (``engine._x`` / public ``engine.config``), the
       string-keyed ``getattr(engine, "_x")`` / ``setattr(engine, "_x", v)``
       reach-through forms, and passing ``engine`` onward as an argument;
    2. via direct ``self.engine`` access WITHOUT first aliasing — an ``Attribute``
       node ``self.engine`` (e.g. ``self.engine._x`` / ``getattr(self.engine, …)``
       / passing ``self.engine`` along). Without this second check a phase body
       could re-couple as ``self.engine._best_scores`` and slip past a guard that
       only matched a bare ``engine`` ``Name`` — the blind spot this closes.

    Both forms are confined to the single ``_collaborators`` factory (which does
    ``engine = self.engine`` — itself a ``self.engine`` access, so it is
    legitimately flagged and allow-listed by each per-module test).

    Nested functions are reported under their own name (``ast.walk`` visits every
    ``FunctionDef``), so a closure that still reaches the engine is caught
    independently of its enclosing method. Setter lambdas inside the factory are
    ``ast.Lambda`` nodes (not ``FunctionDef``), so their ``engine`` references
    are attributed to the enclosing ``_collaborators`` and stay confined there.
    """
    tree = ast.parse(Path(module_path).read_text())
    offenders: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == "engine":
                offenders.add(node.name)
            elif (
                isinstance(sub, ast.Attribute)
                and sub.attr == "engine"
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
            ):
                offenders.add(node.name)
    return offenders
