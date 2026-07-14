"""Guard: all mount-critical memory modules must be direct siblings.

WHY THIS TEST EXISTS
--------------------
The container runtime mounts ``src/ksi/memory/`` flat at ``/app/memory``
and runs ``mcp_server.py`` as a top-level script (not as part of an installed
package).  Because of this, ``mcp_server.py`` imports its sibling modules via
bare names as a fallback::

    try:
        from .knowledge_store import KnowledgeStore
        ...
    except ImportError:
        from knowledge_store import KnowledgeStore   # script-mode path
        ...

If any of the listed modules were moved into a sub-package, the script-mode
import would raise ``ModuleNotFoundError`` and the MCP server would fail to
start inside the container.

See also:
- ``src/ksi/memory/__init__.py`` — module docstring explaining the invariant
- ``src/ksi/runtime/container_host.py`` — ``mcp_server_dir`` mount definition
- ``src/ksi/memory/mcp_server.py`` lines 40-52 — the fallback import block
"""

import pathlib

import ksi.memory

# Modules that mcp_server.py reaches as flat siblings under script-mode import,
# directly or transitively, and that must therefore live in the memory directory.
# Extend this list whenever a new sibling import is added ANYWHERE mcp_server.py
# can pull it in — not only the top-of-file try/except block:
#   - mcp_server.py's top try/except: knowledge_store, store, arc_semantics, forum_bus
#   - lazy try/except in function bodies: embeddings, parity
#   - transitive siblings of those (e.g. _store_common, imported by store/knowledge_store)
MOUNT_CRITICAL_MODULES = [
    "mcp_server.py",
    "knowledge_store.py",
    "knowledge_store_migrations.py",
    "store.py",
    "arc_semantics.py",
    "forum_bus.py",
    "_store_common.py",
    "embeddings.py",
    "parity.py",
]


def test_mount_critical_modules_are_flat_siblings() -> None:
    """Each mount-critical module must be a direct child of the memory dir."""
    memory_dir = pathlib.Path(ksi.memory.__file__).parent

    missing = [name for name in MOUNT_CRITICAL_MODULES if not (memory_dir / name).is_file()]

    assert not missing, (
        f"The following mount-critical modules are missing from {memory_dir}: "
        f"{missing}. "
        "Moving them into a sub-package breaks mcp_server.py's script-mode "
        "imports when the container runs python3 /app/memory/mcp_server.py. "
        "See src/ksi/memory/__init__.py for the full invariant explanation."
    )


def test_no_sub_packages_in_memory_dir() -> None:
    """The memory directory must not contain any sub-packages.

    Sub-packages would break the flat /app/memory container mount: Python
    resolves bare 'from knowledge_store import ...' only when all siblings
    are top-level .py files, not when they hide inside nested __init__.py
    trees.
    """
    memory_dir = pathlib.Path(ksi.memory.__file__).parent

    sub_packages = [p.name for p in memory_dir.iterdir() if p.is_dir() and (p / "__init__.py").is_file()]

    assert not sub_packages, (
        f"Sub-packages found in {memory_dir}: {sub_packages}. "
        "The memory directory must remain flat (no nested packages). "
        "See src/ksi/memory/__init__.py for the full invariant explanation."
    )
