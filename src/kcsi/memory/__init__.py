"""Agent memory system — SQLite + sqlite-vec + FTS5.

Flat-mount invariant
--------------------
The container runtime mounts this directory flat at ``/app/memory`` and
invokes ``mcp_server.py`` directly as a top-level script::

    python3 /app/memory/mcp_server.py

Because ``mcp_server.py`` is run as a script (not as part of an installed
package), it cannot rely on relative package imports.  It therefore uses a
``try/except ImportError`` fallback that imports sibling modules by bare name::

    try:
        from .knowledge_store import KnowledgeStore   # installed-package path
        ...
    except ImportError:
        from knowledge_store import KnowledgeStore    # script-mode fallback
        ...

For these fallback imports to succeed, **every module that ``mcp_server.py``
imports must be a direct sibling of ``mcp_server.py``** inside this directory.
Splitting any of them into a sub-package would break the script-mode path.

**Do NOT introduce sub-packages inside ``src/kcsi/memory/``.**

References:
- ``src/kcsi/runtime/container_host.py`` — the ``mcp_server_dir`` key sets the
  mounted path to ``Path(__file__).parent.parent / "memory"``.
- ``src/kcsi/memory/mcp_server.py`` lines 40-52 — the ``try/except ImportError``
  block that requires all modules to be flat siblings.
- ``tests/memory/test_flat_mount_invariant.py`` — automated guard that asserts
  each mount-critical module lives directly in this directory.
"""
