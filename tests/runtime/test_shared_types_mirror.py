"""Guard: the host and in-container ``shared_types.ts`` copies stay identical.

The container I/O contract between the host runner (``container_runner.ts``)
and the in-container agent runners (``index.ts``, ``openai.ts``) is declared in
two TypeScript files that are SEPARATE compilation units (host-side ``tsx`` vs
container-side ``npm``/``tsc``), so the interfaces must be duplicated:

  - ``runtime_runner/src/shared_types.ts``            (host-side)
  - ``runtime_runner/agent-runner/src/shared_types.ts`` (container-side)

Both files declare ``mirrored here verbatim``. If they drift, the host and the
container can silently disagree on the wire shape (a field added on one side is
ignored on the other), which fails at runtime, not at typecheck. This test pins
them byte-identical so any edit to one MUST be applied to both — the same
single-source-of-truth discipline used for ``shared/retryable_markers.json``
(see ``tests/test_retryable_markers.py``).
"""

from __future__ import annotations

from conftest import REPO_ROOT

HOST_COPY = REPO_ROOT / "runtime_runner" / "src" / "shared_types.ts"
CONTAINER_COPY = REPO_ROOT / "runtime_runner" / "agent-runner" / "src" / "shared_types.ts"


def test_shared_types_copies_are_byte_identical() -> None:
    assert HOST_COPY.exists(), f"missing host-side shared types: {HOST_COPY}"
    assert CONTAINER_COPY.exists(), f"missing container-side shared types: {CONTAINER_COPY}"

    host = HOST_COPY.read_text(encoding="utf-8")
    container = CONTAINER_COPY.read_text(encoding="utf-8")

    assert host == container, (
        "runtime_runner/src/shared_types.ts and "
        "runtime_runner/agent-runner/src/shared_types.ts have drifted.\n"
        "These declare the same host<->container I/O contract and must stay "
        "byte-identical (each file says it is 'mirrored verbatim'). Apply your "
        "change to BOTH copies."
    )
