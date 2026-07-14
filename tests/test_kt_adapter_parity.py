"""Information-parity contract for the KT adapter memo payload.

The adapter memo is rendered into the recipient agent's MEMORY.md, so the
payload handed to the memo-builder MUST be a subset of what the solver itself
sees. These tests pin that contract at the source (`adapter_task_payload`):

- ARC: train pairs + test INPUTS only — never the hidden test ``output`` (gold
  answer).
- polyglot: problem statement + starter code only — never the hidden grader
  ``test_files`` / ``build_files`` / ``test_command``.
"""

from __future__ import annotations

import json

from ksi.models import TaskSpec
from ksi.orchestrator import kt_adapter as ka


def test_arc_adapter_payload_omits_test_outputs():
    task = TaskSpec(
        id="arc-1",
        prompt="solve",
        metadata={
            "task_source": "arc",
            "arc_train_pairs": [{"input": [[1]], "output": [[2]]}],
            "arc_eval_test_pairs": [{"input": [[3, 3]], "output": [[9, 9]]}],  # 9,9 = hidden answer
            "arc_test_inputs": [{"input": [[3, 3]]}],
        },
    )
    payload = ka.adapter_task_payload(task)
    assert "test_pairs" not in payload
    assert payload.get("test_inputs") == [{"input": [[3, 3]]}]
    # Test inputs must carry no output key, and the hidden answer value must not
    # appear anywhere in the payload. (Train-pair outputs ARE solver-visible and
    # are legitimately retained — so we check the test answer specifically.)
    assert all("output" not in ti for ti in payload["test_inputs"])
    blob = json.dumps(payload).replace(" ", "")
    assert "[[9,9]]" not in blob and "9,9" not in blob


def test_arc_adapter_payload_derives_inputs_when_only_pairs_present():
    """Even if arc_test_inputs is absent, outputs are stripped from raw pairs."""
    task = TaskSpec(
        id="arc-2",
        prompt="solve",
        metadata={
            "task_source": "arc",
            "arc_train_pairs": [{"input": [[1]], "output": [[2]]}],
            "arc_eval_test_pairs": [{"input": [[4]], "output": [[7]]}],
        },
    )
    payload = ka.adapter_task_payload(task)
    assert payload.get("test_inputs") == [{"input": [[4]]}]
    assert "test_pairs" not in payload
    assert all("output" not in ti for ti in payload["test_inputs"])
    # hidden test answer [[7]] must not leak (train output [[2]] may remain)
    assert "[[7]]" not in json.dumps(payload).replace(" ", "")


def test_polyglot_adapter_payload_omits_hidden_grader_files():
    task = TaskSpec(
        id="poly-1",
        prompt="solve",
        metadata={
            "task_source": "polyglot",
            "language": "python",
            "exercise_name": "two-bucket",
            "problem_statement": "statement",
            "starter_code": {"two_bucket.py": "def measure(): ..."},
            "test_files": {"two_bucket_test.py": "assert measure() == SECRET"},
            "build_files": {"Makefile": "secret"},
            "test_command": "pytest -q",
        },
    )
    payload = ka.adapter_task_payload(task)
    for hidden in ("test_files", "build_files", "test_command"):
        assert hidden not in payload, f"{hidden} leaked into adapter payload"
    assert payload["problem_statement"] == "statement"
    assert payload["starter_code"] == {"two_bucket.py": "def measure(): ..."}
    assert "SECRET" not in json.dumps(payload)
