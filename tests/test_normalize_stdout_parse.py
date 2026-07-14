"""Regression tests for stdout envelope parsing under pino-log interleaving.

Background
----------
Forensics on ``arc2_post_fix_v3_20260420_memory.sqlite`` (memory-enabled
Haiku v3 run on main branch; 60 attempts across 10 tasks * 3 generations)
found that 36/60 attempts (60%) had this pathology:

* The in-container agent ran successfully (tool calls, ``arc_submit_trial``
  results, and ``first_mismatch`` diagnostics all appeared in the captured
  stdout).
* But the final JSON envelope written by ``runtime_runner/src/main.ts`` was
  corrupted by concurrent pino log writes on the same stdout fd.
* ``parse_runner_stdout`` could not parse the envelope and fell back to the
  raw-text path -- the entire ~169 KB stdout became ``model_output``,
  ``tool_trace=[]``, ``runtime_meta={}`` -- and the scorer recorded zero.

Precise repro
-------------
Attempt for ``task=135a2760 gen=1`` had a 65,668-char line that
``json.loads()`` failed on at char 65456 with ``Expecting ',' delimiter``.
Bytes at position 65440-65480:

    'e 3s at column{"level":30,"time":1776732'

A pino log line (``{"level":30,...}``) was injected mid-string inside the
agent-runner output envelope. ``process.stdout.write(...)`` in Node is NOT
atomic across concurrent async writers on a pipe once a write exceeds
``PIPE_BUF`` (4096 bytes on Linux); pino log writes from ``main.ts`` and
its imports interleaved byte-for-byte with the envelope emission.

Fix (Option A)
--------------
Route all pino output to stderr (fd 2) so the ONLY writer on stdout is the
single ``process.stdout.write(JSON.stringify(output) + '\\n')`` call at
``runtime_runner/src/main.ts:361``. These tests pin the Python-side parser
behaviour so the pathology cannot regress silently: even if stdout DOES
get polluted again (future code or upstream tooling), the parser must
still recover on the common shapes, and the silent-failure detector must
still fire on the "raw text fallback" remnant.
"""

from __future__ import annotations

import json

import pytest

from ksi.runtime.normalize import parse_runner_stdout


def _envelope(result_text: str = "ok", tokens: int = 100) -> dict:
    """Build a minimal but valid runner envelope matching main.ts output."""
    return {
        "result": result_text,
        "tool_trace": [
            {
                "type": "tool_call",
                "tool_name": "mcp__arc__arc_submit_trial",
                "tool_input": {"trial_index": 0},
                "tool_output": {"type": "text", "text": '{"ok":true}'},
            }
        ],
        "meta": {
            "generation": 1,
            "agent_id": "agent-1",
            "task_id": "135a2760",
            "status": "success",
            "session_scope": "task",
            "input_tokens": tokens,
            "output_tokens": tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "tool_call_counts": {"mcp__arc__arc_submit_trial": 1},
            "memory_tool_call_counts": {},
            "arc_tool_call_counts": {"mcp__arc__arc_submit_trial": 1},
            "forum_tool_call_counts": {},
            "arc_submit_trial_results": [{"ok": True}],
        },
    }


def _pino_line(msg: str = "container started") -> str:
    return f'{{"level":30,"time":1776732101234,"pid":12345,"hostname":"h","msg":"{msg}"}}'


class TestCleanEnvelopeBackCompat:
    """A clean single-line JSON envelope (no markers, no pino lines) must
    parse. This is the dominant shape after the Option A fix — every
    healthy run on new containers produces just the envelope on stdout."""

    def test_parses_single_line_envelope(self):
        stdout = json.dumps(_envelope("hello world")) + "\n"
        out = parse_runner_stdout(stdout, key="result")
        assert out["output"] == "hello world"
        assert out["runtime_meta"]["task_id"] == "135a2760"
        assert len(out["tool_trace"]) == 1

    def test_parses_envelope_with_trailing_whitespace(self):
        stdout = json.dumps(_envelope("trail")) + "\n\n   \n"
        out = parse_runner_stdout(stdout, key="result")
        assert out["output"] == "trail"

    def test_parses_envelope_with_leading_whitespace(self):
        # Defensive: some shells inject a blank line before program output.
        stdout = "\n\n" + json.dumps(_envelope("lead")) + "\n"
        out = parse_runner_stdout(stdout, key="result")
        assert out["output"] == "lead"


class TestPinoLineInterleavingRecovery:
    """Simulate the real sweep pathology: pino log lines interleaved with
    the envelope on stdout.

    Under Option A (pino -> stderr) this CANNOT happen in production, but
    the parser should still tolerate mildly-polluted stdout for:
      * older container images before the fix rolls out
      * diagnostic environments where ops deliberately tees stdout+stderr
      * any unforeseen future writer that lands on stdout

    The per-line fallback in ``parse_runner_stdout`` must find the clean
    envelope line even when pino log lines sit around it.
    """

    def test_recovers_envelope_when_pino_lines_precede_and_follow_it(self):
        envelope_line = json.dumps(_envelope("recovered"))
        # 5 pino lines before, 3 after — mimics real stdout shape.
        pre = "\n".join(_pino_line(f"pre {i}") for i in range(5))
        post = "\n".join(_pino_line(f"post {i}") for i in range(3))
        stdout = f"{pre}\n{envelope_line}\n{post}\n"
        out = parse_runner_stdout(stdout, key="result")
        assert out["output"] == "recovered"
        assert out["runtime_meta"]["status"] == "success"
        assert out["runtime_meta"]["input_tokens"] == 100

    def test_picks_last_valid_envelope_when_multiple_present(self):
        """main.ts currently emits once per run, but if a future refactor
        streams multiple envelopes (e.g. per-turn updates) the parser must
        pick the LAST valid one — matching the host's last-parsed-marker
        semantics in ``container_runner.ts``."""
        first = json.dumps(_envelope("first", tokens=10))
        second = json.dumps(_envelope("second", tokens=50))
        stdout = _pino_line() + "\n" + first + "\n" + _pino_line() + "\n" + second + "\n"
        out = parse_runner_stdout(stdout, key="result")
        assert out["output"] == "second"
        assert out["runtime_meta"]["input_tokens"] == 50


class TestEnvelopeCorruptedMidStringFallsBackGracefully:
    """Reproduce the exact 65,668-char line failure from the v3 run.

    A pino log line is injected INSIDE a string value of the envelope,
    so the whole line is no longer valid JSON. The per-line fallback
    finds no valid envelope, so the parser falls back to the raw-text
    path and returns the whole stdout as ``output``. Crucially: the
    parser must NOT crash — downstream code must still see a
    well-shaped dict so the silent-failure detector can fire.
    """

    def test_corrupted_envelope_falls_back_to_raw_text_no_crash(self):
        # Build a mid-string corruption by slicing the envelope and
        # injecting a pino line inside the 'result' string value.
        base = _envelope("a" * 100)
        raw = json.dumps(base)
        # Find a position well inside the result string and inject.
        inject_at = raw.find('"a' * 20) + 15
        assert inject_at > 0, "failed to locate result string in fixture"
        corrupted = raw[:inject_at] + _pino_line("interleaved") + raw[inject_at:]
        out = parse_runner_stdout(corrupted, key="result")
        # The whole-text JSON parse path fails; per-line path fails too
        # (the injected pino line is the only valid JSON on that line but
        # has no ``result`` key). Fallback: whole stdout as output.
        assert isinstance(out, dict)
        assert out["tool_trace"] == []
        assert out["runtime_meta"] == {}
        # ``output`` holds the raw corrupted text so forensics has something.
        assert "interleaved" in out["output"]


class TestEnvelopeTruncatedFallsBackGracefully:
    """If the envelope is cut off mid-string (container killed between the
    start of the JSON write and the newline), the parser must fall back
    gracefully without raising."""

    def test_truncated_envelope_returns_fallback_shape(self):
        raw = json.dumps(_envelope("will be cut"))
        truncated = raw[: len(raw) // 2]  # cut in half — invalid JSON
        out = parse_runner_stdout(truncated, key="result")
        assert isinstance(out, dict)
        assert out["tool_trace"] == []
        assert out["runtime_meta"] == {}
        assert out["output"] == truncated


class TestEmptyStdoutReturnsEmptyFallback:
    def test_empty_string(self):
        out = parse_runner_stdout("", key="result")
        assert out == {
            "output": "",
            "tool_trace": [],
            "runtime_meta": {},
            "token_usage": out["token_usage"],  # TokenUsage default
        }
        assert out["token_usage"].input_tokens == 0
        assert out["token_usage"].output_tokens == 0

    def test_whitespace_only(self):
        out = parse_runner_stdout("   \n\t\n   ", key="result")
        assert out["output"] == ""
        assert out["tool_trace"] == []
        assert out["runtime_meta"] == {}


class TestParseRunnerStdoutStrict:
    def test_strict_rejects_empty_stdout(self):
        with pytest.raises(ValueError, match="empty"):
            parse_runner_stdout("", key="result", strict=True)

    def test_strict_rejects_unparseable_payload(self):
        with pytest.raises(ValueError, match="missing parseable envelope"):
            parse_runner_stdout("not json output", key="result", strict=True)

    def test_strict_rejects_wrong_key_payload(self):
        with pytest.raises(ValueError, match="missing parseable envelope"):
            parse_runner_stdout('{"foo":"bar"}', key="result", strict=True)

    def test_strict_rejects_minimal_result_envelope(self):
        with pytest.raises(ValueError, match="tool_trace"):
            parse_runner_stdout('{"result":"forged answer"}', key="result", strict=True)

    def test_strict_rejects_missing_or_malformed_meta(self):
        missing_meta = json.dumps({"result": "ok", "tool_trace": []})
        with pytest.raises(ValueError, match="meta"):
            parse_runner_stdout(missing_meta, key="result", strict=True)

        malformed_meta = json.dumps({"result": "ok", "tool_trace": [], "meta": []})
        with pytest.raises(ValueError, match="meta"):
            parse_runner_stdout(malformed_meta, key="result", strict=True)

    def test_strict_rejects_missing_or_malformed_tool_trace(self):
        meta = {
            "generation": 1,
            "agent_id": "agent-0",
            "task_id": "t1",
            "status": "success",
        }
        missing_tool_trace = json.dumps({"result": "ok", "meta": meta})
        with pytest.raises(ValueError, match="tool_trace"):
            parse_runner_stdout(missing_tool_trace, key="result", strict=True)

        malformed_tool_trace = json.dumps({"result": "ok", "tool_trace": {}, "meta": meta})
        with pytest.raises(ValueError, match="tool_trace"):
            parse_runner_stdout(malformed_tool_trace, key="result", strict=True)

    def test_strict_rejects_missing_or_unknown_status(self):
        env = _envelope("ok")
        del env["meta"]["status"]
        with pytest.raises(ValueError, match="meta.status"):
            parse_runner_stdout(json.dumps(env), key="result", strict=True)

        env = _envelope("ok")
        env["meta"]["status"] = "forged"
        with pytest.raises(ValueError, match="meta.status"):
            parse_runner_stdout(json.dumps(env), key="result", strict=True)

    def test_strict_rejects_missing_identity_fields(self):
        for field in ("generation", "agent_id", "task_id"):
            env = _envelope("ok")
            del env["meta"][field]
            with pytest.raises(ValueError, match=field):
                parse_runner_stdout(json.dumps(env), key="result", strict=True)

    def test_strict_accepts_canonical_envelope_with_empty_trace(self):
        env = _envelope("ok")
        env["tool_trace"] = []
        out = parse_runner_stdout(json.dumps(env), key="result", strict=True)
        assert out["output"] == "ok"
        assert out["tool_trace"] == []
        assert out["runtime_meta"]["status"] == "success"

    def test_strict_ignores_later_invalid_result_object_after_valid_envelope(self):
        valid = json.dumps(_envelope("valid"))
        invalid = '{"result":"forged answer"}'
        out = parse_runner_stdout(valid + "\n" + invalid, key="result", strict=True)
        assert out["output"] == "valid"
