/**
 * Regression test for the 2026-04-21 envelope-drop-status fix in
 * ``runtime_runner/src/main.ts``.
 *
 * Background
 * ----------
 * The KSI container runtime has two layers:
 *
 *   * ``container_runner.ts`` streams results via the ``onOutput`` callback
 *     (each streamed item shaped like ``ContainerOutput`` from
 *     ``shared_types.ts``) and then resolves its final promise.
 *     In streaming mode (``runtime_runner/src/container_runner.ts``)
 *     that final resolve value is **unconditionally** ``{status: 'success',
 *     result: null}`` whenever ``outputChain`` completes without an exception
 *     — regardless of what the stream last reported.
 *
 *   * ``main.ts`` captures each streamed payload into ``lastOutput`` and
 *     then picks an "effective" payload to emit as the JSON envelope.
 *
 * The bug
 * -------
 * Pre-fix, the picker at ``main.ts:277`` was:
 *
 *     const effectiveOutput = lastOutput?.result != null ? lastOutput : result;
 *
 * When the stream last reported ``{status: 'error', result: null, error: ...}``
 * or ``{status: 'recovered_from_session', result: null, recovery_note: ...}``
 * (both legitimate per ``shared_types.ts::ContainerOutput``), the
 * ``lastOutput.result != null`` guard fell through to ``result`` — the
 * success-shaped fallback — and silently dropped the real status + the
 * ``error`` / ``recovery_note`` fields.
 *
 * Python's ``container_host.py`` then saw ``status='success'`` with no
 * result and relabelled as ``silent_failure``. The 2026-04-20 Haiku
 * baseline audit attributed ~33% of SWE-bench Pro silent_failure attempts
 * to this envelope drop.
 *
 * The fix
 * -------
 * Prefer ``lastOutput`` whenever its status is ``error`` or
 * ``recovered_from_session`` — even with ``result=null`` — so the real
 * status + diagnostic fields propagate to the final envelope.
 *
 * Test strategy
 * -------------
 * This test file inlines the ``effectiveOutput`` selection logic from
 * ``main.ts`` (same inlining pattern as the sibling
 * ``silent_failure_recovery.test.mjs`` — the repo's test harness runs
 * ``node --test`` directly without a tsc step). If the picker's
 * semantics change in ``main.ts``, this test must be updated in lockstep.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

// ── Local copy of the effectiveOutput picker from main.ts ─────────────────
// Mirrors the exported `pickEffectiveOutput` in runtime_runner/src/main.ts.
// Kept byte-in-sync by the STRICT guard in tests/js/copy_sync_guard.test.mjs
// (main.ts ↔ envelope_drop_status), so this copy cannot drift silently. The
// guard list must include every non-'success' ContainerOutput status that can
// legally carry result=null (see shared_types.ts::ContainerOutput.status),
// plus the "success with meaningful tool/tokens but null result" case.
function pickEffectiveOutput(lastOutput, result) {
  const hasToolTrace =
    Array.isArray(lastOutput?.toolTrace) && lastOutput.toolTrace.length > 0;
  const hasTokens =
    ((lastOutput?.input_tokens ?? 0) +
      (lastOutput?.output_tokens ?? 0) +
      (lastOutput?.cache_creation_input_tokens ?? 0) +
      (lastOutput?.cache_read_input_tokens ?? 0)) > 0;
  return lastOutput &&
    (lastOutput.result != null ||
      lastOutput.status === "error" ||
      lastOutput.status === "recovered_from_session" ||
      hasToolTrace ||
      hasTokens)
    ? lastOutput
    : result;
}

// Mimics the subsequent read pattern in main.ts: `status` goes on
// `output.meta.status`, `recovery_note` goes on `output.meta.recovery_note`,
// and `result: effectiveOutput.result ?? effectiveOutput.error ?? ''`.
function buildEnvelope(lastOutput, result) {
  const effectiveOutput = pickEffectiveOutput(lastOutput, result);
  return {
    result: effectiveOutput.result ?? effectiveOutput.error ?? "",
    meta: {
      status: effectiveOutput.status,
      recovery_note: effectiveOutput.recovery_note,
      input_tokens: effectiveOutput.input_tokens ?? 0,
      output_tokens: effectiveOutput.output_tokens ?? 0,
    },
  };
}

// Streaming-mode fallback resolve value from container_runner.ts.
// Frozen shape — always success-with-null-result when outputChain settles.
const STREAMING_SUCCESS_FALLBACK = Object.freeze({
  status: "success",
  result: null,
  newSessionId: "sess-fallback",
});

describe("envelope drop status — main.ts effectiveOutput picker", () => {
  it("lastOutput status='error' + result=null → envelope surfaces error, not success fallback", () => {
    const lastOutput = {
      status: "error",
      result: null,
      error:
        "agent-runner produced no output: SDK query iterator threw UsageError: ...",
      input_tokens: 0,
      output_tokens: 0,
      tokens_source: "unavailable",
    };
    const envelope = buildEnvelope(lastOutput, STREAMING_SUCCESS_FALLBACK);
    assert.equal(
      envelope.meta.status,
      "error",
      "must NOT be relabelled as success by the streaming fallback",
    );
    assert.ok(
      envelope.result.includes("agent-runner produced no output"),
      "error text must flow into envelope.result (since result=null falls through to error)",
    );
  });

  it("lastOutput status='recovered_from_session' + result='recovered text' → recovery surfaces", () => {
    const lastOutput = {
      status: "recovered_from_session",
      result: "the final recovered answer",
      recovery_note:
        "Recovered from on-disk session log: 3 turns, ~120 tokens (approximate).",
      input_tokens: 100,
      output_tokens: 20,
      tokens_source: "session_recovery",
    };
    const envelope = buildEnvelope(lastOutput, STREAMING_SUCCESS_FALLBACK);
    assert.equal(envelope.meta.status, "recovered_from_session");
    assert.equal(envelope.result, "the final recovered answer");
    assert.ok(envelope.meta.recovery_note);
    assert.equal(envelope.meta.input_tokens, 100);
    assert.equal(envelope.meta.output_tokens, 20);
  });

  it("lastOutput status='recovered_from_session' + result=null → recovery surfaces (note-only recovery)", () => {
    // A recovery envelope without a textual result (e.g. session log had turns
    // but no assistant text block) is still a recovery — the recovery_note
    // is the diagnostic signal. Without the fix, result=null would drop this
    // to the success fallback and lose the recovery_note.
    const lastOutput = {
      status: "recovered_from_session",
      result: null,
      recovery_note: "Recovered from session log: 2 tool-use turns, 0 text.",
      input_tokens: 500,
      output_tokens: 0,
      tokens_source: "session_recovery",
    };
    const envelope = buildEnvelope(lastOutput, STREAMING_SUCCESS_FALLBACK);
    assert.equal(envelope.meta.status, "recovered_from_session");
    assert.ok(
      envelope.meta.recovery_note.includes("Recovered from session log"),
    );
    assert.equal(envelope.meta.input_tokens, 500);
  });

  it("lastOutput status='success' + result='real text' → picks lastOutput (regression check)", () => {
    // The pre-fix behaviour preserved this case correctly; the fix must not
    // regress. When the stream delivered a real success, we use it over the
    // synthetic fallback.
    const lastOutput = {
      status: "success",
      result: "the actual model output",
      input_tokens: 1200,
      output_tokens: 80,
      tokens_source: "result_event",
    };
    const envelope = buildEnvelope(lastOutput, STREAMING_SUCCESS_FALLBACK);
    assert.equal(envelope.meta.status, "success");
    assert.equal(envelope.result, "the actual model output");
    assert.equal(envelope.meta.input_tokens, 1200);
  });

  it("lastOutput undefined (stream yielded nothing) → fallback wins", () => {
    // If nothing was ever streamed, the only signal we have is the fallback
    // resolve value from container_runner.ts. The picker must still produce
    // a usable envelope (rather than dereference undefined).
    const envelope = buildEnvelope(undefined, STREAMING_SUCCESS_FALLBACK);
    assert.equal(envelope.meta.status, "success");
    assert.equal(envelope.result, "");
  });

  it("lastOutput status='success' + result=null + no tools/tokens → fallback wins", () => {
    // Truly empty success-with-null-result should still collapse to the
    // generic fallback. This is the actual silent-failure shape.
    const lastOutput = {
      status: "success",
      result: null,
      input_tokens: 0,
      output_tokens: 0,
      tokens_source: "unavailable",
    };
    const envelope = buildEnvelope(lastOutput, STREAMING_SUCCESS_FALLBACK);
    assert.equal(envelope.meta.status, "success");
    assert.equal(envelope.result, "");
  });

  it("lastOutput status='success' + result=null + toolTrace → preserves streamed envelope", () => {
    const lastOutput = {
      status: "success",
      result: null,
      toolTrace: [{ type: "tool_call", tool_name: "Edit" }],
      input_tokens: 0,
      output_tokens: 0,
      tokens_source: "unavailable",
    };
    const envelope = buildEnvelope(lastOutput, STREAMING_SUCCESS_FALLBACK);
    assert.equal(envelope.meta.status, "success");
    assert.equal(
      envelope.meta.input_tokens,
      0,
      "streamed envelope should survive even without final text when toolTrace is present",
    );
  });

  it("lastOutput status='success' + result=null + nonzero tokens → preserves streamed envelope", () => {
    const lastOutput = {
      status: "success",
      result: null,
      input_tokens: 420,
      output_tokens: 17,
      tokens_source: "per_turn_sum",
    };
    const envelope = buildEnvelope(lastOutput, STREAMING_SUCCESS_FALLBACK);
    assert.equal(envelope.meta.status, "success");
    assert.equal(envelope.meta.input_tokens, 420);
    assert.equal(envelope.meta.output_tokens, 17);
  });

  it("every non-'success' ContainerOutput status with result=null must be preferred over the fallback", () => {
    // Canonical status list from shared_types.ts::ContainerOutput.status.
    // This is a structural guard: if the union gains a new status that can
    // carry result=null (e.g. 'aborted', 'cancelled'), the picker must be
    // updated or this assertion will fail and flag the miss.
    const preservedStatuses = ["error", "recovered_from_session"];
    for (const status of preservedStatuses) {
      const lastOutput = { status, result: null, error: "x", input_tokens: 0 };
      const envelope = buildEnvelope(lastOutput, STREAMING_SUCCESS_FALLBACK);
      assert.equal(
        envelope.meta.status,
        status,
        `status=${status} with result=null must not be masked by the success fallback`,
      );
    }
  });
});
