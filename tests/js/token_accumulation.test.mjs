/**
 * Regression test for the 2026-04 zero-token reporting bug.
 *
 * Context
 * -------
 * The agent-runner (runtime_runner/agent-runner/src/index.ts) streams SDK
 * messages and accumulates token usage for each one. Before this fix, the
 * accumulator only inspected ``message.usage`` (top-level). The Claude Agent
 * SDK, however, nests per-turn usage for ``assistant`` messages under
 * ``message.message.usage`` — so the accumulator only picked up the
 * ``result`` event's top-level aggregate. When the stream ended before a
 * result event (e.g. hit ``scheduledMaxMessages=80`` on ARC sessions), the
 * scheduled-fallback path wrote zero tokens even though real work happened.
 *
 * Evidence: live Haiku baseline sweep had 4 ARC2 attempts with
 * ``input_tokens=output_tokens=cache_read=cache_creation=0`` despite
 * ``eval.status=ok`` and 100+ tool_trace entries carrying Read/arc_* calls.
 *
 * This test verifies the reporter sums per-turn usages correctly when the
 * stream carries assistant messages with nested usage but no final result
 * event.
 *
 * The helper implementation is the source of truth in the TypeScript file
 * (`runtime_runner/agent-runner/src/index.ts`, exported as
 * `extractUsageFromSdkMessage`). This test duplicates the helper inline
 * because the repository's test harness runs Node directly (no tsc step);
 * if the TS helper's semantics change, this test must be updated in
 * lockstep.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

// ── Local copy of `extractUsageFromSdkMessage` from index.ts ──────────────
// Faithful type-stripped mirror of runtime_runner/agent-runner/src/index.ts;
// drift-guarded by copy_sync_guard.test.mjs.
function extractUsageFromSdkMessage(message) {
  const zero = {
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  };
  if (!message || typeof message !== 'object') return zero;
  const msg = message;
  // Assistant messages nest under .message.usage; result messages expose .usage
  // at the top level. Check both and return whichever is populated.
  const nestedMessage = (msg.message && typeof msg.message === 'object')
    ? msg.message
    : null;
  const nestedUsage = nestedMessage && typeof nestedMessage.usage === 'object'
    ? nestedMessage.usage
    : null;
  const topUsage = msg.usage && typeof msg.usage === 'object'
    ? msg.usage
    : null;
  const usage = topUsage ?? nestedUsage;
  if (!usage) return zero;
  const num = (v) => {
    const n = Number(v ?? 0);
    return Number.isFinite(n) && n > 0 ? n : 0;
  };
  return {
    input_tokens: num(usage.input_tokens),
    output_tokens: num(usage.output_tokens),
    cache_creation_input_tokens: num(usage.cache_creation_input_tokens),
    cache_read_input_tokens: num(usage.cache_read_input_tokens),
  };
}

// ── Accumulator simulation that mirrors the runQuery loop split ────────────
function accumulate(messages) {
  let perTurn = {
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  };
  let result = {
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  };
  for (const msg of messages) {
    const d = extractUsageFromSdkMessage(msg);
    if (d.input_tokens + d.output_tokens + d.cache_creation_input_tokens + d.cache_read_input_tokens === 0) {
      continue;
    }
    const bucket = msg.type === "result" ? result : perTurn;
    bucket.input_tokens += d.input_tokens;
    bucket.output_tokens += d.output_tokens;
    bucket.cache_creation_input_tokens += d.cache_creation_input_tokens;
    bucket.cache_read_input_tokens += d.cache_read_input_tokens;
  }
  // emission policy from index.ts: prefer result aggregate, fall back to per-turn
  const hasResult =
    result.input_tokens + result.output_tokens + result.cache_creation_input_tokens + result.cache_read_input_tokens > 0;
  const hasPerTurn =
    perTurn.input_tokens + perTurn.output_tokens + perTurn.cache_creation_input_tokens + perTurn.cache_read_input_tokens > 0;
  const emitted = hasResult ? result : perTurn;
  let source;
  if (hasResult) source = "result_event";
  else if (hasPerTurn) source = "per_turn_sum";
  else source = "unavailable";
  return { emitted, source };
}

// ── Fixtures ──────────────────────────────────────────────────────────────

function assistantMsg({ input, output, cacheCreate = 0, cacheRead = 0 }) {
  // Mirrors the actual SDK shape — usage nested under .message.usage.
  // Sampled from a real failing attempt in the baseline-sweep DB.
  return {
    type: "assistant",
    message: {
      model: "claude-haiku-4-5-20251001",
      id: "msg_test",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "…" }],
      stop_reason: null,
      usage: {
        input_tokens: input,
        output_tokens: output,
        cache_creation_input_tokens: cacheCreate,
        cache_read_input_tokens: cacheRead,
      },
    },
    parent_tool_use_id: null,
    session_id: "sess_test",
    uuid: "u_test",
  };
}

function userMsg() {
  return {
    type: "user",
    message: { role: "user", content: [{ type: "tool_result", tool_use_id: "t" }] },
    parent_tool_use_id: null,
    session_id: "sess_test",
  };
}

function resultMsg({ input, output, cacheCreate = 0, cacheRead = 0 }) {
  return {
    type: "result",
    subtype: "success",
    result: "final answer",
    usage: {
      input_tokens: input,
      output_tokens: output,
      cache_creation_input_tokens: cacheCreate,
      cache_read_input_tokens: cacheRead,
    },
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────

describe("extractUsageFromSdkMessage", () => {
  it("pulls nested usage from an assistant message (the 2026-04 bug locus)", () => {
    const msg = assistantMsg({ input: 10, output: 5, cacheRead: 100 });
    const d = extractUsageFromSdkMessage(msg);
    assert.deepEqual(d, {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 100,
    });
  });

  it("pulls top-level usage from a result message", () => {
    const msg = resultMsg({ input: 50, output: 20, cacheCreate: 200, cacheRead: 1000 });
    const d = extractUsageFromSdkMessage(msg);
    assert.deepEqual(d, {
      input_tokens: 50,
      output_tokens: 20,
      cache_creation_input_tokens: 200,
      cache_read_input_tokens: 1000,
    });
  });

  it("returns zeros when no usage fields are present", () => {
    const d = extractUsageFromSdkMessage({ type: "system", subtype: "init" });
    assert.deepEqual(d, {
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    });
  });

  it("handles null/undefined/non-object inputs gracefully", () => {
    for (const bad of [null, undefined, "nope", 42]) {
      const d = extractUsageFromSdkMessage(bad);
      assert.equal(d.input_tokens + d.output_tokens, 0);
    }
  });

  it("prefers top-level usage when both nesting levels are present", () => {
    const msg = {
      type: "result",
      usage: { input_tokens: 999, output_tokens: 111 },
      message: { usage: { input_tokens: 1, output_tokens: 1 } },
    };
    const d = extractUsageFromSdkMessage(msg);
    assert.equal(d.input_tokens, 999);
    assert.equal(d.output_tokens, 111);
  });
});

describe("streaming accumulation — bug regression", () => {
  it("sums per-turn usages when stream ends without a result event", () => {
    // This is the exact failure mode from attempts 275/483: the SDK stream
    // terminated at the scheduledMaxMessages ceiling (80 messages for ARC)
    // without emitting a result event. Per-turn usage on assistant messages
    // is the only signal. Before the fix, accumulation read the wrong field
    // and produced zero; after the fix, it sums correctly.
    const turns = [
      { type: "system", subtype: "init" },
      assistantMsg({ input: 4, output: 100, cacheRead: 12000, cacheCreate: 200 }),
      userMsg(),
      assistantMsg({ input: 2, output: 50, cacheRead: 12400, cacheCreate: 30 }),
      userMsg(),
      assistantMsg({ input: 3, output: 80, cacheRead: 12500, cacheCreate: 40 }),
      // Stream truncated here by max-messages ceiling. NO result event.
    ];
    const { emitted, source } = accumulate(turns);
    assert.equal(source, "per_turn_sum", "must flag tokens as per-turn-sum");
    assert.equal(emitted.input_tokens, 9, "input_tokens must sum across assistant messages");
    assert.equal(emitted.output_tokens, 230);
    assert.equal(emitted.cache_read_input_tokens, 36900);
    assert.equal(emitted.cache_creation_input_tokens, 270);
  });

  it("prefers the result-event aggregate when it is present", () => {
    // Normal healthy session: result event carries the authoritative total.
    const turns = [
      assistantMsg({ input: 10, output: 20 }),
      userMsg(),
      assistantMsg({ input: 15, output: 30 }),
      resultMsg({ input: 500, output: 250, cacheRead: 10000 }),
    ];
    const { emitted, source } = accumulate(turns);
    assert.equal(source, "result_event");
    // Result aggregate is authoritative — the per-turn sum is ignored.
    assert.equal(emitted.input_tokens, 500);
    assert.equal(emitted.output_tokens, 250);
    assert.equal(emitted.cache_read_input_tokens, 10000);
  });

  it("reports 'unavailable' when neither source has data (distinguishable from truly-zero)", () => {
    const turns = [
      { type: "system", subtype: "init" },
      userMsg(),
      { type: "assistant", message: { content: [] } }, // no usage at all
    ];
    const { emitted, source } = accumulate(turns);
    assert.equal(source, "unavailable");
    assert.equal(emitted.input_tokens, 0);
    assert.equal(emitted.output_tokens, 0);
  });

  it("scheduled-fallback path: non-trivial trace with per-turn usage yields non-zero tokens", () => {
    // Simulates attempt 275 shape: 47 assistants + 32 user turns + 0 results,
    // each assistant carrying a small per-turn usage block. Prior to the fix
    // this attempt reported zero tokens despite generating 60KB of session
    // memory and 29 tool calls.
    const turns = [];
    turns.push({ type: "system", subtype: "init" });
    for (let i = 0; i < 47; i += 1) {
      turns.push(assistantMsg({ input: 4, output: 6, cacheRead: 12000 }));
      if (i % 2 === 0) turns.push(userMsg());
    }
    const { emitted, source } = accumulate(turns);
    assert.equal(source, "per_turn_sum");
    assert.equal(emitted.input_tokens, 47 * 4);
    assert.equal(emitted.output_tokens, 47 * 6);
    assert.equal(emitted.cache_read_input_tokens, 47 * 12000);
  });
});
