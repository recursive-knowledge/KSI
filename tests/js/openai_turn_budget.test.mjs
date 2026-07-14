/**
 * Regression test for OpenAI agent-runner turn-budget resolution and the
 * MaxTurnsExceededError salvage path.
 *
 * ## Turn-budget resolution
 *
 * The Claude path resolves its turn budget via `resolveClaudeMaxTurns`
 * (runtime_runner/agent-runner/src/query_config.ts): a unified 150-turn cap
 * for every task source, overridable via KCSI_CLAUDE_MAX_TURNS. A separate
 * message ceiling comes from KCSI_CLAUDE_MAX_MESSAGES (`resolveTurnBudgets`,
 * same file) with a default of 150 for every task source (ARC and
 * per_task_forum both used to get a lower default here before, respectively,
 * #1037 and #1049 raised them to match maxTurns — see
 * claude_turn_budget.test.mjs).
 *
 * Before the original fix, the OpenAI path used a hardcoded 25-turn cap with
 * no task-source awareness. That was raised to per-source defaults
 * (per_task_forum=60, arc=80, other=150). The subsequent MaxTurnsExceeded
 * handling PR (fix/openai-max-turns-handling) raised the ARC default from
 * 80 → 150 after observing ~3-5% of GPT-4o-mini ARC1 attempts throwing
 * uncaught MaxTurnsExceededError with p99 latency >1300s.
 *
 * `resolveOpenAIMaxTurns` is the shared resolver. Must stay in sync with
 * the inline copy in runtime_runner/agent-runner/src/openai.ts.
 *
 * ## MaxTurnsExceededError salvage
 *
 * Previously, when the @openai/agents SDK threw `MaxTurnsExceededError`
 * from `run()`, the exception bubbled up out of `runOpenAIQuery`'s
 * try/finally and killed the attempt with zero tokens reported and an
 * empty toolTrace. The SDK attaches the full `RunState` to the error
 * (via `error.state`), which contains `_generatedItems`,
 * `_modelResponses`, and `_previousResponseId` — the same backing fields
 * that `RunResult.newItems` / `rawResponses` / `lastResponseId` expose as
 * getters (see node_modules/@openai/agents-core/dist/result.mjs lines
 * 61-78). `salvageResultFromState` reconstructs a `result`-shaped object
 * from that state so the existing `usageFromResult` /
 * `extractToolTrace` / `extractBestOutput` helpers work unchanged, and
 * the adapter emits a `status='error'` envelope with salvaged partial
 * tokens + tool_trace instead of losing everything.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

// ── Local copy of resolveOpenAIMaxTurns from openai.ts ─────────────────
// Keep in lockstep with runtime_runner/agent-runner/src/openai.ts.
function resolveOpenAIMaxTurns(taskSource, envOverride) {
  const parsed = Number(envOverride);
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.floor(parsed);
  }
  const src = (taskSource || "").toLowerCase();
  if (src === "per_task_forum") return 60;
  if (src === "arc") return 150;
  return 150;
}

// ── Local copy of salvageResultFromState from openai.ts ─────────────────
// Keep in lockstep with runtime_runner/agent-runner/src/openai.ts. The
// shape returned here must be consumable by `usageFromResult` /
// `extractToolTrace` / `extractBestOutput` — i.e. it must have
// `rawResponses`, `newItems`, `lastResponseId`, `finalOutput`.
function salvageResultFromState(state) {
  const generatedItems = Array.isArray(state?._generatedItems)
    ? state._generatedItems
    : [];
  const modelResponses = Array.isArray(state?._modelResponses)
    ? state._modelResponses
    : [];
  const lastResponse =
    modelResponses.length > 0 ? modelResponses[modelResponses.length - 1] : undefined;
  const lastResponseId =
    (lastResponse && (lastResponse.responseId || lastResponse.response_id)) ||
    state?._previousResponseId ||
    undefined;
  return {
    newItems: generatedItems,
    rawResponses: modelResponses,
    lastResponseId,
    finalOutput: undefined,
  };
}

// ── Synthetic MaxTurnsExceededError mirror ─────────────────────────────
// We can't import the SDK here (tests must run without node_modules in
// CI-offline mode), so we mint a structurally-identical throwable: the
// adapter detects the error by both `instanceof MaxTurnsExceededError`
// AND `err.name === 'MaxTurnsExceededError'`, so name-based detection is
// a supported code path.
class MaxTurnsExceededErrorMock extends Error {
  constructor(message, state) {
    super(message);
    this.name = "MaxTurnsExceededError";
    this.state = state;
  }
}

// ── Simulate the adapter's catch-and-salvage handler ──────────────────
// This mirrors the try/catch around `run()` in openai.ts. If the thrown
// error is MaxTurnsExceededError (by name or instance), we salvage state
// and return an envelope shape; otherwise we re-throw.
function simulateRunOpenAIQueryCatch(thrown, { maxTurns, taskSource, previousResponseId }) {
  const isMaxTurns =
    (thrown instanceof Error && thrown.name === "MaxTurnsExceededError");
  if (!isMaxTurns) {
    throw thrown;
  }
  const salvage = salvageResultFromState(thrown.state);
  const toolTrace = Array.isArray(salvage.newItems) ? salvage.newItems : [];
  const sessionId = salvage.lastResponseId || previousResponseId || undefined;
  const errorMessage =
    `MaxTurnsExceededError (maxTurns=${maxTurns}, ` +
    `taskSource=${taskSource || "unknown"}): ${thrown.message} ` +
    `[salvaged tools=${toolTrace.length} ` +
    `rawResponses=${Array.isArray(salvage.rawResponses) ? salvage.rawResponses.length : 0}]`;
  return {
    status: "error",
    error: errorMessage,
    newSessionId: sessionId,
    toolTraceLength: toolTrace.length,
    rawResponsesLength: Array.isArray(salvage.rawResponses) ? salvage.rawResponses.length : 0,
  };
}

describe("resolveOpenAIMaxTurns — task-source aware turn budget", () => {
  it("defaults to 150 for unknown / unspecified task sources", () => {
    assert.equal(resolveOpenAIMaxTurns("", undefined), 150);
    assert.equal(resolveOpenAIMaxTurns(undefined, undefined), 150);
    assert.equal(resolveOpenAIMaxTurns("swebench_pro", undefined), 150);
    assert.equal(resolveOpenAIMaxTurns("polyglot", undefined), 150);
  });

  it("uses 150 turns for ARC tasks (raised from 80 to match non-forum default)", () => {
    // Pre-fix value was 80; raised to 150 in fix/openai-max-turns-handling
    // after empirical evidence that ~3-5% of GPT-4o-mini ARC attempts were
    // hitting 80 and throwing uncaught MaxTurnsExceededError.
    assert.equal(resolveOpenAIMaxTurns("arc", undefined), 150);
    assert.equal(resolveOpenAIMaxTurns("ARC", undefined), 150);
  });

  it("uses 60 turns for per_task_forum (matches Claude path default)", () => {
    assert.equal(resolveOpenAIMaxTurns("per_task_forum", undefined), 60);
  });

  it("honors KCSI_OPENAI_MAX_TURNS env override when set to a positive integer", () => {
    assert.equal(resolveOpenAIMaxTurns("arc", "200"), 200);
    assert.equal(resolveOpenAIMaxTurns("per_task_forum", "1"), 1);
  });

  it("ignores empty / zero / negative / NaN overrides and falls back to per-source default", () => {
    assert.equal(resolveOpenAIMaxTurns("arc", ""), 150);
    assert.equal(resolveOpenAIMaxTurns("arc", "0"), 150);
    assert.equal(resolveOpenAIMaxTurns("arc", "-5"), 150);
    assert.equal(resolveOpenAIMaxTurns("arc", "abc"), 150);
    assert.equal(resolveOpenAIMaxTurns("swebench_pro", undefined), 150);
  });

  it("never returns the pre-fix 25-turn cap unless explicitly requested", () => {
    // Defensive: confirm no code path silently yields 25 turns.
    assert.notEqual(resolveOpenAIMaxTurns("arc", undefined), 25);
    assert.notEqual(resolveOpenAIMaxTurns("swebench_pro", undefined), 25);
    assert.notEqual(resolveOpenAIMaxTurns("per_task_forum", undefined), 25);
    assert.notEqual(resolveOpenAIMaxTurns("", undefined), 25);
    // Explicit override is still honored — operator opt-in.
    assert.equal(resolveOpenAIMaxTurns("arc", "25"), 25);
  });
});

describe("salvageResultFromState — partial state recovery", () => {
  it("returns empty arrays and undefined session id for missing/null state", () => {
    assert.deepEqual(salvageResultFromState(undefined), {
      newItems: [],
      rawResponses: [],
      lastResponseId: undefined,
      finalOutput: undefined,
    });
    assert.deepEqual(salvageResultFromState(null), {
      newItems: [],
      rawResponses: [],
      lastResponseId: undefined,
      finalOutput: undefined,
    });
    assert.deepEqual(salvageResultFromState({}), {
      newItems: [],
      rawResponses: [],
      lastResponseId: undefined,
      finalOutput: undefined,
    });
  });

  it("exposes _generatedItems as newItems (the extractToolTrace input)", () => {
    const state = {
      _generatedItems: [{ type: "tool_call_item", rawItem: { name: "shell", call_id: "c1" } }],
      _modelResponses: [],
    };
    const salvage = salvageResultFromState(state);
    assert.equal(Array.isArray(salvage.newItems), true);
    assert.equal(salvage.newItems.length, 1);
    assert.equal(salvage.newItems[0].type, "tool_call_item");
  });

  it("exposes _modelResponses as rawResponses (the usageFromResult input)", () => {
    const state = {
      _generatedItems: [],
      _modelResponses: [
        { usage: { inputTokens: 100, outputTokens: 50 }, responseId: "resp_a" },
        { usage: { inputTokens: 200, outputTokens: 75 }, responseId: "resp_b" },
      ],
    };
    const salvage = salvageResultFromState(state);
    assert.equal(Array.isArray(salvage.rawResponses), true);
    assert.equal(salvage.rawResponses.length, 2);
    // lastResponseId falls back to the tail response's responseId.
    assert.equal(salvage.lastResponseId, "resp_b");
  });

  it("falls back to _previousResponseId when modelResponses lacks a responseId", () => {
    const state = {
      _generatedItems: [],
      _modelResponses: [{ usage: { inputTokens: 10, outputTokens: 5 } }],
      _previousResponseId: "resp_prev",
    };
    const salvage = salvageResultFromState(state);
    assert.equal(salvage.lastResponseId, "resp_prev");
  });

  it("accepts snake_case response_id alias", () => {
    const state = {
      _modelResponses: [{ response_id: "resp_snake" }],
    };
    const salvage = salvageResultFromState(state);
    assert.equal(salvage.lastResponseId, "resp_snake");
  });
});

describe("MaxTurnsExceededError handling — salvage into error envelope", () => {
  it("catches MaxTurnsExceededError-by-name and returns status='error' envelope", () => {
    const state = {
      _generatedItems: [
        { type: "tool_call_item", rawItem: { name: "shell", call_id: "c1" } },
        { type: "tool_call_item", rawItem: { name: "apply_patch", call_id: "c2" } },
      ],
      _modelResponses: [{ usage: { inputTokens: 500, outputTokens: 200 }, responseId: "resp_1" }],
    };
    const err = new MaxTurnsExceededErrorMock("Max turns (150) exceeded", state);

    const envelope = simulateRunOpenAIQueryCatch(err, {
      maxTurns: 150,
      taskSource: "arc",
      previousResponseId: "resp_prev",
    });

    assert.equal(envelope.status, "error");
    // Error message must mention MaxTurnsExceeded, the maxTurns value, and the task source
    assert.match(envelope.error, /MaxTurnsExceededError/);
    assert.match(envelope.error, /maxTurns=150/);
    assert.match(envelope.error, /taskSource=arc/);
    // Salvaged tool calls are preserved
    assert.equal(envelope.toolTraceLength, 2);
    assert.equal(envelope.rawResponsesLength, 1);
    // Session id carried forward from the salvaged state
    assert.equal(envelope.newSessionId, "resp_1");
  });

  it("re-throws non-MaxTurnsExceeded errors (does NOT swallow)", () => {
    const otherErr = new Error("some other failure");
    otherErr.name = "ToolCallError";
    assert.throws(
      () =>
        simulateRunOpenAIQueryCatch(otherErr, {
          maxTurns: 150,
          taskSource: "arc",
          previousResponseId: undefined,
        }),
      /some other failure/,
    );
  });

  it("handles MaxTurnsExceededError with empty state (zero salvage)", () => {
    // Agent throws before any tool call completes — we still must not
    // throw; we must return a valid error envelope with zero tokens
    // and zero tools.
    const err = new MaxTurnsExceededErrorMock("Max turns exceeded", {});
    const envelope = simulateRunOpenAIQueryCatch(err, {
      maxTurns: 60,
      taskSource: "per_task_forum",
      previousResponseId: "resp_prev",
    });
    assert.equal(envelope.status, "error");
    assert.match(envelope.error, /maxTurns=60/);
    assert.match(envelope.error, /taskSource=per_task_forum/);
    assert.equal(envelope.toolTraceLength, 0);
    assert.equal(envelope.rawResponsesLength, 0);
    // Session id falls back to previousResponseId when state has none
    assert.equal(envelope.newSessionId, "resp_prev");
  });

  it("includes '(unknown)' taskSource label when task source is empty", () => {
    const err = new MaxTurnsExceededErrorMock("Max turns exceeded", {});
    const envelope = simulateRunOpenAIQueryCatch(err, {
      maxTurns: 150,
      taskSource: "",
      previousResponseId: undefined,
    });
    assert.equal(envelope.status, "error");
    assert.match(envelope.error, /taskSource=unknown/);
  });
});
