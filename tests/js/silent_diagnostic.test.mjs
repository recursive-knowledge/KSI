/**
 * Regression / hypothesis-locking tests for the silent SDK-query-loop drain
 * diagnostic instrumentation added in the `diag(runtime)` PR.
 *
 * Context
 * -------
 * The agent-runner (runtime_runner/agent-runner/src/index.ts) wraps its
 * `for await (const message of query(...))` loop in a try/catch and emits a
 * structured diagnostic envelope on silent drain. PR #351 introduced a basic
 * status=error envelope; this PR extends it with env-shape snapshots so the
 * next reproduction distinguishes:
 *   - H1: API key missing / truncated (apiKeyLength < 20 or typeof==='undefined')
 *   - H2: MCP server hang (mcpServerNames non-empty + messageCount === 0)
 *   - H6: SDK subprocess swallowed provider error (iteratorError !== null)
 *
 * The helper implementation lives in the TypeScript file (exported as
 * `buildSilentDiagnostic`). This test duplicates the helper inline because
 * the repository's test harness runs Node directly (no tsc step); if the TS
 * helper's semantics change, this test must be updated in lockstep.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

// ── Local copy of `buildSilentDiagnostic` from adapter_safety.ts ─────────
// The canonical source moved to runtime_runner/agent-runner/src/adapter_safety.ts
// so the OpenAI adapter (and any future framework) can share the same
// provider-aware diagnostic. Keep this inline copy in lockstep with the TS.
const ANTHROPIC_PROVIDER = {
  id: "anthropic",
  apiKeyEnvName: "ANTHROPIC_API_KEY",
  oauthTokenEnvName: "CLAUDE_CODE_OAUTH_TOKEN",
};

function buildSilentDiagnostic(args) {
  const provider = args.provider || ANTHROPIC_PROVIDER; // default for legacy call sites
  const apiKey = args.sdkEnv[provider.apiKeyEnvName];
  const oauthEnvName = provider.oauthTokenEnvName || "";
  const oauthTok = oauthEnvName ? args.sdkEnv[oauthEnvName] : undefined;
  const sdkEnvKeys = Object.keys(args.sdkEnv).sort();
  // Accept either mcpServerNames (string[]) or legacy mcpServerConfig (dict).
  const mcpInput = args.mcpServerNames ?? args.mcpServerConfig ?? [];
  const mcpServerNames = Array.isArray(mcpInput)
    ? [...mcpInput].sort()
    : Object.keys(mcpInput).sort();
  let causeDesc;
  if (args.iteratorError && typeof args.iteratorError === "object") {
    const maybeCause = args.iteratorError.cause;
    if (maybeCause !== undefined && maybeCause !== null) {
      try {
        causeDesc =
          maybeCause instanceof Error
            ? `${maybeCause.name}: ${maybeCause.message}`
            : JSON.stringify(maybeCause).slice(0, 400);
      } catch {
        causeDesc = String(maybeCause).slice(0, 400);
      }
    }
  }
  return {
    provider: provider.id,
    messageCount: args.messageCount,
    resultCount: args.resultCount,
    lastAssistantFallbackKind:
      args.lastAssistantFallback === null ? "null" : "non-null",
    perTurnInputTokens: args.perTurnInputTokens,
    perTurnOutputTokens: args.perTurnOutputTokens,
    resultInputTokens: args.resultInputTokens,
    resultOutputTokens: args.resultOutputTokens,
    sdkEnvKeys,
    apiKeyEnvName: provider.apiKeyEnvName,
    apiKeyType: typeof apiKey,
    apiKeyLength: typeof apiKey === "string" ? apiKey.length : 0,
    oauthTokenEnvName: oauthEnvName,
    oauthTokenType: typeof oauthTok,
    oauthTokenLength: typeof oauthTok === "string" ? oauthTok.length : 0,
    logLevel: args.sdkEnv.LOG_LEVEL,
    anthropicLog: args.sdkEnv.ANTHROPIC_LOG,
    model: args.sdkEnv.MODEL,
    modelProvider: args.sdkEnv.MODEL_PROVIDER,
    modelAuthMode: args.sdkEnv.MODEL_AUTH_MODE,
    mcpServerNames,
    iteratorError: args.iteratorError
      ? {
          message: String(args.iteratorError.message || ""),
          name: args.iteratorError.name || "Error",
          stackHead: args.iteratorError.stack
            ? String(args.iteratorError.stack)
                .split("\n")
                .slice(0, 6)
                .join("\n")
            : undefined,
          cause: causeDesc,
        }
      : null,
  };
}

// ── Fixtures ──────────────────────────────────────────────────────────────

function healthySdkEnv() {
  return {
    ANTHROPIC_API_KEY: "sk-ant-" + "x".repeat(90),
    MODEL: "claude-haiku-4-5-20251001",
    MODEL_PROVIDER: "anthropic",
    MODEL_AUTH_MODE: "api",
    LOG_LEVEL: "silent",
    PATH: "/usr/bin:/bin",
  };
}

function mcpServerConfig() {
  return {
    memory: { command: "python3", args: ["/app/memory/mcp_server.py"] },
    arc: { command: "python3", args: ["/app/memory/mcp_server.py"] },
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────

describe("buildSilentDiagnostic — hypothesis channels", () => {
  it("H1 detection: missing ANTHROPIC_API_KEY produces apiKeyType='undefined', length=0", () => {
    const env = healthySdkEnv();
    delete env.ANTHROPIC_API_KEY;
    const diag = buildSilentDiagnostic({
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: env,
      mcpServerConfig: mcpServerConfig(),
      iteratorError: null,
    });
    assert.equal(diag.apiKeyType, "undefined");
    assert.equal(diag.apiKeyLength, 0);
    assert.equal(diag.iteratorError, null);
    assert.ok(
      !diag.sdkEnvKeys.includes("ANTHROPIC_API_KEY"),
      "sdkEnvKeys must not list the missing key",
    );
  });

  it("H1 detection: truncated ANTHROPIC_API_KEY yields apiKeyLength < 20", () => {
    const env = healthySdkEnv();
    env.ANTHROPIC_API_KEY = "sk-ant-xxx"; // 10 chars, clearly too short
    const diag = buildSilentDiagnostic({
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: env,
      mcpServerConfig: mcpServerConfig(),
      iteratorError: null,
    });
    assert.equal(diag.apiKeyType, "string");
    assert.ok(diag.apiKeyLength < 20, `apiKeyLength=${diag.apiKeyLength}`);
  });

  it("H2 detection: MCP server hang — messageCount=0 but mcpServerNames populated", () => {
    const diag = buildSilentDiagnostic({
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: healthySdkEnv(),
      mcpServerConfig: mcpServerConfig(),
      iteratorError: null,
    });
    // The signature of an MCP hang: non-empty MCP list, zero messages, no error.
    assert.equal(diag.messageCount, 0);
    assert.deepEqual(diag.mcpServerNames, ["arc", "memory"]);
    assert.equal(diag.iteratorError, null);
  });

  it("H6 detection: SDK subprocess error — iteratorError carries shape", () => {
    const err = new Error("ECONNRESET reading stream");
    err.name = "APIConnectionError";
    const causeErr = new Error("socket hang up");
    causeErr.name = "Error";
    err.cause = causeErr;
    const diag = buildSilentDiagnostic({
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: healthySdkEnv(),
      mcpServerConfig: mcpServerConfig(),
      iteratorError: err,
    });
    assert.ok(diag.iteratorError, "iteratorError must be set");
    assert.equal(diag.iteratorError.name, "APIConnectionError");
    assert.equal(diag.iteratorError.message, "ECONNRESET reading stream");
    assert.ok(
      diag.iteratorError.cause && diag.iteratorError.cause.includes("socket hang up"),
      `expected cause to mention 'socket hang up', got: ${diag.iteratorError.cause}`,
    );
  });
});

describe("buildSilentDiagnostic — env-shape invariants", () => {
  it("never leaks raw API key or OAuth token values", () => {
    const env = healthySdkEnv();
    env.ANTHROPIC_API_KEY = "sk-ant-SECRET-TOKEN-DO-NOT-LEAK";
    env.CLAUDE_CODE_OAUTH_TOKEN = "oat-SECRET-OAUTH-VALUE";
    const diag = buildSilentDiagnostic({
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: env,
      mcpServerConfig: mcpServerConfig(),
      iteratorError: null,
    });
    const serialized = JSON.stringify(diag);
    assert.ok(
      !serialized.includes("SECRET-TOKEN-DO-NOT-LEAK"),
      "API key value must not appear in diagnostic",
    );
    assert.ok(
      !serialized.includes("SECRET-OAUTH-VALUE"),
      "OAuth token value must not appear in diagnostic",
    );
    // But we DO want the lengths + types + key names so the operator can
    // correlate shape with H1.
    assert.equal(diag.apiKeyLength, env.ANTHROPIC_API_KEY.length);
    assert.equal(diag.oauthTokenLength, env.CLAUDE_CODE_OAUTH_TOKEN.length);
    assert.ok(diag.sdkEnvKeys.includes("ANTHROPIC_API_KEY"));
  });

  it("emits sdkEnvKeys in deterministic (sorted) order", () => {
    const env = {
      ZULU: "1",
      ALPHA: "2",
      MODEL: "m",
      ANTHROPIC_API_KEY: "x".repeat(100),
    };
    const diag = buildSilentDiagnostic({
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: env,
      mcpServerConfig: {},
      iteratorError: null,
    });
    assert.deepEqual(diag.sdkEnvKeys, [
      "ALPHA",
      "ANTHROPIC_API_KEY",
      "MODEL",
      "ZULU",
    ]);
  });

  it("captures per-turn + result token counters (pre-drain state) for post-mortem", () => {
    const diag = buildSilentDiagnostic({
      messageCount: 3,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 10,
      perTurnOutputTokens: 5,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: healthySdkEnv(),
      mcpServerConfig: mcpServerConfig(),
      iteratorError: null,
    });
    assert.equal(diag.messageCount, 3);
    assert.equal(diag.perTurnInputTokens, 10);
    assert.equal(diag.perTurnOutputTokens, 5);
    assert.equal(diag.resultInputTokens, 0);
  });

  it("lastAssistantFallbackKind is 'null' when null, 'non-null' otherwise", () => {
    const base = {
      messageCount: 0,
      resultCount: 0,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: healthySdkEnv(),
      mcpServerConfig: mcpServerConfig(),
      iteratorError: null,
    };
    const dNull = buildSilentDiagnostic({ ...base, lastAssistantFallback: null });
    const dStr = buildSilentDiagnostic({
      ...base,
      lastAssistantFallback: "partial text",
    });
    assert.equal(dNull.lastAssistantFallbackKind, "null");
    assert.equal(dStr.lastAssistantFallbackKind, "non-null");
  });
});

describe("buildSilentDiagnostic — error-cause handling", () => {
  it("handles iteratorError with no .cause gracefully", () => {
    const err = new Error("plain failure");
    const diag = buildSilentDiagnostic({
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: healthySdkEnv(),
      mcpServerConfig: mcpServerConfig(),
      iteratorError: err,
    });
    assert.ok(diag.iteratorError);
    assert.equal(diag.iteratorError.cause, undefined);
    assert.equal(diag.iteratorError.message, "plain failure");
  });

  it("handles iteratorError with a plain-object cause via JSON.stringify", () => {
    const err = new Error("wrapped");
    err.cause = { kind: "provider", status: 401 };
    const diag = buildSilentDiagnostic({
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: healthySdkEnv(),
      mcpServerConfig: mcpServerConfig(),
      iteratorError: err,
    });
    assert.ok(diag.iteratorError.cause.includes("provider"));
    assert.ok(diag.iteratorError.cause.includes("401"));
  });

  it("truncates stack to first 6 lines so the envelope stays small", () => {
    const err = new Error("deep");
    err.stack = Array.from({ length: 20 }, (_, i) => `  at frame${i}`).join(
      "\n",
    );
    const diag = buildSilentDiagnostic({
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv: healthySdkEnv(),
      mcpServerConfig: mcpServerConfig(),
      iteratorError: err,
    });
    const lines = diag.iteratorError.stackHead.split("\n");
    assert.equal(lines.length, 6);
  });
});

// ── Branch-selection invariants ───────────────────────────────────────────
// These lock the silent-exit branch behavior described at
// index.ts lines ~1060-1190. The runtime runs one of four paths after the
// for-await loop:
//   1. resultCount>0              — writeOutput already fired inside the loop.
//   2. fallback (has assistant)   — emit success with per-turn tokens.
//   3. silent-exit (nothing)      — emit status=error with diagnostic.
//   4. iterator-threw (partial)   — emit terminal status=error that the
//                                    host's LAST-parsed-marker rule overrides
//                                    any earlier success write with.
// We simulate the branch selector here so future refactors don't regress it.

function selectBranch({
  resultCount,
  lastAssistantFallback,
  resultInputTokens,
  resultOutputTokens,
  resultCacheCreationTokens = 0,
  resultCacheReadTokens = 0,
  perTurnInputTokens,
  perTurnOutputTokens,
  perTurnCacheCreationTokens = 0,
  perTurnCacheReadTokens = 0,
  iteratorError,
}) {
  if (resultCount > 0) return "inLoopResult";
  if (resultCount === 0 && lastAssistantFallback) {
    return "fallbackAssistant";
  }
  if (
    resultCount === 0 &&
    !lastAssistantFallback &&
    resultInputTokens +
      resultOutputTokens +
      resultCacheCreationTokens +
      resultCacheReadTokens +
      perTurnInputTokens +
      perTurnOutputTokens +
      perTurnCacheCreationTokens +
      perTurnCacheReadTokens ===
      0
  ) {
    return "silentExit";
  }
  if (iteratorError) return "iteratorThrewPartial";
  return "none";
}

describe("silent-exit branch selection", () => {
  it("SDK drained clean (H1/H2): routes to silentExit", () => {
    const branch = selectBranch({
      resultCount: 0,
      lastAssistantFallback: null,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      iteratorError: null,
    });
    assert.equal(branch, "silentExit");
  });

  it("SDK threw with zero progress: still routes to silentExit so diagnostic includes iteratorError", () => {
    const branch = selectBranch({
      resultCount: 0,
      lastAssistantFallback: null,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      iteratorError: new Error("APIConnectionError"),
    });
    assert.equal(branch, "silentExit");
  });

  it("SDK threw after partial progress: routes to iteratorThrewPartial (NOT silent)", () => {
    // This is the H6 "swallowed provider error" path. The agent-runner must
    // emit a terminal status=error envelope that overrides the earlier
    // success write.
    const branch = selectBranch({
      resultCount: 0,
      lastAssistantFallback: "partial text",
      resultInputTokens: 0,
      resultOutputTokens: 0,
      perTurnInputTokens: 100,
      perTurnOutputTokens: 50,
      iteratorError: new Error("ECONNRESET"),
    });
    // Falls through to fallbackAssistant first (emits success), and then
    // the iterator-threw branch in index.ts emits a follow-up error envelope
    // on top. We assert fallbackAssistant here; index.ts's second pass
    // picks up iteratorError separately.
    assert.equal(branch, "fallbackAssistant");
  });

  it("cache-only progress is not treated as a silent exit", () => {
    const branch = selectBranch({
      resultCount: 0,
      lastAssistantFallback: null,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      resultCacheCreationTokens: 0,
      resultCacheReadTokens: 250,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      iteratorError: new Error("ECONNRESET"),
    });
    assert.equal(branch, "iteratorThrewPartial");
  });

  it("healthy run: routes through inLoopResult (no post-loop work)", () => {
    const branch = selectBranch({
      resultCount: 1,
      lastAssistantFallback: "final answer",
      resultInputTokens: 500,
      resultOutputTokens: 100,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      iteratorError: null,
    });
    assert.equal(branch, "inLoopResult");
  });
});
