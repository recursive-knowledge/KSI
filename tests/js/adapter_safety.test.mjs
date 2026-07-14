/**
 * Adapter-safety shared-helper regression tests.
 *
 * `runtime_runner/agent-runner/src/adapter_safety.ts` centralises the
 * provider-aware silent-failure diagnostic previously inlined in
 * `index.ts`, plus a fingerprint helper that the OpenAI adapter and any
 * future adapter can call in lockstep. The goal of the extraction is that
 * each new agent framework gets the Haiku-era safety nets (PR #351, #356,
 * #358) for free by calling these helpers from its silent-exit branch.
 *
 * These tests pin the provider-aware behaviour:
 *   - Anthropic provider samples ANTHROPIC_API_KEY and CLAUDE_CODE_OAUTH_TOKEN
 *   - OpenAI provider samples OPENAI_API_KEY and has no OAuth alternative
 *   - Secret VALUES never appear in the diagnostic (shapes only)
 *   - `isSilentFailureEnvelope` matches the Python-side contract in
 *     `src/ksi/runtime/normalize.py::is_silent_agent_failure`
 *
 * This test duplicates the helpers inline because the repo's test harness
 * runs Node directly with no tsc step; if adapter_safety.ts changes, this
 * file must follow in lockstep (same convention as
 * tests/js/silent_diagnostic.test.mjs).
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

// ── Provider configs (duplicated from adapter_safety.ts) ────────────────
const ANTHROPIC_PROVIDER = {
  id: "anthropic",
  apiKeyEnvName: "ANTHROPIC_API_KEY",
  oauthTokenEnvName: "CLAUDE_CODE_OAUTH_TOKEN",
};
const OPENAI_PROVIDER = {
  id: "openai",
  apiKeyEnvName: "OPENAI_API_KEY",
};

// ── buildSilentDiagnostic (duplicated from adapter_safety.ts) ───────────
function buildSilentDiagnostic(args) {
  const apiKey = args.sdkEnv[args.provider.apiKeyEnvName];
  const oauthEnvName = args.provider.oauthTokenEnvName || "";
  const oauthTok = oauthEnvName ? args.sdkEnv[oauthEnvName] : undefined;
  const sdkEnvKeys = Object.keys(args.sdkEnv).sort();
  const mcpServerNames = Array.isArray(args.mcpServerNames)
    ? [...args.mcpServerNames].sort()
    : Object.keys(args.mcpServerNames || {}).sort();

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
    provider: args.provider.id,
    messageCount: args.messageCount,
    resultCount: args.resultCount,
    lastAssistantFallbackKind:
      args.lastAssistantFallback === null ? "null" : "non-null",
    perTurnInputTokens: args.perTurnInputTokens,
    perTurnOutputTokens: args.perTurnOutputTokens,
    resultInputTokens: args.resultInputTokens,
    resultOutputTokens: args.resultOutputTokens,
    sdkEnvKeys,
    apiKeyEnvName: args.provider.apiKeyEnvName,
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

// ── isSilentFailureEnvelope (duplicated from adapter_safety.ts) ─────────
function isSilentFailureEnvelope(envelope) {
  if (!envelope || typeof envelope !== "object") return false;
  const status = String(envelope.status || "").toLowerCase();
  const successLike = status === "" || status === "success" || status === "ok";
  if (!successLike) return false;
  const result = envelope.result;
  const hasOutput =
    typeof result === "string" && result.trim().length > 0;
  const hasTools =
    Array.isArray(envelope.toolTrace) && envelope.toolTrace.length > 0;
  const hasTokens =
    Number(envelope.input_tokens || 0) > 0 ||
    Number(envelope.output_tokens || 0) > 0 ||
    Number(envelope.cache_creation_input_tokens || 0) > 0 ||
    Number(envelope.cache_read_input_tokens || 0) > 0;
  return !hasOutput && !hasTools && !hasTokens;
}

// ── helpers ─────────────────────────────────────────────────────────────
const baseArgs = (provider, sdkEnv = {}) => ({
  provider,
  messageCount: 0,
  resultCount: 0,
  lastAssistantFallback: null,
  perTurnInputTokens: 0,
  perTurnOutputTokens: 0,
  resultInputTokens: 0,
  resultOutputTokens: 0,
  sdkEnv,
  mcpServerNames: [],
  iteratorError: null,
});

// ── tests ───────────────────────────────────────────────────────────────
describe("buildSilentDiagnostic — provider-aware env sampling", () => {
  it("samples ANTHROPIC_API_KEY + CLAUDE_CODE_OAUTH_TOKEN for anthropic provider", () => {
    const apiKeyValue = "sk-ant-abcdefghijklmnop"; // 23 chars
    const oauthValue = "oauthtoken123"; // 13 chars
    const diag = buildSilentDiagnostic(
      baseArgs(ANTHROPIC_PROVIDER, {
        ANTHROPIC_API_KEY: apiKeyValue,
        CLAUDE_CODE_OAUTH_TOKEN: oauthValue,
      }),
    );
    assert.equal(diag.provider, "anthropic");
    assert.equal(diag.apiKeyEnvName, "ANTHROPIC_API_KEY");
    assert.equal(diag.apiKeyType, "string");
    assert.equal(diag.apiKeyLength, apiKeyValue.length);
    assert.equal(diag.oauthTokenEnvName, "CLAUDE_CODE_OAUTH_TOKEN");
    assert.equal(diag.oauthTokenType, "string");
    assert.equal(diag.oauthTokenLength, oauthValue.length);
  });

  it("samples OPENAI_API_KEY for openai provider with empty oauth field", () => {
    const openaiKey = "sk-proj-aaaabbbbccccdddd"; // 24 chars
    const diag = buildSilentDiagnostic(
      baseArgs(OPENAI_PROVIDER, {
        OPENAI_API_KEY: openaiKey,
        ANTHROPIC_API_KEY: "sk-ant-SHOULD-NOT-LEAK",
      }),
    );
    assert.equal(diag.provider, "openai");
    assert.equal(diag.apiKeyEnvName, "OPENAI_API_KEY");
    assert.equal(diag.apiKeyLength, openaiKey.length);
    assert.equal(diag.oauthTokenEnvName, "", "OpenAI has no oauth alternative");
    assert.equal(diag.oauthTokenType, "undefined");
    assert.equal(diag.oauthTokenLength, 0);

    // Critically: even though ANTHROPIC_API_KEY is present in sdkEnv,
    // the openai provider diagnostic MUST NOT surface its length as the
    // primary api-key metric (that would false-implicate the wrong env var).
    // Only the env-key LIST contains it (which is fine — shape, not value).
    assert.deepEqual(diag.sdkEnvKeys.sort(), [
      "ANTHROPIC_API_KEY",
      "OPENAI_API_KEY",
    ]);
  });

  it("reports apiKeyType='undefined' when the key env is missing (H1)", () => {
    const diag = buildSilentDiagnostic(baseArgs(OPENAI_PROVIDER, {}));
    assert.equal(diag.apiKeyType, "undefined");
    assert.equal(diag.apiKeyLength, 0);
  });

  it("flags truncated API keys by length (H1 signal)", () => {
    const diag = buildSilentDiagnostic(
      baseArgs(ANTHROPIC_PROVIDER, { ANTHROPIC_API_KEY: "sk-short" }),
    );
    assert.ok(diag.apiKeyLength < 20, "truncated key should be visibly short");
    assert.equal(diag.apiKeyLength, 8);
  });

  it("surfaces mcpServerNames for H2 diagnosis", () => {
    const diag = buildSilentDiagnostic({
      ...baseArgs(ANTHROPIC_PROVIDER),
      mcpServerNames: ["ksi-memory", "ksi-arc"],
    });
    assert.deepEqual(diag.mcpServerNames, ["ksi-arc", "ksi-memory"]);
  });

  it("accepts mcpServerNames as a dict (legacy index.ts call shape)", () => {
    const diag = buildSilentDiagnostic({
      ...baseArgs(ANTHROPIC_PROVIDER),
      mcpServerNames: { ksi: {}, "ksi-memory": {} },
    });
    assert.deepEqual(diag.mcpServerNames, ["ksi", "ksi-memory"]);
  });

  it("captures iteratorError shape for H6 diagnosis", () => {
    const err = new Error("socket hang up");
    err.name = "FetchError";
    err.stack = "FetchError: socket hang up\n    at agent.js:1";
    Object.defineProperty(err, "cause", {
      value: new Error("ECONNRESET"),
      configurable: true,
      enumerable: false,
    });
    const diag = buildSilentDiagnostic({
      ...baseArgs(OPENAI_PROVIDER),
      iteratorError: err,
    });
    assert.equal(diag.iteratorError.name, "FetchError");
    assert.equal(diag.iteratorError.message, "socket hang up");
    assert.ok(diag.iteratorError.stackHead.includes("FetchError"));
    assert.equal(diag.iteratorError.cause, "Error: ECONNRESET");
  });

  it("NEVER includes raw API key or OAuth token values in the serialized diagnostic", () => {
    const diag = buildSilentDiagnostic(
      baseArgs(ANTHROPIC_PROVIDER, {
        ANTHROPIC_API_KEY: "sk-ant-SECRETsecretSECRET",
        CLAUDE_CODE_OAUTH_TOKEN: "oauth-secret-token-XYZ",
        OPENAI_API_KEY: "sk-proj-SECRET",
      }),
    );
    const s = JSON.stringify(diag);
    assert.ok(!s.includes("SECRETsecretSECRET"));
    assert.ok(!s.includes("oauth-secret-token-XYZ"));
    assert.ok(!s.includes("sk-proj-SECRET"));
    assert.ok(!s.includes("sk-ant-"));
  });
});

describe("isSilentFailureEnvelope — agent-runner fingerprint", () => {
  it("matches an empty success envelope with zero tokens and no tools", () => {
    assert.equal(
      isSilentFailureEnvelope({
        status: "success",
        result: "",
        toolTrace: [],
        input_tokens: 0,
        output_tokens: 0,
      }),
      true,
    );
  });

  it("treats '' and 'ok' statuses the same as 'success' (Python-side parity)", () => {
    for (const status of ["", "success", "ok", "Ok"]) {
      assert.equal(
        isSilentFailureEnvelope({
          status,
          result: "",
          toolTrace: [],
        }),
        true,
        `status=${status}`,
      );
    }
  });

  it("is FALSE when any token bucket is nonzero", () => {
    assert.equal(
      isSilentFailureEnvelope({
        status: "success",
        result: "",
        toolTrace: [],
        input_tokens: 5,
      }),
      false,
    );
    assert.equal(
      isSilentFailureEnvelope({
        status: "success",
        result: "",
        toolTrace: [],
        cache_read_input_tokens: 100,
      }),
      false,
    );
  });

  it("is FALSE when result is a non-empty string", () => {
    assert.equal(
      isSilentFailureEnvelope({
        status: "success",
        result: "done",
        toolTrace: [],
      }),
      false,
    );
  });

  it("is FALSE when toolTrace has any entries", () => {
    assert.equal(
      isSilentFailureEnvelope({
        status: "success",
        result: "",
        toolTrace: [{ tool_name: "Read" }],
      }),
      false,
    );
  });

  it("is FALSE when status already signals error (already classified)", () => {
    assert.equal(
      isSilentFailureEnvelope({ status: "error", result: "", toolTrace: [] }),
      false,
    );
    assert.equal(
      isSilentFailureEnvelope({
        status: "silent_failure",
        result: "",
        toolTrace: [],
      }),
      false,
    );
  });

  it("is FALSE on non-object input", () => {
    assert.equal(isSilentFailureEnvelope(null), false);
    assert.equal(isSilentFailureEnvelope(undefined), false);
    assert.equal(isSilentFailureEnvelope("string"), false);
  });
});
