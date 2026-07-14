/**
 * Copy-sync guards for the inline JS copies of TypeScript helpers (issue
 * #734).
 *
 * Several tests under tests/js/ keep inline JS copies of agent-runner TS
 * helpers because the test harness runs Node directly (no tsc step). Those
 * "must stay in sync" comments were not enforced and the copies drifted
 * (silent_failure_recovery.test.mjs missed the `iterator_drain_pending_tools`
 * trigger branch, the emitPhrase marker sourcing, and the provider-aware
 * buildSilentDiagnostic). This file makes the contract executable:
 *
 *   STRICT guards — the JS copy must be a byte-mirror of the TS source after
 *   conservative type-annotation stripping + comment/whitespace/quote
 *   normalization (see tests/js/_sync_check.mjs):
 *     - extractUsageFromSdkMessage   (index.ts ↔ 4 copies)
 *     - recoverFromSessionLog        (index.ts ↔ 3 copies)
 *     - resolveClaudeMaxTurns        (index.ts ↔ claude_turn_budget)
 *     - resolveTurnBudgets           (query_config.ts ↔ claude_turn_budget)
 *     - resolveOpenAIMaxTurns        (openai.ts ↔ openai_turn_budget)
 *     - salvageResultFromState       (openai.ts ↔ openai_turn_budget)
 *     - buildSilentDiagnostic        (adapter_safety.ts ↔ silent_failure_recovery)
 *     - maybeRecoverFromEmptyScheduledOutcome (index.ts ↔ silent_failure_recovery)
 *     - pickEffectiveOutput          (main.ts ↔ envelope_drop_status)
 *
 *   ANCHOR guards — for copies that deliberately diverge, assert that the
 *   distinctive strings exist on both sides instead:
 *     - iterator_drain_recovery's maybeRecoverFromEmptyScheduledOutcome is a
 *       DELIBERATE TRIM (recovery-success path only; the error path is
 *       covered by silent_failure_recovery's full copy) — pin the trigger
 *       wordings and envelope markers.
 *     - silent_diagnostic.test.mjs's buildSilentDiagnostic carries a
 *       legacy-compat shim (`args.mcpServerNames ?? args.mcpServerConfig`,
 *       `args.provider || ANTHROPIC_PROVIDER`) so its older call sites keep
 *       working — pin the full output field list instead.
 *
 *   DOCUMENTED EXCEPTIONS of the strict comparison (pinned by anchors):
 *     - parameter default values and parameter type annotations are not
 *       byte-compared (only parameter names are); the critical defaults
 *       (CONTAINER_CLAUDE_SESSIONS_ROOT, readFileImpl, recoverImpl) are
 *       anchor-asserted below.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  compareFunctions,
  extractFunction,
  normalizeBody,
} from "./_sync_check.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const readRepo = (...p) => fs.readFileSync(path.join(repoRoot, ...p), "utf-8");
const readTest = (name) => fs.readFileSync(path.join(__dirname, name), "utf-8");

// index.ts was split into focused modules (issue: god-file refactor). The
// helpers these guards pin now live in dedicated files; read each from its
// new home so the strict byte-mirror comparison follows the real source.
const usageTs = readRepo("runtime_runner", "agent-runner", "src", "usage.ts");
const sessionRecoveryTs = readRepo(
  "runtime_runner",
  "agent-runner",
  "src",
  "session_recovery.ts",
);
const queryConfigTs = readRepo(
  "runtime_runner",
  "agent-runner",
  "src",
  "query_config.ts",
);
const runnerConstantsTs = readRepo(
  "runtime_runner",
  "agent-runner",
  "src",
  "runner_constants.ts",
);
const openaiTs = readRepo("runtime_runner", "agent-runner", "src", "openai.ts");
const adapterSafetyTs = readRepo(
  "runtime_runner",
  "agent-runner",
  "src",
  "adapter_safety.ts",
);
const mainTs = readRepo("runtime_runner", "src", "main.ts");

const silentFailure = readTest("silent_failure_recovery.test.mjs");
const sessionRecovery = readTest("session_recovery.test.mjs");
const iteratorDrain = readTest("iterator_drain_recovery.test.mjs");
const tokenAccumulation = readTest("token_accumulation.test.mjs");
const claudeBudget = readTest("claude_turn_budget.test.mjs");
const openaiBudget = readTest("openai_turn_budget.test.mjs");
const silentDiagnostic = readTest("silent_diagnostic.test.mjs");
const envelopeDropStatus = readTest("envelope_drop_status.test.mjs");

function assertInSync(tsSource, jsSource, name, label) {
  const r = compareFunctions(tsSource, jsSource, name);
  assert.ok(r.ok, `${label}: ${r.ok ? "" : r.message}`);
}

// ── Self-test: the normalizer is not vacuous and detects real drift ───────

describe("sync-guard self-test", () => {
  it("normalized bodies are non-trivial (the comparison is not vacuous)", () => {
    const body = normalizeBody(
      extractFunction(sessionRecoveryTs, "recoverFromSessionLog").body,
    );
    assert.ok(
      body.length > 500,
      `normalized recoverFromSessionLog body suspiciously short (${body.length})`,
    );
    assert.ok(body.includes("candidates"), "real identifiers must survive");
    assert.ok(body.includes("'.jsonl'"), "string literals must survive");
  });

  it("detects an injected single-token drift", () => {
    const mutated = silentFailure.replace(
      "turnCount += 1;",
      "turnCount += 2;",
    );
    assert.notEqual(mutated, silentFailure, "mutation must apply");
    const r = compareFunctions(sessionRecoveryTs, mutated, "recoverFromSessionLog");
    assert.equal(r.ok, false, "guard must flag the mutated copy");
    assert.ok(r.message.includes("drifted"), r.message);
  });

  it("detects a dropped trigger branch (the original #734 drift)", () => {
    const mutated = silentFailure.replace(
      "ctx.trigger === 'iterator_drain_pending_tools'",
      "ctx.trigger === 'never_matches'",
    );
    const r = compareFunctions(
      sessionRecoveryTs,
      mutated,
      "maybeRecoverFromEmptyScheduledOutcome",
    );
    assert.equal(r.ok, false, "guard must flag the rewritten trigger branch");
  });

  it("ignores nested regular-function parameter annotations", () => {
    const tsSource = `
      function outer(value: unknown): Record<string, unknown> | undefined {
        function inner(obj: Record<string, unknown>, key: string): Record<string, unknown> | undefined {
          return typeof obj[key] === 'object' ? obj : undefined;
        }
        return inner(value as Record<string, unknown>, 'id');
      }
    `;
    const jsSource = `
      function outer(value) {
        function inner(obj, key) {
          return typeof obj[key] === "object" ? obj : undefined;
        }
        return inner(value, "id");
      }
    `;
    const r = compareFunctions(tsSource, jsSource, "outer");
    assert.equal(r.ok, true, r.ok ? "" : r.message);
  });
});

// ── Strict guards ──────────────────────────────────────────────────────────

describe("strict copy-sync: index.ts helpers", () => {
  it("extractUsageFromSdkMessage — 4 inline copies mirror index.ts", () => {
    for (const [label, src] of [
      ["silent_failure_recovery.test.mjs", silentFailure],
      ["session_recovery.test.mjs", sessionRecovery],
      ["iterator_drain_recovery.test.mjs", iteratorDrain],
      ["token_accumulation.test.mjs", tokenAccumulation],
    ]) {
      assertInSync(usageTs, src, "extractUsageFromSdkMessage", label);
    }
  });

  it("recoverFromSessionLog — 3 inline copies mirror index.ts", () => {
    for (const [label, src] of [
      ["silent_failure_recovery.test.mjs", silentFailure],
      ["session_recovery.test.mjs", sessionRecovery],
      ["iterator_drain_recovery.test.mjs", iteratorDrain],
    ]) {
      assertInSync(sessionRecoveryTs, src, "recoverFromSessionLog", label);
    }
  });

  it("resolveClaudeMaxTurns — claude_turn_budget copy mirrors index.ts", () => {
    assertInSync(
      queryConfigTs,
      claudeBudget,
      "resolveClaudeMaxTurns",
      "claude_turn_budget.test.mjs",
    );
  });

  it("resolveTurnBudgets — claude_turn_budget copy mirrors query_config.ts", () => {
    assertInSync(
      queryConfigTs,
      claudeBudget,
      "resolveTurnBudgets",
      "claude_turn_budget.test.mjs",
    );
  });

  it("maybeRecoverFromEmptyScheduledOutcome — silent_failure_recovery FULL copy mirrors index.ts", () => {
    assertInSync(
      sessionRecoveryTs,
      silentFailure,
      "maybeRecoverFromEmptyScheduledOutcome",
      "silent_failure_recovery.test.mjs",
    );
  });
});

describe("strict copy-sync: openai.ts helpers", () => {
  it("resolveOpenAIMaxTurns — openai_turn_budget copy mirrors openai.ts", () => {
    assertInSync(
      openaiTs,
      openaiBudget,
      "resolveOpenAIMaxTurns",
      "openai_turn_budget.test.mjs",
    );
  });

  it("salvageResultFromState — openai_turn_budget copy mirrors openai.ts", () => {
    assertInSync(
      openaiTs,
      openaiBudget,
      "salvageResultFromState",
      "openai_turn_budget.test.mjs",
    );
  });
});

describe("strict copy-sync: adapter_safety.ts helpers", () => {
  it("buildSilentDiagnostic — silent_failure_recovery copy mirrors adapter_safety.ts", () => {
    assertInSync(
      adapterSafetyTs,
      silentFailure,
      "buildSilentDiagnostic",
      "silent_failure_recovery.test.mjs",
    );
  });
});

describe("strict copy-sync: main.ts helpers", () => {
  it("pickEffectiveOutput — envelope_drop_status copy mirrors main.ts", () => {
    assertInSync(
      mainTs,
      envelopeDropStatus,
      "pickEffectiveOutput",
      "envelope_drop_status.test.mjs",
    );
  });
});

// ── Anchor guards (documented divergences / unchecked defaults) ───────────

describe("anchor guards: deliberate trims and legacy shims", () => {
  // The wording of all three trigger branches, exactly as emitted by the TS
  // recovery_note ternary. A reword in index.ts must show up here.
  const TRIGGER_WORDINGS = [
    "emitted an empty result event with zero tokens and no tool trace",
    "drained mid-conversation with pending tool calls (no tool_result on the wire)",
    "drained without yielding events to the Node wrapper",
  ];

  it("iterator_drain_recovery's TRIMMED maybeRecover copy still carries every trigger wording", () => {
    // Function-scoped: a wording quoted in a comment or test assertion
    // elsewhere in the file must not satisfy the anchor.
    const tsFnBody = extractFunction(
      sessionRecoveryTs,
      "maybeRecoverFromEmptyScheduledOutcome",
    ).body;
    const trimBody = extractFunction(
      iteratorDrain,
      "maybeRecoverFromEmptyScheduledOutcome",
    ).body;
    for (const wording of TRIGGER_WORDINGS) {
      assert.ok(
        tsFnBody.includes(wording),
        `index.ts lost the wording '${wording}' — update the anchors AND the copies`,
      );
      assert.ok(
        trimBody.includes(wording),
        `iterator_drain_recovery.test.mjs trimmed copy lost '${wording}'`,
      );
    }
    for (const marker of ["recovered_from_session", "session_recovery"]) {
      assert.ok(tsFnBody.includes(`'${marker}'`));
      assert.ok(
        trimBody.includes(`'${marker}'`) || trimBody.includes(`"${marker}"`),
        `iterator_drain_recovery.test.mjs lost envelope marker '${marker}'`,
      );
    }
  });

  it("all three trigger tags exist in index.ts and in the synced full copy", () => {
    for (const trigger of [
      "empty_result_event",
      "silent_exit",
      "iterator_drain_pending_tools",
    ]) {
      assert.ok(
        sessionRecoveryTs.includes(`'${trigger}'`),
        `session_recovery.ts lost trigger '${trigger}'`,
      );
      assert.ok(
        silentFailure.includes(`'${trigger}'`) ||
          silentFailure.includes(`"${trigger}"`),
        `silent_failure_recovery.test.mjs lost trigger '${trigger}'`,
      );
    }
  });

  it("silent_diagnostic's legacy-compat buildSilentDiagnostic emits every field of the TS diagnostic", () => {
    // This copy deliberately diverges from adapter_safety.ts: it keeps a
    // shim for pre-refactor call sites. Pin the shim and the output shape.
    assert.ok(
      silentDiagnostic.includes("args.mcpServerNames ?? args.mcpServerConfig"),
      "legacy-compat shim missing — if removed, promote this copy to a strict guard",
    );
    // normalizeBody strips comments, so a field name surviving only in a
    // comment (e.g. the shim's "Accept either mcpServerNames ..." note)
    // cannot satisfy the anchor.
    const tsBody = normalizeBody(
      extractFunction(adapterSafetyTs, "buildSilentDiagnostic").body,
    );
    const jsBody = normalizeBody(
      extractFunction(silentDiagnostic, "buildSilentDiagnostic").body,
    );
    const FIELDS = [
      "provider",
      "messageCount",
      "resultCount",
      "lastAssistantFallbackKind",
      "perTurnInputTokens",
      "perTurnOutputTokens",
      "resultInputTokens",
      "resultOutputTokens",
      "sdkEnvKeys",
      "apiKeyEnvName",
      "apiKeyType",
      "apiKeyLength",
      "oauthTokenEnvName",
      "oauthTokenType",
      "oauthTokenLength",
      "logLevel",
      "anthropicLog",
      "model",
      "modelProvider",
      "modelAuthMode",
      "mcpServerNames",
      "iteratorError",
    ];
    for (const field of FIELDS) {
      assert.ok(
        tsBody.includes(field),
        `adapter_safety.ts buildSilentDiagnostic no longer mentions '${field}' — update this anchor list`,
      );
      assert.ok(
        jsBody.includes(field),
        `silent_diagnostic.test.mjs copy lost diagnostic field '${field}'`,
      );
    }
  });

  it("parameter defaults excluded from the strict compare are pinned here", () => {
    // recoverFromSessionLog's default rootDir + injectable readFileImpl.
    // The CONTAINER_CLAUDE_SESSIONS_ROOT constant now lives in
    // runner_constants.ts; the function defaults live in session_recovery.ts.
    assert.ok(
      runnerConstantsTs.includes(
        "const CONTAINER_CLAUDE_SESSIONS_ROOT = '/home/node/.claude/projects';",
      ),
    );
    assert.ok(
      sessionRecoveryTs.includes("rootDir: string = CONTAINER_CLAUDE_SESSIONS_ROOT"),
    );
    assert.ok(
      sessionRecoveryTs.includes("= (p) => fs.readFileSync(p, 'utf-8')"),
      "session_recovery.ts readFileImpl default changed — update the copies",
    );
    for (const [label, src] of [
      ["silent_failure_recovery.test.mjs", silentFailure],
      ["session_recovery.test.mjs", sessionRecovery],
      ["iterator_drain_recovery.test.mjs", iteratorDrain],
    ]) {
      assert.ok(
        src.includes(
          "const CONTAINER_CLAUDE_SESSIONS_ROOT = '/home/node/.claude/projects';",
        ),
        `${label}: CONTAINER_CLAUDE_SESSIONS_ROOT constant missing/changed`,
      );
      assert.ok(
        src.includes("rootDir = CONTAINER_CLAUDE_SESSIONS_ROOT"),
        `${label}: rootDir default missing/changed`,
      );
      assert.ok(
        src.includes("readFileImpl = (p) => fs.readFileSync(p, 'utf-8')"),
        `${label}: readFileImpl default missing/changed`,
      );
    }
    // maybeRecoverFromEmptyScheduledOutcome's default recoverImpl.
    assert.ok(sessionRecoveryTs.includes("= () => recoverFromSessionLog()"));
    assert.ok(
      silentFailure.includes("recoverImpl = () => recoverFromSessionLog()"),
      "silent_failure_recovery.test.mjs: recoverImpl default missing/changed",
    );
  });
});
