/**
 * Regression tests for the 2026-04-20 v3 audit silent-failure recovery-path
 * gap: the agent-runner emitted `status='success'` with zero tokens / null
 * result whenever the SDK yielded a `result` event that was empty (no text
 * AND no tokens AND empty tool trace). The host then relabelled it to
 * `silent_failure` at DB-write time, but the post-loop recovery branch was
 * already skipped (guarded by `resultCount === 0`, which was now 1), so the
 * on-disk session log — a 160 KB / 60-turn artifact proving the CLI ran to
 * completion — was thrown away.
 *
 * Fix (runtime_runner/agent-runner/src/index.ts):
 *   1. Extracted `maybeRecoverFromEmptyScheduledOutcome()` — single helper
 *      that either returns a `recovered_from_session` envelope (when the
 *      session log has usable turns) or an `error` envelope with the full
 *      `buildSilentDiagnostic` snapshot.
 *   2. The `if (message.type === 'result')` branch now detects the empty
 *      pattern (no effective result text + zero tokens
 *      + empty toolTrace) and routes through the helper instead of
 *      unconditionally emitting success.
 *   3. The post-loop silent-exit branch (was `resultCount === 0 && no
 *      tokens`) now calls the same helper, eliminating duplication.
 *
 * Since then the helper grew (and this file's copies are re-synced to):
 *   - a THIRD trigger, `iterator_drain_pending_tools` (PR #527 / #525):
 *     non-strict tasks whose SDK iterator drains mid-conversation with
 *     pending tool calls route through the same helper;
 *   - marker phrases sourced from runtime_runner/shared/retryable_markers.json
 *     via `emitPhrase(...)` (issue #648) instead of hardcoded strings;
 *   - the provider-aware `buildSilentDiagnostic` that moved to
 *     runtime_runner/agent-runner/src/adapter_safety.ts (takes
 *     `provider` + `mcpServerNames`, emits `provider` / `apiKeyEnvName` /
 *     `oauthTokenEnvName`).
 *
 * This test file duplicates the helpers inline because the repo's test
 * harness runs Node directly (no tsc step). The copies below are faithful
 * type-stripped mirrors of their TS sources and are enforced against drift
 * by tests/js/copy_sync_guard.test.mjs (issue #734) — if the TS changes,
 * that guard fails and the copies here must be re-synced.
 */

import { strict as assert } from "node:assert";
import { describe, it, beforeEach, afterEach } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// ── Marker phrases + emitPhrase, sourced exactly like the TS does ─────────
// index.ts emits its silent-failure log/diagnostic phrases via emitPhrase()
// over markers loaded from runtime_runner/shared/retryable_markers.json
// (issue #648). Rather than hardcoding the phrases here (the exact drift
// issue #734 fixed), load the same JSON and execute the REAL emitPhrase
// body extracted from retryable_markers.ts — same technique as
// retryable_markers_parity.test.mjs.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const markersJson = JSON.parse(
  fs.readFileSync(
    path.join(repoRoot, "runtime_runner", "shared", "retryable_markers.json"),
    "utf-8",
  ),
);
const markersTsSource = fs.readFileSync(
  path.join(
    repoRoot,
    "runtime_runner",
    "agent-runner",
    "src",
    "retryable_markers.ts",
  ),
  "utf-8",
);
const emitPhraseMatch = markersTsSource.match(
  /export function emitPhrase\(marker: string\): string \{([\s\S]*?)\n\}/,
);
assert.ok(emitPhraseMatch, "emitPhrase not found in retryable_markers.ts");
// eslint-disable-next-line no-new-func
const emitPhrase = new Function("marker", emitPhraseMatch[1]);

function requireStreamRaceMarker(startsWith) {
  const found = markersJson.categories.stream_race.find((m) =>
    m.toLowerCase().startsWith(startsWith),
  );
  assert.ok(found, `no stream_race marker starting with '${startsWith}'`);
  return found;
}
const MARKER_SILENT_AGENT_RUNNER_FAILURE = requireStreamRaceMarker(
  "silent agent-runner failure",
);
const MARKER_SDK_EMPTY_RESULT_EVENT = requireStreamRaceMarker(
  "sdk emitted an empty result event",
);
const MARKER_SDK_QUERY_LOOP_DRAINED = requireStreamRaceMarker(
  "sdk query loop drained",
);
const MARKER_SDK_QUERY_ITERATOR_THREW = requireStreamRaceMarker(
  "sdk query iterator threw",
);

// ── Local copy of `extractUsageFromSdkMessage` from index.ts ──────────────
// Faithful type-stripped mirror; drift-guarded by copy_sync_guard.test.mjs.
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

// ── Local copy of `recoverFromSessionLog` from index.ts ───────────────────
// Faithful type-stripped mirror; drift-guarded by copy_sync_guard.test.mjs.
const CONTAINER_CLAUDE_SESSIONS_ROOT = '/home/node/.claude/projects';

function recoverFromSessionLog(
  rootDir = CONTAINER_CLAUDE_SESSIONS_ROOT,
  readFileImpl = (p) => fs.readFileSync(p, 'utf-8'),
) {
  // 1) Locate session log files.
  let candidates = [];
  try {
    if (!fs.existsSync(rootDir)) return null;
    const walk = (dir) => {
      let entries;
      try {
        entries = fs.readdirSync(dir, { withFileTypes: true });
      } catch {
        return;
      }
      for (const ent of entries) {
        const full = path.join(dir, ent.name);
        if (ent.isDirectory()) {
          walk(full);
        } else if (ent.isFile() && ent.name.endsWith('.jsonl')) {
          candidates.push(full);
        }
      }
    };
    walk(rootDir);
  } catch {
    return null;
  }
  if (candidates.length === 0) return null;

  // Pick the most-recently-modified jsonl — that's the log for this session.
  candidates.sort((a, b) => {
    let am = 0;
    let bm = 0;
    try { am = fs.statSync(a).mtimeMs; } catch { /* ignore */ }
    try { bm = fs.statSync(b).mtimeMs; } catch { /* ignore */ }
    return bm - am;
  });
  const sourcePath = candidates[0];

  // 2) Parse the JSONL.
  let raw;
  try {
    raw = readFileImpl(sourcePath);
  } catch {
    return null;
  }
  let turnCount = 0;
  let toolUseCount = 0;
  let lastAssistantText = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheCreationTokens = 0;
  let cacheReadTokens = 0;

  for (const line of raw.split('\n')) {
    if (!line.trim()) continue;
    let entry;
    try {
      entry = JSON.parse(line);
    } catch {
      continue;
    }
    if (!entry || typeof entry !== 'object') continue;
    // Count anything that looks like a turn (assistant / user / tool roles).
    const t = typeof entry.type === 'string' ? entry.type : '';
    if (t === 'assistant' || t === 'user') {
      turnCount += 1;
    }
    // Per-turn usage may be nested under .message.usage (assistant turns).
    const delta = extractUsageFromSdkMessage(entry);
    inputTokens += delta.input_tokens;
    outputTokens += delta.output_tokens;
    cacheCreationTokens += delta.cache_creation_input_tokens;
    cacheReadTokens += delta.cache_read_input_tokens;

    // Collect assistant text + tool-use counts.
    if (t === 'assistant') {
      const inner = (entry.message && typeof entry.message === 'object')
        ? entry.message
        : null;
      const content = inner && Array.isArray(inner.content) ? inner.content : [];
      const textParts = [];
      for (const block of content) {
        if (!block || typeof block !== 'object') continue;
        const b = block;
        if (b.type === 'text' && typeof b.text === 'string') {
          textParts.push(b.text);
        } else if (b.type === 'tool_use') {
          toolUseCount += 1;
        }
      }
      const joined = textParts.join('').trim();
      if (joined.length > 0) {
        lastAssistantText = joined;
      }
    }
  }

  if (lastAssistantText === null && turnCount === 0) {
    // Nothing usable in the log — let the caller fall through to status=error.
    return null;
  }

  return {
    result: lastAssistantText,
    toolUseCount,
    inputTokens,
    outputTokens,
    cacheCreationTokens,
    cacheReadTokens,
    sourcePath,
    turnCount,
  };
}

// ── Local copy of `buildSilentDiagnostic` from adapter_safety.ts ──────────
// The canonical source moved from index.ts to
// runtime_runner/agent-runner/src/adapter_safety.ts and is provider-aware:
// it takes `provider` (a ProviderAuthConfig) plus `mcpServerNames` (string[]
// or a legacy mcpServerConfig dict) and emits `provider` / `apiKeyEnvName` /
// `oauthTokenEnvName` in the diagnostic. Faithful type-stripped mirror;
// drift-guarded by copy_sync_guard.test.mjs.
const ANTHROPIC_PROVIDER = {
  id: 'anthropic',
  apiKeyEnvName: 'ANTHROPIC_API_KEY',
  oauthTokenEnvName: 'CLAUDE_CODE_OAUTH_TOKEN',
};

function buildSilentDiagnostic(args) {
  const apiKey = args.sdkEnv[args.provider.apiKeyEnvName];
  const oauthEnvName = args.provider.oauthTokenEnvName || '';
  const oauthTok = oauthEnvName ? args.sdkEnv[oauthEnvName] : undefined;
  const sdkEnvKeys = Object.keys(args.sdkEnv).sort();
  const mcpServerNames = Array.isArray(args.mcpServerNames)
    ? [...args.mcpServerNames].sort()
    : Object.keys(args.mcpServerNames || {}).sort();

  let causeDesc;
  if (args.iteratorError && typeof args.iteratorError === 'object') {
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
    lastAssistantFallbackKind: args.lastAssistantFallback === null ? 'null' : 'non-null',
    perTurnInputTokens: args.perTurnInputTokens,
    perTurnOutputTokens: args.perTurnOutputTokens,
    resultInputTokens: args.resultInputTokens,
    resultOutputTokens: args.resultOutputTokens,
    sdkEnvKeys,
    apiKeyEnvName: args.provider.apiKeyEnvName,
    apiKeyType: typeof apiKey,
    apiKeyLength: typeof apiKey === 'string' ? apiKey.length : 0,
    oauthTokenEnvName: oauthEnvName,
    oauthTokenType: typeof oauthTok,
    oauthTokenLength: typeof oauthTok === 'string' ? oauthTok.length : 0,
    logLevel: args.sdkEnv.LOG_LEVEL,
    anthropicLog: args.sdkEnv.ANTHROPIC_LOG,
    model: args.sdkEnv.MODEL,
    modelProvider: args.sdkEnv.MODEL_PROVIDER,
    modelAuthMode: args.sdkEnv.MODEL_AUTH_MODE,
    mcpServerNames,
    iteratorError: args.iteratorError
      ? {
          message: String(args.iteratorError.message || ''),
          name: args.iteratorError.name || 'Error',
          stackHead: args.iteratorError.stack
            ? String(args.iteratorError.stack).split('\n').slice(0, 6).join('\n')
            : undefined,
          cause: causeDesc,
        }
      : null,
  };
}

// ── Local copy of `maybeRecoverFromEmptyScheduledOutcome` from index.ts ──
// Faithful type-stripped mirror of runtime_runner/agent-runner/src/index.ts
// (currently lines 503-637); drift-guarded by copy_sync_guard.test.mjs.
// Covers all THREE triggers: 'empty_result_event', 'silent_exit', and
// 'iterator_drain_pending_tools'.
function maybeRecoverFromEmptyScheduledOutcome(
  ctx,
  recoverImpl = () => recoverFromSessionLog(),
) {
  const logLines = [];
  let recovered = null;
  try {
    recovered = recoverImpl();
  } catch (err) {
    logLines.push(
      `Silent-exit recovery threw (trigger=${ctx.trigger}): ` +
      `${err instanceof Error ? err.message : String(err)} — ` +
      `falling through to diagnostic status=error.`,
    );
    recovered = null;
  }

  if (recovered && (recovered.result || recovered.turnCount > 0)) {
    const tokenTotal =
      recovered.inputTokens + recovered.outputTokens
      + recovered.cacheCreationTokens + recovered.cacheReadTokens;
    const recoveryNote =
      `Recovered from on-disk session log at ${recovered.sourcePath}: ` +
      `${recovered.turnCount} turns, ${recovered.toolUseCount} tool_use blocks, ` +
      `~${tokenTotal} tokens (summed from per-turn usage). ` +
      `The claude-agent-sdk iterator ${
        ctx.trigger === 'empty_result_event'
          ? 'emitted an empty result event with zero tokens and no tool trace'
          : ctx.trigger === 'iterator_drain_pending_tools'
            ? 'drained mid-conversation with pending tool calls (no tool_result on the wire)'
            : 'drained without yielding events to the Node wrapper'
      }, but the underlying CLI subprocess ran to completion. This output ` +
      `was reconstructed from that log so downstream evaluators have something ` +
      `to score; treat the tokens as approximate.`;
    logLines.push(
      `Silent-exit recovery succeeded (trigger=${ctx.trigger}): ` +
      `turns=${recovered.turnCount}, ` +
      `tools=${recovered.toolUseCount}, tokens=${tokenTotal}, ` +
      `resultLen=${recovered.result ? recovered.result.length : 0}.`,
    );
    return {
      recovered,
      logLines,
      envelope: {
        status: 'recovered_from_session',
        result: recovered.result,
        newSessionId: ctx.newSessionId,
        toolTrace: ctx.toolTrace.slice(-1000),
        input_tokens: recovered.inputTokens,
        output_tokens: recovered.outputTokens,
        cache_creation_input_tokens: recovered.cacheCreationTokens,
        cache_read_input_tokens: recovered.cacheReadTokens,
        tokens_source: 'session_recovery',
        recovery_note: recoveryNote,
      },
    };
  }

  // No usable log → diagnostic envelope. Reconstruct the iterator error
  // as a real Error instance for buildSilentDiagnostic's typed signature.
  const iteratorError = ctx.iteratorError
    ? (() => {
        const e = new Error(ctx.iteratorError.message);
        e.name = ctx.iteratorError.name;
        if (ctx.iteratorError.stack) e.stack = ctx.iteratorError.stack;
        if (ctx.iteratorError.cause !== undefined) {
          e.cause = ctx.iteratorError.cause;
        }
        return e;
      })()
    : null;
  const diag = buildSilentDiagnostic({
    messageCount: ctx.messageCount,
    resultCount: ctx.resultCount,
    lastAssistantFallback: ctx.lastAssistantFallback,
    perTurnInputTokens: ctx.perTurnInputTokens,
    perTurnOutputTokens: ctx.perTurnOutputTokens,
    resultInputTokens: ctx.resultInputTokens,
    resultOutputTokens: ctx.resultOutputTokens,
    sdkEnv: ctx.sdkEnv,
    provider: ANTHROPIC_PROVIDER,
    mcpServerNames: ctx.mcpServerConfig,
    iteratorError,
  });
  logLines.push(
    // Marker sourced from runtime_runner/shared/retryable_markers.json so the
    // substring engine.py matches stays in lockstep. See issue #648.
    `${emitPhrase(MARKER_SILENT_AGENT_RUNNER_FAILURE)} (trigger=${ctx.trigger}): ` +
    `messages=${diag.messageCount}, ` +
    `results=${diag.resultCount}, ` +
    `assistantFallback=${diag.lastAssistantFallbackKind}, ` +
    `tokens=0/0, ` +
    `iteratorError=${
      diag.iteratorError
        ? diag.iteratorError.name + ':' + diag.iteratorError.message.slice(0, 80)
        : 'null'
    }, ` +
    `session-log recovery=${recovered === null ? 'none' : 'empty'}. ` +
    `Emitting status=error with diagnostic envelope.`,
  );
  logLines.push(`Silent-exit diagnostic: ${JSON.stringify(diag)}`);

  // Marker prefixes sourced from runtime_runner/shared/retryable_markers.json
  // (see issue #648) so they stay in lockstep with engine.py's substring match.
  const triggerSummary = ctx.trigger === 'empty_result_event'
    ? `${emitPhrase(MARKER_SDK_EMPTY_RESULT_EVENT)} (no text, zero tokens, empty tool trace)`
    : ctx.trigger === 'iterator_drain_pending_tools'
      ? `SDK iterator drained mid-conversation with pending tool calls (messageCount=${diag.messageCount})`
      : `${emitPhrase(MARKER_SDK_QUERY_LOOP_DRAINED)} without yielding any assistant/result message (messageCount=${diag.messageCount})`;
  const errorSummary = ctx.iteratorError
    ? `${emitPhrase(MARKER_SDK_QUERY_ITERATOR_THREW)} ${ctx.iteratorError.name}: ${ctx.iteratorError.message.slice(0, 240)}`
    : triggerSummary;

  return {
    recovered,
    logLines,
    envelope: {
      status: 'error',
      result: null,
      newSessionId: ctx.newSessionId,
      toolTrace: ctx.toolTrace.slice(-1000),
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      tokens_source: 'unavailable',
      error:
        `agent-runner produced no output: ${errorSummary}. ` +
        `Session-log recovery also failed (${recovered === null ? 'none' : 'empty log'}). ` +
        `This is the "silent exit" pattern -- auth/startup failure inside the container, ` +
        `MCP server hang, or a claude-agent-sdk stream that closed before emitting any events. ` +
        `trigger=${ctx.trigger} diagnostic=${JSON.stringify(diag)}`,
    },
  };
}

// ── End-to-end simulation of the result-event branch in runQuery ─────────
//
// Mirrors the post-fix code in runtime_runner/agent-runner/src/index.ts:
//   - empty result text + zero tokens + empty toolTrace
//     → route through maybeRecoverFromEmptyScheduledOutcome
//   - otherwise → emit status='success' with the accumulated tokens
//
// Kept separate from maybeRecoverFromEmptyScheduledOutcome so the tests can
// assert on the routing decision directly AND cross-check that the shared
// helper is produced identical output for both call sites.
function simulateResultEventEmit(ctx) {
  const {
    resultTextFromMessage,
    bestStructuredForumText,
    resultInputTokens,
    resultOutputTokens,
    resultCacheCreationTokens,
    resultCacheReadTokens,
    perTurnInputTokens,
    perTurnOutputTokens,
    perTurnCacheCreationTokens,
    perTurnCacheReadTokens,
    toolTrace,
    newSessionId,
    sdkEnv,
    mcpServerConfig,
    iteratorError,
    sessionsRoot,
  } = ctx;

  const effectiveResult =
    bestStructuredForumText || resultTextFromMessage || null;
  const finalInput = resultInputTokens || perTurnInputTokens;
  const finalOutput = resultOutputTokens || perTurnOutputTokens;
  const finalCacheCreate =
    resultCacheCreationTokens || perTurnCacheCreationTokens;
  const finalCacheRead = resultCacheReadTokens || perTurnCacheReadTokens;
  const resultEventHasTokens =
    resultInputTokens +
      resultOutputTokens +
      resultCacheCreationTokens +
      resultCacheReadTokens >
    0;
  const tokensMissing =
    finalInput + finalOutput + finalCacheCreate + finalCacheRead === 0;
  const resultIsEmpty = !effectiveResult || !String(effectiveResult).trim();
  const handledAsEmptyRecovery =
    resultIsEmpty && tokensMissing && toolTrace.length === 0;
  if (handledAsEmptyRecovery) {
    const outcome = maybeRecoverFromEmptyScheduledOutcome(
      {
        messageCount: ctx.messageCount ?? 0,
        resultCount: 1,
        lastAssistantFallback: null,
        perTurnInputTokens,
        perTurnOutputTokens,
        perTurnCacheCreationTokens,
        perTurnCacheReadTokens,
        resultInputTokens,
        resultOutputTokens,
        resultCacheCreationTokens,
        resultCacheReadTokens,
        toolTrace,
        newSessionId,
        sdkEnv,
        mcpServerConfig,
        iteratorError,
        trigger: "empty_result_event",
      },
      () => recoverFromSessionLog(sessionsRoot),
    );
    return { envelope: outcome.envelope, recovered: outcome.recovered };
  }
  return {
    envelope: {
      status: "success",
      result: effectiveResult,
      newSessionId,
      toolTrace: toolTrace.slice(-1000),
      input_tokens: finalInput,
      output_tokens: finalOutput,
      cache_creation_input_tokens: finalCacheCreate,
      cache_read_input_tokens: finalCacheRead,
      tokens_source:
        resultEventHasTokens
          ? "result_event"
          : tokensMissing
            ? "unavailable"
            : "per_turn_sum",
    },
    recovered: null,
  };
}

// ── Fixtures ─────────────────────────────────────────────────────────────

function writeJsonl(filePath, entries) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(
    filePath,
    entries.map((e) => JSON.stringify(e)).join("\n") + "\n",
  );
}

function makeTempRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "kcsi-empty-result-"));
}

function baseCtx(overrides = {}) {
  return {
    resultTextFromMessage: null,
    bestStructuredForumText: null,
    resultInputTokens: 0,
    resultOutputTokens: 0,
    resultCacheCreationTokens: 0,
    resultCacheReadTokens: 0,
    perTurnInputTokens: 0,
    perTurnOutputTokens: 0,
    perTurnCacheCreationTokens: 0,
    perTurnCacheReadTokens: 0,
    toolTrace: [],
    newSessionId: "sess-xyz",
    sdkEnv: {
      ANTHROPIC_API_KEY: "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      LOG_LEVEL: "info",
      MODEL: "claude-haiku-4-5",
      MODEL_PROVIDER: "anthropic",
    },
    mcpServerConfig: {},
    iteratorError: null,
    messageCount: 10,
    sessionsRoot: "/nonexistent/default",
    ...overrides,
  };
}

// ── Tests: empty-result-event routing (the primary v3 bug) ───────────────

describe("empty result event — recovery routing", () => {
  let tmp;
  beforeEach(() => {
    tmp = makeTempRoot();
  });
  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("null result + 0 tokens + empty toolTrace + session log present → emits recovered_from_session", () => {
    const logPath = path.join(tmp, "projects", "slug", "session.jsonl");
    writeJsonl(logPath, [
      { type: "user", message: { role: "user", content: "solve it" } },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "tool_use", id: "t1", name: "Read", input: {} },
            { type: "text", text: "partial reasoning" },
          ],
          usage: { input_tokens: 500, output_tokens: 50 },
        },
      },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "the final recovered answer" }],
          usage: { input_tokens: 300, output_tokens: 40 },
        },
      },
    ]);
    const { envelope, recovered } = simulateResultEventEmit(
      baseCtx({ sessionsRoot: tmp }),
    );
    assert.equal(
      envelope.status,
      "recovered_from_session",
      "must recover, not emit success",
    );
    assert.equal(envelope.result, "the final recovered answer");
    assert.equal(envelope.tokens_source, "session_recovery");
    assert.equal(envelope.input_tokens, 800);
    assert.equal(envelope.output_tokens, 90);
    assert.ok(envelope.recovery_note);
    assert.ok(
      envelope.recovery_note.includes("empty result event"),
      "recovery note must explain the empty-result trigger",
    );
    assert.ok(recovered, "recovered payload must be populated");
    assert.equal(recovered.turnCount, 3);
  });

  it("null result + 0 tokens + empty toolTrace + NO session log → emits status=error, NOT status=success", () => {
    // Directly point at an empty temp dir — recovery returns null; helper
    // MUST emit status=error rather than mask as success.
    const { envelope } = simulateResultEventEmit(
      baseCtx({ sessionsRoot: tmp }),
    );
    assert.equal(
      envelope.status,
      "error",
      "no session log → must NOT emit status=success",
    );
    assert.equal(envelope.result, null);
    assert.equal(envelope.input_tokens, 0);
    assert.equal(envelope.output_tokens, 0);
    assert.equal(envelope.tokens_source, "unavailable");
    assert.ok(envelope.error, "error envelope must carry a diagnostic string");
    assert.ok(
      envelope.error.includes(emitPhrase(MARKER_SDK_EMPTY_RESULT_EVENT)),
      "diagnostic must lead with the marker-sourced empty-result phrase",
    );
    assert.ok(
      envelope.error.includes("trigger=empty_result_event"),
      "diagnostic must tag trigger",
    );
    assert.ok(
      envelope.error.includes("diagnostic="),
      "diagnostic snapshot must be embedded",
    );
    // The diagnostic is provider-aware now (adapter_safety.ts): it names the
    // provider and the auth env vars it inspected.
    assert.ok(
      envelope.error.includes('"provider":"anthropic"'),
      "diagnostic must carry the provider id",
    );
    assert.ok(
      envelope.error.includes('"apiKeyEnvName":"ANTHROPIC_API_KEY"'),
      "diagnostic must name the API-key env var",
    );
    assert.ok(
      envelope.error.includes('"oauthTokenEnvName":"CLAUDE_CODE_OAUTH_TOKEN"'),
      "diagnostic must name the OAuth-token env var",
    );
  });

  it("result event with real text + 0 tokens → still emits status=success (regression check)", () => {
    // An agent that actually spoke but whose SDK result-event usage block
    // was dropped is a different failure mode. We do NOT want to recover
    // from session log here — non-empty text means the agent DID produce
    // a final answer; the zero is a reporting gap, not a silent failure.
    const { envelope } = simulateResultEventEmit(
      baseCtx({
        sessionsRoot: "/nonexistent/should/not/matter",
        resultTextFromMessage: "real answer produced by the agent",
      }),
    );
    assert.equal(envelope.status, "success");
    assert.equal(envelope.result, "real answer produced by the agent");
    assert.equal(envelope.tokens_source, "unavailable");
  });

  it("result event with no text but non-zero tokens → still emits status=success (degenerate edge case)", () => {
    // The agent used tokens but returned nothing textual — e.g., it ran
    // tool_use loops that timed out before a final text block. Not an
    // empty-result silent fail (tokens prove work happened); fall through
    // to success emit with tokens_source='result_event'.
    const { envelope } = simulateResultEventEmit(
      baseCtx({
        sessionsRoot: "/nonexistent/should/not/matter",
        resultTextFromMessage: null,
        resultInputTokens: 1200,
        resultOutputTokens: 30,
      }),
    );
    assert.equal(envelope.status, "success");
    assert.equal(envelope.result, null);
    assert.equal(envelope.input_tokens, 1200);
    assert.equal(envelope.output_tokens, 30);
    assert.equal(envelope.tokens_source, "result_event");
  });

  it("cache-only result event tokens count as result_event usage", () => {
    const { envelope } = simulateResultEventEmit(
      baseCtx({
        sessionsRoot: "/nonexistent/should/not/matter",
        resultTextFromMessage: null,
        resultCacheReadTokens: 1200,
      }),
    );
    assert.equal(envelope.status, "success");
    assert.equal(envelope.result, null);
    assert.equal(envelope.input_tokens, 0);
    assert.equal(envelope.output_tokens, 0);
    assert.equal(envelope.cache_read_input_tokens, 1200);
    assert.equal(envelope.tokens_source, "result_event");
  });

  it("cache-only per-turn usage is not treated as an empty result event", () => {
    const { envelope } = simulateResultEventEmit(
      baseCtx({
        sessionsRoot: "/nonexistent/should/not/matter",
        resultTextFromMessage: null,
        perTurnCacheCreationTokens: 900,
      }),
    );
    assert.equal(envelope.status, "success");
    assert.equal(envelope.result, null);
    assert.equal(envelope.cache_creation_input_tokens, 900);
    assert.equal(envelope.tokens_source, "per_turn_sum");
  });

  it("result event with no text + 0 tokens but NON-EMPTY toolTrace → emits status=success (not recovery)", () => {
    // Tools were invoked (evidence the SDK did yield messages) but the
    // final result event was empty. The empty-result detector intentionally
    // bails when toolTrace has entries — we don't want to overwrite a
    // successful-tool-execution attempt with a diagnostic error.
    const { envelope } = simulateResultEventEmit(
      baseCtx({
        sessionsRoot: "/nonexistent/should/not/matter",
        toolTrace: [{ idx: 1, type: "tool_call", tool_name: "Read" }],
      }),
    );
    assert.equal(envelope.status, "success");
    assert.equal(envelope.result, null);
    assert.equal(envelope.tokens_source, "unavailable");
    assert.equal(envelope.toolTrace.length, 1);
  });
});

// ── Tests: the shared helper is identical for all call sites ─────────────

describe("maybeRecoverFromEmptyScheduledOutcome — call site parity", () => {
  let tmp;
  beforeEach(() => {
    tmp = makeTempRoot();
  });
  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  function ctxCommon(overrides = {}) {
    return {
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      perTurnCacheCreationTokens: 0,
      perTurnCacheReadTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      resultCacheCreationTokens: 0,
      resultCacheReadTokens: 0,
      toolTrace: [],
      newSessionId: "sess-1",
      sdkEnv: { ANTHROPIC_API_KEY: "x".repeat(64), MODEL: "claude-haiku-4-5" },
      mcpServerConfig: {},
      iteratorError: null,
      ...overrides,
    };
  }

  it("recovered envelope is structurally identical for both triggers except recovery_note wording", () => {
    const logPath = path.join(tmp, "projects", "slug", "session.jsonl");
    writeJsonl(logPath, [
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "recovered payload" }],
          usage: { input_tokens: 100, output_tokens: 20 },
        },
      },
    ]);
    const emptyResultOutcome = maybeRecoverFromEmptyScheduledOutcome(
      ctxCommon({ resultCount: 1, trigger: "empty_result_event" }),
      () => recoverFromSessionLog(tmp),
    );
    const silentExitOutcome = maybeRecoverFromEmptyScheduledOutcome(
      ctxCommon({ resultCount: 0, trigger: "silent_exit" }),
      () => recoverFromSessionLog(tmp),
    );
    // Status / result / tokens must be byte-identical — the helper is the
    // single source of truth for envelope shape on recovery.
    assert.equal(
      emptyResultOutcome.envelope.status,
      silentExitOutcome.envelope.status,
    );
    assert.equal(
      emptyResultOutcome.envelope.status,
      "recovered_from_session",
    );
    assert.equal(
      emptyResultOutcome.envelope.result,
      silentExitOutcome.envelope.result,
    );
    assert.equal(
      emptyResultOutcome.envelope.input_tokens,
      silentExitOutcome.envelope.input_tokens,
    );
    assert.equal(
      emptyResultOutcome.envelope.output_tokens,
      silentExitOutcome.envelope.output_tokens,
    );
    assert.equal(
      emptyResultOutcome.envelope.tokens_source,
      silentExitOutcome.envelope.tokens_source,
    );
    // Recovery note MUST differ — it carries the trigger wording so
    // post-mortems can tell which code path fired.
    assert.ok(
      emptyResultOutcome.envelope.recovery_note.includes(
        "empty result event",
      ),
    );
    assert.ok(
      silentExitOutcome.envelope.recovery_note.includes(
        "drained without yielding events",
      ),
    );
  });

  it("error envelope is structurally identical for both triggers, only trigger tag differs", () => {
    // Empty temp dir → recovery returns null → diagnostic error envelope.
    const emptyResultOutcome = maybeRecoverFromEmptyScheduledOutcome(
      ctxCommon({ resultCount: 1, trigger: "empty_result_event" }),
      () => recoverFromSessionLog(tmp),
    );
    const silentExitOutcome = maybeRecoverFromEmptyScheduledOutcome(
      ctxCommon({ resultCount: 0, trigger: "silent_exit" }),
      () => recoverFromSessionLog(tmp),
    );
    assert.equal(emptyResultOutcome.envelope.status, "error");
    assert.equal(silentExitOutcome.envelope.status, "error");
    assert.equal(emptyResultOutcome.envelope.result, null);
    assert.equal(silentExitOutcome.envelope.result, null);
    assert.equal(emptyResultOutcome.envelope.input_tokens, 0);
    assert.equal(silentExitOutcome.envelope.input_tokens, 0);
    assert.equal(emptyResultOutcome.envelope.tokens_source, "unavailable");
    assert.equal(silentExitOutcome.envelope.tokens_source, "unavailable");
    assert.ok(
      emptyResultOutcome.envelope.error.includes(
        "trigger=empty_result_event",
      ),
    );
    assert.ok(
      silentExitOutcome.envelope.error.includes("trigger=silent_exit"),
    );
    // The trigger summaries are marker-sourced (issue #648): the phrases the
    // orchestrator's retry gate substring-matches must survive verbatim.
    assert.ok(
      emptyResultOutcome.envelope.error.includes(
        emitPhrase(MARKER_SDK_EMPTY_RESULT_EVENT),
      ),
    );
    assert.ok(
      silentExitOutcome.envelope.error.includes(
        emitPhrase(MARKER_SDK_QUERY_LOOP_DRAINED),
      ),
    );
  });

  it("recovery helper never emits status=success (all no-recovery paths are status=error)", () => {
    // Belt-and-suspenders: the whole point of the fix is to eliminate the
    // status=success-with-zeros mask. For every no-recovery input the
    // helper must return status=error.
    for (const trigger of [
      "empty_result_event",
      "silent_exit",
      "iterator_drain_pending_tools",
    ]) {
      const outcome = maybeRecoverFromEmptyScheduledOutcome(
        ctxCommon({ trigger }),
        () => null, // simulate no session log
      );
      assert.notEqual(
        outcome.envelope.status,
        "success",
        `trigger=${trigger} must not fall back to status=success`,
      );
      assert.equal(outcome.envelope.status, "error");
    }
  });

  it("recovery throws → still emits status=error, not success", () => {
    const outcome = maybeRecoverFromEmptyScheduledOutcome(
      ctxCommon({ trigger: "empty_result_event" }),
      () => {
        throw new Error("fs explosion");
      },
    );
    assert.equal(outcome.envelope.status, "error");
    assert.ok(
      outcome.logLines.some((l) => l.includes("Silent-exit recovery threw")),
    );
  });

  it("iteratorError present → error summary uses the marker-sourced 'iterator threw' phrase", () => {
    const outcome = maybeRecoverFromEmptyScheduledOutcome(
      ctxCommon({
        trigger: "silent_exit",
        iteratorError: {
          message: "ECONNRESET reading stream",
          name: "APIConnectionError",
          stack: "at frame0\nat frame1",
          cause: { kind: "provider", status: 401 },
        },
      }),
      () => null,
    );
    assert.equal(outcome.envelope.status, "error");
    assert.ok(
      outcome.envelope.error.includes(
        emitPhrase(MARKER_SDK_QUERY_ITERATOR_THREW),
      ),
      "iterator-threw marker phrase must lead the error summary",
    );
    assert.ok(
      outcome.envelope.error.includes("APIConnectionError"),
      "error name must be embedded",
    );
  });
});

// ── Tests: third trigger branch (iterator_drain_pending_tools, PR #527) ──

describe("maybeRecoverFromEmptyScheduledOutcome — iterator_drain_pending_tools trigger", () => {
  let tmp;
  beforeEach(() => {
    tmp = makeTempRoot();
  });
  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  function ctxDrain(overrides = {}) {
    return {
      messageCount: 12,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      perTurnCacheCreationTokens: 0,
      perTurnCacheReadTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      resultCacheCreationTokens: 0,
      resultCacheReadTokens: 0,
      toolTrace: [],
      newSessionId: "sess-drain",
      sdkEnv: { ANTHROPIC_API_KEY: "x".repeat(64), MODEL: "claude-haiku-4-5" },
      mcpServerConfig: {},
      iteratorError: null,
      trigger: "iterator_drain_pending_tools",
      ...overrides,
    };
  }

  it("session log present → recovered_from_session with the pending-tools recovery_note wording", () => {
    const logPath = path.join(tmp, "projects", "slug", "session.jsonl");
    writeJsonl(logPath, [
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "text", text: "answer before the drain" },
            { type: "tool_use", id: "t1", name: "Edit", input: {} },
          ],
          usage: { input_tokens: 700, output_tokens: 90 },
        },
      },
    ]);
    const outcome = maybeRecoverFromEmptyScheduledOutcome(
      ctxDrain(),
      () => recoverFromSessionLog(tmp),
    );
    assert.equal(outcome.envelope.status, "recovered_from_session");
    assert.equal(outcome.envelope.result, "answer before the drain");
    assert.equal(outcome.envelope.tokens_source, "session_recovery");
    assert.ok(
      outcome.envelope.recovery_note.includes(
        "drained mid-conversation with pending tool calls (no tool_result on the wire)",
      ),
      "recovery_note must use the iterator_drain_pending_tools wording",
    );
    assert.ok(
      outcome.logLines.some((l) =>
        l.includes("trigger=iterator_drain_pending_tools"),
      ),
      "log lines must carry the trigger tag",
    );
  });

  it("NO session log → status=error with the pending-tools trigger summary (no marker phrase for this branch)", () => {
    const outcome = maybeRecoverFromEmptyScheduledOutcome(
      ctxDrain({ messageCount: 12 }),
      () => recoverFromSessionLog(tmp), // empty dir → null
    );
    assert.equal(outcome.envelope.status, "error");
    assert.equal(outcome.envelope.result, null);
    assert.equal(outcome.envelope.tokens_source, "unavailable");
    assert.ok(
      outcome.envelope.error.includes(
        "SDK iterator drained mid-conversation with pending tool calls (messageCount=12)",
      ),
      "trigger summary must identify the pending-tools drain and messageCount",
    );
    assert.ok(
      outcome.envelope.error.includes("trigger=iterator_drain_pending_tools"),
      "error must tag the trigger",
    );
    // The shared 'Silent agent-runner failure' log marker still fires so the
    // orchestrator's retry gate recognizes the failure as a stream race.
    assert.ok(
      outcome.logLines.some((l) =>
        l.startsWith(emitPhrase(MARKER_SILENT_AGENT_RUNNER_FAILURE)),
      ),
      "log line must lead with the marker-sourced silent-failure phrase",
    );
  });
});
