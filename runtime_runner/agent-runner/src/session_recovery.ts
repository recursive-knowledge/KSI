/**
 * Last-resort recovery for the "silent exit" pattern: the SDK's async
 * `query(...)` iterator drains without yielding events to the Node wrapper,
 * but the underlying claude-code CLI subprocess DID run to completion inside
 * the container. These helpers reconstruct usable output + approximate token
 * counts from the on-disk session log, and assemble the shared envelopes used
 * by every silent-failure branch in the main query loop.
 *
 * Pure module: nothing here touches stdout/stderr. Callers forward the
 * returned `logLines` to their logger and the `envelope` to writeOutput —
 * which keeps the helpers trivially unit-testable.
 */
import fs from 'fs';
import path from 'path';
import { ContainerOutput } from './shared_types.js';
import { extractUsageFromSdkMessage } from './usage.js';
import { CONTAINER_CLAUDE_SESSIONS_ROOT } from './runner_constants.js';
import {
  ANTHROPIC_PROVIDER,
  buildSilentDiagnostic,
} from './adapter_safety.js';
import {
  emitPhrase,
  MARKER_SDK_EMPTY_RESULT_EVENT,
  MARKER_SDK_QUERY_ITERATOR_THREW,
  MARKER_SDK_QUERY_LOOP_DRAINED,
  MARKER_SILENT_AGENT_RUNNER_FAILURE,
} from './retryable_markers.js';

/**
 * Token usage shape used by the session-log recovery path. Same keys as
 * `UsageDelta` — duplicated here so the recovery helper stays self-contained.
 */
export interface SessionRecoveryResult {
  /** Last assistant message's text content, or null if the log had none. */
  result: string | null;
  /** Count of tool_use blocks seen in assistant messages in the log. */
  toolUseCount: number;
  inputTokens: number;
  outputTokens: number;
  cacheCreationTokens: number;
  cacheReadTokens: number;
  /** Path to the session log file that was used (for diagnostics). */
  sourcePath: string | null;
  /** Number of turns (assistant + user) parsed out of the log. */
  turnCount: number;
}

/**
 * Recover usable output + approximate token counts from the claude-agent-sdk's
 * on-disk session log. This is the last-resort path for the "silent exit"
 * pattern: the SDK's async `query(...)` iterator drains without yielding
 * events to the Node wrapper, but the underlying claude-code CLI subprocess
 * DID run to completion inside the container (15-30 turns, thousands of
 * tokens, real tool calls). Without this helper, the wrapper emits
 * `status='error'` with zeros and the host strips the on-disk session memory
 * on DB write, discarding the one artifact that proves work happened.
 *
 * Forensics example (vuls-86b60e, baseline Haiku sweep 2026-04):
 *   18 turns / 4494 tokens / 6 tool calls were visible in the session log
 *   even though the wrapper emitted 0 tokens and 0 tool calls.
 *
 * Scans `rootDir` (defaults to `/home/node/.claude/projects`) recursively for
 * `*.jsonl` files, picks the most-recently-modified one, and parses its turns.
 * Returns `null` when no log exists or the log has no assistant turns.
 */
export function recoverFromSessionLog(
  rootDir: string = CONTAINER_CLAUDE_SESSIONS_ROOT,
  readFileImpl: (p: string) => string = (p) => fs.readFileSync(p, 'utf-8'),
): SessionRecoveryResult | null {
  // 1) Locate session log files.
  let candidates: string[] = [];
  try {
    if (!fs.existsSync(rootDir)) return null;
    const walk = (dir: string): void => {
      let entries: fs.Dirent[];
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
  let raw: string;
  try {
    raw = readFileImpl(sourcePath);
  } catch {
    return null;
  }
  let turnCount = 0;
  let toolUseCount = 0;
  let lastAssistantText: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheCreationTokens = 0;
  let cacheReadTokens = 0;

  for (const line of raw.split('\n')) {
    if (!line.trim()) continue;
    let entry: Record<string, unknown>;
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
        ? entry.message as Record<string, unknown>
        : null;
      const content = inner && Array.isArray(inner.content) ? inner.content : [];
      const textParts: string[] = [];
      for (const block of content) {
        if (!block || typeof block !== 'object') continue;
        const b = block as Record<string, unknown>;
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

/**
 * Shared context captured by the two scheduled-task recovery call sites.
 * Both the "empty result event" (a `result` message with no text AND no
 * tokens AND no tool trace) and the "silent exit" (SDK iterator drained
 * without emitting any message at all) flow through
 * {@link maybeRecoverFromEmptyScheduledOutcome} so they emit identical
 * envelopes — either `recovered_from_session` (session log had usable
 * turns) or `error` with the full `buildSilentDiagnostic` snapshot (true
 * silent failure with nothing to recover).
 */
export interface ScheduledRecoveryContext {
  messageCount: number;
  resultCount: number;
  lastAssistantFallback: string | null;
  perTurnInputTokens: number;
  perTurnOutputTokens: number;
  perTurnCacheCreationTokens: number;
  perTurnCacheReadTokens: number;
  resultInputTokens: number;
  resultOutputTokens: number;
  resultCacheCreationTokens: number;
  resultCacheReadTokens: number;
  toolTrace: Array<Record<string, unknown>>;
  newSessionId?: string;
  sdkEnv: Record<string, string | undefined>;
  mcpServerConfig: Record<string, unknown>;
  iteratorError: {
    message: string;
    name: string;
    stack?: string;
    cause?: unknown;
  } | null;
  /**
   * Why the caller invoked recovery. Shows up in log lines and the error
   * envelope so the next repro can distinguish the silent patterns.
   *   - `empty_result_event`: a `result` SDK message WAS received but its
   *     payload was empty (no text, no tokens, no tool trace).
   *   - `silent_exit`: SDK iterator drained without yielding ANY message.
   *   - `iterator_drain_pending_tools`: SDK iterator drained mid-conversation
   *     with at least one assistant tool_use message but no following
   *     tool_result on the wire. See #525.
   */
  trigger: 'empty_result_event' | 'silent_exit' | 'iterator_drain_pending_tools';
}

/**
 * Outcome returned by {@link maybeRecoverFromEmptyScheduledOutcome}. The
 * caller should forward `envelope` to `writeOutput` and emit `logLines`
 * via the logger (split so the helper stays pure/testable — no stdout /
 * stderr side effects).
 */
export interface ScheduledRecoveryOutcome {
  envelope: ContainerOutput;
  logLines: string[];
  recovered: SessionRecoveryResult | null;
}

/**
 * Attempt session-log recovery, then emit either `recovered_from_session`
 * (with reconstructed tokens + result) or `error` (with the full
 * `buildSilentDiagnostic` envelope) — never `success` with zero tokens.
 *
 * This is the single code path for BOTH silent patterns on scheduled
 * (benchmark) tasks. Factoring it guarantees both branches produce
 * byte-identical envelopes for identical inputs; the only difference is the
 * `trigger` tag that appears in log lines and (for the error branch) in the
 * diagnostic summary.
 *
 * The helper is pure: it does not call writeOutput, log, or touch stdout /
 * stderr. The caller is responsible for forwarding `logLines` to its logger
 * and `envelope` to `writeOutput`.
 */
export function maybeRecoverFromEmptyScheduledOutcome(
  ctx: ScheduledRecoveryContext,
  recoverImpl: () => SessionRecoveryResult | null = () => recoverFromSessionLog(),
): ScheduledRecoveryOutcome {
  const logLines: string[] = [];
  let recovered: SessionRecoveryResult | null = null;
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
        const e = new Error(ctx.iteratorError!.message);
        e.name = ctx.iteratorError!.name;
        if (ctx.iteratorError!.stack) e.stack = ctx.iteratorError!.stack;
        if (ctx.iteratorError!.cause !== undefined) {
          (e as { cause?: unknown }).cause = ctx.iteratorError!.cause;
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
