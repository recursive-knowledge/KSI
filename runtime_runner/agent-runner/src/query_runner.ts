/**
 * The default (claude-code) scheduled-task query loop.
 *
 * {@link runQuery} streams a single prompt through the Claude Agent SDK,
 * accumulates token usage + a tool trace, and emits exactly one result
 * envelope via writeOutput. The two large decision blocks — what envelope to
 * emit on the SDK `result` event, and what to emit when the stream ends
 * WITHOUT a result event — are factored into {@link buildScheduledResultOutcome}
 * and {@link buildPostLoopOutcome} so the loop body stays readable and each
 * decision is independently testable. Both helpers are side-effect-free with
 * respect to stdout: they return an `envelope` plus `logLines` that the caller
 * forwards to writeOutput / log.
 */
import fs from 'fs';
import path from 'path';
import { McpServerConfig, query } from '@anthropic-ai/claude-agent-sdk';
import { extractAssistantText, extractStructuredForumText } from './extract.js';
import { buildSystemPromptAppend } from './prompt-utils.js';
import { ContainerInput, ContainerOutput } from './shared_types.js';
import { emitPhrase, MARKER_SDK_QUERY_ITERATOR_THREW } from './retryable_markers.js';
import {
  ANTHROPIC_PROVIDER,
  buildSilentDiagnostic,
} from './adapter_safety.js';
import { CONTAINER_WORKSPACE_ROOT } from './runner_constants.js';
import { log } from './runner_log.js';
import { writeOutput } from './output.js';
import { MessageStream } from './message_stream.js';
import { extractUsageFromSdkMessage, hasAnyUsageDelta } from './usage.js';
import {
  recordAssistantToolUses,
  backfillToolResults,
} from './tool_trace.js';
import { resolveTurnBudgets, buildToolPolicy, buildMcpServerConfig } from './query_config.js';
import { createPreCompactHook, createSanitizeBashHook } from './hooks.js';
import { runPhase1Reflection } from './phase1_reflection.js';
import { runPolyglotTestFeedback } from './polyglot_test_feedback.js';
import { maybeRecoverFromEmptyScheduledOutcome } from './session_recovery.js';

/** Iterator-error shape captured from a thrown SDK stream (no values). */
interface IteratorError {
  message: string;
  name: string;
  stack?: string;
  cause?: unknown;
}

/** Final per-turn + result-event usage accumulators read after the loop. */
interface UsageTotals {
  perTurnInputTokens: number;
  perTurnOutputTokens: number;
  perTurnCacheCreationTokens: number;
  perTurnCacheReadTokens: number;
  resultInputTokens: number;
  resultOutputTokens: number;
  resultCacheCreationTokens: number;
  resultCacheReadTokens: number;
}

/**
 * Decide and build the envelope to emit on the SDK `result` event for a
 * scheduled task. Mirrors the original inline branching exactly:
 *   1. empty result event (no text, no tokens, no tool trace) → shared recovery
 *   2. ARC without arc_submit_trial → error (refuse unsubmitted success)
 *   3. forum task with pending tool calls → error (incomplete tool loop)
 *   4. otherwise → success, optionally with a phase-1 reflection
 *
 * Pure with respect to stdout: returns the envelope plus the log lines that
 * would have been emitted, in order, BEFORE writeOutput.
 */
async function buildScheduledResultOutcome(args: {
  resultCount: number;
  messageCount: number;
  effectiveResult: string | null;
  // Raw SDK result text, logged in the `Result #N` preview line. Distinct
  // from `effectiveResult` (which prefers structured forum text) so the log
  // mirrors the original inline behavior exactly.
  rawResultText: string | null;
  lastAssistantFallback: string | null;
  usage: UsageTotals;
  toolTrace: Array<Record<string, unknown>>;
  pendingToolCallsById: Map<string, Record<string, unknown>>;
  newSessionId?: string;
  lastAssistantUuid?: string;
  selectedModel: string | undefined;
  sdkEnv: Record<string, string | undefined>;
  mcpServerConfig: Record<string, unknown>;
  iteratorError: IteratorError | null;
  taskSource: string;
  isForumTask: boolean;
  containerInput: ContainerInput;
  resultSubtype: string | undefined;
}): Promise<{ envelope: ContainerOutput; logLines: string[] }> {
  const {
    resultCount, messageCount, effectiveResult, rawResultText, lastAssistantFallback, usage,
    toolTrace, pendingToolCallsById, newSessionId, lastAssistantUuid,
    selectedModel, sdkEnv, mcpServerConfig, iteratorError, taskSource,
    isForumTask, containerInput, resultSubtype,
  } = args;
  const logLines: string[] = [];

  logLines.push(
    `Result #${resultCount}: subtype=${resultSubtype}${rawResultText ? ` text=${rawResultText.slice(0, 200)}` : ''}`,
  );

  // Prefer the SDK's result-event aggregate (authoritative for the turn);
  // fall back to the per-turn accumulator if the result event shipped
  // without a usage block (degenerate edge case — some SDK variants).
  const finalInput = usage.resultInputTokens || usage.perTurnInputTokens;
  const finalOutput = usage.resultOutputTokens || usage.perTurnOutputTokens;
  const finalCacheCreate = usage.resultCacheCreationTokens || usage.perTurnCacheCreationTokens;
  const finalCacheRead = usage.resultCacheReadTokens || usage.perTurnCacheReadTokens;
  const resultEventHasTokens = hasAnyUsageDelta(
    usage.resultInputTokens,
    usage.resultOutputTokens,
    usage.resultCacheCreationTokens,
    usage.resultCacheReadTokens,
  );
  const tokensMissing = !hasAnyUsageDelta(finalInput, finalOutput, finalCacheCreate, finalCacheRead);
  if (tokensMissing) {
    logLines.push('WARN: result event had zero token usage and no per-turn usage was seen');
  }

  const tokensSource = resultEventHasTokens
    ? 'result_event'
    : (tokensMissing ? 'unavailable' : 'per_turn_sum');

  // 2026-04-20 audit v3 fix. Scheduled/bench tasks: if the SDK's `result`
  // message arrives EMPTY — no usable text AND zero tokens AND nothing in
  // the tool trace — route through the shared silent-exit recovery helper
  // instead of unconditionally emitting status='success' with zeros. The
  // shared helper either recovers to `recovered_from_session` or emits a
  // diagnostic `status='error'` envelope — never masks a real failure as
  // success. This comment intentionally mentions "empty result event" so
  // container-rebuild verification can grep for it in the baked bundle.
  // (tsc --outDir compiles this file to /tmp/dist/query_runner.js — the
  // string moved out of index.js when runQuery was split into this module.)
  const resultIsEmpty = !effectiveResult || !effectiveResult.trim();
  const handledAsEmptyRecovery =
    resultIsEmpty
    && tokensMissing
    && toolTrace.length === 0;
  if (handledAsEmptyRecovery) {
    const outcome = maybeRecoverFromEmptyScheduledOutcome({
      messageCount,
      resultCount,
      lastAssistantFallback,
      perTurnInputTokens: usage.perTurnInputTokens,
      perTurnOutputTokens: usage.perTurnOutputTokens,
      perTurnCacheCreationTokens: usage.perTurnCacheCreationTokens,
      perTurnCacheReadTokens: usage.perTurnCacheReadTokens,
      resultInputTokens: usage.resultInputTokens,
      resultOutputTokens: usage.resultOutputTokens,
      resultCacheCreationTokens: usage.resultCacheCreationTokens,
      resultCacheReadTokens: usage.resultCacheReadTokens,
      toolTrace,
      newSessionId,
      sdkEnv,
      mcpServerConfig,
      iteratorError,
      trigger: 'empty_result_event',
    });
    for (const line of outcome.logLines) logLines.push(line);
    return { envelope: outcome.envelope, logLines };
  }

  // ARC runs natively now (no arc_submit_trial tool call): its success signal
  // is a workspace prediction file that the host-side
  // runtime_runner/src/main.ts synthesises into the trace after the container
  // exits, so there is no in-trace arc_submit_trial guard here.

  if (isForumTask && pendingToolCallsById.size > 0) {
    const error = `Scheduled forum task ended with pending tool call(s): ${pendingToolCallsById.size}; refusing fallback success.`;
    logLines.push(`ERROR: ${error}`);
    return {
      envelope: {
        status: 'error',
        result: null,
        newSessionId,
        toolTrace: toolTrace.slice(-1000),
        input_tokens: finalInput,
        output_tokens: finalOutput,
        cache_creation_input_tokens: finalCacheCreate,
        cache_read_input_tokens: finalCacheRead,
        tokens_source: tokensSource,
        error,
      },
      logLines,
    };
  }

  // Phase-1 reflection (Path a). When the feature flag is on and we are on
  // the scheduled-task success path, write a barrier sentinel for the host's
  // BarrierWatcher, wait for the eval result, then run ONE follow-up SDK turn
  // to capture the agent's structured 3-5 sentence reflection. All failure
  // modes degrade gracefully: we ALWAYS emit the success envelope;
  // phase1_reflection is just optionally populated.
  let phase1Text: string | undefined;
  let phase1Meta: ContainerOutput['phase1_reflection_meta'];
  let phase1TokenUsage: ContainerOutput['phase1_reflection_token_usage'];
  const phase1Cfg = containerInput.phase1Reflection;
  if (
    phase1Cfg
    && phase1Cfg.enabled
    && effectiveResult
  ) {
    const outcome = await runPhase1Reflection({
      workspaceDir: CONTAINER_WORKSPACE_ROOT,
      agentId: phase1Cfg.agentId || 'agent',
      taskId: containerInput.memoryMcp?.taskId
        || containerInput.arcTools?.taskId
        || '',
      modelOutput: effectiveResult,
      resumeSessionId: newSessionId,
      resumeAt: lastAssistantUuid,
      selectedModel,
      sdkEnv,
      pollTimeoutMs: phase1Cfg.evalResultPollTimeoutMs,
      logger: log,
    });
    if (outcome.captured && outcome.text) {
      phase1Text = outcome.text;
    }
    phase1Meta = {
      enabled: true,
      captured: outcome.captured,
      note: outcome.note,
      elapsed_ms: outcome.elapsedMs,
    };
    // Surface reflection-turn tokens whether or not the reflection text was
    // captured — even a silent-exit on the follow-up turn can have consumed
    // input tokens, and dropping them produces an artificial cache-cost
    // mismatch. Only include when the SDK reported any usage.
    if (
      outcome.tokenUsage
      && (
        outcome.tokenUsage.input_tokens
        || outcome.tokenUsage.output_tokens
        || outcome.tokenUsage.cache_creation_input_tokens
        || outcome.tokenUsage.cache_read_input_tokens
      )
    ) {
      phase1TokenUsage = outcome.tokenUsage;
    }
  } else if (phase1Cfg && phase1Cfg.enabled) {
    phase1Meta = {
      enabled: true,
      captured: false,
      note: 'phase1: skipped (no effectiveResult to reflect on)',
    };
  }

  // Polyglot test-feedback retry loop (Aider protocol). When the feature flag
  // is on and we are on the scheduled-task success path, write a barrier
  // sentinel with the agent's live model output, wait for the host to run
  // the real evaluator, and — on failure — run resumed SDK turns (tools
  // re-enabled) feeding back capped test-runner output. Unlike phase1
  // reflection, this DOES change what gets returned as `result`: the last
  // round's on-disk edits are what gets graded.
  //
  // Finding #9: nothing prevents phase1_reflection and this feature from
  // both being eligible for the same polyglot task, and this block reuses
  // the SAME `newSessionId`/`lastAssistantUuid` checkpoint the phase1
  // block above resumed from (not an updated one — `runPhase1Reflection`'s
  // return type carries no session/checkpoint field to update them with).
  // This is safe by construction: phase1's reflection turn runs with
  // `allowedTools: []`/`mcpServers: {}` (no filesystem access), so it
  // cannot have mutated the workspace between the two resumes — there is
  // no edit to lose by forking both turns from the same checkpoint. If
  // phase1_reflection ever gains tool access, this assumption breaks and
  // this block would need the reflection turn's own updated checkpoint.
  let polyglotTFResult = effectiveResult;
  let polyglotTFMeta: ContainerOutput['polyglot_test_feedback_meta'];
  let polyglotTFTokenUsage: ContainerOutput['polyglot_test_feedback_token_usage'];
  const polyglotTFCfg = containerInput.polyglotTestFeedback;
  if (polyglotTFCfg && polyglotTFCfg.enabled && effectiveResult && polyglotTFCfg.triesRemaining > 1) {
    const outcome = await runPolyglotTestFeedback({
      workspaceDir: CONTAINER_WORKSPACE_ROOT,
      agentId: polyglotTFCfg.agentId,
      fileList: polyglotTFCfg.fileList,
      modelOutput: effectiveResult,
      triesRemaining: polyglotTFCfg.triesRemaining,
      maxLines: polyglotTFCfg.maxLines,
      resumeSessionId: newSessionId,
      resumeAt: lastAssistantUuid,
      selectedModel,
      sdkEnv,
      allowedTools: polyglotTFCfg.allowedTools,
      mcpServers: polyglotTFCfg.mcpServers as Record<string, McpServerConfig>,
      maxTurnsPerRound: polyglotTFCfg.maxTurnsPerRound,
      pollTimeoutMs: polyglotTFCfg.evalResultPollTimeoutMs,
      logger: log,
    });
    polyglotTFResult = outcome.finalResult;
    polyglotTFMeta = {
      enabled: true,
      rounds_used: outcome.roundsUsed,
      attempt_1_eval_summary: outcome.attempt1EvalSummary,
      captured: outcome.captured,
      note: outcome.note,
      final_eval_matches_output: outcome.finalEvalMatchesOutput,
    };
    if (
      outcome.tokenUsage.input_tokens || outcome.tokenUsage.output_tokens
      || outcome.tokenUsage.cache_creation_input_tokens || outcome.tokenUsage.cache_read_input_tokens
    ) {
      polyglotTFTokenUsage = outcome.tokenUsage;
    }
  } else if (polyglotTFCfg && polyglotTFCfg.enabled && polyglotTFCfg.triesRemaining > 1) {
    polyglotTFMeta = {
      enabled: true,
      rounds_used: 0,
      attempt_1_eval_summary: null,
      captured: false,
      note: 'polyglot_test_feedback: skipped (no effectiveResult to retry on)',
      final_eval_matches_output: false,
    };
  }

  return {
    envelope: {
      status: 'success',
      result: polyglotTFResult,
      newSessionId,
      toolTrace: toolTrace.slice(-1000),
      input_tokens: finalInput,
      output_tokens: finalOutput,
      cache_creation_input_tokens: finalCacheCreate,
      cache_read_input_tokens: finalCacheRead,
      tokens_source: tokensSource,
      ...(phase1Text ? { phase1_reflection: phase1Text } : {}),
      ...(phase1Meta ? { phase1_reflection_meta: phase1Meta } : {}),
      ...(phase1TokenUsage ? { phase1_reflection_token_usage: phase1TokenUsage } : {}),
      ...(polyglotTFMeta ? { polyglot_test_feedback_meta: polyglotTFMeta } : {}),
      ...(polyglotTFTokenUsage ? { polyglot_test_feedback_token_usage: polyglotTFTokenUsage } : {}),
    },
    logLines,
  };
}

/**
 * Decide and build the envelope to emit when the SDK stream ENDED without a
 * `result` event (or after the iterator threw). Mirrors the original inline
 * post-loop branching exactly. Returns `envelope: null` when no envelope
 * should be written (the normal break-after-result path already wrote one).
 */
function buildPostLoopOutcome(args: {
  resultCount: number;
  messageCount: number;
  lastAssistantFallback: string | null;
  bestStructuredForumText: string | null;
  usage: UsageTotals;
  toolTrace: Array<Record<string, unknown>>;
  pendingToolCallsById: Map<string, Record<string, unknown>>;
  newSessionId?: string;
  sdkEnv: Record<string, string | undefined>;
  mcpServerConfig: Record<string, unknown>;
  iteratorError: IteratorError | null;
  taskSource: string;
  isForumTask: boolean;
  containerInput: ContainerInput;
}): { envelope: ContainerOutput | null; logLines: string[] } {
  const {
    resultCount, messageCount, lastAssistantFallback, bestStructuredForumText,
    usage, toolTrace, pendingToolCallsById, newSessionId, sdkEnv,
    mcpServerConfig, iteratorError, taskSource, isForumTask, containerInput,
  } = args;
  const logLines: string[] = [];

  // Scheduled/bench safety fallback: if SDK stream ended without an explicit
  // `result` message, emit the latest assistant text so callers still get
  // output. Token usage comes from the per-turn accumulator since there is
  // no final SDK aggregate to pull from.
  //
  // ARC is stricter: no scheduled ARC run may be counted as success unless it
  // actually reached arc_submit_trial and has no dangling tool calls.
  if (resultCount === 0 && lastAssistantFallback) {
    const fallbackMissing = (
      usage.perTurnInputTokens + usage.perTurnOutputTokens
      + usage.perTurnCacheCreationTokens + usage.perTurnCacheReadTokens
    ) === 0;
    const hasPendingToolCalls = pendingToolCallsById.size > 0;
    // ARC runs natively now: it succeeds via a workspace prediction file that
    // the host-side runtime synthesises into the trace post-exit, so there is
    // no in-trace arc_submit_trial requirement here.
    if (fallbackMissing) {
      logLines.push('WARN: scheduled task ended without result event and no per-turn usage accumulated');
    } else {
      logLines.push(
        `Scheduled fallback emitting per-turn tokens: input=${usage.perTurnInputTokens} `
        + `output=${usage.perTurnOutputTokens} cache_read=${usage.perTurnCacheReadTokens} `
        + `cache_create=${usage.perTurnCacheCreationTokens}`,
      );
    }
    // Strict-protocol tasks (ARC + forum) require complete tool loops:
    //   - ARC must reach arc_submit_trial; otherwise scoring has no submission.
    //   - Forum tasks must complete forum tool calls; otherwise the bus has
    //     no signal to drain.
    // For these, refusing the fallback (and surfacing a diagnostic) is
    // correct.
    //
    // Non-strict tasks (polyglot, swebench_pro) DON'T need a specific tool
    // sequence — the eval pipeline reads workspace state via `git diff` plus
    // a test suite. Try session-log recovery first; only fall back to a
    // diagnostic error if the log isn't usable.
    const isStrictProtocol = isForumTask || taskSource === 'arc';
    if (hasPendingToolCalls && isStrictProtocol) {
      const reason = `pending tool call(s) never returned: ${pendingToolCallsById.size}`;
      const protocol = isForumTask ? 'forum tool loop' : 'ARC tool loop';
      const error = `Scheduled task ended before a complete ${protocol} (${reason}); refusing fallback success.`;
      logLines.push(`ERROR: ${error}`);
      return {
        envelope: {
          status: 'error',
          result: null,
          newSessionId,
          toolTrace: toolTrace.slice(-1000),
          input_tokens: usage.perTurnInputTokens,
          output_tokens: usage.perTurnOutputTokens,
          cache_creation_input_tokens: usage.perTurnCacheCreationTokens,
          cache_read_input_tokens: usage.perTurnCacheReadTokens,
          tokens_source: fallbackMissing ? 'unavailable' : 'per_turn_sum',
          error,
        },
        logLines,
      };
    } else if (hasPendingToolCalls) {
      // Non-strict task with iterator drain mid-conversation. Try session-log
      // recovery before declaring failure — see #525.
      logLines.push(
        `Iterator drained with ${pendingToolCallsById.size} pending tool call(s) ` +
        `on non-strict task (taskSource=${taskSource ?? 'none'}); attempting ` +
        `session-log recovery before falling back.`,
      );
      const outcome = maybeRecoverFromEmptyScheduledOutcome({
        messageCount,
        resultCount,
        lastAssistantFallback,
        perTurnInputTokens: usage.perTurnInputTokens,
        perTurnOutputTokens: usage.perTurnOutputTokens,
        perTurnCacheCreationTokens: usage.perTurnCacheCreationTokens,
        perTurnCacheReadTokens: usage.perTurnCacheReadTokens,
        resultInputTokens: usage.resultInputTokens,
        resultOutputTokens: usage.resultOutputTokens,
        resultCacheCreationTokens: usage.resultCacheCreationTokens,
        resultCacheReadTokens: usage.resultCacheReadTokens,
        toolTrace,
        newSessionId,
        sdkEnv,
        mcpServerConfig,
        iteratorError,
        trigger: 'iterator_drain_pending_tools',
      });
      for (const line of outcome.logLines) logLines.push(line);
      return { envelope: outcome.envelope, logLines };
    } else {
      return {
        envelope: {
          status: 'success',
          result: bestStructuredForumText || lastAssistantFallback,
          newSessionId,
          toolTrace: toolTrace.slice(-1000),
          input_tokens: usage.perTurnInputTokens,
          output_tokens: usage.perTurnOutputTokens,
          cache_creation_input_tokens: usage.perTurnCacheCreationTokens,
          cache_read_input_tokens: usage.perTurnCacheReadTokens,
          tokens_source: fallbackMissing ? 'unavailable' : 'per_turn_sum',
        },
        logLines,
      };
    }
  } else if (
    resultCount === 0 &&
    !lastAssistantFallback &&
    !hasAnyUsageDelta(
      usage.resultInputTokens,
      usage.resultOutputTokens,
      usage.resultCacheCreationTokens,
      usage.resultCacheReadTokens,
    ) &&
    !hasAnyUsageDelta(
      usage.perTurnInputTokens,
      usage.perTurnOutputTokens,
      usage.perTurnCacheCreationTokens,
      usage.perTurnCacheReadTokens,
    )
  ) {
    // Silent-exit branch. Delegates to the shared
    // `maybeRecoverFromEmptyScheduledOutcome` helper so this and the
    // empty-result-event branch produce identical envelopes. Two-layer
    // response: FIRST try to recover from the on-disk claude-code session
    // log; if recovery fails, fall through to the diagnostic status='error'
    // envelope with the full `buildSilentDiagnostic` snapshot.
    const outcome = maybeRecoverFromEmptyScheduledOutcome({
      messageCount,
      resultCount,
      lastAssistantFallback,
      perTurnInputTokens: usage.perTurnInputTokens,
      perTurnOutputTokens: usage.perTurnOutputTokens,
      perTurnCacheCreationTokens: usage.perTurnCacheCreationTokens,
      perTurnCacheReadTokens: usage.perTurnCacheReadTokens,
      resultInputTokens: usage.resultInputTokens,
      resultOutputTokens: usage.resultOutputTokens,
      resultCacheCreationTokens: usage.resultCacheCreationTokens,
      resultCacheReadTokens: usage.resultCacheReadTokens,
      toolTrace,
      newSessionId,
      sdkEnv,
      mcpServerConfig,
      iteratorError,
      trigger: 'silent_exit',
    });
    for (const line of outcome.logLines) logLines.push(line);
    return { envelope: outcome.envelope, logLines };
  } else if (iteratorError && resultCount === 0) {
    // Iterator threw without ever producing a `result` event, but per-turn
    // tokens accumulated (otherwise the silent-exit branch above handles it).
    // Emit a status=error envelope carrying the iterator error + diagnostic so
    // the next repro has the crash context.
    //
    // The `resultCount === 0` guard is load-bearing (#945): when a result WAS
    // produced, the loop already `break`s and `buildScheduledResultOutcome`
    // already wrote a success envelope. But `break` runs the async iterator's
    // `.return()` cleanup, which can throw and set `iteratorError` AFTER the
    // success write. Without this guard, we would emit a follow-up
    // status=error envelope; because the host's streaming parser keeps the
    // LAST parsed marker pair (runtime_runner/src/container_output.ts), that
    // late error would clobber the otherwise-successful run. With the guard, a
    // post-result iterator error falls through to `envelope: null` and the
    // success stands.
    const diag = buildSilentDiagnostic({
      messageCount,
      resultCount,
      lastAssistantFallback,
      perTurnInputTokens: usage.perTurnInputTokens,
      perTurnOutputTokens: usage.perTurnOutputTokens,
      resultInputTokens: usage.resultInputTokens,
      resultOutputTokens: usage.resultOutputTokens,
      sdkEnv,
      provider: ANTHROPIC_PROVIDER,
      mcpServerNames: mcpServerConfig,
      iteratorError: (() => {
        const e = new Error(iteratorError.message);
        e.name = iteratorError.name;
        if (iteratorError.stack) e.stack = iteratorError.stack;
        if (iteratorError.cause !== undefined) (e as { cause?: unknown }).cause = iteratorError.cause;
        return e;
      })(),
    });
    logLines.push(
      `SDK iterator threw after partial output (messages=${diag.messageCount}, ` +
      `results=${diag.resultCount}). Emitting terminal status=error envelope.`,
    );
    logLines.push(`Iterator-threw diagnostic: ${JSON.stringify(diag)}`);
    const resultEventHasTokens = hasAnyUsageDelta(
      usage.resultInputTokens,
      usage.resultOutputTokens,
      usage.resultCacheCreationTokens,
      usage.resultCacheReadTokens,
    );
    const perTurnHasTokens = hasAnyUsageDelta(
      usage.perTurnInputTokens,
      usage.perTurnOutputTokens,
      usage.perTurnCacheCreationTokens,
      usage.perTurnCacheReadTokens,
    );
    return {
      envelope: {
        status: 'error',
        result: null,
        newSessionId,
        toolTrace: toolTrace.slice(-1000),
        input_tokens: usage.resultInputTokens || usage.perTurnInputTokens,
        output_tokens: usage.resultOutputTokens || usage.perTurnOutputTokens,
        cache_creation_input_tokens: usage.resultCacheCreationTokens || usage.perTurnCacheCreationTokens,
        cache_read_input_tokens: usage.resultCacheReadTokens || usage.perTurnCacheReadTokens,
        tokens_source: resultEventHasTokens
          ? 'result_event'
          : (perTurnHasTokens ? 'per_turn_sum' : 'unavailable'),
        // Marker prefix from runtime_runner/shared/retryable_markers.json (#648):
        // this is the envelope `error` text the orchestrator classifies.
        error:
          `${emitPhrase(MARKER_SDK_QUERY_ITERATOR_THREW)} mid-stream: ${iteratorError.name}: ${iteratorError.message.slice(0, 240)}. ` +
          `diagnostic=${JSON.stringify(diag)}`,
      },
      logLines,
    };
  }
  return { envelope: null, logLines };
}

/**
 * Run a single query and stream results via writeOutput.
 * Uses MessageStream (AsyncIterable) to keep isSingleUserTurn=false,
 * allowing agent teams subagents to run to completion.
 */
export async function runQuery(
  prompt: string,
  sessionId: string | undefined,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
): Promise<{ newSessionId?: string; lastAssistantUuid?: string }> {
  // Always feed Claude through an async stream. Passing a plain string prompt
  // makes scheduled benchmark jobs a single user turn in the Claude Agent SDK:
  // Haiku can emit the first tool call and then the iterator drains before tool
  // results / MCP calls continue. Close the async iterable immediately after
  // the initial prompt: the SDK still treats the session as non-single-turn,
  // but its streamInput() loop can finish and wait for the final result
  // instead of hanging on an open input stream.
  const stream = new MessageStream();
  stream.push(prompt);
  stream.end();

  let newSessionId: string | undefined;
  let lastAssistantUuid: string | undefined;
  let messageCount = 0;
  let resultCount = 0;
  let lastAssistantFallback: string | null = null;
  let bestStructuredForumText: string | null = null;
  const forumTaskSources = new Set([
    'per_task_forum',
    'cross_task_forum',
  ]);
  const isForumTask = forumTaskSources.has(
    (containerInput.memoryMcp?.taskSource || '').toLowerCase(),
  );
  const toolTrace: Array<Record<string, unknown>> = [];
  // Map tool_use_id -> tool_call trace entry so we can backfill tool_output
  // when the matching tool_result arrives on a subsequent user message. The
  // SDK emits tool_call (no output) and tool_result as separate messages;
  // without correlation, downstream scorers (e.g., ARC fast-path) never see
  // the output and fall back to fragile text parsing.
  const pendingToolCallsById = new Map<string, Record<string, unknown>>();
  const TOOL_OUTPUT_MAX_CHARS = 64 * 1024;
  // Per-turn accumulator: sums usage from every assistant/user message's
  // nested `message.usage` block. This is the source of truth when the stream
  // is cut off (max-messages ceiling, timeout, cancellation) before emitting
  // a final `result` event.
  let perTurnInputTokens = 0;
  let perTurnOutputTokens = 0;
  let perTurnCacheCreationTokens = 0;
  let perTurnCacheReadTokens = 0;
  // Result-event aggregator: the SDK's `result` message carries a top-level
  // `usage` that is the authoritative session total for that turn. We track
  // it separately and prefer it over the per-turn sum when available.
  let resultInputTokens = 0;
  let resultOutputTokens = 0;
  let resultCacheCreationTokens = 0;
  let resultCacheReadTokens = 0;

  // Load global CLAUDE.md as additional system context (shared across all groups)
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  let globalClaudeMd: string | undefined;
  if (fs.existsSync(globalClaudeMdPath)) {
    globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }
  const seedContextPath = `${CONTAINER_WORKSPACE_ROOT}/.seed_context`;
  let seedContext: string | undefined;
  if (fs.existsSync(seedContextPath)) {
    seedContext = fs.readFileSync(seedContextPath, 'utf-8');
  }
  const systemPromptAppend = buildSystemPromptAppend(globalClaudeMd, seedContext);

  // Discover additional directories mounted at /workspace/extra/*
  // These are passed to the SDK so their CLAUDE.md files are loaded automatically
  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) {
        extraDirs.push(fullPath);
      }
    }
  }
  if (extraDirs.length > 0) {
    log(`Additional directories: ${extraDirs.join(', ')}`);
  }

  const selectedModel = sdkEnv.MODEL;
  // taskSource falls back to containerInput.arcTools?.taskSource so that
  // `--no-memory` runs (where memoryMcp is undefined) still get the ARC
  // offline-web-tool gate and the ARC MCP registration below.
  const taskSource = (
    containerInput.memoryMcp?.taskSource
    || containerInput.arcTools?.taskSource
    || ''
  ).toLowerCase();
  const { scheduledMaxTurns, scheduledMaxMessages } = resolveTurnBudgets(
    taskSource,
    sdkEnv,
  );
  if (selectedModel) {
    log(`Using model: ${selectedModel}`);
  }

  // Build allowed tools list — conditionally include MCP tools.
  // Scheduled swarm tasks keep coding-native tools enabled, but explicitly
  // exclude team/orchestration tools so the provider runtime does not invent a
  // second coordination layer on top of the KSI protocol.
  //
  // ARC is a sealed offline puzzle benchmark — web tools must be disabled so
  // agents cannot google published ARC solutions and contaminate the eval.
  // Scheduled ARC jobs also disable Claude-native file/shell tools.
  const {
    isArcWithoutMcp,
    isScheduledMcpProtocolTask,
    allowedToolsList,
    disallowedToolsList,
  } = buildToolPolicy(containerInput, taskSource, isForumTask, sdkEnv);

  // Build MCP server config (memory, ARC). This also pushes the
  // registered servers' wildcard/allowlist tools onto allowedToolsList.
  const mcpServerConfig = buildMcpServerConfig(
    containerInput,
    sdkEnv,
    taskSource,
    allowedToolsList,
  );

  // Capture SDK iterator errors so the silent-exit branch below can
  // distinguish "iterator drained cleanly with zero messages" (auth/MCP
  // startup failure) from "iterator threw mid-stream" (provider error the
  // subprocess swallowed).
  let iteratorError: IteratorError | null = null;
  try {
  for await (const message of query({
    prompt: stream ?? prompt,
    options: {
      model: selectedModel || undefined,
      cwd: CONTAINER_WORKSPACE_ROOT,
      additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
      resume: sessionId,
      systemPrompt: systemPromptAppend
        ? { type: 'preset' as const, preset: 'claude_code' as const, append: systemPromptAppend }
        : undefined,
      allowedTools: allowedToolsList,
      disallowedTools: disallowedToolsList.length > 0 ? disallowedToolsList : undefined,
      tools: (isScheduledMcpProtocolTask && !isArcWithoutMcp) ? [] : undefined,
      env: sdkEnv,
      maxTurns: scheduledMaxTurns,
      permissionMode: 'bypassPermissions',
      allowDangerouslySkipPermissions: true,
      settingSources: ['project', 'user'],
      mcpServers: mcpServerConfig,
      hooks: {
        PreCompact: [{ hooks: [createPreCompactHook(containerInput.assistantName)] }],
        PreToolUse: [{ matcher: 'Bash', hooks: [createSanitizeBashHook()] }],
      },
      stderr: (chunk: string) => {
        process.stderr.write(`[claude-cli] ${chunk}`);
      },
    }
  })) {
    messageCount++;
    const msgType = message.type === 'system' ? `system/${(message as { subtype?: string }).subtype}` : message.type;
    log(`[msg #${messageCount}] type=${msgType}`);
    const subtype = (message as { subtype?: string }).subtype;
    const record: Record<string, unknown> = {
      idx: messageCount,
      type: message.type,
      subtype: subtype || null,
      ts: new Date().toISOString(),
    };
    const maybeToolName = (message as { tool_name?: string }).tool_name;
    if (maybeToolName) record.tool_name = maybeToolName;
    const maybeToolUseId = (message as { tool_use_id?: string }).tool_use_id;
    if (maybeToolUseId) record.tool_use_id = maybeToolUseId;
    const maybeResult = (message as { result?: string }).result;
    if (typeof maybeResult === 'string' && maybeResult.trim()) {
      record.result_excerpt = maybeResult.slice(0, 240);
    }

    // Extract tool_use blocks from assistant messages and backfill tool_output
    // onto prior tool_call entries when user tool_result blocks arrive.
    recordAssistantToolUses(message, messageCount, record, toolTrace, pendingToolCallsById);
    backfillToolResults(message, pendingToolCallsById, TOOL_OUTPUT_MAX_CHARS);

    toolTrace.push(record);
    if (toolTrace.length > 2000) toolTrace.shift();

    // Accumulate token usage from SDK messages. Route the delta to either the
    // per-turn accumulator or the result aggregator depending on message type:
    //   - assistant/user messages  → per-turn sum (nested under .message.usage)
    //   - result messages          → result aggregate (top-level .usage)
    const usageDelta = extractUsageFromSdkMessage(message);
    if (
      usageDelta.input_tokens
      || usageDelta.output_tokens
      || usageDelta.cache_creation_input_tokens
      || usageDelta.cache_read_input_tokens
    ) {
      if (message.type === 'result') {
        resultInputTokens += usageDelta.input_tokens;
        resultOutputTokens += usageDelta.output_tokens;
        resultCacheCreationTokens += usageDelta.cache_creation_input_tokens;
        resultCacheReadTokens += usageDelta.cache_read_input_tokens;
      } else {
        perTurnInputTokens += usageDelta.input_tokens;
        perTurnOutputTokens += usageDelta.output_tokens;
        perTurnCacheCreationTokens += usageDelta.cache_creation_input_tokens;
        perTurnCacheReadTokens += usageDelta.cache_read_input_tokens;
      }
    }

    if (message.type === 'assistant' && 'uuid' in message) {
      lastAssistantUuid = (message as { uuid: string }).uuid;
    }

    if (isForumTask && message.type === 'assistant') {
      const structuredForumText = extractStructuredForumText(message);
      if (structuredForumText) {
        bestStructuredForumText = structuredForumText;
      }
    }

    if (message.type === 'assistant') {
      const assistantText = extractAssistantText(message);
      lastAssistantFallback = assistantText || (() => {
        try {
          return JSON.stringify(message).slice(0, 50000);
        } catch {
          return '[assistant_message_without_text]';
        }
      })();
    }

    if (message.type === 'system' && message.subtype === 'init') {
      newSessionId = message.session_id;
      log(`Session initialized: ${newSessionId}`);
    }

    if (message.type === 'system' && (message as { subtype?: string }).subtype === 'task_notification') {
      const tn = message as unknown as { task_id: string; status: string; summary: string };
      log(`Task notification: task=${tn.task_id} status=${tn.status} summary=${tn.summary}`);
    }

    if (message.type === 'result') {
      resultCount++;
      const textResult = 'result' in message ? (message as { result?: string }).result : null;
      const effectiveResult = bestStructuredForumText || textResult || null;
      const outcome = await buildScheduledResultOutcome({
        resultCount,
        messageCount,
        effectiveResult,
        rawResultText: textResult ?? null,
        lastAssistantFallback,
        usage: {
          perTurnInputTokens,
          perTurnOutputTokens,
          perTurnCacheCreationTokens,
          perTurnCacheReadTokens,
          resultInputTokens,
          resultOutputTokens,
          resultCacheCreationTokens,
          resultCacheReadTokens,
        },
        toolTrace,
        pendingToolCallsById,
        newSessionId,
        lastAssistantUuid,
        selectedModel,
        sdkEnv,
        mcpServerConfig,
        iteratorError,
        taskSource,
        isForumTask,
        containerInput,
        resultSubtype: message.subtype,
      });
      for (const line of outcome.logLines) log(line);
      writeOutput(outcome.envelope);
      // Scheduled/bench tasks are single-shot: break immediately after
      // emitting the envelope (success, recovered, or diagnostic error)
      // so we don't block waiting for the SDK generator to complete (the
      // CLI subprocess may linger after producing its result).
      log('Scheduled task: result received, breaking out of query loop');
      break;
    }

    if (
      scheduledMaxMessages != null &&
      messageCount >= scheduledMaxMessages
    ) {
      log(
        `Scheduled task: reached message ceiling (${scheduledMaxMessages}), stopping query loop with fallback output`,
      );
      break;
    }
  }
  } catch (err) {
    // SDK iterator threw. Capture error shape (no values) so the silent-exit
    // branch below can emit a diagnostic envelope that distinguishes this
    // "iterator-threw" path from a clean drain. We DO NOT swallow the error
    // -- control falls through to the fallback branches.
    iteratorError = {
      message: err instanceof Error ? err.message : String(err),
      name: err instanceof Error ? err.name : 'Error',
      stack: err instanceof Error && err.stack ? err.stack : undefined,
      cause: err instanceof Error ? (err as { cause?: unknown }).cause : undefined,
    };
    log(
      // Marker prefix from runtime_runner/shared/retryable_markers.json (#648)
      // so this diagnostic stays in lockstep with the classified envelopes.
      `${emitPhrase(MARKER_SDK_QUERY_ITERATOR_THREW)}: name=${iteratorError.name} ` +
      `message=${iteratorError.message.slice(0, 240)}`,
    );
    if (iteratorError.stack) {
      log(`SDK iterator stack (head):\n${iteratorError.stack.split('\n').slice(0, 8).join('\n')}`);
    }
    if (iteratorError.cause !== undefined && iteratorError.cause !== null) {
      try {
        const causeStr =
          iteratorError.cause instanceof Error
            ? `${iteratorError.cause.name}: ${iteratorError.cause.message}`
            : JSON.stringify(iteratorError.cause).slice(0, 400);
        log(`SDK iterator cause: ${causeStr}`);
      } catch {
        log(`SDK iterator cause: ${String(iteratorError.cause).slice(0, 400)}`);
      }
    }
  }

  const outcome = buildPostLoopOutcome({
    resultCount,
    messageCount,
    lastAssistantFallback,
    bestStructuredForumText,
    usage: {
      perTurnInputTokens,
      perTurnOutputTokens,
      perTurnCacheCreationTokens,
      perTurnCacheReadTokens,
      resultInputTokens,
      resultOutputTokens,
      resultCacheCreationTokens,
      resultCacheReadTokens,
    },
    toolTrace,
    pendingToolCallsById,
    newSessionId,
    sdkEnv,
    mcpServerConfig,
    iteratorError,
    taskSource,
    isForumTask,
    containerInput,
  });
  for (const line of outcome.logLines) log(line);
  if (outcome.envelope) writeOutput(outcome.envelope);

  log(`Query done. Messages: ${messageCount}, results: ${resultCount}, lastAssistantUuid: ${lastAssistantUuid || 'none'}`);
  return { newSessionId, lastAssistantUuid };
}
