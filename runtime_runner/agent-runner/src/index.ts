/**
 * KSI shared agent runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF, like before)
 *   IPC:   Pending messages written as JSON files to /workspace/ipc/input/
 *          ({type:"message", text:"..."}.json) are drained into the initial
 *          prompt. The host writes /workspace/ipc/input/_close as a
 *          best-effort shutdown signal after the first output marker.
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent teams result).
 *   Final marker after loop ends signals completion.
 *
 * This module is the thin entrypoint: it reads input, dispatches to the
 * appropriate provider adapter (OpenAI, Anthropic-direct forum, or the
 * default claude-code `runQuery` — which also serves ARC natively via
 * attempt files), and installs the terminal-diagnostic
 * process handlers. The per-adapter machinery lives in sibling modules
 * (query_runner, query_config, session_recovery, phase1_reflection, hooks,
 * tool_trace, usage, ...).
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { runAnthropicDirectForumQuery } from './anthropic_direct_forum.js';
import { runOpenAIQuery } from './openai.js';
import { ContainerInput } from './shared_types.js';
import { log } from './runner_log.js';
import { IPC_INPUT_DIR, IPC_INPUT_CLOSE_SENTINEL } from './runner_constants.js';
import { writeOutput, setOutputProtocolNonce } from './output.js';
import { readStdin, drainIpcInput } from './ipc.js';
import { runQuery } from './query_runner.js';
import {
  ANTHROPIC_PROVIDER,
  OPENAI_PROVIDER,
  buildSilentDiagnostic,
} from './adapter_safety.js';
import { ProxyAgent, setGlobalDispatcher } from 'undici';

// ── Back-compat re-exports ───────────────────────────────────────────────
// index.ts has historically been the public surface for these symbols (the
// JS regression tests document them as "exported from index.ts"). They now
// live in focused modules; re-export so any external import keeps resolving.
export type { UsageDelta } from './usage.js';
export { extractUsageFromSdkMessage } from './usage.js';
export { resolveClaudeMaxTurns } from './query_config.js';
export type {
  SessionRecoveryResult,
  ScheduledRecoveryContext,
  ScheduledRecoveryOutcome,
} from './session_recovery.js';
export {
  recoverFromSessionLog,
  maybeRecoverFromEmptyScheduledOutcome,
} from './session_recovery.js';
export { PHASE1_REFLECTION_BARRIER_NAME, runPhase1Reflection } from './phase1_reflection.js';
export type { SilentDiagnostic, ProviderAuthConfig } from './adapter_safety.js';
export {
  ANTHROPIC_PROVIDER,
  OPENAI_PROVIDER,
  buildSilentDiagnostic,
  isSilentFailureEnvelope,
} from './adapter_safety.js';

/** When the container is launched under egress isolation, the host injects
 *  HTTPS_PROXY. Node's global fetch (undici) does NOT honor it automatically,
 *  so install a ProxyAgent dispatcher. This routes anthropic_direct_transport
 *  (global fetch) and in-process @openai/agents calls through the allowlisting
 *  proxy. The spawned claude CLI subprocess reads HTTPS_PROXY itself. */
function installEgressDispatcher(): void {
  const proxy = process.env.HTTPS_PROXY || process.env.https_proxy;
  if (!proxy) return;
  try {
    setGlobalDispatcher(new ProxyAgent(proxy));
    log(`Egress dispatcher installed via ${proxy}`);
  } catch (err) {
    log(`Failed to install egress dispatcher: ${String(err)}`);
  }
}

async function flushStdStreams(): Promise<void> {
  await new Promise<void>((resolve) => process.stdout.write('', () => resolve()));
  await new Promise<void>((resolve) => process.stderr.write('', () => resolve()));
}

/**
 * Structurally validate stdin JSON before trusting it as a ContainerInput.
 * `JSON.parse` returns `any`, so a version-skewed host (e.g. snake_case
 * `workspace_key`) would otherwise sail through and surface later as a cryptic
 * undefined-field failure. Throwing here routes to the parse-error envelope.
 */
function assertContainerInput(value: unknown): asserts value is ContainerInput {
  if (typeof value !== 'object' || value === null) {
    throw new Error('container input is not a JSON object');
  }
  const v = value as Record<string, unknown>;
  if (typeof v.prompt !== 'string') {
    throw new Error("container input missing string field 'prompt'");
  }
  if (typeof v.workspaceKey !== 'string') {
    throw new Error("container input missing string field 'workspaceKey'");
  }
}

async function main(): Promise<void> {
  installEgressDispatcher();
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    const parsedInput: unknown = JSON.parse(stdinData);
    assertContainerInput(parsedInput);
    containerInput = parsedInput;
    setOutputProtocolNonce(
      typeof containerInput.protocolNonce === 'string'
        ? containerInput.protocolNonce
        : '',
    );
    // Delete the temp file the entrypoint wrote — it contains secrets
    try { fs.unlinkSync('/tmp/input.json'); } catch { /* may not exist */ }
    log(`Received input for workspace: ${containerInput.workspaceKey}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`
    });
    process.exit(1);
  }

  // Build SDK env: merge secrets into process.env for the SDK only.
  // Secrets never touch process.env itself, so Bash subprocesses can't see them.
  const sdkEnv: Record<string, string | undefined> = { ...process.env };
  for (const [key, value] of Object.entries(containerInput.secrets || {})) {
    sdkEnv[key] = value;
  }
  sdkEnv.CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS = '0';

  const provider = String(sdkEnv.MODEL_PROVIDER || 'anthropic').toLowerCase();
  const scheduledTaskSource = (
    containerInput.memoryMcp?.taskSource
    || containerInput.arcTools?.taskSource
    || ''
  ).toLowerCase();
  const scheduledForumTaskSources = new Set([
    'per_task_forum',
    'cross_task_forum',
  ]);

  let sessionId = containerInput.sessionId;
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  // Build initial prompt (drain any pending IPC messages too)
  let prompt = `[AUTOMATED TASK - This task was dispatched programmatically by the orchestrator.]\n\n${containerInput.prompt}`;
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // Single-shot dispatch: every adapter branch emits its envelope and exits.
  try {
    log(`Starting query (session: ${sessionId || 'new'})...`);
    if (provider === 'openai') {
      const queryResult = await runOpenAIQuery(prompt, sessionId, containerInput, sdkEnv);
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }
      // When the OpenAI adapter returns `status: 'error'` (currently only
      // from the MaxTurnsExceededError salvage path), emit an error
      // envelope with salvaged tokens/tool_trace rather than a
      // `status='success'`. The SDK consumed real tokens and may have
      // completed tool calls before hitting the turn cap — preserving
      // them in the envelope keeps the Python host's accounting honest
      // and lets the seeding pipeline see partial progress. In
      // scheduled-task mode we still exit after writing the envelope,
      // matching the success path's single-shot semantics.
      const openaiStatus: 'success' | 'error' = queryResult.status === 'error' ? 'error' : 'success';
      // NB: ARC runs natively for every provider now (attempt files, no
      // arc_submit_trial tool). The native ARC trace is synthesized host-side
      // AFTER the container exits (runtime_runner/src/main.ts +
      // arc_nomcp_synth.ts, gated on payload.arc_no_mcp), so an in-container
      // guard requiring an arc_submit_trial tool call would wrongly fail every
      // native ARC attempt — hence there is no such guard here.
      writeOutput({
        status: openaiStatus,
        result: queryResult.resultText,
        newSessionId: sessionId,
        toolTrace: queryResult.toolTrace,
        input_tokens: queryResult.input_tokens,
        output_tokens: queryResult.output_tokens,
        cache_creation_input_tokens: queryResult.cache_creation_input_tokens,
        cache_read_input_tokens: queryResult.cache_read_input_tokens,
        ...(queryResult.error ? { error: queryResult.error } : {}),
        // Polyglot test-feedback retry-loop diagnostics + dedicated token
        // usage (Aider protocol). Only present when the feature flag was on
        // (task_source='polyglot', triesRemaining > 1). The host-side
        // runtime_runner/src/main.ts copies these envelope fields into
        // runtime_meta generically, same as the Claude path's envelope.
        ...(queryResult.polyglot_test_feedback_meta
          ? { polyglot_test_feedback_meta: queryResult.polyglot_test_feedback_meta }
          : {}),
        ...(queryResult.polyglot_test_feedback_token_usage
          ? { polyglot_test_feedback_token_usage: queryResult.polyglot_test_feedback_token_usage }
          : {}),
      });
      log(
        `Scheduled task mode (openai): single-shot query complete (status=${openaiStatus}), returning`,
      );
      await flushStdStreams();
      process.exit(openaiStatus === 'error' ? 1 : 0);
    } else if (
      scheduledForumTaskSources.has(scheduledTaskSource)
      && String(sdkEnv.KSI_ANTHROPIC_FORUM_ADAPTER || 'direct').toLowerCase() !== 'claude-code'
    ) {
      const queryResult = await runAnthropicDirectForumQuery(prompt, containerInput, sdkEnv);
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }
      const anthropicStatus: 'success' | 'error' =
        queryResult.status === 'error' ? 'error' : 'success';
      writeOutput({
        status: anthropicStatus,
        result: anthropicStatus === 'error' ? null : queryResult.resultText,
        newSessionId: sessionId,
        toolTrace: queryResult.toolTrace.slice(-1000),
        input_tokens: queryResult.input_tokens,
        output_tokens: queryResult.output_tokens,
        cache_creation_input_tokens: queryResult.cache_creation_input_tokens,
        cache_read_input_tokens: queryResult.cache_read_input_tokens,
        tokens_source: queryResult.tokens_source,
        ...(queryResult.error ? { error: queryResult.error } : {}),
        // Cross-task shared-container per-round outputs. Only present
        // when the feature flag was on AND this was a cross_task_forum
        // dispatch — the host engine.py harvests them from runtime_meta.
        ...(queryResult.cross_task_round_0_result
          ? { cross_task_round_0_result: queryResult.cross_task_round_0_result }
          : {}),
        ...(queryResult.cross_task_round_1_result
          ? { cross_task_round_1_result: queryResult.cross_task_round_1_result }
          : {}),
        ...(queryResult.cross_task_shared_container_meta
          ? { cross_task_shared_container_meta: queryResult.cross_task_shared_container_meta }
          : {}),
      });
      log(
        `Scheduled task mode (anthropic direct forum): single-shot query complete (status=${anthropicStatus}), returning`,
      );
      await flushStdStreams();
      process.exit(anthropicStatus === 'error' ? 1 : 0);
    } else {
      const queryResult = await runQuery(prompt, sessionId, containerInput, sdkEnv);
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }

      // Bench/scheduled mode is single-shot: do not wait for IPC follow-ups.
      log('Scheduled task mode: single-shot query complete, returning');
      await flushStdStreams();
      process.exit(0);
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    // Provider-aware diagnostic: OpenAI errors surface OPENAI_API_KEY shape
    // (not ANTHROPIC_API_KEY). Env VALUES are never captured — only
    // lengths and types. See adapter_safety.ts.
    const providerCfg = provider === 'openai' ? OPENAI_PROVIDER : ANTHROPIC_PROVIDER;
    // mcpServerConfig is scoped inside runQuery and not visible here in the
    // outer catch block — pass an empty map so the diagnostic reports "no
    // MCP servers known" at this failure point instead of failing to build.
    const diag = buildSilentDiagnostic({
      provider: providerCfg,
      messageCount: 0,
      resultCount: 0,
      lastAssistantFallback: null,
      perTurnInputTokens: 0,
      perTurnOutputTokens: 0,
      resultInputTokens: 0,
      resultOutputTokens: 0,
      sdkEnv,
      // mcpServerConfig is scoped to runQuery()/runOpenAIQuery() internals
      // and not reachable here. MCP state is less useful at the outer catch
      // than it is inside the silent-exit branches, so an empty list is fine.
      mcpServerNames: [],
      iteratorError: err instanceof Error ? err : new Error(errorMessage),
    });
    log(`Agent-error diagnostic: ${JSON.stringify(diag)}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      tokens_source: 'unavailable',
      error: `${errorMessage}. diagnostic=${JSON.stringify(diag)}`,
    });
    process.exit(1);
  }
}

/**
 * Write a terminal diagnostic envelope for a fatal process-level error
 * (unhandled promise rejection or uncaught exception). These reach here
 * when the SDK's internal async stack rejects without the for-await loop
 * seeing it -- this is the path that causes silent-drain bugs where
 * nothing is written to stdout at all. Emitting an OUTPUT envelope here
 * gives the host a structured error to record instead of the previous
 * "container exited cleanly with no markers" mystery.
 */
function emitTerminalDiagnostic(kind: 'unhandledRejection' | 'uncaughtException', err: unknown): void {
  const message = err instanceof Error ? err.message : String(err);
  const name = err instanceof Error ? err.name : 'Error';
  const stackHead = err instanceof Error && err.stack
    ? err.stack.split('\n').slice(0, 8).join('\n')
    : undefined;
  // Provider-aware env sampling: OpenAI failures surface OPENAI_API_KEY
  // shape, Anthropic (and unknown) failures surface the
  // ANTHROPIC_API_KEY + CLAUDE_CODE_OAUTH_TOKEN pair. Previously hardcoded
  // to Anthropic, which false-implicated the wrong env var on OpenAI
  // silent-exit forensics.
  const providerCfg =
    (process.env.MODEL_PROVIDER || '').toLowerCase() === 'openai'
      ? OPENAI_PROVIDER
      : ANTHROPIC_PROVIDER;
  const innerErr = err instanceof Error ? err : new Error(message);
  const diagBody = buildSilentDiagnostic({
    provider: providerCfg,
    messageCount: 0,
    resultCount: 0,
    lastAssistantFallback: null,
    perTurnInputTokens: 0,
    perTurnOutputTokens: 0,
    resultInputTokens: 0,
    resultOutputTokens: 0,
    sdkEnv: process.env as Record<string, string | undefined>,
    mcpServerNames: [],
    iteratorError: innerErr,
  });
  const diag = { kind, ...diagBody };
  const causeDesc = diagBody.iteratorError?.cause;
  try {
    console.error(`[agent-runner] FATAL ${kind}: ${name}: ${message}`);
    if (stackHead) console.error(stackHead);
    if (causeDesc) console.error(`cause: ${causeDesc}`);
    console.error(`[agent-runner] ${kind} diagnostic: ${JSON.stringify(diag)}`);
  } catch {
    /* ignore */
  }
  try {
    writeOutput({
      status: 'error',
      result: null,
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      tokens_source: 'unavailable',
      error:
        `agent-runner fatal ${kind}: ${name}: ${message.slice(0, 240)}. ` +
        `diagnostic=${JSON.stringify(diag)}`,
    });
  } catch {
    /* last-resort: ensure we never throw from the handler */
  }
}

process.on('unhandledRejection', (reason) => {
  emitTerminalDiagnostic('unhandledRejection', reason);
  // Exit non-zero so the host reclassifies the attempt. We give stdout a
  // moment to flush before exiting; setImmediate is enough since writeOutput
  // goes through console.log which drains on nextTick.
  setImmediate(() => process.exit(1));
});

process.on('uncaughtException', (err) => {
  emitTerminalDiagnostic('uncaughtException', err);
  setImmediate(() => process.exit(1));
});

const entryPath = process.argv[1] ? path.resolve(process.argv[1]) : '';
if (fileURLToPath(import.meta.url) === entryPath) {
  main();
}
