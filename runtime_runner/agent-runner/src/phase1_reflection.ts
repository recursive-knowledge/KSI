/**
 * Phase-1 reflection round-trip. After a scheduled task's success path, the
 * agent is asked for a short structured self-reflection on the attempt:
 *   1. Write a barrier sentinel describing the just-completed attempt.
 *   2. Wait for the host's BarrierWatcher to evaluate and respond.
 *   3. Run ONE additional `query()` turn (tools disabled) capturing a 3-5
 *      sentence reflection.
 *
 * {@link runPhase1Reflection} always resolves — never throws — so the caller
 * can always ship its success envelope with `phase1_reflection` optionally
 * populated.
 */
import { query } from '@anthropic-ai/claude-agent-sdk';
import { MessageStream } from './message_stream.js';
import { UsageDelta, extractUsageFromSdkMessage } from './usage.js';
import { extractAssistantText } from './extract.js';
import { CONTAINER_WORKSPACE_ROOT, IPC_POLL_MS } from './runner_constants.js';
import {
  responsePath as barrierResponsePath,
  waitForBarrierFile,
  writeSentinelFile as writeBarrierSentinel,
} from './barrier.js';

/**
 * Barrier name used by the Phase 1 reflection round-trip. Centralized so
 * the host-side BarrierWatcher in Python uses the same identifier — these
 * two literals MUST agree (see `src/ksi/runtime/barrier.py`).
 */
export const PHASE1_REFLECTION_BARRIER_NAME = 'phase1_reflection';

/**
 * Default upper-bound on how long the in-container reflection step waits
 * for the host's BarrierWatcher to write back the eval result. Chosen so
 * the host's `evaluator.evaluate(...)` (typically subprocess-based for
 * polyglot/swebench, sub-second for ARC) has comfortable headroom while
 * the watcher still finishes within the host's `effective_timeout + 120s`
 * subprocess backstop.
 */
const PHASE1_EVAL_RESULT_POLL_TIMEOUT_MS_DEFAULT = 120_000;

/**
 * Reflection prompt template. Kept short so an in-session follow-up turn
 * stays cheap and the synthetic user turn doesn't dominate the cache key.
 *
 * The phrasing intentionally asks for the THREE structured fields
 * (load-bearing assumption, proposed change, predicted outcome) inline in
 * 3-5 sentences — extracting these into a per-task distill payload is the
 * downstream consumer's job (see `distillation/distiller.py:139`).
 */
function buildPhase1ReflectionPrompt(evalSummary: string): string {
  return [
    'Task evaluation result:',
    evalSummary,
    '',
    'In 3-5 sentences, write a concise structured reflection on this attempt.',
    'Cover (a) the load-bearing assumption you made, (b) the single change you',
    'would make if you had another shot, and (c) what outcome you predict that',
    'change would produce. No headings, no code blocks — just the reflection.',
  ].join('\n');
}

/**
 * Summarize the eval result (received via barrier from the host) into a
 * compact human-readable line for the reflection prompt. Defensive against
 * partial / unexpected shapes — never throws.
 */
function summarizeEvalResultForPrompt(payload: Record<string, unknown> | null): string {
  if (!payload) return 'evaluator returned no result';
  const score = payload.native_score ?? payload.score;
  const status = payload.status ?? (payload.eval_result as Record<string, unknown> | undefined)?.status;
  const resolved = payload.resolved ?? (payload.eval_result as Record<string, unknown> | undefined)?.resolved;
  const parts: string[] = [];
  if (score != null && score !== '') parts.push(`score=${String(score)}`);
  if (status != null && status !== '') parts.push(`status=${String(status)}`);
  if (resolved != null) parts.push(`resolved=${String(Boolean(resolved))}`);
  return parts.length > 0 ? parts.join(', ') : 'unknown evaluation outcome';
}

/**
 * Best-effort extract the assistant's last reply text from an SDK message.
 * Reuses the same shape that ``extractAssistantText`` operates on but is
 * lenient about the synthetic-followup flow (we don't care about ToolUse
 * blocks, only the textual reflection the model wrote).
 */
function extractTextFromAssistantMessage(message: unknown): string {
  const direct = extractAssistantText(message);
  if (typeof direct === 'string' && direct.trim()) return direct.trim();
  // ``extractAssistantText`` may return null for messages whose content is
  // entirely tool_use blocks; the reflection prompt explicitly disallows
  // tools so this branch should be rare. When it does happen, fall back to
  // a JSON dump excerpt so debugging still has signal.
  try {
    return JSON.stringify(message).slice(0, 4000);
  } catch {
    return '';
  }
}

/**
 * Prefer the result-event usage aggregate when the SDK shipped one;
 * fall back to the summed per-turn deltas otherwise. Mirrors the
 * source-of-truth selection in the main scheduled-task loop.
 */
function pickReflectionUsage(
  resultUsage: UsageDelta,
  perTurnUsage: UsageDelta,
): {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
} {
  const resultHasAny = (
    resultUsage.input_tokens
    || resultUsage.output_tokens
    || resultUsage.cache_creation_input_tokens
    || resultUsage.cache_read_input_tokens
  ) > 0;
  return resultHasAny ? resultUsage : perTurnUsage;
}

/**
 * Phase-1 reflection round-trip:
 *   1. Write a barrier sentinel describing the just-completed attempt.
 *   2. Wait for the host's BarrierWatcher to evaluate and respond.
 *   3. If the host responded, run ONE additional ``query()`` turn in the
 *      same SDK session with a synthetic user prompt asking for a 3-5
 *      sentence structured reflection.
 *   4. Capture the assistant's reply text and return it.
 *
 * Always returns a result object — never throws. On any failure (timeout,
 * SDK silent-exit on the follow-up, etc.) the result has `captured: false`
 * and a `note` explaining why so the caller can ship a graceful envelope.
 *
 * The follow-up `query()` deliberately disables every tool the agent had —
 * we want a textual reflection, not more tool use. We also pin
 * `maxTurns: 1` so the SDK can't decide to escalate into another tool loop.
 */
export async function runPhase1Reflection(args: {
  workspaceDir: string;
  agentId: string;
  taskId: string;
  modelOutput: string | null;
  resumeSessionId: string | undefined;
  resumeAt: string | undefined;
  selectedModel: string | undefined;
  sdkEnv: Record<string, string | undefined>;
  pollTimeoutMs?: number;
  logger?: (msg: string) => void;
}): Promise<{
  captured: boolean;
  text?: string;
  note?: string;
  elapsedMs: number;
  hostPayload?: Record<string, unknown> | null;
  // Usage observed on the reflection turn's `result` event (or summed
  // from per-turn deltas if the result event lacked a usage block).
  // Surfaced so the host can record a `phase1_reflection` token-phases
  // row — without this the reflection turn's tokens vanish.
  tokenUsage?: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens: number;
    cache_read_input_tokens: number;
  };
}> {
  const startedAt = Date.now();
  const log0 = args.logger || ((m: string) => process.stderr.write(`[phase1] ${m}\n`));
  const sentinelPayload = {
    schema: 'phase1_reflection.v1',
    agent_id: args.agentId,
    task_id: args.taskId,
    // Pass the full model_output through the sentinel so the host's
    // BarrierWatcher evaluates exactly what the engine's _eval_stage would
    // have seen. The previous 200KB cap caused two problems:
    //   1. Score disagreement on swebench_pro patches >200KB — the
    //      watcher-side eval saw a truncated patch, the engine-side eval
    //      saw the full one, so the persisted attempt's reflection was
    //      built off a different score than the persisted score.
    //   2. Re-truncation broke the engine-side eval-result reuse the host
    //      now performs (see container_host.py phase1_eval_result cache):
    //      different inputs => can't safely reuse.
    // 8MB is a generous safety bound — large patches typically <1MB even
    // for monorepo-spanning fixes; pathological cases get cropped rather
    // than blowing memory. The sentinel file lives on the host's local
    // tmpfs, so the cost of writing it is negligible.
    model_output: typeof args.modelOutput === 'string'
      ? args.modelOutput.slice(0, 8_000_000)
      : null,
  };
  let sentinelTarget: string;
  try {
    sentinelTarget = writeBarrierSentinel(
      args.workspaceDir,
      PHASE1_REFLECTION_BARRIER_NAME,
      args.agentId,
      sentinelPayload,
    );
  } catch (err) {
    const note = `phase1: failed to write barrier sentinel: ${err instanceof Error ? err.message : String(err)}`;
    log0(note);
    return { captured: false, note, elapsedMs: Date.now() - startedAt };
  }
  log0(`wrote barrier sentinel ${sentinelTarget}`);

  const responseFile = barrierResponsePath(
    args.workspaceDir,
    PHASE1_REFLECTION_BARRIER_NAME,
    args.agentId,
  );
  const timeoutMs = args.pollTimeoutMs ?? PHASE1_EVAL_RESULT_POLL_TIMEOUT_MS_DEFAULT;
  const evalPayload = await waitForBarrierFile(responseFile, timeoutMs, {
    pollIntervalMs: IPC_POLL_MS,
  });
  if (!evalPayload) {
    const note = `phase1: host barrier response not received within ${timeoutMs}ms`;
    log0(note);
    return { captured: false, note, elapsedMs: Date.now() - startedAt };
  }
  log0(`received barrier response keys=${Object.keys(evalPayload).join(',')}`);

  const evalSummary = summarizeEvalResultForPrompt(evalPayload);
  const followupPrompt = buildPhase1ReflectionPrompt(evalSummary);

  // Run a single follow-up SDK turn. If the SDK silent-exits (memory
  // `sdk_stream_race_silent_failure` notes ~3-5% under load), capture
  // nothing and fall through; the caller emits the success envelope
  // without `phase1_reflection`.
  let reflectionText = '';
  // Accumulate token usage from the reflection turn so the host can
  // emit a ``phase1_reflection`` row in ``token_phases``. Without this
  // accounting the reflection turn's input/output/cache tokens silently
  // vanish from cost reports.
  //
  // Mirror the dual-source capture in ``runQuery``:
  //   - prefer the result event's aggregate usage when present
  //   - fall back to per-turn deltas if the result event omits usage
  let reflectionResultUsage: UsageDelta = {
    input_tokens: 0, output_tokens: 0,
    cache_creation_input_tokens: 0, cache_read_input_tokens: 0,
  };
  let reflectionPerTurnUsage: UsageDelta = {
    input_tokens: 0, output_tokens: 0,
    cache_creation_input_tokens: 0, cache_read_input_tokens: 0,
  };
  try {
    const stream = new MessageStream();
    stream.push(followupPrompt);
    stream.end();
    let lastAssistant: unknown = null;
    for await (const message of query({
      prompt: stream,
      options: {
        model: args.selectedModel || undefined,
        cwd: CONTAINER_WORKSPACE_ROOT,
        resume: args.resumeSessionId,
        resumeSessionAt: args.resumeAt,
        // Reflection is a textual self-report — disable every tool so the
        // model can't accidentally escalate this turn into more tool work.
        // ``allowedTools: []`` is the floor on built-in (Bash/Read/Edit/...) tools.
        // ``mcpServers: {}`` is the cleanest way to disable every MCP server
        // for this turn — earlier code listed glob patterns
        // (``mcp__memory__*`` etc.) on ``disallowedTools``, which the SDK
        // matcher MAY treat as literal strings depending on version, leaving
        // the agent's MCP tools (memory, arc, ksi) callable on the
        // reflection turn. Setting ``mcpServers: {}`` removes the MCP
        // surface entirely so we don't have to hand-enumerate every concrete
        // tool name and keep that list in sync with future MCP server
        // additions.
        allowedTools: [],
        disallowedTools: [
          'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
          'WebSearch', 'WebFetch', 'TodoWrite', 'NotebookEdit',
          'Task', 'TaskOutput', 'TaskStop',
        ],
        mcpServers: {},
        env: args.sdkEnv,
        maxTurns: 1,
        permissionMode: 'bypassPermissions',
        allowDangerouslySkipPermissions: true,
        settingSources: ['project', 'user'],
        stderr: () => {},
      },
    })) {
      const msgType = (message as { type?: string }).type;
      // Always extract usage — assistant messages nest it under
      // ``message.usage`` (per-turn deltas), result events expose it at
      // the top level (turn aggregate). See extractUsageFromSdkMessage.
      const delta = extractUsageFromSdkMessage(message);
      if (
        delta.input_tokens || delta.output_tokens
        || delta.cache_creation_input_tokens || delta.cache_read_input_tokens
      ) {
        if (msgType === 'result') {
          reflectionResultUsage = {
            input_tokens: reflectionResultUsage.input_tokens + delta.input_tokens,
            output_tokens: reflectionResultUsage.output_tokens + delta.output_tokens,
            cache_creation_input_tokens:
              reflectionResultUsage.cache_creation_input_tokens + delta.cache_creation_input_tokens,
            cache_read_input_tokens:
              reflectionResultUsage.cache_read_input_tokens + delta.cache_read_input_tokens,
          };
        } else {
          reflectionPerTurnUsage = {
            input_tokens: reflectionPerTurnUsage.input_tokens + delta.input_tokens,
            output_tokens: reflectionPerTurnUsage.output_tokens + delta.output_tokens,
            cache_creation_input_tokens:
              reflectionPerTurnUsage.cache_creation_input_tokens + delta.cache_creation_input_tokens,
            cache_read_input_tokens:
              reflectionPerTurnUsage.cache_read_input_tokens + delta.cache_read_input_tokens,
          };
        }
      }
      if (msgType === 'assistant') {
        lastAssistant = message;
      }
      if (msgType === 'result') {
        // result event — break out, the SDK is done with this turn
        const textResult = (message as { result?: string }).result;
        if (typeof textResult === 'string' && textResult.trim()) {
          reflectionText = textResult.trim();
        }
        break;
      }
    }
    if (!reflectionText && lastAssistant) {
      reflectionText = extractTextFromAssistantMessage(lastAssistant).trim();
    }
  } catch (err) {
    const note = `phase1: SDK follow-up threw: ${err instanceof Error ? err.message : String(err)}`;
    log0(note);
    return {
      captured: false,
      note,
      elapsedMs: Date.now() - startedAt,
      hostPayload: evalPayload,
      tokenUsage: pickReflectionUsage(reflectionResultUsage, reflectionPerTurnUsage),
    };
  }

  // Same dual-source preference the main runQuery loop uses: trust the
  // SDK's result-event aggregate when it shipped one, otherwise sum the
  // per-turn deltas. Either way the host gets a single phase1_reflection
  // usage row.
  const reflectionUsage = pickReflectionUsage(reflectionResultUsage, reflectionPerTurnUsage);
  if (!reflectionText) {
    const note = 'phase1: SDK follow-up yielded no assistant text';
    log0(note);
    return {
      captured: false,
      note,
      elapsedMs: Date.now() - startedAt,
      hostPayload: evalPayload,
      tokenUsage: reflectionUsage,
    };
  }
  // Cap to a sane length — distillation already truncates, but trimming
  // here keeps the runtime_meta small.
  const capped = reflectionText.slice(0, 8000);
  log0(
    `captured reflection (${capped.length} chars) in ${Date.now() - startedAt}ms `
    + `(in=${reflectionUsage.input_tokens}, out=${reflectionUsage.output_tokens}, `
    + `cc=${reflectionUsage.cache_creation_input_tokens}, cr=${reflectionUsage.cache_read_input_tokens})`,
  );
  return {
    captured: true,
    text: capped,
    elapsedMs: Date.now() - startedAt,
    hostPayload: evalPayload,
    tokenUsage: reflectionUsage,
  };
}
