/**
 * Polyglot test-feedback retry loop ("Aider protocol"). After a polyglot
 * task's first submission, the host runs the real PolyglotHarnessEvaluator
 * against the agent's live on-disk edits while the container waits; on
 * failure, the agent gets ONE resumed SDK turn (tools re-enabled) per
 * remaining try, seeing its own capped test-runner output, mirroring
 * Aider's --tries 2 / 50-line-cap benchmark protocol.
 *
 * Deliberately separate from phase1_reflection.ts: that feature disables
 * tools and never changes what gets graded. This one re-enables tools and
 * its last round's on-disk edits ARE what gets graded (via the unchanged
 * post-hoc evaluate() call in execution_phase.py).
 */
import { McpServerConfig, query } from '@anthropic-ai/claude-agent-sdk';
import { UsageDelta, extractUsageFromSdkMessage } from './usage.js';
import { extractAssistantText } from './extract.js';
import { CONTAINER_WORKSPACE_ROOT, IPC_POLL_MS } from './runner_constants.js';
import {
  responsePath as barrierResponsePath,
  waitForBarrierFile,
  writeSentinelFile as writeBarrierSentinel,
} from './barrier.js';
import {
  POLYGLOT_TEST_FEEDBACK_BARRIER_NAME,
  addUsage,
  buildTestFeedbackPrompt,
  classifyBarrierEval,
  extractCappedTail,
  extractRawTestOutput,
  retryRoundCount,
  selectRoundUsage,
  shouldBailOnRoundError,
  summarizeEvalResult,
  zeroUsage,
} from './polyglot_test_feedback_core.js';

// Back-compat re-exports: these were historically exported from this module
// (before the provider-agnostic core was extracted for the OpenAI path).
export {
  POLYGLOT_TEST_FEEDBACK_BARRIER_NAME,
  buildTestFeedbackPrompt,
  extractCappedTail,
} from './polyglot_test_feedback_core.js';

const POLL_TIMEOUT_MS_DEFAULT = 120_000;

export async function runPolyglotTestFeedback(args: {
  workspaceDir: string;
  agentId: string;
  fileList: string;
  modelOutput: string | null;
  triesRemaining: number;
  maxLines: number;
  resumeSessionId: string | undefined;
  resumeAt: string | undefined;
  selectedModel: string | undefined;
  sdkEnv: Record<string, string | undefined>;
  allowedTools: string[];
  mcpServers: Record<string, McpServerConfig>;
  maxTurnsPerRound: number;
  pollTimeoutMs?: number;
  logger?: (msg: string) => void;
}): Promise<{
  finalResult: string | null;
  roundsUsed: number;
  attempt1EvalSummary: Record<string, unknown> | null;
  captured: boolean;
  note?: string;
  tokenUsage: UsageDelta;
  // True only when the LAST barrier round's evaluation genuinely reflects
  // the final graded state (i.e. `evalPayload.resolved === true` fired and
  // no further agent turn ran afterward). False whenever the loop exhausted
  // its tries after an edit turn -- that last evaluate() call scored the
  // PRE-turn state, so it must never be reused as the final score.
  finalEvalMatchesOutput: boolean;
}> {
  const log0 = args.logger || ((m: string) => process.stderr.write(`[polyglot_test_feedback] ${m}\n`));
  let currentOutput = args.modelOutput;
  let currentSessionId = args.resumeSessionId;
  let currentResumeAt = args.resumeAt;
  let attempt1EvalSummary: Record<string, unknown> | null = null;
  let totalUsage = zeroUsage();
  let roundsUsed = 0;

  for (let round = 0; round < retryRoundCount(args.triesRemaining); round += 1) {
    const sentinelTarget = writeBarrierSentinel(
      args.workspaceDir,
      POLYGLOT_TEST_FEEDBACK_BARRIER_NAME,
      args.agentId,
      {
        schema: 'polyglot_test_feedback.v1',
        agent_id: args.agentId,
        // 8MB safety bound, matching phase1_reflection.ts's cap: this
        // sentinel content IS what gets scored, so cropping it must be a
        // last-resort safety valve, not routine truncation.
        model_output: typeof currentOutput === 'string' ? currentOutput.slice(0, 8_000_000) : null,
      },
    );
    log0(`round ${round}: wrote barrier sentinel ${sentinelTarget}`);

    const responseFile = barrierResponsePath(args.workspaceDir, POLYGLOT_TEST_FEEDBACK_BARRIER_NAME, args.agentId);
    const timeoutMs = args.pollTimeoutMs ?? POLL_TIMEOUT_MS_DEFAULT;
    const evalPayload = await waitForBarrierFile(responseFile, timeoutMs, { pollIntervalMs: IPC_POLL_MS });
    if (!evalPayload) {
      return {
        finalResult: currentOutput,
        roundsUsed,
        attempt1EvalSummary,
        captured: false,
        note: `polyglot_test_feedback: host barrier response not received within ${timeoutMs}ms`,
        tokenUsage: totalUsage,
        finalEvalMatchesOutput: false,
      };
    }

    if (round === 0) attempt1EvalSummary = summarizeEvalResult(evalPayload);
    const evalDecision = classifyBarrierEval(evalPayload);
    if (evalDecision === 'error') {
      // Host-side failure (evaluator crash, timeout, or a round-limit
      // refusal) -- NOT a real test failure. Bail out instead of building a
      // retry prompt from empty test output and burning a real retry round
      // reacting to an infra hiccup (errors-timeouts.md Finding 1).
      return {
        finalResult: currentOutput,
        roundsUsed,
        attempt1EvalSummary,
        captured: false,
        note: `polyglot_test_feedback: host reported an error on round ${round}: ${evalPayload.error}`,
        tokenUsage: totalUsage,
        finalEvalMatchesOutput: false,
      };
    }
    if (evalDecision === 'resolved') {
      return {
        finalResult: currentOutput,
        roundsUsed,
        attempt1EvalSummary,
        captured: true,
        tokenUsage: totalUsage,
        finalEvalMatchesOutput: true,
      };
    }

    const rawTestOutput = extractRawTestOutput(evalPayload);
    const cappedOutput = extractCappedTail(rawTestOutput, args.maxLines);
    const prompt = buildTestFeedbackPrompt({ testOutput: cappedOutput, fileList: args.fileList });

    let lastAssistant: unknown = null;
    let roundResult = '';
    let resultReceivedThisRound = false;
    let roundResultUsage = zeroUsage();
    let roundPerTurnUsage = zeroUsage();
    try {
      for await (const message of query({
        prompt,
        options: {
          model: args.selectedModel || undefined,
          cwd: CONTAINER_WORKSPACE_ROOT,
          resume: currentSessionId,
          resumeSessionAt: currentResumeAt,
          allowedTools: args.allowedTools,
          mcpServers: args.mcpServers,
          env: args.sdkEnv,
          maxTurns: args.maxTurnsPerRound,
          permissionMode: 'bypassPermissions',
          allowDangerouslySkipPermissions: true,
          settingSources: ['project', 'user'],
          stderr: () => {},
        },
      })) {
        const msgType = (message as { type?: string }).type;
        // Assistant messages nest usage under `.message.usage` (per-turn
        // delta); result events expose it at the top level (turn
        // aggregate). Track the two separately and pick one below --
        // summing both double-counts tokens. See usage.ts and
        // phase1_reflection.ts's pickReflectionUsage for the same pattern.
        const delta = extractUsageFromSdkMessage(message);
        if (msgType === 'result') {
          roundResultUsage = addUsage(roundResultUsage, delta);
        } else {
          roundPerTurnUsage = addUsage(roundPerTurnUsage, delta);
        }
        if (msgType === 'assistant') lastAssistant = message;
        if (
          msgType === 'system'
          && (message as { subtype?: string }).subtype === 'init'
          && typeof (message as { session_id?: string }).session_id === 'string'
        ) {
          currentSessionId = (message as { session_id: string }).session_id;
        }
        if (msgType === 'result') {
          const textResult = (message as { result?: string }).result;
          if (typeof textResult === 'string' && textResult.trim()) roundResult = textResult.trim();
          resultReceivedThisRound = true;
          break;
        }
      }
    } catch (err) {
      // `break` right after a captured `result` event runs the async
      // iterator's `.return()` cleanup, which can throw AFTER the result
      // was already captured (see query_runner.ts's `resultCount === 0`
      // guard for the same documented SDK behavior, issue #945). If we
      // already have this round's result, don't discard it -- fall
      // through and finish the round normally instead of reporting a
      // spurious failure.
      if (shouldBailOnRoundError(resultReceivedThisRound)) {
        return {
          finalResult: currentOutput,
          roundsUsed,
          attempt1EvalSummary,
          captured: false,
          note: `polyglot_test_feedback: SDK round ${round} threw: ${err instanceof Error ? err.message : String(err)}`,
          tokenUsage: totalUsage,
          finalEvalMatchesOutput: false,
        };
      }
    }
    totalUsage = addUsage(totalUsage, selectRoundUsage(roundResultUsage, roundPerTurnUsage));
    if (!roundResult && lastAssistant) {
      roundResult = extractAssistantText(lastAssistant) || currentOutput || '';
    }
    currentOutput = roundResult || currentOutput;
    roundsUsed += 1;
  }

  // Tries are exhausted after the last edit turn. Evaluate the FINAL
  // on-disk state once more so execution_phase.py's cache-reuse can skip
  // its own redundant post-hoc evaluate() call in the common case (a retry
  // actually happened), not just when attempt 1 passed outright. This does
  // NOT add a new Docker evaluation beyond what execution_phase.py would
  // otherwise run itself -- it relocates that unavoidable "did it work"
  // check into a barrier round the cache mechanism can reuse
  // (performance.md Finding 1).
  const finalSentinelTarget = writeBarrierSentinel(
    args.workspaceDir,
    POLYGLOT_TEST_FEEDBACK_BARRIER_NAME,
    args.agentId,
    {
      schema: 'polyglot_test_feedback.v1',
      agent_id: args.agentId,
      model_output: typeof currentOutput === 'string' ? currentOutput.slice(0, 8_000_000) : null,
    },
  );
  log0(`final round: wrote barrier sentinel ${finalSentinelTarget}`);
  const finalResponseFile = barrierResponsePath(args.workspaceDir, POLYGLOT_TEST_FEEDBACK_BARRIER_NAME, args.agentId);
  const finalTimeoutMs = args.pollTimeoutMs ?? POLL_TIMEOUT_MS_DEFAULT;
  const finalEvalPayload = await waitForBarrierFile(finalResponseFile, finalTimeoutMs, { pollIntervalMs: IPC_POLL_MS });
  if (!finalEvalPayload || (typeof finalEvalPayload.error === 'string' && finalEvalPayload.error)) {
    return {
      finalResult: currentOutput,
      roundsUsed,
      attempt1EvalSummary,
      captured: true,
      note: finalEvalPayload
        ? `polyglot_test_feedback: host reported an error on the final post-loop check: ${finalEvalPayload.error}`
        : `polyglot_test_feedback: host barrier response not received within ${finalTimeoutMs}ms on the final post-loop check`,
      tokenUsage: totalUsage,
      finalEvalMatchesOutput: false,
    };
  }
  return {
    finalResult: currentOutput,
    roundsUsed,
    attempt1EvalSummary,
    captured: true,
    tokenUsage: totalUsage,
    finalEvalMatchesOutput: true,
  };
}
