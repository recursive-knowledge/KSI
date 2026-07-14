/**
 * Polyglot test-feedback retry loop ("Aider protocol") for the OpenAI
 * provider path.
 *
 * Same protocol and meta semantics as the Claude Agent SDK loop in
 * `polyglot_test_feedback.ts` (see its module docstring): after the first
 * submission, write a barrier sentinel with the live model output, wait for
 * the host to run the real PolyglotHarnessEvaluator, and — on failure —
 * feed the capped test-runner output back to the model for a bounded retry
 * round whose on-disk edits ARE what gets graded. The two loops share the
 * provider-agnostic core (`polyglot_test_feedback_core.ts`); this one keeps
 * the provider-specific round run behind the injected `runRound` callback
 * (built in `openai.ts` over the existing @openai/agents tool loop), so this
 * module stays SDK-free and behaviorally testable
 * (tests/js/polyglot_test_feedback_openai.test.mjs).
 */
import { UsageDelta } from './usage.js';
import { IPC_POLL_MS } from './runner_constants.js';
import {
  responsePath as barrierResponsePath,
  waitForBarrierFile,
  writeSentinelFile as writeBarrierSentinel,
} from './barrier.js';
import {
  POLYGLOT_TEST_FEEDBACK_BARRIER_NAME,
  addUsage,
  buildTestFeedbackPrompt,
  extractCappedTail,
  extractRawTestOutput,
  summarizeEvalResult,
  zeroUsage,
} from './polyglot_test_feedback_core.js';

const POLL_TIMEOUT_MS_DEFAULT = 120_000;

/**
 * One provider retry round: run `prompt` against the model with the task's
 * workspace tools re-enabled, bounded by the config's maxTurnsPerRound.
 * `text` is the round's assistant text (empty/null keeps the previous
 * output). Implementations must salvage recoverable SDK errors themselves
 * (e.g. MaxTurnsExceededError — the round's on-disk edits still count);
 * anything thrown here aborts the loop gracefully with a diagnostic note.
 */
export type OpenAIPolyglotRoundRunner = (
  prompt: string,
  round: number,
) => Promise<{ text: string | null; usage: UsageDelta }>;

export async function runOpenAIPolyglotTestFeedback(args: {
  workspaceDir: string;
  agentId: string;
  fileList: string;
  modelOutput: string | null;
  triesRemaining: number;
  maxLines: number;
  pollTimeoutMs?: number;
  runRound: OpenAIPolyglotRoundRunner;
  logger?: (msg: string) => void;
}): Promise<{
  finalResult: string | null;
  roundsUsed: number;
  attempt1EvalSummary: Record<string, unknown> | null;
  captured: boolean;
  note?: string;
  tokenUsage: UsageDelta;
  // Same contract as the Claude loop: true only when the LAST barrier
  // round's evaluation genuinely reflects the final graded state (no agent
  // turn ran after it).
  finalEvalMatchesOutput: boolean;
}> {
  const log0 = args.logger || ((m: string) => process.stderr.write(`[polyglot_test_feedback_openai] ${m}\n`));
  let currentOutput = args.modelOutput;
  let attempt1EvalSummary: Record<string, unknown> | null = null;
  let totalUsage = zeroUsage();
  let roundsUsed = 0;

  for (let round = 0; round < Math.max(0, args.triesRemaining - 1); round += 1) {
    const sentinelTarget = writeBarrierSentinel(
      args.workspaceDir,
      POLYGLOT_TEST_FEEDBACK_BARRIER_NAME,
      args.agentId,
      {
        schema: 'polyglot_test_feedback.v1',
        agent_id: args.agentId,
        // 8MB safety bound, matching the Claude loop's cap: this sentinel
        // content IS what gets scored, so cropping it must be a last-resort
        // safety valve, not routine truncation.
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
    if (typeof evalPayload.error === 'string' && evalPayload.error) {
      // Host-side failure (evaluator crash, timeout, or a round-limit
      // refusal) -- NOT a real test failure. Bail out instead of building a
      // retry prompt from empty test output (same rationale as the Claude
      // loop, errors-timeouts.md Finding 1).
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
    if (evalPayload.resolved === true) {
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

    let roundText: string | null = null;
    try {
      const roundOutcome = await args.runRound(prompt, round);
      totalUsage = addUsage(totalUsage, roundOutcome.usage);
      roundText = roundOutcome.text;
    } catch (err) {
      return {
        finalResult: currentOutput,
        roundsUsed,
        attempt1EvalSummary,
        captured: false,
        note: `polyglot_test_feedback: OpenAI round ${round} threw: ${err instanceof Error ? err.message : String(err)}`,
        tokenUsage: totalUsage,
        finalEvalMatchesOutput: false,
      };
    }
    if (typeof roundText === 'string' && roundText.trim()) {
      currentOutput = roundText.trim();
    }
    roundsUsed += 1;
  }

  // Tries are exhausted after the last edit turn. Evaluate the FINAL
  // on-disk state once more so execution_phase.py's cache-reuse can skip
  // its own redundant post-hoc evaluate() call — identical rationale to the
  // Claude loop's post-loop barrier check (performance.md Finding 1).
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
