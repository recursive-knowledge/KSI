/**
 * Provider-agnostic core of the polyglot test-feedback retry loop ("Aider
 * protocol"): barrier name, test-output capping, the Aider-style retry
 * prompt, eval-summary shaping, and token-usage arithmetic.
 *
 * Extracted from `polyglot_test_feedback.ts` (the Claude Agent SDK
 * round-runner) so the OpenAI round-runner
 * (`polyglot_test_feedback_openai.ts`) can share it. This module must stay
 * free of provider SDK imports — `@anthropic-ai/claude-agent-sdk` and
 * `@openai/agents` are only installed inside the Docker image, and the
 * OpenAI loop's behavioral tests (tests/js/polyglot_test_feedback_openai.test.mjs)
 * tsx-load this chain on a clean checkout.
 */
import { UsageDelta } from './usage.js';

export const POLYGLOT_TEST_FEEDBACK_BARRIER_NAME = 'polyglot_test_feedback';

/** Cap `text` to its last `maxLines` lines (or fewer if shorter). */
export function extractCappedTail(text: string, maxLines: number): string {
  const lines = text.split('\n');
  if (lines.length <= maxLines) return text;
  return lines.slice(lines.length - maxLines).join('\n');
}

/** Aider-style framing: "the tests are correct, don't change them, fix the code". */
export function buildTestFeedbackPrompt(args: { testOutput: string; fileList: string }): string {
  return [
    'See the testing errors below.',
    '',
    args.testOutput,
    '',
    "The tests are correct, don't try and change them.",
    `Fix the code in ${args.fileList} to resolve the errors.`,
  ].join('\n');
}

export function summarizeEvalResult(payload: Record<string, unknown> | null): Record<string, unknown> | null {
  if (!payload) return null;
  return {
    native_score: payload.native_score ?? null,
    resolved: payload.resolved ?? null,
    status: payload.status ?? null,
    test_exit_code: payload.test_exit_code ?? null,
  };
}

export function extractRawTestOutput(payload: Record<string, unknown> | null): string {
  if (!payload) return '';
  const stdout = typeof payload.test_stdout_tail === 'string' ? payload.test_stdout_tail : '';
  const stderr = typeof payload.test_stderr_tail === 'string' ? payload.test_stderr_tail : '';
  return [stdout, stderr].filter(Boolean).join('\n');
}

export function zeroUsage(): UsageDelta {
  return { input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 };
}

export function addUsage(a: UsageDelta, b: UsageDelta): UsageDelta {
  return {
    input_tokens: a.input_tokens + b.input_tokens,
    output_tokens: a.output_tokens + b.output_tokens,
    cache_creation_input_tokens: a.cache_creation_input_tokens + b.cache_creation_input_tokens,
    cache_read_input_tokens: a.cache_read_input_tokens + b.cache_read_input_tokens,
  };
}

// ---------------------------------------------------------------------------
// Pure control-flow decisions of the retry loop. Extracted from
// `polyglot_test_feedback.ts` (and mirrored by the OpenAI loop) so the
// load-bearing round-count bound, early-exit classification, usage pick, and
// round-error bail can be unit-tested directly without driving a provider
// SDK's `query()`. The loops CALL these so the real path and the tests
// exercise the same logic.
// ---------------------------------------------------------------------------

/**
 * Number of retry rounds the loop runs for a given `triesRemaining` (Aider
 * `--tries`): one submission is the first attempt, so N tries buy N-1 retry
 * rounds. Floored at 0 so a non-positive `triesRemaining` runs no rounds.
 */
export function retryRoundCount(triesRemaining: number): number {
  return Math.max(0, triesRemaining - 1);
}

/**
 * How the loop should react to a (non-null) barrier eval payload:
 *   - `error`    -> host-side failure (evaluator crash, timeout, or a
 *                   round-limit refusal). Bail out; do NOT burn a retry round
 *                   reacting to an infra hiccup (errors-timeouts.md Finding 1).
 *   - `resolved` -> the submission passed. Early-exit with
 *                   finalEvalMatchesOutput=true.
 *   - `continue` -> a genuine test failure. Build a retry prompt and run a
 *                   round.
 * Order matters: an `error` response is a bail even if it also carries
 * `resolved`, mirroring the original inline check order. Callers handle the
 * null/timeout payload (a missing barrier response) before calling this.
 */
export type BarrierEvalDecision = 'error' | 'resolved' | 'continue';
export function classifyBarrierEval(payload: Record<string, unknown>): BarrierEvalDecision {
  if (typeof payload.error === 'string' && payload.error) return 'error';
  if (payload.resolved === true) return 'resolved';
  return 'continue';
}

/**
 * Pick the round's token usage. Assistant messages report a per-turn usage
 * delta; the terminal `result` event reports the turn aggregate. Prefer the
 * result aggregate when it carries any nonzero field, else fall back to the
 * summed per-turn deltas -- summing BOTH would double-count (see usage.ts and
 * phase1_reflection.ts's pickReflectionUsage for the same pattern).
 */
export function selectRoundUsage(resultUsage: UsageDelta, perTurnUsage: UsageDelta): UsageDelta {
  const resultHasUsage = (
    resultUsage.input_tokens || resultUsage.output_tokens
    || resultUsage.cache_creation_input_tokens || resultUsage.cache_read_input_tokens
  );
  return resultHasUsage ? resultUsage : perTurnUsage;
}

/**
 * A round's async iterator can throw during its post-`break` cleanup AFTER
 * the `result` event was already captured (SDK issue #945). Only bail when no
 * result was captured this round; otherwise the throw is spurious and the
 * already-captured result stands.
 */
export function shouldBailOnRoundError(resultReceivedThisRound: boolean): boolean {
  return !resultReceivedThisRound;
}
