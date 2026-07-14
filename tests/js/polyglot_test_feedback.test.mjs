/**
 * Unit tests for the pure helper functions in
 * `runtime_runner/agent-runner/src/polyglot_test_feedback.ts`.
 *
 * `runPolyglotTestFeedback` itself drives `@anthropic-ai/claude-agent-sdk`'s
 * `query()` directly (no dependency-injection seam), and this repo has no
 * jest/sinon-style mocking convention for that import (confirmed: no
 * existing tests/js/*.test.mjs mocks/stubs `query` — see
 * `phase1_reflection.ts`, which is structurally identical and also has no
 * test file). Per the task brief, the SDK-touching loop is left to a real
 * end-to-end Docker smoke (Task 11); this file covers only the pure,
 * side-effect-free helpers.
 *
 * Unlike `anthropic_direct_transport.ts` (tested via a real tsx `import()` in
 * anthropic_transport_retry.test.mjs) or `web_tools.ts` (same, in
 * web_tools_gating.test.mjs), this module has a module-level
 * `import { query } from '@anthropic-ai/claude-agent-sdk'`. That package is
 * only installed inside the Docker image / a manually-run `npm install` in
 * `runtime_runner/agent-runner/` — CI's `npm ci` (working-directory
 * `runtime_runner/`) never installs it, and neither `dist/` nor
 * `node_modules/` under `runtime_runner/agent-runner/` are committed. So a
 * direct tsx `import()` of this file throws `ERR_MODULE_NOT_FOUND` on a
 * clean checkout (verified). Instead, this follows the
 * retryable_markers_parity.test.mjs pattern: read the source as text and
 * extract+execute the two pure function bodies directly, exercising the
 * real TS logic without requiring the SDK import to resolve.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsSource = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'polyglot_test_feedback.ts'),
  'utf-8',
);
// The pure helpers were extracted to the provider-agnostic core module so
// the OpenAI round-runner (polyglot_test_feedback_openai.ts) can share them;
// the SDK-driving loop (and its pinned findings below) stays in
// polyglot_test_feedback.ts.
const coreSource = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'polyglot_test_feedback_core.ts'),
  'utf-8',
);

// Extract and execute the REAL function body (pure logic, no types inside).
// `signature` must be the full signature text up to and including the
// function's opening brace (the arg-type object literals in this file's
// signatures also contain `{`, so we can't just search for the next one).
function extractFunctionBody(signature) {
  const start = coreSource.indexOf(signature);
  assert.ok(start >= 0, `${signature} not found in polyglot_test_feedback_core.ts`);
  const braceStart = start + signature.length - 1;
  const end = coreSource.indexOf('\n}', braceStart);
  assert.ok(end > braceStart, `end of body for "${signature}" not found`);
  return coreSource.slice(braceStart + 1, end);
}

// eslint-disable-next-line no-new-func
const extractCappedTail = new Function(
  'text',
  'maxLines',
  extractFunctionBody('export function extractCappedTail(text: string, maxLines: number): string {'),
);
// eslint-disable-next-line no-new-func
const buildTestFeedbackPrompt = new Function(
  'args',
  extractFunctionBody(
    'export function buildTestFeedbackPrompt(args: { testOutput: string; fileList: string }): string {',
  ),
);

test('extractCappedTail caps to the last N lines', () => {
  const stdout = Array.from({ length: 100 }, (_, i) => `line ${i}`).join('\n');
  const capped = extractCappedTail(stdout, 50);
  const lines = capped.split('\n');
  assert.equal(lines.length, 50);
  assert.equal(lines[0], 'line 50');
  assert.equal(lines[49], 'line 99');
});

test('extractCappedTail returns full text when under the cap', () => {
  const stdout = 'line 1\nline 2\nline 3';
  assert.equal(extractCappedTail(stdout, 50), stdout);
});

test('buildTestFeedbackPrompt mirrors Aider framing and includes file list', () => {
  const prompt = buildTestFeedbackPrompt({
    testOutput: 'AssertionError: expected 5 got 6',
    fileList: 'bowling.py',
  });
  assert.match(prompt, /tests are correct/i);
  assert.match(prompt, /bowling\.py/);
  assert.match(prompt, /AssertionError: expected 5 got 6/);
});

// Source-pin regressions for the SDK-driving loop's fixes (the loop itself
// can't be unit-tested without a query() mocking seam this repo doesn't
// have -- see the module docstring above -- so these pin the fix's presence
// in source text rather than exercising it behaviorally).

test('Finding #5: finalEvalMatchesOutput is true only on genuinely-final-state paths', () => {
  // The return type declares the field...
  assert.match(tsSource, /finalEvalMatchesOutput: boolean;/);
  // ...exactly two return statements set it to true: the `resolved === true`
  // early exit inside the loop, and the post-loop final re-evaluation
  // (performance.md Finding 1 fix) that confirms the LAST edit's on-disk
  // state -- both are cases where no further agent turn can run afterward.
  const trueSites = tsSource.match(/finalEvalMatchesOutput: true,/g) || [];
  assert.equal(trueSites.length, 2, 'finalEvalMatchesOutput: true must appear at exactly two return sites');
  const falseSites = tsSource.match(/finalEvalMatchesOutput: false,/g) || [];
  assert.ok(falseSites.length >= 3, 'the barrier-timeout, SDK-error, and evaluator-error returns must all set it false');
});

test('Finding #10: an evaluator_error / host-refused round does not consume a retry turn', () => {
  // A host-side failure response (evaluator crash, or a round-cap refusal)
  // is surfaced as an `error` field; the loop must bail out on it BEFORE
  // building a retry prompt from empty test output (errors-timeouts.md
  // Finding 1) -- it must not fall through to buildTestFeedbackPrompt. The
  // error/resolved/continue classification is now the pure
  // `classifyBarrierEval` (behaviorally unit-tested in
  // polyglot_test_feedback_core.test.mjs); the loop dispatches on its verdict.
  assert.match(tsSource, /classifyBarrierEval\(evalPayload\)/);
  assert.match(tsSource, /evalDecision === 'error'/);
  assert.match(tsSource, /host reported an error on round/);
});

test('Finding #11: the loop performs a final post-loop barrier check after tries are exhausted', () => {
  // performance.md Finding 1: re-evaluating the FINAL on-disk state (not
  // just each round's pre-edit state) lets execution_phase.py's cache-reuse
  // skip its own redundant evaluate() call in the common (retry-happened)
  // case, not just when attempt 1 passed outright.
  const loopEnd = tsSource.indexOf('roundsUsed += 1;\n  }');
  assert.ok(loopEnd >= 0, 'could not locate the end of the retry-loop body');
  const postLoop = tsSource.slice(loopEnd);
  assert.match(postLoop, /writeBarrierSentinel/, 'a sentinel must be written again after the loop');
  assert.match(postLoop, /waitForBarrierFile/, 'the final round must wait for a barrier response');
});

test('Finding #3: a captured result before a post-break throw is not discarded', () => {
  assert.match(tsSource, /resultReceivedThisRound = true/);
  // The bail-vs-keep decision is now the pure `shouldBailOnRoundError`
  // (behaviorally unit-tested in polyglot_test_feedback_core.test.mjs).
  assert.match(tsSource, /if \(shouldBailOnRoundError\(resultReceivedThisRound\)\) \{/);
});

test('Finding #4: round usage is picked (result aggregate vs per-turn sum), not summed', () => {
  assert.match(tsSource, /roundResultUsage = addUsage\(roundResultUsage, delta\)/);
  assert.match(tsSource, /roundPerTurnUsage = addUsage\(roundPerTurnUsage, delta\)/);
  // The pick (result aggregate when nonzero, else per-turn sum) is now the
  // pure `selectRoundUsage` (behaviorally unit-tested in
  // polyglot_test_feedback_core.test.mjs).
  assert.match(tsSource, /selectRoundUsage\(roundResultUsage, roundPerTurnUsage\)/);
  // Must NOT unconditionally sum every message's delta into one running total.
  assert.doesNotMatch(tsSource, /totalUsage = addUsage\(totalUsage, delta\);/);
});
