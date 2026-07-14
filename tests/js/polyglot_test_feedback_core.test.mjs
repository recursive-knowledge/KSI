/**
 * Behavioral unit tests for the pure control-flow decisions of the polyglot
 * test-feedback retry loop, extracted from `polyglot_test_feedback.ts` into
 * the provider-agnostic core so they can be exercised directly (no
 * `@anthropic-ai/claude-agent-sdk` `query()` to drive):
 *
 *   - retryRoundCount(triesRemaining)      -- the round-count bound
 *   - classifyBarrierEval(payload)         -- error / resolved / continue
 *   - selectRoundUsage(resultU, perTurnU)  -- result-aggregate vs per-turn pick
 *   - shouldBailOnRoundError(received)     -- the post-break catch guard
 *
 * `polyglot_test_feedback_core.ts` is deliberately SDK-free (its only import,
 * a `UsageDelta` type, is erased), so — unlike `polyglot_test_feedback.ts` —
 * it can be tsx-loaded on a clean checkout. Same tsx-spawn pattern as
 * tests/js/polyglot_test_feedback_openai.test.mjs: run the real TS logic under
 * tsx, print a JSON result table, and assert on it here as node:test cases.
 * The Claude loop CALLS these functions (source-pinned in
 * polyglot_test_feedback.test.mjs), so the real path and these tests exercise
 * the same logic.
 */
import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const tsxAvailable = fs.existsSync(tsxBin);

const coreTs = path.join(
  repoRoot, 'runtime_runner', 'agent-runner', 'src', 'polyglot_test_feedback_core.ts',
);

function runCases() {
  const script = `
import {
  retryRoundCount,
  classifyBarrierEval,
  selectRoundUsage,
  shouldBailOnRoundError,
} from ${JSON.stringify(coreTs)};

const u = (i, o, cc, cr) => ({
  input_tokens: i, output_tokens: o,
  cache_creation_input_tokens: cc, cache_read_input_tokens: cr,
});

const out = {};

// retryRoundCount: N tries buy N-1 retry rounds, floored at 0.
out.roundCount = {
  neg1: retryRoundCount(-1),
  zero: retryRoundCount(0),
  one: retryRoundCount(1),
  two: retryRoundCount(2),
  three: retryRoundCount(3),
};

// classifyBarrierEval: error > resolved > continue (order-sensitive).
out.classify = {
  error: classifyBarrierEval({ error: 'RuntimeError: boom' }),
  resolved: classifyBarrierEval({ resolved: true, native_score: 1.0 }),
  fail: classifyBarrierEval({ resolved: false, native_score: 0.0 }),
  errorWinsOverResolved: classifyBarrierEval({ error: 'boom', resolved: true }),
  emptyErrorIsContinue: classifyBarrierEval({ error: '', resolved: false }),
  nonStringErrorIsContinue: classifyBarrierEval({ error: 123, resolved: false }),
  resolvedMustBeStrictTrue: classifyBarrierEval({ resolved: 'true' }),
  emptyPayload: classifyBarrierEval({}),
};

// selectRoundUsage: prefer the result aggregate when any field is nonzero,
// else fall back to the summed per-turn deltas (never both).
out.usage = {
  resultNonzero: selectRoundUsage(u(10, 5, 0, 2), u(999, 999, 999, 999)),
  resultAllZeroFallsBack: selectRoundUsage(u(0, 0, 0, 0), u(7, 3, 1, 0)),
  onlyCacheReadCountsAsNonzero: selectRoundUsage(u(0, 0, 0, 4), u(7, 3, 1, 0)),
  onlyCacheCreationCountsAsNonzero: selectRoundUsage(u(0, 0, 9, 0), u(7, 3, 1, 0)),
  bothZero: selectRoundUsage(u(0, 0, 0, 0), u(0, 0, 0, 0)),
};

// shouldBailOnRoundError: bail only when no result was captured this round.
out.bail = {
  noResult: shouldBailOnRoundError(false),
  gotResult: shouldBailOnRoundError(true),
};

process.stdout.write(JSON.stringify(out));
`;
  return spawnSync(tsxBin, ['--eval', script, '--conditions=node'], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, NO_COLOR: '1', FORCE_COLOR: '0' },
  });
}

describe('polyglot_test_feedback_core — pure control-flow decisions', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/', () => {});
    return;
  }

  const res = runCases();
  it('tsx evaluated the real core module', () => {
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
  });
  if (res.status !== 0) return;
  const out = JSON.parse(res.stdout);

  it('retryRoundCount: N tries buy N-1 retry rounds, floored at 0', () => {
    assert.equal(out.roundCount.neg1, 0, 'negative triesRemaining runs no rounds');
    assert.equal(out.roundCount.zero, 0, 'triesRemaining=0 runs no rounds');
    assert.equal(out.roundCount.one, 0, 'triesRemaining=1 (attempt only) runs no retry rounds');
    assert.equal(out.roundCount.two, 1, 'triesRemaining=2 runs exactly one retry round');
    assert.equal(out.roundCount.three, 2, 'triesRemaining=3 runs two retry rounds');
  });

  it('classifyBarrierEval: error > resolved > continue, order-sensitive', () => {
    assert.equal(out.classify.error, 'error');
    assert.equal(out.classify.resolved, 'resolved');
    assert.equal(out.classify.fail, 'continue');
    // A host error takes precedence even when the payload also says resolved.
    assert.equal(out.classify.errorWinsOverResolved, 'error');
  });

  it('classifyBarrierEval: only a non-empty string error counts as an error', () => {
    assert.equal(out.classify.emptyErrorIsContinue, 'continue');
    assert.equal(out.classify.nonStringErrorIsContinue, 'continue');
  });

  it('classifyBarrierEval: resolved must be strictly boolean true', () => {
    assert.equal(out.classify.resolvedMustBeStrictTrue, 'continue', "'true' string is not resolved");
    assert.equal(out.classify.emptyPayload, 'continue');
  });

  it('selectRoundUsage: prefers the result aggregate when any field is nonzero', () => {
    assert.deepEqual(out.usage.resultNonzero, {
      input_tokens: 10, output_tokens: 5, cache_creation_input_tokens: 0, cache_read_input_tokens: 2,
    });
  });

  it('selectRoundUsage: falls back to per-turn sum when the result aggregate is all-zero', () => {
    assert.deepEqual(out.usage.resultAllZeroFallsBack, {
      input_tokens: 7, output_tokens: 3, cache_creation_input_tokens: 1, cache_read_input_tokens: 0,
    });
  });

  it('selectRoundUsage: a nonzero cache field alone keeps the result aggregate', () => {
    assert.deepEqual(out.usage.onlyCacheReadCountsAsNonzero, {
      input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 4,
    });
    assert.deepEqual(out.usage.onlyCacheCreationCountsAsNonzero, {
      input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 9, cache_read_input_tokens: 0,
    });
  });

  it('selectRoundUsage: both all-zero yields the (zero) per-turn value', () => {
    assert.deepEqual(out.usage.bothZero, {
      input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0,
    });
  });

  it('shouldBailOnRoundError: bail iff no result was captured this round', () => {
    assert.equal(out.bail.noResult, true, 'no captured result -> bail');
    assert.equal(out.bail.gotResult, false, 'captured result before the throw -> keep it, do not bail');
  });
});
