/**
 * Coverage for the OpenAI-provider polyglot test-feedback retry loop
 * (`runtime_runner/agent-runner/src/polyglot_test_feedback_openai.ts`).
 *
 * Background: the Aider-protocol retry loop (PR #1032) only ran on the
 * Claude Agent SDK path (`query_runner.ts` → `runPolyglotTestFeedback`).
 * The OpenAI branch in `index.ts` called `runOpenAIQuery(...)`, wrote its
 * envelope, and exited — the loop was unreachable and no
 * `polyglot_test_feedback_meta` / `_token_usage` fields were ever emitted
 * under `MODEL_PROVIDER=openai`.
 *
 * Unlike `polyglot_test_feedback.ts` (whose module-level
 * `@anthropic-ai/claude-agent-sdk` import forces source-pin-only testing),
 * the OpenAI loop module is deliberately SDK-free: the provider-specific
 * round run is dependency-injected (`runRound`), so this file can exercise
 * the REAL loop behaviorally via tsx (same pattern as
 * web_tools_gating.test.mjs), with this test acting as the fake host on the
 * barrier-file protocol. Wiring into `openai.ts` / `index.ts` is pinned in
 * source text below (those files import the OpenAI SDK, so they cannot be
 * tsx-imported on a clean checkout).
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

const agentRunnerSrc = path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src');
const openaiLoopTs = path.join(agentRunnerSrc, 'polyglot_test_feedback_openai.ts');
const coreTs = path.join(agentRunnerSrc, 'polyglot_test_feedback_core.ts');
const openaiTs = path.join(agentRunnerSrc, 'openai.ts');
const indexTs = path.join(agentRunnerSrc, 'index.ts');

// ---------------------------------------------------------------------------
// Behavioral: run the real loop under tsx with a fake barrier host + a
// stubbed runRound.
// ---------------------------------------------------------------------------

function runScenarios() {
  const script = `
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { runOpenAIPolyglotTestFeedback } from ${JSON.stringify(openaiLoopTs)};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Fake host: answer each barrier sentinel in order with the given payloads.
 * Mirrors the host BarrierWatcher contract: consume the sentinel, then
 * atomically write the response. Returns the sentinel payloads it consumed.
 */
const FAKE_HOST_DEADLINE_MS = 10000;
async function fakeHost(workspaceDir, agentId, responses) {
  const sentinel = path.join(workspaceDir, \`.barrier.polyglot_test_feedback.\${agentId}.ready\`);
  const response = path.join(workspaceDir, \`.barrier.polyglot_test_feedback.\${agentId}.response\`);
  const seen = [];
  for (let i = 0; i < responses.length; i += 1) {
    // Hard deadline: if a loop-shape regression leaves sentinel i unwritten,
    // throw (rejecting the scenario) instead of spinning forever and keeping
    // the tsx child alive -> a crisp failure, not a CI-timeout hang.
    const start = Date.now();
    while (!fs.existsSync(sentinel)) {
      if (Date.now() - start > FAKE_HOST_DEADLINE_MS) {
        throw new Error(\`fakeHost: sentinel \${i} not written within \${FAKE_HOST_DEADLINE_MS}ms\`);
      }
      await sleep(20);
    }
    seen.push(JSON.parse(fs.readFileSync(sentinel, 'utf-8')));
    fs.unlinkSync(sentinel);
    fs.writeFileSync(response + '.tmp', JSON.stringify(responses[i]));
    fs.renameSync(response + '.tmp', response);
  }
  return seen;
}

async function scenario({ responses, triesRemaining, maxLines, rounds, pollTimeoutMs, throwOnRound }) {
  const workspaceDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tfoai-'));
  const prompts = [];
  let roundIdx = 0;
  const runRound = async (prompt, round) => {
    prompts.push({ prompt, round });
    if (throwOnRound !== undefined && round === throwOnRound) {
      throw new Error('boom');
    }
    const r = rounds[roundIdx] || { text: 'unexpected-extra-round', usage: { input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 } };
    roundIdx += 1;
    return r;
  };
  const hostPromise = fakeHost(workspaceDir, 'agent-1', responses);
  // Surface a fakeHost deadline rejection as a scenario failure: race the loop
  // against a guard that ONLY ever rejects (on the host's hard deadline). When
  // the host resolves normally the guard never settles, so the real outcome
  // wins the race.
  const hostGuard = hostPromise.then(
    () => new Promise(() => {}),
    (err) => Promise.reject(err),
  );
  const outcome = await Promise.race([
    runOpenAIPolyglotTestFeedback({
      workspaceDir,
      agentId: 'agent-1',
      fileList: 'bowling.py',
      modelOutput: 'attempt-1 answer',
      triesRemaining,
      maxLines,
      pollTimeoutMs,
      runRound,
      logger: () => {},
    }),
    hostGuard,
  ]);
  // The loop has returned, so the host has consumed all its responses (or is
  // about to hit its deadline and reject). Awaiting it here also propagates a
  // late deadline rejection rather than hanging the child on a pending timer.
  const sentinels = await hostPromise;
  fs.rmSync(workspaceDir, { recursive: true, force: true });
  return { outcome, prompts, sentinels };
}

// Fixtures mirror the real PolyglotHarnessEvaluator payload shape: it emits
// status:'ok' for BOTH a pass and a fail (src/ksi/benchmarks/polyglot_harness.py
// ~L657 — the pass/fail verdict lives in resolved/native_score/test_exit_code,
// not status). The evaluator-error response is {status:'evaluator_error', error}
// (src/ksi/runtime/container_host.py ~L1160).
const pass = { resolved: true, native_score: 1.0, status: 'ok', test_exit_code: 0 };
const fail = {
  resolved: false, native_score: 0.0, status: 'ok', test_exit_code: 1,
  test_stdout_tail: 'l1\\nl2\\nl3\\nl4\\nl5',
  test_stderr_tail: 'E: assertion failed',
};
const usage1 = { input_tokens: 10, output_tokens: 5, cache_creation_input_tokens: 0, cache_read_input_tokens: 2 };
const usage2 = { input_tokens: 7, output_tokens: 3, cache_creation_input_tokens: 1, cache_read_input_tokens: 0 };

// tsx --eval compiles to CJS (no top-level await): run inside an async IIFE.
(async () => {
const out = {};

// S1: attempt-1 already passes -> zero rounds, eval matches output.
out.s1 = await scenario({ responses: [pass], triesRemaining: 2, maxLines: 3, rounds: [] });

// S2: attempt-1 fails, one retry round, final post-loop eval passes.
out.s2 = await scenario({
  responses: [fail, pass],
  triesRemaining: 2,
  maxLines: 3,
  rounds: [{ text: 'round-1 answer', usage: usage1 }],
});

// S3: host reports an evaluator error on round 0 -> bail, no retry burned.
// Production shape: {status:'evaluator_error', error:'<Type>: <msg>'}.
out.s3 = await scenario({
  responses: [{ status: 'evaluator_error', error: 'RuntimeError: kaboom' }],
  triesRemaining: 2,
  maxLines: 3,
  rounds: [],
});

// S4: host never answers -> graceful timeout.
out.s4 = await scenario({ responses: [], triesRemaining: 2, maxLines: 3, rounds: [], pollTimeoutMs: 300 });

// S5: retry ran, tries exhausted, final eval still failing -> the final
// eval nonetheless reflects the final on-disk state.
out.s5 = await scenario({
  responses: [fail, fail],
  triesRemaining: 2,
  maxLines: 3,
  rounds: [{ text: 'round-1 answer', usage: usage1 }],
});

// S6: the provider round itself throws -> bail with a note.
out.s6 = await scenario({ responses: [fail], triesRemaining: 2, maxLines: 3, rounds: [], throwOnRound: 0 });

// S7: two retry rounds (tries=3), usage accumulates, output chains.
out.s7 = await scenario({
  responses: [fail, fail, pass],
  triesRemaining: 3,
  maxLines: 3,
  rounds: [
    { text: 'round-1 answer', usage: usage1 },
    { text: 'round-2 answer', usage: usage2 },
  ],
});

process.stdout.write(JSON.stringify(out));
})().catch((err) => { console.error(err); process.exit(1); });
`;
  return spawnSync(tsxBin, ['--eval', script, '--conditions=node'], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, NO_COLOR: '1', FORCE_COLOR: '0' },
  });
}

describe('runOpenAIPolyglotTestFeedback — behavioral (fake barrier host + stub rounds)', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/');
    return;
  }

  const res = runScenarios();
  it('tsx evaluated the real module', () => {
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
  });
  if (res.status !== 0) return;
  const out = JSON.parse(res.stdout);

  it('S1: passing attempt 1 exits with zero rounds and final_eval_matches_output=true', () => {
    const { outcome, prompts, sentinels } = out.s1;
    assert.equal(outcome.roundsUsed, 0);
    assert.equal(outcome.captured, true);
    assert.equal(outcome.finalEvalMatchesOutput, true);
    assert.equal(outcome.finalResult, 'attempt-1 answer');
    assert.equal(prompts.length, 0);
    assert.deepEqual(outcome.attempt1EvalSummary, {
      native_score: 1, resolved: true, status: 'ok', test_exit_code: 0,
    });
    // Sentinel carries the schema + live model output for the host evaluator.
    assert.equal(sentinels.length, 1);
    assert.equal(sentinels[0].schema, 'polyglot_test_feedback.v1');
    assert.equal(sentinels[0].agent_id, 'agent-1');
    assert.equal(sentinels[0].model_output, 'attempt-1 answer');
  });

  it('S2: failing attempt 1 runs one retry round with the capped Aider-style prompt', () => {
    const { outcome, prompts } = out.s2;
    assert.equal(prompts.length, 1);
    assert.equal(prompts[0].round, 0);
    // stdout tail (5 lines) + stderr tail joined = 6 lines, capped to last 3.
    assert.ok(prompts[0].prompt.includes('l4\nl5\nE: assertion failed'), `prompt was: ${prompts[0].prompt}`);
    assert.ok(!prompts[0].prompt.includes('l1'), 'capped prompt must drop lines above maxLines');
    assert.match(prompts[0].prompt, /tests are correct/i);
    assert.match(prompts[0].prompt, /bowling\.py/);
    assert.equal(outcome.roundsUsed, 1);
    assert.equal(outcome.captured, true);
    assert.equal(outcome.finalEvalMatchesOutput, true);
    assert.equal(outcome.finalResult, 'round-1 answer');
    assert.deepEqual(outcome.tokenUsage, {
      input_tokens: 10, output_tokens: 5, cache_creation_input_tokens: 0, cache_read_input_tokens: 2,
    });
    assert.deepEqual(outcome.attempt1EvalSummary, {
      native_score: 0, resolved: false, status: 'ok', test_exit_code: 1,
    });
  });

  it('S3: a host-side evaluator error bails out without burning a retry round', () => {
    const { outcome, prompts } = out.s3;
    assert.equal(prompts.length, 0);
    assert.equal(outcome.roundsUsed, 0);
    assert.equal(outcome.captured, false);
    assert.equal(outcome.finalEvalMatchesOutput, false);
    assert.match(outcome.note, /host reported an error on round 0/);
    assert.equal(outcome.finalResult, 'attempt-1 answer');
  });

  it('S4: a barrier timeout degrades gracefully with a note', () => {
    const { outcome, prompts } = out.s4;
    assert.equal(prompts.length, 0);
    assert.equal(outcome.captured, false);
    assert.equal(outcome.finalEvalMatchesOutput, false);
    assert.match(outcome.note, /host barrier response not received within 300ms/);
    assert.equal(outcome.finalResult, 'attempt-1 answer');
  });

  it('S5: exhausted tries with a still-failing final eval keeps final_eval_matches_output=true', () => {
    const { outcome } = out.s5;
    assert.equal(outcome.roundsUsed, 1);
    assert.equal(outcome.captured, true);
    // The final barrier round evaluated the LAST edit's on-disk state; it
    // "matches output" even though the tests still fail.
    assert.equal(outcome.finalEvalMatchesOutput, true);
    assert.equal(outcome.finalResult, 'round-1 answer');
  });

  it('S6: a thrown provider round bails with a diagnostic note', () => {
    const { outcome } = out.s6;
    assert.equal(outcome.roundsUsed, 0);
    assert.equal(outcome.captured, false);
    assert.equal(outcome.finalEvalMatchesOutput, false);
    assert.match(outcome.note, /round 0 threw: boom/);
    assert.equal(outcome.finalResult, 'attempt-1 answer');
  });

  it('S7: multiple retry rounds accumulate usage and chain model output through sentinels', () => {
    const { outcome, prompts, sentinels } = out.s7;
    assert.equal(prompts.length, 2);
    assert.equal(outcome.roundsUsed, 2);
    assert.equal(outcome.captured, true);
    assert.equal(outcome.finalEvalMatchesOutput, true);
    assert.equal(outcome.finalResult, 'round-2 answer');
    assert.deepEqual(outcome.tokenUsage, {
      input_tokens: 17, output_tokens: 8, cache_creation_input_tokens: 1, cache_read_input_tokens: 2,
    });
    // Sentinel N carries the output as of that barrier point.
    assert.equal(sentinels.length, 3);
    assert.equal(sentinels[0].model_output, 'attempt-1 answer');
    assert.equal(sentinels[1].model_output, 'round-1 answer');
    assert.equal(sentinels[2].model_output, 'round-2 answer');
  });
});

// ---------------------------------------------------------------------------
// Source pins: the loop module stays SDK-free, and the OpenAI adapter +
// index.ts envelope actually wire it in (openai.ts imports @openai/agents,
// so it cannot be tsx-imported on a clean checkout — same constraint as
// query_runner_polyglot_test_feedback.test.mjs documents for the Claude path).
// ---------------------------------------------------------------------------

describe('polyglot_test_feedback_openai.ts — module constraints', () => {
  const src = fs.readFileSync(openaiLoopTs, 'utf-8');

  it('imports NO provider SDK (must stay tsx-loadable without @openai/agents or the Claude SDK)', () => {
    assert.doesNotMatch(src, /from '@openai\/agents'/);
    assert.doesNotMatch(src, /from '@anthropic-ai\/claude-agent-sdk'/);
  });

  it('shares the provider-agnostic core (barrier name, capping, prompt, eval summary)', () => {
    assert.match(src, /from '\.\/polyglot_test_feedback_core\.js'/);
    assert.match(src, /extractCappedTail/);
    assert.match(src, /buildTestFeedbackPrompt/);
    assert.match(src, /POLYGLOT_TEST_FEEDBACK_BARRIER_NAME/);
  });
});

describe('openai.ts — retry loop wired into runOpenAIQuery', () => {
  const src = fs.readFileSync(openaiTs, 'utf-8');

  it('imports runOpenAIPolyglotTestFeedback from ./polyglot_test_feedback_openai.js', () => {
    assert.match(
      src,
      /import \{ runOpenAIPolyglotTestFeedback \} from '\.\/polyglot_test_feedback_openai\.js';/,
    );
  });

  it('gates on enabled + non-empty result text + triesRemaining > 1 (mirrors query_runner.ts)', () => {
    assert.match(
      src,
      /polyglotTFCfg && polyglotTFCfg\.enabled && resultText\.trim\(\) && polyglotTFCfg\.triesRemaining > 1/,
    );
  });

  it('records a skip note when there is no result text, mirroring the Claude path', () => {
    assert.match(src, /polyglot_test_feedback: skipped \(no effectiveResult to retry on\)/);
  });

  it('the retry loop changes the returned resultText (the graded output)', () => {
    assert.match(src, /resultText: polyglotTFResult/);
  });

  it('salvages MaxTurnsExceededError inside a retry round instead of aborting the loop', () => {
    const loopIdx = src.indexOf('runOpenAIPolyglotTestFeedback({');
    assert.ok(loopIdx !== -1, 'runOpenAIPolyglotTestFeedback call not found');
    // NOTE: a `});`-bounded slice truncates early (template literals inside
    // the call contain `});`), so bound the block by a generous fixed window.
    const callBlock = src.slice(loopIdx, loopIdx + 4000);
    assert.match(callBlock, /isMaxTurnsErr\(/);
    assert.match(callBlock, /salvageResultFromState\(/);
  });

  it('chains the Responses-API conversation via previousResponseId across rounds', () => {
    // Behavioral coverage stops at the injected runRound seam (openai.ts imports
    // @openai/agents, so it is not tsx-loadable). Pin the previousResponseId
    // chaining directly: dropping either assignment would silently restart the
    // conversation each round yet pass every other pin and full CI.
    const loopIdx = src.indexOf('runOpenAIPolyglotTestFeedback({');
    assert.ok(loopIdx !== -1, 'runOpenAIPolyglotTestFeedback call not found');
    const callBlock = src.slice(loopIdx, loopIdx + 4000);
    // Each round consumes the running currentResponseId as previousResponseId.
    assert.ok(
      callBlock.includes('previousResponseId: currentResponseId || undefined,'),
      'round options must chain previousResponseId from currentResponseId',
    );
    // The MaxTurns salvage path must advance currentResponseId...
    assert.ok(
      callBlock.includes('if (salvage.lastResponseId) currentResponseId = salvage.lastResponseId;'),
      'salvage path must chain currentResponseId = salvage.lastResponseId',
    );
    // ...and so must the normal (non-salvage) round result.
    assert.ok(
      callBlock.includes('if (roundResult?.lastResponseId) currentResponseId = roundResult.lastResponseId;'),
      'normal path must chain currentResponseId = roundResult.lastResponseId',
    );
  });

  it('passes the host config through (agentId, fileList, tries, maxLines, poll timeout)', () => {
    const loopIdx = src.indexOf('runOpenAIPolyglotTestFeedback({');
    const callBlock = src.slice(loopIdx, loopIdx + 4000);
    for (const arg of [
      'agentId: polyglotTFCfg.agentId',
      'fileList: polyglotTFCfg.fileList',
      'triesRemaining: polyglotTFCfg.triesRemaining',
      'maxLines: polyglotTFCfg.maxLines',
      'pollTimeoutMs: polyglotTFCfg.evalResultPollTimeoutMs',
    ]) {
      assert.ok(callBlock.includes(arg), `runOpenAIPolyglotTestFeedback call missing arg: ${arg}`);
    }
  });

  it('bounds each retry round with the config maxTurnsPerRound', () => {
    assert.match(src, /maxTurns: polyglotTFCfg\.maxTurnsPerRound/);
  });
});

describe('index.ts — OpenAI envelope carries polyglot_test_feedback fields', () => {
  const src = fs.readFileSync(indexTs, 'utf-8');

  it('splices polyglot_test_feedback_meta and _token_usage into the OpenAI writeOutput envelope', () => {
    assert.match(
      src,
      /\.\.\.\(queryResult\.polyglot_test_feedback_meta\s*\?\s*\{ polyglot_test_feedback_meta: queryResult\.polyglot_test_feedback_meta \}\s*:\s*\{\}\),/,
    );
    assert.match(
      src,
      /\.\.\.\(queryResult\.polyglot_test_feedback_token_usage\s*\?\s*\{ polyglot_test_feedback_token_usage: queryResult\.polyglot_test_feedback_token_usage \}\s*:\s*\{\}\),/,
    );
  });
});

describe('polyglot_test_feedback_core.ts — extracted core stays provider-agnostic', () => {
  const src = fs.readFileSync(coreTs, 'utf-8');

  it('holds the shared helpers and no provider SDK import', () => {
    assert.doesNotMatch(src, /from '@openai\/agents'/);
    assert.doesNotMatch(src, /from '@anthropic-ai\/claude-agent-sdk'/);
    assert.match(src, /export const POLYGLOT_TEST_FEEDBACK_BARRIER_NAME = 'polyglot_test_feedback';/);
    assert.match(src, /export function extractCappedTail\(/);
    assert.match(src, /export function buildTestFeedbackPrompt\(/);
    assert.match(src, /export function summarizeEvalResult\(/);
    assert.match(src, /export function extractRawTestOutput\(/);
  });

  it('the Claude-path module re-exports the historically-public symbols', () => {
    const claudeSrc = fs.readFileSync(
      path.join(agentRunnerSrc, 'polyglot_test_feedback.ts'), 'utf-8',
    );
    assert.match(
      claudeSrc,
      /export \{\n  POLYGLOT_TEST_FEEDBACK_BARRIER_NAME,\n  buildTestFeedbackPrompt,\n  extractCappedTail,\n\} from '\.\/polyglot_test_feedback_core\.js';/,
    );
  });
});
