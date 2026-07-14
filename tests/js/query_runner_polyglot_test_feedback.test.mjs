/**
 * Regression/wiring test for the polyglot test-feedback retry loop
 * (`runtime_runner/agent-runner/src/polyglot_test_feedback.ts`):
 * it must be wired into the scheduled-task success path in
 * `runtime_runner/agent-runner/src/query_runner.ts`, and the new
 * `ContainerInput.polyglotTestFeedback` / `ContainerOutput.polyglot_test_feedback_meta`
 * / `_token_usage` fields must exist on both copies of `shared_types.ts`
 * (host `runtime_runner/src/` and container `runtime_runner/agent-runner/src/`
 * — the file is deliberately duplicated across the two compilation units).
 *
 * `polyglot_test_feedback.ts` (and, transitively, `query_runner.ts`) imports
 * `query` from `@anthropic-ai/claude-agent-sdk`, which is only installed
 * inside the Docker image / a manually-run `npm install` in
 * `runtime_runner/agent-runner/` — CI's `npm ci` (working directory
 * `runtime_runner/`) never installs it, and neither `dist/` nor
 * `node_modules/` under `runtime_runner/agent-runner/` are committed.
 * `shared_types.ts`'s interfaces are also erased at compile time (TS
 * interfaces produce no runtime JS), so importing them from a built
 * `dist/shared_types.js` would fail even if `dist/` existed. So — following
 * the same source-pin idiom already used by
 * `tests/js/post_loop_late_iterator_error.test.mjs` and
 * `tests/js/polyglot_test_feedback.test.mjs` — this file reads the real
 * `.ts` sources as text and asserts the wiring is present, rather than
 * importing compiled output.
 */
import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');

const HOST_SHARED_TYPES = path.join(repoRoot, 'runtime_runner', 'src', 'shared_types.ts');
const CONTAINER_SHARED_TYPES = path.join(
  repoRoot, 'runtime_runner', 'agent-runner', 'src', 'shared_types.ts',
);
const QUERY_RUNNER_SRC = path.join(
  repoRoot, 'runtime_runner', 'agent-runner', 'src', 'query_runner.ts',
);

describe('shared_types.ts — PolyglotTestFeedbackConfig + ContainerInput/Output wiring', () => {
  for (const [label, srcPath] of [
    ['host copy', HOST_SHARED_TYPES],
    ['container copy', CONTAINER_SHARED_TYPES],
  ]) {
    it(`${label} defines PolyglotTestFeedbackConfig with the fields the retry loop needs`, () => {
      const src = fs.readFileSync(srcPath, 'utf-8');
      assert.match(src, /export interface PolyglotTestFeedbackConfig \{/);
      const body = src.slice(
        src.indexOf('export interface PolyglotTestFeedbackConfig {'),
        src.indexOf('}', src.indexOf('export interface PolyglotTestFeedbackConfig {')),
      );
      for (const field of [
        'enabled: boolean;',
        'agentId: string;',
        'triesRemaining: number;',
        'maxLines: number;',
        'fileList: string;',
        'allowedTools: string[];',
        'maxTurnsPerRound: number;',
      ]) {
        assert.ok(body.includes(field), `PolyglotTestFeedbackConfig missing field: ${field}`);
      }
    });

    it(`${label} ContainerInput carries an optional polyglotTestFeedback config block`, () => {
      const src = fs.readFileSync(srcPath, 'utf-8');
      assert.match(src, /polyglotTestFeedback\?: PolyglotTestFeedbackConfig;/);
    });

    it(`${label} ContainerOutput carries polyglot_test_feedback_meta + _token_usage`, () => {
      const src = fs.readFileSync(srcPath, 'utf-8');
      assert.match(src, /polyglot_test_feedback_meta\?: \{/);
      assert.match(src, /rounds_used: number;/);
      assert.match(src, /attempt_1_eval_summary: Record<string, unknown> \| null;/);
      assert.match(src, /polyglot_test_feedback_token_usage\?: \{/);
    });
  }

  it('host and container copies of shared_types.ts stay byte-identical (documented invariant)', () => {
    const hostSrc = fs.readFileSync(HOST_SHARED_TYPES, 'utf-8');
    const containerSrc = fs.readFileSync(CONTAINER_SHARED_TYPES, 'utf-8');
    assert.equal(hostSrc, containerSrc);
  });
});

describe('query_runner.ts — runPolyglotTestFeedback wired into buildScheduledResultOutcome', () => {
  const src = fs.readFileSync(QUERY_RUNNER_SRC, 'utf-8');

  it('imports runPolyglotTestFeedback from ./polyglot_test_feedback.js', () => {
    assert.match(
      src,
      /import \{ runPolyglotTestFeedback \} from '\.\/polyglot_test_feedback\.js';/,
    );
  });

  it('the polyglot test-feedback block runs after the phase1Reflection block and before the return', () => {
    const phase1Idx = src.indexOf('const phase1Cfg = containerInput.phase1Reflection;');
    const polyglotCfgIdx = src.indexOf('const polyglotTFCfg = containerInput.polyglotTestFeedback;');
    const returnIdx = src.indexOf('return {\n    envelope: {\n      status: \'success\',');
    assert.ok(phase1Idx !== -1, 'phase1Cfg assignment not found');
    assert.ok(polyglotCfgIdx !== -1, 'polyglotTFCfg assignment not found');
    assert.ok(returnIdx !== -1, 'success return block not found');
    assert.ok(
      phase1Idx < polyglotCfgIdx && polyglotCfgIdx < returnIdx,
      'expected ordering: phase1Reflection block, then polyglotTestFeedback block, then the final return',
    );
  });

  it('gates the retry loop on enabled + effectiveResult + triesRemaining > 1', () => {
    assert.match(
      src,
      /if \(polyglotTFCfg && polyglotTFCfg\.enabled && effectiveResult && polyglotTFCfg\.triesRemaining > 1\) \{/,
    );
  });

  it('Finding #2: records a skip note when effectiveResult is falsy, mirroring phase1Reflection', () => {
    // Unlike the pre-fix code (which had no else-branch and silently
    // skipped, unlike the matching phase1Reflection else-branch just
    // above), the polyglot block must record why it was skipped.
    assert.match(
      src,
      /else if \(polyglotTFCfg && polyglotTFCfg\.enabled && polyglotTFCfg\.triesRemaining > 1\) \{/,
    );
    assert.match(src, /polyglot_test_feedback: skipped \(no effectiveResult to retry on\)/);
  });

  it('calls runPolyglotTestFeedback with the resumed session + retry-round config', () => {
    const callIdx = src.indexOf('const outcome = await runPolyglotTestFeedback({');
    assert.ok(callIdx !== -1, 'runPolyglotTestFeedback call not found');
    const callBlock = src.slice(callIdx, src.indexOf('});', callIdx));
    for (const arg of [
      'workspaceDir: CONTAINER_WORKSPACE_ROOT',
      'agentId: polyglotTFCfg.agentId',
      'fileList: polyglotTFCfg.fileList',
      'modelOutput: effectiveResult',
      'triesRemaining: polyglotTFCfg.triesRemaining',
      'maxLines: polyglotTFCfg.maxLines',
      'resumeSessionId: newSessionId',
      'resumeAt: lastAssistantUuid',
      'selectedModel',
      'sdkEnv',
      'allowedTools: polyglotTFCfg.allowedTools',
      'mcpServers:',
      'maxTurnsPerRound: polyglotTFCfg.maxTurnsPerRound',
      'pollTimeoutMs: polyglotTFCfg.evalResultPollTimeoutMs',
    ]) {
      assert.ok(callBlock.includes(arg), `runPolyglotTestFeedback call missing arg: ${arg}`);
    }
  });

  it('the retry loop changes what result gets returned/graded (result: polyglotTFResult, not effectiveResult)', () => {
    assert.match(src, /result: polyglotTFResult,/);
    // effectiveResult must no longer be the field emitted as `result` in the
    // success envelope — the whole point of this feature (unlike
    // phase1_reflection) is that a retry round's edits ARE what gets graded.
    assert.doesNotMatch(src, /status: 'success',\n\s*result: effectiveResult,/);
  });

  it('splices polyglot_test_feedback_meta and _token_usage into the success envelope', () => {
    assert.match(src, /\.\.\.\(polyglotTFMeta \? \{ polyglot_test_feedback_meta: polyglotTFMeta \} : \{\}\),/);
    assert.match(
      src,
      /\.\.\.\(polyglotTFTokenUsage \? \{ polyglot_test_feedback_token_usage: polyglotTFTokenUsage \} : \{\}\),/,
    );
  });
});
