/**
 * Tests for the shared Anthropic-direct transport's retry/backoff policy
 * (`createMessage` in anthropic_direct_transport.ts).
 *
 * Both direct adapters (ARC + forum) route every LLM call through
 * `createMessage`. Previously it was a single `fetch` with no retry, so a
 * transient 429/5xx or a dropped connection failed the whole attempt. The
 * transport now retries 429 + 5xx + network-level errors with exponential
 * backoff, tunable via KCSI_ANTHROPIC_MAX_RETRIES /
 * KCSI_ANTHROPIC_RETRY_BASE_MS (base set to 0 here to keep tests fast).
 *
 * These run the real TS through tsx with a stubbed global fetch so the
 * actual retry control-flow is exercised (not a regex over source text).
 */

import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');

const tsxBin = path.join(
  repoRoot,
  'runtime_runner',
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'tsx.cmd' : 'tsx',
);
const tsxSkip = fs.existsSync(tsxBin)
  ? undefined
  : 'runtime_runner/node_modules/.bin/tsx is not installed';

function runTsxFixture(source) {
  const result = spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    cwd: path.join(repoRoot, 'runtime_runner'),
    encoding: 'utf-8',
    env: { ...process.env, NODE_NO_WARNINGS: '1' },
  });
  assert.equal(
    result.status,
    0,
    `tsx fixture failed with status ${result.status}\n--- stdout ---\n${result.stdout}\n--- stderr ---\n${result.stderr}`,
  );
  return result.stdout.trim();
}

const OK_RESPONSE = `{
  ok: true,
  status: 200,
  headers: { get: () => null },
  async text() { return JSON.stringify({ id: 'r', content: [{ type: 'text', text: 'hi' }], usage: { input_tokens: 1, output_tokens: 1 } }); },
}`;

function errResponse(status) {
  return `{
    ok: false,
    status: ${status},
    headers: { get: () => null },
    async text() { return JSON.stringify({ error: { message: 'boom ${status}' } }); },
  }`;
}

describe('anthropic_direct_transport createMessage — retry/backoff', () => {
  it('retries a 429 then succeeds', { skip: tsxSkip }, () => {
    runTsxFixture(`
      import { strict as assert } from 'node:assert';
      const { createMessage } = await import('./agent-runner/src/anthropic_direct_transport.ts');
      let calls = 0;
      globalThis.fetch = async () => {
        calls += 1;
        return calls === 1 ? ${errResponse(429)} : ${OK_RESPONSE};
      };
      const res = await createMessage(
        { ANTHROPIC_API_KEY: 'k', KCSI_ANTHROPIC_RETRY_BASE_MS: '0' },
        { model: 'm' },
        'test',
      );
      assert.equal(calls, 2, 'should retry once after 429');
      assert.equal(res.id, 'r');
    `);
  });

  it('retries a 503 then succeeds', { skip: tsxSkip }, () => {
    runTsxFixture(`
      import { strict as assert } from 'node:assert';
      const { createMessage } = await import('./agent-runner/src/anthropic_direct_transport.ts');
      let calls = 0;
      globalThis.fetch = async () => {
        calls += 1;
        return calls < 3 ? ${errResponse(503)} : ${OK_RESPONSE};
      };
      const res = await createMessage(
        { ANTHROPIC_API_KEY: 'k', KCSI_ANTHROPIC_RETRY_BASE_MS: '0' },
        { model: 'm' },
        'test',
      );
      assert.equal(calls, 3);
      assert.equal(res.id, 'r');
    `);
  });

  it('retries a network-level fetch rejection then succeeds', { skip: tsxSkip }, () => {
    runTsxFixture(`
      import { strict as assert } from 'node:assert';
      const { createMessage } = await import('./agent-runner/src/anthropic_direct_transport.ts');
      let calls = 0;
      globalThis.fetch = async () => {
        calls += 1;
        if (calls === 1) throw new Error('ECONNRESET');
        return ${OK_RESPONSE};
      };
      const res = await createMessage(
        { ANTHROPIC_API_KEY: 'k', KCSI_ANTHROPIC_RETRY_BASE_MS: '0' },
        { model: 'm' },
        'test',
      );
      assert.equal(calls, 2);
      assert.equal(res.id, 'r');
    `);
  });

  it('does NOT retry a non-retryable 400', { skip: tsxSkip }, () => {
    runTsxFixture(`
      import { strict as assert } from 'node:assert';
      const { createMessage } = await import('./agent-runner/src/anthropic_direct_transport.ts');
      let calls = 0;
      globalThis.fetch = async () => { calls += 1; return ${errResponse(400)}; };
      await assert.rejects(
        createMessage({ ANTHROPIC_API_KEY: 'k', KCSI_ANTHROPIC_RETRY_BASE_MS: '0' }, { model: 'm' }, 'test'),
        /Anthropic API error 400: boom 400/,
      );
      assert.equal(calls, 1, '4xx (non-429) must not be retried');
    `);
  });

  it('exhausts retries on persistent 500 and throws', { skip: tsxSkip }, () => {
    runTsxFixture(`
      import { strict as assert } from 'node:assert';
      const { createMessage } = await import('./agent-runner/src/anthropic_direct_transport.ts');
      let calls = 0;
      globalThis.fetch = async () => { calls += 1; return ${errResponse(500)}; };
      await assert.rejects(
        createMessage(
          { ANTHROPIC_API_KEY: 'k', KCSI_ANTHROPIC_RETRY_BASE_MS: '0', KCSI_ANTHROPIC_MAX_RETRIES: '2' },
          { model: 'm' },
          'test',
        ),
        /Anthropic API error 500/,
      );
      assert.equal(calls, 3, 'maxRetries=2 means 1 initial + 2 retries = 3 calls');
    `);
  });

  it('disables retries when KCSI_ANTHROPIC_MAX_RETRIES=0', { skip: tsxSkip }, () => {
    runTsxFixture(`
      import { strict as assert } from 'node:assert';
      const { createMessage } = await import('./agent-runner/src/anthropic_direct_transport.ts');
      let calls = 0;
      globalThis.fetch = async () => { calls += 1; return ${errResponse(429)}; };
      await assert.rejects(
        createMessage(
          { ANTHROPIC_API_KEY: 'k', KCSI_ANTHROPIC_MAX_RETRIES: '0' },
          { model: 'm' },
          'test',
        ),
        /Anthropic API error 429/,
      );
      assert.equal(calls, 1);
    `);
  });

  it('requires ANTHROPIC_API_KEY with the adapter label', { skip: tsxSkip }, () => {
    runTsxFixture(`
      import { strict as assert } from 'node:assert';
      const { createMessage } = await import('./agent-runner/src/anthropic_direct_transport.ts');
      await assert.rejects(
        createMessage({}, { model: 'm' }, 'Anthropic direct ARC'),
        /ANTHROPIC_API_KEY is required for Anthropic direct ARC runs/,
      );
    `);
  });
});
