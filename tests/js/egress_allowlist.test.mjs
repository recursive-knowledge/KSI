import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const MOD = path.join(repoRoot, 'runtime_runner', 'src', 'egress_allowlist.ts');

function derive(env) {
  const src = `
    import { deriveEgressAllowlist } from ${JSON.stringify(MOD)};
    console.log(JSON.stringify(deriveEgressAllowlist(${JSON.stringify(env)})));
  `;
  const res = spawnSync('npx', ['tsx', '--eval', src], {
    cwd: repoRoot, encoding: 'utf-8', env: { ...process.env },
  });
  assert.equal(res.status, 0, res.stderr);
  return JSON.parse(res.stdout.trim());
}

describe('deriveEgressAllowlist', () => {
  it('defaults to anthropic when provider unset', () => {
    assert.deepEqual(derive({}), ['api.anthropic.com']);
  });
  it('uses openai host for openai provider', () => {
    assert.deepEqual(derive({ MODEL_PROVIDER: 'openai' }), ['api.openai.com']);
  });
  it('adds the host of a base-URL override', () => {
    const out = derive({ MODEL_PROVIDER: 'anthropic', ANTHROPIC_BASE_URL: 'https://proxy.corp:8443/v1' });
    assert.ok(out.includes('api.anthropic.com'));
    assert.ok(out.includes('proxy.corp'));
  });
  it('appends KCSI_EGRESS_ALLOW hosts', () => {
    const out = derive({ MODEL_PROVIDER: 'openai', KCSI_EGRESS_ALLOW: 'bedrock-runtime.us-east-1.amazonaws.com, sts.amazonaws.com' });
    assert.ok(out.includes('api.openai.com'));
    assert.ok(out.includes('bedrock-runtime.us-east-1.amazonaws.com'));
    assert.ok(out.includes('sts.amazonaws.com'));
  });
  it('lowercases and dedupes', () => {
    const out = derive({ MODEL_PROVIDER: 'anthropic', KCSI_EGRESS_ALLOW: 'API.Anthropic.com' });
    assert.deepEqual(out, ['api.anthropic.com']);
  });
  it('ignores a malformed base-URL override', () => {
    const out = derive({ MODEL_PROVIDER: 'anthropic', ANTHROPIC_BASE_URL: 'not-a-url' });
    assert.deepEqual(out, ['api.anthropic.com']);
  });
});
