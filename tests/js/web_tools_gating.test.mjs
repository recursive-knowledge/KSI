/**
 * Issue #666 — BEHAVIORAL coverage of the web-tool gating decision.
 *
 * The companion tests/js/web_tools_default_off.test.mjs pins the gating in
 * source text (and re-implements the truthiness for a smoke check). This file
 * exercises the REAL exported functions in
 * runtime_runner/agent-runner/src/web_tools.ts — the single source of truth
 * that index.ts imports — by spawning tsx, so a refactor that changes behavior
 * (not just source text) is caught and the test cannot drift from a copy.
 *
 * web_tools.ts has NO @anthropic-ai/claude-agent-sdk import (that's why the
 * gating was extracted out of index.ts), so tsx can load it standalone with
 * only runtime_runner deps. When tsx is not installed the test skips, matching
 * the other tsx-dependent tests in this directory.
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
const webToolsTs = path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'web_tools.ts');
const tsxAvailable = fs.existsSync(tsxBin);

// Run buildWebToolGating / isWebToolsAllowed for a matrix of cases inside tsx
// and return the parsed results.
function evalGating() {
  const script = `
import { buildWebToolGating, isWebToolsAllowed, WEB_TOOLS } from ${JSON.stringify(webToolsTs)};

const cases = [
  // [label, sdkEnv, isOffline]
  ['non-arc default (unset)', {}, false],
  ['non-arc explicit on "1"', { KCSI_ALLOW_WEB_TOOLS: '1' }, false],
  ['non-arc explicit on "true"', { KCSI_ALLOW_WEB_TOOLS: 'true' }, false],
  ['non-arc explicit off "0"', { KCSI_ALLOW_WEB_TOOLS: '0' }, false],
  ['non-arc explicit off "false"', { KCSI_ALLOW_WEB_TOOLS: 'false' }, false],
  ['non-arc whitespace', { KCSI_ALLOW_WEB_TOOLS: '   ' }, false],
  ['arc default', {}, true],
  ['arc with flag on (ARC always wins)', { KCSI_ALLOW_WEB_TOOLS: '1' }, true],
];

const out = cases.map(([label, env, isOffline]) => ({
  label,
  flag: isWebToolsAllowed(env),
  gating: buildWebToolGating(env, isOffline),
}));
process.stdout.write(JSON.stringify({ WEB_TOOLS, out }));
`;
  return spawnSync(tsxBin, ['--eval', script, '--conditions=node'], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env },
  });
}

describe('web_tools.ts gating — behavioral (issue #666)', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/');
    return;
  }

  const res = evalGating();
  it('tsx evaluated the real module', () => {
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
  });
  if (res.status !== 0) return;
  const { WEB_TOOLS, out } = JSON.parse(res.stdout);
  const byLabel = Object.fromEntries(out.map((r) => [r.label, r]));
  const WEB = ['WebSearch', 'WebFetch'];
  assert.deepEqual(WEB_TOOLS, WEB);

  it('default (flag unset) DENIES web tools on a non-ARC benchmark', () => {
    const r = byLabel['non-arc default (unset)'];
    assert.equal(r.flag, false);
    assert.equal(r.gating.webToolsEnabled, false);
    assert.deepEqual(r.gating.allowlistWebTools, []);
    // The load-bearing denial: web tools are in disallowedTools.
    assert.deepEqual(r.gating.disallowedWebTools, WEB);
    assert.match(r.gating.reason, /default-off/);
  });

  it('explicit opt-in ENABLES web tools on a non-ARC benchmark', () => {
    for (const label of ['non-arc explicit on "1"', 'non-arc explicit on "true"']) {
      const r = byLabel[label];
      assert.equal(r.gating.webToolsEnabled, true, label);
      assert.deepEqual(r.gating.allowlistWebTools, WEB, label);
      assert.deepEqual(r.gating.disallowedWebTools, [], label);
      assert.equal(r.gating.reason, 'KCSI_ALLOW_WEB_TOOLS=1', label);
    }
  });

  it('false-y / whitespace flag values DENY (default-off preserved)', () => {
    for (const label of ['non-arc explicit off "0"', 'non-arc explicit off "false"', 'non-arc whitespace']) {
      const r = byLabel[label];
      assert.equal(r.gating.webToolsEnabled, false, label);
      assert.deepEqual(r.gating.disallowedWebTools, WEB, label);
    }
  });

  it('ARC always DENIES, even with the flag on', () => {
    for (const label of ['arc default', 'arc with flag on (ARC always wins)']) {
      const r = byLabel[label];
      assert.equal(r.gating.webToolsEnabled, false, label);
      assert.deepEqual(r.gating.allowlistWebTools, [], label);
      assert.deepEqual(r.gating.disallowedWebTools, WEB, label);
      assert.match(r.gating.reason, /ARC offline benchmark/, label);
    }
  });
});
