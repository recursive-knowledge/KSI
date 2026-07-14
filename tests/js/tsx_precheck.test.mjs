/**
 * Central tsx-availability canary (issue #979).
 *
 * Most behavioral JS tests in this directory spawn
 * `runtime_runner/node_modules/.bin/tsx` to import the real
 * `runtime_runner/{src,agent-runner/src}/*.ts` modules. When that binary is
 * missing they `it.skip(...)` — the correct local behavior in a fresh git
 * worktree without the gitignored `node_modules`. But the same skip is a
 * FALSE GREEN in CI: if the runtime_runner `npm ci` step is ever dropped or
 * fails, every tsx-dependent suite silently skips and `node --test` reports
 * 100% pass with ZERO behavioral TS coverage — including web-tool gating
 * (a security control) and pino->stderr routing (a regression that once caused
 * ~60% silent ARC failures).
 *
 * `tests/js/swebench_repo_sanitize.test.mjs` already fails loudly in CI for its
 * own (security) suite. This canary generalizes that guard to the whole
 * directory: a single test that FAILS (not skips) when `process.env.CI` is set
 * and tsx is unavailable, so a missing toolchain can never masquerade as a
 * green run. Locally (CI unset) it is a no-op pass.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const tsxAvailable = fs.existsSync(tsxBin);

describe('tsx toolchain precheck (#979)', () => {
  it('tsx must be available whenever CI is set (else tsx-dependent suites silently skip)', () => {
    if (process.env.CI && !tsxAvailable) {
      assert.fail(
        `tsx missing in CI (expected at ${tsxBin}). The tsx-dependent JS suites ` +
          '(web-tool gating, stdout/pino isolation, anthropic transport retry, ' +
          'arc no-mcp synthesis, session-state path guard, workspace stamp, ' +
          '.git sanitization) would all silently skip and report a false green. ' +
          'Run `npm ci` in runtime_runner before `node --test tests/js/*.test.mjs`.',
      );
    }
    // Locally (CI unset): a missing tsx is an expected, allowed skip-condition
    // for the other suites, so this canary is a no-op pass.
    assert.ok(true);
  });
});
