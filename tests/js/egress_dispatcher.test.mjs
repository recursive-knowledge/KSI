import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const idx = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'index.ts'),
  'utf-8',
);

describe('egress dispatcher install', () => {
  it('installs a ProxyAgent dispatcher when HTTPS_PROXY is set', () => {
    assert.match(idx, /setGlobalDispatcher/);
    assert.match(idx, /ProxyAgent/);
    assert.match(idx, /HTTPS_PROXY/);
  });
});
