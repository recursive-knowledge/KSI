import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');
const src = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'src', 'container_args.ts'),
  'utf-8',
);

describe('container loads the embedding cache offline (#923 M2)', () => {
  it('sets HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE so a read-only warm cache loads lock-free', () => {
    assert.match(src, /HF_HUB_OFFLINE=1/);
    assert.match(src, /TRANSFORMERS_OFFLINE=1/);
  });
});
