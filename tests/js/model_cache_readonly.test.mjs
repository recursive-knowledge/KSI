import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');
const src = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'src', 'container_mounts.ts'),
  'utf-8',
);

describe('model cache mounts are read-only (#923 M2)', () => {
  it('mounts huggingface + sentence-transformers caches read-only', () => {
    // Scope to the model-cache mount block to avoid matching other mounts.
    const start = src.indexOf('Shared model cache');
    const end = src.indexOf('Copy agent-runner source');
    assert.ok(start !== -1 && end !== -1 && end > start, 'model-cache block not found');
    const block = src.slice(start, end);
    assert.match(block, /huggingFaceCacheDir,[\s\S]*?readonly:\s*true/);
    assert.match(block, /sentenceTransformersCacheDir,[\s\S]*?readonly:\s*true/);
    assert.doesNotMatch(block, /readonly:\s*false/);
  });
});
