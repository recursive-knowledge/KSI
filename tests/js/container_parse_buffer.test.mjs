import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const runnerTs = path.join(repoRoot, 'runtime_runner', 'src', 'container_runner.ts');
const outputTs = path.join(repoRoot, 'runtime_runner', 'src', 'container_output.ts');
const tsxAvailable = fs.existsSync(tsxBin);

function sampleTrimmedBuffers() {
  const source = `
    import {
      STREAM_PARSE_BUFFER_MAX_CHARS,
      trimStreamParseBuffer,
    } from ${JSON.stringify(runnerTs)};
    import { OUTPUT_START_MARKER } from ${JSON.stringify(outputTs)};

    const max = 64;
    const noMarker = trimStreamParseBuffer('x'.repeat(200), max);
    const prefixed = trimStreamParseBuffer('noise'.repeat(20) + OUTPUT_START_MARKER + '{"status"', max);
    const incomplete = trimStreamParseBuffer(OUTPUT_START_MARKER + 'y'.repeat(200), max);
    const laterStart = trimStreamParseBuffer(
      OUTPUT_START_MARKER + 'old'.repeat(100) + OUTPUT_START_MARKER + '{"status"',
      max,
    );
    process.stdout.write(JSON.stringify({
      defaultMax: STREAM_PARSE_BUFFER_MAX_CHARS,
      startMarker: OUTPUT_START_MARKER,
      noMarker,
      prefixed,
      incomplete,
      laterStart,
      lengths: {
        noMarker: noMarker.length,
        prefixed: prefixed.length,
        incomplete: incomplete.length,
        laterStart: laterStart.length,
      },
    }));
  `;
  return spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, NODE_NO_WARNINGS: '1' },
  });
}

describe('container stream parse buffer trimming', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/');
    return;
  }

  const result = sampleTrimmedBuffers();
  it('evaluates the real container_runner module', () => {
    assert.equal(result.status, 0, `tsx failed\nstdout=${result.stdout}\nstderr=${result.stderr}`);
  });
  if (result.status !== 0) return;

  const sample = JSON.parse(result.stdout);

  it('keeps the production cap finite', () => {
    assert.equal(sample.defaultMax, 2_000_000);
  });

  it('drops marker-free noise down to the possible marker suffix', () => {
    assert.ok(sample.lengths.noMarker < sample.startMarker.length);
  });

  it('discards noise before the first output marker', () => {
    assert.ok(sample.prefixed.startsWith(sample.startMarker));
    assert.ok(sample.lengths.prefixed <= 64);
  });

  it('bounds an oversized incomplete marker block', () => {
    assert.ok(sample.incomplete.startsWith(sample.startMarker));
    assert.ok(sample.lengths.incomplete <= 64);
  });

  it('prefers a later marker when an earlier incomplete block overflows', () => {
    assert.ok(sample.laterStart.startsWith(sample.startMarker));
    assert.ok(sample.lengths.laterStart <= 64);
    assert.match(sample.laterStart, /\{"status"/);
  });
});
