/**
 * Behavioral regression tests for F1: wipe_workspace_per_task=false still
 * recopied repo.
 *
 * Before the fix, seedWorkspace() unconditionally `removePathIfExists()` +
 * `copyRepo()` even when wipeWorkspacePerTask=false, destroying warmed
 * repo-local state (node_modules, build caches).
 *
 * The fix persists a content-fingerprint stamp inside repoDst and skips
 * the copy when both stamp and source content match. These tests verify
 * the actual behavior — not source-code text — by spawning tsx, calling
 * seedWorkspace, and observing whether sentinel files survive across calls.
 *
 * Test plan (audit handoff F1 contract):
 *   1. First seed copies the repo and writes the stamp
 *   2. Second seed with wipe=false and unchanged source SKIPS the copy
 *      (sentinel file added between calls survives)
 *   3. Source content change INVALIDATES the stamp → second seed re-copies
 *      (sentinel is gone; new content visible)
 *   4. Adding a new file in source invalidates the stamp (the audit's
 *      specific concern: ext4 directory mtime doesn't propagate on this
 *      operation, so a directory-mtime stamp would silently miss it)
 *   5. wipe=true ALWAYS re-copies regardless of stamp
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const workspaceTs = path.join(repoRoot, 'runtime_runner', 'src', 'workspace.ts');
const tsxAvailable = fs.existsSync(tsxBin);

function runFixture(script, env = {}) {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-stamp-test-'));
  try {
    const res = spawnSync(
      tsxBin,
      ['--eval', script, '--conditions=node'],
      {
        // The container_runner derives workspace paths from process.cwd();
        // use a fresh tmp dir so seeds land in isolation.
        cwd: tmpDir,
        encoding: 'utf8',
        env: {
          ...process.env,
          ...env,
        },
      },
    );
    return { ...res, tmpDir };
  } catch (err) {
    fs.rmSync(tmpDir, { recursive: true, force: true });
    throw err;
  }
}

function makeFixtureScript(srcDir, mutationSteps) {
  return `
import { seedWorkspace } from ${JSON.stringify(workspaceTs)};
import fs from 'node:fs';
import path from 'node:path';

const srcDir = ${JSON.stringify(srcDir)};
const workspaceKey = 'task__test__test__abcdef0123';

function payload() {
  return {
    workspace_seed: {
      instruction_md: 'noop',
      memory_md: 'noop',
      task_md: 'noop',
      repo_source_path: srcDir,
    },
  };
}

// Mirror workspace_scope.ts resolution for a 'task__' key:
// parseWorkspaceKey strips the 'task__' prefix, so the on-disk dir is:
//   <cwd>/workspaces/tasks/<key-without-prefix>/workspace/repo
const workspaceKeyDir = workspaceKey.startsWith('task__')
  ? workspaceKey.slice('task__'.length)
  : workspaceKey;
const repoDst = path.join(process.cwd(), 'workspaces', 'tasks', workspaceKeyDir, 'workspace', 'repo');
const sentinelPath = path.join(repoDst, '.sentinel');
const stampPath = path.join(repoDst, '.repo-stamp');

const log = [];
${mutationSteps}
process.stdout.write(JSON.stringify(log));
`;
}

describe('workspace stamp-file optimization (F1) — behavioral', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/');
    return;
  }

  it('first seed copies the repo and writes a stamp', () => {
    const tmpSrc = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-stamp-src-'));
    fs.writeFileSync(path.join(tmpSrc, 'data.txt'), 'original');
    try {
      const script = makeFixtureScript(tmpSrc, `
seedWorkspace(payload(), workspaceKey, false);
log.push({ step: 1, dataExists: fs.existsSync(path.join(repoDst, 'data.txt')) });
log.push({ step: 1, stampExists: fs.existsSync(stampPath) });
log.push({ step: 1, dataContent: fs.readFileSync(path.join(repoDst, 'data.txt'), 'utf-8') });
`);
      const res = runFixture(script);
      try {
        assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
        const log = JSON.parse(res.stdout);
        assert.deepEqual(log[0], { step: 1, dataExists: true });
        assert.deepEqual(log[1], { step: 1, stampExists: true });
        assert.deepEqual(log[2], { step: 1, dataContent: 'original' });
      } finally {
        fs.rmSync(res.tmpDir, { recursive: true, force: true });
      }
    } finally {
      fs.rmSync(tmpSrc, { recursive: true, force: true });
    }
  });

  it('second seed with wipe=false and unchanged source SKIPS the copy', () => {
    const tmpSrc = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-stamp-src-'));
    fs.writeFileSync(path.join(tmpSrc, 'data.txt'), 'original');
    try {
      const script = makeFixtureScript(tmpSrc, `
seedWorkspace(payload(), workspaceKey, false);
fs.writeFileSync(sentinelPath, 'survived');
log.push({ step: 'seed1', dataContent: fs.readFileSync(path.join(repoDst, 'data.txt'), 'utf-8') });

seedWorkspace(payload(), workspaceKey, false);
log.push({ step: 'seed2', sentinelSurvived: fs.existsSync(sentinelPath) });
log.push({ step: 'seed2', sentinelContent: fs.existsSync(sentinelPath) ? fs.readFileSync(sentinelPath, 'utf-8') : null });
`);
      const res = runFixture(script);
      try {
        assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
        const log = JSON.parse(res.stdout);
        assert.deepEqual(log[0], { step: 'seed1', dataContent: 'original' });
        assert.deepEqual(log[1], { step: 'seed2', sentinelSurvived: true });
        assert.deepEqual(log[2], { step: 'seed2', sentinelContent: 'survived' });
      } finally {
        fs.rmSync(res.tmpDir, { recursive: true, force: true });
      }
    } finally {
      fs.rmSync(tmpSrc, { recursive: true, force: true });
    }
  });

  it('source content change invalidates the stamp and triggers re-copy', () => {
    const tmpSrc = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-stamp-src-'));
    fs.writeFileSync(path.join(tmpSrc, 'data.txt'), 'original');
    try {
      const script = makeFixtureScript(tmpSrc, `
seedWorkspace(payload(), workspaceKey, false);
fs.writeFileSync(sentinelPath, 'survived');

fs.writeFileSync(path.join(srcDir, 'data.txt'), 'updated content');

seedWorkspace(payload(), workspaceKey, false);
log.push({ step: 'seed2', sentinelSurvived: fs.existsSync(sentinelPath) });
log.push({ step: 'seed2', dataContent: fs.readFileSync(path.join(repoDst, 'data.txt'), 'utf-8') });
`);
      const res = runFixture(script);
      try {
        assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
        const log = JSON.parse(res.stdout);
        assert.deepEqual(log[0], { step: 'seed2', sentinelSurvived: false },
          'sentinel should be removed when source changed');
        assert.deepEqual(log[1], { step: 'seed2', dataContent: 'updated content' },
          'new source content should be visible after invalidation');
      } finally {
        fs.rmSync(res.tmpDir, { recursive: true, force: true });
      }
    } finally {
      fs.rmSync(tmpSrc, { recursive: true, force: true });
    }
  });

  it('source file addition invalidates the stamp', () => {
    // The audit's specific concern: file added inside a directory whose
    // parent dir mtime doesn't always propagate on ext4. With the old
    // mtime-only stamp this would silently miss the addition; with the
    // fingerprint walk it must be detected.
    const tmpSrc = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-stamp-src-'));
    fs.writeFileSync(path.join(tmpSrc, 'data.txt'), 'original');
    try {
      const script = makeFixtureScript(tmpSrc, `
seedWorkspace(payload(), workspaceKey, false);
fs.writeFileSync(sentinelPath, 'survived');

fs.writeFileSync(path.join(srcDir, 'extra.txt'), 'new file');

seedWorkspace(payload(), workspaceKey, false);
log.push({ step: 'seed2', sentinelSurvived: fs.existsSync(sentinelPath) });
log.push({ step: 'seed2', extraExists: fs.existsSync(path.join(repoDst, 'extra.txt')) });
`);
      const res = runFixture(script);
      try {
        assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
        const log = JSON.parse(res.stdout);
        assert.deepEqual(log[0], { step: 'seed2', sentinelSurvived: false },
          'addition of a new file in source must invalidate the stamp');
        assert.deepEqual(log[1], { step: 'seed2', extraExists: true });
      } finally {
        fs.rmSync(res.tmpDir, { recursive: true, force: true });
      }
    } finally {
      fs.rmSync(tmpSrc, { recursive: true, force: true });
    }
  });

  it('wipe=true ALWAYS re-copies regardless of stamp', () => {
    const tmpSrc = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-stamp-src-'));
    fs.writeFileSync(path.join(tmpSrc, 'data.txt'), 'original');
    try {
      const script = makeFixtureScript(tmpSrc, `
seedWorkspace(payload(), workspaceKey, false);
fs.writeFileSync(sentinelPath, 'survived');

seedWorkspace(payload(), workspaceKey, true);
log.push({ step: 'seed2', sentinelSurvived: fs.existsSync(sentinelPath) });
log.push({ step: 'seed2', dataExists: fs.existsSync(path.join(repoDst, 'data.txt')) });
`);
      const res = runFixture(script);
      try {
        assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
        const log = JSON.parse(res.stdout);
        assert.deepEqual(log[0], { step: 'seed2', sentinelSurvived: false },
          'wipe=true must wipe the workspace root regardless of stamp match');
        assert.deepEqual(log[1], { step: 'seed2', dataExists: true });
      } finally {
        fs.rmSync(res.tmpDir, { recursive: true, force: true });
      }
    } finally {
      fs.rmSync(tmpSrc, { recursive: true, force: true });
    }
  });

  // Source-text guard: defense against regressing back to mtime-only.
  // The previous implementation used `<src>:<mtimeMs>` as the stamp,
  // which silently missed file-content changes inside an unchanged
  // parent directory. The fingerprint walk is the load-bearing fix.
  const workspaceTsContent = fs.readFileSync(workspaceTs, 'utf-8');

  it('uses content fingerprint, not directory mtime', () => {
    assert.match(workspaceTsContent, /computeRepoFingerprint/,
      'must compute a content fingerprint, not rely on directory mtime');
    assert.doesNotMatch(
      workspaceTsContent,
      /writeText\(stampPath,\s*`\$\{srcResolved\}:\$\{srcMtime\}`\)/,
      'stamp format must not be source:mtime',
    );
  });
});
