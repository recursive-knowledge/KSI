import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const mainTs = path.join(repoRoot, 'runtime_runner', 'src', 'main.ts');
const tsxAvailable = fs.existsSync(tsxBin);
const gitAvailable = spawnSync('git', ['--version'], { encoding: 'utf8' }).status === 0;

function runFixture() {
  const source = `
    import { captureWorkspaceDiff } from ${JSON.stringify(mainTs)};
    import fs from 'node:fs';
    import os from 'node:os';
    import path from 'node:path';
    import { spawnSync } from 'node:child_process';

    const parent = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-workspace-diff-'));
    const repo = path.join(parent, 'repo');
    const outside = path.join(parent, 'outside-secret.txt');
    fs.mkdirSync(repo);
    fs.writeFileSync(outside, 'SECRET_OUTSIDE_REPO\\\\n');

    const git = (...args) => {
      const res = spawnSync('git', ['-C', repo, ...args], { encoding: 'utf8' });
      if (res.status !== 0) throw new Error('git ' + args.join(' ') + ' failed: ' + res.stderr);
      return res.stdout.trim();
    };
    git('init', '-q');
    git('config', 'user.email', 'ksi@test');
    git('config', 'user.name', 'ksi');
    fs.writeFileSync(path.join(repo, 'tracked.txt'), 'base\\\\n');
    git('add', 'tracked.txt');
    git('commit', '-q', '-m', 'base');

    fs.symlinkSync(outside, path.join(repo, 'leak.txt'));
    fs.writeFileSync(path.join(repo, 'new.txt'), 'SAFE_NEW_FILE\\\\n');

    process.stdout.write(JSON.stringify(captureWorkspaceDiff(repo)));
  `;
  return spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, NODE_NO_WARNINGS: '1' },
  });
}

describe('workspace diff capture rejects untracked symlink content', () => {
  if (!tsxAvailable || !gitAvailable) {
    it.skip('tsx and/or git not installed');
    return;
  }

  const result = runFixture();
  it('evaluates the real host runtime module', () => {
    assert.equal(result.status, 0, `fixture failed\nstdout=${result.stdout}\nstderr=${result.stderr}`);
  });
  if (result.status !== 0) return;

  const captured = JSON.parse(result.stdout);

  it('captures normal untracked file content', () => {
    assert.match(captured.diff, /diff --git a\/new\.txt b\/new\.txt/);
    assert.match(captured.diff, /SAFE_NEW_FILE/);
  });

  it('does not follow untracked symlinks outside the repo', () => {
    assert.ok(captured.changedFiles.includes('leak.txt'));
    assert.doesNotMatch(captured.diff, /diff --git a\/leak\.txt b\/leak\.txt/);
    assert.doesNotMatch(captured.diff, /SECRET_OUTSIDE_REPO/);
  });
});
