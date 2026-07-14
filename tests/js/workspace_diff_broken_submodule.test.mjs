import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const mainTs = path.join(repoRoot, 'runtime_runner', 'src', 'main.ts');
const tsxAvailable = fs.existsSync(tsxBin);
const gitAvailable = spawnSync('git', ['--version'], { encoding: 'utf8' }).status === 0;

// Regression: a SWE-bench Pro instance whose submodule failed to clone is left
// with a broken gitlink (a ``.git`` file pointing at a missing gitdir). Before
// the --ignore-submodules=all fix, ``git diff HEAD`` exited 128 on it and
// gitOutput swallowed the failure to '', silently dropping every real
// tracked-file edit the agent made (scored no_patch). This proves the agent's
// tracked edit still reaches captureWorkspaceDiff despite the broken submodule.
function runFixture() {
  const source = `
    import { captureWorkspaceDiff } from ${JSON.stringify(mainTs)};
    import fs from 'node:fs';
    import os from 'node:os';
    import path from 'node:path';
    import { spawnSync } from 'node:child_process';

    const parent = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-broken-submodule-'));
    const sub = path.join(parent, 'sub_src');
    const repo = path.join(parent, 'repo');
    const git = (cwd, ...args) => {
      const res = spawnSync('git', ['-C', cwd, ...args], { encoding: 'utf8' });
      if (res.status !== 0) throw new Error('git ' + args.join(' ') + ' failed: ' + res.stderr);
      return res.stdout.trim();
    };

    // a small standalone repo to use as the submodule source
    fs.mkdirSync(sub);
    git(sub, 'init', '-q');
    git(sub, 'config', 'user.email', 'ksi@test');
    git(sub, 'config', 'user.name', 'ksi');
    fs.writeFileSync(path.join(sub, 's.txt'), 'hi\\\\n');
    git(sub, 'add', '-A');
    git(sub, 'commit', '-q', '-m', 'init');

    // superproject with the submodule wired in
    fs.mkdirSync(repo);
    git(repo, 'init', '-q');
    git(repo, 'config', 'user.email', 'ksi@test');
    git(repo, 'config', 'user.name', 'ksi');
    fs.writeFileSync(path.join(repo, 'tracked.py'), 'base\\\\n');
    git(repo, 'add', '-A');
    git(repo, 'commit', '-q', '-m', 'base');
    git(repo, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', sub, 'sub');
    git(repo, 'commit', '-q', '-m', 'addsub');

    // simulate the failed-clone state: break the submodule gitlink
    fs.writeFileSync(path.join(repo, 'sub', '.git'), 'gitdir: /nonexistent/broken/path\\\\n');
    // agent edit to a tracked file that MUST survive capture
    fs.writeFileSync(path.join(repo, 'tracked.py'), 'base\\\\nAGENT_EDIT_LINE\\\\n');

    process.stdout.write(JSON.stringify(captureWorkspaceDiff(repo)));
  `;
  return spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, NODE_NO_WARNINGS: '1' },
  });
}

describe('workspace diff capture survives a broken submodule gitlink', () => {
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

  it('captures the tracked-file edit despite the broken submodule', () => {
    assert.match(captured.diff, /diff --git a\/tracked\.py b\/tracked\.py/);
    assert.match(captured.diff, /AGENT_EDIT_LINE/);
    assert.ok(captured.changedFiles.includes('tracked.py'));
    assert.equal(captured.captureError, undefined);
  });
});
