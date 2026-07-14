/**
 * Behavioral regression test for issue #924: SWE-bench Pro copied the FULL
 * upstream `.git` (live `origin` + the future fix commit, reachable via
 * refs/remotes/origin/*) into the solver workspace, letting an agent read the
 * graded answer OFFLINE with `git log HEAD..origin/master` / `git show
 * <test_commit>:<f>` — bypassing the #666 web-tool gate.
 *
 * The fix: copyRepo() calls sanitizeRepoHistory() on the disposable per-task
 * copy, dropping all remotes + refs and pruning unreachable objects so only the
 * detached base_commit HEAD survives. The grading `git diff HEAD` (main.ts)
 * must still produce the agent's patch.
 *
 * These tests assert the actual git state of the seeded workspace, not source
 * text: build a cache-shaped repo (origin remote + future commit + detached
 * HEAD@base), run seedWorkspace via tsx, then probe the resulting repo with git.
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
const gitAvailable = spawnSync('git', ['--version'], { encoding: 'utf8' }).status === 0;

function git(cwd, args, input) {
  return spawnSync('git', ['-C', cwd, ...args], { encoding: 'utf8', input });
}

// Build a repo shaped like a SWE-bench Pro cache clone: a base commit, a later
// "fix" commit (the answer) reachable only via refs/remotes/origin/master + a
// tag, a live origin URL, and a detached HEAD at base (as cache prep leaves it).
function makeCacheShapedRepo({ packed = true } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-sanitize-src-'));
  const g = (...a) => {
    const r = git(dir, a);
    if (r.status !== 0) throw new Error(`git ${a.join(' ')} failed: ${r.stderr}`);
    return (r.stdout || '').trim();
  };
  // `git init -b <name>` needs git >= 2.28; init then `branch -M` works on any
  // git version and on the unborn branch (avoids a hard error on old CI images).
  g('init', '-q');
  g('branch', '-M', 'master');
  g('config', 'user.email', 'kcsi@test');
  g('config', 'user.name', 'kcsi');
  // A real SWE-bench Pro base_commit is NOT a root commit — it sits on top of
  // the project's history. Give base a prior commit so verifySanitized is
  // exercised against a base that has ancestors (the condition that makes a
  // naive `rev-list --all --count == 1` gate fail-closed on every real task).
  fs.writeFileSync(path.join(dir, 'file.txt'), 'older project history\n');
  g('add', '-A');
  g('commit', '-q', '-m', 'pre-base history');
  fs.writeFileSync(path.join(dir, 'file.txt'), 'base content\n');
  g('add', '-A');
  g('commit', '-q', '-m', 'base');
  const base = g('rev-parse', 'HEAD');
  fs.writeFileSync(path.join(dir, 'file.txt'), 'THE FUTURE FIX (answer)\n');
  fs.writeFileSync(path.join(dir, 'fix_test.py'), 'def test_fix(): assert True\n');
  g('add', '-A');
  g('commit', '-q', '-m', 'fix: the graded answer + hidden test');
  const fix = g('rev-parse', 'HEAD');
  g('tag', 'test_commit');
  // Simulate a clone's remote-tracking refs + live origin, then detach at base.
  g('update-ref', 'refs/remotes/origin/master', fix);
  g('remote', 'add', 'origin', 'https://github.com/example/repo.git');
  // Real SWE-bench Pro caches are full GitHub clones whose history lives in a
  // PACKFILE — pack here (while the fix is still reachable) so the test exercises
  // gc's packed-unreachable prune path, not just the trivial loose-object path.
  if (packed) {
    g('gc', '-q');
  }
  g('checkout', '-q', '-f', '--detach', base);
  g('clean', '-fdx');
  return { dir, base, fix };
}

// A SWE-bench Pro cache for the 2nd+ task of a repo is a `git clone --shared`
// of a sibling cache: its objects live in the sibling via .git/objects/info/
// alternates (an absolute host path), not locally. Returns that shared clone as
// the repo_source_path, plus the sibling source dir that must outlive sanitize.
function makeSharedCacheShapedRepo() {
  const { dir: sibling, base, fix } = makeCacheShapedRepo({ packed: true });
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-sanitize-shared-'));
  const shared = path.join(parent, 'clone');
  const c = spawnSync('git', ['clone', '-q', '--shared', '--no-checkout', sibling, shared], {
    encoding: 'utf8',
  });
  if (c.status !== 0) throw new Error(`shared clone failed: ${c.stderr}`);
  const sg = (...a) => {
    const r = git(shared, a);
    if (r.status !== 0) throw new Error(`git ${a.join(' ')} failed: ${r.stderr}`);
    return (r.stdout || '').trim();
  };
  sg('config', 'user.email', 'kcsi@test');
  sg('config', 'user.name', 'kcsi');
  sg('checkout', '-q', '-f', '--detach', base);
  const altPath = path.join(shared, '.git', 'objects', 'info', 'alternates');
  assert.ok(fs.existsSync(altPath), 'fixture precondition: shared clone must have an alternates link');
  return { sibling, parent, shared, base, fix };
}

// Build a superproject with an initialized git submodule (issue #1256). The
// submodule's own history carries a "future" commit ahead of the pinned commit
// and a live `origin` remote — mirroring how a SWE-bench Pro repo (e.g.
// openlibrary vendoring infogami) leaves `.git/modules/<path>` populated after
// `git submodule update --init`. Returns the superproject as repo_source_path.
function makeSubmoduleCacheShapedRepo() {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-sanitize-sub-'));
  const subSrc = path.join(parent, 'dep');
  const superSrc = path.join(parent, 'super');
  const mk = (dir) => {
    fs.mkdirSync(dir, { recursive: true });
    const g = (...a) => {
      const r = git(dir, a);
      if (r.status !== 0) throw new Error(`git ${a.join(' ')} failed: ${r.stderr}`);
      return (r.stdout || '').trim();
    };
    g('init', '-q');
    g('branch', '-M', 'master');
    g('config', 'user.email', 'kcsi@test');
    g('config', 'user.name', 'kcsi');
    return g;
  };
  // Dependency (submodule) repo: pinned commit + a later "future" commit.
  const gd = mk(subSrc);
  fs.writeFileSync(path.join(subSrc, 'dep.py'), 'VALUE = 1  # pinned\n');
  gd('add', '-A');
  gd('commit', '-q', '-m', 'dep: pinned');
  const subPin = gd('rev-parse', 'HEAD');
  fs.writeFileSync(path.join(subSrc, 'dep.py'), 'VALUE = 999  # SUBMODULE FUTURE ANSWER\n');
  gd('add', '-A');
  gd('commit', '-q', '-m', 'dep: future answer');
  const subFuture = gd('rev-parse', 'HEAD');
  gd('checkout', '-q', subPin);
  // Superproject: add the submodule at the pinned commit. file:// submodules are
  // blocked by default on modern git — allow them for the fixture only.
  const gs = mk(superSrc);
  const addRes = git(superSrc, [
    '-c', 'protocol.file.allow=always',
    'submodule', 'add', subSrc, 'vendor/dep',
  ]);
  if (addRes.status !== 0) throw new Error(`submodule add failed: ${addRes.stderr}`);
  git(superSrc, ['-C', 'vendor/dep', 'checkout', '-q', subPin]);
  gs('add', '-A');
  gs('commit', '-q', '-m', 'super: vendor dep as submodule');
  const superBase = gs('rev-parse', 'HEAD');
  gs('checkout', '-q', '-f', '--detach', superBase);
  const modulesDir = path.join(superSrc, '.git', 'modules', 'vendor', 'dep');
  assert.ok(fs.existsSync(modulesDir), 'fixture precondition: .git/modules/vendor/dep must exist');
  return { parent, superSrc, superBase, subPin, subFuture };
}

function seedFixtureScript(srcDir) {
  const workspaceKey = 'task__t__t__abcdef0123';
  return `
import { seedWorkspace } from ${JSON.stringify(workspaceTs)};
import path from 'node:path';
const workspaceKey = ${JSON.stringify(workspaceKey)};
function payload() {
  return { workspace_seed: { instruction_md: 'x', memory_md: 'x', task_md: 'x', repo_source_path: ${JSON.stringify(srcDir)} } };
}
seedWorkspace(payload(), workspaceKey, false);
const repoDst = path.join(process.cwd(), 'workspaces', 'tasks', workspaceKey.slice('task__'.length), 'workspace', 'repo');
process.stdout.write(repoDst);
`;
}

describe('SWE-bench Pro workspace .git sanitization (#924) — behavioral', () => {
  if (process.env.CI && (!tsxAvailable || !gitAvailable)) {
    // This is a SECURITY regression suite — silently skipping it in CI would
    // let a sanitizer regression land green. CI must provision tsx + git
    // (runtime_runner `npm ci` installs tsx). Fail loudly instead of skipping.
    it('SECURITY: tsx + git must be available in CI to run sanitization checks', () => {
      assert.fail(
        `tsx/git missing in CI (tsxAvailable=${tsxAvailable}, gitAvailable=${gitAvailable}); ` +
          'run `npm ci` in runtime_runner so the .git-sanitization tests actually execute',
      );
    });
    return;
  }
  if (!tsxAvailable || !gitAvailable) {
    it.skip('tsx and/or git not available');
    return;
  }

  it('strips origin + future history but keeps base HEAD and a working git diff', () => {
    const { dir: srcDir, base, fix } = makeCacheShapedRepo();
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-sanitize-cwd-'));
    try {
      const res = spawnSync(tsxBin, ['--eval', seedFixtureScript(srcDir), '--conditions=node'], {
        cwd,
        encoding: 'utf8',
        env: { ...process.env },
      });
      assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
      const repoDst = res.stdout.trim();
      assert.ok(fs.existsSync(path.join(repoDst, '.git')), 'workspace repo should still be a git repo');

      // The answer must be unreachable by ref AND physically pruned.
      assert.equal(git(repoDst, ['remote']).stdout.trim(), '', 'no remotes should remain');
      assert.equal(
        git(repoDst, ['rev-list', '--all', '--not', 'HEAD', '--count']).stdout.trim(),
        '0',
        'nothing should be reachable beyond the base commit history',
      );
      assert.notEqual(
        git(repoDst, ['cat-file', '-t', fix]).status,
        0,
        'the future fix object must be pruned (not recoverable via cat-file)',
      );
      assert.notEqual(
        git(repoDst, ['log', '--oneline', `${base}..origin/master`]).status,
        0,
        'origin/master must not resolve',
      );
      assert.equal(
        git(repoDst, ['fsck', '--unreachable', '--no-dangling']).stdout
          .split('\n')
          .filter((l) => l.startsWith('unreachable ')).length,
        0,
        'no unreachable objects of ANY type (commit/tree/blob) should remain',
      );
      assert.ok(
        !fs.existsSync(path.join(repoDst, '.git', 'objects', 'info', 'alternates')),
        'no alternates link should remain',
      );

      // Grading still works: HEAD is base, and a worktree edit shows in diff.
      assert.equal(git(repoDst, ['rev-parse', 'HEAD']).stdout.trim(), base, 'HEAD must stay at base_commit');
      fs.writeFileSync(path.join(repoDst, 'file.txt'), 'base content\nagent edit\n');
      const diff = git(repoDst, ['diff', '--no-ext-diff', 'HEAD', '--']).stdout;
      assert.match(diff, /\+agent edit/, 'git diff HEAD must capture the agent edit');
    } finally {
      fs.rmSync(srcDir, { recursive: true, force: true });
      fs.rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('strips a --shared cache copy: future objects in the alternate are removed', () => {
    // Regression for the #924 follow-up: a `git clone --shared` cache borrows
    // its objects (incl. the fix) from a sibling via .git/objects/info/alternates.
    // gc alone cannot prune the alternate store, so without absorbing + dropping
    // the link the fix stays recoverable on the host. The sibling source must
    // still exist while seedWorkspace sanitizes (the repack absorbs from it).
    const { sibling, parent, shared, base, fix } = makeSharedCacheShapedRepo();
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-sanitize-shared-cwd-'));
    try {
      const res = spawnSync(tsxBin, ['--eval', seedFixtureScript(shared), '--conditions=node'], {
        cwd,
        encoding: 'utf8',
        env: { ...process.env },
      });
      assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
      const repoDst = res.stdout.trim();

      assert.ok(
        !fs.existsSync(path.join(repoDst, '.git', 'objects', 'info', 'alternates')),
        'alternates link must be removed so borrowed objects cannot be followed',
      );
      assert.notEqual(
        git(repoDst, ['cat-file', '-t', fix]).status,
        0,
        'the future fix object (borrowed via alternate) must be unrecoverable on the host',
      );
      assert.equal(
        git(repoDst, ['rev-list', '--all', '--not', 'HEAD', '--count']).stdout.trim(),
        '0',
        'nothing should be reachable beyond the base commit history',
      );
      // Grading integrity: base_commit objects were absorbed locally, so HEAD
      // resolves and a worktree edit still shows in `git diff HEAD`.
      assert.equal(git(repoDst, ['rev-parse', 'HEAD']).stdout.trim(), base, 'HEAD must stay at base_commit');
      fs.writeFileSync(path.join(repoDst, 'file.txt'), 'base content\nagent edit\n');
      assert.match(
        git(repoDst, ['diff', '--no-ext-diff', 'HEAD', '--']).stdout,
        /\+agent edit/,
        'git diff HEAD must capture the agent edit',
      );
    } finally {
      fs.rmSync(sibling, { recursive: true, force: true });
      fs.rmSync(parent, { recursive: true, force: true });
      fs.rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('sanitizeRepoHistory is a no-op on a non-git source (polyglot/ARC)', () => {
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-sanitize-nogit-'));
    try {
      const script = `
import { sanitizeRepoHistory } from ${JSON.stringify(workspaceTs)};
import fs from 'node:fs';
import path from 'node:path';
const d = path.join(process.cwd(), 'plain');
fs.mkdirSync(d, { recursive: true });
fs.writeFileSync(path.join(d, 'solution.py'), 'print(1)\\n');
sanitizeRepoHistory(d); // must not throw
process.stdout.write(String(fs.existsSync(path.join(d, 'solution.py'))));
`;
      const res = spawnSync(tsxBin, ['--eval', script, '--conditions=node'], { cwd, encoding: 'utf8', env: { ...process.env } });
      assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
      assert.equal(res.stdout.trim(), 'true', 'non-git dir contents must be untouched');
    } finally {
      fs.rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('fails closed and does NOT mutate the external gitdir when .git is a pointer file', () => {
    // A `.git` pointer FILE (worktree / --separate-git-dir) would make
    // `git -C repoDst` operate on an external git dir. The sanitizer must
    // refuse rather than delete refs / gc someone else's repo.
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-sanitize-ptr-'));
    try {
      // An external "real" repo whose refs must survive untouched.
      const ext = path.join(cwd, 'external');
      fs.mkdirSync(ext, { recursive: true });
      const eg = (...a) => git(ext, a);
      eg('init', '-q');
      eg('branch', '-M', 'master');
      eg('config', 'user.email', 'kcsi@test');
      eg('config', 'user.name', 'kcsi');
      fs.writeFileSync(path.join(ext, 'f.txt'), 'x\n');
      eg('add', '-A');
      eg('commit', '-q', '-m', 'c0');
      const refsBefore = git(ext, ['for-each-ref', '--format=%(refname)']).stdout.trim();
      assert.ok(refsBefore.includes('refs/heads/master'), 'precondition: external repo has a branch ref');

      // A workspace dir whose `.git` is a pointer FILE into the external repo.
      const work = path.join(cwd, 'workspace');
      fs.mkdirSync(work, { recursive: true });
      fs.writeFileSync(path.join(work, '.git'), `gitdir: ${path.join(ext, '.git')}\n`);

      const script = `
import { sanitizeRepoHistory } from ${JSON.stringify(workspaceTs)};
try {
  sanitizeRepoHistory(${JSON.stringify(work)});
  process.stdout.write('NO_THROW');
} catch (e) {
  process.stdout.write('THREW:' + String(e && e.message));
}
`;
      const res = spawnSync(tsxBin, ['--eval', script, '--conditions=node'], { cwd, encoding: 'utf8', env: { ...process.env } });
      assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
      assert.match(res.stdout, /^THREW:/, 'sanitizer must throw on a .git pointer file');
      assert.match(res.stdout, /not a real directory/, 'throw reason must name the pointer-file guard');
      // The external repo must be completely untouched (no refs deleted, no gc).
      assert.equal(
        git(ext, ['for-each-ref', '--format=%(refname)']).stdout.trim(),
        refsBefore,
        'external gitdir refs must be untouched',
      );
    } finally {
      fs.rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('removes submodule git stores (.git/modules) while keeping working-tree files (#1256)', () => {
    const { parent, superSrc, subPin, subFuture } = makeSubmoduleCacheShapedRepo();
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-sanitize-sub-cwd-'));
    try {
      const res = spawnSync(tsxBin, ['--eval', seedFixtureScript(superSrc), '--conditions=node'], {
        cwd,
        encoding: 'utf8',
        env: { ...process.env },
      });
      assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
      const repoDst = res.stdout.trim();

      // The nested submodule git store must be gone — its history + origin
      // remote would otherwise be reachable in-container via `git -C vendor/dep`.
      assert.ok(
        !fs.existsSync(path.join(repoDst, '.git', 'modules')),
        '.git/modules must be removed so submodule history/remote is unreachable',
      );
      // Even if the submodule .git gitfile survives, it now dangles: no object
      // store, so neither the pinned nor the future submodule commit resolves.
      assert.notEqual(
        git(path.join(repoDst, 'vendor', 'dep'), ['cat-file', '-t', subFuture]).status,
        0,
        'the submodule future-answer commit must be unrecoverable',
      );
      assert.notEqual(
        git(path.join(repoDst, 'vendor', 'dep'), ['cat-file', '-t', subPin]).status,
        0,
        'the submodule git store must be fully gone (pinned commit also unrecoverable)',
      );
      // The agent still needs the submodule WORKING-TREE files (the point of
      // initializing the submodule) — those are plain files, untouched.
      assert.ok(
        fs.existsSync(path.join(repoDst, 'vendor', 'dep', 'dep.py')),
        'submodule working-tree files must survive sanitization',
      );
      assert.match(
        fs.readFileSync(path.join(repoDst, 'vendor', 'dep', 'dep.py'), 'utf8'),
        /VALUE = 1/,
        'submodule working-tree content must be the pinned version, intact',
      );
      // The superproject itself is still a valid, sanitized git repo.
      assert.equal(git(repoDst, ['remote']).stdout.trim(), '', 'no superproject remotes should remain');
    } finally {
      fs.rmSync(parent, { recursive: true, force: true });
      fs.rmSync(cwd, { recursive: true, force: true });
    }
  });
});
