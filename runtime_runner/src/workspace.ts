import { execFileSync } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

import { resolveWorkspaceRootPath } from './workspace_scope.js';

import { KsiPayload } from './types.js';

export const CONTAINER_WORKSPACE_ROOT = '/workspace/task';
export const CONTAINER_ACTIVE_WORKSPACE_DIR = `${CONTAINER_WORKSPACE_ROOT}/workspace`;
const DEFAULT_REPO_CONTAINER_PATH = `${CONTAINER_ACTIVE_WORKSPACE_DIR}/repo`;

export function shortHash(input: string): string {
  return crypto.createHash('sha1').update(input).digest('hex').slice(0, 10);
}

function normalizeKeyPart(
  value: string,
  fallback: string,
  maxLen: number,
): string {
  const cleaned = (value || fallback).replace(/[^a-zA-Z0-9._-]/g, '_').replace(/^[._-]+|[._-]+$/g, '');
  const bounded = (cleaned || fallback).slice(0, maxLen).replace(/[._-]+$/g, '');
  return bounded || fallback;
}

export function safeTaskDir(taskId: string): string {
  return normalizeKeyPart(taskId, 'task', 80);
}

export function safeExperimentDir(experimentName: string): string {
  return normalizeKeyPart(experimentName, 'default', 24);
}

export function toWorkspaceKey(
  payload: KsiPayload,
  scope: 'task' | 'agent',
): string {
  if (scope === 'agent') {
    return `agent__${safeTaskDir(payload.agent_id || 'agent')}`;
  }
  const rawTaskId = payload.task?.id || 'task';
  const rawExperiment = payload.experiment_name || 'default';
  const taskId = safeTaskDir(rawTaskId);
  const experiment = safeExperimentDir(rawExperiment);
  const digest = shortHash(`${rawExperiment}::${rawTaskId}`);
  return `task__${experiment}__${taskId}__${digest}`;
}

function ensureDir(p: string): void {
  fs.mkdirSync(p, { recursive: true });
}

function writeText(filePath: string, content: string): void {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, content, 'utf-8');
}

function removeFileIfExists(filePath: string): void {
  try {
    fs.rmSync(filePath, { force: true });
  } catch {
    // best-effort cleanup only
  }
}

function removePathIfExists(targetPath: string): void {
  if (!fs.existsSync(targetPath)) {
    return;
  }
  try {
    fs.rmSync(targetPath, {
      recursive: true,
      force: true,
      maxRetries: 8,
      retryDelay: 50,
    });
    return;
  } catch {
    // Fall through to child-by-child cleanup for occasional ENOTEMPTY races.
  }

  try {
    const stat = fs.lstatSync(targetPath);
    if (!stat.isDirectory()) {
      fs.rmSync(targetPath, { force: true });
      return;
    }
    for (const entry of fs.readdirSync(targetPath)) {
      removePathIfExists(path.join(targetPath, entry));
    }
    fs.rmdirSync(targetPath);
  } catch {
    // best-effort cleanup only
  }
}

export function buildWorkspaceMemoryMd(seedMemoryMd: string): string {
  // Write the full seed content so agents that Read MEMORY.md get the full
  // bundle directly. The same content is also appended to the system prompt
  // via .seed_context as a silent backup, but many agents attend to
  // MEMORY.md first.
  const trimmed = seedMemoryMd.trim();
  if (!trimmed) {
    return '';
  }
  return trimmed;
}

function copyRepo(repoSourcePath: string, repoDst: string): void {
  if (!repoSourcePath) {
    ensureDir(repoDst);
    writeText(path.join(repoDst, '.keep'), '');
    return;
  }

  const src = path.resolve(repoSourcePath);
  if (!fs.existsSync(src) || !fs.statSync(src).isDirectory()) {
    ensureDir(repoDst);
    writeText(path.join(repoDst, '.keep'), '');
    return;
  }

  // verbatimSymlinks: true is critical — without it Node resolves relative
  // symlink targets against the SOURCE path and records the absolute result
  // at the destination. Real-world hit on SWE-bench Pro openlibrary: source
  // has `config -> conf` (relative symlink); without verbatim mode the dest
  // gets `config -> /data/.../repo_cache/<instance_id>/conf` (absolute,
  // host-only). When the agent runs, `git diff` in the bind-mounted repo
  // emits a symlink-rewrite hunk that gets submitted to the SWE-bench Pro
  // grader as the patch — the grader rejects it (host paths don't exist
  // inside the grader container) and zero tests run.
  fs.cpSync(src, repoDst, { recursive: true, verbatimSymlinks: true });

  // The cache clone we just copied carries the FULL upstream history + a live
  // `origin` remote (incl. the future fix commit and post-fix hidden tests).
  // Strip them from this disposable per-task copy so the agent cannot read the
  // graded answer offline via git. See sanitizeRepoHistory / issue #924.
  sanitizeRepoHistory(repoDst);
}

// Ref namespaces deleted during sanitization. refs/notes and refs/replace are
// included for completeness (a clone may carry them) even though GitHub does not
// expose them by default — anything that keeps a future commit reachable must go.
const SANITIZE_REF_NAMESPACES = [
  'refs/remotes',
  'refs/tags',
  'refs/heads',
  'refs/notes',
  'refs/replace',
];

// Sanitizer-behavior version, embedded in the workspace `.repo-stamp`. BUMP
// this whenever sanitizeRepoHistory's stripping/verification behavior changes,
// so already-stamped workspaces (incl. pre-sanitizer copies with full upstream
// .git) are re-copied + re-sanitized instead of reused via the stamp fast path.
const SANITIZE_VERSION = 'v3';

// Canonical `.repo-stamp` contents: source path + content fingerprint +
// sanitizer version. Reader and writer MUST agree, so it lives in one place.
function stampValue(srcResolved: string, fingerprint: string): string {
  return `${srcResolved}:${fingerprint}:sanitize-${SANITIZE_VERSION}`;
}

/**
 * Strip upstream history from a freshly copied workspace repo.
 *
 * SWE-bench Pro workspaces are copied (copyRepo) from a cache clone that carries
 * the FULL upstream history and a live `origin` remote — including the future
 * fix commit and post-fix hidden-test files, all reachable via
 * refs/remotes/origin/*. An agent with `git` + Bash can read the graded answer
 * OFFLINE (`git log HEAD..origin/master -- <f>`, `git show <test_commit>:<f>`),
 * bypassing the #666 web-tool gate entirely (issue #924).
 *
 * The copy is disposable (one per task), so we neutralize it: drop every remote
 * and delete every ref, leaving only the detached `base_commit` HEAD that cache
 * prep checked out, then prune the now-unreachable objects so the future commits
 * cannot be recovered via `git fsck`/`git cat-file`. The detached HEAD and
 * working tree are untouched, so the grading `git diff HEAD` (main.ts) still
 * yields the agent's patch. No-op when there is no `.git` (polyglot/ARC).
 *
 * The shared cache keeps its `origin` for narrow updates — only the per-task
 * copy is sanitized.
 *
 * Alternates: cache prep may create the cache via `git clone --shared` (issue
 * #924 follow-up), so the copy borrows objects from a sibling cache via
 * `.git/objects/info/alternates` (an absolute host path). `gc` cannot prune
 * objects that live in the alternate, so we first `git repack -a -d` (no `-l`)
 * to absorb the reachable objects into a local pack — dropping the unreachable
 * future ones — then delete the alternates link so the borrowed future objects
 * become unrecoverable on the host too, not just inside the (unmounted-alternate)
 * container.
 *
 * Submodules: an initialized git submodule ships a SEPARATE nested object store
 * at `.git/modules/<path>` (full history + live `origin`) that the top-level
 * strip above does not reach; we delete `.git/modules` wholesale so a submodule
 * cannot leak answer-bearing history offline (issue #1256). Working-tree files
 * are untouched.
 *
 * Fail-closed: individual git steps are best-effort, but a final verification
 * asserts the invariant (HEAD valid, no remotes/refs, only base reachable, no
 * unreachable objects, no alternates link, no `.git/modules`) and THROWS on
 * violation. A security
 * control must not silently no-op — a swallowed gc failure or missing git would
 * otherwise leave the graded answer recoverable with zero signal, and (under
 * `--wipe-workspace-per-task false`) get cached as "done" via the repo stamp.
 */
export function sanitizeRepoHistory(repoDst: string): void {
  const gitPath = path.join(repoDst, '.git');
  if (!fs.existsSync(gitPath)) {
    return;
  }
  // `.git` must be a real directory living inside repoDst. A gitdir-pointer
  // FILE (`git worktree` / `clone --separate-git-dir`) or a symlink would make
  // the `git -C repoDst ...` calls below mutate an EXTERNAL git dir — deleting
  // refs / running gc on someone else's repo or the shared cache. Per-task
  // workspace repos are always self-contained (repo_cache.py uses plain
  // `git clone`), so fail closed rather than follow the pointer.
  if (!fs.lstatSync(gitPath).isDirectory()) {
    throw new Error(
      `sanitizeRepoHistory refusing to operate: ${gitPath} is not a real ` +
        `directory (gitdir pointer/symlink); workspace repo must be self-contained`,
    );
  }
  const git = (args: string[], input?: string): void => {
    try {
      execFileSync('git', ['-C', repoDst, ...args], {
        stdio: input === undefined ? 'ignore' : ['pipe', 'ignore', 'ignore'],
        input,
      });
    } catch {
      // Best-effort: a missing remote/ref or an old git is non-fatal here —
      // the final verifySanitized() below is the authoritative gate.
    }
  };
  const capture = (args: string[]): string => {
    try {
      return execFileSync('git', ['-C', repoDst, ...args], {
        encoding: 'utf-8',
        stdio: ['ignore', 'pipe', 'ignore'],
      });
    } catch {
      return '';
    }
  };
  // Pin HEAD to its current commit as a detached ref so deleting branch refs
  // below cannot orphan it (cache prep already detaches at base_commit; this is
  // a safety net for the on-a-branch case).
  git(['checkout', '-q', '--detach', 'HEAD']);
  // Drop all remotes — also kills `git fetch origin <future-sha>`.
  for (const remote of capture(['remote']).split('\n').map((s) => s.trim()).filter(Boolean)) {
    git(['remote', 'remove', remote]);
  }
  // Delete every ref in one batch; the detached HEAD keeps base_commit reachable.
  const refs = capture(['for-each-ref', '--format=%(refname)', ...SANITIZE_REF_NAMESPACES])
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean);
  if (refs.length > 0) {
    git(['update-ref', '--stdin'], refs.map((r) => `delete ${r}\n`).join(''));
  }
  // Expire reflogs so they cannot keep future objects reachable.
  git(['reflog', 'expire', '--expire=now', '--all']);
  // For a `--shared` copy the future objects live in the alternate, which `gc`
  // cannot prune. Absorb the reachable objects locally (repack WITHOUT `-l`),
  // then drop the alternates link so the borrowed future objects are gone.
  const alternatesPath = path.join(repoDst, '.git', 'objects', 'info', 'alternates');
  if (fs.existsSync(alternatesPath)) {
    git(['repack', '-a', '-d']);
    removeFileIfExists(alternatesPath);
  }
  // Physically prune the now-unreachable future objects.
  git(['gc', '--prune=now', '--quiet']);

  // Submodule git stores (`.git/modules/<path>`) are SEPARATE object stores the
  // top-level history strip above never touches: each carries the submodule's
  // full history plus a live `origin` remote, and `copyRepo`'s cpSync copies
  // them verbatim into the disposable per-task workspace. An initialized
  // submodule (issue #1256: SWE-bench Pro repos like openlibrary vendor a
  // git-submodule dependency) would therefore leave the submodule's history +
  // remote reachable in-container via `git -C <submodule> log --all` — the same
  // offline-answer-leak class as #924, but for any answer-bearing submodule.
  // The agent needs only the submodule WORKING-TREE files (already copied as
  // plain files), never its git metadata, so remove the nested git stores
  // wholesale. The submodule `<path>/.git` gitfiles then dangle harmlessly
  // (nothing left to reach).
  const modulesPath = path.join(repoDst, '.git', 'modules');
  removePathIfExists(modulesPath);

  verifySanitized(repoDst, capture, alternatesPath, modulesPath);
}

/**
 * Authoritative post-sanitization check. Throws if the workspace repo could
 * still leak the graded answer — see sanitizeRepoHistory's fail-closed note.
 */
function verifySanitized(
  repoDst: string,
  capture: (args: string[]) => string,
  alternatesPath: string,
  modulesPath: string,
): void {
  const fail = (reason: string): never => {
    throw new Error(`sanitizeRepoHistory failed for ${repoDst}: ${reason}`);
  };
  // git must actually work and HEAD must resolve (catches missing/old git: a
  // broken git makes every capture() return '' which would otherwise look like
  // "already clean").
  const head = capture(['rev-parse', 'HEAD']).trim();
  if (!/^[0-9a-f]{40}$/.test(head)) {
    fail(`HEAD does not resolve to a commit (got ${JSON.stringify(head)})`);
  }
  const remotes = capture(['remote']).trim();
  if (remotes) {
    fail(`remotes still present: ${remotes.replace(/\n/g, ',')}`);
  }
  const refs = capture(['for-each-ref', '--format=%(refname)', ...SANITIZE_REF_NAMESPACES]).trim();
  if (refs) {
    fail(`refs still present: ${refs.replace(/\n/g, ',')}`);
  }
  // Nothing may stay reachable outside the detached HEAD's own history. A real
  // base_commit is deep in history with thousands of legitimate ancestors, so
  // the invariant is "no commit reachable from any ref that is not an ancestor
  // of HEAD" (`--all --not HEAD` == 0), NOT "exactly one commit total" (which
  // only holds for a root base_commit and would brick every real task). `--all`
  // still spans every ref namespace — including ones not in
  // SANITIZE_REF_NAMESPACES (refs/stash, refs/bisect, …) — so a future commit
  // reachable via any stray ref is still caught here.
  const beyondHead = capture(['rev-list', '--all', '--not', 'HEAD', '--count']).trim();
  if (beyondHead !== '0') {
    fail(`commits reachable beyond base history: ${JSON.stringify(beyondHead)}`);
  }
  const unreachable = capture(['fsck', '--unreachable', '--no-dangling'])
    .split('\n')
    .map((s) => s.trim())
    .filter((l) => l.startsWith('unreachable '));
  if (unreachable.length > 0) {
    fail(`${unreachable.length} unreachable object(s) survived gc (e.g. ${unreachable[0]})`);
  }
  if (fs.existsSync(alternatesPath)) {
    fail('alternates link still present — borrowed objects remain recoverable');
  }
  // Submodule git stores must be gone: a surviving `.git/modules/<path>` keeps
  // the submodule's history + `origin` remote reachable in-container (#1256).
  if (fs.existsSync(modulesPath)) {
    fail('.git/modules still present — submodule git stores remain recoverable');
  }
}

/**
 * Compute a content-based fingerprint of `src` for the workspace repo stamp.
 *
 * Walks the source tree and hashes (relpath, size, file_mtime) tuples sorted
 * by relpath. We use *file* mtime (not directory mtime) because on ext4 the
 * directory mtime does NOT propagate when a file's content changes — using
 * the dir's mtime would silently miss source updates and reuse a stale repo.
 *
 * Symlinks are captured by their target string (not followed) so symlink
 * changes invalidate the stamp without risking infinite traversal. Each
 * call is O(n) in the file count of the source; bounded enough for typical
 * benchmark sources (ARC: hundreds of files; SWE-bench Pro: low thousands).
 *
 * Returns a 16-char sha1 prefix — collision-resistant for our scale, short
 * enough to keep the stamp file compact.
 */
function computeRepoFingerprint(srcResolved: string): string {
  const hash = crypto.createHash('sha1');
  const entries: Array<{ rel: string; payload: string }> = [];

  function walk(absPath: string, relPath: string): void {
    let stat: fs.Stats;
    try {
      stat = fs.lstatSync(absPath);
    } catch {
      return;
    }
    if (stat.isSymbolicLink()) {
      let target = '';
      try {
        target = fs.readlinkSync(absPath);
      } catch {
        // ignore unreadable link
      }
      entries.push({ rel: `L:${relPath}`, payload: `link|${target}` });
      return;
    }
    if (stat.isDirectory()) {
      let children: string[] = [];
      try {
        children = fs.readdirSync(absPath);
      } catch {
        return;
      }
      children.sort();
      for (const child of children) {
        walk(path.join(absPath, child), relPath ? `${relPath}/${child}` : child);
      }
      return;
    }
    if (stat.isFile()) {
      entries.push({ rel: relPath, payload: `file|${stat.size}|${stat.mtimeMs}` });
    }
  }

  walk(srcResolved, '');
  entries.sort((a, b) => (a.rel < b.rel ? -1 : a.rel > b.rel ? 1 : 0));
  for (const e of entries) {
    hash.update(`${e.rel}|${e.payload}\n`);
  }
  return hash.digest('hex').slice(0, 16);
}

export function seedWorkspace(
  payload: KsiPayload,
  workspaceKey: string,
  wipeWorkspacePerTask: boolean,
): void {
  const workspaceRootPath = resolveWorkspaceRootPath(workspaceKey);
  ensureDir(workspaceRootPath);
  const seedContextPath = path.join(workspaceRootPath, '.seed_context');
  removeFileIfExists(seedContextPath);

  const workspaceRoot = path.join(workspaceRootPath, 'workspace');
  if (wipeWorkspacePerTask) {
    removePathIfExists(workspaceRoot);
  }
  ensureDir(workspaceRoot);

  const seed = payload.workspace_seed || {};
  const instruction = (seed.instruction_md || '').trim() + '\n';
  const seedMemory = (seed.memory_md || '').trim();
  const memory = buildWorkspaceMemoryMd(seedMemory).trim() + '\n';
  const taskMd = (seed.task_md || '').trim() + '\n';
  const taskFiles = seed.task_files || {};

  if (seedMemory) {
    writeText(seedContextPath, seedMemory + '\n');
  }

  writeText(path.join(workspaceRoot, 'INSTRUCTION.md'), instruction);
  writeText(path.join(workspaceRoot, 'MEMORY.md'), memory);

  const tools = (seed.tools_md || '').trim();
  if (tools) {
    writeText(path.join(workspaceRoot, 'TOOLS.md'), tools + '\n');
  }
  writeText(path.join(workspaceRoot, 'TASK.md'), taskMd);
  for (const [name, content] of Object.entries(taskFiles)) {
    const fileName = path.basename(String(name || '').trim());
    if (!fileName || fileName === 'TASK.md') continue;
    writeText(path.join(workspaceRoot, fileName), (content || '').trim() + '\n');
  }

  const repoDst = path.join(workspaceRoot, 'repo');
  const repoSource = seed.repo_source_path || '';
  const stampPath = path.join(repoDst, '.repo-stamp');

  // The stamp captures (resolved-source-path, content-fingerprint,
  // sanitizer-version). Using a content fingerprint (not directory mtime)
  // because on Linux ext4 the parent dir's mtime does not propagate when a
  // file inside it changes — an mtime-only stamp would silently reuse a stale
  // repo on real source updates. See computeRepoFingerprint for the walk
  // strategy. The sanitizer-version (SANITIZE_VERSION) is part of the stamp so
  // a workspace copied before the .git sanitizer existed (or under an older
  // sanitizer) does NOT match the fast path and is re-copied + re-sanitized —
  // otherwise, under --wipe-workspace-per-task false, a pre-sanitizer stamped
  // SWE-bench Pro workspace would be reused with its full upstream .git history
  // (the future fix commit + hidden tests) still readable by the solver. See
  // sanitizeRepoHistory / issue #924.
  let needsCopy = true;
  if (!wipeWorkspacePerTask && repoSource && fs.existsSync(stampPath)) {
    try {
      const stamp = fs.readFileSync(stampPath, 'utf-8').trim();
      const srcResolved = path.resolve(repoSource);
      if (fs.existsSync(srcResolved)) {
        const fingerprint = computeRepoFingerprint(srcResolved);
        if (stamp === stampValue(srcResolved, fingerprint)) {
          needsCopy = false;
        }
      }
    } catch {
      // Stamp unreadable or source stat failed — fall through to copy.
    }
  }

  if (needsCopy) {
    removePathIfExists(repoDst);
    copyRepo(repoSource, repoDst);
    if (repoSource) {
      try {
        const srcResolved = path.resolve(repoSource);
        if (fs.existsSync(srcResolved)) {
          const fingerprint = computeRepoFingerprint(srcResolved);
          writeText(stampPath, stampValue(srcResolved, fingerprint));
        }
      } catch {
        // Non-fatal — worst case is an extra copy next time.
      }
    }
  }
}

export function buildPrompt(payload: KsiPayload): string {
  const instruction = (payload.execution_prompt || '').trim();
  const configuredRepoPath = String(payload.runtime?.repo_container_path || '').trim();
  const repoContainerPath = configuredRepoPath.startsWith('/')
    ? configuredRepoPath.replace(/\/+$/, '') || DEFAULT_REPO_CONTAINER_PATH
    : DEFAULT_REPO_CONTAINER_PATH;
  const taskHint = [
    'Use this workspace:',
    '- Only edit files under the active task repo path.',
    '- Shared guidance:',
    `  - ${CONTAINER_ACTIVE_WORKSPACE_DIR}/INSTRUCTION.md`,
    `  - ${CONTAINER_WORKSPACE_ROOT}/.seed_context (the full seeded memory payload injected into the system prompt when present)`,
    `  - ${CONTAINER_ACTIVE_WORKSPACE_DIR}/MEMORY.md (full seeded memory bundle when present)`,
    `  - ${CONTAINER_ACTIVE_WORKSPACE_DIR}/TOOLS.md (available tools and when to use them)`,
    `- Active task workspace: ${CONTAINER_ACTIVE_WORKSPACE_DIR}`,
    `  - ${CONTAINER_ACTIVE_WORKSPACE_DIR}/TASK.md`,
    `  - ${repoContainerPath}`,
  ].join('\n');
  return `${instruction}\n\n${taskHint}\n`;
}
