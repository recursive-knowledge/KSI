import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

import { synthesizeArcSubmitTraceFromPrediction } from './arc_nomcp_synth.js';
import { runContainerAgent } from './container_runner.js';
import { RegisteredWorkspace } from './container_types.js';

import { collectNativeSessionMemory } from './native_memory.js';
import { cleanupWorkspace, loadSessionForAgent, saveSessionForAgent } from './sessions.js';
import { ContainerOutput } from './shared_types.js';
import { RuntimeScope, KcsiPayload } from './types.js';
import {
  CONTAINER_ACTIVE_WORKSPACE_DIR,
  buildPrompt,
  seedWorkspace,
  toWorkspaceKey,
} from './workspace.js';
import { resolveWorkspaceRootPath } from './workspace_scope.js';

function die(msg: string): never {
  throw new Error(msg);
}

function readJsonFile<T>(filePath: string): T {
  const raw = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(raw) as T;
}

/**
 * Validate the parsed payload shape at the process boundary. The unsound
 * `readJsonFile<KcsiPayload>` cast otherwise lets a malformed payload (null,
 * a non-object, or one missing `agent_id`/`task.id`) flow downstream and
 * silently surface as `undefined` dereferences. Fail loudly instead.
 */
function assertKcsiPayload(
  value: unknown,
  filePath: string,
): asserts value is KcsiPayload {
  function fail(pathName: string, expectation: string): never {
    die(`payload at ${filePath} ${pathName} must be ${expectation}`);
  }
  function requireString(obj: Record<string, unknown>, key: string, pathName: string): void {
    if (typeof obj[key] !== 'string' || !obj[key]) {
      fail(`${pathName}.${key}`, 'a non-empty string');
    }
  }
  function optionalString(obj: Record<string, unknown>, key: string, pathName: string): void {
    if (obj[key] !== undefined && typeof obj[key] !== 'string') {
      fail(`${pathName}.${key}`, 'a string when present');
    }
  }
  function optionalBoolean(obj: Record<string, unknown>, key: string, pathName: string): void {
    if (obj[key] !== undefined && typeof obj[key] !== 'boolean') {
      fail(`${pathName}.${key}`, 'a boolean when present');
    }
  }
  function optionalNumber(obj: Record<string, unknown>, key: string, pathName: string): void {
    if (obj[key] !== undefined || Object.prototype.hasOwnProperty.call(obj, key)) {
      if (typeof obj[key] !== 'number' || !Number.isFinite(obj[key])) {
        fail(`${pathName}.${key}`, 'a finite number when present');
      }
    }
  }
  function requireNumber(obj: Record<string, unknown>, key: string, pathName: string): void {
    if (typeof obj[key] !== 'number' || !Number.isFinite(obj[key])) {
      fail(`${pathName}.${key}`, 'a finite number');
    }
  }
  function objectAt(
    obj: Record<string, unknown>,
    key: string,
    pathName: string,
  ): Record<string, unknown> | undefined {
    const child = obj[key];
    if (child === undefined) return undefined;
    if (child === null || typeof child !== 'object' || Array.isArray(child)) {
      fail(`${pathName}.${key}`, 'a JSON object when present');
    }
    return child as Record<string, unknown>;
  }
  function stringRecord(obj: Record<string, unknown>, key: string, pathName: string): void {
    const record = objectAt(obj, key, pathName);
    if (!record) return;
    for (const [childKey, childValue] of Object.entries(record)) {
      if (typeof childValue !== 'string') {
        fail(`${pathName}.${key}.${childKey}`, 'a string');
      }
    }
  }
  function requireStringArray(obj: Record<string, unknown>, key: string, pathName: string): void {
    const valueAtKey = obj[key];
    if (!Array.isArray(valueAtKey) || valueAtKey.some((item) => typeof item !== 'string')) {
      fail(`${pathName}.${key}`, 'an array of strings');
    }
  }

  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    die(`payload at ${filePath} is not a JSON object`);
  }
  const obj = value as Record<string, unknown>;
  requireNumber(obj, 'generation', 'root');
  if (typeof obj.agent_id !== 'string' || !obj.agent_id) {
    die(`payload at ${filePath} is missing required string field "agent_id"`);
  }
  optionalString(obj, 'experiment_name', 'root');
  optionalString(obj, 'execution_prompt', 'root');
  optionalBoolean(obj, 'arc_no_mcp', 'root');
  if (
    obj.task === null ||
    typeof obj.task !== 'object' ||
    Array.isArray(obj.task)
  ) {
    die(`payload at ${filePath} is missing required object field "task"`);
  }
  const task = obj.task as Record<string, unknown>;
  if (typeof task.id !== 'string' || !task.id) {
    die(`payload at ${filePath} task is missing required string field "id"`);
  }
  optionalString(task, 'repo', 'task');
  optionalString(task, 'prompt', 'task');
  objectAt(task, 'metadata', 'task');

  const workspaceSeed = objectAt(obj, 'workspace_seed', 'root');
  if (workspaceSeed) {
    for (const key of ['instruction_md', 'memory_md', 'task_md', 'tools_md', 'repo_source_path']) {
      optionalString(workspaceSeed, key, 'workspace_seed');
    }
    stringRecord(workspaceSeed, 'task_files', 'workspace_seed');
  }

  const runtime = objectAt(obj, 'runtime', 'root');
  if (runtime) {
    const sessionScope = runtime.session_scope;
    if (sessionScope !== undefined && sessionScope !== 'task' && sessionScope !== 'agent') {
      fail('runtime.session_scope', '"task" or "agent" when present');
    }
    optionalBoolean(runtime, 'wipe_workspace_per_task', 'runtime');
    for (const key of [
      'container_image',
      'official_container_image',
      'runner_image',
      'repo_container_path',
      'official_repo_container_path',
      'runner_root',
    ]) {
      optionalString(runtime, key, 'runtime');
    }
  }

  const knowledge = objectAt(obj, 'knowledge', 'root');
  if (knowledge) {
    requireString(knowledge, 'db_path', 'knowledge');
    requireString(knowledge, 'mcp_server_dir', 'knowledge');
    optionalString(knowledge, 'snapshot_path', 'knowledge');
    optionalBoolean(knowledge, 'disable_memory_tools', 'knowledge');
    optionalNumber(knowledge, 'forum_generation', 'knowledge');
    optionalString(knowledge, 'experiment_name', 'knowledge');
  }

  const runtimeAudit = objectAt(obj, 'runtime_audit', 'root');
  if (runtimeAudit) {
    requireString(runtimeAudit, 'db_path', 'runtime_audit');
  }

  const arcTools = objectAt(obj, 'arc_tools', 'root');
  if (arcTools) {
    if (typeof arcTools.enable !== 'boolean') {
      fail('arc_tools.enable', 'a boolean');
    }
    requireString(arcTools, 'mcp_server_dir', 'arc_tools');
    optionalString(arcTools, 'task_source', 'arc_tools');
    optionalString(arcTools, 'task_id', 'arc_tools');
    optionalString(arcTools, 'snapshot_path', 'arc_tools');
  }

  const phase1Reflection = objectAt(obj, 'phase1_reflection', 'root');
  if (phase1Reflection) {
    if (typeof phase1Reflection.enabled !== 'boolean') {
      fail('phase1_reflection.enabled', 'a boolean');
    }
    optionalNumber(phase1Reflection, 'eval_result_poll_timeout_ms', 'phase1_reflection');
  }

  const crossTaskSharedContainer = objectAt(obj, 'cross_task_shared_container', 'root');
  if (crossTaskSharedContainer) {
    if (typeof crossTaskSharedContainer.enabled !== 'boolean') {
      fail('cross_task_shared_container.enabled', 'a boolean');
    }
    optionalString(crossTaskSharedContainer, 'barrier_name', 'cross_task_shared_container');
    optionalNumber(crossTaskSharedContainer, 'response_poll_timeout_ms', 'cross_task_shared_container');
  }

  const polyglotFeedback = objectAt(obj, 'polyglot_test_feedback', 'root');
  if (polyglotFeedback) {
    if (typeof polyglotFeedback.enabled !== 'boolean') {
      fail('polyglot_test_feedback.enabled', 'a boolean');
    }
    requireString(polyglotFeedback, 'agentId', 'polyglot_test_feedback');
    requireNumber(polyglotFeedback, 'triesRemaining', 'polyglot_test_feedback');
    requireNumber(polyglotFeedback, 'maxLines', 'polyglot_test_feedback');
    requireString(polyglotFeedback, 'fileList', 'polyglot_test_feedback');
    requireStringArray(polyglotFeedback, 'allowedTools', 'polyglot_test_feedback');
    if (!objectAt(polyglotFeedback, 'mcpServers', 'polyglot_test_feedback')) {
      fail('polyglot_test_feedback.mcpServers', 'a JSON object');
    }
    requireNumber(polyglotFeedback, 'maxTurnsPerRound', 'polyglot_test_feedback');
    optionalNumber(polyglotFeedback, 'evalResultPollTimeoutMs', 'polyglot_test_feedback');
  }
}

/** Coerce an unknown metadata value to a finite number, or undefined. */
function coerceOptionalNumber(value: unknown): number | undefined {
  if (value === undefined || value === null || value === '') {
    return undefined;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

/** Coerce an unknown metadata value to a non-empty string, or undefined. */
function coerceOptionalString(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  const s = String(value);
  return s.length > 0 ? s : undefined;
}

function parseStructuredToolOutput(
  value: unknown,
): Record<string, unknown> | null {
  if (!value) {
    return null;
  }
  if (typeof value === 'object' && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    if (
      typeof record.type === 'string' &&
      record.type === 'text' &&
      typeof record.text === 'string'
    ) {
      try {
        const parsed = JSON.parse(record.text);
        return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
          ? (parsed as Record<string, unknown>)
          : null;
      } catch {
        return null;
      }
    }
    return record;
  }
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>)
        : null;
    } catch {
      return null;
    }
  }
  return null;
}

function normalizeToolName(name: string): string {
  if (name.startsWith('mcp__memory__')) {
    return name.slice('mcp__memory__'.length);
  }
  return name;
}

function excerptValue(value: unknown, maxChars = 220): string {
  if (value == null) {
    return '';
  }
  let text = '';
  if (typeof value === 'string') {
    text = value;
  } else {
    try {
      text = JSON.stringify(value);
    } catch {
      text = String(value);
    }
  }
  const compact = text.replace(/\s+/g, ' ').trim();
  if (compact.length <= maxChars) {
    return compact;
  }
  return `${compact.slice(0, maxChars - 3)}...`;
}

function objectKeys(value: unknown): string[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
  return Object.keys(value as Record<string, unknown>);
}

function buildNewFileDiff(relPath: string, content: string): string {
  const body = content.endsWith('\n') ? content.slice(0, -1) : content;
  const lines = body ? body.split('\n') : [];
  const hunkSize = lines.length;
  const patch = [
    `diff --git a/${relPath} b/${relPath}`,
    'new file mode 100644',
    '--- /dev/null',
    `+++ b/${relPath}`,
    `@@ -0,0 +1,${hunkSize} @@`,
    ...lines.map((line) => `+${line}`),
  ];
  return `${patch.join('\n')}\n`;
}

function resolveRepoRelativePath(repoDir: string, relPath: string): string | null {
  if (!relPath || path.isAbsolute(relPath)) return null;
  const repoRoot = path.resolve(repoDir);
  const fullPath = path.resolve(repoRoot, relPath);
  if (!fullPath.startsWith(repoRoot + path.sep)) return null;
  return fullPath;
}

function pathInside(parent: string, child: string): boolean {
  const rel = path.relative(parent, child);
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

function readSmallRepoTextFile(repoDir: string, relPath: string, maxBytes = 1_000_000): string | null {
  try {
    const fullPath = resolveRepoRelativePath(repoDir, relPath);
    if (!fullPath) return null;
    const linkStat = fs.lstatSync(fullPath);
    if (!linkStat.isFile() || linkStat.size > maxBytes) return null;

    const repoReal = fs.realpathSync(repoDir);
    const fileReal = fs.realpathSync(fullPath);
    if (!pathInside(repoReal, fileReal)) return null;

    const stat = fs.statSync(fullPath);
    if (!stat.isFile() || stat.size > maxBytes) return null;
    return fs.readFileSync(fullPath, 'utf-8');
  } catch {
    return null;
  }
}

function gitOutput(args: string[], cwd: string): { output: string; error?: string } {
  try {
    return { output: execFileSync('git', args, {
      cwd,
      encoding: 'utf-8',
      maxBuffer: 32 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'pipe'],
    }) };
  } catch (err) {
    // Do not silently swallow git failures: a non-zero ``git diff`` (e.g. a
    // broken submodule gitlink) previously masqueraded as an empty diff,
    // silently losing the agent's patch. Log it so the failure is visible in
    // the run log even though we still return '' to keep the caller resilient.
    const stderr = (err as { stderr?: Buffer | string })?.stderr;
    const detail = stderr ? String(stderr).trim().split('\n').slice(0, 3).join(' | ') : String(err);
    console.error(`[workspace-diff] git ${args.join(' ')} failed in ${cwd}: ${detail}`);
    return { output: '', error: detail };
  }
}

// Path patterns excluded from the workspace diff: cache/build artifacts that
// some upstream repos forget to .gitignore. Without this filter, an agent's
// test run produces __pycache__ or .pyc files that get reported as a synthetic
// new-file diff and submitted to the grader as "the patch", drowning out the
// real edits and causing false-negative SWE-bench Pro scores.
const WORKSPACE_DIFF_NOISE_PATHSPEC = [
  ':(exclude,glob)**/__pycache__/**',
  ':(exclude,glob)**/*.pyc',
  ':(exclude,glob)**/*.pyo',
  ':(exclude,glob)**/.pytest_cache/**',
  ':(exclude,glob)**/.mypy_cache/**',
  ':(exclude,glob)**/.ruff_cache/**',
  ':(exclude,glob)**/.coverage',
  ':(exclude,glob)**/.coverage.*',
  ':(exclude,glob)**/.DS_Store',
  ':(exclude,glob)**/*.egg-info/**',
  ':(exclude,glob)**/*.log',
  // .repo-stamp is a kcsi-internal seeding-fingerprint file written by
  // workspace.ts at the repo root. It is host-state, not part of the
  // benchmark patch. Suppress to keep the workspace diff focused on the
  // agent's actual edits.
  ':(exclude,glob).repo-stamp',
];

const WORKSPACE_DIFF_NOISE_RE = /(?:^|\/)(?:__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|node_modules|\.DS_Store|\.repo-stamp)(?:\/|$)|\.pyc$|\.pyo$|\.log$|\.coverage(?:\..*)?$|\.egg-info(?:\/|$)/;

function isNoisePath(relPath: string): boolean {
  return WORKSPACE_DIFF_NOISE_RE.test(relPath);
}

export function captureWorkspaceDiff(repoDir: string): { diff: string; changedFiles: string[]; captureError?: string } {
  if (!fs.existsSync(repoDir)) {
    return { diff: '', changedFiles: [] };
  }
  // Polyglot exercise repos and any other workspace seeded from non-git source
  // have no own .git. Without this guard, ``git diff`` walks up the filesystem
  // looking for a repo and ends up capturing the kcsi host repo's
  // working-tree diff as the agent's patch — pure noise that pollutes
  // workspace_diff / workspace_changed_files for any downstream consumer.
  // Note: ``.git`` may be a directory (normal repo) or a regular file (linked
  // worktree); ``existsSync`` covers both.
  if (!fs.existsSync(path.join(repoDir, '.git'))) {
    return { diff: '', changedFiles: [] };
  }
  // ``--ignore-submodules=all`` is load-bearing: SWE-bench Pro instances whose
  // submodules failed to clone (private repo, dead ``git://`` URL, network
  // timeout) are left with a broken gitlink, and a plain ``git diff HEAD``
  // then exits 128 (``fatal: not a git repository: .../.git/modules/...``),
  // which gitOutput swallows to '' — silently dropping every real tracked-file
  // edit the agent made and yielding a bogus empty patch (scored ``no_patch``).
  // Ignoring submodules keeps the diff exit-0 and captures the agent's edits.
  const diffArgs = ['diff', '--no-ext-diff', '--binary', '--ignore-submodules=all', 'HEAD', '--', ...WORKSPACE_DIFF_NOISE_PATHSPEC];
  const nameArgs = ['diff', '--name-only', '--ignore-submodules=all', 'HEAD', '--', ...WORKSPACE_DIFF_NOISE_PATHSPEC];
  const tracked = gitOutput(diffArgs, repoDir);
  const names = gitOutput(nameArgs, repoDir);
  const untracked = gitOutput(['ls-files', '--others', '--exclude-standard'], repoDir);
  const captureError = tracked.error || names.error || untracked.error;
  const trackedNames = names.output
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !isNoisePath(line));
  const untrackedNames = untracked.output
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !isNoisePath(line));
  const untrackedDiffs: string[] = [];
  for (const relPath of untrackedNames.slice(0, 50)) {
    const content = readSmallRepoTextFile(repoDir, relPath);
    if (content != null) {
      untrackedDiffs.push(buildNewFileDiff(relPath, content));
    }
  }
  return {
    diff: [tracked.output, ...untrackedDiffs].filter(Boolean).join('\n'),
    changedFiles: Array.from(new Set([...trackedNames, ...untrackedNames])),
    ...(captureError ? { captureError } : {}),
  };
}

function walkFiles(root: string, limit = 80): string[] {
  const out: string[] = [];
  function visit(dir: string): void {
    if (out.length >= limit) return;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (out.length >= limit) return;
      if (entry.name === '.git' || entry.name === '.repo-stamp') continue;
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        visit(fullPath);
      } else if (entry.isFile()) {
        out.push(fullPath);
      }
    }
  }
  visit(root);
  return out;
}

function collectPolyglotSolutionFiles(repoDir: string, taskMeta: Record<string, unknown>): Record<string, string> {
  if (!fs.existsSync(repoDir)) return {};
  const language = String(taskMeta.language || '').toLowerCase();
  const defaultFiles: Record<string, string> = {
    python: 'solution.py',
    rust: 'src/lib.rs',
    go: 'solution.go',
    javascript: 'solution.js',
    java: 'Solution.java',
    cpp: 'solution.cpp',
  };
  const extensions: Record<string, string[]> = {
    python: ['.py'],
    rust: ['.rs'],
    go: ['.go'],
    javascript: ['.js'],
    java: ['.java'],
    cpp: ['.cpp', '.cc', '.cxx', '.hpp', '.h'],
  };
  const excluded = new Set([
    ...objectKeys(taskMeta.test_files),
    ...objectKeys(taskMeta.build_files),
  ]);
  const candidates = new Set(objectKeys(taskMeta.starter_code));
  if (defaultFiles[language]) candidates.add(defaultFiles[language]);

  const solutionFiles: Record<string, string> = {};
  for (const relPath of candidates) {
    if (excluded.has(relPath)) continue;
    const content = readSmallRepoTextFile(repoDir, relPath);
    if (content != null) solutionFiles[relPath] = content;
  }
  if (Object.keys(solutionFiles).length > 0) return solutionFiles;

  const allowedExt = extensions[language] || [];
  for (const fullPath of walkFiles(repoDir)) {
    const relPath = path.relative(repoDir, fullPath);
    const base = path.basename(relPath).toLowerCase();
    if (excluded.has(relPath) || base.includes('test')) continue;
    if (allowedExt.length > 0 && !allowedExt.some((ext) => relPath.endsWith(ext))) continue;
    const content = readSmallRepoTextFile(repoDir, relPath);
    if (content != null) solutionFiles[relPath] = content;
    if (Object.keys(solutionFiles).length >= 12) break;
  }
  return solutionFiles;
}

// Generic (non-language-aware) counterpart to collectPolyglotSolutionFiles,
// for task sources with no `language` metadata to key off of. Reads every
// file under repoDir, capped at 12 (same cap as the polyglot fallback loop
// above) — fine for small self-contained demo tasks; a task with more than
// 12 files will silently lose the extras from this content-capture channel
// (host_workspace_repo_dir, when it survives, is unaffected). Mirrors the
// polyglot fallback's exclusions: skip `score.json` (never let a captured
// copy masquerade as the eval command's own override — see
// kcsi.eval.command's matching forgery-close guard), any basename
// containing `test` (case-insensitive, same rule as
// collectPolyglotSolutionFiles), and `__pycache__/`/`*.pyc` noise.
function collectGenericSolutionFiles(repoDir: string): Record<string, string> {
  if (!fs.existsSync(repoDir)) return {};
  const solutionFiles: Record<string, string> = {};
  for (const fullPath of walkFiles(repoDir)) {
    const relPath = path.relative(repoDir, fullPath);
    const base = path.basename(relPath).toLowerCase();
    if (base === 'score.json' || base.includes('test')) continue;
    if (base.endsWith('.pyc') || relPath.split(path.sep).includes('__pycache__')) continue;
    const content = readSmallRepoTextFile(repoDir, relPath);
    if (content != null) solutionFiles[relPath] = content;
    if (Object.keys(solutionFiles).length >= 12) break;
  }
  return solutionFiles;
}

function captureWorkspaceArtifacts(
  workspaceKey: string,
  taskSource: string,
  taskMeta: Record<string, unknown>,
): { repoDir: string; diff: string; changedFiles: string[]; captureError?: string; solutionFiles: Record<string, string> } {
  const workspaceRoot = resolveWorkspaceRootPath(workspaceKey);
  const repoDir = path.join(workspaceRoot, 'workspace', 'repo');
  const codingTask = taskSource === 'swebench_pro' || taskSource === 'polyglot' || taskSource === 'custom';
  if (!codingTask) {
    return { repoDir, diff: '', changedFiles: [], solutionFiles: {} };
  }
  if (taskSource === 'custom') {
    // `custom` seeds repoDir from a plain (non-git) directory (kcsi.tasks.custom),
    // so the git-diff-based capture below always returns empty for it —
    // captureWorkspaceDiff bails out whenever `.git` is absent. Capture file
    // CONTENT instead (like polyglot's solution-file fallback): this
    // survives even when the workspace is later wiped before the `command`
    // evaluator (kcsi.eval.command) runs, unlike a bare directory path.
    const solutionFiles = collectGenericSolutionFiles(repoDir);
    // Note: `changedFiles` here is every captured file, not a real diff
    // against a prior state (there is no git history to diff against) —
    // it only exists to satisfy the "did the agent produce anything"
    // gate below that decides whether to publish `host_workspace_repo_dir`.
    return { repoDir, diff: '', changedFiles: Object.keys(solutionFiles), solutionFiles };
  }
  const { diff, changedFiles, captureError } = captureWorkspaceDiff(repoDir);
  const solutionFiles = taskSource === 'polyglot'
    ? collectPolyglotSolutionFiles(repoDir, taskMeta)
    : {};
  return { repoDir, diff, changedFiles, captureError, solutionFiles };
}

function buildRuntimeTranscript(
  rawTrace: Array<Record<string, unknown>>,
  resultText: string,
): string {
  const lines: string[] = ['# runtime_transcript'];
  for (const entry of rawTrace) {
    if (entry.type === 'tool_call' && typeof entry.tool_name === 'string') {
      lines.push(`tool_call: ${entry.tool_name}`);
      const toolInput = excerptValue(entry.tool_input);
      if (toolInput) {
        lines.push(`tool_input: ${toolInput}`);
      }
      const toolOutput = excerptValue(entry.tool_output);
      if (toolOutput) {
        lines.push(`tool_output: ${toolOutput}`);
      }
      continue;
    }
    if (entry.type === 'reasoning' && typeof entry.text === 'string') {
      const text = excerptValue(entry.text, 400);
      if (text) {
        lines.push(`reasoning: ${text}`);
      }
      continue;
    }
    if (entry.type === 'message' && typeof entry.text === 'string') {
      const text = excerptValue(entry.text, 400);
      if (text) {
        lines.push(`message: ${text}`);
      }
    }
  }
  const finalText = excerptValue(resultText, 4000);
  if (finalText) {
    lines.push('final_output:');
    lines.push(finalText);
  }
  return lines.join('\n').trim();
}

/**
 * Pick the payload to emit as the final JSON envelope.
 *
 * Prefer the last streamed `lastOutput` over the success-shaped streaming
 * fallback (`result`, see container_runner.ts) whenever it carries a real
 * signal: a non-null result, a non-'success' status that can legally carry
 * `result=null` (`error` / `recovered_from_session`, per
 * shared_types.ts::ContainerOutput.status), a non-empty tool trace, or
 * non-zero token usage. Otherwise fall back to `result`. Extracting this as
 * a named export keeps it enrolled in the copy-sync guard
 * (tests/js/copy_sync_guard.test.mjs) so the inline test copy in
 * tests/js/envelope_drop_status.test.mjs cannot drift silently.
 */
export function pickEffectiveOutput(
  lastOutput: ContainerOutput | undefined,
  result: ContainerOutput,
): ContainerOutput {
  const hasToolTrace =
    Array.isArray(lastOutput?.toolTrace) && lastOutput.toolTrace.length > 0;
  const hasTokens =
    ((lastOutput?.input_tokens ?? 0) +
      (lastOutput?.output_tokens ?? 0) +
      (lastOutput?.cache_creation_input_tokens ?? 0) +
      (lastOutput?.cache_read_input_tokens ?? 0)) > 0;
  return lastOutput &&
    (lastOutput.result != null ||
      lastOutput.status === 'error' ||
      lastOutput.status === 'recovered_from_session' ||
      hasToolTrace ||
      hasTokens)
    ? lastOutput
    : result;
}

async function main(): Promise<void> {
  const payloadPath = process.argv[2];
  if (!payloadPath) {
    die('usage: tsx runtime_runner/src/main.ts <payload.json>');
  }
  if (!fs.existsSync(payloadPath)) {
    die(`payload not found: ${payloadPath}`);
  }

  const payload = readJsonFile<KcsiPayload>(payloadPath);
  assertKcsiPayload(payload, payloadPath);
  const scope: RuntimeScope =
    payload.runtime?.session_scope === 'agent' ? 'agent' : 'task';
  const wipeWorkspacePerTask =
    payload.runtime?.wipe_workspace_per_task !== false;
  const workspaceKey = toWorkspaceKey(payload, scope);
  const workspaceRootPath = resolveWorkspaceRootPath(workspaceKey);
  // Inside the container the bind mount is hostPath=workspaceRootPath
  // mapped to containerPath=CONTAINER_WORKSPACE_ROOT (=/workspace/task);
  // see container_runner.ts::buildVolumeMounts. So the host directory that
  // is observed by the container as ``/workspace/task`` IS workspaceRootPath
  // itself, NOT workspaceRootPath/workspace/task (that inner subtree is the
  // *active* working dir CONTAINER_ACTIVE_WORKSPACE_DIR=/workspace/task/workspace,
  // which is not what the agent-runner targets when writing the barrier
  // sentinel — see runtime_runner/agent-runner/src/index.ts where
  // ``workspaceDir: CONTAINER_WORKSPACE_ROOT`` is passed into runPhase1Reflection).
  // The earlier ``path.join(workspaceRootPath, 'workspace', 'task')`` value
  // pointed at a directory that does not even exist on disk, so the
  // BarrierWatcher polled an empty path and never observed the sentinel —
  // confirmed empirically on commit 9e1fa515 with phase1 polyglot smoke
  // (see project_phase1_barrier_diagnosis.md).
  // Emit on stderr ASAP so the Python host's BarrierWatcher can read it
  // from a tempfile referenced via KCSI_BARRIER_WORKSPACE_FILE without
  // having to replicate `toWorkspaceKey()` (which has drifted between
  // sides before — see CLAUDE.md's discussion of the workspace-key formula).
  const workspaceTaskDir = workspaceRootPath;
  process.stderr.write(`WORKSPACE_PATH=${workspaceTaskDir}\n`);
  const barrierWorkspaceFile = process.env.KCSI_BARRIER_WORKSPACE_FILE;
  if (barrierWorkspaceFile) {
    try {
      fs.mkdirSync(path.dirname(barrierWorkspaceFile), { recursive: true });
      fs.writeFileSync(barrierWorkspaceFile, workspaceTaskDir, 'utf-8');
    } catch (err) {
      process.stderr.write(
        `WARN: failed to write KCSI_BARRIER_WORKSPACE_FILE=${barrierWorkspaceFile}: ${
          err instanceof Error ? err.message : String(err)
        }\n`,
      );
    }
  }

  seedWorkspace(payload, workspaceKey, wipeWorkspacePerTask);

  let sessionId: string | undefined;
  if (scope === 'agent') {
    sessionId = loadSessionForAgent(payload.agent_id);
  }

  // Knowledge-related mounts (/app/memory-db, /app/memory, /app/memory-snapshot)
  // are emitted by runContainerAgent() from input.memoryMcp. We previously
  // also pushed them here via the (since-removed) additionalMounts config,
  // which produced duplicate -v flags on every docker run. We now only set
  // the host-side env vars that container_runner.ts forwards into the
  // container via explicit -e flags.
  const knowledgeDbPath = payload.knowledge?.db_path || '';
  if (knowledgeDbPath) {
    const dbDir = path.dirname(path.resolve(knowledgeDbPath));
    if (fs.existsSync(dbDir)) {
      const dbFilename = path.basename(path.resolve(knowledgeDbPath));
      process.env.KNOWLEDGE_DB_PATH = `/app/memory-db/${dbFilename}`;

      const runtimeDbPath = payload.runtime_audit?.db_path || '';
      if (runtimeDbPath) {
        const runtimeFilename = path.basename(path.resolve(runtimeDbPath));
        process.env.RUNTIME_DB_PATH = `/app/memory-db/${runtimeFilename}`;
      }

      if (payload.knowledge?.forum_generation !== undefined) {
        process.env.FORUM_GENERATION = String(payload.knowledge.forum_generation);
      }
      if (payload.knowledge?.experiment_name) {
        process.env.EXPERIMENT_NAME = payload.knowledge.experiment_name;
      }
    }
  }

  process.env.KCSI_DISABLE_AGENT_TEAMS = '1';

  const workspaceRuntime: RegisteredWorkspace = {
    name: `kcsi-${payload.agent_id}`,
    folder: workspaceKey,
    trigger: '@Kcsi',
    added_at: new Date().toISOString(),
    requiresTrigger: false,
  };

  try {
    let lastOutput: ContainerOutput | undefined;
    let latestSessionId: string | undefined = sessionId;

    const taskMeta = (payload.task?.metadata ?? {}) as Record<string, unknown>;
    const taskSource = String(taskMeta.task_source ?? '').toLowerCase();
    const forumToolTaskSources = new Set([
      'cross_task_forum',
      'per_task_forum',
    ]);
    const needsForumTools = forumToolTaskSources.has(taskSource);
    const memoryMcp = payload.knowledge && (!payload.knowledge.disable_memory_tools || needsForumTools)
      ? {
          dbPath: payload.knowledge.db_path,
          serverDir: payload.knowledge.mcp_server_dir,
          snapshotPath: payload.knowledge.snapshot_path,
          // Threaded through for container_mounts.ts (issue #1009) to
          // bind-mount the runtime-audit DB individually; RUNTIME_DB_PATH
          // (set below) already assumes it, so both must stay in sync.
          runtimeDbPath: payload.runtime_audit?.db_path || undefined,
          taskId: payload.task?.id || '',
          taskSource,
          forumGeneration: coerceOptionalNumber(taskMeta.forum_generation),
          forumRound: coerceOptionalNumber(taskMeta.forum_round),
          forumAgentId: coerceOptionalString(taskMeta.forum_agent_id),
          forumExpectedAgents: coerceOptionalNumber(
            taskMeta.forum_expected_agents,
          ),
          forumTaskIds: Array.isArray(taskMeta.forum_task_ids)
            ? taskMeta.forum_task_ids
                .map((x) => String(x || '').trim())
                .filter(Boolean)
            : [],
          experiment: payload.experiment_name,
        }
      : undefined;

    // ARC MCP config is independent of memory. Before this hoist, the ARC
    // block was read from payload.knowledge, which meant `--no-memory` runs
    // silently shipped zero arc_* tools into the container.
    //
    // `payload.arc_no_mcp` (boolean, set by Python container_host when the
    // --arc-no-mcp flag is on) forces `arcTools.enable=false` regardless of
    // what payload.arc_tools.enable says. The agent-runner side detects
    // `taskSource === 'arc' && !arcTools?.enable` and falls back to native
    // tools (Bash/Read/Write/Edit/Glob/Grep) plus a prediction.json file
    // submission contract.
    const arcNoMcp = Boolean(payload.arc_no_mcp);
    const arcTools = payload.arc_tools
      ? {
          enable: Boolean(payload.arc_tools.enable) && !arcNoMcp,
          mcpServerDir: payload.arc_tools.mcp_server_dir,
          taskSource: String(
            payload.arc_tools.task_source ?? taskMeta.task_source ?? '',
          ),
          taskId: String(payload.arc_tools.task_id || payload.task?.id || ''),
          snapshotPath: payload.arc_tools.snapshot_path || undefined,
        }
      : undefined;

    // Cache-stable forum split. The Python host writes
    // `forum_cacheable_prefix` / `forum_variable_suffix` into task.metadata
    // for forum-phase tasks (per_task_forum / cross_task_forum). The
    // direct forum adapter consumes them to place cache_control only on
    // the prefix; non-forum / non-direct adapters ignore them.
    const rawForumCacheablePrefix = taskMeta.forum_cacheable_prefix;
    const rawForumVariableSuffix = taskMeta.forum_variable_suffix;
    const forumCacheablePrefix =
      typeof rawForumCacheablePrefix === 'string' && rawForumCacheablePrefix.length > 0
        ? rawForumCacheablePrefix
        : undefined;
    const forumVariableSuffix =
      typeof rawForumVariableSuffix === 'string' && rawForumVariableSuffix.length > 0
        ? rawForumVariableSuffix
        : undefined;
    const phase1ReflectionCfg = payload.phase1_reflection?.enabled
      ? {
          enabled: true,
          agentId: payload.agent_id,
          evalResultPollTimeoutMs: payload.phase1_reflection.eval_result_poll_timeout_ms,
        }
      : undefined;

    // Cross-task shared-container feature flag block. The Python host
    // sets ``payload.cross_task_shared_container = {enabled: true, ...}``
    // for forum tasks where the container should run BOTH round 0 and
    // round 1 in the same SDK / Anthropic-Messages-API session, signalling
    // the host between rounds via the BarrierProtocol. The forum adapter
    // ignores the block when the task source is not ``cross_task_forum``.
    const crossTaskSharedContainerCfg = payload.cross_task_shared_container?.enabled
      ? {
          enabled: true,
          agentId: payload.agent_id,
          barrierName: payload.cross_task_shared_container.barrier_name,
          responsePollTimeoutMs:
            payload.cross_task_shared_container.response_poll_timeout_ms,
        }
      : undefined;

    // Polyglot test-feedback retry loop config. Unlike phase1Reflection /
    // crossTaskSharedContainer above, container_host.py already builds this
    // dict with camelCase keys matching `PolyglotTestFeedbackConfig`
    // (shared_types.ts) directly, so no field-by-field remap is needed.
    const polyglotTestFeedbackCfg = payload.polyglot_test_feedback?.enabled
      ? payload.polyglot_test_feedback
      : undefined;

    const result = await runContainerAgent(
      workspaceRuntime,
      {
        prompt: buildPrompt(payload),
        sessionId,
        workspaceKey,
        assistantName: 'KCSI',
        memoryMcp,
        arcTools,
        forumCacheablePrefix,
        forumVariableSuffix,
        phase1Reflection: phase1ReflectionCfg,
        crossTaskSharedContainer: crossTaskSharedContainerCfg,
        polyglotTestFeedback: polyglotTestFeedbackCfg,
      },
      () => {},
      async (streamed) => {
        lastOutput = streamed;
        if (streamed.newSessionId) {
          latestSessionId = streamed.newSessionId;
        }
      },
    );

    if (!latestSessionId && result.newSessionId) {
      latestSessionId = result.newSessionId;
    }
    if (scope === 'agent' && latestSessionId) {
      saveSessionForAgent(payload.agent_id, latestSessionId);
    }

    // Prefer the streamed `lastOutput` when it carries a real signal —
    // not just a non-null `result`, but also:
    //   * `status='error'` / `status='recovered_from_session'`
    //   * a non-empty `toolTrace`
    //   * non-zero token counters
    // Any of those can be masked by the streaming-mode fallback in
    // container_runner.ts, which unconditionally resolves with
    // `{status: 'success', result: null}` when the outputChain completes
    // without an exception, regardless of what the stream last reported.
    // Before this broader guard, a streamed envelope with
    // `status='success'`, `result=null`, but real tool/tokens (for example
    // a tool-heavy attempt that never produced final text) was discarded in
    // favour of the generic fallback and Python's container_host could
    // relabel the attempt as `silent_failure`. That is a real attempt and
    // must survive even if the model never emitted a final answer string.
    const effectiveOutput = pickEffectiveOutput(lastOutput, result);
    const rawTrace = effectiveOutput.toolTrace ?? [];

    // --arc-no-mcp: synthesize arc_set_output_grid / arc_submit_trial trace
    // entries from the prediction.json file the agent wrote, so the existing
    // tool-count and arcSubmitTrialResults aggregation, plus the downstream
    // Python scorer, see the same shape they would for an MCP-tool submission.
    // Must run BEFORE toolCallCounts so the synthetic entries are counted.
    //
    // `payload.arc_no_mcp` alone is a sufficient (registry-derived) gate here
    // (#1026 follow-up to #1020): Python only ever sets it True when
    // `resolve_source(source).supports_mcp_arc` is True (container_host.py's
    // `arc_no_mcp_active = bool(self.arc_no_mcp) and _supports_mcp_arc`), and
    // ARC is currently the only registered source with that flag — so a
    // literal `taskSource === 'arc'` check here was redundant with, not
    // additive to, the registry-backed value already threaded through the
    // payload.
    if (Boolean(payload.arc_no_mcp)) {
      synthesizeArcSubmitTraceFromPrediction(workspaceKey, rawTrace);
    }

    const toolCallCounts: Record<string, number> = {};
    for (const entry of rawTrace) {
      if (entry.type === 'tool_call' && typeof entry.tool_name === 'string') {
        const name = entry.tool_name;
        toolCallCounts[name] = (toolCallCounts[name] || 0) + 1;
      }
    }
    const memoryToolCallCounts: Record<string, number> = {};
    const arcToolCallCounts: Record<string, number> = {};
    const forumToolCallCounts: Record<string, number> = {};
    const arcSubmitTrialResults: Array<Record<string, unknown>> = [];
    let arcLastSubmitResult: Record<string, unknown> | undefined;
    // #1046: the live `arc_load_task` tool accepts an agent-suppliable
    // `max_trials` override that the runtime genuinely enforces, independent
    // of the task-metadata-derived default. Record the *effective* value (as
    // reported back by the tool itself) so the host-side scorer
    // (`ArcSessionEvaluator`) can honor it instead of re-deriving a value
    // that could drift from what was actually enforced.
    let arcEffectiveMaxTrials: number | undefined;
    for (const [name, count] of Object.entries(toolCallCounts)) {
      if (
        name.startsWith('mcp__memory__') ||
        name === 'query' ||
        name === 'forum_read'
      ) {
        memoryToolCallCounts[name] = count;
      }
      if (name.startsWith('arc_')) {
        arcToolCallCounts[name] = count;
      }
      if (
        name.startsWith('mcp__memory__forum_') ||
        name === 'forum_read'
      ) {
        forumToolCallCounts[name] = count;
      }
    }
    for (const entry of rawTrace) {
      if (entry.type !== 'tool_call' || typeof entry.tool_name !== 'string') {
        continue;
      }
      const normalizedName = normalizeToolName(entry.tool_name);
      if (normalizedName === 'arc_load_task') {
        const parsed = parseStructuredToolOutput(entry.tool_output);
        const maxTrials = parsed ? coerceOptionalNumber(parsed.max_trials) : undefined;
        if (maxTrials !== undefined) {
          // Last load wins, mirroring arcLastSubmitResult below — a session
          // reload replaces the prior effective budget.
          arcEffectiveMaxTrials = maxTrials;
        }
        continue;
      }
      if (normalizedName !== 'arc_submit_trial') {
        continue;
      }
      const parsed = parseStructuredToolOutput(entry.tool_output);
      if (!parsed) {
        continue;
      }
      arcSubmitTrialResults.push(parsed);
      arcLastSubmitResult = parsed;
    }

    const nativeSessionMemory = collectNativeSessionMemory(workspaceKey);
    const normalizedRuntimeTranscript =
      nativeSessionMemory.trim() ||
      buildRuntimeTranscript(rawTrace, effectiveOutput.result ?? effectiveOutput.error ?? '');
    const workspaceArtifacts = captureWorkspaceArtifacts(workspaceKey, taskSource, taskMeta);
    const meta: Record<string, unknown> = {
      generation: payload.generation,
      agent_id: payload.agent_id,
      task_id: payload.task?.id || '',
      status: effectiveOutput.status,
      session_scope: scope,
      input_tokens: effectiveOutput.input_tokens ?? 0,
      output_tokens: effectiveOutput.output_tokens ?? 0,
      cache_creation_input_tokens: effectiveOutput.cache_creation_input_tokens ?? 0,
      cache_read_input_tokens: effectiveOutput.cache_read_input_tokens ?? 0,
      tokens_source: effectiveOutput.tokens_source
        ?? ((effectiveOutput.input_tokens ?? 0) === 0
          && (effectiveOutput.output_tokens ?? 0) === 0
          && (effectiveOutput.cache_creation_input_tokens ?? 0) === 0
          && (effectiveOutput.cache_read_input_tokens ?? 0) === 0
          ? 'unavailable'
          : 'result_event'),
      workspace_key: workspaceKey,
      session_id: latestSessionId || '',
      active_task_dir: CONTAINER_ACTIVE_WORKSPACE_DIR,
      knowledge_db_path: payload.knowledge?.db_path || '',
      container_image: process.env.KCSI_CONTAINER_IMAGE || process.env.CONTAINER_IMAGE || '',
      official_container_image: payload.runtime?.official_container_image || '',
      runner_image: payload.runtime?.runner_image || '',
      repo_container_path: payload.runtime?.repo_container_path || '',
      official_repo_container_path: payload.runtime?.official_repo_container_path || '',
      runner_root: payload.runtime?.runner_root || '',
      model_requested: process.env.MODEL || '',
      raw_native_session_memory: nativeSessionMemory,
      native_session_memory: normalizedRuntimeTranscript,
      tool_call_counts: toolCallCounts,
      memory_tool_call_counts: memoryToolCallCounts,
      arc_tool_call_counts: arcToolCallCounts,
      forum_tool_call_counts: forumToolCallCounts,
      arc_submit_trial_results: arcSubmitTrialResults,
      arc_last_submit_result: arcLastSubmitResult,
      // #1046: the effective max_trials the live session actually enforced
      // (from the last `arc_load_task` call's own reported value), so
      // `ArcSessionEvaluator.evaluate` can score against it instead of
      // re-deriving a possibly-stale value from task metadata.
      arc_effective_max_trials: arcEffectiveMaxTrials,
      // Present only when status='recovered_from_session' — explains the
      // session-log recovery path so downstream consumers can distinguish
      // a real success from a reconstructed one.
      recovery_note: effectiveOutput.recovery_note,
      // Phase-1 self-reflection text + diagnostic, threaded through for
      // ``src/kcsi/orchestrator/engine.py`` to write into
      // ``attempt.content.reflection``. Both fields are optional —
      // ``phase1_reflection`` is absent when the feature flag is off OR
      // the host barrier didn't respond OR the SDK follow-up turn
      // silent-exited.
      phase1_reflection: effectiveOutput.phase1_reflection,
      phase1_reflection_meta: effectiveOutput.phase1_reflection_meta,
      // Surface the reflection-turn token usage so the host-side engine can
      // record a dedicated `phase1_reflection` row in `token_phases`.
      // Without this propagation the reflection turn's tokens vanish from
      // cost reports.
      phase1_reflection_token_usage: effectiveOutput.phase1_reflection_token_usage,
      // Polyglot test-feedback retry-loop diagnostics + token usage. Only
      // present when the feature flag was on (task_source='polyglot' and
      // triesRemaining > 1); Task 7 (host-side) reads
      // `runtime_meta.polyglot_test_feedback_meta` to bookkeep rounds_used
      // and `polyglot_test_feedback_token_usage` for a dedicated
      // `token_phases` row, mirroring the phase1_reflection pattern above.
      polyglot_test_feedback_meta: effectiveOutput.polyglot_test_feedback_meta,
      polyglot_test_feedback_token_usage: effectiveOutput.polyglot_test_feedback_token_usage,
      // Cross-task shared-container per-round outputs. Only present when
      // the feature flag was on. The host's engine.py harvests these into
      // the cross-task forum drain pipeline and bookkeeps round-1 tokens
      // under the ``cross_task_forum_round_1`` phase slug.
      cross_task_round_0_result: effectiveOutput.cross_task_round_0_result,
      cross_task_round_1_result: effectiveOutput.cross_task_round_1_result,
      cross_task_shared_container_meta: effectiveOutput.cross_task_shared_container_meta,
      // Propagate the envelope's diagnostic `error` text so the host-side
      // SilentAgentRuntimeError can surface it instead of falling back to
      // the generic "agent-runner returned status=error" literal. Without
      // this, the runner's actual failure reason (e.g. "Scheduled task
      // ended before a complete tool loop (pending tool call(s) never
      // returned: 1)") is silently dropped at the host boundary. See #525.
      error: effectiveOutput.error,
    };
    if (workspaceArtifacts.diff) {
      meta.workspace_diff = workspaceArtifacts.diff;
    }
    if (workspaceArtifacts.changedFiles.length > 0) {
      meta.workspace_changed_files = workspaceArtifacts.changedFiles;
    }
    if (workspaceArtifacts.captureError) {
      meta.workspace_diff_capture_error = workspaceArtifacts.captureError;
    }
    if (Object.keys(workspaceArtifacts.solutionFiles).length > 0) {
      meta.workspace_solution_files = workspaceArtifacts.solutionFiles;
    }
    if (
      workspaceArtifacts.repoDir &&
      (workspaceArtifacts.diff ||
        workspaceArtifacts.changedFiles.length > 0 ||
        Object.keys(workspaceArtifacts.solutionFiles).length > 0)
    ) {
      meta.host_workspace_repo_dir = workspaceArtifacts.repoDir;
    }

    const output = {
      result: effectiveOutput.result ?? effectiveOutput.error ?? '',
      tool_trace: rawTrace,
      meta,
    };
    process.stdout.write(JSON.stringify(output) + '\n');
  } finally {
    if (scope === 'task' && wipeWorkspacePerTask) {
      cleanupWorkspace(workspaceKey);
    }
  }
}

const entryPath = process.argv[1] ? path.resolve(process.argv[1]) : '';
if (import.meta.url === `file://${entryPath}`) {
  main().catch((err) => {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
}
