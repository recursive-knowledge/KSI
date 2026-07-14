import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';

import {
  Agent,
  MCPServerStdio,
  MaxTurnsExceededError,
  applyDiff,
  extractAllTextOutput,
  run,
  setDefaultOpenAIKey,
  tool,
} from '@openai/agents';

import { extractStructuredForumText } from './extract.js';
import { isOpenAIForumPhase, selectOpenAINativeTools } from './openai_tool_selection.js';
import { usageFromResult } from './openai_usage.js';
import { runOpenAIPolyglotTestFeedback } from './polyglot_test_feedback_openai.js';
import { buildSystemPromptAppend } from './prompt-utils.js';
import { MARKER_INVALID_PROMPT, MARKER_USAGE_POLICY } from './retryable_markers.js';
import { ContainerInput, ContainerOutput } from './shared_types.js';
import { buildOpenAIMemoryMcpEnv } from './memory_mcp_env.js';

export { buildOpenAIMemoryMcpEnv } from './memory_mcp_env.js';

export type OpenAIContainerInput = ContainerInput;

const CONTAINER_WORKSPACE_ROOT = '/workspace/task';

function log(message: string): void {
  console.error(`[agent-runner/openai] ${message}`);
}

function safeStringify(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (value == null) {
    return '';
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function sanitizeShellEnv(
  sdkEnv: Record<string, string | undefined>,
): Record<string, string> {
  const secretNames = new Set([
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'CLAUDE_CODE_OAUTH_REFRESH_TOKEN',
    'CLAUDE_CODE_OAUTH_SCOPES',
    'HF_TOKEN',
    'HUGGING_FACE_HUB_TOKEN',
  ]);
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(sdkEnv)) {
    if (!value || secretNames.has(key)) {
      continue;
    }
    out[key] = value;
  }
  return out;
}

function resolveWorkspacePath(filePath: string, cwd: string): string {
  const resolved = path.resolve(cwd, filePath);
  const rel = path.relative(cwd, resolved);
  if (rel.startsWith('..') || path.isAbsolute(rel)) {
    throw new Error(`Path escapes workspace: ${filePath}`);
  }
  return resolved;
}

// ---------------------------------------------------------------------------
// Flag-gated OpenAI scaffold-parity native filesystem tools (issue #634).
//
// Default OFF. When `OPENAI_PARITY_TOOLS` is truthy these function tools are
// added to the OpenAI agent loop so its tool surface more closely mirrors the
// Claude path's native Read/Write/Edit/Glob/Grep (the Claude side ships these
// 8 native tools via the claude_code preset; the OpenAI side historically had
// only shell + apply_patch). Every path is confined to the workspace root via
// `resolveWorkspacePath` — exactly the same sandbox the existing
// shell/apply_patch confinement uses — so enabling parity tools cannot widen
// filesystem reach beyond the existing shell tool.
//
// Confinement is LEXICAL (path.resolve/path.relative), not realpath-based, so a
// symlink FILE that lives inside the workspace but points outside it is
// followed on read/write — identical to what `shell` (cat/tee) already allows,
// hence no wider reach; just don't mistake this for symlink-safe isolation.
// ---------------------------------------------------------------------------

const PARITY_READ_MAX_BYTES = 256 * 1024;
const PARITY_GLOB_MAX_RESULTS = 1000;
const PARITY_GREP_MAX_MATCHES = 1000;
const PARITY_WALK_MAX_ENTRIES = 50000;
// Per-file ceiling for grep: files larger than this are skipped rather than
// read whole into memory (OOM guard). 8 MiB comfortably covers source files.
const PARITY_GREP_MAX_FILE_BYTES = 8 * 1024 * 1024;
// Wall-clock budget for a single grep call. A model-supplied regex runs
// in-process (no subprocess to SIGKILL like `shell`), so cap aggregate scan
// time to bound damage from a pathological/backtracking pattern across many
// lines/files. Note: this bounds the loop between regex.test calls; it cannot
// preempt a single catastrophic test (JS regex is not interruptible without a
// worker), which is acceptable given the container + turn-budget envelope and
// that `shell` already exposes equivalent CPU to the agent.
const PARITY_GREP_TIME_BUDGET_MS = 5000;
// Directory names never descended into during glob/grep walks. Mirrors the
// spirit of the Claude tools' default ignores and keeps bounded walks from
// exploding on dependency trees / VCS metadata inside a task workspace.
const PARITY_WALK_SKIP_DIRS = new Set([
  '.git',
  'node_modules',
  '.venv',
  'venv',
  '__pycache__',
  '.mypy_cache',
  '.pytest_cache',
  '.tox',
  'dist',
  'build',
  '.next',
  'target',
]);

export function isParityToolsEnabled(
  sdkEnv: Record<string, string | undefined>,
): boolean {
  const raw = String(sdkEnv.OPENAI_PARITY_TOOLS ?? '').trim().toLowerCase();
  if (!raw) return false;
  return !['0', 'false', 'no', 'off'].includes(raw);
}

/**
 * Translate a simple glob pattern (supporting `**`, `*`, and `?`) into a
 * RegExp anchored to the full relative path. Kept dependency-free so the
 * parity tools add no new npm imports to the agent-runner.
 */
export function globToRegExp(pattern: string): RegExp {
  let re = '';
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i];
    if (ch === '*') {
      if (pattern[i + 1] === '*') {
        // `**` matches across path separators.
        re += '.*';
        i++;
        // consume an immediately-following slash so `**/x` matches `x` too.
        if (pattern[i + 1] === '/') {
          re += '(?:/)?';
          i++;
        }
      } else {
        // single `*` does not cross a path separator.
        re += '[^/]*';
      }
    } else if (ch === '?') {
      re += '[^/]';
    } else if ('.+^${}()|[]\\'.includes(ch)) {
      re += `\\${ch}`;
    } else {
      re += ch;
    }
  }
  return new RegExp(`^${re}$`);
}

/**
 * Bounded recursive walk of the workspace, yielding workspace-relative file
 * paths. Caps total entries inspected and skips heavy/VCS dirs so a single
 * tool call can never wedge the run. Symlinks are not followed.
 */
function walkWorkspace(root: string): string[] {
  const out: string[] = [];
  const stack: string[] = [root];
  let inspected = 0;
  while (stack.length > 0) {
    const dir = stack.pop() as string;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (inspected >= PARITY_WALK_MAX_ENTRIES) {
        return out;
      }
      inspected++;
      const abs = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) {
        continue;
      }
      if (entry.isDirectory()) {
        if (PARITY_WALK_SKIP_DIRS.has(entry.name)) {
          continue;
        }
        stack.push(abs);
      } else if (entry.isFile()) {
        out.push(path.relative(root, abs));
      }
    }
  }
  return out;
}

interface ParityFsTools {
  readFile: (input: { path: string; max_bytes?: number }) => Promise<string>;
  writeFile: (input: { path: string; content: string }) => Promise<string>;
  editFile: (input: {
    path: string;
    old_string: string;
    new_string: string;
    replace_all?: boolean;
  }) => Promise<string>;
  glob: (input: { pattern: string }) => Promise<string>;
  grep: (input: {
    pattern: string;
    path?: string;
    max_matches?: number;
    ignore_case?: boolean;
  }) => Promise<string>;
}

/**
 * Build the parity filesystem tool implementations. Pure functions over the
 * sandboxed `cwd`; `resolveWorkspacePath` rejects any path that escapes the
 * workspace root (`..`, absolute paths). Each returns a JSON string so the
 * tool-call trace stays inspectable by downstream analysis.
 */
export function createParityFsTools(cwd: string): ParityFsTools {
  return {
    readFile: async ({ path: filePath, max_bytes }) => {
      let resolved: string;
      try {
        resolved = resolveWorkspacePath(String(filePath ?? ''), cwd);
      } catch (err) {
        return safeStringify({ status: 'failed', output: err instanceof Error ? err.message : String(err) });
      }
      const cap = Math.max(
        1,
        Math.min(Number(max_bytes || PARITY_READ_MAX_BYTES), PARITY_READ_MAX_BYTES),
      );
      let buf: Buffer;
      try {
        // Bounded read: pull at most cap+1 bytes (the +1 only flags truncation)
        // instead of slurping the whole file into memory, so a multi-GB
        // workspace file cannot OOM the runner.
        const fd = fs.openSync(resolved, 'r');
        try {
          const readBuf = Buffer.allocUnsafe(cap + 1);
          const n = fs.readSync(fd, readBuf, 0, cap + 1, 0);
          buf = readBuf.subarray(0, n);
        } finally {
          fs.closeSync(fd);
        }
      } catch (err) {
        return safeStringify({ status: 'failed', output: err instanceof Error ? err.message : String(err) });
      }
      const truncated = buf.length > cap;
      const content = buf.subarray(0, cap).toString('utf-8');
      return safeStringify({ status: 'ok', path: filePath, truncated, content });
    },
    writeFile: async ({ path: filePath, content }) => {
      let resolved: string;
      try {
        resolved = resolveWorkspacePath(String(filePath ?? ''), cwd);
      } catch (err) {
        return safeStringify({ status: 'failed', output: err instanceof Error ? err.message : String(err) });
      }
      try {
        fs.mkdirSync(path.dirname(resolved), { recursive: true });
        fs.writeFileSync(resolved, String(content ?? ''), 'utf-8');
      } catch (err) {
        return safeStringify({ status: 'failed', output: err instanceof Error ? err.message : String(err) });
      }
      return safeStringify({ status: 'ok', path: filePath, bytes: Buffer.byteLength(String(content ?? '')) });
    },
    editFile: async ({ path: filePath, old_string, new_string, replace_all }) => {
      let resolved: string;
      let current: string;
      try {
        resolved = resolveWorkspacePath(String(filePath ?? ''), cwd);
        current = fs.readFileSync(resolved, 'utf-8');
      } catch (err) {
        return safeStringify({ status: 'failed', output: err instanceof Error ? err.message : String(err) });
      }
      const oldStr = String(old_string ?? '');
      const newStr = String(new_string ?? '');
      if (oldStr === '') {
        return safeStringify({ status: 'failed', output: 'old_string must be non-empty' });
      }
      const occurrences = current.split(oldStr).length - 1;
      if (occurrences === 0) {
        return safeStringify({ status: 'failed', output: 'old_string not found in file' });
      }
      if (occurrences > 1 && !replace_all) {
        return safeStringify({
          status: 'failed',
          output: `old_string matched ${occurrences} times; pass replace_all=true or provide a unique snippet`,
        });
      }
      // Splice by index rather than String.prototype.replace: `.replace(str, str)`
      // still interprets `$&`/`$$`/`` $` ``/`$'`/`$n` in the REPLACEMENT string as
      // special patterns, which would corrupt any new_string containing `$`
      // (shell `$$`, regex backrefs, JS template `$`). split/join (replace_all)
      // is already literal; mirror that for the single-match path.
      const next = replace_all
        ? current.split(oldStr).join(newStr)
        : (() => {
            const idx = current.indexOf(oldStr);
            return current.slice(0, idx) + newStr + current.slice(idx + oldStr.length);
          })();
      fs.writeFileSync(resolved, next, 'utf-8');
      return safeStringify({ status: 'ok', path: filePath, replacements: replace_all ? occurrences : 1 });
    },
    glob: async ({ pattern }) => {
      const pat = String(pattern ?? '').trim();
      if (!pat) {
        return safeStringify({ status: 'failed', output: 'pattern is required' });
      }
      const regex = globToRegExp(pat);
      const matches: string[] = [];
      for (const rel of walkWorkspace(cwd)) {
        // Match against both the full relative path and the basename so a
        // bare `*.py` pattern (no directory) still matches nested files.
        if (regex.test(rel) || regex.test(path.basename(rel))) {
          matches.push(rel);
          if (matches.length >= PARITY_GLOB_MAX_RESULTS) break;
        }
      }
      matches.sort();
      return safeStringify({ status: 'ok', pattern: pat, count: matches.length, matches });
    },
    grep: async ({ pattern, path: subPath, max_matches, ignore_case }) => {
      const pat = String(pattern ?? '');
      if (!pat) {
        return safeStringify({ status: 'failed', output: 'pattern is required' });
      }
      let regex: RegExp;
      try {
        regex = new RegExp(pat, ignore_case ? 'i' : '');
      } catch (err) {
        return safeStringify({
          status: 'failed',
          output: `invalid regex: ${err instanceof Error ? err.message : String(err)}`,
        });
      }
      const cap = Math.max(
        1,
        Math.min(Number(max_matches || PARITY_GREP_MAX_MATCHES), PARITY_GREP_MAX_MATCHES),
      );
      // Confine the search root to the requested subpath when given, else the
      // whole workspace. `resolveWorkspacePath` enforces the sandbox; wrap it so
      // an escaping `path` returns a graceful failed-status (like read/write/
      // edit) instead of throwing out of the tool.
      let searchRootAbs: string;
      try {
        searchRootAbs = subPath ? resolveWorkspacePath(String(subPath), cwd) : cwd;
      } catch (err) {
        return safeStringify({ status: 'failed', output: err instanceof Error ? err.message : String(err) });
      }
      let relFiles: string[];
      if (fs.existsSync(searchRootAbs) && fs.statSync(searchRootAbs).isFile()) {
        relFiles = [path.relative(cwd, searchRootAbs)];
      } else {
        relFiles = walkWorkspace(searchRootAbs).map((rel) =>
          path.relative(cwd, path.join(searchRootAbs, rel)),
        );
      }
      const out: Array<{ path: string; line: number; text: string }> = [];
      const deadline = Date.now() + PARITY_GREP_TIME_BUDGET_MS;
      let timedOut = false;
      for (const rel of relFiles) {
        if (out.length >= cap) break;
        if (Date.now() > deadline) {
          timedOut = true;
          break;
        }
        let content: string;
        try {
          const abs = resolveWorkspacePath(rel, cwd);
          // Skip oversized files instead of slurping them into memory (OOM guard).
          if (fs.statSync(abs).size > PARITY_GREP_MAX_FILE_BYTES) continue;
          const buf = fs.readFileSync(abs);
          // Skip obvious binaries (NUL byte in the first 4KB).
          if (buf.subarray(0, 4096).includes(0)) continue;
          content = buf.toString('utf-8');
        } catch {
          continue;
        }
        const lines = content.split('\n');
        for (let i = 0; i < lines.length; i++) {
          if (out.length >= cap) break;
          if (Date.now() > deadline) {
            timedOut = true;
            break;
          }
          if (regex.test(lines[i])) {
            out.push({ path: rel, line: i + 1, text: lines[i].slice(0, 2000) });
          }
        }
        if (timedOut) break;
      }
      return safeStringify({ status: 'ok', pattern: pat, count: out.length, matches: out, ...(timedOut ? { timed_out: true } : {}) });
    },
  };
}

/**
 * Concise agentic-coding system prompt used when `OPENAI_PARITY_TOOLS` is on.
 * Written to give the OpenAI agent guidance in the *spirit* of the claude_code
 * preset (explore-before-edit, run tests, minimal surgical diffs) WITHOUT
 * copying Anthropic's preset text. Parity here is approximate: the prompt
 * texts differ by design and OpenAI has no equivalent system-prompt preset.
 */
export function buildParityAgenticInstructions(): string {
  return [
    'You are an autonomous software-engineering agent working inside a sealed',
    'task workspace. You have native tools to read, search, write, and edit',
    'files plus a shell. Work methodically:',
    '',
    '- Start by reading TASK.md (or the task description) and the files it names.',
    '- Explore before editing: use glob/grep to locate the relevant code and read',
    '  enough surrounding context to understand it before changing anything.',
    '- Make the smallest correct change. Prefer surgical edits (edit_file with a',
    '  unique old_string) over rewriting whole files.',
    '- After editing, verify: run the project\'s tests or the command the task',
    '  specifies, and read the output. Fix regressions you introduce.',
    '- Use the shell for running commands, builds, and tests; use read_file/',
    '  write_file/edit_file for file content so changes are explicit and traceable.',
    '- All paths are confined to the workspace root; do not attempt to escape it.',
    '- Do not stop until the task\'s required output/format is produced. When the',
    '  task asks for a specific output format, follow it exactly.',
    '- Be concise in prose; let tool calls do the work.',
  ].join('\n');
}

function createShell(
  cwd: string,
  sdkEnv: Record<string, string | undefined>,
): {
  run: (action: {
    commands: string[];
    maxOutputLength?: number;
    timeoutMs?: number;
  }) => Promise<{
    output: Array<{
      stdout: string;
      stderr: string;
      outcome: { type: 'exit'; exitCode: number };
    }>;
  }>;
} {
  return {
    run: async (action) => {
      const commands = Array.isArray(action.commands) ? action.commands : [];
      const command = commands.join('\n');
      const timeoutMs = Math.max(1, Number(action.timeoutMs || 120000));
      const maxOutputLength = Math.max(
        1024,
        Number(action.maxOutputLength || 50000),
      );

      return await new Promise((resolve) => {
        // Use `-c` not `-lc`. The login-shell flag re-runs /etc/profile,
        // which on Debian/Ubuntu base images RESETS PATH to
        // /usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games — stripping
        // language toolchains that SWE-bench Pro images install at non-default
        // locations (Go at /usr/local/go/bin, Rust at /root/.cargo/bin, etc.).
        // Hit on every Go-based teleport / vuls / navidrome instance: the
        // agent's `bash run_script.sh` couldn't find `go`, run_script.sh
        // produced no test output, and the grader recorded
        // NO_TESTS_FOUND_OR_PARSING_ERROR. Non-login shells inherit the
        // sanitized env we explicitly pass, which already carries the base
        // image's PATH from the docker container env.
        const proc = spawn('/bin/sh', ['-c', command], {
          cwd,
          env: sanitizeShellEnv(sdkEnv),
          stdio: ['ignore', 'pipe', 'pipe'],
        });
        let stdout = '';
        let stderr = '';
        const killTimer = setTimeout(() => {
          proc.kill('SIGKILL');
        }, timeoutMs);
        proc.stdout.on('data', (chunk) => {
          stdout += chunk.toString();
          if (stdout.length > maxOutputLength) {
            stdout = stdout.slice(0, maxOutputLength);
          }
        });
        proc.stderr.on('data', (chunk) => {
          stderr += chunk.toString();
          if (stderr.length > maxOutputLength) {
            stderr = stderr.slice(0, maxOutputLength);
          }
        });
        proc.on('close', (code) => {
          clearTimeout(killTimer);
          resolve({
            output: [
              {
                stdout,
                stderr,
                outcome: { type: 'exit', exitCode: code ?? 0 },
              },
            ],
          });
        });
        proc.on('error', (err) => {
          clearTimeout(killTimer);
          resolve({
            output: [
              {
                stdout,
                stderr: err.message,
                outcome: { type: 'exit', exitCode: 1 },
              },
            ],
          });
        });
      });
    },
  };
}

function createEditor(cwd: string): {
  createFile: (operation: { path: string; diff: string }) => Promise<{ status: 'completed' | 'failed'; output?: string }>;
  updateFile: (operation: { path: string; diff: string }) => Promise<{ status: 'completed' | 'failed'; output?: string }>;
  deleteFile: (operation: { path: string }) => Promise<{ status: 'completed' | 'failed'; output?: string }>;
} {
  return {
    createFile: async (operation) => {
      try {
        const filePath = resolveWorkspacePath(operation.path, cwd);
        const content = applyDiff('', toV4ACreate(operation.diff), 'create');
        fs.mkdirSync(path.dirname(filePath), { recursive: true });
        fs.writeFileSync(filePath, content, 'utf-8');
        return { status: 'completed' };
      } catch (err) {
        return {
          status: 'failed',
          output: err instanceof Error ? err.message : String(err),
        };
      }
    },
    updateFile: async (operation) => {
      try {
        const filePath = resolveWorkspacePath(operation.path, cwd);
        const current = fs.readFileSync(filePath, 'utf-8');
        const next = applyDiff(current, operation.diff, 'default');
        fs.writeFileSync(filePath, next, 'utf-8');
        return { status: 'completed' };
      } catch (err) {
        return {
          status: 'failed',
          output: err instanceof Error ? err.message : String(err),
        };
      }
    },
    deleteFile: async (operation) => {
      try {
        const filePath = resolveWorkspacePath(operation.path, cwd);
        fs.rmSync(filePath, { force: true });
        return { status: 'completed' };
      } catch (err) {
        return {
          status: 'failed',
          output: err instanceof Error ? err.message : String(err),
        };
      }
    },
  };
}

function prefixRawCreateContent(text: string): string {
  if (!text) return '';
  const hasTrailingNewline = text.endsWith('\n');
  const body = hasTrailingNewline ? text.slice(0, -1) : text;
  const lines = body ? body.split('\n') : [''];
  const prefixed = lines.map((line) => `+${line}`).join('\n');
  return hasTrailingNewline ? `${prefixed}\n` : prefixed;
}

function isV4ACreateBody(text: string): boolean {
  if (!text || text.includes('*** Begin Patch')) return false;
  const body = text.endsWith('\n') ? text.slice(0, -1) : text;
  if (!body) return false;
  const lines = body.split('\n');
  // A single raw line such as "+foo" is indistinguishable from a one-line
  // V4A create body. Prefer preserving literal raw content in that ambiguous
  // case; callers that need a literal leading plus in V4A can pass "++foo".
  if (lines.length === 1 && lines[0].startsWith('+') && !lines[0].startsWith('++')) {
    return false;
  }
  return lines.every((line) => line.startsWith('+'));
}

function extractUnifiedCreateBody(lines: string[]): string | null {
  const oldPathIndex = lines.findIndex((line) => line === '--- /dev/null');
  if (oldPathIndex < 0 || !lines[oldPathIndex + 1]?.startsWith('+++ ')) {
    return null;
  }
  const body: string[] = [];
  let inHunk = false;
  for (const line of lines.slice(oldPathIndex + 2)) {
    if (line.startsWith('diff --git ')) break;
    if (line.startsWith('@@ ')) {
      inHunk = true;
      continue;
    }
    if (!inHunk || line === '\\ No newline at end of file') continue;
    if (line.startsWith('+')) {
      body.push(line);
    }
  }
  return inHunk ? body.join('\n') : null;
}

export function toV4ACreate(diff: string): string {
  const text = String(diff ?? '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  if (!text) return '';
  if (isV4ACreateBody(text)) return text;

  const lines = text.split('\n');
  const addFileIndex = lines.findIndex((line) => line.startsWith('*** Add File:'));
  if (addFileIndex >= 0 && lines.some((line) => line.startsWith('*** Begin Patch'))) {
    const body: string[] = [];
    for (const line of lines.slice(addFileIndex + 1)) {
      if (
        line.startsWith('*** End Patch') ||
        line.startsWith('*** Update File:') ||
        line.startsWith('*** Delete File:')
      ) {
        break;
      }
      body.push(line.startsWith('+') ? line : `+${line}`);
    }
    return body.join('\n');
  }

  const unifiedCreateBody = extractUnifiedCreateBody(lines);
  if (unifiedCreateBody != null) {
    return unifiedCreateBody;
  }

  return prefixRawCreateContent(text);
}

async function createMcpServers(
  containerInput: OpenAIContainerInput,
  sdkEnv: Record<string, string | undefined>,
): Promise<MCPServerStdio[]> {
  const servers: MCPServerStdio[] = [];
  if (!fs.existsSync('/app/memory/mcp_server.py')) {
    return servers;
  }

  // taskSource falls back to arcTools.taskSource when memoryMcp is absent
  // (mirrors the parity guard in index.ts so --no-memory ARC runs still
  // register the arc MCP server).
  const taskSource = (
    containerInput.memoryMcp?.taskSource
    || containerInput.arcTools?.taskSource
    || ''
  ).toLowerCase();

  if (containerInput.memoryMcp) {
    const memoryServer = new MCPServerStdio({
      name: 'ksi-memory',
      fullCommand: 'python3 /app/memory/mcp_server.py',
      cacheToolsList: true,
      env: buildOpenAIMemoryMcpEnv(containerInput, sdkEnv, taskSource),
    } as any);
    await memoryServer.connect();
    servers.push(memoryServer);
  }

  // ARC no longer registers an MCP server: it runs natively for every
  // provider (the agent reads payload.json and writes attempt files, and the
  // host synthesizes the arc_submit_trial trace after exit). The legacy ARC
  // MCP server registration was removed with the direct-ARC adapters.

  return servers;
}

/**
 * Map OpenAI per-response usage into the Claude-aligned bucket shape used by
 * `src/ksi/tokens.py::TokenUsage` (which sums all four buckets in `.total`).
 *
 * OpenAI reports `inputTokens` as the TOTAL input (including the cached
 * portion) and exposes the cached subset via `inputTokensDetails[].cachedTokens`.
 * Claude, by contrast, reports `input_tokens` as *fresh* input only and
 * separates the cached portion into `cache_read_input_tokens`. Naively copying
 * OpenAI's fields into Claude's slots double-counted cached tokens (once in
 * input, once in cache_read) and previously mis-bucketed reasoning tokens into
 * `cache_creation_input_tokens` — also a double-count, since reasoning is
 * already counted in `outputTokens`.
 *
 * Resulting semantics (mirrors Claude):
 *   input_tokens                  = inputTotal - cachedTokens   (fresh input)
 *   cache_read_input_tokens       = cachedTokens
 *   cache_creation_input_tokens   = 0    (OpenAI has no separate create charge)
 *   output_tokens                 = outputTotal (reasoning stays inside)
 *
 * So `input + output + cache_read + cache_creation == inputTotal + outputTotal`
 * exactly, with cache visibility preserved for cost-analysis downstream.
 *
 * See tests/js/openai_token_shape.test.mjs for the regression pins.
 */
/**
 * Resolve the per-task-source turn budget for the OpenAI agent-runner.
 *
 * Mirrors the Claude path's `KSI_CLAUDE_MAX_MESSAGES` resolution logic
 * in `resolveTurnBudgets` (`runtime_runner/agent-runner/src/query_config.ts`). Defaults:
 *   discussion phases → 60
 *   arc          → 150
 *   everything else → 150
 *
 * `KSI_OPENAI_MAX_TURNS` overrides the default when set to a positive
 * integer; empty / non-numeric / non-positive values fall back to the
 * per-source default. Previously this was a hardcoded 25 for every task
 * source, which aborted ARC runs mid-attempt and was ~6x tighter than the
 * Claude path. The ARC-specific default was raised from 80 → 150 in the
 * MaxTurnsExceeded-handling fix (fix/openai-max-turns-handling) — the GPT
 * ARC1 latency tail showed ~3-5% of attempts hitting 80 turns and throwing
 * uncaught MaxTurnsExceededError; raising the cap to 150 matches the
 * Claude-side budget for non-discussion tasks and gives MCP-heavy GPT-4o-mini
 * turns enough runway to actually converge. See
 * tests/js/openai_turn_budget.test.mjs.
 */
export function resolveOpenAIMaxTurns(
  taskSource: string | undefined,
  envOverride: string | undefined,
): number {
  const parsed = Number(envOverride);
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.floor(parsed);
  }
  const src = (taskSource || '').toLowerCase();
  if (src === 'per_task_forum') return 60;
  if (src === 'arc') return 150;
  return 150;
}

/**
 * Build a synthetic `result`-shaped object from a `MaxTurnsExceededError`'s
 * `state` field so we can reuse `usageFromResult` / `extractToolTrace` /
 * `extractBestOutput` on partial runs.
 *
 * The SDK's `RunResult` exposes `newItems`, `rawResponses`, and
 * `lastResponseId` as getters over `state._generatedItems`,
 * `state._modelResponses`, and the last entry of `state._modelResponses`
 * respectively (see
 * node_modules/@openai/agents-core/dist/result.mjs lines 61-78). When a
 * run throws mid-way, the RunResult is never constructed but the state is
 * still populated and attached to the error — we reconstruct the same
 * shape manually so downstream helpers work unchanged.
 *
 * Exported for testability (avoids the need to import the SDK in tests).
 */
export function salvageResultFromState(state: any): {
  newItems: unknown[];
  rawResponses: unknown[];
  lastResponseId: string | undefined;
  finalOutput: unknown;
} {
  const generatedItems = Array.isArray(state?._generatedItems)
    ? state._generatedItems
    : [];
  const modelResponses = Array.isArray(state?._modelResponses)
    ? state._modelResponses
    : [];
  const lastResponse: any =
    modelResponses.length > 0 ? modelResponses[modelResponses.length - 1] : undefined;
  const lastResponseId =
    (lastResponse && (lastResponse.responseId || lastResponse.response_id)) ||
    state?._previousResponseId ||
    undefined;
  return {
    newItems: generatedItems,
    rawResponses: modelResponses,
    lastResponseId,
    finalOutput: undefined,
  };
}

function extractToolTrace(result: any): Array<Record<string, unknown>> {
  const outputsByCallId = new Map<string, unknown>();
  for (const item of result?.newItems || []) {
    if (item?.type !== 'tool_call_output_item') continue;
    const rawItem = item.rawItem || item.raw_item || {};
    const callId = rawItem.callId || rawItem.call_id || rawItem.id;
    if (callId) {
      outputsByCallId.set(String(callId), item.output);
    }
  }

  const trace: Array<Record<string, unknown>> = [];
  for (const item of result?.newItems || []) {
    if (
      item?.type !== 'tool_call_item' &&
      item?.type !== 'tool_search_call_item' &&
      item?.type !== 'tool_approval_item'
    ) {
      continue;
    }
    const rawItem = item.rawItem || item.raw_item || {};
    const callId = rawItem.callId || rawItem.call_id || rawItem.id;
    const argumentsValue =
      rawItem.arguments ||
      rawItem.input ||
      rawItem.action ||
      rawItem.actions ||
      item.arguments ||
      item.input ||
      {};
    let parsedArguments: unknown = argumentsValue;
    if (typeof parsedArguments === 'string') {
      try {
        parsedArguments = JSON.parse(parsedArguments);
      } catch {
        parsedArguments = { raw_arguments: parsedArguments };
      }
    }
    const name =
      rawItem.name ||
      rawItem.toolName ||
      item.toolName ||
      rawItem.type ||
      item.name ||
      'unknown';
    trace.push({
      type: 'tool_call',
      tool_name: String(name),
      tool_input: parsedArguments,
      tool_output: callId ? outputsByCallId.get(String(callId)) : undefined,
      name: String(name),
      input: parsedArguments,
      output: callId ? outputsByCallId.get(String(callId)) : undefined,
    });
  }
  return trace;
}

function extractBestOutput(result: any, isForumTask: boolean): string {
  const finalOutput = result?.finalOutput;
  let fallback =
    typeof finalOutput === 'string'
      ? finalOutput
      : safeStringify(finalOutput);
  const extractedText = extractAllTextOutput(result?.newItems || []);
  if (extractedText.trim()) {
    fallback = extractedText;
  }
  for (const item of result?.newItems || []) {
    if (item?.type !== 'message_output_item') continue;
    const raw = item.rawItem || item.raw_item || item;
    const structured = isForumTask ? extractStructuredForumText(raw) : null;
    if (structured) {
      return structured;
    }
    const text =
      typeof item.content === 'string'
        ? item.content
        : safeStringify(
            item.outputText ||
              item.output_text ||
              raw?.content ||
              raw?.text ||
              '',
          );
    if (text.trim()) {
      fallback = text;
    }
  }
  return fallback;
}

function buildModelSettings(
  model: string,
  sdkEnv: Record<string, string | undefined>,
): Record<string, unknown> | undefined {
  const effort = String(sdkEnv.REASONING_EFFORT || '').trim();
  if (!effort || !model.toLowerCase().includes('gpt-5')) {
    return undefined;
  }
  return { reasoning: { effort } };
}

function isInvalidPromptError(err: unknown): boolean {
  const message = safeStringify(err instanceof Error ? err.message : err);
  const lower = message.toLowerCase();
  // The two exact phrases shared with the orchestrator's non_retryable category
  // are sourced from the single-source JSON (retryable_markers.json) so a reword
  // can't silently desync this OpenAI prompt-rejection gate from engine.py
  // (#648). "violating"/"flagged" stay literal on purpose: they are deliberately
  // broader substrings of the canonical "flagged as potentially violating"
  // marker, used to catch the policy-violation family in OpenAI's response text.
  return (
    lower.includes(MARKER_INVALID_PROMPT) ||
    (lower.includes('prompt') &&
      (lower.includes(MARKER_USAGE_POLICY) ||
        lower.includes('violating') ||
        lower.includes('flagged')))
  );
}

function buildCompactRetryPrompt(containerInput: OpenAIContainerInput): string {
  const taskSource = (
    containerInput.memoryMcp?.taskSource || containerInput.arcTools?.taskSource || ''
  ).toLowerCase();
  if (taskSource === 'arc') {
    return [
      'Use the active workspace at /workspace/task/workspace.',
      'Read TASK.md.',
      'Use the ARC tools to load the task, inspect state, set an output grid, and submit.',
      'You have max_trials=2 per test input; always submit two trials per test before advancing. If you have one credible answer, submit it twice; if you can construct a meaningfully different second hypothesis, submit that on the second trial instead.',
      'Return only the JSON grid requested by TASK.md.',
    ].join('\n');
  }
  if (isOpenAIForumPhase(taskSource)) {
    return [
      'Use the active workspace at /workspace/task/workspace.',
      'Read TASK.md.',
      'Contribute to the forum with the available forum tools.',
      'Call forum_signal_done when finished.',
    ].join('\n');
  }
  return [
    'Use the active workspace at /workspace/task/workspace.',
    'Read TASK.md and follow its required output format.',
  ].join('\n');
}

// NOTE: the hosted `shellTool` / `applyPatchTool` helpers from @openai/agents
// are only accepted by the Responses API for a narrow set of reasoning
// models (o1/o3/gpt-5 family) and fail with 400 "Tool 'shell' is not
// supported" on gpt-4o-mini. `runOpenAIQuery` below builds the agent inline
// using function-tool variants (shellFnTool / applyPatchFnTool) that work
// across every OpenAI model.

export async function runOpenAIQuery(
  prompt: string,
  previousResponseId: string | undefined,
  containerInput: OpenAIContainerInput,
  sdkEnv: Record<string, string | undefined>,
): Promise<{
  newSessionId?: string;
  resultText: string;
  toolTrace: Array<Record<string, unknown>>;
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
  /**
   * Envelope status. Present when a known-recoverable error path triggered
   * (currently only MaxTurnsExceededError). Omitted on success so the
   * caller can treat absence as success.
   */
  status?: 'error';
  /**
   * Descriptive error message when `status === 'error'`. Meant for the
   * Python host's `error` field on the ContainerOutput envelope.
   */
  error?: string;
  /**
   * Polyglot test-feedback retry-loop diagnostics + dedicated token usage
   * (Aider protocol; same shape/semantics as the Claude path's fields in
   * query_runner.ts). Present iff `containerInput.polyglotTestFeedback` was
   * enabled with triesRemaining > 1. When a retry round ran, `resultText`
   * above already carries the LAST round's assistant text — the graded
   * output. index.ts splices both fields into the OpenAI envelope.
   */
  polyglot_test_feedback_meta?: ContainerOutput['polyglot_test_feedback_meta'];
  polyglot_test_feedback_token_usage?: ContainerOutput['polyglot_test_feedback_token_usage'];
}> {
  const cwd = CONTAINER_WORKSPACE_ROOT;
  const seedContextPath = `${CONTAINER_WORKSPACE_ROOT}/.seed_context`;
  let seedContext: string | undefined;
  if (fs.existsSync(seedContextPath)) {
    seedContext = fs.readFileSync(seedContextPath, 'utf-8');
  }
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  let globalClaudeMd: string | undefined;
  if (fs.existsSync(globalClaudeMdPath)) {
    globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }
  const systemPromptAppend = buildSystemPromptAppend(globalClaudeMd, seedContext);
  const taskSource = (
    containerInput.memoryMcp?.taskSource
    || containerInput.arcTools?.taskSource
    || ''
  ).toLowerCase();
  const isForumTask = isOpenAIForumPhase(taskSource);

  const maxTurns = resolveOpenAIMaxTurns(taskSource, sdkEnv.KSI_OPENAI_MAX_TURNS);
  const selectedModel = sdkEnv.MODEL;
  if (!selectedModel) {
    throw new Error(
      'MODEL env var is required for the OpenAI agent-runner; ' +
      'set it via the provider profile (e.g. MODEL=gpt-4o-mini).',
    );
  }
  // The host-side Python intentionally passes OPENAI_API_KEY through stdin's
  // `secrets` field (→ sdkEnv) rather than `-e` / process.env, so Bash
  // subprocesses that inherit the container env don't see the secret. But
  // @openai/agents' default client pulls from `process.env.OPENAI_API_KEY`
  // when no apiKey is provided — which is undefined here, so the Agent
  // constructor throws inside #getClient (surfaces as
  // `resolveModelForAgent` in the stack). Register the key with the SDK's
  // module-level setter instead; this keeps it off process.env while still
  // reaching the OpenAI client.
  const openaiKey = sdkEnv.OPENAI_API_KEY;
  if (!openaiKey) {
    throw new Error(
      'OPENAI_API_KEY not found in sdkEnv; set it via the provider profile ' +
      '(configs/ksi/.env.openai) and ensure container_host.py plumbs it.',
    );
  }
  setDefaultOpenAIKey(openaiKey);
  const modelSettings = buildModelSettings(selectedModel, sdkEnv);
  log(`Using model: ${selectedModel} | maxTurns: ${maxTurns} | taskSource: ${taskSource || '(unknown)'}`);

  // ARC-specific trial-budget guidance must live in the agent's system
  // instructions (not in TASK.md / TOOLS.md / compact retry prompt) because
  // gpt-5.4-mini reaches the answer via MCP arc tools and never reads task
  // workspace files via `shell`. Without this, GPT submits exactly one trial
  // per test on every task, leaving the second trial unused — see PR #593.
  const arcSystemAppend =
    taskSource === 'arc'
      ? 'For ARC tasks: you have max_trials=2 per test input. arc_submit_trial gives no correctness signal, so always submit two trials per test before advancing. If you have one credible answer, submit it twice; if you can construct a meaningfully different second hypothesis, submit that on the second trial instead.'
      : '';
  // Flag-gated scaffold parity (issue #634): default OFF. With the flag on,
  // non-ARC OpenAI runs get a richer agentic-coding system prompt (in the
  // spirit of the claude_code preset, not a verbatim copy) prepended ahead of
  // any global CLAUDE.md / seed-context append. ARC keeps its dedicated
  // trial-budget guidance below.
  const parityEnabled = isParityToolsEnabled(sdkEnv);
  const parityPrefix =
    parityEnabled && taskSource !== 'arc' ? `${buildParityAgenticInstructions()}\n\n` : '';
  const baseInstructions =
    `${parityPrefix}${systemPromptAppend?.trim() || ''}`.trim() ||
    'You are running inside the KSI container runtime. Use the provided tools and workspace carefully.';
  const defaultInstructions = arcSystemAppend
    ? `${baseInstructions}\n\n${arcSystemAppend}`
    : baseInstructions;
  const minimalInstructions = arcSystemAppend
    ? `You are running inside the KSI container runtime. Use the provided tools and workspace carefully. Read task files from the workspace instead of relying on preloaded prompt text.\n\n${arcSystemAppend}`
    : 'You are running inside the KSI container runtime. Use the provided tools and workspace carefully. Read task files from the workspace instead of relying on preloaded prompt text.';

  // The hosted `shellTool` / `applyPatchTool` variants in @openai/agents are
  // only accepted by OpenAI's Responses API for a narrow set of reasoning
  // models (o1/o3/gpt-5 family). Sending them to gpt-4o-mini fails with
  //   400 Tool 'shell' is not supported with gpt-4o-mini.
  // which aborts the container before any real work can happen and was the
  // first hard blocker for the GPT-4o-mini KT sweep. Declaring the same
  // capabilities as plain function tools works across every OpenAI model —
  // both gpt-4o-mini and reasoning models can call them via standard
  // function-calling — and the on-host implementation (createShell /
  // createEditor) is unchanged.
  const shellImpl = createShell(cwd, sdkEnv);
  const editorImpl = createEditor(cwd);
  const shellFnTool = tool({
    name: 'shell',
    description:
      'Execute one or more shell commands in the task workspace. Commands are joined by newlines and run under /bin/sh. Returns the captured stdout/stderr plus the exit code of the last invocation.',
    strict: false,
    parameters: {
      type: 'object',
      properties: {
        commands: {
          type: 'array',
          items: { type: 'string' },
          description: 'Shell commands to execute, joined with newlines.',
        },
        timeoutMs: {
          type: 'number',
          description: 'Optional timeout in milliseconds (default 120000).',
        },
        maxOutputLength: {
          type: 'number',
          description: 'Optional cap on captured stdout/stderr (default 50000).',
        },
      },
      required: ['commands'],
      additionalProperties: false,
    } as any,
    execute: async (input: any) => {
      const { commands, timeoutMs, maxOutputLength } = input ?? {};
      const result = await shellImpl.run({
        commands: Array.isArray(commands) ? commands : [String(commands ?? '')],
        timeoutMs,
        maxOutputLength,
      });
      return safeStringify(result.output);
    },
  } as any);
  const applyPatchFnTool = tool({
    name: 'apply_patch',
    description:
      'Create, update, or delete a file in the task workspace. For create/update, pass the new file content (or a unified diff) in `diff`. For delete, `diff` is ignored.',
    strict: false,
    parameters: {
      type: 'object',
      properties: {
        operation: {
          type: 'string',
          enum: ['create', 'update', 'delete'],
          description: 'Which edit to perform.',
        },
        path: {
          type: 'string',
          description: 'Workspace-relative file path. Escapes workspace root are rejected.',
        },
        diff: {
          type: 'string',
          description: 'File content (for create) or unified diff (for update). Ignored for delete.',
        },
      },
      required: ['operation', 'path'],
      additionalProperties: false,
    } as any,
    execute: async (input: any) => {
      const { operation, path: filePath, diff } = input ?? {};
      const body = String(diff ?? '');
      let out: { status: 'completed' | 'failed'; output?: string };
      switch (operation) {
        case 'create':
          out = await editorImpl.createFile({ path: filePath, diff: body });
          break;
        case 'update':
          out = await editorImpl.updateFile({ path: filePath, diff: body });
          break;
        case 'delete':
          out = await editorImpl.deleteFile({ path: filePath });
          break;
        default:
          out = { status: 'failed', output: `Unknown operation: ${operation}` };
      }
      return safeStringify(out);
    },
  } as any);

  // Flag-gated scaffold-parity native filesystem tools (issue #634). Default
  // OFF. When enabled, mirror the Claude path's native Read/Write/Edit/Glob/
  // Grep with function-tool equivalents. All are sandboxed to `cwd` via
  // `resolveWorkspacePath` (same confinement shell/apply_patch already use).
  const parityFs = createParityFsTools(cwd);
  const readFileFnTool = tool({
    name: 'read_file',
    description:
      'Read a UTF-8 text file from the task workspace. Returns {content, truncated}. Paths are workspace-relative; escapes are rejected.',
    strict: false,
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Workspace-relative file path.' },
        max_bytes: { type: 'number', description: 'Optional read cap (default/maximum 262144).' },
      },
      required: ['path'],
      additionalProperties: false,
    } as any,
    execute: async (input: any) => parityFs.readFile(input ?? {}),
  } as any);
  const writeFileFnTool = tool({
    name: 'write_file',
    description:
      'Create or overwrite a file in the task workspace with the given content. Parent directories are created as needed.',
    strict: false,
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Workspace-relative file path.' },
        content: { type: 'string', description: 'Full file content to write.' },
      },
      required: ['path', 'content'],
      additionalProperties: false,
    } as any,
    execute: async (input: any) => parityFs.writeFile(input ?? {}),
  } as any);
  const editFileFnTool = tool({
    name: 'edit_file',
    description:
      'Replace an exact string in a workspace file (string-replace semantics, like the Claude Edit tool). Fails if old_string is absent or non-unique unless replace_all=true.',
    strict: false,
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Workspace-relative file path.' },
        old_string: { type: 'string', description: 'Exact text to replace (must be unique unless replace_all).' },
        new_string: { type: 'string', description: 'Replacement text.' },
        replace_all: { type: 'boolean', description: 'Replace every occurrence instead of requiring uniqueness.' },
      },
      required: ['path', 'old_string', 'new_string'],
      additionalProperties: false,
    } as any,
    execute: async (input: any) => parityFs.editFile(input ?? {}),
  } as any);
  const globFnTool = tool({
    name: 'glob',
    description:
      'Find files in the task workspace matching a glob pattern (supports **, *, ?). Returns workspace-relative paths. Heavy/VCS dirs (node_modules, .git, ...) are skipped.',
    strict: false,
    parameters: {
      type: 'object',
      properties: {
        pattern: { type: 'string', description: 'Glob pattern, e.g. "**/*.py" or "src/*.ts".' },
      },
      required: ['pattern'],
      additionalProperties: false,
    } as any,
    execute: async (input: any) => parityFs.glob(input ?? {}),
  } as any);
  const grepFnTool = tool({
    name: 'grep',
    description:
      'Search workspace file contents for a JavaScript regular expression. Returns {path, line, text} matches. Searches the whole workspace unless `path` narrows it. Binary files are skipped.',
    strict: false,
    parameters: {
      type: 'object',
      properties: {
        pattern: { type: 'string', description: 'Regular expression to search for.' },
        path: { type: 'string', description: 'Optional workspace-relative file or directory to confine the search.' },
        ignore_case: { type: 'boolean', description: 'Case-insensitive match.' },
        max_matches: { type: 'number', description: 'Cap on returned matches (default/maximum 1000).' },
      },
      required: ['pattern'],
      additionalProperties: false,
    } as any,
    execute: async (input: any) => parityFs.grep(input ?? {}),
  } as any);
  const parityFsTools = [readFileFnTool, writeFileFnTool, editFileFnTool, globFnTool, grepFnTool];

  const mcpServers = await createMcpServers(containerInput, sdkEnv);
  // ARC used to get `tools: []` and rely on the ARC MCP server for grid
  // fetches/submission. That server is gone — ARC now runs natively (the agent
  // reads payload.json and writes attempt files via the shell/native tools),
  // so it uses the same tool surface as every other task.
  // Trust boundary (issue #1221): forum/MCP-protocol phases (per_task_forum /
  // cross_task_forum) are MCP-only — omit the native shell/apply_patch/parity
  // filesystem tools so a forum agent cannot shell out and read/write the raw
  // knowledge/runtime SQLite DBs mounted into the container, bypassing the
  // memory MCP layer. Benchmark task phases (arc/polyglot/swebench/tb2) keep
  // the full native surface. Mirrors the Anthropic direct-forum adapter, which
  // already exposes forum MCP tools only.
  const tools = selectOpenAINativeTools({
    isForumTask,
    parityEnabled,
    shellFnTool,
    applyPatchFnTool,
    parityFsTools,
  });
  const makeAgent = (instructions: string): Agent =>
    new Agent({
      name: 'KsiOpenAIContainerAgent',
      model: selectedModel,
      ...(modelSettings ? { modelSettings } : {}),
      instructions,
      tools,
      mcpServers,
    } as any);

  try {
    const agent = makeAgent(defaultInstructions);

    const runOptions = {
      previousResponseId: previousResponseId || undefined,
      maxTurns,
    } as any;
    // Two distinct recoverable-error paths:
    //   1. MaxTurnsExceededError — salvage partial state from err.state so
    //      completed tool calls + tokens aren't thrown away (PR #383).
    //   2. InvalidPromptError — retry once with a compact prompt, then
    //      again with minimal instructions + fresh session (PR #392).
    //   Other errors (ToolCallError, ModelBehaviorError, UserError, …)
    //   are deliberately NOT caught here — they propagate to index.ts's
    //   outer try/catch, which emits the full silent-diagnostic envelope.
    const handleMaxTurns = (err: unknown) => {
      const salvage = salvageResultFromState((err as any)?.state);
      const salvagedUsage = usageFromResult(salvage);
      const salvagedToolTrace = extractToolTrace(salvage);
      const salvagedText = extractBestOutput(salvage, isForumTask);
      const salvagedSessionId = salvage.lastResponseId || previousResponseId || undefined;
      const errorMessage =
        `MaxTurnsExceededError (maxTurns=${maxTurns}, ` +
        `taskSource=${taskSource || 'unknown'}): ` +
        `${err instanceof Error && err.message ? err.message : 'agent exceeded turn budget'} ` +
        `[salvaged tools=${salvagedToolTrace.length} ` +
        `rawResponses=${Array.isArray(salvage.rawResponses) ? salvage.rawResponses.length : 0}]`;
      log(
        `OpenAI run hit MaxTurnsExceededError (maxTurns=${maxTurns}, taskSource=${taskSource || 'unknown'}); ` +
        `salvaging partial state: tools=${salvagedToolTrace.length} ` +
        `input_tokens=${salvagedUsage.input_tokens} output_tokens=${salvagedUsage.output_tokens} ` +
        `responseId=${salvagedSessionId || 'none'}`,
      );
      return {
        status: 'error' as const,
        error: errorMessage,
        newSessionId: salvagedSessionId,
        resultText: salvagedText,
        toolTrace: salvagedToolTrace,
        ...salvagedUsage,
      };
    };
    const isMaxTurnsErr = (err: unknown): boolean =>
      err instanceof MaxTurnsExceededError ||
      (err instanceof Error && err.name === 'MaxTurnsExceededError');

    let result: any;
    try {
      result = await run(agent, prompt, runOptions);
    } catch (err) {
      if (isMaxTurnsErr(err)) {
        return handleMaxTurns(err);
      }
      if (!isInvalidPromptError(err)) {
        throw err;
      }
      log('OpenAI rejected initial prompt; retrying once with compact prompt.');
      try {
        result = await run(agent, buildCompactRetryPrompt(containerInput), runOptions);
      } catch (retryErr) {
        if (isMaxTurnsErr(retryErr)) {
          return handleMaxTurns(retryErr);
        }
        if (!isInvalidPromptError(retryErr)) {
          throw retryErr;
        }
        log('OpenAI rejected compact retry; retrying once with minimal instructions.');
        const fallbackAgent = makeAgent(minimalInstructions);
        try {
          result = await run(
            fallbackAgent,
            buildCompactRetryPrompt(containerInput),
            {
              ...runOptions,
              previousResponseId: undefined,
            } as any,
          );
        } catch (fallbackErr) {
          if (isMaxTurnsErr(fallbackErr)) {
            return handleMaxTurns(fallbackErr);
          }
          throw fallbackErr;
        }
      }
    }

    const usage = usageFromResult(result);
    const toolTrace = extractToolTrace(result);
    const resultText = extractBestOutput(result, isForumTask);
    const newSessionId = result?.lastResponseId || undefined;

    if (
      !resultText.trim() &&
      toolTrace.length === 0 &&
      usage.input_tokens === 0 &&
      usage.output_tokens === 0
    ) {
      throw new Error(
        `OpenAI run produced no observable output: ${JSON.stringify({
          newItems: Array.isArray(result?.newItems)
            ? result.newItems.map((item: any) => item?.type || 'unknown')
            : [],
          rawResponses: Array.isArray(result?.rawResponses)
            ? result.rawResponses.length
            : 0,
          finalOutputType: typeof result?.finalOutput,
          lastResponseId: result?.lastResponseId || null,
        })}`,
      );
    }

    log(`OpenAI run complete. responseId=${newSessionId || 'none'} tools=${toolTrace.length}`);

    // Polyglot test-feedback retry loop (Aider protocol) — OpenAI path.
    // Mirrors query_runner.ts's post-success block on the Claude path: write
    // a barrier sentinel with the live model output, wait for the host to
    // run the real evaluator, and on failure run bounded retry rounds with
    // the same workspace tools. Unlike phase1 reflection, this DOES change
    // what gets returned as the graded result. Rounds continue the same
    // Responses-API conversation via previousResponseId chaining.
    let polyglotTFResult = resultText;
    let polyglotTFMeta: ContainerOutput['polyglot_test_feedback_meta'];
    let polyglotTFTokenUsage: ContainerOutput['polyglot_test_feedback_token_usage'];
    const polyglotTFCfg = containerInput.polyglotTestFeedback;
    if (polyglotTFCfg && polyglotTFCfg.enabled && resultText.trim() && polyglotTFCfg.triesRemaining > 1) {
      let currentResponseId: string | undefined = newSessionId;
      const outcome = await runOpenAIPolyglotTestFeedback({
        workspaceDir: cwd,
        agentId: polyglotTFCfg.agentId,
        fileList: polyglotTFCfg.fileList,
        modelOutput: resultText,
        triesRemaining: polyglotTFCfg.triesRemaining,
        maxLines: polyglotTFCfg.maxLines,
        pollTimeoutMs: polyglotTFCfg.evalResultPollTimeoutMs,
        logger: log,
        runRound: async (roundPrompt: string, round: number) => {
          const roundOptions = {
            previousResponseId: currentResponseId || undefined,
            maxTurns: polyglotTFCfg.maxTurnsPerRound,
          } as any;
          let roundResult: any;
          try {
            roundResult = await run(agent, roundPrompt, roundOptions);
          } catch (err) {
            if (isMaxTurnsErr(err)) {
              // The round hit its turn cap mid-flight, but its completed
              // tool calls already edited the workspace — salvage the
              // partial state (same as handleMaxTurns above) and let the
              // loop's next barrier eval score the on-disk result instead
              // of aborting the whole retry protocol.
              log(
                `polyglot_test_feedback: retry round ${round} hit ` +
                `MaxTurnsExceededError (maxTurns=${polyglotTFCfg.maxTurnsPerRound}); salvaging partial state.`,
              );
              const salvage = salvageResultFromState((err as any)?.state);
              if (salvage.lastResponseId) currentResponseId = salvage.lastResponseId;
              return {
                text: extractBestOutput(salvage, isForumTask),
                usage: usageFromResult(salvage),
              };
            }
            throw err;
          }
          if (roundResult?.lastResponseId) currentResponseId = roundResult.lastResponseId;
          return {
            text: extractBestOutput(roundResult, isForumTask),
            usage: usageFromResult(roundResult),
          };
        },
      });
      polyglotTFResult = outcome.finalResult ?? resultText;
      polyglotTFMeta = {
        enabled: true,
        rounds_used: outcome.roundsUsed,
        attempt_1_eval_summary: outcome.attempt1EvalSummary,
        captured: outcome.captured,
        note: outcome.note,
        final_eval_matches_output: outcome.finalEvalMatchesOutput,
      };
      if (
        outcome.tokenUsage.input_tokens || outcome.tokenUsage.output_tokens
        || outcome.tokenUsage.cache_creation_input_tokens || outcome.tokenUsage.cache_read_input_tokens
      ) {
        polyglotTFTokenUsage = outcome.tokenUsage;
      }
    } else if (polyglotTFCfg && polyglotTFCfg.enabled && polyglotTFCfg.triesRemaining > 1) {
      polyglotTFMeta = {
        enabled: true,
        rounds_used: 0,
        attempt_1_eval_summary: null,
        captured: false,
        note: 'polyglot_test_feedback: skipped (no effectiveResult to retry on)',
        final_eval_matches_output: false,
      };
    }

    return {
      newSessionId,
      resultText: polyglotTFResult,
      toolTrace,
      ...usage,
      ...(polyglotTFMeta ? { polyglot_test_feedback_meta: polyglotTFMeta } : {}),
      ...(polyglotTFTokenUsage ? { polyglot_test_feedback_token_usage: polyglotTFTokenUsage } : {}),
    };
  } finally {
    await Promise.all(
      mcpServers.map(async (server) => {
        try {
          await server.close();
        } catch {
          // best-effort
        }
      }),
    );
  }
}
