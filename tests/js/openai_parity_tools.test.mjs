/**
 * Issue #634 — flag-gated OpenAI scaffold parity.
 *
 * These are source-text invariant pins (the CI runs `node tests/js/*.test.mjs`
 * with only runtime_runner deps installed; the agent-runner SDK deps
 * @openai/agents et al. live only in the container image, so we cannot import
 * openai.ts here). They guarantee:
 *   1. The parity tools are GATED behind OPENAI_PARITY_TOOLS
 *      (default OFF — no behavior change with the flag unset).
 *   2. The richer native tool surface (read_file/write_file/edit_file/glob/
 *      grep) is present and confined to the workspace sandbox.
 *   3. The flag is forwarded host -> container by container_runner.ts.
 *
 * The pure, dependency-free `globToRegExp` helper cannot be imported from the
 * (SDK-dependent) openai.ts here, so it is covered two ways: (a) a faithful
 * re-implementation of its algorithm is exercised behaviorally against concrete
 * glob patterns, and (b) source pins assert openai.ts still encodes the same
 * translation branches, so the re-implementation cannot silently drift from the
 * real function.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');

const openai = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'openai.ts'),
  'utf-8',
);
const containerRunner = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'src', 'container_args.ts'),
  'utf-8',
);
const toolSelection = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'openai_tool_selection.ts'),
  'utf-8',
);

describe('OpenAI scaffold parity — flag gating (default OFF)', () => {
  it('reads the OPENAI_PARITY_TOOLS flag and treats false-y values as off', () => {
    assert.match(openai, /function isParityToolsEnabled\(/);
    assert.match(openai, /OPENAI_PARITY_TOOLS/);
    // The off-list must include the standard false-y tokens so the default and
    // explicit-disable both keep parity OFF.
    assert.match(openai, /\['0', 'false', 'no', 'off'\]/);
  });

  it('only adds parity fs tools when the flag is enabled', () => {
    // The native tool set is assembled by selectOpenAINativeTools; its parity
    // branch keeps shell+apply_patch and adds parityFsTools only when enabled.
    assert.match(openai, /selectOpenAINativeTools\(\{/);
    assert.match(
      toolSelection,
      /parityEnabled\s*\?\s*\[opts\.shellFnTool, opts\.applyPatchFnTool, \.\.\.opts\.parityFsTools\]\s*:\s*\[opts\.shellFnTool, opts\.applyPatchFnTool\]/s,
    );
  });

  it('only prepends the richer agentic system prompt when the flag is enabled', () => {
    assert.match(openai, /buildParityAgenticInstructions\(\)/);
    assert.match(openai, /parityEnabled && taskSource !== 'arc'/);
  });
});

describe('OpenAI scaffold parity — native tool surface', () => {
  it('declares read_file/write_file/edit_file/glob/grep function tools', () => {
    for (const name of ['read_file', 'write_file', 'edit_file', 'glob', 'grep']) {
      assert.match(openai, new RegExp(`name: '${name}'`), `missing tool ${name}`);
    }
  });

  it('edit_file uses string-replace semantics with replace_all guard', () => {
    assert.match(openai, /old_string/);
    assert.match(openai, /new_string/);
    assert.match(openai, /replace_all/);
    assert.match(openai, /matched \$\{occurrences\} times/);
  });

  it('confines every parity path to the workspace sandbox', () => {
    // createParityFsTools must route through resolveWorkspacePath (the same
    // confinement shell/apply_patch use).
    const factoryStart = openai.indexOf('export function createParityFsTools(');
    assert.ok(factoryStart >= 0, 'createParityFsTools must exist');
    const factoryBody = openai.slice(factoryStart, factoryStart + 4000);
    assert.match(factoryBody, /resolveWorkspacePath\(/);
  });

  it('bounds recursive walks (entry cap + skip heavy/VCS dirs)', () => {
    assert.match(openai, /PARITY_WALK_MAX_ENTRIES/);
    assert.match(openai, /PARITY_WALK_SKIP_DIRS/);
    assert.match(openai, /'node_modules'/);
    assert.match(openai, /'\.git'/);
  });

  it('does not follow symlinks during walks', () => {
    assert.match(openai, /entry\.isSymbolicLink\(\)/);
  });
});

// Faithful re-implementation of openai.ts::globToRegExp. Kept byte-aligned with
// the source; the source pins below fail if the real function drifts from this.
function globToRegExpReimpl(pattern) {
  let re = '';
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i];
    if (ch === '*') {
      if (pattern[i + 1] === '*') {
        re += '.*';
        i++;
        if (pattern[i + 1] === '/') {
          re += '(?:/)?';
          i++;
        }
      } else {
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

describe('globToRegExp translation contract', () => {
  it('** matches across path separators; * and ? do not', () => {
    assert.ok(globToRegExpReimpl('**/*.py').test('a/b/x.py'), '**/*.py should match nested');
    assert.ok(globToRegExpReimpl('**/*.py').test('x.py'), '**/ should be optional');
    assert.ok(globToRegExpReimpl('*.py').test('x.py'), '*.py matches a bare file');
    assert.ok(!globToRegExpReimpl('*.py').test('a/x.py'), '* must not cross a separator');
    assert.ok(globToRegExpReimpl('src/*.ts').test('src/a.ts'), 'src/*.ts matches one level');
    assert.ok(!globToRegExpReimpl('src/*.ts').test('src/a/b.ts'), 'src/*.ts is single-level');
    assert.ok(globToRegExpReimpl('a?b').test('axb'), '? matches one char');
    assert.ok(!globToRegExpReimpl('a?b').test('ab'), '? requires a char');
    assert.ok(!globToRegExpReimpl('a?b').test('a/b'), '? must not match a separator');
  });

  it('escapes regex metacharacters so they match literally', () => {
    assert.ok(globToRegExpReimpl('a.b').test('a.b'), 'dot is literal');
    assert.ok(!globToRegExpReimpl('a.b').test('axb'), 'dot must not be a wildcard');
    assert.ok(globToRegExpReimpl('v1.2+x').test('v1.2+x'), 'plus is literal');
  });

  it('openai.ts still encodes the same translation branches (drift guard)', () => {
    const start = openai.indexOf('export function globToRegExp(');
    assert.ok(start >= 0, 'globToRegExp must exist');
    const body = openai.slice(start, start + 1200);
    assert.match(body, /pattern\[i \+ 1\] === '\*'/, '** lookahead branch');
    assert.match(body, /re \+= '\.\*'/, '** -> .*');
    assert.match(body, /re \+= '\[\^\/\]\*'/, "single * -> [^/]*");
    assert.match(body, /re \+= '\[\^\/\]'/, '? -> [^/]');
    assert.match(body, /new RegExp\(`\^\$\{re\}\$`\)/, 'anchored ^...$');
  });
});

describe('OpenAI scaffold parity — hardening (regressions)', () => {
  it('edit_file single-match path splices by index (no $-pattern corruption)', () => {
    // String.prototype.replace would interpret $&/$$ in new_string; the fix
    // must splice via indexOf/slice instead.
    assert.doesNotMatch(openai, /current\.replace\(oldStr, newStr\)/, 'must not use String.replace for the literal edit');
    assert.match(openai, /current\.indexOf\(oldStr\)/);
  });

  it('readFile bounds the read instead of slurping the whole file', () => {
    assert.match(openai, /fs\.readSync\(/);
  });

  it('grep bounds CPU/memory: time budget, per-file size cap, guarded subpath', () => {
    assert.match(openai, /PARITY_GREP_TIME_BUDGET_MS/);
    assert.match(openai, /PARITY_GREP_MAX_FILE_BYTES/);
    // subPath resolution must be guarded so an escaping path fails gracefully.
    assert.match(openai, /searchRootAbs = subPath \? resolveWorkspacePath/);
  });
});

describe('container_runner forwards the parity flag', () => {
  it('whitelists OPENAI_PARITY_TOOLS for container env forwarding', () => {
    assert.match(containerRunner, /'OPENAI_PARITY_TOOLS'/);
  });
});
