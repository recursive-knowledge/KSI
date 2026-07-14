/**
 * Issue #1221 — OpenAI forum phases must be MCP-only.
 *
 * Forum/MCP-protocol phases (`per_task_forum`, `cross_task_forum`) talk to the
 * shared discussion/knowledge substrate exclusively through the memory MCP
 * server. Handing those runs native `shell` / `apply_patch` (or the flag-gated
 * parity filesystem tools) lets a forum prompt shell out and read/write the raw
 * SQLite DBs mounted into the container, bypassing the MCP layer.
 *
 * The pure, dependency-free `selectOpenAINativeTools` helper is exercised for
 * real through `tsx` (like arc_nomcp_synthesis.test.mjs) so we test the shipped
 * TypeScript, not a copy. Skipped only if `runtime_runner/node_modules` is not
 * installed. The always-run source pins below guarantee openai.ts routes forum
 * runs through the selector regardless of tsx availability.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(
  repoRoot,
  'runtime_runner',
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'tsx.cmd' : 'tsx',
);
const tsxSkip = !fs.existsSync(tsxBin);

// Drive the real helper through tsx; return {isForumTask,parity} -> tool names.
function selectNames({ isForumTask, parityEnabled }) {
  const source = `
    import { selectOpenAINativeTools } from "./runtime_runner/agent-runner/src/openai_tool_selection.ts";
    const tools = selectOpenAINativeTools({
      isForumTask: ${JSON.stringify(isForumTask)},
      parityEnabled: ${JSON.stringify(parityEnabled)},
      shellFnTool: { name: "shell" },
      applyPatchFnTool: { name: "apply_patch" },
      parityFsTools: [
        { name: "read_file" }, { name: "write_file" },
        { name: "edit_file" }, { name: "glob" }, { name: "grep" },
      ],
    });
    console.log(JSON.stringify(tools.map((t) => t.name)));
  `;
  const result = spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    encoding: 'utf8',
    cwd: repoRoot,
  });
  assert.equal(result.status, 0, `tsx exited nonzero: ${result.status}\n${result.stderr}`);
  return JSON.parse(result.stdout.trim());
}

// Drive isOpenAIForumPhase through tsx for a batch of task sources.
function forumPhaseFlags(sources) {
  const source = `
    import { isOpenAIForumPhase } from "./runtime_runner/agent-runner/src/openai_tool_selection.ts";
    const srcs = ${JSON.stringify(sources)};
    console.log(JSON.stringify(srcs.map((s) => isOpenAIForumPhase(s))));
  `;
  const result = spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    encoding: 'utf8',
    cwd: repoRoot,
  });
  assert.equal(result.status, 0, `tsx exited nonzero: ${result.status}\n${result.stderr}`);
  return JSON.parse(result.stdout.trim());
}

describe('OpenAI forum phases are MCP-only (behavioral)', () => {
  it('omits shell/apply_patch/parity for forum runs (parity off AND on)', { skip: tsxSkip }, () => {
    assert.deepEqual(selectNames({ isForumTask: true, parityEnabled: false }), []);
    assert.deepEqual(selectNames({ isForumTask: true, parityEnabled: true }), []);
  });

  it('keeps shell+apply_patch for a benchmark task phase (parity off)', { skip: tsxSkip }, () => {
    assert.deepEqual(selectNames({ isForumTask: false, parityEnabled: false }), ['shell', 'apply_patch']);
  });

  it('adds parity fs tools for a benchmark task phase when parity is enabled', { skip: tsxSkip }, () => {
    assert.deepEqual(selectNames({ isForumTask: false, parityEnabled: true }), [
      'shell',
      'apply_patch',
      'read_file',
      'write_file',
      'edit_file',
      'glob',
      'grep',
    ]);
  });

  it('classifies only the two forum phases as MCP-only', { skip: tsxSkip }, () => {
    const flags = forumPhaseFlags([
      'per_task_forum',
      'cross_task_forum',
      'CROSS_TASK_FORUM',
      'arc',
      'polyglot',
      'swebench_pro',
      'terminal_bench_2',
      '',
    ]);
    assert.deepEqual(flags, [true, true, true, false, false, false, false, false]);
  });
});

describe('openai.ts routes forum runs through the MCP-only selector (source pins)', () => {
  const openai = fs.readFileSync(
    path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'openai.ts'),
    'utf-8',
  );

  it('imports and assembles the native tool set via shared forum helpers', () => {
    assert.match(
      openai,
      /import \{ isOpenAIForumPhase, selectOpenAINativeTools \} from '\.\/openai_tool_selection\.js'/,
    );
    assert.doesNotMatch(openai, /const FORUM_PHASES = new Set/);
    assert.match(openai, /const isForumTask = isOpenAIForumPhase\(taskSource\)/);
    const start = openai.indexOf('const tools = selectOpenAINativeTools({');
    assert.ok(start >= 0, 'openai.ts must build tools via selectOpenAINativeTools');
    const block = openai.slice(start, start + 300);
    assert.match(block, /isForumTask,/);
    assert.match(block, /parityEnabled,/);
  });

  it('no longer inlines the unconditional shell+apply_patch tool array', () => {
    // The forum-blind inline form must be gone from openai.ts (moved to helper).
    assert.doesNotMatch(
      openai,
      /const tools = parityEnabled\s*\?\s*\[shellFnTool, applyPatchFnTool/,
    );
  });
});
