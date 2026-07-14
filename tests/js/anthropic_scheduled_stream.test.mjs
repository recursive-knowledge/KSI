import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');
const readAgentSrc = (name) => fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', name),
  'utf-8',
);
// index.ts is now a thin dispatch entrypoint; the scheduled-loop machinery it
// used to contain lives in query_runner.ts / query_config.ts / tool_trace.ts.
// Concatenate them so these source-pinning assertions follow the code.
const agentRunner = [
  'index.ts',
  'query_runner.ts',
  'query_config.ts',
  'tool_trace.ts',
].map(readAgentSrc).join('\n');
const directForumRunner = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'anthropic_direct_forum.ts'),
  'utf-8',
);
const directHistory = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'anthropic_direct_history.ts'),
  'utf-8',
);
const tsxBin = path.join(
  repoRoot,
  'runtime_runner',
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'tsx.cmd' : 'tsx',
);
const tsxSkip = fs.existsSync(tsxBin)
  ? undefined
  : 'runtime_runner/node_modules/.bin/tsx is not installed';

function runTsxFixture(source) {
  const result = spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    cwd: path.join(repoRoot, 'runtime_runner'),
    encoding: 'utf-8',
    env: {
      ...process.env,
      NODE_NO_WARNINGS: '1',
    },
  });
  assert.equal(
    result.status,
    0,
    [
      `tsx fixture failed with status ${result.status}`,
      result.stdout,
      result.stderr,
    ].filter(Boolean).join('\n'),
  );
  return result.stdout.trim();
}

describe('Anthropic scheduled benchmark streaming', () => {
  it('uses MessageStream for scheduled jobs so Claude can continue after tool calls', () => {
    assert.doesNotMatch(
      agentRunner,
      /const useInteractiveStream = /,
      'scheduled jobs must not fall back to plain string prompts',
    );
    assert.match(
      agentRunner,
      /const stream = new MessageStream\(\);\s*stream\.push\(prompt\);\s*stream\.end\(\);/,
      'scheduled async input must close after the initial prompt',
    );
    assert.doesNotMatch(
      agentRunner,
      /ipcPolling/,
      'interactive IPC polling was removed with the NanoClaw interactive mode',
    );
    assert.match(agentRunner, /prompt: stream \?\? prompt,/);
  });

  it('denies native Claude file/shell tools on scheduled MCP-protocol (forum) jobs', () => {
    // ARC now runs natively (attempt files), so the only MCP-protocol-only
    // scheduled jobs are the forum phases. These pins guard that forum jobs
    // still cannot reach native file/shell tools (issue #1115).
    assert.match(
      agentRunner,
      /isScheduledMcpProtocolTask[\s\S]*\? \[\]/,
      'scheduled forum jobs must not allow native Claude file/shell tools',
    );
    assert.match(
      agentRunner,
      /const protocolNativeToolDenials = isMcpProtocolOnlyTask \? \[\.\.\.NATIVE_FILE_SHELL_TOOLS\] : \[\]/,
      'scheduled MCP protocol jobs must deny native Claude tools through disallowedTools',
    );
    // The subagent-spawning tools MUST be in the denial list: under
    // bypassPermissions a spawned subagent does not inherit the parent's
    // disallowedTools, so an undenied `Task` re-opens the raw-DB exfil that
    // issue #1115 closes. Pin them inside the NATIVE_FILE_SHELL_TOOLS array.
    const nativeToolsBlock = agentRunner.match(
      /export const NATIVE_FILE_SHELL_TOOLS = \[([\s\S]*?)\]/,
    );
    assert.ok(nativeToolsBlock, 'NATIVE_FILE_SHELL_TOOLS array must be declared');
    for (const toolName of [
      'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
      'Task', 'TaskOutput', 'TaskStop',
    ]) {
      assert.match(
        nativeToolsBlock[1],
        new RegExp(`'${toolName}'`),
        `NATIVE_FILE_SHELL_TOOLS must deny ${toolName}`,
      );
    }
    assert.match(
      agentRunner,
      /tools:\s*\(?isScheduledMcpProtocolTask\b[^?\n]*\?\s*\[\]\s*:\s*undefined/,
      'scheduled MCP protocol jobs must restrict built-in tools via the SDK tools option',
    );
    assert.match(
      agentRunner,
      /refusing fallback success/,
      'partial scheduled forum turns must not be emitted as success',
    );
  });

  it('treats split forum phases as MCP protocol tasks with explicit failures', () => {
    assert.match(
      agentRunner,
      /runAnthropicDirectForumQuery/,
      'scheduled Anthropic forum phases must bypass the Claude Code MCP bridge by default',
    );
    assert.match(
      agentRunner,
      /KSI_ANTHROPIC_FORUM_ADAPTER\s*\|\|\s*'direct'/,
      'direct forum adapter must be the default with an opt-out env override',
    );
    for (const phase of ['per_task_forum', 'cross_task_forum']) {
      assert.match(agentRunner, new RegExp(`'${phase}'`));
    }
    assert.match(agentRunner, /const isForumTask = forumTaskSources\.has/);
    assert.match(
      agentRunner,
      /Scheduled forum task ended with pending tool call\(s\)/,
      'forum result-event path must not accept dangling tool calls',
    );
    assert.match(
      agentRunner,
      /forum tool loop/,
      'forum fallback path must report a forum-specific incomplete tool loop',
    );
  });

  it('implements direct Anthropic forum MCP calls through the Python forum server', () => {
    assert.match(directForumRunner, /python3/);
    assert.match(directForumRunner, /\/app\/memory\/mcp_server\.py/);
    assert.match(directForumRunner, /tools\/call/);
    assert.match(directForumRunner, /forum_signal_done/);
    assert.match(
      directForumRunner,
      /DIRECT_FORUM_TOOL_ALLOWLIST/,
      'direct forum adapter should expose only protocol-required MCP tools',
    );
    assert.match(
      directForumRunner,
      /'forum_read'/,
      'direct forum adapter should allow forum_read for later-round reply grounding',
    );
    assert.match(
      directForumRunner,
      /buildInitialPrompt\(prompt,\s*containerInput\)/,
      'direct forum adapter should retain orchestrator prompt context',
    );
    assert.match(
      directForumRunner,
      /mcp__memory__/,
      'direct forum adapter must preserve memory-prefixed tool trace names',
    );
  });

  it('DIRECT_FORUM_TOOL_ALLOWLIST matches the asserted forum tool surface', () => {
    // Drift guard (mirrors tests/js/arc_mcp_registration.test.mjs, issue #693):
    // the request-body assertion further down keys off a hardcoded forum tool
    // list, so a tool added to DIRECT_FORUM_TOOL_ALLOWLIST without updating
    // that list (or vice versa) would drift silently. Parse the real Set and
    // require an exact match against the canonical surface below. This guard
    // only pins the Set<->literal agreement; Python handler coverage for these
    // tools is enforced separately by tests/test_forum_mcp_handler_coverage.py.
    const m = directForumRunner.match(
      /const DIRECT_FORUM_TOOL_ALLOWLIST\s*=\s*new Set\(\[([\s\S]*?)\]\)/,
    );
    assert.ok(
      m,
      'anthropic_direct_forum.ts must declare DIRECT_FORUM_TOOL_ALLOWLIST',
    );
    // Drop line comments so a commented-out entry can't leak into the parse.
    const body = m[1].replace(/\/\/[^\n]*/g, '');
    const sourceAllowlist = [...body.matchAll(/'([^']+)'/g)]
      .map((x) => x[1])
      .sort();
    assert.deepEqual(
      sourceAllowlist,
      ['forum_post', 'forum_read', 'forum_signal_done', 'knowledge', 'query'],
      'DIRECT_FORUM_TOOL_ALLOWLIST drifted from the asserted forum tool ' +
        'surface; update the request-body assertion in this file to match.',
    );
  });

  it('the forum direct adapter does not mutate tool_results mid-loop', () => {
    // Compaction's rewrite-in-place behavior on consumed tool_results was
    // the root cause of `cache_read=0` on Haiku ARC (PR #503 → issue #535)
    // — and the same pathology produced `cache_read=0` across forum
    // phases (verified across 12+ post-fix audit DBs in the 2026-04-28
    // audit). The fix uses block-form system + initial user with
    // `cache_control: ephemeral`, no compaction, rolling marker on the most
    // recent tool_result.
    //
    // The compaction helper itself remains in `anthropic_direct_history.ts`
    // for now (other code paths and tooling reference it), but the forum
    // adapter does not call it.
    assert.match(directHistory, /compactConsumedToolResults/);
    assert.match(directHistory, /ksi compacted tool result/);
    assert.doesNotMatch(directForumRunner, /compactConsumedToolResults\(/);
  });

  it('compacts real Anthropic message arrays while preserving recent tool results', { skip: tsxSkip }, () => {
    runTsxFixture(`
      import { strict as assert } from 'node:assert';
      import { compactConsumedToolResults } from './agent-runner/src/anthropic_direct_history.ts';

      const messages = [
        { role: 'user', content: 'start' },
        { role: 'assistant', content: [{ type: 'tool_use', id: 'old-call', name: 'query', input: {} }] },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'old-call', content: [{ type: 'text', text: 'old payload ' + 'x'.repeat(2000) }] }] },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'already', content: [{ type: 'text', text: '[ksi compacted tool result] query result was already delivered.' }] }] },
        { role: 'assistant', content: [{ type: 'tool_use', id: 'recent-call', name: 'forum_post', input: {} }] },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'recent-call', content: [{ type: 'text', text: 'fresh forum_post payload' }] }] },
      ];
      const toolUseNamesById = new Map([
        ['old-call', 'query'],
        ['already', 'query'],
        ['recent-call', 'forum_post'],
      ]);

      compactConsumedToolResults(messages, toolUseNamesById, 1);

      assert.match(
        messages[2].content[0].content[0].text,
        /^\\[ksi compacted tool result\\] query result was already delivered/,
      );
      assert.equal(
        messages[3].content[0].content[0].text,
        '[ksi compacted tool result] query result was already delivered.',
        'pre-compacted results should not be rewritten',
      );
      assert.equal(
        messages[5].content[0].content[0].text,
        'fresh forum_post payload',
        'the most recent tool-result message should remain full fidelity',
      );
      assert.equal(messages[1].content[0].name, 'query', 'assistant tool_use blocks must remain intact');
    `);
  });

  it('filters direct forum tools and sends the orchestrator prompt into Anthropic', { skip: tsxSkip }, () => {
    runTsxFixture(`
      import { strict as assert } from 'node:assert';
      import fs from 'node:fs';
      import os from 'node:os';
      import path from 'node:path';

      const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-forum-mcp-'));
      const fakePython = path.join(tempDir, 'python3');
      fs.writeFileSync(fakePython, \`#!/usr/bin/env node
const tools = [
  { name: 'query', description: 'query', inputSchema: { type: 'object', properties: {} } },
  { name: 'knowledge', description: 'knowledge', inputSchema: { type: 'object', properties: {} } },
  { name: 'forum_read', description: 'forum_read', inputSchema: { type: 'object', properties: {} } },
  { name: 'forum_post', description: 'forum_post', inputSchema: { type: 'object', properties: {} } },
  { name: 'forum_signal_done', description: 'forum_signal_done', inputSchema: { type: 'object', properties: {} } },
  { name: 'forum_post_insight', description: 'legacy write tool', inputSchema: { type: 'object', properties: {} } },
  { name: 'shell', description: 'native shell escape', inputSchema: { type: 'object', properties: {} } },
];
let buffer = '';
process.stdin.setEncoding('utf8');
function send(id, result) {
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, result }) + '\\\\n');
}
function handle(raw) {
  const msg = JSON.parse(raw);
  if (!msg.id) return;
  if (msg.method === 'initialize') {
    send(msg.id, { protocolVersion: '2024-11-05', capabilities: {} });
  } else if (msg.method === 'tools/list') {
    send(msg.id, { tools });
  } else if (msg.method === 'tools/call') {
    send(msg.id, { content: [{ type: 'text', text: JSON.stringify({ status: 'ok', tool: msg.params.name }) }] });
  } else {
    send(msg.id, {});
  }
}
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let newline;
  while ((newline = buffer.indexOf('\\\\n')) >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (line) handle(line);
  }
});
\`);
      fs.chmodSync(fakePython, 0o755);
      process.env.PATH = tempDir + path.delimiter + process.env.PATH;

      const anthropicBodies = [];
      globalThis.fetch = async (_url, init) => {
        const body = JSON.parse(String(init.body));
        anthropicBodies.push(body);
        const response = anthropicBodies.length === 1
          ? {
              id: 'forum-turn-1',
              content: [{ type: 'tool_use', id: 'done-1', name: 'forum_signal_done', input: {} }],
              usage: { input_tokens: 11, output_tokens: 7 },
            }
          : {
              id: 'forum-turn-2',
              content: [{ type: 'text', text: 'forum complete' }],
              usage: { input_tokens: 13, output_tokens: 5 },
            };
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify(response);
          },
        };
      };

      const { runAnthropicDirectForumQuery } = await import('./agent-runner/src/anthropic_direct_forum.ts');
      const orchestratorPrompt = 'ORCHESTRATOR CONTEXT: use the assigned rubric and task packet.';
      const result = await runAnthropicDirectForumQuery(
        orchestratorPrompt,
        {
          assistantName: 'agent-a',
          memoryMcp: {
            dbPath: '/tmp/forum.db',
            snapshotPath: '/tmp/forum-snapshot.json',
            taskSource: 'per_task_forum',
            taskId: 'task-1',
            forumGeneration: 3,
            forumRound: 2,
            forumAgentId: 'agent-a',
            forumExpectedAgents: 4,
            forumTaskIds: ['task-1'],
            experiment: 'exp',
          },
        },
        {
          MODEL: 'claude-test',
          ANTHROPIC_API_KEY: 'test-key',
          KSI_ANTHROPIC_DIRECT_FORUM_MAX_TURNS: '2',
        },
      );

      assert.equal(result.status, undefined);
      assert.equal(result.resultText, 'forum complete');
      assert.deepEqual(
        anthropicBodies[0].tools.map((tool) => tool.name).sort(),
        ['forum_post', 'forum_read', 'forum_signal_done', 'knowledge', 'query'],
      );
      // Initial user message is now block-form so the text block can carry
      // cache_control: ephemeral. Pull the text out of the first block.
      const initialUserBlocks = anthropicBodies[0].messages[0].content;
      assert.ok(Array.isArray(initialUserBlocks), 'initial user must be block-form for cache_control');
      const initialUserText = initialUserBlocks[0].text;
      assert.equal(initialUserBlocks[0].type, 'text');
      assert.deepEqual(
        initialUserBlocks[0].cache_control,
        { type: 'ephemeral' },
        'initial user text block must carry cache_control: ephemeral',
      );
      assert.ok(
        initialUserText.includes(orchestratorPrompt),
        'direct forum must retain the orchestrator prompt context',
      );
      assert.ok(
        initialUserText.includes('call forum_read()'),
        'later forum rounds should tell the model to ground replies in forum_read',
      );
      // System is now also block-form with cache_control.
      const systemBlocks = anthropicBodies[0].system;
      assert.ok(Array.isArray(systemBlocks), 'system must be block-form for cache_control');
      assert.deepEqual(
        systemBlocks[0].cache_control,
        { type: 'ephemeral' },
        'system text block must carry cache_control: ephemeral',
      );
      assert.ok(
        result.toolTrace.some((entry) => entry.tool_name === 'mcp__memory__forum_signal_done'),
        'successful direct forum runs must record forum_signal_done in the trace',
      );
    `);
  });
});
