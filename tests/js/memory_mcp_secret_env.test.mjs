/**
 * Memory MCP subprocesses share a process namespace with the in-container
 * agent. Do not place host Hugging Face tokens in those subprocess envs:
 * shell-enabled agents can read same-UID process environments through /proc.
 */

import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const tsxAvailable = fs.existsSync(tsxBin);

function buildEnvSamples() {
  const queryConfigTs = path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'query_config.ts');
  const memoryMcpEnvTs = path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'memory_mcp_env.ts');
  const directForumTs = path.join(
    repoRoot,
    'runtime_runner',
    'agent-runner',
    'src',
    'anthropic_direct_forum.ts',
  );
  const source = `
    import fs from 'fs';
    import { buildMcpServerConfig } from ${JSON.stringify(queryConfigTs)};
    import { buildOpenAIMemoryMcpEnv } from ${JSON.stringify(memoryMcpEnvTs)};
    import { buildMemoryMcpEnv as buildDirectForumMemoryMcpEnv } from ${JSON.stringify(directForumTs)};

    const originalExistsSync = fs.existsSync.bind(fs);
    fs.existsSync = (candidate) => candidate === '/app/memory/mcp_server.py' || originalExistsSync(candidate);

    const sdkEnv = {
      HF_TOKEN: 'hf_secret',
      HUGGING_FACE_HUB_TOKEN: 'hf_hub_secret',
      HF_HOME: '/cache/huggingface',
      SENTENCE_TRANSFORMERS_HOME: '/cache/sentence-transformers',
      KSI_EMBEDDING_MODEL: 'local-embedding-model',
      MEMORY_ENABLE_SEMANTIC_SEARCH: '1',
    };
    const memoryMcp = {
      dbPath: '/host/knowledge.sqlite',
      snapshotPath: '/host/snapshot.json',
      taskSource: 'arc',
      forumTaskIds: ['a', 'b'],
      experiment: 'exp',
    };
    const allowedTools = [];
    const scheduled = buildMcpServerConfig({ memoryMcp }, sdkEnv, 'arc', allowedTools).memory.env;
    const openai = buildOpenAIMemoryMcpEnv({ memoryMcp }, sdkEnv, 'arc');
    const directForum = buildDirectForumMemoryMcpEnv({
      memoryMcp: { ...memoryMcp, taskSource: 'per_task_forum' },
    }, sdkEnv);
    process.stdout.write(JSON.stringify({ scheduled, openai, directForum, allowedTools }));
  `;
  const result = spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, `tsx failed: status=${result.status}\nstdout=${result.stdout}\nstderr=${result.stderr}`);
  return JSON.parse(result.stdout);
}

describe('memory MCP subprocess envs do not receive host HF tokens', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/');
    return;
  }

  it('omits token secrets while retaining cache configuration', () => {
    const samples = buildEnvSamples();
    for (const [name, env] of Object.entries({
      scheduled: samples.scheduled,
      openai: samples.openai,
      directForum: samples.directForum,
    })) {
      assert.equal(env.HF_TOKEN, undefined, `${name} env must not include HF_TOKEN`);
      assert.equal(
        env.HUGGING_FACE_HUB_TOKEN,
        undefined,
        `${name} env must not include HUGGING_FACE_HUB_TOKEN`,
      );
      assert.equal(env.HF_HOME, '/cache/huggingface');
      assert.equal(env.SENTENCE_TRANSFORMERS_HOME, '/cache/sentence-transformers');
      assert.equal(env.KSI_EMBEDDING_MODEL, 'local-embedding-model');
    }
    assert.deepEqual(samples.allowedTools, ['mcp__memory__*']);
  });
});
