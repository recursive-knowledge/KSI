import path from 'path';

import { isOpenAIForumPhase } from './openai_tool_selection.js';
import type { ContainerInput } from './shared_types.js';

export function buildOpenAIMemoryMcpEnv(
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  taskSource: string,
): Record<string, string> {
  if (!containerInput.memoryMcp) {
    throw new Error('OpenAI memory MCP env requires memoryMcp config.');
  }
  const dbFile = path.basename(containerInput.memoryMcp.dbPath);
  const snapshotFile = containerInput.memoryMcp.snapshotPath
    ? path.basename(containerInput.memoryMcp.snapshotPath)
    : '';
  const memoryToolset = isOpenAIForumPhase(taskSource) ? 'forum' : 'task';
  return {
    // dbFile may be '' when a forum snapshot failed to build host-side; the
    // MCP server then runs without a knowledge store (degraded, never live).
    KNOWLEDGE_DB_PATH: dbFile ? `/app/memory-db/${dbFile}` : '',
    // Snapshot DBs never change while mounted: immutable=1 lets SQLite skip
    // all locking/wal-index access (required for safety on macOS bind mounts).
    KNOWLEDGE_DB_IMMUTABLE: containerInput.memoryMcp.dbIsSnapshot ? '1' : '',
    MEMORY_SNAPSHOT_PATH: snapshotFile ? `/app/memory-db/${snapshotFile}` : '',
    MCP_TOOLSET: memoryToolset,
    FORUM_GENERATION: String(containerInput.memoryMcp.forumGeneration ?? 0),
    FORUM_ROUND: String(containerInput.memoryMcp.forumRound ?? 0),
    FORUM_AGENT_ID: containerInput.memoryMcp.forumAgentId ?? '',
    FORUM_EXPECTED_AGENTS: String(containerInput.memoryMcp.forumExpectedAgents ?? 0),
    FORUM_TASK_IDS: (containerInput.memoryMcp.forumTaskIds || []).join(','),
    MEMORY_EXPERIMENT: containerInput.memoryMcp.experiment ?? '',
    EXPERIMENT_NAME: process.env.EXPERIMENT_NAME || containerInput.memoryMcp.experiment || '',
    MEMORY_ENABLE_SEMANTIC_SEARCH: sdkEnv.MEMORY_ENABLE_SEMANTIC_SEARCH || '1',
    KSI_EMBEDDING_MODEL:
      sdkEnv.KSI_EMBEDDING_MODEL || 'google/embeddinggemma-300m',
    USE_TF: sdkEnv.USE_TF || '0',
    TOKENIZERS_PARALLELISM: sdkEnv.TOKENIZERS_PARALLELISM || 'false',
    HF_HOME: sdkEnv.HF_HOME || '/home/node/.cache/huggingface',
    SENTENCE_TRANSFORMERS_HOME:
      sdkEnv.SENTENCE_TRANSFORMERS_HOME || '/home/node/.cache/sentence-transformers',
  };
}
