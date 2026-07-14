import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import path from 'path';

import { AnthropicMessage } from './anthropic_direct_history.js';
import {
  ContainerInput,
  CrossTaskRoundResult,
  CrossTaskRoundUsage,
} from './shared_types.js';
import {
  responsePath as barrierResponsePath,
  waitForBarrierFile,
  writeSentinelFile as writeBarrierSentinel,
} from './barrier.js';
import {
  type AnthropicBlock,
  type AnthropicResponse,
  accumulateUsage,
  createMessage,
  textFromBlocks,
} from './anthropic_direct_transport.js';

const TOOL_OUTPUT_MAX_CHARS = 64 * 1024;
const MCP_REQUEST_TIMEOUT_MS = 60_000;

/**
 * Barrier name used by the cross-task R0->R1 shared-container round-trip.
 * Centralized so the host-side BarrierWatcher uses the same identifier
 * (see ``src/kcsi/orchestrator/engine.py`` cross-task forum phase). These
 * two literals MUST agree.
 */
export const CROSS_TASK_R1_BARRIER_NAME = 'cross_task_r1';

/** Default upper bound on how long the in-container R0->R1 wait blocks. */
const CROSS_TASK_R1_POLL_TIMEOUT_MS_DEFAULT = 600_000;
const CROSS_TASK_BARRIER_POLL_INTERVAL_MS = 500;
const CONTAINER_WORKSPACE_ROOT = '/workspace/task';

interface PendingMcpRequest {
  resolve: (value: Record<string, unknown>) => void;
  reject: (err: Error) => void;
  timer: NodeJS.Timeout;
}

export interface AnthropicDirectForumResult {
  status?: 'error';
  error?: string;
  newSessionId?: string;
  resultText: string;
  toolTrace: Array<Record<string, unknown>>;
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
  tokens_source: 'per_turn_sum' | 'unavailable';
  /**
   * Cross-task shared-container per-round outputs. Present only when the
   * ``crossTaskSharedContainer`` feature flag was on. The forum adapter
   * runs round 0, writes a barrier sentinel, waits for the host's response
   * carrying the round-1 prompt suffix, then continues the same Anthropic
   * Messages-API loop with a synthetic user turn for round 1. Per-round
   * tool traces and token usage are bookkept here for the host engine to
   * persist under the appropriate ``cross_task_forum_round_<n>`` phase
   * slugs and forum-bus drain windows.
   *
   * ``cross_task_round_1_result`` is absent when (a) the host barrier
   * timed out (graceful degrade — R0 envelope shipped, R1 skipped) or
   * (b) the R1 turn threw before a terminal state.
   */
  cross_task_round_0_result?: CrossTaskRoundResult;
  cross_task_round_1_result?: CrossTaskRoundResult;
  cross_task_shared_container_meta?: {
    enabled: boolean;
    r1_captured: boolean;
    note?: string;
    timed_out?: boolean;
    elapsed_ms?: number;
  };
}

// Forum task sources this direct adapter accepts (the gate at the top of
// runAnthropicDirectForumQuery rejects anything else). Both are container
// `task_source` values the orchestrator emits:
//   - `per_task_forum`   — the per-task forum phase (engine.py
//                          `_per_task_forum_phase`). Shares the single per-task
//                          guidance path in this adapter.
//   - `cross_task_forum` — the cross-task phase; the only mode with a dedicated
//                          in-adapter branch (the R0->R1 shared-container
//                          barrier, see `taskSource === 'cross_task_forum'`).
//
// History (#683): `forum_self` was removed (never emitted anywhere — dead), and
// the per-task wire tag `forum_debate` was renamed to `per_task_forum` so it
// matches the canonical phase / source_phase / config name used elsewhere.
const FORUM_PHASES = new Set([
  'per_task_forum',
  'cross_task_forum',
]);
const DIRECT_FORUM_TOOL_ALLOWLIST = new Set([
  'query',
  'knowledge',
  'forum_read',
  'forum_post',
  'forum_signal_done',
]);

function log(message: string): void {
  console.error(`[agent-runner/anthropic-direct-forum] ${message}`);
}

function safeStringify(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value == null) return '';
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

class McpStdioClient {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private nextId = 1;
  private pending = new Map<number, PendingMcpRequest>();
  private stdoutBuffer = '';
  private stderrTail = '';

  constructor(
    private command: string,
    private args: string[],
    private env: Record<string, string>,
  ) {}

  start(): void {
    this.proc = spawn(this.command, this.args, {
      env: this.env,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.proc.stdout.setEncoding('utf8');
    this.proc.stderr.setEncoding('utf8');
    this.proc.stdout.on('data', (chunk: string) => {
      this.stdoutBuffer += chunk;
      while (true) {
        const newline = this.stdoutBuffer.indexOf('\n');
        if (newline < 0) break;
        const line = this.stdoutBuffer.slice(0, newline).trim();
        this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);
        if (line) this.handleLine(line);
      }
    });
    this.proc.stderr.on('data', (chunk: string) => {
      this.stderrTail = (this.stderrTail + chunk).slice(-8000);
    });
    this.proc.on('close', (code) => {
      const err = new Error(`MCP server exited with code ${code}; stderr=${this.stderrTail.slice(-1000)}`);
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer);
        pending.reject(err);
      }
      this.pending.clear();
    });
  }

  async initialize(): Promise<Array<Record<string, unknown>>> {
    this.start();
    await this.request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'kcsi-direct-forum', version: '1.0.0' },
    });
    this.notify('notifications/initialized', {});
    const listed = await this.request('tools/list', {});
    const tools = (listed.tools || []) as Array<Record<string, unknown>>;
    return tools;
  }

  request(method: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    if (!this.proc?.stdin.writable) {
      return Promise.reject(new Error('MCP server is not running'));
    }
    const id = this.nextId++;
    const payload = { jsonrpc: '2.0', id, method, params };
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP request timed out: ${method}`));
      }, MCP_REQUEST_TIMEOUT_MS);
      this.pending.set(id, { resolve, reject, timer });
      this.proc!.stdin.write(`${JSON.stringify(payload)}\n`);
    });
  }

  notify(method: string, params: Record<string, unknown>): void {
    if (!this.proc?.stdin.writable) return;
    this.proc.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method, params })}\n`);
  }

  close(): void {
    if (!this.proc) return;
    try {
      this.proc.stdin.end();
    } catch {
      // best effort
    }
    try {
      this.proc.kill();
    } catch {
      // best effort
    }
  }

  private handleLine(line: string): void {
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(line) as Record<string, unknown>;
    } catch {
      return;
    }
    const id = Number(msg.id);
    if (!Number.isFinite(id)) return;
    const pending = this.pending.get(id);
    if (!pending) return;
    this.pending.delete(id);
    clearTimeout(pending.timer);
    if (msg.error && typeof msg.error === 'object') {
      const err = msg.error as Record<string, unknown>;
      pending.reject(new Error(String(err.message || 'MCP request failed')));
      return;
    }
    const result = msg.result && typeof msg.result === 'object'
      ? msg.result as Record<string, unknown>
      : {};
    pending.resolve(result);
  }
}

export function buildMemoryMcpEnv(
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
): Record<string, string> {
  if (!containerInput.memoryMcp) {
    throw new Error('Anthropic direct forum adapter requires memoryMcp config.');
  }
  const taskSource = String(containerInput.memoryMcp.taskSource || '').toLowerCase();
  if (!FORUM_PHASES.has(taskSource)) {
    throw new Error(`Anthropic direct forum adapter only supports forum phases, got ${taskSource || '(empty)'}.`);
  }
  const dbFile = path.basename(containerInput.memoryMcp.dbPath);
  const snapshotFile = containerInput.memoryMcp.snapshotPath
    ? path.basename(containerInput.memoryMcp.snapshotPath)
    : '';
  const env: Record<string, string> = {
    PATH: process.env.PATH || '/usr/local/bin:/usr/bin:/bin',
    HOME: process.env.HOME || '/home/node',
    KNOWLEDGE_DB_PATH: `/app/memory-db/${dbFile}`,
    MEMORY_SNAPSHOT_PATH: snapshotFile ? `/app/memory-db/${snapshotFile}` : '',
    MCP_TOOLSET: 'forum',
    FORUM_GENERATION: String(containerInput.memoryMcp.forumGeneration ?? 0),
    FORUM_ROUND: String(containerInput.memoryMcp.forumRound ?? 0),
    FORUM_AGENT_ID: containerInput.memoryMcp.forumAgentId ?? '',
    FORUM_EXPECTED_AGENTS: String(containerInput.memoryMcp.forumExpectedAgents ?? 0),
    FORUM_TASK_IDS: (containerInput.memoryMcp.forumTaskIds || []).join(','),
    MEMORY_EXPERIMENT: containerInput.memoryMcp.experiment ?? '',
    EXPERIMENT_NAME: process.env.EXPERIMENT_NAME || containerInput.memoryMcp.experiment || '',
    MEMORY_ENABLE_SEMANTIC_SEARCH: sdkEnv.MEMORY_ENABLE_SEMANTIC_SEARCH || '1',
    KCSI_EMBEDDING_MODEL: sdkEnv.KCSI_EMBEDDING_MODEL || 'google/embeddinggemma-300m',
    USE_TF: sdkEnv.USE_TF || '0',
    TOKENIZERS_PARALLELISM: sdkEnv.TOKENIZERS_PARALLELISM || 'false',
    HF_HOME: sdkEnv.HF_HOME || '/home/node/.cache/huggingface',
    SENTENCE_TRANSFORMERS_HOME:
      sdkEnv.SENTENCE_TRANSFORMERS_HOME || '/home/node/.cache/sentence-transformers',
  };
  return env;
}

function convertMcpToolsForAnthropic(tools: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  return tools
    .filter((tool) => typeof tool.name === 'string' && DIRECT_FORUM_TOOL_ALLOWLIST.has(tool.name))
    .map((tool) => ({
      name: tool.name,
      description: String(tool.description || ''),
      input_schema:
        (tool.inputSchema && typeof tool.inputSchema === 'object'
          ? tool.inputSchema
          : { type: 'object', properties: {} }),
    }));
}

function prefixedToolName(name: string): string {
  return name.startsWith('mcp__memory__') ? name : `mcp__memory__${name}`;
}

/**
 * Strip the rolling `cache_control` marker from prior turns before placing it
 * on the latest user block. Only text blocks in the INITIAL user message are
 * spared: their `cache_control` is the stable cache floor that lets every
 * subsequent turn read (system + tools + initial_user) from cache. Later
 * synthetic text turns (for example shared-container R1 prompts) are rolling
 * markers too; if they persist after a tool call, the next tool_result marker
 * can exceed Anthropic's 4-breakpoint cap.
 *
 * See the cache_stability invariant in `gotchas.md`: never mutate cached prefix
 * content for input-token savings.
 */
export function clearRollingCacheControl(messages: AnthropicMessage[]): void {
  for (const [msgIndex, msg] of messages.entries()) {
    if (msg.role !== 'user') continue;
    const content = (msg as { content?: unknown }).content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!block || typeof block !== 'object') continue;
      const blockObj = block as Record<string, unknown>;
      if (msgIndex === 0 && blockObj.type === 'text') continue;
      if ('cache_control' in blockObj) {
        delete blockObj.cache_control;
      }
    }
  }
}

function buildSystemPrompt(): string {
  return [
    'You are contributing to a scheduled discussion phase.',
    'The orchestrator-provided discussion prompt in the user message is authoritative for evidence, context, and objectives.',
    'Use only the provided discussion and knowledge tools. Native file and shell tools are unavailable in this direct scheduled adapter.',
    'Minimize tool turns: perform the required retrieval, post concise high-value observations, then signal done.',
    'Call forum_signal_done exactly once when finished.',
  ].join('\n');
}

/**
 * Build the cache-stable portion of the scheduled-guidance text.
 *
 * DELIBERATELY OMITS the round number and the round-conditional
 * `liveBoardGuidance` line — those vary per round and would invalidate
 * the cached prefix hash between rounds for the same agent. The
 * round-specific portion is built by `buildScheduledGuidanceSuffix` and
 * emitted after the cache_control marker by the caller.
 *
 * Cross-task forum ALSO omits `agentId`: this block is concatenated into
 * the `cache_control`-marked prefix (see `buildInitialPrefixText`), and
 * the cross-task prefix is meant to be byte-identical across all N agents
 * in a (generation, round) so cross-AGENT cache reuse can fire (≈1 write
 * + N−1 reads instead of N writes). The agent's identity is delivered in
 * the variable suffix instead — the Python `variable_suffix` opens with
 * the `You are agent {id} …` line (see `build_cross_task_discussion_parts`
 * in `src/kcsi/forum/prompt.py`), so nothing is lost. `generation` is
 * retained because it is the same for every agent in a generation, so it
 * does not break cross-agent prefix stability.
 *
 * Per-task forum KEEPS `agentId` in the header: its cached prefix (in
 * `build_per_task_discussion_parts`) is deliberately per-agent (an agent
 * discusses only its own task pages), so it targets within-agent /
 * cross-round reuse, not cross-agent reuse.
 *
 * See PR #572 follow-up: PR #572 placed the full `scheduledGuidance`
 * inside the cached prefix block, but the embedded round label still
 * mutated the prefix hash between rounds for the same agent —
 * defeating cross-round cache reuse for stable agent/generation pairs.
 */
export function buildScheduledGuidancePrefix(containerInput: ContainerInput): string {
  const memory = containerInput.memoryMcp;
  const taskSource = String(memory?.taskSource || '').toLowerCase();
  const generation = memory?.forumGeneration ?? 0;
  const agentId = memory?.forumAgentId || containerInput.assistantName || 'agent';
  if (taskSource === 'cross_task_forum') {
    // NOTE: agentId is intentionally omitted here — see docstring. The
    // cross-task cached prefix must be identical across every agent so
    // cross-agent cache reuse can fire.
    return [
      `Scheduled cross-task discussion phase, generation ${generation}.`,
      'Target page: __cross_task__.',
      'Protocol: call query(task_id="__cross_task__", query=<specific cross-task pattern>, max_records=8) before posting.',
      'Optionally call knowledge(task_id="__cross_task__") if needed.',
      'Post one or two concise cross-task observations with forum_post(task_id="__cross_task__", text=...).',
      'Call forum_signal_done() exactly once when finished.',
    ].join('\n');
  }
  const taskIds = (memory?.forumTaskIds || [])
    .map((taskId) => String(taskId || '').trim())
    .filter(Boolean);
  const taskList = taskIds.length > 0 ? taskIds.join(', ') : '(current assigned task)';
  return [
    `Scheduled per-task discussion phase for ${agentId}, generation ${generation}.`,
    `Task pages to discuss: ${taskList}.`,
    'For each task page, call knowledge(task_id=<task_id>) and query(task_id=<task_id>, query=<specific evidence>, max_records=8) before posting.',
    'Post one concise transferable insight per task with forum_post(task_id=<task_id>, text=...).',
    'Call forum_signal_done() exactly once when finished.',
  ].join('\n');
}

/**
 * Build the round-specific portion of the scheduled-guidance text.
 *
 * Returns the round label and (when round >= 2) the live-board guidance
 * line. This text is appended to the variable suffix block, OUTSIDE the
 * cache_control marker, so it can change between rounds without
 * invalidating the cached prefix.
 *
 * Returns the empty string when there is nothing round-specific to emit
 * (so the caller can omit the suffix block entirely if no other suffix
 * content exists).
 */
function buildScheduledGuidanceSuffix(containerInput: ContainerInput): string {
  const memory = containerInput.memoryMcp;
  const round = memory?.forumRound ?? 0;
  const liveBoardGuidance = round >= 2
    ? 'Because this is a later discussion round, call forum_read() before posting or replying so current-board parent_post_id choices are grounded in the live discussion state.'
    : '';
  return [
    `Round ${round}.`,
    liveBoardGuidance,
  ].filter(Boolean).join('\n');
}

function buildInitialPrompt(prompt: string, containerInput: ContainerInput): string {
  const orchestratorPrompt = prompt.trim();
  const guidancePrefix = buildScheduledGuidancePrefix(containerInput);
  const guidanceSuffix = buildScheduledGuidanceSuffix(containerInput);
  const scheduledGuidance = guidanceSuffix
    ? [guidancePrefix, guidanceSuffix].join('\n')
    : guidancePrefix;
  if (!orchestratorPrompt) {
    return scheduledGuidance;
  }
  return [
    orchestratorPrompt,
    '',
    '## Direct Anthropic scheduled guidance',
    scheduledGuidance,
  ].join('\n');
}

/**
 * Build the cache-stable prefix portion of the initial user message for
 * a forum task. When the Python host supplied a split via
 * `containerInput.forumCacheablePrefix`, this function returns
 * (prefix + scheduled-guidance) and reserves the variable suffix for a
 * second user-message block; the cache_control marker should land on
 * the block containing this returned text. When no split is supplied
 * (legacy callers, non-forum tasks), this falls back to the legacy
 * concatenation that includes the entire prompt body.
 *
 * The scheduled guidance itself is split into a stable
 * (agentId/generation) portion that joins the cached prefix and a
 * round-specific portion (round label + live-board hint) that joins
 * the variable suffix. See `buildScheduledGuidancePrefix` and
 * `buildScheduledGuidanceSuffix`.
 *
 * See cache_stability_over_token_reduction invariant + PR #564
 * regression note + PR #572 follow-up.
 */
function buildInitialPrefixText(
  prompt: string,
  containerInput: ContainerInput,
): { prefix: string; suffix: string } {
  const split = containerInput.forumCacheablePrefix;
  const guidancePrefix = buildScheduledGuidancePrefix(containerInput);
  const guidanceSuffix = buildScheduledGuidanceSuffix(containerInput);
  if (typeof split === 'string' && split.length > 0) {
    // Cached block: stable orchestrator prefix + the agent/generation
    // (round-independent) portion of the scheduled guidance. The round
    // label and round-conditional live-board hint live in the variable
    // suffix block so they can mutate per round without changing the
    // cached prefix hash.
    const prefix = [
      split.trim(),
      '',
      '## Direct Anthropic scheduled guidance',
      guidancePrefix,
    ].join('\n');
    const variableSuffix = (containerInput.forumVariableSuffix || '').trim();
    const suffixParts: string[] = [];
    if (guidanceSuffix) {
      suffixParts.push(guidanceSuffix);
    }
    if (variableSuffix) {
      suffixParts.push(variableSuffix);
    }
    const suffix = suffixParts.join('\n\n');
    return { prefix, suffix };
  }
  // Legacy path: no split available — fall back to the historical
  // behavior of placing the entire prompt body before the scheduled
  // guidance. Mark suffix empty so the caller emits a single block.
  return { prefix: buildInitialPrompt(prompt, containerInput), suffix: '' };
}

async function callMcpTool(
  client: McpStdioClient,
  name: string,
  input: unknown,
): Promise<{ text: string; isError: boolean }> {
  try {
    const args = input && typeof input === 'object' && !Array.isArray(input)
      ? input as Record<string, unknown>
      : {};
    const result = await client.request('tools/call', { name, arguments: args });
    const content = Array.isArray(result.content) ? result.content : [];
    const textParts: string[] = [];
    for (const block of content) {
      if (block && typeof block === 'object') {
        const b = block as Record<string, unknown>;
        if (b.type === 'text' && typeof b.text === 'string') {
          textParts.push(b.text);
        }
      }
    }
    return { text: textParts.join('') || safeStringify(result), isError: false };
  } catch (err) {
    return {
      text: JSON.stringify({ error: err instanceof Error ? err.message : String(err) }),
      isError: true,
    };
  }
}

/**
 * Run a single round of the Anthropic Messages-API forum loop. Mutates
 * ``messages`` in place (appending assistant + tool-result turns) and
 * accumulates per-round token usage into ``perRoundUsage`` and per-round
 * trace into ``perRoundTrace``. Returns when the assistant emits a
 * tool-call-less reply or when ``maxTurns`` is exhausted.
 *
 * Extracted so the shared-container R0->R1 path can call it twice on the
 * same ``messages`` array without duplicating the rolling-cache-control
 * logic.
 */
async function runForumRound(args: {
  messages: AnthropicMessage[];
  cachedSystem: Array<Record<string, unknown>>;
  tools: Array<Record<string, unknown>>;
  selectedModel: string;
  maxTokens: number;
  maxTurns: number;
  sdkEnv: Record<string, string | undefined>;
  mcpClient: McpStdioClient;
  /** Aggregate token bag (R0+R1 combined). Always updated. */
  totalUsage: AnthropicDirectForumResult;
  /** Per-round token bag. Reset by caller before each round. */
  perRoundUsage: CrossTaskRoundUsage;
  /** Per-round trace bag. Reset by caller before each round. */
  perRoundTrace: Array<Record<string, unknown>>;
  /** Aggregate trace. Always appended-to. */
  totalTrace: Array<Record<string, unknown>>;
  /** 0-indexed; only used for logging. */
  roundIndex: number;
}): Promise<{ assistantTextLast: string; signaledDone: boolean }> {
  let lastAssistantText = '';
  let signaledDone = false;
  for (let turn = 1; turn <= args.maxTurns; turn++) {
    // Strip the previous turn's rolling marker (text-block markers are
    // spared so the initial-user cache floor survives) and place the new
    // one on the most recent tool_result block. Three stable cache
    // breakpoints under Anthropic's 4-marker cap: cached system + initial
    // user text block + rolling tool_result.
    clearRollingCacheControl(args.messages);
    if (args.messages.length > 0) {
      const cacheTargetMsg = args.messages[args.messages.length - 1] as {
        role?: string; content?: unknown;
      };
      if (
        cacheTargetMsg.role === 'user'
        && Array.isArray(cacheTargetMsg.content)
        && cacheTargetMsg.content.length > 0
      ) {
        const lastBlock = cacheTargetMsg.content[cacheTargetMsg.content.length - 1] as Record<string, unknown>;
        // Only roll the marker onto a `tool_result` block — its actual
        // intent. NEVER mark a `text` block: on R0 turn 1 the last user
        // block is the initial message's VARIABLE suffix (agent/round
        // guidance; the cross-task agent identity lives here per #1259),
        // and the R1 synthetic turn's block is the per-agent R1 prompt —
        // both are `type:'text'`. Marking either wastes a per-agent cache
        // write that no other agent can cross-read and (because
        // `clearRollingCacheControl` spares text blocks) is never stripped
        // (deep-review #1264 High 1). The shared static prefix (block 0)
        // keeps its construction-time marker, so the cache floor survives.
        if (lastBlock && typeof lastBlock === 'object' && lastBlock.type === 'tool_result') {
          lastBlock.cache_control = { type: 'ephemeral' };
        }
      }
    }

    const response = await createMessage(args.sdkEnv, {
      model: args.selectedModel,
      max_tokens: args.maxTokens,
      system: args.cachedSystem,
      messages: args.messages,
      tools: args.tools,
    }, 'Anthropic direct forum');
    if (response.id) args.totalUsage.newSessionId = response.id;
    // Update both aggregate and per-round usage from the same response.
    accumulateUsage(args.totalUsage, response);
    accumulateUsage(args.perRoundUsage, response);

    const assistantText = textFromBlocks(response.content);
    const traceTs = new Date().toISOString();
    if (assistantText) {
      lastAssistantText = assistantText;
      const entry = {
        type: 'message',
        text: assistantText,
        idx: turn,
        round: args.roundIndex,
        ts: traceTs,
      };
      args.totalTrace.push(entry);
      args.perRoundTrace.push(entry);
    }

    const toolUses = (response.content || [])
      .filter((block) => block.type === 'tool_use' && block.name && block.id);
    if (toolUses.length === 0) {
      // Assistant ended this round with a text-only reply.
      break;
    }

    args.messages.push({ role: 'assistant', content: response.content });
    const toolResults: Array<Record<string, unknown>> = [];
    for (const block of toolUses) {
      const toolName = String(block.name);
      const toolInput = block.input ?? {};
      if (toolName === 'forum_signal_done' || toolName === 'mcp__memory__forum_signal_done') {
        signaledDone = true;
      }
      const output = await callMcpTool(args.mcpClient, toolName, toolInput);
      let outputText = output.text;
      if (outputText.length > TOOL_OUTPUT_MAX_CHARS) {
        outputText = outputText.slice(0, TOOL_OUTPUT_MAX_CHARS);
      }
      const traceEntry = {
        type: 'tool_call',
        tool_name: prefixedToolName(toolName),
        tool_use_id: block.id,
        tool_input: toolInput,
        tool_output: outputText,
        ...(output.isError ? { tool_is_error: true } : {}),
        idx: turn,
        round: args.roundIndex,
        ts: traceTs,
      };
      args.totalTrace.push(traceEntry);
      args.perRoundTrace.push(traceEntry);
      toolResults.push({
        type: 'tool_result',
        tool_use_id: block.id,
        content: [{ type: 'text', text: outputText }],
        ...(output.isError ? { is_error: true } : {}),
      });
    }
    args.messages.push({ role: 'user', content: toolResults });
    // Note: do NOT break on signaledDone here. After the model calls
    // forum_signal_done it typically emits one final text confirmation in
    // the next turn (which becomes lastAssistantText). The natural loop
    // exit on `toolUses.length === 0` above handles termination. Pre-PR
    // #575 behaviour relied on this; tests assert the post-signal_done
    // text turn is captured.
  }
  return { assistantTextLast: lastAssistantText, signaledDone };
}

export async function runAnthropicDirectForumQuery(
  prompt: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
): Promise<AnthropicDirectForumResult> {
  const selectedModel = String(sdkEnv.MODEL || '').trim();
  if (!selectedModel) {
    throw new Error('MODEL env var is required for Anthropic direct forum runs.');
  }
  const maxTurns = Math.max(
    1,
    Number(
      sdkEnv.KCSI_ANTHROPIC_DIRECT_FORUM_MAX_TURNS ||
        sdkEnv.KCSI_CLAUDE_MAX_TURNS ||
        25,
    ),
  );
  const maxTokens = Math.max(256, Number(sdkEnv.KCSI_ANTHROPIC_MAX_TOKENS || 4096));
  const mcpClient = new McpStdioClient(
    'python3',
    ['/app/memory/mcp_server.py'],
    buildMemoryMcpEnv(containerInput, sdkEnv),
  );

  const result: AnthropicDirectForumResult = {
    resultText: '',
    toolTrace: [],
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    tokens_source: 'per_turn_sum',
  };

  // Detect cross-task shared-container mode. Only meaningful for
  // ``cross_task_forum`` task source. The host (engine.py) only sets the
  // flag in that case, but we double-check the task source defensively
  // so a misrouted dispatch doesn't break per-task forum runs.
  const sharedCfg = containerInput.crossTaskSharedContainer;
  const taskSource = String(containerInput.memoryMcp?.taskSource || '').toLowerCase();
  const sharedEnabled = Boolean(
    sharedCfg && sharedCfg.enabled && taskSource === 'cross_task_forum',
  );
  const sharedAgentId = sharedCfg?.agentId || containerInput.memoryMcp?.forumAgentId || 'agent';
  const barrierName = sharedCfg?.barrierName || CROSS_TASK_R1_BARRIER_NAME;
  const responseTimeoutMs = Number(sharedCfg?.responsePollTimeoutMs)
    || CROSS_TASK_R1_POLL_TIMEOUT_MS_DEFAULT;
  const sharedStartedAt = sharedEnabled ? Date.now() : 0;

  try {
    const mcpTools = await mcpClient.initialize();
    const tools = convertMcpToolsForAnthropic(mcpTools);
    log(
      `Using model: ${selectedModel} | maxTurns: ${maxTurns} | shared=${sharedEnabled} `
      + `| tools=${tools.map((t) => t.name).join(',')}`,
    );

    // Block-form initial user message so we can attach `cache_control` to its
    // text block. Combined with the cached system block below, the
    // (system + tools + initial_user_prefix) prefix is cache-written on
    // turn 1 and cache-read on every turn after. Mirrors the ARC adapter's
    // working pattern. See cache_stability_over_token_reduction invariant
    // — compaction was deliberately removed; prompt caching is the sole
    // cost-control mechanism for forum turns.
    //
    // PR #564 (V2 forum) regressed cache_read=0 by mixing per-agent /
    // per-generation content (prior posts, native memory, peer posts)
    // into the cached block. The Python host now ships a cache-stable
    // split via `forumCacheablePrefix` / `forumVariableSuffix`. The
    // cache_control marker sits ONLY on the prefix block; the variable
    // suffix is appended as a second text block in the same user
    // message — same content delivered to the model, but the cache key
    // is taken from the bytes up to and including the marker, so the
    // prefix hash stays stable across agents and generations.
    const { prefix: initialUserPrefix, suffix: initialUserSuffix } =
      buildInitialPrefixText(prompt, containerInput);
    const initialUserBlocks: Array<Record<string, unknown>> = [
      {
        type: 'text' as const,
        text: initialUserPrefix,
        cache_control: { type: 'ephemeral' as const },
      },
    ];
    if (initialUserSuffix) {
      // Variable suffix — no cache_control. Anthropic's cache key is the
      // content hash up to and including the cache_control marker, so
      // mutations after the marker do NOT invalidate the cached prefix.
      initialUserBlocks.push({
        type: 'text' as const,
        text: initialUserSuffix,
      });
    }
    const messages: AnthropicMessage[] = [
      { role: 'user', content: initialUserBlocks as AnthropicMessage['content'] },
    ];
    // Forum runner intentionally does NOT inject .seed_context into a second
    // system block: the cache budget is reserved for the shared stable floor
    // and the rolling tool_result marker. The initial-user suffix is mutable
    // per agent/round and deliberately has no cache_control marker.
    // Forum agents discuss attempts already present in their conversation
    // thread — the per-task seed bundle is the downstream artifact of forum
    // distillation, not an input forum agents need to consume.
    const cachedSystem = [
      { type: 'text' as const, text: buildSystemPrompt(), cache_control: { type: 'ephemeral' as const } },
    ];

    // ── Round 0 ─────────────────────────────────────────────────────────
    const r0Usage: CrossTaskRoundUsage = {
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    };
    const r0Trace: Array<Record<string, unknown>> = [];
    const r0Outcome = await runForumRound({
      messages,
      cachedSystem,
      tools,
      selectedModel,
      maxTokens,
      maxTurns,
      sdkEnv,
      mcpClient,
      totalUsage: result,
      perRoundUsage: r0Usage,
      perRoundTrace: r0Trace,
      totalTrace: result.toolTrace,
      roundIndex: 0,
    });
    let assistantTextLast = r0Outcome.assistantTextLast;
    const r0SignaledDone = r0Outcome.signaledDone || result.toolTrace.some((entry) =>
      String(entry.tool_name || '') === 'mcp__memory__forum_signal_done',
    );

    if (sharedEnabled) {
      // Surface the round-0 result block so the host engine can drain
      // and attribute round-0 tokens distinctly from round-1 tokens.
      result.cross_task_round_0_result = {
        resultText: assistantTextLast,
        toolTrace: r0Trace.slice(),
        tokenUsage: { ...r0Usage },
        signaledDone: r0SignaledDone,
      };
      // Default meta block — captured flips to true if R1 actually ran.
      result.cross_task_shared_container_meta = {
        enabled: true,
        r1_captured: false,
        elapsed_ms: Date.now() - sharedStartedAt,
      };
    }

    // ── Barrier: R0 -> R1 handshake ────────────────────────────────────
    let r1RanToCompletion = false;
    let r1ErrorNote: string | undefined;
    // True only for the graceful-degrade case: this agent's own poll
    // window elapsed waiting for the host's response file. Distinct
    // from every other r1ErrorNote cause (sentinel write failure,
    // missing r1_prompt_text, R1 turn threw), which are genuine errors
    // the host engine must still count as forum-agent failures.
    let r1TimedOut = false;
    if (sharedEnabled) {
      // Write a barrier sentinel describing what R0 produced. The host
      // BarrierWatcher drains the forum bus, computes per-agent R1
      // prompt suffixes (peer posts), and writes the response file.
      let sentinelTarget: string;
      try {
        sentinelTarget = writeBarrierSentinel(
          CONTAINER_WORKSPACE_ROOT,
          barrierName,
          sharedAgentId,
          {
            schema: 'cross_task_r1.v1',
            agent_id: sharedAgentId,
            generation: containerInput.memoryMcp?.forumGeneration ?? 0,
            r0_signaled_done: r0SignaledDone,
            r0_token_usage: r0Usage,
          },
        );
        log(`R0->R1 barrier sentinel written: ${sentinelTarget}`);
      } catch (err) {
        r1ErrorNote = `cross_task_r1: failed to write barrier sentinel: ${
          err instanceof Error ? err.message : String(err)
        }`;
        log(r1ErrorNote);
      }

      let hostPayload: Record<string, unknown> | null = null;
      if (!r1ErrorNote) {
        const responseFile = barrierResponsePath(
          CONTAINER_WORKSPACE_ROOT,
          barrierName,
          sharedAgentId,
        );
        hostPayload = await waitForBarrierFile(responseFile, responseTimeoutMs, {
          pollIntervalMs: CROSS_TASK_BARRIER_POLL_INTERVAL_MS,
        });
        if (!hostPayload) {
          r1ErrorNote = `cross_task_r1: host barrier response not received within ${responseTimeoutMs}ms`;
          r1TimedOut = true;
          log(r1ErrorNote);
        } else {
          log(`R0->R1 barrier response received keys=${Object.keys(hostPayload).join(',')}`);
        }
      }

      if (hostPayload && !r1ErrorNote) {
        // Build a synthetic user turn carrying the R1 prompt suffix
        // (peer posts gathered by the host from drained R0 forum bus).
        // The variable suffix mutates AFTER the cache_control floor on
        // the initial user prefix, so this addition does NOT invalidate
        // the cached prefix — same invariant as the per-agent /
        // per-round suffix in the legacy two-dispatch path.
        const r1PromptText = String(hostPayload.r1_prompt_text || '').trim();
        if (!r1PromptText) {
          r1ErrorNote = 'cross_task_r1: host response missing r1_prompt_text';
          log(r1ErrorNote);
        } else {
          // Before R1 begins, defensively release any cache_control marker
          // on the R0 initial-user NON-prefix (suffix) blocks. Since #1264
          // the rolling logic only marks `tool_result` blocks, so the R0
          // *suffix* text block is no longer pinned on R0 turn 1 — this
          // clear is now a belt-and-suspenders guard (covering any stray
          // marker on a non-prefix block). Were a stale text-block marker
          // ever left in place it would push the R1 marker set past
          // Anthropic's per-request cap of 4 (system + initial-user prefix +
          // stale suffix + R1 prompt + rolling tool_result) and 400 every
          // multi-turn R1 round. The stable prefix floor (block 0) keeps its
          // marker.
          const initialUserContent = messages[0]?.content;
          if (Array.isArray(initialUserContent)) {
            for (let i = 1; i < initialUserContent.length; i++) {
              const block = initialUserContent[i] as Record<string, unknown> | null;
              if (block && typeof block === 'object' && 'cache_control' in block) {
                delete block.cache_control;
              }
            }
          }
          // Synthetic user turn — fresh content, no cache_control marker.
          // The rolling-cache logic in runForumRound only marks
          // `tool_result` blocks (#1264), so this per-agent R1 prompt text
          // stays unmarked (a per-agent write here would never be
          // cross-read); the stable prefix floor (block 0) remains the cache
          // anchor and R1's `tool_result` turns pick up the rolling marker.
          messages.push({
            role: 'user',
            content: [
              {
                type: 'text' as const,
                text: r1PromptText,
              },
            ] as AnthropicMessage['content'],
          });

          // ── Round 1 ─────────────────────────────────────────────────
          const r1Usage: CrossTaskRoundUsage = {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
          };
          const r1Trace: Array<Record<string, unknown>> = [];
          let r1Outcome: { assistantTextLast: string; signaledDone: boolean };
          try {
            r1Outcome = await runForumRound({
              messages,
              cachedSystem,
              tools,
              selectedModel,
              maxTokens,
              maxTurns,
              sdkEnv,
              mcpClient,
              totalUsage: result,
              perRoundUsage: r1Usage,
              perRoundTrace: r1Trace,
              totalTrace: result.toolTrace,
              roundIndex: 1,
            });
          } catch (err) {
            r1ErrorNote = `cross_task_r1: R1 turn threw: ${
              err instanceof Error ? err.message : String(err)
            }`;
            log(r1ErrorNote);
            r1Outcome = { assistantTextLast: '', signaledDone: false };
          }
          if (!r1ErrorNote) {
            assistantTextLast = r1Outcome.assistantTextLast || assistantTextLast;
            r1RanToCompletion = true;
            result.cross_task_round_1_result = {
              resultText: r1Outcome.assistantTextLast,
              toolTrace: r1Trace.slice(),
              tokenUsage: { ...r1Usage },
              signaledDone: r1Outcome.signaledDone,
            };
          }
        }
      }

      result.cross_task_shared_container_meta = {
        enabled: true,
        r1_captured: r1RanToCompletion,
        ...(r1ErrorNote ? { note: r1ErrorNote } : {}),
        ...(r1TimedOut ? { timed_out: true } : {}),
        elapsed_ms: Date.now() - sharedStartedAt,
      };
    }

    // Choose a final result text. For the shared-container path the
    // "round" we report is whichever ran most recently — round-1 if it
    // happened, round-0 otherwise. Empty is acceptable; the resultText
    // is mostly informational since downstream consumes the per-round
    // structures and the forum bus.
    const hasTokens =
      result.input_tokens +
      result.output_tokens +
      result.cache_creation_input_tokens +
      result.cache_read_input_tokens > 0;
    result.tokens_source = hasTokens ? 'per_turn_sum' : 'unavailable';
    result.resultText = assistantTextLast;

    // For the shared-container path the contract is "R0 must have signaled
    // done; R1 absence is OK". For the legacy path we keep the original
    // contract ("must have signaled done at all").
    const finalSignaledDone = sharedEnabled
      ? r0SignaledDone
      : result.toolTrace.some((entry) =>
          String(entry.tool_name || '') === 'mcp__memory__forum_signal_done',
        );
    if (!finalSignaledDone) {
      result.status = 'error';
      result.error = 'Anthropic direct forum run ended without forum_signal_done.';
      result.resultText = '';
    }
    log(
      `Direct forum run complete. status=${result.status || 'success'} `
      + `tools=${result.toolTrace.filter((entry) => entry.type === 'tool_call').length} `
      + `tokens=${result.input_tokens}/${result.output_tokens} shared=${sharedEnabled}`
      + (sharedEnabled ? ` r1_captured=${r1RanToCompletion}` : ''),
    );
    return result;
  } finally {
    mcpClient.close();
  }
}
