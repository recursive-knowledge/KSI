/**
 * Per-query configuration for the scheduled-task Claude Agent SDK loop:
 * turn/message budgets, the allowed/disallowed tool policy, and the memory
 * MCP server config. Extracted from the main runner; behavior is unchanged.
 * (ARC no longer registers an MCP server — it runs natively via attempt
 * files.)
 */
import fs from 'fs';
import path from 'path';
import type { McpServerConfig } from '@anthropic-ai/claude-agent-sdk';
import { ContainerInput } from './shared_types.js';
import { buildWebToolGating } from './web_tools.js';
import { log } from './runner_log.js';

export const NATIVE_FILE_SHELL_TOOLS = [
  'Bash',
  'Read',
  'Write',
  'Edit',
  'Glob',
  'Grep',
  'TodoWrite',
  'NotebookEdit',
  // Subagent-spawning tools: a forum agent under the claude_code preset runs
  // with permissionMode 'bypassPermissions', and a spawned subagent does not
  // inherit the parent's disallowedTools (Claude Agent SDK #172/#189) — so an
  // undenied `Task` re-opens the raw-DB exfil path (issue #1115). Deny the
  // spawn on the main loop, matching phase1_reflection.ts's canonical list.
  'Task',
  'TaskOutput',
  'TaskStop',
];

export function resolveClaudeMaxTurns(
  // Kept in the signature for API stability with existing call sites; the
  // unified 150-cap policy below ignores it. Override via env if needed.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _taskSource?: string,
  envOverride?: string,
): number {
  const override = Number(envOverride);
  if (Number.isFinite(override) && override > 0) {
    return Math.floor(override);
  }
  // Universal default: every scheduled task gets 150 turns to match the OpenAI
  // agents-sdk path. Prior split (25 for ARC, 80 for polyglot/swebench_pro,
  // inherited from original SDK defaults) was tighter than production needs —
  // Haiku × SWE-bench Pro routinely hit 68-72 tool calls leaving only ~10%
  // headroom, and the no-MCP ARC variant reliably hit the 25 cap mid-analysis.
  // 150 gives every code path the same headroom and removes turn budget as a
  // confound; under MCP tools an ARC task still finishes in ~5 turns.
  // Override via KSI_CLAUDE_MAX_TURNS.
  return 150;
}

/**
 * Turn / message budgets for a scheduled (bench) task. Extracted from
 * runQuery for readability; the computation is unchanged.
 */
export function resolveTurnBudgets(
  taskSource: string,
  sdkEnv: Record<string, string | undefined>,
): { scheduledMaxTurns: number | undefined; scheduledMaxMessages: number | undefined } {
  const scheduledMaxTurns = resolveClaudeMaxTurns(taskSource, sdkEnv.KSI_CLAUDE_MAX_TURNS);
  // ARC used to get a lower 80-message ceiling here, set back when its
  // maxTurns cap was still 25 (so 80 messages was never the binding
  // constraint). When maxTurns was raised to 150 above (no-MCP ARC sessions
  // reliably hit the old 25-turn cap mid-analysis), this ceiling was left
  // behind and became the new, tighter bottleneck: native/no-MCP ARC
  // sessions doing Bash-based exploration hit 80 raw messages (~25-27 tool
  // round-trips) well before 150 turns, forcing a fallback stop before the
  // agent ever writes its prediction file — reproducing the exact failure
  // the turn-cap fix was meant to eliminate (#1037). per_task_forum had the
  // same shape of mismatch (60-message ceiling against the same 150-turn
  // cap, #1049) — match every task source.
  const defaultMaxMessages = 150;
  // Guard the env override the same way resolveClaudeMaxTurns does: a
  // non-numeric/empty/non-positive KSI_CLAUDE_MAX_MESSAGES falls back to the
  // default instead of yielding NaN (which would make the >= message ceiling
  // never fire and silently disable the cap).
  const messagesOverride = Number(sdkEnv.KSI_CLAUDE_MAX_MESSAGES);
  const scheduledMaxMessages = Math.max(
    1,
    Number.isFinite(messagesOverride) && messagesOverride > 0
      ? messagesOverride
      : defaultMaxMessages,
  );
  return { scheduledMaxTurns, scheduledMaxMessages };
}

export interface ToolPolicy {
  isArcWithoutMcp: boolean;
  isScheduledMcpProtocolTask: boolean;
  allowedToolsList: string[];
  disallowedToolsList: string[];
}

const NATIVE_TOOL_LEAK_SECRET_ENV_VARS = [
  'OPENAI_API_KEY',
  'ANTHROPIC_API_KEY',
  'CLAUDE_CODE_OAUTH_TOKEN',
  'CLAUDE_CODE_OAUTH_REFRESH_TOKEN',
  'CLAUDE_CODE_OAUTH_SCOPES',
  'HF_TOKEN',
  'HUGGING_FACE_HUB_TOKEN',
];

function truthyEnvFlag(value: string | undefined): boolean {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase());
}

export function sdkEnvHasNativeToolLeakSecret(sdkEnv: Record<string, string | undefined>): boolean {
  return NATIVE_TOOL_LEAK_SECRET_ENV_VARS.some((name) => Boolean(sdkEnv[name]));
}

export function egressIsolationDisabled(sdkEnv: Record<string, string | undefined>): boolean {
  // Mirrors the host-side check in runtime_runner/src/container_args.ts. Only
  // meaningful when the host forwards KSI_EGRESS into the container env.
  return String(sdkEnv.KSI_EGRESS || '').trim().toLowerCase() === 'open';
}

export function shouldDenyNativeToolsForSdkSecrets(sdkEnv: Record<string, string | undefined>): boolean {
  // Native file/shell tools are the ONLY way coding agents (polyglot,
  // swebench_pro, terminal_bench_2) edit files and run tests, and the provider
  // secret is ALWAYS present in sdkEnv — index.ts merges containerInput.secrets
  // so the SDK can authenticate. A blanket secret-presence denial therefore
  // strips every coding task's tools on every run (0 solves). Two controls
  // already contain the leak on the default isolated path: the Bash PreToolUse
  // hook unsets every SECRET_ENV_VAR before each command (hooks.ts), and egress
  // isolation (#923) blocks exfiltration of anything a tool does read (e.g.
  // /proc/self/environ via Read). The residual read-then-exfiltrate path only
  // exists when egress isolation is disabled (KSI_EGRESS=open — debugging
  // only, never production), so scope the denial to that mode, where breaking
  // native tools is an acceptable cost, instead of firing on every run.
  return (
    sdkEnvHasNativeToolLeakSecret(sdkEnv)
    && egressIsolationDisabled(sdkEnv)
    && !truthyEnvFlag(sdkEnv.KSI_ALLOW_UNSAFE_CLAUDE_NATIVE_TOOLS_WITH_SECRETS)
  );
}

/**
 * Compute the allowed/disallowed tool lists and the offline/MCP-protocol
 * flags for a query. ARC is offline (no web tools); scheduled ARC/forum jobs
 * are MCP-protocol-only (no native file/shell tools) unless ARC runs without
 * its MCP server. Extracted from runQuery; behavior is unchanged. The
 * returned `allowedToolsList` is later appended to by buildMcpServerConfig
 * (mcp__memory__*).
 */
export function buildToolPolicy(
  containerInput: ContainerInput,
  taskSource: string,
  isForumTask: boolean,
  sdkEnv: Record<string, string | undefined>,
): ToolPolicy {
  const isOffline = taskSource === 'arc';
  // Web tools (WebSearch/WebFetch) are a benchmark-solution leak vector
  // (issue #666): default OFF for ALL benchmark tasks (an operator opts in
  // per-run with KSI_ALLOW_WEB_TOOLS=1); ARC stays strictly offline
  // regardless of the flag. Default-OFF is the new baseline and creates a
  // code-era boundary: results recorded before this fix had web tools
  // available on non-ARC Claude runs (see benchmarks/docs/web_tools_policy.md). The
  // gating decision lives in ./web_tools (SDK-free, behaviorally testable —
  // see tests/js/web_tools_gating.test.mjs).
  const webToolGating = buildWebToolGating(sdkEnv, isOffline);
  // ARC always runs natively now (no ARC MCP server), so every ARC task gets
  // the native file/shell tools rather than the MCP-protocol-only surface.
  const isArcWithoutMcp = taskSource === 'arc';
  const isScheduledMcpProtocolTask = isOffline || isForumTask;
  const isMcpProtocolOnlyTask = isScheduledMcpProtocolTask && !isArcWithoutMcp;
  const denyNativeToolsForSecrets = shouldDenyNativeToolsForSdkSecrets(sdkEnv);
  const allowedToolsList: string[] =
    isMcpProtocolOnlyTask
      ? []
      : denyNativeToolsForSecrets
        ? [...webToolGating.allowlistWebTools]
        : [
            'Bash',
            'Read', 'Write', 'Edit', 'Glob', 'Grep',
            ...webToolGating.allowlistWebTools,
            'TodoWrite',
            'NotebookEdit',
          ];
  // CRITICAL: disallowedTools is what actually DENIES preset tools — the
  // claude_code preset loads tools into context, so omitting them from
  // allowedTools is not enough. This carries web-tool denials whenever web
  // tools are not enabled (issue #666) and native file/shell denials for
  // scheduled MCP-protocol tasks such as forum jobs (issue #1115).
  // disallowedTools is a documented SDK Options field; runtime_runner/
  // agent-runner pins @anthropic-ai/claude-agent-sdk at ^0.1.0, where its
  // location in the typings varies by SDK version — trust the JSDoc, not a
  // specific file path.
  const protocolNativeToolDenials = isMcpProtocolOnlyTask ? [...NATIVE_FILE_SHELL_TOOLS] : [];
  const secretNativeToolDenials = denyNativeToolsForSecrets ? [...NATIVE_FILE_SHELL_TOOLS] : [];
  const disallowedToolsList: string[] = [
    ...new Set([
      ...webToolGating.disallowedWebTools,
      ...protocolNativeToolDenials,
      ...secretNativeToolDenials,
    ]),
  ];
  log(
    `Web tools (WebSearch/WebFetch): ${webToolGating.webToolsEnabled ? 'ENABLED' : 'DISABLED'} `
    + `[${webToolGating.reason}]`,
  );
  if (denyNativeToolsForSecrets) {
    log(
      'Native Claude file/shell tools: DISABLED '
      + '[SDK env contains credentials; set KSI_ALLOW_UNSAFE_CLAUDE_NATIVE_TOOLS_WITH_SECRETS=1 to override]',
    );
  }
  return { isArcWithoutMcp, isScheduledMcpProtocolTask, allowedToolsList, disallowedToolsList };
}

/**
 * Build the memory MCP server config map and push the corresponding
 * wildcard/allowlist tools onto `allowedToolsList` in place. Extracted from
 * runQuery; the server wiring and registration guard are unchanged. Mutates
 * `allowedToolsList` (preserving the original push site) and returns the
 * assembled config. (ARC no longer registers an MCP server.)
 */
export function buildMcpServerConfig(
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  taskSource: string,
  allowedToolsList: string[],
): Record<string, McpServerConfig> {
  const mcpServerConfig: Record<string, McpServerConfig> = {};

  // Register memory MCP server when config is present and server file exists.
  if (containerInput.memoryMcp && fs.existsSync('/app/memory/mcp_server.py')) {
    const dbFile = path.basename(containerInput.memoryMcp.dbPath);
    const snapshotFile = containerInput.memoryMcp.snapshotPath
      ? path.basename(containerInput.memoryMcp.snapshotPath)
      : '';
    // Cross-task forum containers also need the forum toolset (forum_post,
    // forum_signal_done, knowledge) — previously only the per-task forum
    // received them, so cross-task agents posted nothing.
    const forumPhases = new Set([
      'cross_task_forum',
      'per_task_forum',
    ]);
    const memoryToolset = forumPhases.has(taskSource) ? 'forum' : 'task';
    mcpServerConfig.memory = {
      command: 'python3',
      args: ['/app/memory/mcp_server.py'],
      env: {
        KNOWLEDGE_DB_PATH: `/app/memory-db/${dbFile}`,
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
      },
    };
    allowedToolsList.push('mcp__memory__*');
    log('Memory MCP server registered');
  }

  // ARC no longer registers an MCP server: it runs natively for every provider
  // (the agent reads payload.json and writes attempt files, and the host
  // synthesizes the arc_submit_trial trace post-exit). The legacy ARC MCP
  // registration was removed alongside the direct-ARC adapters.

  return mcpServerConfig;
}
