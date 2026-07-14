/**
 * Pure, dependency-free helpers for deciding which NATIVE (non-MCP) tools an
 * OpenAI container run should expose. Kept free of the `@openai/agents` SDK
 * import so it can be unit-tested directly under CI's `node --test tests/js`
 * (which installs only the host-side runtime_runner deps).
 *
 * Trust boundary (issue #1221): forum/MCP-protocol phases (`per_task_forum`,
 * `cross_task_forum`) are meant to be MCP-mediated ONLY — a forum agent talks
 * to the shared discussion/knowledge substrate exclusively through the memory
 * MCP server. Handing those runs native `shell` / `apply_patch` (and the
 * flag-gated parity filesystem tools) lets a forum prompt/tool call shell out
 * and read/write the raw knowledge/runtime SQLite DBs mounted into the
 * container, bypassing the MCP layer. The Anthropic direct-forum adapter
 * already enforces this ("Native file and shell tools are unavailable in this
 * direct scheduled adapter."); this brings the OpenAI path to parity.
 */

export const OPENAI_FORUM_PHASES: ReadonlySet<string> = new Set([
  'cross_task_forum',
  'per_task_forum',
]);

/** True for forum/MCP-protocol phases that must be MCP-only. */
export function isOpenAIForumPhase(taskSource: string | undefined | null): boolean {
  return OPENAI_FORUM_PHASES.has((taskSource || '').toLowerCase());
}

/**
 * Select the native tool set for an OpenAI run.
 *
 * - Forum phases (`isForumTask`): no native tools at all — MCP-only.
 * - Benchmark task phases (arc/polyglot/swebench/tb2): keep shell + apply_patch,
 *   plus the flag-gated parity filesystem tools when parity is enabled.
 *
 * Generic over the tool object type so it can be exercised with lightweight
 * fakes in tests without importing the `@openai/agents` `tool()` factory.
 */
export function selectOpenAINativeTools<T>(opts: {
  isForumTask: boolean;
  parityEnabled: boolean;
  shellFnTool: T;
  applyPatchFnTool: T;
  parityFsTools: T[];
}): T[] {
  if (opts.isForumTask) {
    return [];
  }
  return opts.parityEnabled
    ? [opts.shellFnTool, opts.applyPatchFnTool, ...opts.parityFsTools]
    : [opts.shellFnTool, opts.applyPatchFnTool];
}
