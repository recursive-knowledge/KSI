export type RuntimeScope = 'task' | 'agent';

export interface RuntimeConfig {
  session_scope?: RuntimeScope;
  wipe_workspace_per_task?: boolean;
  container_image?: string;
  official_container_image?: string;
  runner_image?: string;
  repo_container_path?: string;
  official_repo_container_path?: string;
  runner_root?: string;
}

export interface WorkspaceSeed {
  instruction_md?: string;
  memory_md?: string;
  task_md?: string;
  tools_md?: string;
  task_files?: Record<string, string>;
  repo_source_path?: string;
}

export interface TaskPayload {
  id: string;
  repo?: string;
  prompt?: string;
  metadata?: Record<string, unknown>;
}

export interface KsiPayload {
  generation: number;
  agent_id: string;
  task: TaskPayload;
  workspace_seed?: WorkspaceSeed;
  execution_prompt?: string;
  runtime?: RuntimeConfig;
  experiment_name?: string;
  knowledge?: {
    db_path: string;
    mcp_server_dir: string;
    snapshot_path?: string;
    disable_memory_tools?: boolean;
    forum_generation?: number;
    experiment_name?: string;
  };
  runtime_audit?: {
    db_path: string;
  };
  /**
   * ARC task metadata for the native path — hoisted out of `memory` so that
   * running with `--no-memory` does not silently strip it. There is no ARC
   * MCP server anymore; ARC always runs natively.
   */
  arc_tools?: {
    enable: boolean;
    mcp_server_dir: string;
    task_source?: string;
    task_id?: string;
    /**
     * Absolute path to an ARC snapshot JSON written by the Python host.
     * Used to carry the ARC train/test payload into the container under
     * `--no-memory` (where agent-facing memory tools are disabled).
     */
    snapshot_path?: string;
  };
  /**
   * ARC native-mode flag, set by the Python host for ARC tasks. ARC always
   * runs natively (there is no ARC MCP server): native Claude Code tools are
   * on the allowlist, and the in-container agent submits by writing attempt
   * files, which `main.ts` synthesizes into `arc_submit_trial` trace entries
   * after the SDK returns.
   */
  arc_no_mcp?: boolean;
  /**
   * Phase-1 self-reflection feature flag block. When `enabled` is true
   * the in-container agent runs a barrier-protocol round-trip with the
   * host after the task succeeds and captures a structured reflection.
   * Optional; absent = disabled.
   */
  phase1_reflection?: {
    enabled: boolean;
    eval_result_poll_timeout_ms?: number;
  };
  /**
   * Cross-task shared-container feature flag block. When `enabled` is true,
   * the in-container forum agent runs both round 0 and round 1 in the
   * same Anthropic-Messages-API session, coordinating the R0->R1
   * transition via the host barrier protocol. Optional; absent = disabled
   * (legacy two-dispatch path).
   */
  cross_task_shared_container?: {
    enabled: boolean;
    barrier_name?: string;
    response_poll_timeout_ms?: number;
  };
  /**
   * Polyglot test-feedback retry loop config. Set by the Python host
   * (`container_host.py::_maybe_setup_polyglot_test_feedback`) for polyglot
   * tasks whose `polyglot_test_feedback_tries` > 1. The keys already match
   * `PolyglotTestFeedbackConfig` (shared_types.ts) camelCase-for-camelCase,
   * so this is a near-passthrough rather than a field-by-field remap.
   * Optional; absent = disabled.
   */
  polyglot_test_feedback?: {
    enabled: boolean;
    agentId: string;
    triesRemaining: number;
    maxLines: number;
    fileList: string;
    allowedTools: string[];
    mcpServers: Record<string, unknown>;
    maxTurnsPerRound: number;
    evalResultPollTimeoutMs?: number;
  };
}

export interface SessionState {
  session_id: string;
}
