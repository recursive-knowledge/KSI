/**
 * Canonical container I/O type definitions for the KCSI runtime.
 *
 * These interfaces define the contract between the host container runner
 * (container_runner.ts) and the in-container agent runners (index.ts, openai.ts).
 *
 * This file is kept in sync across:
 *   - runtime_runner/src/shared_types.ts           (host-side, used by container_runner.ts)
 *   - runtime_runner/agent-runner/src/shared_types.ts  (container-side, used by index.ts, openai.ts)
 *
 * These are separate TypeScript compilation units (host vs container), so the
 * file exists in both locations. Any change must be applied to both copies.
 */

export interface MemoryMcpConfig {
  dbPath: string;
  serverDir: string;
  snapshotPath?: string;
  /**
   * Host-side path to the optional runtime-audit sqlite DB (`--runtime-db-path`).
   * Threaded through so container_mounts.ts can bind-mount it (and its WAL/SHM
   * sidecars) individually alongside dbPath instead of relying on a whole-
   * directory mount (issue #1009).
   */
  runtimeDbPath?: string;
  taskId?: string;
  taskSource?: string;
  forumGeneration?: number;
  forumRound?: number;
  forumAgentId?: string;
  forumExpectedAgents?: number;
  forumTaskIds?: string[];
  experiment?: string;
}

/**
 * Carries ARC task metadata (source/id/snapshot) for the native ARC path.
 * Deliberately independent of MemoryMcpConfig so that `--no-memory` (disabled
 * knowledge MCP) does not silently strip this metadata — ARC and memory are
 * conceptually separate concerns. There is no ARC MCP server anymore; ARC
 * always runs natively (the agent reads payload.json and writes attempt files).
 */
export interface ArcToolsConfig {
  /** Retained but always false — no ARC MCP server exists to register. */
  enable: boolean;
  /** Host-side directory containing mcp_server.py to mount as /app/memory. */
  mcpServerDir: string;
  /** Task source string (echoed from payload so guards work without memoryMcp). */
  taskSource?: string;
  /** Current ARC task id, used by native fallbacks when the model omits args. */
  taskId?: string;
  /**
   * Host-side path to an ARC task snapshot JSON, carrying the train/test
   * payload for the native path. The snapshot shape is
   * `{arc_payload_by_task: {task_id: payload}}`.
   */
  snapshotPath?: string;
}

/**
 * Phase-1 reflection feature flag block. When `enabled` is true, the
 * scheduled-task path in agent-runner/src/index.ts wakes a host-side
 * BarrierWatcher after the SDK emits its terminal `result` event, hands
 * the watcher the eval result, and then runs ONE additional `query()`
 * turn in the same SDK session asking the agent for a 3-5 sentence
 * structured reflection. The captured assistant text is shipped back as
 * `ContainerOutput.phase1_reflection` and threaded into runtime_meta.
 *
 * Behind a flag because (a) it costs an extra SDK turn (and any cache
 * miss it implies), (b) it adds wall-clock latency between the eval
 * and the next phase, and (c) we want to dial it up gradually after
 * smoke-testing.
 */
export interface Phase1ReflectionConfig {
  enabled: boolean;
  /** Logical agent id used to namespace the barrier sentinel/response pair. */
  agentId: string;
  /** Upper bound (ms) on how long we wait for the host to respond with the eval. */
  evalResultPollTimeoutMs?: number;
}

/**
 * Polyglot test-feedback retry loop feature flag block ("Aider protocol").
 * When `enabled` is true, the scheduled-task path in query_runner.ts writes a
 * barrier sentinel with the agent's live model output, waits for the host to
 * run the real PolyglotHarnessEvaluator and respond with the eval result, and
 * — on failure — runs one resumed SDK turn (tools re-enabled) per remaining
 * try, feeding the agent its own capped test-runner output. Unlike
 * Phase1ReflectionConfig, a round here CAN change what gets graded: the last
 * round's on-disk edits become the new `result`.
 */
export interface PolyglotTestFeedbackConfig {
  enabled: boolean;
  /** Logical agent id used to namespace the barrier sentinel/response pair. */
  agentId: string;
  /** Remaining submission attempts, Aider `--tries`-style (includes the one already spent). */
  triesRemaining: number;
  /** Cap applied to the tail of the test-runner output shown back to the agent. */
  maxLines: number;
  /** Human-readable list of the task's solution file(s), echoed into the retry prompt. */
  fileList: string;
  /** Upper bound (ms) on how long we wait for the host to respond with the eval. */
  evalResultPollTimeoutMs?: number;
  /** Tools re-enabled for the retry round's resumed SDK turn. */
  allowedTools: string[];
  /** MCP server config for the retry round's resumed SDK turn. */
  mcpServers: Record<string, unknown>;
  /** Max SDK turns allowed per retry round. */
  maxTurnsPerRound: number;
}

/**
 * Cross-task shared-container feature-flag block. When ``enabled`` is true,
 * the cross-task forum agent stays in the SAME Anthropic-Messages-API
 * conversation across forum rounds 0 and 1. After the agent emits
 * ``forum_signal_done`` for round 0, the container writes a barrier
 * sentinel with the round-0 token usage, waits for the host's
 * BarrierWatcher to compute and write back this agent's round-1 prompt
 * (peer posts from THIS-gen R0 + any other variable suffix), then continues
 * the same ``messages[]`` loop with a new synthetic user turn carrying the
 * R1 prompt content.
 *
 * Behind a flag because (a) it changes the dispatch shape (one container
 * per agent for both rounds instead of two), (b) the host has to drain the
 * forum bus between R0 and R1 mid-flight, and (c) the round-1 token usage
 * is bookkept under a new ``cross_task_forum_round_1`` phase slug rather
 * than the existing one. Default off.
 */
export interface CrossTaskSharedContainerConfig {
  enabled: boolean;
  /** Logical agent id used to namespace the barrier sentinel/response pair. */
  agentId: string;
  /** The barrier name. Always ``"cross_task_r1"`` in production; configurable for tests. */
  barrierName?: string;
  /**
   * Upper bound (ms) on how long the in-container R0->R1 wait blocks before
   * giving up and emitting an R0-only envelope. Mirrors the Phase 1
   * reflection's ``evalResultPollTimeoutMs`` semantics.
   */
  responsePollTimeoutMs?: number;
}

/**
 * Token usage accumulated on a single cross-task forum round. Surfaced
 * separately from the (R0+R1 combined) top-level usage so the host can
 * record per-round rows in ``token_phases`` (slugs:
 * ``cross_task_forum_round_0`` and ``cross_task_forum_round_1``).
 */
export interface CrossTaskRoundUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

/**
 * Per-round result block emitted when the cross-task shared-container
 * feature flag is on. ``resultText`` is the assistant's last text reply
 * for that round (typically empty since round-N ends with a tool-use
 * turn, but kept for symmetry). ``toolTrace`` is the per-round subset of
 * the trace and ``tokenUsage`` is the per-round token aggregate.
 *
 * Always optional in ``ContainerOutput`` — absent on legacy single-round
 * dispatches. ``cross_task_round_1_result`` is also absent when the host
 * barrier timed out and we degraded gracefully to R0-only.
 */
export interface CrossTaskRoundResult {
  resultText: string;
  toolTrace: Array<Record<string, unknown>>;
  tokenUsage: CrossTaskRoundUsage;
  signaledDone: boolean;
}

export interface ContainerInput {
  prompt: string;
  sessionId?: string;
  workspaceKey: string;
  assistantName?: string;
  secrets?: Record<string, string>;
  /**
   * Host-generated nonce for authenticating container stdout envelopes.
   * The nonce is passed over stdin and is never included in the model prompt.
   */
  protocolNonce?: string;
  /** Memory MCP server config -- when set, mounts DB + server into container. */
  memoryMcp?: MemoryMcpConfig;
  /** ARC task metadata for the native path — independent of memoryMcp. */
  arcTools?: ArcToolsConfig;
  /**
   * Cache-stable split of the forum prompt body, set by the Python host
   * for forum-phase tasks (per_task_forum / cross_task_forum). The
   * direct forum adapter places `cache_control` on a block containing
   * only `forumCacheablePrefix` and appends `forumVariableSuffix` as a
   * separate block. Adapters that don't support a split delivery (the
   * file-using runner that surfaces the body via TASK.md) ignore these
   * fields and fall back to ``prompt``.
   */
  forumCacheablePrefix?: string;
  forumVariableSuffix?: string;
  /** Phase-1 self-reflection feature flag block. Optional; absent = disabled. */
  phase1Reflection?: Phase1ReflectionConfig;
  /** Polyglot test-feedback retry loop config. Optional; absent = disabled (non-polyglot tasks). */
  polyglotTestFeedback?: PolyglotTestFeedbackConfig;
  /**
   * Cross-task shared-container feature flag block. Optional; absent =
   * disabled (legacy two-dispatch round 0 + round 1 path).
   *
   * Only meaningful when ``memoryMcp.taskSource === 'cross_task_forum'``;
   * the direct forum adapter is the only consumer.
   */
  crossTaskSharedContainer?: CrossTaskSharedContainerConfig;
}

export interface ContainerOutput {
  status: 'success' | 'error' | 'recovered_from_session';
  result: string | null;
  newSessionId?: string;
  error?: string;
  toolTrace?: Array<Record<string, unknown>>;
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
  /** Echoes ContainerInput.protocolNonce so host parsing can reject forged stdout markers. */
  protocolNonce?: string;
  /**
   * Provenance of the emitted token counts. Set by the agent-runner so
   * downstream consumers (Python host, analysis scripts) can distinguish:
   *   - 'result_event'    : pulled from the SDK's final `result` message usage
   *                         (most sessions)
   *   - 'per_turn_sum'    : stream ended without a result event (e.g. message
   *                         ceiling hit); tokens summed from per-turn
   *                         assistant `message.usage` deltas
   *   - 'session_recovery': SDK iterator drained without yielding events but
   *                         the in-container session-log JSONL held real turns;
   *                         tokens summed from those turns and output extracted
   *                         from the last assistant message. Paired with
   *                         `status='recovered_from_session'`.
   *   - 'unavailable'     : neither source had data — the reported zeros are
   *                         a reporting gap, NOT a truly zero session
   * Absent on legacy outputs (interactive session updates). Optional.
   */
  tokens_source?: 'result_event' | 'per_turn_sum' | 'session_recovery' | 'unavailable';
  /**
   * Only set when `status='recovered_from_session'`. Explains the recovery
   * path so downstream consumers can distinguish a real success from one
   * that was reconstructed from the session log after a silent-exit bug.
   */
  recovery_note?: string;
  /**
   * Phase-1 self-reflection text captured by an in-session follow-up
   * `query()` turn after the host returned the eval result via the barrier
   * protocol. Present iff the feature flag was on AND the host responded
   * within the timeout AND the SDK follow-up did not silent-exit.
   *
   * The string is the assistant's verbatim 3-5 sentence reflection
   * (load-bearing assumption + proposed change + predicted outcome).
   * Always optional — missing/empty MUST not fail the attempt.
   */
  phase1_reflection?: string;
  /**
   * Diagnostic shape for phase-1 reflection. Lets the host distinguish
   * "feature off", "host eval timed out", "follow-up SDK turn silent-exited",
   * and "captured" cases. `note` carries an English-language reason when
   * `captured` is false. Always optional.
   */
  phase1_reflection_meta?: {
    enabled: boolean;
    captured: boolean;
    note?: string;
    elapsed_ms?: number;
  };
  /**
   * Token usage observed on the reflection turn's `result` event (or
   * summed per-turn deltas as a fallback). Surfaced separately from the
   * scheduled-task usage block so the host can record a dedicated
   * ``phase1_reflection`` row in ``token_phases``. Without this field
   * the reflection-turn tokens silently vanish from cost reports.
   */
  phase1_reflection_token_usage?: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens: number;
    cache_read_input_tokens: number;
  };
  /**
   * Polyglot test-feedback retry-loop diagnostics. Present iff the task was
   * polyglot and `triesRemaining > 1`. Unlike phase1_reflection, this
   * feature's rounds DO change `result` (the graded output) when a retry ran.
   */
  polyglot_test_feedback_meta?: {
    enabled: boolean;
    rounds_used: number;
    attempt_1_eval_summary: Record<string, unknown> | null;
    captured: boolean;
    note?: string;
    // True only when the last barrier round's evaluation genuinely
    // reflects the final graded state (no agent turn ran after it) --
    // see polyglot_test_feedback.ts's finalEvalMatchesOutput. Host-side
    // reuse of the watcher's cached eval must gate on this.
    final_eval_matches_output?: boolean;
  };
  polyglot_test_feedback_token_usage?: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens: number;
    cache_read_input_tokens: number;
  };
  /**
   * Round-0 result block, emitted by the cross-task shared-container path
   * (direct forum adapter) when ``crossTaskSharedContainer.enabled`` is
   * true. The legacy single-dispatch path leaves this absent.
   */
  cross_task_round_0_result?: CrossTaskRoundResult;
  /**
   * Round-1 result block. Emitted only when (a) the shared-container feature
   * flag is on AND (b) the host barrier responded with the R1 prompt within
   * the configured timeout AND (c) the agent ran R1 to a terminal state
   * (signaled done OR maxTurns hit). Absent when the host timed out or
   * the R1 turn threw — the host treats absence as "R1 was skipped",
   * NOT a failure.
   */
  cross_task_round_1_result?: CrossTaskRoundResult;
  /**
   * Diagnostic for the shared-container path. ``captured`` reports whether
   * R1 was actually run; ``note`` carries an English reason when R1 was
   * skipped (timeout, exception). ``timed_out`` is true only for the
   * graceful-degrade case — the host barrier never produced a response
   * within this agent's own poll window — and is absent/false for every
   * other skip reason (sentinel write failure, missing r1_prompt_text, the
   * R1 turn throwing), which are genuine errors. Always optional — only
   * present when ``crossTaskSharedContainer.enabled`` is true.
   */
  cross_task_shared_container_meta?: {
    enabled: boolean;
    r1_captured: boolean;
    note?: string;
    timed_out?: boolean;
    elapsed_ms?: number;
  };
}
