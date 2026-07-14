/**
 * Container-side paths and IPC constants shared across the agent-runner
 * modules. Centralized so the host-coupled literals (workspace root, IPC
 * mailbox) have a single definition; module-local constants that only one
 * file needs stay in that file.
 */
import path from 'path';

/** In-container workspace root the SDK runs against (mounted by the host). */
export const CONTAINER_WORKSPACE_ROOT = '/workspace/task';

/** Where PreCompact archives land. Derived from the workspace root. */
export const CONTAINER_CONVERSATIONS_DIR = `${CONTAINER_WORKSPACE_ROOT}/conversations`;

/**
 * In-container path where the claude-agent-sdk writes per-session JSONL
 * transcripts. The runtime mounts the per-workspace `.claude/` directory
 * here from the host (see container_runner.ts — mount groupSessionsDir →
 * /home/node/.claude). Each session's turns land under `projects/<slug>/`.
 */
export const CONTAINER_CLAUDE_SESSIONS_ROOT = '/home/node/.claude/projects';

/** Host→container IPC mailbox: pending message JSON files are dropped here. */
export const IPC_INPUT_DIR = '/workspace/ipc/input';

/** Best-effort shutdown sentinel the host writes after the first output marker. */
export const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');

/** Poll interval (ms) for IPC drains and barrier-file waits. */
export const IPC_POLL_MS = 500;
