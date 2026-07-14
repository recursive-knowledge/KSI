import path from 'path';

// Absolute paths needed for container mounts
const PROJECT_ROOT = process.cwd();

export const WORKSPACES_DIR = path.resolve(PROJECT_ROOT, 'workspaces');
export const TASK_WORKSPACES_DIR = path.join(WORKSPACES_DIR, 'tasks');
export const AGENT_WORKSPACES_DIR = path.join(WORKSPACES_DIR, 'agents');
export const GLOBAL_WORKSPACE_DIR = path.join(WORKSPACES_DIR, 'global');
export const RUNTIME_STATE_DIR = path.resolve(PROJECT_ROOT, 'runtime_state');
export const RUNTIME_IPC_DIR = path.join(RUNTIME_STATE_DIR, 'ipc');
export const RUNTIME_PROVIDER_SESSIONS_DIR = path.join(
  RUNTIME_STATE_DIR,
  'provider_sessions',
);
export const RUNTIME_RUNNER_OVERLAYS_DIR = path.join(
  RUNTIME_STATE_DIR,
  'runner_overlays',
);
export const RUNTIME_SESSION_STATE_DIR = path.join(
  RUNTIME_STATE_DIR,
  'session_state',
);
export const CONTAINER_IMAGE =
  process.env.KSI_CONTAINER_IMAGE ||
  process.env.CONTAINER_IMAGE ||
  'ksi-agent:bench';
export const CONTAINER_TIMEOUT = parseInt(
  process.env.KSI_CONTAINER_TIMEOUT || process.env.CONTAINER_TIMEOUT || '1800000',
  10,
); // <= 0 disables the hard container deadline; positive values are milliseconds.
export const CONTAINER_MAX_OUTPUT_SIZE = parseInt(
  process.env.KSI_CONTAINER_MAX_OUTPUT_SIZE ||
    process.env.CONTAINER_MAX_OUTPUT_SIZE ||
    '10485760',
  10,
); // 10MB default
export const IDLE_TIMEOUT = parseInt(
  process.env.KSI_IDLE_TIMEOUT || process.env.IDLE_TIMEOUT || '1800000',
  10,
); // 30min default — how long to keep container alive after last result

// Timezone for scheduled tasks
// Uses system timezone by default
export const TIMEZONE =
  process.env.TZ || Intl.DateTimeFormat().resolvedOptions().timeZone;
