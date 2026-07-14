/**
 * Volume-mount construction for the shared container runtime.
 *
 * Builds the bind-mount set for a container launch: the per-workspace scoped
 * root, provider session dir, IPC namespace, model caches, agent-runner source
 * overlay, and the conditional memory MCP mounts. Also owns the
 * runner-root path resolution and the best-effort recursive removal helper.
 */
import fs from 'fs';
import path from 'path';

import { RUNTIME_STATE_DIR } from './config.js';
import {
  resolveGlobalWorkspacePath,
  resolveWorkspaceIpcPath,
  resolveWorkspaceRootPath,
  resolveWorkspaceRunnerOverlayPath,
  resolveWorkspaceSessionPath,
} from './workspace_scope.js';
import { logger } from './logger.js';
import { RegisteredWorkspace } from './container_types.js';
import { CONTAINER_WORKSPACE_ROOT } from './workspace.js';
import { ContainerInput } from './shared_types.js';

export interface VolumeMount {
  hostPath: string;
  containerPath: string;
  readonly: boolean;
}

export function removePathIfExists(targetPath: string): void {
  if (!fs.existsSync(targetPath)) {
    return;
  }
  try {
    fs.rmSync(targetPath, {
      recursive: true,
      force: true,
      maxRetries: 8,
      retryDelay: 50,
    });
    return;
  } catch {
    // Fall through to child-by-child cleanup for occasional ENOTEMPTY/EEXIST races.
  }

  try {
    const stat = fs.lstatSync(targetPath);
    if (!stat.isDirectory()) {
      fs.rmSync(targetPath, { force: true });
      return;
    }
    for (const entry of fs.readdirSync(targetPath)) {
      removePathIfExists(path.join(targetPath, entry));
    }
    fs.rmdirSync(targetPath);
  } catch {
    // best-effort
  }
}

export function resolveRunnerRoot(): string {
  const raw = (process.env.KSI_RUNNER_ROOT || '/app').trim();
  if (!raw.startsWith('/') || raw === '/') {
    return '/app';
  }
  return raw.replace(/\/+$/, '') || '/app';
}

export function runnerPath(child: string): string {
  return path.posix.join(resolveRunnerRoot(), child);
}

function resolveTaskRepoContainerPath(): string {
  const raw = (process.env.KSI_TASK_REPO_CONTAINER_PATH || '').trim();
  if (!raw.startsWith('/') || raw === '/' || raw.includes(':')) {
    return '';
  }
  return raw.replace(/\/+$/, '');
}

export function buildVolumeMounts(
  workspaceRuntime: RegisteredWorkspace,
): VolumeMount[] {
  const mounts: VolumeMount[] = [];
  const projectRoot = process.cwd();
  const workspaceRoot = resolveWorkspaceRootPath(workspaceRuntime.folder);

  // Each task/agent runtime only gets its own scoped workspace root
  mounts.push({
    hostPath: workspaceRoot,
    containerPath: CONTAINER_WORKSPACE_ROOT,
    readonly: false,
  });

  // Global memory directory (read-only)
  // Only directory mounts are supported, not file mounts
  const globalDir = resolveGlobalWorkspacePath();
  if (fs.existsSync(globalDir)) {
    mounts.push({
      hostPath: globalDir,
      containerPath: '/workspace/global',
      readonly: true,
    });
  }

  const isTaskScopedWorkspace = workspaceRuntime.folder.startsWith('task__');

  // Per-workspace provider session directory.
  // Each task/agent workspace gets its own .claude/ to prevent cross-workspace session access.
  const groupSessionRoot = resolveWorkspaceSessionPath(workspaceRuntime.folder);
  if (isTaskScopedWorkspace) {
    removePathIfExists(groupSessionRoot);
  }
  const groupSessionsDir = path.join(groupSessionRoot, '.claude');
  fs.mkdirSync(groupSessionsDir, { recursive: true });
  const settingsFile = path.join(groupSessionsDir, 'settings.json');
  const enableAgentTeams = !process.env.KSI_DISABLE_AGENT_TEAMS;
  fs.writeFileSync(
    settingsFile,
    JSON.stringify(
      {
        env: {
          // Explicitly pin agent-team behavior so scheduled benchmark runs do
          // not inherit stale defaults from prior local setups.
          // https://code.claude.com/docs/en/agent-teams#orchestrate-teams-of-claude-code-sessions
          CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: enableAgentTeams ? '1' : '0',
          // Load CLAUDE.md from additional mounted directories
          // https://code.claude.com/docs/en/memory#load-memory-from-additional-directories
          CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: '1',
          // Enable Claude's memory feature (persists user preferences between sessions)
          // https://code.claude.com/docs/en/memory#manage-auto-memory
          CLAUDE_CODE_DISABLE_AUTO_MEMORY: '0',
        },
      },
      null,
      2,
    ) + '\n',
  );

  // Sync skills from container/skills/ into each group's .claude/skills/
  const skillsSrc = path.join(process.cwd(), 'container', 'skills');
  const skillsDst = path.join(groupSessionsDir, 'skills');
  if (fs.existsSync(skillsSrc)) {
    for (const skillDir of fs.readdirSync(skillsSrc)) {
      const srcDir = path.join(skillsSrc, skillDir);
      if (!fs.statSync(srcDir).isDirectory()) continue;
      const dstDir = path.join(skillsDst, skillDir);
      // verbatimSymlinks: preserve relative symlinks as-is. Node's default
      // resolves them against the source absolute path and writes host-only
      // absolute targets, which break inside the container (see CLAUDE.md
      // gotcha + PR #560 — this rewrote SWE-bench-Pro repo symlinks and made
      // every affected task score 0).
      fs.cpSync(srcDir, dstDir, { recursive: true, verbatimSymlinks: true });
    }
  }
  mounts.push({
    hostPath: groupSessionsDir,
    containerPath: '/home/node/.claude',
    readonly: false,
  });

  // Per-workspace IPC namespace.
  // This prevents cross-workspace privilege escalation via IPC.
  const groupIpcDir = resolveWorkspaceIpcPath(workspaceRuntime.folder);
  if (isTaskScopedWorkspace) {
    removePathIfExists(groupIpcDir);
  }
  fs.mkdirSync(path.join(groupIpcDir, 'input'), { recursive: true });
  mounts.push({
    hostPath: groupIpcDir,
    containerPath: '/workspace/ipc',
    readonly: false,
  });

  // Shared model cache for local embedding search. Pre-warmed host-side
  // (#923 M2: the orchestrator populates runtime_state/model_cache before any
  // container launches), then mounted READ-ONLY so a container cannot poison
  // the shared model weights for other/future containers. The container loads
  // the model offline (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE in container_args.ts)
  // so a warm read-only cache loads without lock-file or network writes.
  const huggingFaceCacheDir = path.join(RUNTIME_STATE_DIR, 'model_cache', 'huggingface');
  const sentenceTransformersCacheDir = path.join(
    RUNTIME_STATE_DIR,
    'model_cache',
    'sentence-transformers',
  );
  fs.mkdirSync(huggingFaceCacheDir, { recursive: true });
  fs.mkdirSync(sentenceTransformersCacheDir, { recursive: true });
  mounts.push(
    {
      hostPath: huggingFaceCacheDir,
      containerPath: '/home/node/.cache/huggingface',
      readonly: true,
    },
    {
      hostPath: sentenceTransformersCacheDir,
      containerPath: '/home/node/.cache/sentence-transformers',
      readonly: true,
    },
  );

  // Copy agent-runner source into a per-workspace writable location so agents
  // can customize it (add tools, change behavior) without affecting other
  // workspaces. Recompiled on container startup via entrypoint.sh.
  const agentRunnerSrcCandidates = [
    path.join(projectRoot, 'runtime_runner', 'agent-runner', 'src'),
  ];
  const agentRunnerSrc = agentRunnerSrcCandidates.find((p) => fs.existsSync(p));
  const groupAgentRunnerDir = path.join(
    resolveWorkspaceRunnerOverlayPath(workspaceRuntime.folder),
    'agent-runner-src',
  );
  if (agentRunnerSrc) {
    fs.mkdirSync(path.dirname(groupAgentRunnerDir), { recursive: true });
    removePathIfExists(groupAgentRunnerDir);
    // verbatimSymlinks: see the skills copy above. Without it, any relative
    // symlink in agent-runner/src is rewritten to a host-absolute path and the
    // in-container `tsc` step fails before any task work begins.
    fs.cpSync(agentRunnerSrc, groupAgentRunnerDir, { recursive: true, verbatimSymlinks: true });
  }
  mounts.push({
    hostPath: groupAgentRunnerDir,
    containerPath: runnerPath('src'),
    readonly: false,
  });

  // Keep the container's agent-runner dependencies aligned with the mounted
  // KSI-owned source tree. Without this, the image-baked /app/node_modules
  // can drift from runtime_runner/agent-runner imports and fail compilation
  // before the SDK starts.
  //
  // Only mount a host node_modules directory that actually contains the
  // agent-runner SDK — using the host-side runtime_runner node_modules
  // (which declares only pino/tsx) would shadow the image's baked SDKs and
  // crash every container with TS2307 errors. We check for a sentinel package
  // (@anthropic-ai/claude-agent-sdk) to confirm the candidate is the right
  // one.
  const agentRunnerSentinel = path.join(
    '@anthropic-ai',
    'claude-agent-sdk',
    'package.json',
  );
  const agentRunnerNodeModulesCandidates = [
    path.join(projectRoot, 'container', 'agent-runner', 'node_modules'),
    path.join(projectRoot, 'runtime_runner', 'agent-runner', 'node_modules'),
  ];
  const agentRunnerNodeModules = agentRunnerNodeModulesCandidates.find(
    (p) => fs.existsSync(path.join(p, agentRunnerSentinel)),
  );
  if (agentRunnerNodeModules) {
    mounts.push({
      hostPath: agentRunnerNodeModules,
      containerPath: runnerPath('node_modules'),
      readonly: true,
    });
  }

  const taskRepoContainerPath = isTaskScopedWorkspace
    ? resolveTaskRepoContainerPath()
    : '';
  if (taskRepoContainerPath && taskRepoContainerPath !== resolveRunnerRoot()) {
    const taskRepoHostPath = path.join(workspaceRoot, 'workspace', 'repo');
    if (fs.existsSync(taskRepoHostPath)) {
      mounts.push({
        hostPath: taskRepoHostPath,
        containerPath: taskRepoContainerPath,
        readonly: false,
      });
    }
  }

  return mounts;
}

/**
 * Bind-mount a sqlite DB's main file plus whichever WAL/SHM sidecars
 * currently exist on the host, as individual files under `/app/memory-db`
 * (issue #1009 — defense-in-depth narrowing of the previous whole-directory
 * mount).
 *
 * Both `MemoryStore` and `KnowledgeStore` (src/ksi/memory/) run
 * `PRAGMA journal_mode=WAL`, so a DB under active use grows `-wal`/`-shm`
 * sidecars next to the main file for as long as a writer holds it open. A
 * sidecar that doesn't exist yet is skipped rather than mounted: Docker
 * silently creates an empty DIRECTORY at a bind-mount source path that
 * doesn't exist, which would present the wrong file type to sqlite inside
 * the container instead of erroring loudly.
 *
 * Verified empirically (see PR description) that a reader opening the main
 * file read-only via a bind-mounted `-wal` with NO `-shm` mounted at all
 * still sees the latest WAL-only (uncommitted-to-main-file) rows correctly
 * — sqlite falls back to creating its own local `-shm` in the container's
 * already-writable `/app/memory-db` directory (chmod 777 at image build
 * time) rather than failing.
 */
function addSqliteFileMounts(
  mounts: VolumeMount[],
  dbPath: string,
  readonly: boolean,
  workspaceRuntime: RegisteredWorkspace,
  label: string,
): void {
  const resolved = path.resolve(dbPath);
  for (const suffix of ['', '-wal', '-shm']) {
    const hostPath = `${resolved}${suffix}`;
    if (!fs.existsSync(hostPath)) {
      if (suffix === '') {
        logger.warn(
          { group: workspaceRuntime.name, dbPath: hostPath },
          `${label} file does not exist — it will not be mounted`,
        );
      }
      continue;
    }
    mounts.push({
      hostPath,
      containerPath: `/app/memory-db/${path.basename(hostPath)}`,
      readonly,
    });
  }
}

/**
 * Append the conditional memory MCP mounts to an already-built mount set.
 *
 * Mounts the knowledge DB directory and MCP server code when agent-facing
 * memory tools are configured. (ARC no longer mounts an MCP server or snapshot
 * — it runs natively via workspace attempt files.)
 */
export function appendMemoryAndArcMounts(
  mounts: VolumeMount[],
  input: ContainerInput,
  workspaceRuntime: RegisteredWorkspace,
): void {
  // Mount the knowledge DB directory and MCP server code when agent-facing
  // memory tools are configured.
  if (input.memoryMcp) {
    if (!path.isAbsolute(input.memoryMcp.dbPath)) {
      logger.warn(
        { group: workspaceRuntime.name, dbPath: input.memoryMcp.dbPath },
        'memoryMcp.dbPath is not absolute — skipping memory mounts to avoid resolving against wrong cwd',
      );
    } else {
      const dbDir = path.dirname(input.memoryMcp.dbPath);
      const taskSource = String(input.memoryMcp.taskSource || '').toLowerCase();
      // Every forum phase (per-task, cross-task) needs write access to
      // /app/memory-db so ForumBus.append can persist posts and
      // forum_signal_done can INSERT. Previously the RO mount made cross-task
      // posts silently vanish.
      const forumWritesNeeded =
        taskSource === 'cross_task_forum'
        || taskSource === 'per_task_forum';
      if (fs.existsSync(dbDir)) {
        // At-rest redaction invariant: the raw knowledge/runtime-audit sqlite DBs
        // store hidden-test tails (`test_stdout_tail`/`test_stderr_tail`) UNREDACTED
        // at rest — MCP-side redaction is read-time only. A TASK (execute) agent
        // with native Bash/Read (`--arc-no-mcp`) could otherwise
        // `grep -a test_stdout_tail /app/memory-db/*.sqlite` and lift hidden
        // assertions straight off the mounted file, bypassing the MCP redactor.
        //
        // Task agents do NOT need the raw DB file: the `task` MCP toolset
        // exposes ZERO tools (query results are pre-injected into MEMORY.md via
        // the already-redacted snapshot), and the `arc` toolset reads the
        // answer-stripped snapshot — neither reads the knowledge DB. Only forum
        // agents (`cross_task_forum`/`per_task_forum`) genuinely need the DB
        // mounted: they write posts / query via MCP (with read-time redaction).
        // So mount the raw DB files ONLY for forum containers; task agents get
        // just the redacted snapshot (mounted unconditionally below).
        //
        // #1009 defense-in-depth: even for forum, bind-mount only the specific
        // files needed (knowledge + runtime-audit sqlite DBs and their WAL/SHM
        // sidecars) rather than the whole per-experiment directory, so an agent
        // can't enumerate sibling files that land in the same subdirectory.
        if (forumWritesNeeded) {
          // This branch only runs for forum containers, which need the DB
          // mounted read-write (ForumBus.append / forum_signal_done INSERT),
          // so the mounts are unconditionally read-write here.
          addSqliteFileMounts(mounts, input.memoryMcp.dbPath, false, workspaceRuntime, 'Knowledge DB');
          if (input.memoryMcp.runtimeDbPath) {
            addSqliteFileMounts(
              mounts,
              input.memoryMcp.runtimeDbPath,
              false,
              workspaceRuntime,
              'Runtime-audit DB',
            );
          }
        }
        if (input.memoryMcp.snapshotPath) {
          // query_config.ts sets MEMORY_SNAPSHOT_PATH to /app/memory-db/<basename>.
          const resolvedSnapshotPath = path.resolve(input.memoryMcp.snapshotPath);
          if (fs.existsSync(resolvedSnapshotPath)) {
            mounts.push({
              hostPath: resolvedSnapshotPath,
              containerPath: `/app/memory-db/${path.basename(resolvedSnapshotPath)}`,
              readonly: true,
            });
          } else {
            logger.warn(
              { group: workspaceRuntime.name, snapshotPath: resolvedSnapshotPath },
              'Memory snapshot file does not exist — snapshot-backed memory MCP will not be available',
            );
          }
        }
        // ForumBus (src/ksi/memory/forum_bus.py) is a file-backed bus with
        // dynamically-named, per-generation JSONL files under
        // <dbDir>/forum_bus/ — genuinely directory-shaped, multi-writer
        // state, not a fixed filename we can allowlist like the DBs above.
        // Only mount it for task sources whose MCP_TOOLSET actually exposes
        // forum tools (query_config.ts's `forumPhases` set, matching
        // `forumWritesNeeded` below) — every other toolset ("task", "arc")
        // returns zero tools from mcp_server.py's _build_tools(), including
        // no forum_read, so mounting forum_bus/ for them has no legitimate
        // functional use and would only expose raw, unfiltered forum JSONL
        // (bypassing the exclude_task_ids hold-out filter that forum_read
        // enforces at the MCP layer) to containers with native Bash/Read.
        if (forumWritesNeeded) {
          // Pre-create it (mirroring ForumBus.__init__'s own mkdir) so Docker
          // never auto-vivifies an empty directory over a not-yet-existing
          // path, and mount it explicitly so a read-only mount here still lets
          // ForumBus correctly detect "can't write" via OSError — without this
          // mount, /app/memory-db is a normal (chmod 777) container-local
          // directory, so ForumBus's mkdir/touch calls would silently succeed
          // against a copy that's invisible to the host, instead of no-op'ing
          // with the "bus is read-only" log line it currently emits.
          const busDir = path.join(dbDir, 'forum_bus');
          try {
            fs.mkdirSync(busDir, { recursive: true });
            mounts.push({
              hostPath: busDir,
              containerPath: '/app/memory-db/forum_bus',
              // Always read-write: this branch only runs when
              // forumWritesNeeded is true.
              readonly: false,
            });
          } catch (err) {
            logger.warn(
              {
                group: workspaceRuntime.name,
                busDir,
                err: err instanceof Error ? err.message : String(err),
              },
              'Could not create forum_bus directory for mounting — forum posts may not persist',
            );
          }
        }
      } else {
        logger.warn(
          { group: workspaceRuntime.name, dbDir },
          'Knowledge DB directory does not exist — memory MCP will not be available',
        );
      }
      if (fs.existsSync(input.memoryMcp.serverDir)) {
        mounts.push({
          hostPath: input.memoryMcp.serverDir,
          containerPath: '/app/memory',
          readonly: true,
        });
      } else {
        logger.warn(
          { group: workspaceRuntime.name, serverDir: input.memoryMcp.serverDir },
          'Memory MCP server directory does not exist — memory MCP will not be available',
        );
      }
      // (No separate /app/memory-snapshot mount here: that legacy
      // whole-directory mount was removed — nothing container-side reads it
      // any more, since the snapshot file is already delivered above at
      // /app/memory-db/<basename>, and the whole-dir mount defeated the
      // per-file narrowing this function otherwise does.)
    }
  }

  // NB: there is no ARC MCP server mount anymore. ARC runs natively for every
  // provider (the agent reads payload.json from the workspace and writes
  // attempt files), so it neither spawns /app/memory/mcp_server.py nor reads a
  // snapshot from /app/memory-db. The legacy `--no-memory` ARC MCP mount block
  // was removed with the ARC MCP server registration.
}
