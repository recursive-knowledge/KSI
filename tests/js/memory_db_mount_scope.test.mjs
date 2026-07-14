/**
 * Issue #1009 — narrow the /app/memory-db mount from the whole per-experiment
 * directory down to the specific files a container needs (knowledge DB,
 * optional runtime-audit DB, their WAL/SHM sidecars, the memory snapshot,
 * and the forum_bus/ subdirectory for forum-write phases only), so an agent
 * with native Bash/Read (--arc-no-mcp) can't enumerate or read sibling files
 * that happen to land in the same subdirectory. Also covers the follow-up
 * fix that removed the legacy whole-directory /app/memory-snapshot mount,
 * which had otherwise defeated this narrowing whenever a memory snapshot
 * was present.
 *
 * Exercises the REAL appendMemoryAndArcMounts() export from
 * runtime_runner/src/container_mounts.ts via tsx (not a source-text copy),
 * against real fixture files/directories on disk, so a regression that
 * silently reverts to a whole-directory mount (or drops WAL/SHM/forum_bus
 * handling) is caught.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const containerMountsTs = path.join(repoRoot, 'runtime_runner', 'src', 'container_mounts.ts');
const tsxAvailable = fs.existsSync(tsxBin);

// Call the real appendMemoryAndArcMounts() inside tsx with a controlled
// ContainerInput + a temp per-experiment directory standing in for the real
// knowledge-db subdirectory, and return the resulting VolumeMount[].
function evalMounts(input) {
  const script = `
import { appendMemoryAndArcMounts } from ${JSON.stringify(containerMountsTs)};
const mounts = [];
appendMemoryAndArcMounts(mounts, ${JSON.stringify(input)}, { name: 'test-group', folder: 'task__x', trigger: '@Kcsi', added_at: new Date().toISOString() });
process.stdout.write(JSON.stringify(mounts));
`;
  return spawnSync(tsxBin, ['--eval', script, '--conditions=node'], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
}

function makeExperimentDir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

describe('memory-db mount scoping (issue #1009)', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/');
    return;
  }

  it('does NOT mount the raw knowledge DB file for a non-forum (task/execute) agent — only the redacted snapshot (information-parity: raw *_tail is unredacted at rest)', () => {
    // Raw knowledge/runtime-audit sqlite DBs hold hidden-test tails
    // (test_stdout_tail / test_stderr_tail) UNREDACTED at rest; MCP redaction
    // is read-time only. A task agent with native Bash/Read must not be able to
    // `grep -a test_stdout_tail /app/memory-db/*.sqlite`, so the raw DB file
    // must not be bind-mounted into a non-forum container. The `task` MCP
    // toolset exposes zero tools (results pre-injected via the redacted
    // snapshot), so nothing legitimate breaks.
    for (const taskSource of ['arc', 'polyglot', 'swebench_pro', 'terminal_bench_2', '']) {
      const dbDir = makeExperimentDir('kcsi-memdb-task-');
      const dbPath = path.join(dbDir, 'knowledge.sqlite');
      const runtimeDbPath = path.join(dbDir, 'runtime.sqlite');
      const snapshotPath = path.join(dbDir, 'snapshot.json');
      fs.writeFileSync(dbPath, '');
      fs.writeFileSync(`${dbPath}-wal`, '');
      fs.writeFileSync(runtimeDbPath, '');
      fs.writeFileSync(snapshotPath, '{}');
      const serverDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-memdb-server-'));

      const res = evalMounts({
        prompt: 'p',
        workspaceKey: 'w',
        memoryMcp: { dbPath, serverDir, runtimeDbPath, snapshotPath, taskSource },
      });
      assert.equal(res.status, 0, `tsx failed for ${JSON.stringify(taskSource)}: stderr=${res.stderr}\nstdout=${res.stdout}`);
      const mounts = JSON.parse(res.stdout);

      // No raw knowledge DB (nor its WAL/SHM sidecars) mounted.
      assert.ok(
        !mounts.some((m) => m.containerPath === '/app/memory-db/knowledge.sqlite'),
        `raw knowledge DB must NOT be mounted for taskSource=${JSON.stringify(taskSource)}: ${JSON.stringify(mounts)}`,
      );
      assert.ok(
        !mounts.some((m) => m.containerPath.startsWith('/app/memory-db/knowledge.sqlite-')),
        `knowledge DB WAL/SHM sidecars must NOT be mounted for taskSource=${JSON.stringify(taskSource)}: ${JSON.stringify(mounts)}`,
      );
      // No raw runtime-audit DB mounted either (also unredacted at rest).
      assert.ok(
        !mounts.some((m) => m.containerPath.includes('runtime.sqlite')),
        `raw runtime-audit DB must NOT be mounted for taskSource=${JSON.stringify(taskSource)}: ${JSON.stringify(mounts)}`,
      );
      // The redacted snapshot IS still delivered (task memory channel).
      assert.ok(
        mounts.some((m) => m.containerPath === '/app/memory-db/snapshot.json' && m.readonly === true),
        `redacted snapshot must still be mounted for taskSource=${JSON.stringify(taskSource)}: ${JSON.stringify(mounts)}`,
      );
      // Never the whole per-experiment directory.
      assert.ok(
        !mounts.some((m) => m.containerPath === '/app/memory-db' && m.hostPath === dbDir),
        `whole per-experiment directory should not be mounted: ${JSON.stringify(mounts)}`,
      );
    }
  });

  it('mounts WAL/SHM sidecars read-only when they exist on the host (forum container)', () => {
    const dbDir = makeExperimentDir('kcsi-memdb-wal-');
    const dbPath = path.join(dbDir, 'knowledge.sqlite');
    fs.writeFileSync(dbPath, '');
    fs.writeFileSync(`${dbPath}-wal`, '');
    fs.writeFileSync(`${dbPath}-shm`, '');
    const serverDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-memdb-server-'));

    const res = evalMounts({
      prompt: 'p',
      workspaceKey: 'w',
      memoryMcp: { dbPath, serverDir, taskSource: 'per_task_forum' },
    });
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const mounts = JSON.parse(res.stdout);

    assert.ok(mounts.some((m) => m.containerPath === '/app/memory-db/knowledge.sqlite-wal'));
    assert.ok(mounts.some((m) => m.containerPath === '/app/memory-db/knowledge.sqlite-shm'));
  });

  it('does not mount a WAL/SHM sidecar that does not exist on the host (avoids Docker auto-vivifying a directory) (forum container)', () => {
    const dbDir = makeExperimentDir('kcsi-memdb-nowal-');
    const dbPath = path.join(dbDir, 'knowledge.sqlite');
    fs.writeFileSync(dbPath, '');
    // Deliberately no -wal / -shm files.
    const serverDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-memdb-server-'));

    const res = evalMounts({
      prompt: 'p',
      workspaceKey: 'w',
      memoryMcp: { dbPath, serverDir, taskSource: 'per_task_forum' },
    });
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const mounts = JSON.parse(res.stdout);

    assert.ok(!mounts.some((m) => m.containerPath.endsWith('.sqlite-wal')));
    assert.ok(!mounts.some((m) => m.containerPath.endsWith('.sqlite-shm')));
  });

  it('mounts the knowledge DB and forum_bus/ read-write for a forum-write task source', () => {
    for (const taskSource of ['per_task_forum', 'cross_task_forum']) {
      const dbDir = makeExperimentDir('kcsi-memdb-forum-');
      const dbPath = path.join(dbDir, 'knowledge.sqlite');
      fs.writeFileSync(dbPath, '');
      const serverDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-memdb-server-'));

      const res = evalMounts({
        prompt: 'p',
        workspaceKey: 'w',
        memoryMcp: { dbPath, serverDir, taskSource },
      });
      assert.equal(res.status, 0, `tsx failed for ${taskSource}: stderr=${res.stderr}`);
      const mounts = JSON.parse(res.stdout);

      const dbMount = mounts.find((m) => m.containerPath === '/app/memory-db/knowledge.sqlite');
      assert.ok(dbMount, `missing db mount for ${taskSource}`);
      assert.equal(dbMount.readonly, false, `${taskSource} must mount the DB read-write`);

      const busMount = mounts.find((m) => m.containerPath === '/app/memory-db/forum_bus');
      assert.ok(busMount, `missing forum_bus mount for ${taskSource}: ${JSON.stringify(mounts)}`);
      assert.equal(busMount.hostPath, path.join(dbDir, 'forum_bus'));
      assert.equal(busMount.readonly, false, `${taskSource} must mount forum_bus read-write`);
      assert.ok(fs.existsSync(busMount.hostPath), 'forum_bus dir must be pre-created on the host');
    }
  });

  it('does NOT mount forum_bus/ for a non-forum task (its MCP_TOOLSET exposes zero forum tools, so raw access would only bypass the exclude_task_ids hold-out filter)', () => {
    for (const taskSource of ['arc', 'polyglot', 'swebench_pro', '']) {
      const dbDir = makeExperimentDir('kcsi-memdb-forum-ro-');
      const dbPath = path.join(dbDir, 'knowledge.sqlite');
      fs.writeFileSync(dbPath, '');
      const serverDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-memdb-server-'));

      const res = evalMounts({
        prompt: 'p',
        workspaceKey: 'w',
        memoryMcp: { dbPath, serverDir, taskSource },
      });
      assert.equal(res.status, 0, `tsx failed for taskSource=${taskSource}: stderr=${res.stderr}`);
      const mounts = JSON.parse(res.stdout);

      assert.ok(
        !mounts.some((m) => m.containerPath === '/app/memory-db/forum_bus'),
        `forum_bus must not be mounted for taskSource=${JSON.stringify(taskSource)}: ${JSON.stringify(mounts)}`,
      );
      assert.ok(
        !fs.existsSync(path.join(dbDir, 'forum_bus')),
        `forum_bus dir must not even be pre-created on the host for taskSource=${JSON.stringify(taskSource)}`,
      );
    }
  });

  it('mounts the optional runtime-audit DB (+ sidecars) individually when runtimeDbPath is set', () => {
    const dbDir = makeExperimentDir('kcsi-memdb-runtime-');
    const dbPath = path.join(dbDir, 'knowledge.sqlite');
    const runtimeDbPath = path.join(dbDir, 'runtime.sqlite');
    fs.writeFileSync(dbPath, '');
    fs.writeFileSync(runtimeDbPath, '');
    fs.writeFileSync(`${runtimeDbPath}-wal`, '');
    const serverDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-memdb-server-'));

    const res = evalMounts({
      prompt: 'p',
      workspaceKey: 'w',
      memoryMcp: { dbPath, serverDir, runtimeDbPath, taskSource: 'per_task_forum' },
    });
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const mounts = JSON.parse(res.stdout);

    assert.ok(mounts.some((m) => m.containerPath === '/app/memory-db/runtime.sqlite'));
    assert.ok(mounts.some((m) => m.containerPath === '/app/memory-db/runtime.sqlite-wal'));
    assert.ok(!mounts.some((m) => m.containerPath === '/app/memory-db/runtime.sqlite-shm'));
  });

  it('does not mount a runtime DB at all when runtimeDbPath is absent', () => {
    const dbDir = makeExperimentDir('kcsi-memdb-noruntime-');
    const dbPath = path.join(dbDir, 'knowledge.sqlite');
    fs.writeFileSync(dbPath, '');
    const serverDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-memdb-server-'));

    const res = evalMounts({
      prompt: 'p',
      workspaceKey: 'w',
      memoryMcp: { dbPath, serverDir, taskSource: 'arc' },
    });
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const mounts = JSON.parse(res.stdout);
    assert.ok(!mounts.some((m) => m.containerPath.includes('runtime.sqlite')));
  });

  it('mounts the memory snapshot file individually at /app/memory-db/<basename>', () => {
    const dbDir = makeExperimentDir('kcsi-memdb-snap-');
    const dbPath = path.join(dbDir, 'knowledge.sqlite');
    const snapshotPath = path.join(dbDir, 'snapshot.json');
    fs.writeFileSync(dbPath, '');
    fs.writeFileSync(snapshotPath, '{}');
    const serverDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-memdb-server-'));

    const res = evalMounts({
      prompt: 'p',
      workspaceKey: 'w',
      memoryMcp: { dbPath, serverDir, snapshotPath, taskSource: 'arc' },
    });
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const mounts = JSON.parse(res.stdout);
    const snapMount = mounts.find((m) => m.containerPath === '/app/memory-db/snapshot.json');
    assert.ok(snapMount, `missing snapshot mount: ${JSON.stringify(mounts)}`);
    assert.equal(snapMount.readonly, true);

    // The legacy whole-directory /app/memory-snapshot mount must be gone —
    // it defeated the per-file narrowing above (nothing container-side reads
    // /app/memory-snapshot any more; MEMORY_SNAPSHOT_PATH always points at
    // /app/memory-db/<basename>, delivered above).
    assert.ok(
      !mounts.some((m) => m.containerPath === '/app/memory-snapshot'),
      `legacy whole-directory /app/memory-snapshot mount must not exist: ${JSON.stringify(mounts)}`,
    );
  });

  it('does NOT mount an ARC MCP server dir or snapshot (ARC runs natively now)', () => {
    // ARC no longer registers an MCP server: the agent reads payload.json from
    // the workspace and writes attempt files, so appendMemoryAndArcMounts must
    // emit no /app/memory server-dir mount and no /app/memory-db snapshot mount
    // even when arcTools is present on the input.
    const snapDir = makeExperimentDir('kcsi-memdb-arcsnap-');
    const snapshotPath = path.join(snapDir, 'arc_snapshot.json');
    fs.writeFileSync(snapshotPath, '{}');
    const arcServerDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kcsi-arc-server-'));

    const res = evalMounts({
      prompt: 'p',
      workspaceKey: 'w',
      arcTools: { enable: true, mcpServerDir: arcServerDir, taskSource: 'arc', taskId: 't1', snapshotPath },
    });
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const mounts = JSON.parse(res.stdout);
    assert.ok(
      !mounts.some((m) => m.containerPath === '/app/memory-db/arc_snapshot.json'),
      `ARC snapshot must not be mounted (native ARC): ${JSON.stringify(mounts)}`,
    );
    assert.ok(
      !mounts.some((m) => m.hostPath === arcServerDir || m.containerPath === '/app/memory'),
      `ARC MCP server dir must not be mounted (native ARC): ${JSON.stringify(mounts)}`,
    );
  });
});
