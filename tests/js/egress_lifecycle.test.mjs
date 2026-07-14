import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
// The egress lifecycle (network creation, proxy sidecar, teardown) lives in
// container_args.ts alongside buildContainerArgs and the docker-network helpers;
// the caller (runContainerAgent) lives in container_runner.ts and resolves the
// infra once via ensureEgressInfra(). Pin both surfaces.
const args = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'src', 'container_args.ts'),
  'utf-8',
);
const runner = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'src', 'container_runner.ts'),
  'utf-8',
);

const CONTAINER_ARGS_MOD = path.join(repoRoot, 'runtime_runner', 'src', 'container_args.ts');

// `buildContainerArgs` is a pure function (mounts + egress descriptor in, a
// `docker run` argv array out) with no docker/network dependency for the
// default (bridge) open-mode network, so it's genuinely callable here without
// gating on a live daemon — unlike ensureEgressInfra/teardownEgressInfra below
// (real network create + proxy sidecar container, left source-pinned).
function buildContainerArgs(egress, env = {}) {
  const src = `
    import { buildContainerArgs } from ${JSON.stringify(CONTAINER_ARGS_MOD)};
    const args = buildContainerArgs([], 'test-container', ${JSON.stringify(egress)});
    console.log(JSON.stringify(args));
  `;
  const res = spawnSync('npx', ['tsx', '--eval', src], {
    cwd: repoRoot,
    encoding: 'utf-8',
    env: { ...process.env, ...env },
  });
  assert.equal(res.status, 0, res.stderr);
  return JSON.parse(res.stdout.trim());
}

function runContainerArgsSnippet(source, env = {}) {
  const res = spawnSync('npx', ['tsx', '--eval', source], {
    cwd: repoRoot,
    encoding: 'utf-8',
    env: { ...process.env, ...env },
  });
  assert.equal(res.status, 0, res.stderr);
  return JSON.parse(res.stdout.trim());
}

function envPairs(argv) {
  const out = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '-e') out.push(argv[i + 1]);
  }
  return out;
}

describe('buildContainerArgs egress behavior (real invocation, not source-pin)', () => {
  // Issue #979 MEDIUM: the source-pin describe block below verifies string
  // literals, not behavior — a change that preserves the literals but breaks
  // the conditional (e.g. moving --dns into the wrong branch) would still
  // pass it. These call the REAL function and inspect the actual returned
  // docker argv.
  const egress = { internalNet: 'ksi-egress-net', proxyAlias: 'ksi-proxy', proxyPort: 3128 };

  it('isolated mode: attaches to the internal network and blackholes DNS (issue #934)', () => {
    const argv = buildContainerArgs(egress);
    assert.equal(argv[argv.indexOf('--network') + 1], 'ksi-egress-net');
    const dnsIdx = argv.indexOf('--dns');
    assert.ok(dnsIdx >= 0, 'expected a --dns flag in isolated mode');
    assert.equal(argv[dnsIdx + 1], '0.0.0.0');
  });

  it('isolated mode: injects upper+lower-case proxy env vars pointed at the sidecar', () => {
    const pairs = envPairs(buildContainerArgs(egress));
    assert.ok(pairs.includes('HTTPS_PROXY=http://ksi-proxy:3128'));
    assert.ok(pairs.includes('HTTP_PROXY=http://ksi-proxy:3128'));
    assert.ok(pairs.includes('https_proxy=http://ksi-proxy:3128'));
    assert.ok(pairs.includes('http_proxy=http://ksi-proxy:3128'));
    assert.ok(pairs.some((p) => p.startsWith('NO_PROXY=')));
    assert.ok(pairs.some((p) => p.startsWith('no_proxy=')));
  });

  it('open mode (egress=null): no --dns blackhole, no proxy env, resolves the plain docker network', () => {
    const argv = buildContainerArgs(null, { KSI_DOCKER_NETWORK: '' });
    assert.equal(argv.indexOf('--dns'), -1, 'must NOT blackhole DNS in open mode');
    const pairs = envPairs(argv);
    assert.ok(!pairs.some((p) => p.startsWith('HTTPS_PROXY=') || p.startsWith('HTTP_PROXY=')));
    assert.ok(!pairs.some((p) => p.startsWith('https_proxy=') || p.startsWith('http_proxy=')));
    assert.ok(!pairs.some((p) => p.startsWith('NO_PROXY=') || p.startsWith('no_proxy=')));
    assert.equal(argv[argv.indexOf('--network') + 1], 'bridge');
  });
});

describe('container_args egress lifecycle', () => {
  it('honors the KSI_EGRESS=open escape hatch', () => {
    assert.match(args, /KSI_EGRESS/);
    assert.match(args, /'open'/);
  });
  it('creates an internal (no-route) network when isolated', () => {
    assert.match(args, /--internal/);
  });
  it('derives the allowlist and passes it to the proxy sidecar', () => {
    assert.match(args, /deriveEgressAllowlist/);
    assert.match(args, /KSI_EGRESS_ALLOWLIST/);
  });
  it('injects HTTPS_PROXY for the agent container when isolated', () => {
    assert.match(args, /HTTPS_PROXY/);
    assert.match(args, /HTTP_PROXY/);
    assert.match(args, /NO_PROXY/);
  });
  it('blackholes the embedded resolver upstream so external DNS lookups fail (issue #934)', () => {
    // The agent only needs Docker service discovery (the proxy container name);
    // the proxy resolves provider hostnames upstream on its behalf. Pointing the
    // agent's --dns at a non-routable blackhole keeps service discovery working
    // (answered locally by 127.0.0.11) while making every external name lookup
    // SERVFAIL — closing the DNS-tunnel exfil residual left open by --internal.
    //
    // Pin STRUCTURE, not just presence: --dns must be (a) pushed onto the
    // isolated-only `isolationArgs`, (b) inside the `if (egress)` branch and
    // NOT the open-mode `else`, and (c) spread into the assembled run args.
    // This catches the two regressions a bare presence-check would miss:
    // dropping --dns from the args array, or moving it into open mode.
    assert.match(args, /isolationArgs\.push\(\s*'--dns',\s*'0\.0\.0\.0'\s*\)/);
    const isoStart = args.indexOf('if (egress) {');
    const elseStart = args.indexOf('} else {', isoStart);
    const argsLine = args.indexOf('const args', elseStart);
    assert.ok(
      isoStart >= 0 && elseStart > isoStart && argsLine > elseStart,
      'expected `if (egress) { … } else { … } … const args` structure in buildContainerArgs',
    );
    assert.match(args.slice(isoStart, elseStart), /'--dns', '0\.0\.0\.0'/); // in the isolated branch
    assert.doesNotMatch(args.slice(elseStart, argsLine), /--dns/); // NOT in the open-mode branch
    assert.match(args, /const args[^\n]*\.\.\.isolationArgs/); // isolationArgs reaches the run args
  });
  it('launches the proxy via the egress_proxy_main entry', () => {
    assert.match(args, /egress_proxy_main\.js/);
  });
  it('labels the proxy with the allowlist signature so changed policy is not reused', () => {
    const out = runContainerArgsSnippet(
      `
        import { egressResourceNames, proxyContainerRunArgs, egressAllowlistSignature } from ${JSON.stringify(CONTAINER_ARGS_MOD)};
        const allowlist = ['api.openai.com', 'api.anthropic.com'];
        const argv = proxyContainerRunArgs(egressResourceNames(), allowlist);
        console.log(JSON.stringify({
          signature: egressAllowlistSignature(allowlist),
          hasLabel: argv.includes('--label'),
          labelValue: argv[argv.indexOf('--label') + 1],
        }));
      `,
      { KSI_RUN_ID: 'allowlist-test' },
    );
    assert.equal(out.hasLabel, true);
    assert.equal(out.labelValue, `ksi.egress.allowlist-sha256=${out.signature}`);
  });
  it('tears the infra down idempotently on signals/exit', () => {
    assert.match(args, /teardownEgressInfra/);
    assert.match(args, /process\.(once|on)\(['"](exit|SIGINT|SIGTERM)['"]/);
  });
  it('uses a sanitized shared run id for egress resource names', () => {
    const out = runContainerArgsSnippet(
      `
        import { egressResourceNames, sanitizeEgressRunId } from ${JSON.stringify(CONTAINER_ARGS_MOD)};
        console.log(JSON.stringify({
          sanitized: sanitizeEgressRunId(process.env.KSI_RUN_ID),
          names: egressResourceNames(),
        }));
      `,
      { KSI_RUN_ID: ' Campaign 123/Blue ' },
    );
    assert.equal(out.sanitized, 'campaign-123-blue');
    assert.equal(out.names.internalNet, 'ksi-egress-int-campaign-123-blue');
    assert.equal(out.names.externalNet, 'ksi-egress-ext-campaign-123-blue');
    assert.equal(out.names.proxyContainer, 'ksi-egress-proxy-campaign-123-blue');
  });
  it('tracks one process lease idempotently and releases it', () => {
    const leaseRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-egress-lease-test-'));
    try {
      const out = runContainerArgsSnippet(
        `
          import { acquireEgressLease, activeEgressLeaseCount, releaseEgressLease } from ${JSON.stringify(CONTAINER_ARGS_MOD)};
          const first = acquireEgressLease();
          const second = acquireEgressLease();
          const held = activeEgressLeaseCount();
          releaseEgressLease();
          const released = activeEgressLeaseCount();
          console.log(JSON.stringify({ samePath: first === second, held, released }));
        `,
        { KSI_RUN_ID: 'lease-test', KSI_EGRESS_LEASE_DIR: leaseRoot },
      );
      assert.equal(out.samePath, true);
      assert.equal(out.held, 1);
      assert.equal(out.released, 0);
    } finally {
      fs.rmSync(leaseRoot, { recursive: true, force: true });
    }
  });
  it('keeps a sibling lease alive after the current process releases its lease', () => {
    const leaseRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-egress-sibling-lease-test-'));
    try {
      const out = runContainerArgsSnippet(
        `
          import fs from 'fs';
          import path from 'path';
          import { acquireEgressLease, activeEgressLeaseCount, releaseEgressLease } from ${JSON.stringify(CONTAINER_ARGS_MOD)};
          const ownLease = acquireEgressLease();
          const siblingLease = path.join(path.dirname(ownLease), 'sibling.lease');
          fs.writeFileSync(siblingLease, JSON.stringify({ pid: process.pid }), 'utf-8');
          const held = activeEgressLeaseCount();
          releaseEgressLease();
          const afterRelease = activeEgressLeaseCount();
          fs.unlinkSync(siblingLease);
          const afterSiblingRemoved = activeEgressLeaseCount();
          fs.writeFileSync(path.join(path.dirname(ownLease), 'bogus-identity.lease'), JSON.stringify({
            pid: process.pid,
            process_identity: 'not-this-process',
          }), 'utf-8');
          const afterBogusIdentity = activeEgressLeaseCount();
          console.log(JSON.stringify({ held, afterRelease, afterSiblingRemoved, afterBogusIdentity }));
        `,
        { KSI_RUN_ID: 'sibling-lease-test', KSI_EGRESS_LEASE_DIR: leaseRoot },
      );
      assert.equal(out.held, 2);
      assert.equal(out.afterRelease, 1);
      assert.equal(out.afterSiblingRemoved, 0);
      assert.equal(out.afterBogusIdentity, 0);
    } finally {
      fs.rmSync(leaseRoot, { recursive: true, force: true });
    }
  });
  it('does not break a stale-looking lock whose owner process is still alive', () => {
    const leaseRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-egress-lock-test-'));
    try {
      const out = runContainerArgsSnippet(
        `
          import fs from 'fs';
          import path from 'path';
          import { sanitizeEgressRunId, shouldBreakEgressLeaseLock } from ${JSON.stringify(CONTAINER_ARGS_MOD)};
          const runId = sanitizeEgressRunId(process.env.KSI_RUN_ID);
          const lockDir = path.join(process.env.KSI_EGRESS_LEASE_DIR, runId, '.lock');
          fs.mkdirSync(lockDir, { recursive: true });
          fs.writeFileSync(path.join(lockDir, 'owner.json'), JSON.stringify({ pid: process.pid }), 'utf-8');
          const old = new Date(Date.now() - 600_000);
          fs.utimesSync(lockDir, old, old);
          const liveOwner = shouldBreakEgressLeaseLock(lockDir);
          fs.writeFileSync(path.join(lockDir, 'owner.json'), JSON.stringify({ pid: -1 }), 'utf-8');
          const deadOwner = shouldBreakEgressLeaseLock(lockDir);
          console.log(JSON.stringify({ liveOwner, deadOwner }));
        `,
        { KSI_RUN_ID: 'lock-test', KSI_EGRESS_LEASE_DIR: leaseRoot },
      );
      assert.equal(out.liveOwner, false);
      assert.equal(out.deadOwner, true);
    } finally {
      fs.rmSync(leaseRoot, { recursive: true, force: true });
    }
  });
  it('guards teardown with active lease counts so siblings keep shared infra', () => {
    const teardownStart = args.indexOf('function teardownEgressInfra()');
    assert.ok(teardownStart >= 0, 'teardownEgressInfra must exist');
    const teardownBody = args.slice(teardownStart);
    assert.match(teardownBody, /releaseEgressLease\(\)/);
    assert.match(teardownBody, /activeEgressLeaseCount\(\) > 0/);
    assert.match(teardownBody, /removeEgressResources\(names\)/);
  });
  it('fails closed by asserting the internal network is truly --internal', () => {
    assert.match(args, /network/);
    assert.match(args, /inspect/);
    assert.match(args, /\{\{\.Internal\}\}/);
  });
});

describe('container_runner egress wiring', () => {
  it('resolves egress infra once and threads it into buildContainerArgs', () => {
    assert.match(runner, /ensureEgressInfra/);
    assert.match(runner, /buildContainerArgs\(mounts, containerName, egress\)/);
  });
});
