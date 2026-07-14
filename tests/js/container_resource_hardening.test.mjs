/**
 * Issue #1012 — resource/capability hardening for untrusted agent-generated
 * code running inside `docker run` containers.
 *
 * Exercises the REAL buildContainerArgs() export from
 * runtime_runner/src/container_args.ts via tsx (not a source-text copy), so a
 * refactor that drops a flag from the assembled args array is caught even if
 * the literal string stays somewhere else in the file.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const containerArgsTs = path.join(repoRoot, 'runtime_runner', 'src', 'container_args.ts');
const tsxAvailable = fs.existsSync(tsxBin);

// Call the real buildContainerArgs() inside tsx with a controlled env, in
// non-egress ("open") mode so no docker network/proxy calls are required
// (KCSI_DOCKER_NETWORK defaults to the builtin "bridge", which
// ensureDockerNetwork() no-ops on).
function evalBuildArgs(extraEnv) {
  const script = `
import { buildContainerArgs } from ${JSON.stringify(containerArgsTs)};
const args = buildContainerArgs([], 'test-container', null);
process.stdout.write(JSON.stringify(args));
`;
  return spawnSync(tsxBin, ['--eval', script, '--conditions=node'], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, ...extraEnv },
  });
}

describe('container_args resource hardening (issue #1012)', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/');
    return;
  }

  it('drops all capabilities and blocks privilege escalation by default', () => {
    const res = evalBuildArgs({});
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
    const args = JSON.parse(res.stdout);
    assert.ok(args.includes('--cap-drop=ALL'), `missing --cap-drop=ALL in ${JSON.stringify(args)}`);
    assert.ok(
      args.includes('--security-opt=no-new-privileges'),
      `missing --security-opt=no-new-privileges in ${JSON.stringify(args)}`,
    );
  });

  it('caps process count with a generous default pids-limit', () => {
    const res = evalBuildArgs({});
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const args = JSON.parse(res.stdout);
    assert.ok(args.includes('--pids-limit=4096'), `missing --pids-limit=4096 in ${JSON.stringify(args)}`);
  });

  it('honors KCSI_CONTAINER_PIDS_LIMIT to override the default', () => {
    const res = evalBuildArgs({ KCSI_CONTAINER_PIDS_LIMIT: '256' });
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const args = JSON.parse(res.stdout);
    assert.ok(args.includes('--pids-limit=256'), `expected overridden pids-limit in ${JSON.stringify(args)}`);
    assert.ok(!args.includes('--pids-limit=4096'));
  });

  it('does NOT set --memory or --cpus by default (avoid silently breaking larger workloads)', () => {
    const res = evalBuildArgs({});
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const args = JSON.parse(res.stdout);
    assert.ok(!args.some((a) => a.startsWith('--memory')), `unexpected --memory in ${JSON.stringify(args)}`);
    assert.ok(!args.some((a) => a.startsWith('--cpus')), `unexpected --cpus in ${JSON.stringify(args)}`);
  });

  it('sets --memory/--cpus only when explicitly opted in via env', () => {
    const res = evalBuildArgs({ KCSI_CONTAINER_MEMORY_LIMIT: '8g', KCSI_CONTAINER_CPU_LIMIT: '4' });
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const args = JSON.parse(res.stdout);
    assert.ok(args.includes('--memory=8g'), `missing opt-in --memory in ${JSON.stringify(args)}`);
    assert.ok(args.includes('--cpus=4'), `missing opt-in --cpus in ${JSON.stringify(args)}`);
  });
});

// Issue #1025 (follow-up to #1019): the egress-proxy sidecar container started
// by ensureEgressInfra() must get the same hardening flags as agent
// containers. Exercises the REAL proxyContainerRunArgs() export via tsx
// rather than starting an actual docker container.
function evalProxyRunArgs(extraEnv) {
  const script = `
import { proxyContainerRunArgs, egressResourceNames } from ${JSON.stringify(containerArgsTs)};
const args = proxyContainerRunArgs(egressResourceNames(), ['api.anthropic.com']);
process.stdout.write(JSON.stringify(args));
`;
  return spawnSync(tsxBin, ['--eval', script, '--conditions=node'], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, ...extraEnv },
  });
}

describe('egress-proxy sidecar resource hardening (issue #1025)', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/');
    return;
  }

  it('drops all capabilities and blocks privilege escalation on the proxy container', () => {
    const res = evalProxyRunArgs({});
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}\nstdout=${res.stdout}`);
    const args = JSON.parse(res.stdout);
    assert.ok(args.includes('--cap-drop=ALL'), `missing --cap-drop=ALL in ${JSON.stringify(args)}`);
    assert.ok(
      args.includes('--security-opt=no-new-privileges'),
      `missing --security-opt=no-new-privileges in ${JSON.stringify(args)}`,
    );
  });

  it('caps process count on the proxy container with the same default pids-limit', () => {
    const res = evalProxyRunArgs({});
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const args = JSON.parse(res.stdout);
    assert.ok(args.includes('--pids-limit=4096'), `missing --pids-limit=4096 in ${JSON.stringify(args)}`);
  });

  it('honors KCSI_CONTAINER_PIDS_LIMIT override on the proxy container too', () => {
    const res = evalProxyRunArgs({ KCSI_CONTAINER_PIDS_LIMIT: '256' });
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const args = JSON.parse(res.stdout);
    assert.ok(args.includes('--pids-limit=256'), `expected overridden pids-limit in ${JSON.stringify(args)}`);
  });

  it('still constructs a valid docker run invocation for the proxy (name/network/entrypoint intact)', () => {
    const res = evalProxyRunArgs({});
    assert.equal(res.status, 0, `tsx failed: stderr=${res.stderr}`);
    const args = JSON.parse(res.stdout);
    assert.ok(args.includes('/tmp/dist/egress_proxy_main.js'), `missing proxy entry script in ${JSON.stringify(args)}`);
    assert.ok(args.includes('--entrypoint'), `missing --entrypoint in ${JSON.stringify(args)}`);
  });
});
