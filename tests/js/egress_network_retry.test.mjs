import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const MOD = path.join(repoRoot, 'runtime_runner', 'src', 'container_args.ts');

// Drives ensureEgressDockerNetwork with a fake `runDocker` (no real docker
// daemon involved) that scripts a sequence of {status, stdout} results per
// call, so the retry/backoff/failure behavior under docker-daemon
// contention (issue: concurrent `network create` calls transiently failing)
// is verified as real behavior, not just source-pinned text.
function runWithFakeDocker({ name, opts, calls }) {
  const src = `
    import { ensureEgressDockerNetwork } from ${JSON.stringify(MOD)};
    const calls = ${JSON.stringify(calls)};
    let i = 0;
    const seen = [];
    function fakeRunDocker(args, timeoutMs) {
      seen.push(args.join(' '));
      const result = calls[Math.min(i, calls.length - 1)];
      i += 1;
      return result;
    }
    try {
      ensureEgressDockerNetwork(${JSON.stringify(name)}, {
        ...${JSON.stringify(opts)},
        delayMs: 1,
        runDocker: fakeRunDocker,
      });
      console.log(JSON.stringify({ ok: true, seen }));
    } catch (err) {
      console.log(JSON.stringify({ ok: false, error: String(err && err.message || err), seen }));
    }
  `;
  const res = spawnSync('npx', ['tsx', '--eval', src], {
    cwd: repoRoot, encoding: 'utf-8', env: { ...process.env },
  });
  assert.equal(res.status, 0, res.stderr);
  return JSON.parse(res.stdout.trim());
}

// Like runWithFakeDocker, but with an injected recording `sleeper` (so the
// exponential-backoff schedule is observable without real sleeping) and
// controllable child-process env (so the KCSI_EGRESS_NET_READY_* override
// parsing is exercised for real). Env keys default to '' (= unset for the
// envInt fallback path) so the host machine's environment can't leak in.
function runScenario({ name, opts = {}, calls, env = {} }) {
  const src = `
    import { ensureEgressDockerNetwork } from ${JSON.stringify(MOD)};
    const calls = ${JSON.stringify(calls)};
    let i = 0;
    const seen = [];
    const sleeps = [];
    function fakeRunDocker(args, timeoutMs) {
      seen.push(args.join(' '));
      const result = calls[Math.min(i, calls.length - 1)];
      i += 1;
      return result;
    }
    try {
      ensureEgressDockerNetwork(${JSON.stringify(name)}, {
        ...${JSON.stringify(opts)},
        runDocker: fakeRunDocker,
        sleeper: (ms) => { sleeps.push(ms); },
      });
      console.log(JSON.stringify({ ok: true, seen, sleeps }));
    } catch (err) {
      console.log(JSON.stringify({ ok: false, error: String(err && err.message || err), seen, sleeps }));
    }
  `;
  const res = spawnSync('npx', ['tsx', '--eval', src], {
    cwd: repoRoot,
    encoding: 'utf-8',
    env: {
      ...process.env,
      KCSI_EGRESS_NET_READY_ATTEMPTS: '',
      KCSI_EGRESS_NET_READY_DELAY_MS: '',
      ...env,
    },
  });
  assert.equal(res.status, 0, res.stderr);
  return JSON.parse(res.stdout.trim());
}

describe('ensureEgressDockerNetwork retry behavior', () => {
  it('succeeds immediately when create+inspect both work on the first try', () => {
    const result = runWithFakeDocker({
      name: 'kcsi-egress-ext-test',
      opts: { requireInternal: false, attempts: 5 },
      calls: [
        { status: 0 }, // network create
        { status: 0, stdout: 'false' }, // network inspect
      ],
    });
    assert.equal(result.ok, true);
    assert.equal(result.seen.length, 2);
  });

  it('retries past a transient inspect failure and succeeds', () => {
    const result = runWithFakeDocker({
      name: 'kcsi-egress-ext-test',
      opts: { requireInternal: false, attempts: 5 },
      calls: [
        { status: 0 }, // attempt 1: create
        { status: 1 }, // attempt 1: inspect fails (network not found yet)
        { status: 0 }, // cleanup rm (from the retry path)
        { status: 0 }, // attempt 2: create
        { status: 0, stdout: 'false' }, // attempt 2: inspect succeeds
      ],
    });
    assert.equal(result.ok, true);
  });

  it('gives up and throws after exhausting all attempts', () => {
    const result = runWithFakeDocker({
      name: 'kcsi-egress-ext-test',
      opts: { requireInternal: false, attempts: 3 },
      calls: [{ status: 1 }],
    });
    assert.equal(result.ok, false);
    assert.match(result.error, /did not become ready after 3 attempt/);
  });

  it('requires the --internal flag to actually be set when requireInternal is true', () => {
    const result = runWithFakeDocker({
      name: 'kcsi-egress-int-test',
      opts: { requireInternal: true, attempts: 2 },
      calls: [
        { status: 0 }, // create returns ok
        { status: 0, stdout: 'false' }, // but inspect says NOT internal (name clash) -> must not accept
      ],
    });
    assert.equal(result.ok, false);
    assert.match(result.error, /must be --internal/);
  });
});

describe('ensureEgressDockerNetwork backoff and env overrides (concurrent launch-storm hardening)', () => {
  const alwaysFail = [{ status: 1 }];

  it('defaults to a 10-attempt budget when no attempts option or env override is given', () => {
    const result = runScenario({
      name: 'kcsi-egress-int-test',
      opts: { requireInternal: true },
      calls: alwaysFail,
    });
    assert.equal(result.ok, false);
    assert.match(result.error, /did not become ready after 10 attempt/);
    // One sleep between each pair of consecutive attempts.
    assert.equal(result.sleeps.length, 9);
  });

  it('uses the exact default backoff schedule (300/600/1200/2400/4800 then capped at 5000) with no opts or env override', () => {
    const result = runScenario({
      name: 'kcsi-egress-int-test',
      opts: { requireInternal: true },
      calls: alwaysFail,
    });
    assert.equal(result.ok, false);
    assert.match(result.error, /did not become ready after 10 attempt/);
    assert.deepEqual(result.sleeps, [300, 600, 1200, 2400, 4800, 5000, 5000, 5000, 5000]);
  });

  it('falls back to the default attempt budget on duration/scientific/zero/negative KCSI_EGRESS_NET_READY_ATTEMPTS values', () => {
    for (const bad of ['600s', '1e3', '0', '-3']) {
      const result = runScenario({
        name: 'kcsi-egress-int-test',
        opts: { requireInternal: true },
        calls: alwaysFail,
        env: { KCSI_EGRESS_NET_READY_ATTEMPTS: bad },
      });
      assert.equal(result.ok, false, `attempts=${bad}`);
      assert.match(result.error, /did not become ready after 10 attempt/, `attempts=${bad}`);
    }
  });

  it('sleeps with exponential backoff between attempts, capped at 5000ms', () => {
    const result = runScenario({
      name: 'kcsi-egress-ext-test',
      opts: { requireInternal: false, attempts: 7, delayMs: 1000 },
      calls: alwaysFail,
    });
    assert.equal(result.ok, false);
    assert.deepEqual(result.sleeps, [1000, 2000, 4000, 5000, 5000, 5000]);
  });

  it('honors KCSI_EGRESS_NET_READY_ATTEMPTS when no explicit attempts option is passed', () => {
    const result = runScenario({
      name: 'kcsi-egress-int-test',
      opts: { requireInternal: true },
      calls: alwaysFail,
      env: { KCSI_EGRESS_NET_READY_ATTEMPTS: '3' },
    });
    assert.equal(result.ok, false);
    assert.match(result.error, /did not become ready after 3 attempt/);
  });

  it('falls back to the default attempt budget on a garbage KCSI_EGRESS_NET_READY_ATTEMPTS', () => {
    const result = runScenario({
      name: 'kcsi-egress-int-test',
      opts: { requireInternal: true },
      calls: alwaysFail,
      env: { KCSI_EGRESS_NET_READY_ATTEMPTS: 'banana' },
    });
    assert.equal(result.ok, false);
    assert.match(result.error, /did not become ready after 10 attempt/);
  });

  it('honors KCSI_EGRESS_NET_READY_DELAY_MS as the backoff base delay', () => {
    const result = runScenario({
      name: 'kcsi-egress-ext-test',
      opts: { requireInternal: false, attempts: 4 },
      calls: alwaysFail,
      env: { KCSI_EGRESS_NET_READY_DELAY_MS: '50' },
    });
    assert.equal(result.ok, false);
    assert.deepEqual(result.sleeps, [50, 100, 200]);
  });

  it('explicit attempts option wins over the env override', () => {
    const result = runScenario({
      name: 'kcsi-egress-int-test',
      opts: { requireInternal: true, attempts: 2 },
      calls: alwaysFail,
      env: { KCSI_EGRESS_NET_READY_ATTEMPTS: '8' },
    });
    assert.equal(result.ok, false);
    assert.match(result.error, /did not become ready after 2 attempt/);
  });
});
