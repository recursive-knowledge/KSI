import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const MOD = path.join(repoRoot, 'runtime_runner', 'src', 'container_args.ts');

// Drives reapStaleEgressNetworks with a fake `runDocker` that answers
// `network ls`, `inspect <proxy>`, `network inspect -f '{{len .Containers}}'`,
// and `network rm` calls from a scripted table keyed by the joined args —
// no real docker daemon involved, so the "only remove networks with zero
// attached containers AND a dead sibling proxy" safety logic is verified as
// real behavior (this is exactly the logic that must not race a concurrently
// STARTING sibling process's egress setup).
// A fixed, far-in-the-past timestamp for "genuinely stale" fixtures — safely
// older than any graceMs used in these tests regardless of wall-clock time.
// `docker network inspect -f '{{json .Created}}'` emits a QUOTED RFC3339Nano
// string (JSON-encoded), so fixtures carry the surrounding double quotes.
const ANCIENT_CREATED = '"2020-01-01T00:00:00.000000000Z"';
// Same age but with a non-US UTC offset — proves the parse path no longer
// depends on V8 recognizing a zone abbreviation (the old bare `{{.Created}}`
// Go-format output was NaN for CET/JST/IST/... hosts).
const ANCIENT_CREATED_NON_US_OFFSET = '"2020-01-01T00:00:00.000000000+05:30"';

// `graceMs: null` omits the argument so the function's REAL default
// (env-overridable via KCSI_EGRESS_REAP_GRACE_MS) is exercised; `env` controls
// the child process env for those override tests. KCSI_EGRESS_REAP_GRACE_MS
// defaults to '' (= unset for the envInt fallback path) so the host machine's
// environment can't leak in.
function runReaper(networkLsOutput, responses, graceMs = null, env = {}) {
  const src = `
    import { reapStaleEgressNetworks } from ${JSON.stringify(MOD)};
    const responses = ${JSON.stringify(responses)};
    const calls = [];
    function fakeRunDocker(args) {
      const key = args.join(' ');
      calls.push(key);
      if (args[0] === 'network' && args[1] === 'ls') {
        return { status: 0, stdout: ${JSON.stringify(networkLsOutput)} };
      }
      if (key in responses) return responses[key];
      // default: "does not exist" / "no containers" so unmatched calls don't
      // accidentally look like a live sibling.
      return { status: 1, stdout: '' };
    }
    reapStaleEgressNetworks(fakeRunDocker${graceMs == null ? '' : `, ${JSON.stringify(graceMs)}`});
    console.log(JSON.stringify({ calls }));
  `;
  const res = spawnSync('npx', ['tsx', '--eval', src], {
    cwd: repoRoot,
    encoding: 'utf-8',
    env: { ...process.env, KCSI_EGRESS_REAP_GRACE_MS: '', ...env },
  });
  assert.equal(res.status, 0, res.stderr);
  return JSON.parse(res.stdout.trim());
}

describe('reapStaleEgressNetworks', () => {
  it('removes a network pair whose sibling proxy is gone, has zero attached containers, and is older than the grace period', () => {
    const { calls } = runReaper('kcsi-egress-int-1234\nkcsi-egress-ext-1234\n', {
      'inspect kcsi-egress-proxy-1234': { status: 1 }, // proxy container gone
      'network inspect -f {{len .Containers}} kcsi-egress-int-1234': { status: 0, stdout: '0' },
      'network inspect -f {{len .Containers}} kcsi-egress-ext-1234': { status: 0, stdout: '0' },
      'network inspect -f {{json .Created}} kcsi-egress-int-1234': { status: 0, stdout: ANCIENT_CREATED },
      'network inspect -f {{json .Created}} kcsi-egress-ext-1234': { status: 0, stdout: ANCIENT_CREATED },
    });
    assert.ok(calls.includes('network rm kcsi-egress-int-1234'));
    assert.ok(calls.includes('network rm kcsi-egress-ext-1234'));
  });

  it('leaves a network alone when it was created recently, even with a dead proxy and zero containers (race-prevention: a concurrently-starting sibling)', () => {
    // 5s ago, JSON-quoted like real `{{json .Created}}` output
    const recentCreated = JSON.stringify(new Date(Date.now() - 5_000).toISOString());
    const { calls } = runReaper('kcsi-egress-int-4321\nkcsi-egress-ext-4321\n', {
      'inspect kcsi-egress-proxy-4321': { status: 1 },
      'network inspect -f {{len .Containers}} kcsi-egress-int-4321': { status: 0, stdout: '0' },
      'network inspect -f {{len .Containers}} kcsi-egress-ext-4321': { status: 0, stdout: '0' },
      'network inspect -f {{json .Created}} kcsi-egress-int-4321': { status: 0, stdout: recentCreated },
      'network inspect -f {{json .Created}} kcsi-egress-ext-4321': { status: 0, stdout: recentCreated },
    }, 120_000);
    assert.ok(!calls.includes('network rm kcsi-egress-int-4321'));
    assert.ok(!calls.includes('network rm kcsi-egress-ext-4321'));
  });

  it('removes a network whose age exceeds a small injected grace period', () => {
    const { calls } = runReaper('kcsi-egress-int-2222\n', {
      'inspect kcsi-egress-proxy-2222': { status: 1 },
      'network inspect -f {{len .Containers}} kcsi-egress-int-2222': { status: 0, stdout: '0' },
      'network inspect -f {{json .Created}} kcsi-egress-int-2222': { status: 0, stdout: ANCIENT_CREATED },
    }, 1_000);
    assert.ok(calls.includes('network rm kcsi-egress-int-2222'));
  });

  it('removes stale networks for sanitized non-numeric campaign run ids', () => {
    const runId = 'campaign-123.blue';
    const { calls } = runReaper(`kcsi-egress-int-${runId}\nkcsi-egress-ext-${runId}\n`, {
      [`inspect kcsi-egress-proxy-${runId}`]: { status: 1 },
      [`network inspect -f {{len .Containers}} kcsi-egress-int-${runId}`]: { status: 0, stdout: '0' },
      [`network inspect -f {{len .Containers}} kcsi-egress-ext-${runId}`]: { status: 0, stdout: '0' },
      [`network inspect -f {{json .Created}} kcsi-egress-int-${runId}`]: { status: 0, stdout: ANCIENT_CREATED },
      [`network inspect -f {{json .Created}} kcsi-egress-ext-${runId}`]: { status: 0, stdout: ANCIENT_CREATED },
    }, 1_000);
    assert.ok(calls.includes(`network rm kcsi-egress-int-${runId}`));
    assert.ok(calls.includes(`network rm kcsi-egress-ext-${runId}`));
  });

  it('removes an ancient network whose Created timestamp carries a non-US UTC offset (timezone-robust parse)', () => {
    const { calls } = runReaper('kcsi-egress-int-8888\n', {
      'inspect kcsi-egress-proxy-8888': { status: 1 },
      'network inspect -f {{len .Containers}} kcsi-egress-int-8888': { status: 0, stdout: '0' },
      'network inspect -f {{json .Created}} kcsi-egress-int-8888': {
        status: 0,
        stdout: ANCIENT_CREATED_NON_US_OFFSET,
      },
    });
    assert.ok(calls.includes('network rm kcsi-egress-int-8888'));
  });

  it('fail-safe: leaves a network alone when the Created-timestamp inspect fails', () => {
    const { calls } = runReaper('kcsi-egress-int-3333\n', {
      'inspect kcsi-egress-proxy-3333': { status: 1 },
      'network inspect -f {{len .Containers}} kcsi-egress-int-3333': { status: 0, stdout: '0' },
      'network inspect -f {{json .Created}} kcsi-egress-int-3333': { status: 1, stdout: '' },
    });
    assert.ok(!calls.includes('network rm kcsi-egress-int-3333'));
  });

  it('fail-safe: leaves a network alone when the Created timestamp is unparsable', () => {
    const { calls } = runReaper('kcsi-egress-int-7777\n', {
      'inspect kcsi-egress-proxy-7777': { status: 1 },
      'network inspect -f {{len .Containers}} kcsi-egress-int-7777': { status: 0, stdout: '0' },
      'network inspect -f {{json .Created}} kcsi-egress-int-7777': { status: 0, stdout: 'not-a-timestamp' },
    });
    assert.ok(!calls.includes('network rm kcsi-egress-int-7777'));
  });

  it('leaves a network alone when its sibling proxy container is still alive', () => {
    const { calls } = runReaper('kcsi-egress-int-5678\nkcsi-egress-ext-5678\n', {
      'inspect kcsi-egress-proxy-5678': { status: 0 }, // proxy still running (or mid-setup)
      'network inspect -f {{len .Containers}} kcsi-egress-int-5678': { status: 0, stdout: '0' },
      'network inspect -f {{len .Containers}} kcsi-egress-ext-5678': { status: 0, stdout: '0' },
    });
    assert.ok(!calls.includes('network rm kcsi-egress-int-5678'));
    assert.ok(!calls.includes('network rm kcsi-egress-ext-5678'));
  });

  it('leaves a network alone when it still has an attached container even if the proxy is gone', () => {
    const { calls } = runReaper('kcsi-egress-int-9999\n', {
      'inspect kcsi-egress-proxy-9999': { status: 1 },
      'network inspect -f {{len .Containers}} kcsi-egress-int-9999': { status: 0, stdout: '1' }, // an agent container is still attached
    });
    assert.ok(!calls.includes('network rm kcsi-egress-int-9999'));
  });

  it('ignores networks that do not match the kcsi-egress-(int|ext)-<id> pattern', () => {
    const { calls } = runReaper('bridge\nhost\nsome-other-network\n', {});
    assert.ok(!calls.some((c) => c.startsWith('network rm')));
  });

  describe('default grace period (launch-storm hardening)', () => {
    // A network 200s old: older than the pre-incident 120s default, younger
    // than the new 600s default. The 2026-07-03 incident showed proxies can
    // take minutes to attach under docker contention (3 concurrent campaigns
    // fanning out 18 containers), so a sibling mid-setup at 200s must NOT be
    // reaped under the shipped default.
    const created200sAgo = () => JSON.stringify(new Date(Date.now() - 200_000).toISOString());
    const created660sAgo = () => JSON.stringify(new Date(Date.now() - 660_000).toISOString());

    function fixtures(runId, createdStdout) {
      return {
        [`inspect kcsi-egress-proxy-${runId}`]: { status: 1 },
        [`network inspect -f {{len .Containers}} kcsi-egress-int-${runId}`]: { status: 0, stdout: '0' },
        [`network inspect -f {{json .Created}} kcsi-egress-int-${runId}`]: { status: 0, stdout: createdStdout },
      };
    }

    it('leaves a 200s-old network alone under the default grace (default must exceed 200s)', () => {
      const { calls } = runReaper(
        'kcsi-egress-int-6001\n', fixtures('6001', created200sAgo()), null,
      );
      assert.ok(!calls.includes('network rm kcsi-egress-int-6001'));
    });

    it('reaps a 660s-old network under the default grace (default must not exceed 660s)', () => {
      const { calls } = runReaper(
        'kcsi-egress-int-6002\n', fixtures('6002', created660sAgo()), null,
      );
      assert.ok(calls.includes('network rm kcsi-egress-int-6002'));
    });

    it('honors KCSI_EGRESS_REAP_GRACE_MS as the default grace', () => {
      const { calls } = runReaper(
        'kcsi-egress-int-6003\n', fixtures('6003', created200sAgo()), null,
        { KCSI_EGRESS_REAP_GRACE_MS: '1000' },
      );
      assert.ok(calls.includes('network rm kcsi-egress-int-6003'));
    });

    it('falls back to the shipped default grace on a garbage KCSI_EGRESS_REAP_GRACE_MS', () => {
      const { calls } = runReaper(
        'kcsi-egress-int-6004\n', fixtures('6004', created200sAgo()), null,
        { KCSI_EGRESS_REAP_GRACE_MS: 'soon' },
      );
      assert.ok(!calls.includes('network rm kcsi-egress-int-6004'));
    });

    it('falls back to the shipped default grace on duration/scientific/zero/negative KCSI_EGRESS_REAP_GRACE_MS values (200s network not reaped)', () => {
      for (const bad of ['600s', '1e3', '0', '-3']) {
        const { calls } = runReaper(
          'kcsi-egress-int-6006\n', fixtures('6006', created200sAgo()), null,
          { KCSI_EGRESS_REAP_GRACE_MS: bad },
        );
        assert.ok(!calls.includes('network rm kcsi-egress-int-6006'), `grace=${bad}`);
      }
    });

    it('an explicit graceMs argument wins over the env override', () => {
      const { calls } = runReaper(
        'kcsi-egress-int-6005\n', fixtures('6005', created200sAgo()), 1_000,
        { KCSI_EGRESS_REAP_GRACE_MS: '900000' },
      );
      assert.ok(calls.includes('network rm kcsi-egress-int-6005'));
    });
  });
});
