import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');

// Exercise the TS module's pure logic through a tsx one-shot so we test the
// real source, not a reimplementation. Mirrors the runTsxFixture pattern in
// anthropic_transport_retry.test.mjs.
function runTsx(body) {
  const src = `
    import { isAllowed, parseAllowlist } from ${JSON.stringify(
      path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'egress_proxy.ts'),
    )};
    ${body}
  `;
  const res = spawnSync('npx', ['tsx', '--eval', src], {
    cwd: repoRoot,
    encoding: 'utf-8',
    // Force color OFF: the assertions string-compare a bare `console.log(boolean)`,
    // which Node colorizes (util.inspect) when FORCE_COLOR is set — e.g. this
    // harness exports FORCE_COLOR=3, turning `true` into `\x1B[33mtrue\x1B[39m`
    // and breaking `assert.equal(out, 'true')`. CI (no FORCE_COLOR) is unaffected.
    env: { ...process.env, NO_COLOR: '1', FORCE_COLOR: '0' },
  });
  assert.equal(res.status, 0, `tsx failed: ${res.stderr}`);
  return res.stdout.trim();
}

describe('egress_proxy allowlist', () => {
  it('allows an allowlisted host on 443', () => {
    const out = runTsx(`
      const al = parseAllowlist('api.anthropic.com, api.openai.com');
      console.log(isAllowed('api.anthropic.com', 443, al));
    `);
    assert.equal(out, 'true');
  });

  it('denies a non-allowlisted host', () => {
    const out = runTsx(`
      const al = parseAllowlist('api.anthropic.com');
      console.log(isAllowed('exercism.io', 443, al));
    `);
    assert.equal(out, 'false');
  });

  it('denies a non-443/80 port even if host is allowlisted', () => {
    const out = runTsx(`
      const al = parseAllowlist('api.anthropic.com');
      console.log(isAllowed('api.anthropic.com', 22, al));
    `);
    assert.equal(out, 'false');
  });

  it('does not suffix-match (no api.anthropic.com.evil.com bypass)', () => {
    const out = runTsx(`
      const al = parseAllowlist('api.anthropic.com');
      console.log(isAllowed('api.anthropic.com.evil.com', 443, al));
    `);
    assert.equal(out, 'false');
  });

  it('parseAllowlist is case-insensitive and trims', () => {
    const out = runTsx(`
      const al = parseAllowlist('  API.Anthropic.com , ');
      console.log(isAllowed('api.anthropic.com', 443, al));
    `);
    assert.equal(out, 'true');
  });
});

describe('egress_proxy CONNECT tunnel', () => {
  it('tunnels an allowlisted host and 403s a denied host', () => {
    const fixture = `
      import http from 'node:http';
      import net from 'node:net';
      import { createEgressProxy } from ${JSON.stringify(
        path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'egress_proxy.ts'),
      )};
      const target = http.createServer((_q, s) => { s.end('ok'); });
      await new Promise((r) => target.listen(0, '127.0.0.1', r));
      const tport = target.address().port;
      const proxy = createEgressProxy(new Set(['127.0.0.1']), new Set([tport]));
      await new Promise((r) => proxy.listen(0, '127.0.0.1', r));
      const pport = proxy.address().port;
      function connect(hostport) {
        return new Promise((resolve) => {
          const s = net.connect(pport, '127.0.0.1', () => {
            s.write('CONNECT ' + hostport + ' HTTP/1.1\\r\\nHost: ' + hostport + '\\r\\n\\r\\n');
          });
          let buf = '';
          s.on('data', (d) => { buf += d; if (buf.includes('\\r\\n\\r\\n')) { resolve(buf.split('\\r\\n')[0]); s.destroy(); } });
        });
      }
      const allowed = await connect('127.0.0.1:' + tport);
      const denied = await connect('blocked.example:443');
      console.log(JSON.stringify({ allowed, denied }));
      target.close(); proxy.close();
    `;
    const res = spawnSync('npx', ['tsx', '--input-type=module', '--eval', fixture], {
      cwd: repoRoot, encoding: 'utf-8', env: { ...process.env },
    });
    assert.equal(res.status, 0, res.stderr);
    const out = JSON.parse(
      res.stdout.split('\n').find((l) => l.trim().startsWith('{')),
    );
    assert.match(out.allowed, /200 Connection Established/);
    assert.match(out.denied, /403 Forbidden/);
  });
});
