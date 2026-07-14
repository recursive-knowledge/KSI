import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const MAIN = path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'egress_proxy_main.ts');

describe('egress_proxy_main', () => {
  it('starts, prints READY, and 403s a denied host then exits', () => {
    // Driver: spawn the entry on an ephemeral port with a known allowlist,
    // wait for READY, probe a denied host, assert 403, kill.
    const driver = `
      import net from 'node:net';
      import { spawn } from 'node:child_process';
      const proc = spawn('npx', ['tsx', ${JSON.stringify(MAIN)}], {
        env: { ...process.env, KCSI_EGRESS_PROXY_PORT: '0', KCSI_EGRESS_ALLOWLIST: 'api.anthropic.com' },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let buf = '';
      const port = await new Promise((resolve, reject) => {
        proc.stdout.on('data', (d) => {
          buf += d;
          const m = buf.match(/READY listening on (\\d+)/);
          if (m) resolve(Number(m[1]));
        });
        setTimeout(() => reject(new Error('no READY: ' + buf)), 10000);
      });
      const status = await new Promise((resolve) => {
        const s = net.connect(port, '127.0.0.1', () => {
          s.write('CONNECT blocked.example:443 HTTP/1.1\\r\\nHost: blocked.example\\r\\n\\r\\n');
        });
        let b = '';
        s.on('data', (d) => { b += d; if (b.includes('\\r\\n\\r\\n')) { resolve(b.split('\\r\\n')[0]); s.destroy(); } });
      });
      console.log(status);
      proc.kill('SIGKILL');
      process.exit(0);
    `;
    const res = spawnSync('npx', ['tsx', '--input-type=module', '--eval', driver], {
      cwd: repoRoot, encoding: 'utf-8', env: { ...process.env },
    });
    assert.equal(res.status, 0, res.stderr);
    assert.match(res.stdout, /403 Forbidden/);
  });
});
