// shell_tool_path_inheritance.test.mjs
//
// Regression guard: the agent's shell tool must NOT use a login shell
// (`/bin/sh -lc`). On Debian/Ubuntu base images — which most SWE-bench Pro
// per-instance images derive from — /etc/profile resets PATH to
// `/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games`, stripping the
// language toolchains those images install at non-default locations:
//   * Go at /usr/local/go/bin and /go/bin
//   * Rust at /root/.cargo/bin
//   * Any project-local installs under /opt/...
//
// When the agent's shell tool runs `bash run_script.sh ...`, a login shell
// loses these paths and the SWE-bench Pro grader's run_script.sh fails to
// find its toolchain. The grader then reports NO_TESTS_FOUND_OR_PARSING_ERROR
// and the instance scores 0 even though the agent's diff might be correct.
//
// We assert the source uses `-c` (non-login) and explicitly forbid `-lc`.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const OPENAI_ADAPTER = path.join(REPO_ROOT, 'runtime_runner', 'agent-runner', 'src', 'openai.ts');

test('agent-runner shell tool does not spawn login shells (-lc)', () => {
  const src = fs.readFileSync(OPENAI_ADAPTER, 'utf-8');
  // No login-shell flags anywhere — neither in the createShell impl nor in
  // any future helper. -lc inside a literal string is fine (e.g., comment),
  // so we look for the actual spawn arguments.
  const offending = src.match(/spawn\([^)]*['"]-lc['"]/);
  assert.strictEqual(
    offending,
    null,
    `openai.ts spawns a login shell (-lc), which strips PATH on Debian/Ubuntu base images. Use '-c' so the spawned shell inherits the env we explicitly pass.`,
  );
});

test('agent-runner shell tool spawns /bin/sh -c with the inherited env', () => {
  const src = fs.readFileSync(OPENAI_ADAPTER, 'utf-8');
  // Positive assertion: confirm we're using the non-login form.
  assert.match(
    src,
    /spawn\(['"]\/bin\/sh['"],\s*\[['"]-c['"], command\]/,
    `Expected createShell to spawn ['/bin/sh', '-c', command]; the non-login flag is critical so language toolchains stay on PATH.`,
  );
});
