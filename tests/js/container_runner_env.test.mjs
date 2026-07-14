import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');
// The container-runner god file was split (issue #913 follow-up): env
// forwarding + secret reading live in container_args.ts; runner-root path
// resolution lives in container_mounts.ts.
const containerArgs = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'src', 'container_args.ts'),
  'utf-8',
);
const containerMounts = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'src', 'container_mounts.ts'),
  'utf-8',
);

describe('container_runner env forwarding', () => {
  it('forwards OpenAI runtime knobs into the container', () => {
    assert.match(containerArgs, /'KCSI_OPENAI_MAX_TURNS'/);
    assert.match(containerArgs, /'OPENAI_AGENTS_DISABLE_TRACING'/);
  });

  it('forwards the unsafe Claude native-tool override explicitly', () => {
    assert.match(containerArgs, /'KCSI_ALLOW_UNSAFE_CLAUDE_NATIVE_TOOLS_WITH_SECRETS'/);
  });

  it('supports moving the runner away from official SWE repo paths', () => {
    assert.match(containerMounts, /KCSI_RUNNER_ROOT/);
    assert.match(containerMounts, /KCSI_TASK_REPO_CONTAINER_PATH/);
    assert.match(containerMounts, /runnerPath\('src'\)/);
    assert.match(containerMounts, /runnerPath\('node_modules'\)/);
  });

  it('lets provider-profile process env override .env secrets', () => {
    const readSecretsStart = containerArgs.indexOf('function readSecrets()');
    const readEnvFileAt = containerArgs.indexOf('const fromFile = readEnvFile', readSecretsStart);
    const processEnvAt = containerArgs.indexOf('process.env[key]', readSecretsStart);
    const fromFileFallbackAt = containerArgs.indexOf('fromFile[key]', readEnvFileAt);

    assert.ok(readSecretsStart >= 0, 'readSecrets() must exist');
    assert.ok(processEnvAt > readSecretsStart, 'readSecrets() must inspect process.env');
    assert.ok(readEnvFileAt > processEnvAt, 'process.env secrets must be read before .env');
    assert.ok(fromFileFallbackAt > readEnvFileAt, '.env secrets must be fallback-only');
  });
});
