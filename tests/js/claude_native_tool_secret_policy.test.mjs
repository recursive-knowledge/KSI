import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const queryConfigTs = path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'query_config.ts');
const tsxAvailable = fs.existsSync(tsxBin);

function evaluatePolicies() {
  const source = `
    import {
      NATIVE_FILE_SHELL_TOOLS,
      buildToolPolicy,
      sdkEnvHasNativeToolLeakSecret,
      shouldDenyNativeToolsForSdkSecrets,
    } from ${JSON.stringify(queryConfigTs)};

    const cases = {
      noSecret: buildToolPolicy({}, 'polyglot', false, {}),
      // Secret present but egress isolation ON (production default): native
      // tools MUST stay available or every coding task loses its file/shell
      // tools. This is the normal production shape.
      secretIsolated: buildToolPolicy({}, 'polyglot', false, { ANTHROPIC_API_KEY: 'sk-ant-secret' }),
      // Secret present AND egress open (debug only): deny native tools.
      secret: buildToolPolicy(
        {},
        'polyglot',
        false,
        { ANTHROPIC_API_KEY: 'sk-ant-secret', KSI_EGRESS: 'open' },
      ),
      secretWithWeb: buildToolPolicy(
        {},
        'polyglot',
        false,
        { ANTHROPIC_API_KEY: 'sk-ant-secret', KSI_EGRESS: 'open', KSI_ALLOW_WEB_TOOLS: '1' },
      ),
      override: buildToolPolicy(
        {},
        'polyglot',
        false,
        {
          ANTHROPIC_API_KEY: 'sk-ant-secret',
          KSI_EGRESS: 'open',
          KSI_ALLOW_UNSAFE_CLAUDE_NATIVE_TOOLS_WITH_SECRETS: '1',
        },
      ),
      forum: buildToolPolicy({}, 'per_task_forum', true, {}),
    };
    process.stdout.write(JSON.stringify({
      nativeTools: NATIVE_FILE_SHELL_TOOLS,
      hasSecret: sdkEnvHasNativeToolLeakSecret({ HF_TOKEN: 'hf-secret' }),
      // Secret present but isolated egress: NOT denied (leak is contained).
      denySecretIsolated: shouldDenyNativeToolsForSdkSecrets({ HF_TOKEN: 'hf-secret' }),
      // Secret present AND egress open: denied.
      denySecret: shouldDenyNativeToolsForSdkSecrets({ HF_TOKEN: 'hf-secret', KSI_EGRESS: 'open' }),
      denyOverride: shouldDenyNativeToolsForSdkSecrets({
        HF_TOKEN: 'hf-secret',
        KSI_EGRESS: 'open',
        KSI_ALLOW_UNSAFE_CLAUDE_NATIVE_TOOLS_WITH_SECRETS: 'true',
      }),
      cases,
    }));
  `;
  return spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: { ...process.env, NODE_NO_WARNINGS: '1' },
  });
}

describe('Claude native tool policy with SDK env secrets', () => {
  if (!tsxAvailable) {
    it.skip('tsx not installed; run npm install in runtime_runner/');
    return;
  }

  const result = evaluatePolicies();
  it('evaluates the real query_config module', () => {
    assert.equal(result.status, 0, `tsx failed\nstdout=${result.stdout}\nstderr=${result.stderr}`);
  });
  if (result.status !== 0) return;

  const sample = JSON.parse(result.stdout);

  it('detects leak-sensitive secrets and honors egress isolation + the unsafe override', () => {
    assert.equal(sample.hasSecret, true);
    // Secret present but egress isolated: NOT denied (leak is contained).
    assert.equal(sample.denySecretIsolated, false);
    // Secret present AND egress open: denied.
    assert.equal(sample.denySecret, true);
    // Even with egress open, the explicit unsafe override restores tools.
    assert.equal(sample.denyOverride, false);
  });

  it('keeps native tools available when no SDK env secret is present', () => {
    assert.ok(sample.cases.noSecret.allowedToolsList.includes('Bash'));
    assert.ok(sample.cases.noSecret.allowedToolsList.includes('Read'));
    assert.ok(!sample.cases.noSecret.disallowedToolsList.includes('Bash'));
  });

  it('keeps native tools available under isolated egress even with secrets (production default)', () => {
    assert.ok(sample.cases.secretIsolated.allowedToolsList.includes('Bash'));
    assert.ok(sample.cases.secretIsolated.allowedToolsList.includes('Read'));
    assert.ok(!sample.cases.secretIsolated.disallowedToolsList.includes('Bash'));
  });

  it('denies every native file/shell tool when credentials are present AND egress is open', () => {
    for (const toolName of sample.nativeTools) {
      assert.ok(!sample.cases.secret.allowedToolsList.includes(toolName), `${toolName} must not be allowed`);
      assert.ok(sample.cases.secret.disallowedToolsList.includes(toolName), `${toolName} must be denied`);
    }
  });

  it('can still allow web tools while denying local native tools', () => {
    assert.deepEqual(sample.cases.secretWithWeb.allowedToolsList, ['WebSearch', 'WebFetch']);
    for (const toolName of sample.nativeTools) {
      assert.ok(sample.cases.secretWithWeb.disallowedToolsList.includes(toolName), `${toolName} must be denied`);
    }
  });

  it('restores legacy native-tool behavior only under the unsafe override', () => {
    assert.ok(sample.cases.override.allowedToolsList.includes('Bash'));
    assert.ok(sample.cases.override.allowedToolsList.includes('Read'));
    assert.ok(!sample.cases.override.disallowedToolsList.includes('Bash'));
  });

  it('preserves scheduled forum native-tool denial', () => {
    assert.deepEqual(sample.cases.forum.allowedToolsList, []);
    for (const toolName of sample.nativeTools) {
      assert.ok(sample.cases.forum.disallowedToolsList.includes(toolName), `${toolName} must be denied`);
    }
  });
});
