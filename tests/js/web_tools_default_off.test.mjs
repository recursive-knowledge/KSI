/**
 * Issue #666 — default-OFF WebSearch/WebFetch for benchmark tasks.
 *
 * WebSearch/WebFetch are a benchmark-solution leak vector: on non-ARC
 * benchmarks (swebench_pro / polyglot / terminal_bench_2) an agent could fetch
 * the upstream fix/issue/PR instead of solving the task, and only Claude got
 * these tools (GPT had none) — a provider asymmetry on top of the
 * contamination risk. This fix makes web tools default OFF for ALL benchmark
 * tasks; an operator opts a run in with KSI_ALLOW_WEB_TOOLS=1. ARC stays
 * strictly offline regardless of the flag.
 *
 * These are source-text invariant pins. CI runs `node tests/js/*.test.mjs`
 * with only runtime_runner deps installed; index.ts imports the
 * @anthropic-ai/claude-agent-sdk (container-only) at module top, so we cannot
 * `import` it here. We therefore pin the gating in source and additionally
 * exercise the pure truthy semantics by re-implementing them.
 *
 * They guarantee:
 *   1. The web-tool opt-in is GATED behind KSI_ALLOW_WEB_TOOLS (default OFF).
 *   2. Web tools are added to allowedTools ONLY when enabled (flag on + non-ARC).
 *   3. The DENIAL is enforced via disallowedTools (not mere omission) whenever
 *      web tools are not enabled — the claude_code preset would otherwise
 *      re-add them to the model's context. This is the critical correctness
 *      point.
 *   4. ARC always denies web tools regardless of the flag.
 *   5. The flag is forwarded host -> container (container_host.py +
 *      container_runner.ts).
 *   6. A self-documenting trace line reports web-tool status + flag source.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');

// The web-tool gating wiring (buildWebToolGating, the allow/disallow lists)
// lives in query_config.ts; the SDK query that passes disallowedToolsList
// lives in query_runner.ts. Concatenate so these assertions follow the code.
const index = [
  'index.ts',
  'query_config.ts',
  'query_runner.ts',
].map((name) => fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', name),
  'utf-8',
)).join('\n');
const webTools = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'web_tools.ts'),
  'utf-8',
);
const containerRunner = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'src', 'container_args.ts'),
  'utf-8',
);
const containerHost = fs.readFileSync(
  path.join(repoRoot, 'src', 'ksi', 'runtime', 'container_host.py'),
  'utf-8',
);

// Re-implementation of the exported isWebToolsAllowed truthy semantics for the
// always-on (no-tsx) smoke check below. The REAL exported function in
// web_tools.ts is exercised behaviorally (via tsx) in
// tests/js/web_tools_gating.test.mjs; here we only pin the source contract.
function isWebToolsAllowed(env) {
  const raw = String(env.KSI_ALLOW_WEB_TOOLS ?? '').trim().toLowerCase();
  if (!raw) return false;
  return !['0', 'false', 'no', 'off'].includes(raw);
}

describe('Web tools — flag truthiness (default OFF)', () => {
  it('treats unset / empty as OFF', () => {
    assert.equal(isWebToolsAllowed({}), false);
    assert.equal(isWebToolsAllowed({ KSI_ALLOW_WEB_TOOLS: '' }), false);
    assert.equal(isWebToolsAllowed({ KSI_ALLOW_WEB_TOOLS: '   ' }), false);
  });

  it('treats false-y tokens as OFF', () => {
    for (const v of ['0', 'false', 'no', 'off', 'FALSE', 'Off', 'NO']) {
      assert.equal(
        isWebToolsAllowed({ KSI_ALLOW_WEB_TOOLS: v }),
        false,
        `expected ${v} to be OFF`,
      );
    }
  });

  it('treats 1/true and other non-empty values as ON', () => {
    for (const v of ['1', 'true', 'yes', 'on', 'TRUE']) {
      assert.equal(
        isWebToolsAllowed({ KSI_ALLOW_WEB_TOOLS: v }),
        true,
        `expected ${v} to be ON`,
      );
    }
  });
});

describe('Web tools — web_tools.ts gating logic (issue #666)', () => {
  it('exports isWebToolsAllowed reading KSI_ALLOW_WEB_TOOLS with false-y off-list', () => {
    assert.match(webTools, /export function isWebToolsAllowed\(/);
    assert.match(webTools, /KSI_ALLOW_WEB_TOOLS/);
    assert.match(webTools, /\['0', 'false', 'no', 'off'\]/);
  });

  it('derives webToolsEnabled = flag AND not-ARC', () => {
    assert.match(webTools, /export function buildWebToolGating\(/);
    assert.match(webTools, /const webToolsAllowedByFlag = isWebToolsAllowed\(sdkEnv\)/);
    // ARC always wins: the effective enable must AND-gate on !isOffline.
    assert.match(
      webTools,
      /const webToolsEnabled = webToolsAllowedByFlag && !isOffline/,
    );
  });

  it('adds web tools to allowedTools ONLY when webToolsEnabled', () => {
    // allowlistWebTools is the web tools only when enabled, else [].
    assert.match(
      webTools,
      /allowlistWebTools: webToolsEnabled \? \[\.\.\.WEB_TOOLS\] : \[\]/,
    );
  });

  it('CRITICAL: denies web tools via disallowedTools whenever not enabled', () => {
    // The main query runs with the claude_code preset, which loads the full
    // default tool surface (WebSearch/WebFetch included) regardless of
    // allowedTools. Only disallowedTools removes them from context. The denial
    // must therefore carry the web tools whenever NOT enabled.
    assert.match(
      webTools,
      /disallowedWebTools: webToolsEnabled \? \[\] : \[\.\.\.WEB_TOOLS\]/,
    );
    assert.match(webTools, /export const WEB_TOOLS: readonly string\[\] = \['WebSearch', 'WebFetch'\]/);
  });
});

describe('Web tools — index.ts wiring (issue #666)', () => {
  it('imports the SDK-free gating helper instead of inlining it', () => {
    assert.match(index, /import \{ buildWebToolGating \} from '\.\/web_tools\.js'/);
    // The old always-on-for-non-ARC allowlist pattern must be gone.
    assert.doesNotMatch(
      index,
      /\.\.\.\(isOffline \? \[\] : \['WebSearch', 'WebFetch'\]\)/,
    );
    // And the old inlined disallowed ternary must be gone (now from gating).
    assert.doesNotMatch(
      index,
      /const disallowedToolsList: string\[\] = isOffline\s*\?\s*\['WebSearch', 'WebFetch'\]\s*:\s*\[\]/s,
    );
  });

  it('builds the gating from sdkEnv + isOffline and uses its arrays', () => {
    assert.match(index, /const webToolGating = buildWebToolGating\(sdkEnv, isOffline\)/);
    assert.match(index, /\.\.\.webToolGating\.allowlistWebTools/);
    assert.match(index, /\.\.\.webToolGating\.disallowedWebTools/);
  });

  it('passes disallowedToolsList to the SDK query so the denial reaches the CLI', () => {
    assert.match(
      index,
      /disallowedTools: disallowedToolsList\.length > 0 \? disallowedToolsList : undefined/,
    );
  });

  it('logs a self-documenting web-tool status line with the flag source', () => {
    assert.match(index, /Web tools \(WebSearch\/WebFetch\):/);
    assert.match(index, /webToolGating\.reason/);
    // ARC reason, flag-on reason, default-off reason all present (in web_tools.ts).
    assert.match(webTools, /ARC offline benchmark \(web tools always denied\)/);
    assert.match(webTools, /KSI_ALLOW_WEB_TOOLS=1/);
    assert.match(webTools, /default-off/);
  });
});

describe('Web tools — host -> container threading (issue #666)', () => {
  it('container_args.ts forwards KSI_ALLOW_WEB_TOOLS into the container env', () => {
    assert.match(containerRunner, /'KSI_ALLOW_WEB_TOOLS'/);
  });

  it('container_host.py threads KSI_ALLOW_WEB_TOOLS from host env into the runner env', () => {
    assert.match(containerHost, /KSI_ALLOW_WEB_TOOLS/);
    // Threaded via the same setdefault-from-os.environ pattern as the other
    // runtime knobs (does not clobber a provider-profile-supplied value).
    assert.match(
      containerHost,
      /for web_key in \("KSI_ALLOW_WEB_TOOLS",\):/,
    );
  });
});
