// Parity guard for the TypeScript view of the shared retryable-error markers.
//
// Issue #648 centralised the markers in runtime_runner/shared/retryable_markers.json,
// consumed by both Python (kcsi/errors.py, pinned by tests/test_retryable_markers.py)
// and TypeScript (runtime_runner/agent-runner/src/retryable_markers.ts). The TS side
// keeps an inline VENDORED fallback (it reads the JSON via fs at runtime, but must not
// crash if the file is missing). That fallback is a hand-maintained copy and can silently
// drift from the JSON — the exact failure #648 exists to prevent. The Python vendored copy
// is pinned by test_vendored_fallback_matches_json; this is the symmetric guard for TS.
//
// Runs under `node tests/js/*.test.mjs` (CI: node 20). It cannot import the .ts module
// directly (no type stripping on node 20), so it reads the source as text — the same
// pattern as container_runner_env.test.mjs et al. — and additionally extracts and EXECUTES
// the pure emitPhrase() body to pin the casing round-trip.
import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');

const json = JSON.parse(
  fs.readFileSync(
    path.join(repoRoot, 'runtime_runner', 'shared', 'retryable_markers.json'),
    'utf-8',
  ),
);
const tsSource = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'retryable_markers.ts'),
  'utf-8',
);

// Isolate the `const VENDORED = { ... };` object literal so category lookups below
// can't accidentally match `stream_race?` in loadCategory()'s type annotation etc.
function vendoredBlock() {
  const start = tsSource.indexOf('const VENDORED');
  assert.ok(start >= 0, 'const VENDORED not found in retryable_markers.ts');
  const end = tsSource.indexOf('\n};', start);
  assert.ok(end > start, 'end of VENDORED object literal not found');
  return tsSource.slice(start, end + 3);
}

// Extract an ordered array of single-quoted string literals for `name: [ ... ]`.
function vendoredCategory(block, name) {
  const m = block.match(new RegExp(`${name}:\\s*\\[([\\s\\S]*?)\\]`));
  assert.ok(m, `VENDORED.${name} array not found in retryable_markers.ts`);
  return [...m[1].matchAll(/'((?:[^'\\]|\\.)*)'/g)].map((x) => x[1]);
}

describe('retryable_markers.ts <-> retryable_markers.json parity', () => {
  const block = vendoredBlock();
  const vendoredCategoryNames = [...block.matchAll(/^\s{2}(\w+):\s*\[/gm)].map((m) => m[1]);

  it('VENDORED has at least the categories it is expected to mirror', () => {
    assert.ok(
      vendoredCategoryNames.includes('stream_race') && vendoredCategoryNames.includes('non_retryable'),
      `VENDORED categories were ${JSON.stringify(vendoredCategoryNames)}`,
    );
  });

  it('every VENDORED category is byte-identical (and same order) to the JSON', () => {
    for (const name of vendoredCategoryNames) {
      assert.ok(json.categories[name], `JSON has no category '${name}' but TS vendors it`);
      assert.deepEqual(
        vendoredCategory(block, name),
        json.categories[name],
        `VENDORED.${name} drifted from retryable_markers.json — update both`,
      );
    }
  });

  it('every loadCategory(...) the TS reads has a vendored fallback (no spread-of-undefined crash)', () => {
    const loaded = [...tsSource.matchAll(/loadCategory\(\s*'([^']+)'\s*\)/g)].map((m) => m[1]);
    assert.ok(loaded.length >= 2, `expected loadCategory calls, found ${JSON.stringify(loaded)}`);
    for (const name of loaded) {
      assert.ok(json.categories[name], `loadCategory('${name}') but JSON has no such category`);
      assert.ok(
        vendoredCategoryNames.includes(name),
        `loadCategory('${name}') has no VENDORED.${name} fallback — runtime would throw if the JSON were unreadable`,
      );
    }
  });

  it('every requireMarker(...) prefix exists in its JSON category (a reword fails CI, not container runtime)', () => {
    // requireMarker(LIST, 'category', 'prefix')
    // requireMarker(LIST, 'category', 'prefix') — calls span multiple lines and
    // carry a trailing comma before the close paren.
    const calls = [
      ...tsSource.matchAll(/requireMarker\(\s*\w+\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*,?\s*\)/g),
    ].map((m) => ({ category: m[1], prefix: m[2] }));
    assert.ok(calls.length >= 6, `expected requireMarker call sites, found ${calls.length}`);
    for (const { category, prefix } of calls) {
      const markers = json.categories[category];
      assert.ok(markers, `requireMarker references unknown category '${category}'`);
      assert.ok(
        markers.some((mk) => mk.toLowerCase().startsWith(prefix.toLowerCase())),
        `requireMarker('${category}','${prefix}') has no matching marker in the JSON — `
          + `a reword here throws at container load; fix the JSON or the call.`,
      );
    }
  });
});

describe('emitPhrase casing round-trip', () => {
  // Extract and execute the REAL emitPhrase body (pure string logic, no types in body).
  const m = tsSource.match(/export function emitPhrase\(marker: string\): string \{([\s\S]*?)\n\}/);
  assert.ok(m, 'emitPhrase not found in retryable_markers.ts');
  // eslint-disable-next-line no-new-func
  const emitPhrase = new Function('marker', m[1]);

  it('lowercasing the emitted phrase yields the original marker (substring match unaffected)', () => {
    for (const marker of json.categories.stream_race) {
      assert.equal(
        emitPhrase(marker).toLowerCase(),
        marker,
        `emitPhrase('${marker}') does not lowercase back to the marker`,
      );
    }
  });

  it('reproduces the historical visible casing', () => {
    assert.equal(emitPhrase('sdk query loop drained'), 'SDK query loop drained');
    assert.equal(emitPhrase('sdk query iterator threw'), 'SDK query iterator threw');
    assert.equal(emitPhrase('sdk emitted an empty result event'), 'SDK emitted an empty result event');
    assert.equal(emitPhrase('silent agent-runner failure'), 'Silent agent-runner failure');
  });
});
