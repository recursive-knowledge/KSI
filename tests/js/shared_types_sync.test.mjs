/**
 * Sync guard for the two deliberately-duplicated shared_types.ts copies:
 *
 *   - runtime_runner/src/shared_types.ts            (host-side compilation unit)
 *   - runtime_runner/agent-runner/src/shared_types.ts (container-side)
 *
 * Both files state "Any change must be applied to both copies", but the
 * discipline is manual and has already failed once: issue #731 — the
 * agent-runner copy grew `phase1_reflection_token_usage` while the host
 * copy didn't, forcing main.ts to re-declare the output shape inline and
 * extract the field via `as`-casts.
 *
 * This test derives the set of `export interface` names from EACH file,
 * asserts those name-sets are identical (catching an interface added to
 * only one copy), then brace-extracts every interface body from BOTH
 * files as text and asserts the TOP-LEVEL field-name sets are identical.
 * It deliberately does NOT compare docstrings or field types/bodies —
 * full byte-identity of the two copies is pinned separately by
 * tests/runtime/test_shared_types_mirror.py.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');

const copies = [
  {
    label: 'runtime_runner/src/shared_types.ts',
    src: fs.readFileSync(
      path.join(repoRoot, 'runtime_runner', 'src', 'shared_types.ts'),
      'utf-8',
    ),
  },
  {
    label: 'runtime_runner/agent-runner/src/shared_types.ts',
    src: fs.readFileSync(
      path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'shared_types.ts'),
      'utf-8',
    ),
  },
];

/** Remove block and line comments so brace counting and field parsing
 *  can't be confused by braces or field-like text inside docstrings. */
function stripComments(src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

/** Return the body text between the braces of `interface <name> { ... }`,
 *  matching nested braces. Input must already be comment-stripped. */
function extractInterfaceBody(src, name, label) {
  const m = src.match(new RegExp(`interface\\s+${name}\\s*\\{`));
  assert.ok(m, `${label} must declare interface ${name}`);
  const start = m.index + m[0].length;
  let depth = 1;
  let i = start;
  while (i < src.length && depth > 0) {
    if (src[i] === '{') depth += 1;
    else if (src[i] === '}') depth -= 1;
    i += 1;
  }
  assert.equal(depth, 0, `${label}: unbalanced braces in interface ${name}`);
  return src.slice(start, i - 1);
}

/** Parse the names of TOP-LEVEL fields only (nested object-literal fields
 *  like `phase1_reflection_meta.enabled` are skipped via depth tracking). */
function topLevelFieldNames(body) {
  const names = [];
  let depth = 0;
  for (const line of body.split('\n')) {
    if (depth === 0) {
      const m = line.match(/^\s*(?:readonly\s+)?['"]?(\w+)['"]?\??:/);
      if (m) names.push(m[1]);
    }
    for (const ch of line) {
      if (ch === '{') depth += 1;
      else if (ch === '}') depth -= 1;
    }
  }
  return names;
}

/** Names of every `export interface` in (already comment-stripped) source. */
function exportedInterfaceNames(src) {
  return [...src.matchAll(/export interface (\w+)/g)].map((m) => m[1]);
}

const interfaceNamesPerCopy = copies.map(({ label, src }) => {
  const names = exportedInterfaceNames(stripComments(src));
  assert.ok(
    names.length > 0,
    `${label}: parsed zero exported interfaces — the extraction regex may have rotted`,
  );
  return { label, names };
});

describe('shared_types.ts copies stay field-synchronized (#731)', () => {
  it('both copies export the same set of interfaces', () => {
    const [host, agent] = interfaceNamesPerCopy;
    assert.deepEqual(
      [...host.names].sort(),
      [...agent.names].sort(),
      `exported interface sets diverged between ${host.label} and ` +
        `${agent.label}. Both files say "Any change must be applied to ` +
        'both copies" — add/remove the interface in the other copy too.',
    );
  });

  for (const interfaceName of interfaceNamesPerCopy[0].names) {
    it(`${interfaceName} has identical top-level field names in both copies`, () => {
      const fieldSets = copies.map(({ label, src }) => {
        const body = extractInterfaceBody(stripComments(src), interfaceName, label);
        const names = topLevelFieldNames(body);
        assert.ok(
          names.length > 0,
          `${label}: parsed zero fields from interface ${interfaceName} — ` +
            'the extraction regex may have rotted',
        );
        return { label, names };
      });
      const [host, agent] = fieldSets;
      assert.deepEqual(
        [...host.names].sort(),
        [...agent.names].sort(),
        `interface ${interfaceName} field names diverged between ${host.label} ` +
          `and ${agent.label}. Both files say "Any change must be applied to ` +
          'both copies" — apply your field addition/removal to the other copy too.',
      );
    });
  }
});

describe('sanity: the parser sees known fields', () => {
  it('ContainerOutput includes phase1_reflection_token_usage in both copies', () => {
    for (const { label, src } of copies) {
      const body = extractInterfaceBody(stripComments(src), 'ContainerOutput', label);
      const names = topLevelFieldNames(body);
      assert.ok(
        names.includes('phase1_reflection_token_usage'),
        `${label}: ContainerOutput must carry phase1_reflection_token_usage (#731)`,
      );
      assert.ok(
        names.includes('status'),
        `${label}: ContainerOutput must carry status`,
      );
    }
  });
});
