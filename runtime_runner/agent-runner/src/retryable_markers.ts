/**
 * Retryable-error marker phrases — TypeScript view of the single source of
 * truth at ``runtime_runner/shared/retryable_markers.json``. Exposes the
 * ``stream_race`` category (emitted by index.ts so the orchestrator recognises
 * an SDK stream-race as retryable) and the ``non_retryable`` category (whose
 * prompt-rejection phrases the OpenAI adapter matches; see openai.ts).
 *
 * The Python orchestrator (``src/kcsi/orchestrator/engine.py`` via
 * ``src/kcsi/errors.py::load_retryable_markers``) classifies a task error as
 * transient (retry) by case-insensitive substring match against these phrases.
 * The agent-runner must therefore EMIT these exact phrases (as a substring of
 * its error/diagnostic text) for the orchestrator to recognise an SDK
 * stream-race as retryable. Historically these were hardcoded on both sides
 * and drifted; now both read the same JSON. See issue #648.
 *
 * Loading strategy (mirrors the Python ``load_retryable_markers`` fallback):
 * read ``retryable_markers.json`` at runtime from a small set of candidate
 * paths (next to the compiled module, the source tree, or the repo's
 * ``runtime_runner/shared`` dir). If none is readable, fall back to a vendored
 * copy below so the runner can never crash on a missing/garbled file. Reading
 * via ``fs`` (rather than a TS ``import ... json``) avoids the tsc
 * resolveJsonModule emit/copy pitfall where the JSON is not written to outDir.
 *
 * The vendored fallbacks MUST stay byte-identical to their JSON categories;
 * tests/test_retryable_markers.py pins the JSON, and the agent-runner is built
 * from the same shared file at image-build time.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Vendored fallbacks — keep in lockstep with the matching categories in
// runtime_runner/shared/retryable_markers.json. The Python side pins JSON<->
// vendored parity in tests/test_retryable_markers.py; the TS copies are built
// from the same shared file at image-build time.
const VENDORED: Readonly<Record<string, readonly string[]>> = {
  // categories.stream_race
  stream_race: [
    'sdk query loop drained',
    'sdk query iterator threw',
    'sdk emitted an empty result event',
    'silent agent-runner failure',
  ],
  // categories.non_retryable — the deterministic refusal/parse markers the
  // orchestrator (engine.py) treats as non-retryable. The OpenAI adapter reuses
  // the two prompt-rejection phrases ("invalid prompt", "usage policy") so its
  // compact-retry gate can't silently desync from the shared list (#648).
  non_retryable: [
    'invalid prompt',
    'usage policy',
    'flagged as potentially violating',
    'no patch',
    'missing report',
    'parse_error',
  ],
};

function loadCategory(category: string): readonly string[] {
  let hereDir: string;
  try {
    hereDir = path.dirname(fileURLToPath(import.meta.url));
  } catch {
    hereDir = process.cwd();
  }
  const candidates = [
    path.join(hereDir, 'retryable_markers.json'),
    path.join(hereDir, '..', 'shared', 'retryable_markers.json'),
    path.join(hereDir, '..', '..', 'shared', 'retryable_markers.json'),
    path.join(hereDir, '..', '..', '..', 'runtime_runner', 'shared', 'retryable_markers.json'),
  ];
  for (const candidate of candidates) {
    try {
      const raw = fs.readFileSync(candidate, 'utf-8');
      const parsed = JSON.parse(raw) as {
        categories?: Record<string, unknown>;
      };
      const list = parsed?.categories?.[category];
      if (Array.isArray(list) && list.length > 0) {
        return Object.freeze(list.map((m) => String(m)));
      }
    } catch {
      // try next candidate
    }
  }
  return Object.freeze([...VENDORED[category]]);
}

/**
 * The ``stream_race`` category: phrases the orchestrator treats as a retryable
 * SDK stream-race / silent-failure signature. Order matches the JSON.
 */
export const STREAM_RACE_MARKERS: readonly string[] = loadCategory('stream_race');

/**
 * The ``non_retryable`` category: deterministic refusal/parse markers the
 * orchestrator treats as non-retryable. Exposed so the OpenAI adapter can match
 * the shared prompt-rejection phrases instead of hardcoding its own copy.
 */
export const NON_RETRYABLE_MARKERS: readonly string[] = loadCategory('non_retryable');

/**
 * Resolve a single marker by its leading token so call sites read
 * intention-revealingly and a JSON reword is caught loudly rather than silently
 * diverging. Throws if the expected marker is absent so a mismatched build
 * fails loudly instead of emitting/matching a non-existent phrase.
 */
function requireMarker(
  markers: readonly string[],
  category: string,
  startsWith: string,
): string {
  const found = markers.find((m) =>
    m.toLowerCase().startsWith(startsWith.toLowerCase()),
  );
  if (!found) {
    throw new Error(
      `retryable_markers.json: no ${category} marker starting with ` +
        `"${startsWith}" (have: ${markers.join(', ')})`,
    );
  }
  return found;
}

/** "silent agent-runner failure" */
export const MARKER_SILENT_AGENT_RUNNER_FAILURE = requireMarker(
  STREAM_RACE_MARKERS,
  'stream_race',
  'silent agent-runner failure',
);
/** "sdk emitted an empty result event" */
export const MARKER_SDK_EMPTY_RESULT_EVENT = requireMarker(
  STREAM_RACE_MARKERS,
  'stream_race',
  'sdk emitted an empty result event',
);
/** "sdk query loop drained" */
export const MARKER_SDK_QUERY_LOOP_DRAINED = requireMarker(
  STREAM_RACE_MARKERS,
  'stream_race',
  'sdk query loop drained',
);
/** "sdk query iterator threw" */
export const MARKER_SDK_QUERY_ITERATOR_THREW = requireMarker(
  STREAM_RACE_MARKERS,
  'stream_race',
  'sdk query iterator threw',
);

/** "invalid prompt" — shared non_retryable phrase used by the OpenAI adapter. */
export const MARKER_INVALID_PROMPT = requireMarker(
  NON_RETRYABLE_MARKERS,
  'non_retryable',
  'invalid prompt',
);
/** "usage policy" — shared non_retryable phrase used by the OpenAI adapter. */
export const MARKER_USAGE_POLICY = requireMarker(
  NON_RETRYABLE_MARKERS,
  'non_retryable',
  'usage policy',
);

/**
 * Render a (lowercase) marker for human-facing emission while preserving the
 * exact substring the orchestrator matches (case-insensitive). Reproduces the
 * historical visible casing so this refactor is byte-for-byte non-behavioural:
 *   - a leading "sdk " token becomes "SDK " ("sdk query loop drained" ->
 *     "SDK query loop drained")
 *   - otherwise the first letter is capitalised ("silent agent-runner failure"
 *     -> "Silent agent-runner failure")
 * Lowercasing the result yields the original marker, so substring matching in
 * engine.py is unaffected regardless of the visible casing.
 */
export function emitPhrase(marker: string): string {
  if (marker.toLowerCase().startsWith('sdk ')) {
    return 'SDK ' + marker.slice(4);
  }
  return marker.charAt(0).toUpperCase() + marker.slice(1);
}
