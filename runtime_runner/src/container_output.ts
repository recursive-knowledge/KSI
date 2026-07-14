/**
 * Parsing of the container stdout envelope for the shared container runtime.
 *
 * The in-container agent-runner wraps its JSON result in OUTPUT_START/END
 * sentinel markers; these helpers locate, validate, and extract the
 * {@link ContainerOutput} payload from the (possibly log-interleaved) stdout
 * stream. Pure functions with no process/IO dependencies.
 */
import { ContainerOutput } from './shared_types.js';

// Sentinel markers for robust output parsing (must match agent-runner)
export const OUTPUT_START_MARKER = '---KSI_OUTPUT_START---';
export const OUTPUT_END_MARKER = '---KSI_OUTPUT_END---';

export function outputHasExpectedNonce(
  output: ContainerOutput,
  expectedNonce: string,
): boolean {
  return Boolean(expectedNonce) && output.protocolNonce === expectedNonce;
}

export const CONTAINER_OUTPUT_STATUSES = new Set<ContainerOutput['status']>([
  'success',
  'error',
  'recovered_from_session',
]);

/**
 * Structural guard for the container stdout envelope. The previous
 * `JSON.parse(...) as ContainerOutput` casts accepted any valid JSON
 * (e.g. `42`, `null`, or an object with no/invalid `status`) and let it flow
 * downstream as a typed `ContainerOutput`, where `.status`/`.result` then read
 * back as `undefined`. Require a real object carrying a known `status` AND the
 * required `result` field (`string | null`) — every legitimate emitter sets
 * `result`, so an object missing it is malformed and must fall through to the
 * line-scan / recovery path rather than narrow to a valid envelope.
 */
export function isContainerOutput(value: unknown): value is ContainerOutput {
  if (value === null || typeof value !== 'object') {
    return false;
  }
  const status = (value as { status?: unknown }).status;
  if (typeof status !== 'string' || !CONTAINER_OUTPUT_STATUSES.has(status as ContainerOutput['status'])) {
    return false;
  }
  // `result` is a required field (`string | null`); reject objects that omit it
  // or carry a wrong type.
  const result = (value as { result?: unknown }).result;
  return result === null || typeof result === 'string';
}

export function parseContainerOutputBlock(
  block: string,
  expectedNonce?: string,
): ContainerOutput | null {
  const text = String(block || '').trim();
  if (!text) {
    return null;
  }

  try {
    const parsed: unknown = JSON.parse(text);
    if (isContainerOutput(parsed)) {
      if (expectedNonce && !outputHasExpectedNonce(parsed, expectedNonce)) {
        return null;
      }
      return parsed;
    }
    // Valid JSON but not a ContainerOutput shape — fall through to line scan.
  } catch {
    // Some SDK/runtime logs can interleave inside the marker block.
    // Fall back to scanning for the JSON line instead of dropping the result.
  }

  for (const line of text.split('\n')) {
    const candidate = line.trim();
    if (!candidate.startsWith('{')) {
      continue;
    }
    try {
      const parsed: unknown = JSON.parse(candidate);
      if (!isContainerOutput(parsed)) {
        continue;
      }
      if (expectedNonce && !outputHasExpectedNonce(parsed, expectedNonce)) {
        continue;
      }
      return parsed;
    } catch {
      // Keep scanning.
    }
  }
  return null;
}

export function extractLastContainerOutput(
  stdoutText: string,
  expectedNonce: string,
): ContainerOutput | null {
  const text = String(stdoutText || '');
  let cursor = 0;
  let lastParsed: ContainerOutput | null = null;
  while (cursor < text.length) {
    const startIdx = text.indexOf(OUTPUT_START_MARKER, cursor);
    if (startIdx === -1) {
      break;
    }
    const endIdx = text.indexOf(
      OUTPUT_END_MARKER,
      startIdx + OUTPUT_START_MARKER.length,
    );
    if (endIdx === -1) {
      break;
    }
    const block = text
      .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
      .trim();
    const parsed = parseContainerOutputBlock(block, expectedNonce);
    if (parsed) {
      lastParsed = parsed;
    }
    cursor = endIdx + OUTPUT_END_MARKER.length;
  }
  return lastParsed;
}
