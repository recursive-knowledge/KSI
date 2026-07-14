/**
 * Container-side helpers for the host<->container barrier protocol.
 *
 * See `src/ksi/runtime/barrier.py` for the canonical protocol description.
 *
 * Sentinel naming (relative to /workspace/task):
 *   - sentinel:  .barrier.<name>.<agentId>.ready    (container writes)
 *   - response:  .barrier.<name>.<agentId>.response (host writes)
 *
 * The container side:
 *   1. Writes a sentinel JSON file atomically (tmp + rename) when it has
 *      reached a barrier point.
 *   2. Polls for the response file, with a poll interval matching the
 *      existing IPC poll cadence and a configurable timeout.
 *   3. On timeout, returns null so callers can degrade gracefully.
 *
 * No business logic lives here — the helpers are deliberately thin so
 * future barrier types (Phase 3 R0->R1, etc.) can layer their semantics
 * on top.
 */

import fs from 'fs';
import path from 'path';

export const DEFAULT_BARRIER_POLL_MS = 500;

function sanitizeNamePart(value: string): string {
  return (value || '').replace(/[^A-Za-z0-9_-]/g, '_');
}

export function sentinelFilename(name: string, agentId: string): string {
  return `.barrier.${sanitizeNamePart(name)}.${sanitizeNamePart(agentId)}.ready`;
}

export function responseFilename(name: string, agentId: string): string {
  return `.barrier.${sanitizeNamePart(name)}.${sanitizeNamePart(agentId)}.response`;
}

export function sentinelPath(workspaceDir: string, name: string, agentId: string): string {
  return path.join(workspaceDir, sentinelFilename(name, agentId));
}

export function responsePath(workspaceDir: string, name: string, agentId: string): string {
  return path.join(workspaceDir, responseFilename(name, agentId));
}

/**
 * Atomically write the sentinel file with a JSON payload describing the
 * barrier event the host should respond to.
 */
export function writeSentinelFile(
  workspaceDir: string,
  name: string,
  agentId: string,
  payload: Record<string, unknown>,
): string {
  const target = sentinelPath(workspaceDir, name, agentId);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const tmp = `${target}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(payload), { encoding: 'utf-8' });
  fs.renameSync(tmp, target);
  return target;
}

/**
 * Poll for a barrier response file written by the host. Returns the parsed
 * JSON payload on success, or null if the timeout elapsed before the file
 * appeared. Callers MUST treat null as "graceful degrade" — barriers are
 * advisory, not load-bearing.
 *
 * The function deliberately swallows JSON-parse errors and returns null;
 * a partially-written response (which the atomic-write helper on the host
 * side prevents in normal operation) is treated identically to a missing
 * one. The caller can log a warning if needed via the `onAttempt` hook.
 */
export async function waitForBarrierFile(
  filePath: string,
  timeoutMs: number,
  options: {
    pollIntervalMs?: number;
    onAttempt?: (info: { elapsedMs: number; exists: boolean }) => void;
  } = {},
): Promise<Record<string, unknown> | null> {
  const pollMs = Math.max(50, options.pollIntervalMs ?? DEFAULT_BARRIER_POLL_MS);
  const start = Date.now();
  while (true) {
    let exists = false;
    try {
      exists = fs.existsSync(filePath);
    } catch {
      exists = false;
    }
    if (options.onAttempt) {
      try {
        options.onAttempt({ elapsedMs: Date.now() - start, exists });
      } catch {
        // never let an instrumentation hook break the wait loop
      }
    }
    if (exists) {
      try {
        const raw = fs.readFileSync(filePath, 'utf-8');
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          // Best-effort cleanup so a re-entry on the same workspace doesn't
          // see the stale response. Failure to unlink is non-fatal.
          try { fs.unlinkSync(filePath); } catch { /* ignore */ }
          return parsed as Record<string, unknown>;
        }
      } catch {
        // fall through and treat as missing — host will overwrite atomically
      }
    }
    if (Date.now() - start >= timeoutMs) {
      return null;
    }
    await new Promise<void>((resolve) => setTimeout(resolve, pollMs));
  }
}
