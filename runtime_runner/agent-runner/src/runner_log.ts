/**
 * Shared stderr logger for the in-container agent runner. Kept in its own
 * module so every extracted adapter helper logs with the same `[agent-runner]`
 * prefix without re-declaring it (and without a circular import back to
 * index.ts).
 */
export function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}
