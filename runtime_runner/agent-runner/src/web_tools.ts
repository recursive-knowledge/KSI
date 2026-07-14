/**
 * Web-tool gating for the in-container Claude agent-runner (issue #666).
 *
 * Extracted from index.ts so the gating decision has a single source of truth
 * that is BEHAVIORALLY testable: index.ts imports the container-only
 * @anthropic-ai/claude-agent-sdk at module top, so a test cannot `import`
 * index.ts without the SDK installed. This module has NO SDK dependency, so
 * tests/js/web_tools_gating.test.mjs can import the real functions (via tsx)
 * and exercise the actual returned tool arrays — not just pin source text.
 *
 * Policy: native web tools (WebSearch/WebFetch) are OFF by default for ALL
 * benchmark tasks. An operator opts a specific run in with
 * `KCSI_ALLOW_WEB_TOOLS=1`. ARC stays strictly offline regardless of the
 * flag (callers AND-gate with `!isOffline`).
 */

/** The native web tools gated by this module. */
export const WEB_TOOLS: readonly string[] = ['WebSearch', 'WebFetch'];

/**
 * Whether native web tools (WebSearch/WebFetch) may be offered to the Claude
 * agent on this run, based solely on the `KCSI_ALLOW_WEB_TOOLS` flag.
 *
 * Truthiness mirrors the Python `_is_enabled_env` helper
 * (src/kcsi/runtime/container_host.py): any non-empty value that is not one of
 * 0/false/no/off counts as enabled. ARC offlineness is applied by the caller,
 * not here.
 */
export function isWebToolsAllowed(
  sdkEnv: Record<string, string | undefined>,
): boolean {
  const raw = String(sdkEnv.KCSI_ALLOW_WEB_TOOLS ?? '').trim().toLowerCase();
  if (!raw) return false;
  return !['0', 'false', 'no', 'off'].includes(raw);
}

/** Resolved web-tool gating for a single run. */
export interface WebToolGating {
  /** Flag truthiness, before the ARC override. */
  webToolsAllowedByFlag: boolean;
  /** Effective state: flag on AND not ARC. */
  webToolsEnabled: boolean;
  /** Entry to splice into `allowedTools` ([] or the web tools). */
  allowlistWebTools: string[];
  /**
   * Entry for `disallowedTools` — THE load-bearing denial. The claude_code
   * preset re-adds web tools to the model's context regardless of
   * `allowedTools`; only `disallowedTools` removes them. So this is
   * `['WebSearch','WebFetch']` whenever web tools are not enabled (ARC always,
   * and every benchmark unless the operator sets KCSI_ALLOW_WEB_TOOLS=1).
   */
  disallowedWebTools: string[];
  /** Self-documenting reason string for the status log line. */
  reason: string;
}

/**
 * Resolve the full web-tool gating for a run. `isOffline` is true for ARC,
 * which forces web tools off regardless of the flag.
 */
export function buildWebToolGating(
  sdkEnv: Record<string, string | undefined>,
  isOffline: boolean,
): WebToolGating {
  const webToolsAllowedByFlag = isWebToolsAllowed(sdkEnv);
  const webToolsEnabled = webToolsAllowedByFlag && !isOffline;
  const reason = isOffline
    ? 'ARC offline benchmark (web tools always denied)'
    : webToolsAllowedByFlag
      ? 'KCSI_ALLOW_WEB_TOOLS=1'
      : 'default-off (set KCSI_ALLOW_WEB_TOOLS=1 to enable; issue #666)';
  return {
    webToolsAllowedByFlag,
    webToolsEnabled,
    allowlistWebTools: webToolsEnabled ? [...WEB_TOOLS] : [],
    disallowedWebTools: webToolsEnabled ? [] : [...WEB_TOOLS],
    reason,
  };
}

