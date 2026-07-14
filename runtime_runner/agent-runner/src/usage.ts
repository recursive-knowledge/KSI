/**
 * Token-usage accounting for Claude Agent SDK messages.
 *
 * The SDK surfaces usage at two different nesting levels depending on the
 * message type:
 *   - Assistant messages: `message.message.usage` (per-turn delta)
 *   - Result messages:    `message.usage`         (final session aggregate)
 * {@link extractUsageFromSdkMessage} returns zeros for missing/malformed input
 * so callers can accumulate without null-checks.
 */
export interface UsageDelta {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

export function extractUsageFromSdkMessage(message: unknown): UsageDelta {
  const zero: UsageDelta = {
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  };
  if (!message || typeof message !== 'object') return zero;
  const msg = message as Record<string, unknown>;
  // Assistant messages nest under .message.usage; result messages expose .usage
  // at the top level. Check both and return whichever is populated.
  const nestedMessage = (msg.message && typeof msg.message === 'object')
    ? msg.message as Record<string, unknown>
    : null;
  const nestedUsage = nestedMessage && typeof nestedMessage.usage === 'object'
    ? nestedMessage.usage as Record<string, unknown>
    : null;
  const topUsage = msg.usage && typeof msg.usage === 'object'
    ? msg.usage as Record<string, unknown>
    : null;
  const usage = topUsage ?? nestedUsage;
  if (!usage) return zero;
  const num = (v: unknown): number => {
    const n = Number(v ?? 0);
    return Number.isFinite(n) && n > 0 ? n : 0;
  };
  return {
    input_tokens: num(usage.input_tokens),
    output_tokens: num(usage.output_tokens),
    cache_creation_input_tokens: num(usage.cache_creation_input_tokens),
    cache_read_input_tokens: num(usage.cache_read_input_tokens),
  };
}

export function hasAnyUsageDelta(
  inputTokens: number,
  outputTokens: number,
  cacheCreationTokens: number,
  cacheReadTokens: number,
): boolean {
  return (inputTokens + outputTokens + cacheCreationTokens + cacheReadTokens) > 0;
}
