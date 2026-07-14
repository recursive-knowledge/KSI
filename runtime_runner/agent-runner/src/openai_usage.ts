/**
 * Shared OpenAI token-usage extraction for the agent-runner.
 *
 * `usageFromResult` was previously duplicated across multiple OpenAI adapter
 * modules. It lives here in a standalone module so consumers import one
 * implementation without risking an import cycle through `openai.ts` (#982 #6).
 *
 * OpenAI reports total input (cached + fresh); we split the cached portion into
 * `cache_read_input_tokens` so the four buckets sum to inputTotal + outputTotal
 * exactly (Claude-aligned shape). OpenAI has no separate cache-creation bucket.
 */
export function usageFromResult(result: any): {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
} {
  const detailValue = (detail: unknown, snakeKey: string, camelKey: string): number => {
    if (!detail || typeof detail !== 'object') {
      return 0;
    }
    const raw = (detail as Record<string, unknown>)[camelKey] ??
      (detail as Record<string, unknown>)[snakeKey];
    return Number(raw || 0);
  };

  let raw_input_tokens = 0;
  let output_tokens = 0;
  let cache_read_input_tokens = 0;
  for (const response of result?.rawResponses || []) {
    const usage = response?.usage;
    if (!usage) continue;

    raw_input_tokens += Number(usage.inputTokens ?? usage.input_tokens ?? 0);
    output_tokens += Number(usage.outputTokens ?? usage.output_tokens ?? 0);

    const inputDetails = Array.isArray(usage.inputTokensDetails)
      ? usage.inputTokensDetails
      : Array.isArray(usage.input_tokens_details)
        ? usage.input_tokens_details
        : [usage.inputTokensDetails ?? usage.input_tokens_details].filter(Boolean);

    for (const detail of inputDetails) {
      cache_read_input_tokens += detailValue(detail, 'cached_tokens', 'cachedTokens');
    }
  }
  const cache_creation_input_tokens = 0;
  const input_tokens = Math.max(
    0,
    raw_input_tokens - cache_read_input_tokens - cache_creation_input_tokens,
  );
  return {
    input_tokens,
    output_tokens,
    // OpenAI has no separate cache-creation bucket.
    cache_creation_input_tokens: 0,
    cache_read_input_tokens,
  };
}
