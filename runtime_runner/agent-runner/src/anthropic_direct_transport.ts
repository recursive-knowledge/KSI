/**
 * Shared transport layer for the Anthropic "direct" forum adapter
 * (`anthropic_direct_forum.ts`).
 *
 * The adapter bypasses the Claude Code SDK and speaks the raw Anthropic
 * Messages API so it can own its own prompt-cache architecture (the SDK's
 * in-place compaction was the root cause of cache_read=0 on Haiku ARC — see
 * PRs #484/#503/#519). The cache-marker logic (`clearRollingCacheControl`,
 * block-form cached prefixes) deliberately stays inline in the adapter and is
 * pinned by per-file guard tests, because it is the load-bearing invariant.
 *
 * The pieces here are the parts that carry NO cache semantics: the endpoint
 * constants, the message/response shapes, the HTTP call, and the text
 * extractor. Keeping them in one place keeps the transport independent of the
 * cache architecture the adapter pins. (This module was shared with the
 * now-removed direct-ARC adapter.)
 */

export const ANTHROPIC_MESSAGES_URL = 'https://api.anthropic.com/v1/messages';
export const ANTHROPIC_VERSION = '2023-06-01';

// Transient failures the API recommends retrying: rate limits (429) and the
// 5xx family. Everything else (4xx auth/validation) is a hard error.
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);
const DEFAULT_MAX_RETRIES = 4;
const DEFAULT_RETRY_BASE_MS = 1000;
const MAX_RETRY_BACKOFF_MS = 30_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));
}

function clampInt(value: string | undefined, fallback: number, min: number, max: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(n)));
}

/** Parse a `Retry-After` header (delta-seconds form) into milliseconds. */
function retryAfterMs(headers: Headers | undefined): number | null {
  const raw = headers?.get?.('retry-after');
  if (!raw) return null;
  const secs = Number(raw);
  return Number.isFinite(secs) && secs >= 0 ? secs * 1000 : null;
}

export interface AnthropicBlock {
  type: string;
  text?: string;
  id?: string;
  name?: string;
  input?: unknown;
}

export interface AnthropicResponse {
  id?: string;
  type?: string;
  role?: string;
  content?: AnthropicBlock[];
  stop_reason?: string;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
  };
  error?: {
    type?: string;
    message?: string;
  };
}

/**
 * POST a Messages request to the Anthropic API.
 *
 * `label` is woven into error messages so a failure is attributable to the
 * calling adapter (e.g. "Anthropic direct ARC runs" vs "Anthropic direct
 * forum runs"). Throws on a missing key, a non-JSON 2xx body, or a non-2xx
 * status that is not retryable / has exhausted retries.
 *
 * Transient failures (429 + 5xx, plus network-level fetch rejections) are
 * retried with exponential backoff, honoring a numeric `Retry-After` header
 * when present. Tunable via `KCSI_ANTHROPIC_MAX_RETRIES` (default 4) and
 * `KCSI_ANTHROPIC_RETRY_BASE_MS` (default 1000); set retries to 0 to
 * disable. The callers still degrade gracefully on a final thrown error.
 */
export async function createMessage(
  sdkEnv: Record<string, string | undefined>,
  body: Record<string, unknown>,
  label: string,
): Promise<AnthropicResponse> {
  const apiKey = String(sdkEnv.ANTHROPIC_API_KEY || '').trim();
  if (!apiKey) {
    throw new Error(`ANTHROPIC_API_KEY is required for ${label} runs.`);
  }
  const maxRetries = clampInt(sdkEnv.KCSI_ANTHROPIC_MAX_RETRIES, DEFAULT_MAX_RETRIES, 0, 10);
  const baseMs = clampInt(sdkEnv.KCSI_ANTHROPIC_RETRY_BASE_MS, DEFAULT_RETRY_BASE_MS, 0, 60_000);

  const backoffMs = (attempt: number): number =>
    Math.min(MAX_RETRY_BACKOFF_MS, baseMs * 2 ** attempt);

  let attempt = 0;
  for (;;) {
    let response: Response;
    let text: string;
    try {
      response = await fetch(ANTHROPIC_MESSAGES_URL, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': ANTHROPIC_VERSION,
        },
        body: JSON.stringify(body),
      });
      text = await response.text();
    } catch (err) {
      // Network-level failure (DNS, connection reset, etc.) — retryable.
      if (attempt < maxRetries) {
        await sleep(backoffMs(attempt));
        attempt += 1;
        continue;
      }
      throw err instanceof Error
        ? new Error(`Anthropic API request failed for ${label} after ${attempt + 1} attempt(s): ${err.message}`)
        : err;
    }

    if (!response.ok) {
      if (RETRYABLE_STATUS.has(response.status) && attempt < maxRetries) {
        await sleep(retryAfterMs(response.headers) ?? backoffMs(attempt));
        attempt += 1;
        continue;
      }
      let msg = text.slice(0, 500);
      try {
        msg = (JSON.parse(text) as AnthropicResponse).error?.message || msg;
      } catch { /* non-JSON error body — keep the raw slice */ }
      throw new Error(`Anthropic API error ${response.status}: ${msg}`);
    }

    try {
      return JSON.parse(text) as AnthropicResponse;
    } catch {
      throw new Error(`Anthropic API returned non-JSON response (${response.status}): ${text.slice(0, 500)}`);
    }
  }
}

/** Concatenate and trim the text blocks of an assistant message. */
export function textFromBlocks(content: AnthropicBlock[] | undefined): string {
  if (!Array.isArray(content)) return '';
  return content
    .filter((block) => block.type === 'text' && typeof block.text === 'string')
    .map((block) => block.text)
    .join('')
    .trim();
}

/** The four token counters every direct-adapter result/round bag carries. */
export interface UsageTotals {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

/** Add one response's `usage` block onto a running total in place. */
export function accumulateUsage(target: UsageTotals, response: AnthropicResponse): void {
  const usage = response.usage || {};
  target.input_tokens += Number(usage.input_tokens || 0);
  target.output_tokens += Number(usage.output_tokens || 0);
  target.cache_creation_input_tokens += Number(usage.cache_creation_input_tokens || 0);
  target.cache_read_input_tokens += Number(usage.cache_read_input_tokens || 0);
}
