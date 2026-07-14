/**
 * Adapter-safety helpers shared across agent-runner adapters.
 *
 * Every agent framework that plugs into the KSI runtime emits the same
 * ContainerOutput envelope (see shared_types.ts). Each adapter must
 * detect its own silent-exit modes, produce a diagnostic envelope on
 * failure, and avoid leaking secrets to stdout / DB. This module
 * centralises the provider-aware diagnostic builder and the envelope
 * fingerprint check so new adapters inherit the Haiku-era safety nets
 * (PR #351 silent-failure detection, PR #356 diagnostic envelope) for
 * free — they just call buildSilentDiagnostic with their ProviderAuthConfig.
 *
 * See tests/js/adapter_safety.test.mjs for the regression pins.
 */

/** Identifier for the backing LLM provider. New adapters add a new id. */
export type ProviderId = 'anthropic' | 'openai' | string;

/**
 * Names of the env vars to sample when building a silent-fail diagnostic
 * for a given provider. Only key shapes (type, length) are captured —
 * never values — but knowing WHICH env var the adapter expects helps the
 * forensics narrative when the key is missing or truncated.
 */
export interface ProviderAuthConfig {
  id: ProviderId;
  /** Env var name of the primary API key (e.g. ANTHROPIC_API_KEY). */
  apiKeyEnvName: string;
  /** Env var name of an OAuth/session-token alternative, or undefined. */
  oauthTokenEnvName?: string;
}

export const ANTHROPIC_PROVIDER: ProviderAuthConfig = {
  id: 'anthropic',
  apiKeyEnvName: 'ANTHROPIC_API_KEY',
  oauthTokenEnvName: 'CLAUDE_CODE_OAUTH_TOKEN',
};

export const OPENAI_PROVIDER: ProviderAuthConfig = {
  id: 'openai',
  apiKeyEnvName: 'OPENAI_API_KEY',
};

/**
 * Structured diagnostic snapshot captured when an adapter's run loop
 * either drains with no output or throws mid-stream. Key shapes only —
 * no secret values.
 */
export interface SilentDiagnostic {
  provider: ProviderId;
  messageCount: number;
  resultCount: number;
  lastAssistantFallbackKind: 'null' | 'non-null';
  perTurnInputTokens: number;
  perTurnOutputTokens: number;
  resultInputTokens: number;
  resultOutputTokens: number;
  sdkEnvKeys: string[];
  apiKeyEnvName: string;
  apiKeyType: string;
  apiKeyLength: number;
  oauthTokenEnvName: string;
  oauthTokenType: string;
  oauthTokenLength: number;
  logLevel: string | undefined;
  anthropicLog: string | undefined;
  model: string | undefined;
  modelProvider: string | undefined;
  modelAuthMode: string | undefined;
  mcpServerNames: string[];
  iteratorError: {
    message: string;
    name: string;
    stackHead?: string;
    cause?: string;
  } | null;
}

export function buildSilentDiagnostic(args: {
  provider: ProviderAuthConfig;
  messageCount: number;
  resultCount: number;
  lastAssistantFallback: string | null;
  perTurnInputTokens: number;
  perTurnOutputTokens: number;
  resultInputTokens: number;
  resultOutputTokens: number;
  sdkEnv: Record<string, string | undefined>;
  /** Either a pre-computed list of server names, or a legacy mcpServerConfig dict. */
  mcpServerNames: string[] | Record<string, unknown>;
  iteratorError: Error | null;
}): SilentDiagnostic {
  const apiKey = args.sdkEnv[args.provider.apiKeyEnvName];
  const oauthEnvName = args.provider.oauthTokenEnvName || '';
  const oauthTok = oauthEnvName ? args.sdkEnv[oauthEnvName] : undefined;
  const sdkEnvKeys = Object.keys(args.sdkEnv).sort();
  const mcpServerNames = Array.isArray(args.mcpServerNames)
    ? [...args.mcpServerNames].sort()
    : Object.keys(args.mcpServerNames || {}).sort();

  let causeDesc: string | undefined;
  if (args.iteratorError && typeof args.iteratorError === 'object') {
    const maybeCause = (args.iteratorError as { cause?: unknown }).cause;
    if (maybeCause !== undefined && maybeCause !== null) {
      try {
        causeDesc =
          maybeCause instanceof Error
            ? `${maybeCause.name}: ${maybeCause.message}`
            : JSON.stringify(maybeCause).slice(0, 400);
      } catch {
        causeDesc = String(maybeCause).slice(0, 400);
      }
    }
  }

  return {
    provider: args.provider.id,
    messageCount: args.messageCount,
    resultCount: args.resultCount,
    lastAssistantFallbackKind: args.lastAssistantFallback === null ? 'null' : 'non-null',
    perTurnInputTokens: args.perTurnInputTokens,
    perTurnOutputTokens: args.perTurnOutputTokens,
    resultInputTokens: args.resultInputTokens,
    resultOutputTokens: args.resultOutputTokens,
    sdkEnvKeys,
    apiKeyEnvName: args.provider.apiKeyEnvName,
    apiKeyType: typeof apiKey,
    apiKeyLength: typeof apiKey === 'string' ? apiKey.length : 0,
    oauthTokenEnvName: oauthEnvName,
    oauthTokenType: typeof oauthTok,
    oauthTokenLength: typeof oauthTok === 'string' ? oauthTok.length : 0,
    logLevel: args.sdkEnv.LOG_LEVEL,
    anthropicLog: args.sdkEnv.ANTHROPIC_LOG,
    model: args.sdkEnv.MODEL,
    modelProvider: args.sdkEnv.MODEL_PROVIDER,
    modelAuthMode: args.sdkEnv.MODEL_AUTH_MODE,
    mcpServerNames,
    iteratorError: args.iteratorError
      ? {
          message: String(args.iteratorError.message || ''),
          name: args.iteratorError.name || 'Error',
          stackHead: args.iteratorError.stack
            ? String(args.iteratorError.stack).split('\n').slice(0, 6).join('\n')
            : undefined,
          cause: causeDesc,
        }
      : null,
  };
}

/**
 * ContainerOutput fingerprint that matches the Python host's silent-failure
 * classifier (`src/ksi/runtime/normalize.py::is_silent_agent_failure`).
 * An envelope is a silent failure when it claims success but carries no
 * output, no tools, and no tokens. Adapters can call this before emitting
 * to proactively reclassify themselves rather than relying on host-side
 * reclassification.
 */
export function isSilentFailureEnvelope(envelope: {
  status?: string;
  result?: string | null;
  toolTrace?: Array<unknown>;
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
} | null | undefined): boolean {
  if (!envelope || typeof envelope !== 'object') return false;
  const status = String(envelope.status || '').toLowerCase();
  const successLike = status === '' || status === 'success' || status === 'ok';
  if (!successLike) return false;
  const result = envelope.result;
  const hasOutput = typeof result === 'string' && result.trim().length > 0;
  const hasTools = Array.isArray(envelope.toolTrace) && envelope.toolTrace.length > 0;
  const hasTokens =
    Number(envelope.input_tokens || 0) > 0 ||
    Number(envelope.output_tokens || 0) > 0 ||
    Number(envelope.cache_creation_input_tokens || 0) > 0 ||
    Number(envelope.cache_read_input_tokens || 0) > 0;
  return !hasOutput && !hasTools && !hasTokens;
}
