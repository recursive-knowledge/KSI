/** Derive the egress allowlist (provider API host + optional base-URL host +
 *  operator-supplied extras) for the container egress proxy. Pure: env in,
 *  host list out. */
function addUrlHost(set: Set<string>, raw: string | undefined): void {
  if (!raw || !raw.trim()) return;
  try {
    set.add(new URL(raw).hostname.toLowerCase());
  } catch {
    // Malformed base URL — ignore; provider default still applies.
  }
}

export function deriveEgressAllowlist(
  env: NodeJS.ProcessEnv = process.env,
): string[] {
  const provider = String(env.MODEL_PROVIDER || 'anthropic').toLowerCase();
  const hosts = new Set<string>();

  if (provider === 'openai') {
    hosts.add('api.openai.com');
    addUrlHost(hosts, env.OPENAI_BASE_URL);
  } else {
    // Default + explicit anthropic; any unrecognised provider also falls here.
    hosts.add('api.anthropic.com');
    addUrlHost(hosts, env.ANTHROPIC_BASE_URL);
  }

  for (const raw of String(env.KSI_EGRESS_ALLOW || '').split(',')) {
    const host = raw.trim().toLowerCase();
    if (host) hosts.add(host);
  }

  return [...hosts];
}
