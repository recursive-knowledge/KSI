import { createEgressProxy, parseAllowlist } from './egress_proxy.js';

const port = Number(process.env.KCSI_EGRESS_PROXY_PORT || 8080);
const allowlist = parseAllowlist(process.env.KCSI_EGRESS_ALLOWLIST);

const server = createEgressProxy(allowlist);
server.on('error', (err) => {
  console.error('[egress-proxy] fatal:', err instanceof Error ? err.message : String(err));
  process.exit(1);
});
server.listen(port, '0.0.0.0', () => {
  const addr = server.address();
  const actual = typeof addr === 'object' && addr ? addr.port : port;
  // The host poller waits for this exact marker. Keep "READY listening on <port>".
  // eslint-disable-next-line no-console
  console.log(
    `[egress-proxy] READY listening on ${actual} allow=${[...allowlist].join(',') || '(none)'}`,
  );
});
