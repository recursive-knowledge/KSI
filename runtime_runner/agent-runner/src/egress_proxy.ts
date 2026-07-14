import http from 'node:http';
import net from 'node:net';

/** Only the provider API ports are ever needed. */
const ALLOWED_PORTS = new Set([443, 80]);

/** Exact-hostname, case-insensitive match. No wildcard/suffix matching by
 *  design — suffix matching would allow `api.anthropic.com.evil.com`. */
export function isAllowed(
  host: string,
  port: number,
  allowlist: ReadonlySet<string>,
  allowedPorts: ReadonlySet<number> = ALLOWED_PORTS,
): boolean {
  if (!allowedPorts.has(port)) return false;
  return allowlist.has(host.toLowerCase());
}

export function parseAllowlist(raw: string | undefined): Set<string> {
  return new Set(
    (raw || '')
      .split(',')
      .map((h) => h.trim().toLowerCase())
      .filter((h) => h.length > 0),
  );
}

/** A CONNECT-only forward proxy. Plain HTTP proxying is rejected; only
 *  allowlisted CONNECT tunnels are established.
 *
 *  @param allowedPorts  @internal — test seam; production callers use the
 *                       default 443/80. Lets tests allow an ephemeral port
 *                       without needing root. */
export function createEgressProxy(
  allowlist: ReadonlySet<string>,
  allowedPorts: ReadonlySet<number> = ALLOWED_PORTS,
): http.Server {
  const server = http.createServer((_req, res) => {
    res.writeHead(405, { 'content-type': 'text/plain' });
    res.end('egress proxy: only CONNECT is supported\n');
  });

  server.on('connect', (req, clientSocket, head) => {
    const target = req.url || '';
    const lastColon = target.lastIndexOf(':');
    const host = lastColon > 0 ? target.slice(0, lastColon) : '';
    const port = lastColon > 0 ? Number(target.slice(lastColon + 1)) : NaN;

    if (!host || !Number.isInteger(port) || !isAllowed(host, port, allowlist, allowedPorts)) {
      clientSocket.write('HTTP/1.1 403 Forbidden\r\n\r\n');
      clientSocket.destroy();
      return;
    }

    const upstream = net.connect(port, host, () => {
      clientSocket.write('HTTP/1.1 200 Connection Established\r\n\r\n');
      upstream.write(head);
      upstream.pipe(clientSocket);
      clientSocket.pipe(upstream);
    });
    upstream.on('error', () => clientSocket.destroy());
    clientSocket.on('error', () => upstream.destroy());
  });

  return server;
}
