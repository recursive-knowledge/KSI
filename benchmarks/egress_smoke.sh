#!/usr/bin/env bash
# Live merge-gate smoke for container egress isolation (#923 B1).
# Requires: built ksi-agent:bench image + provider creds in env.
# Proves: (1) provider API works THROUGH the proxy; (2) a blocked host fails.
set -euo pipefail

RUN_ID="smoke$$"
INT="ksi-egress-int-${RUN_ID}"
EXT="ksi-egress-ext-${RUN_ID}"
PROXY="ksi-egress-proxy-${RUN_ID}"
# Honor the suite-wide CONTAINER_IMAGE (set by common.sh / used by run_ksi.sh);
# KSI_CONTAINER_IMAGE kept as a back-compat alias.
IMG="${CONTAINER_IMAGE:-${KSI_CONTAINER_IMAGE:-ksi-agent:bench}}"
ALLOW="api.anthropic.com,api.openai.com"

cleanup() {
  docker rm -f "$PROXY" >/dev/null 2>&1 || true
  docker network rm "$INT" "$EXT" >/dev/null 2>&1 || true
}
trap cleanup EXIT

command -v docker >/dev/null 2>&1 || { echo "FAIL: docker is not available" >&2; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || {
  echo "FAIL: Docker image '$IMG' is missing" >&2
  exit 1
}

docker network create --internal "$INT" >/dev/null
docker network create "$EXT" >/dev/null
docker run -d --rm --name "$PROXY" --network "$EXT" \
  -e "KSI_EGRESS_PROXY_PORT=8080" -e "KSI_EGRESS_ALLOWLIST=${ALLOW}" \
  --entrypoint node "$IMG" /tmp/dist/egress_proxy_main.js >/dev/null
docker network connect "$INT" "$PROXY"

# Wait for READY.
ready=0
for _ in $(seq 1 50); do
  if docker logs "$PROXY" 2>&1 | grep -q "\[egress-proxy\] READY"; then
    ready=1
    break
  fi
  sleep 0.3
done
if [[ "$ready" -ne 1 ]]; then
  echo "FAIL: egress proxy did not become READY" >&2
  docker logs "$PROXY" >&2 || true
  exit 1
fi

PROXY_URL="http://${PROXY}:8080"
run_in_agent() {
  # --dns 0.0.0.0 mirrors the production agent container (#934): blackholes the
  # embedded resolver's external forwarding while keeping docker service discovery.
  docker run --rm --network "$INT" --dns 0.0.0.0 \
    -e "HTTPS_PROXY=${PROXY_URL}" -e "HTTP_PROXY=${PROXY_URL}" \
    -e "NO_PROXY=localhost,127.0.0.1" \
    --entrypoint bash "$IMG" -c "$1"
}

echo "== agent container sanity check =="
run_in_agent "true" >/dev/null
echo "OK: agent container can start on isolated network"

echo "== blocked host (expect failure) =="
if run_in_agent "curl -fsS --max-time 10 https://exercism.io >/dev/null 2>&1"; then
  echo "FAIL: reached blocked host exercism.io"; exit 1
else
  echo "OK: exercism.io blocked"
fi

echo "== allowed host reachability (expect TLS handshake to succeed) =="
# 401/anything-but-connection-error proves the tunnel reached the API.
if run_in_agent "curl -sS --max-time 15 -o /dev/null -w '%{http_code}' https://api.anthropic.com/v1/messages | grep -Eq '^[0-9]{3}$'"; then
  echo "OK: api.anthropic.com reachable through proxy"
else
  echo "FAIL: could not reach api.anthropic.com through proxy"; exit 1
fi

echo "== external DNS lookup (expect SERVFAIL) =="
# The embedded resolver must not forward external names off-box (#934).
if run_in_agent "getent hosts google.com >/dev/null 2>&1"; then
  echo "FAIL: external DNS resolved (DNS-tunnel exfil channel open)"; exit 1
else
  echo "OK: external DNS lookup fails"
fi

echo "== service discovery (proxy name must still resolve) =="
# Blackholing external DNS must not break docker service discovery.
if run_in_agent "getent hosts ${PROXY} >/dev/null 2>&1"; then
  echo "OK: proxy container name resolves"
else
  echo "FAIL: proxy container name no longer resolves"; exit 1
fi

echo "ALL EGRESS SMOKE CHECKS PASSED"
