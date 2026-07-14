/**
 * Stdout result protocol. Each result is framed in
 * OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs so the host's streaming
 * parser (runtime_runner/src/container_runner.ts) can extract envelopes from
 * interleaved log output. An optional per-run nonce is stamped onto every
 * envelope so the host can reject markers from a stale/previous container.
 */
import { ContainerOutput } from './shared_types.js';

export const OUTPUT_START_MARKER = '---KCSI_OUTPUT_START---';
export const OUTPUT_END_MARKER = '---KCSI_OUTPUT_END---';

let outputProtocolNonce = '';

/** Set the protocol nonce stamped on every subsequent {@link writeOutput}. */
export function setOutputProtocolNonce(nonce: string): void {
  outputProtocolNonce = nonce;
}

export function writeOutput(output: ContainerOutput): void {
  const framedOutput = outputProtocolNonce
    ? { ...output, protocolNonce: outputProtocolNonce }
    : output;
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(framedOutput));
  console.log(OUTPUT_END_MARKER);
}
