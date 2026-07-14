/**
 * Wire-protocol sentinel lockstep guard.
 *
 * The stdout envelope markers are defined independently in TWO separate npm
 * packages: the host-side parser (runtime_runner/src/container_output.ts) and
 * the in-container writer (runtime_runner/agent-runner/src/output.ts). If either
 * OUTPUT_START_MARKER / OUTPUT_END_MARKER drifts, the host stdout parser
 * silently stops finding envelopes. This test reads both files as text and
 * asserts the two literal marker strings are byte-equal across packages.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const readRepo = (...p) => fs.readFileSync(path.join(repoRoot, ...p), "utf-8");

const hostTs = readRepo("runtime_runner", "src", "container_output.ts");
const agentTs = readRepo(
  "runtime_runner",
  "agent-runner",
  "src",
  "output.ts",
);

function extractMarker(src, name, label) {
  const m = src.match(
    new RegExp(`export const ${name}\\s*=\\s*'([^']*)'`),
  );
  assert.ok(m, `${label}: could not find 'export const ${name}'`);
  return m[1];
}

describe("output marker lockstep: host parser ↔ agent-runner writer", () => {
  it("OUTPUT_START_MARKER is byte-equal across both packages", () => {
    const host = extractMarker(hostTs, "OUTPUT_START_MARKER", "container_output.ts");
    const agent = extractMarker(agentTs, "OUTPUT_START_MARKER", "output.ts");
    assert.equal(
      host,
      agent,
      "OUTPUT_START_MARKER drifted between container_output.ts and agent-runner/output.ts",
    );
  });

  it("OUTPUT_END_MARKER is byte-equal across both packages", () => {
    const host = extractMarker(hostTs, "OUTPUT_END_MARKER", "container_output.ts");
    const agent = extractMarker(agentTs, "OUTPUT_END_MARKER", "output.ts");
    assert.equal(
      host,
      agent,
      "OUTPUT_END_MARKER drifted between container_output.ts and agent-runner/output.ts",
    );
  });
});
