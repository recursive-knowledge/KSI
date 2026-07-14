/**
 * Tests for the container-side barrier helpers
 * (`runtime_runner/agent-runner/src/barrier.ts`). Following the convention in
 * the rest of `tests/js/`, the helpers under test are mirrored as plain JS
 * here so the tests can run without compiling TypeScript or installing the
 * agent-runner dependency tree. The TS source remains the source of truth
 * — when changing one, update the other.
 */

import { strict as assert } from "node:assert";
import { describe, it, beforeEach, afterEach } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// ── Local mirror of barrier.ts helpers (keep in sync) ─────────────────────
function sanitizeNamePart(value) {
  return (value || "").replace(/[^A-Za-z0-9_-]/g, "_");
}

function sentinelFilename(name, agentId) {
  return `.barrier.${sanitizeNamePart(name)}.${sanitizeNamePart(agentId)}.ready`;
}

function responseFilename(name, agentId) {
  return `.barrier.${sanitizeNamePart(name)}.${sanitizeNamePart(agentId)}.response`;
}

function writeSentinelFile(workspaceDir, name, agentId, payload) {
  const target = path.join(workspaceDir, sentinelFilename(name, agentId));
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const tmp = `${target}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(payload), { encoding: "utf-8" });
  fs.renameSync(tmp, target);
  return target;
}

async function waitForBarrierFile(filePath, timeoutMs, options = {}) {
  const pollMs = Math.max(50, options.pollIntervalMs ?? 500);
  const start = Date.now();
  while (true) {
    let exists = false;
    try { exists = fs.existsSync(filePath); } catch { exists = false; }
    if (options.onAttempt) {
      try { options.onAttempt({ elapsedMs: Date.now() - start, exists }); }
      catch { /* never let an instrumentation hook break the wait loop */ }
    }
    if (exists) {
      try {
        const raw = fs.readFileSync(filePath, "utf-8");
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          try { fs.unlinkSync(filePath); } catch { /* ignore */ }
          return parsed;
        }
      } catch { /* fall through */ }
    }
    if (Date.now() - start >= timeoutMs) return null;
    await new Promise((resolve) => setTimeout(resolve, pollMs));
  }
}

// ── Test setup ────────────────────────────────────────────────────────────
let scratchDir;

beforeEach(() => {
  scratchDir = fs.mkdtempSync(path.join(os.tmpdir(), "barrier-test-"));
});

afterEach(() => {
  try { fs.rmSync(scratchDir, { recursive: true, force: true }); } catch { /* ignore */ }
});

describe("barrier filename helpers", () => {
  it("namespaces sentinel and response by name and agent id", () => {
    assert.equal(sentinelFilename("phase1_reflection", "agent-7"), ".barrier.phase1_reflection.agent-7.ready");
    assert.equal(responseFilename("phase1_reflection", "agent-7"), ".barrier.phase1_reflection.agent-7.response");
  });

  it("sanitizes path-traversal-flavoured inputs", () => {
    const name = sentinelFilename("../evil", "../escape");
    assert.equal(name.includes("/"), false);
    assert.equal(name.includes(".."), false);
  });
});

describe("writeSentinelFile", () => {
  it("writes JSON atomically with a tmp+rename", () => {
    const target = writeSentinelFile(scratchDir, "b", "x", { hello: "world" });
    assert.equal(fs.existsSync(target), true);
    const body = JSON.parse(fs.readFileSync(target, "utf-8"));
    assert.deepEqual(body, { hello: "world" });
    // No leftover .tmp file.
    const leftovers = fs.readdirSync(scratchDir).filter((f) => f.endsWith(".tmp"));
    assert.deepEqual(leftovers, []);
  });
});

describe("waitForBarrierFile", () => {
  it("returns parsed JSON when the file appears within timeout", async () => {
    const filePath = path.join(scratchDir, responseFilename("b", "x"));
    setTimeout(() => {
      fs.writeFileSync(filePath, JSON.stringify({ score: 0.75 }), "utf-8");
    }, 60);
    const got = await waitForBarrierFile(filePath, 2000, { pollIntervalMs: 50 });
    assert.deepEqual(got, { score: 0.75 });
    // Consumed by the helper.
    assert.equal(fs.existsSync(filePath), false);
  });

  it("returns null when the timeout elapses without the file", async () => {
    const filePath = path.join(scratchDir, responseFilename("b", "x"));
    const start = Date.now();
    const got = await waitForBarrierFile(filePath, 200, { pollIntervalMs: 50 });
    const elapsed = Date.now() - start;
    assert.equal(got, null);
    assert.ok(elapsed >= 180, `expected timeout >= 180ms, got ${elapsed}`);
  });

  it("treats invalid JSON as still-missing (graceful)", async () => {
    const filePath = path.join(scratchDir, responseFilename("b", "x"));
    fs.writeFileSync(filePath, "{not valid", "utf-8");
    const got = await waitForBarrierFile(filePath, 200, { pollIntervalMs: 50 });
    assert.equal(got, null);
  });

  it("treats non-object JSON (arrays, scalars) as missing", async () => {
    const filePath = path.join(scratchDir, responseFilename("b", "x"));
    fs.writeFileSync(filePath, JSON.stringify([1, 2, 3]), "utf-8");
    const got = await waitForBarrierFile(filePath, 200, { pollIntervalMs: 50 });
    assert.equal(got, null);
  });

  it("invokes onAttempt for instrumentation without breaking on hook errors", async () => {
    const filePath = path.join(scratchDir, responseFilename("b", "x"));
    let attempts = 0;
    const got = await waitForBarrierFile(filePath, 200, {
      pollIntervalMs: 50,
      onAttempt: () => { attempts += 1; throw new Error("hook explode"); },
    });
    assert.equal(got, null);
    assert.ok(attempts >= 2, `expected multiple attempts, saw ${attempts}`);
  });
});
