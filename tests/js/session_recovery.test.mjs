/**
 * Regression test for the 2026-04 silent-exit session-log recovery path.
 *
 * Context
 * -------
 * Forensics on the Haiku baseline sweep showed that the claude-agent-sdk's
 * async `query(...)` iterator sometimes drains without yielding any events
 * to the Node wrapper, even though the underlying claude-code CLI subprocess
 * DOES run to completion inside the container: 15-30 turns, thousands of
 * tokens, real tool calls captured on-disk at
 * `/home/node/.claude/projects/<slug>/*.jsonl`. Before the recovery path,
 * the wrapper classified this as a hard silent failure with
 * `status='error'`, 0 tokens, and empty model_output -- discarding the one
 * artifact that proved work happened.
 *
 * Fix (runtime_runner/agent-runner/src/index.ts):
 *   `recoverFromSessionLog()` scans the in-container session-log directory,
 *   picks the most-recently-modified JSONL, parses turns, and returns:
 *     * the LAST assistant message's joined text content as `result`
 *     * a count of tool_use blocks (toolUseCount)
 *     * summed per-turn usage (input/output/cache tokens)
 *   The silent-exit branch then emits `status='recovered_from_session'` with
 *   `tokens_source='session_recovery'` and a `recovery_note` instead of
 *   `status='error'` when the log had usable content.
 *
 * If the session log has NO assistant turns (true silent failure), recovery
 * returns null and the original `status='error'` path fires.
 *
 * This test verifies:
 *   1. Recovery happy path: 3-turn log with a final assistant message
 *      yields `result` = last assistant text, summed tokens, tool count.
 *   2. Recovery with tool_use blocks increments `toolUseCount`.
 *   3. Empty session-log directory returns null (→ status=error path).
 *   4. Non-existent root returns null.
 *   5. Most-recently-modified JSONL is picked when several exist.
 *
 * The helper implementation is the source of truth in the TypeScript file
 * (`runtime_runner/agent-runner/src/index.ts`, exported as
 * `recoverFromSessionLog`). This test duplicates the helper inline because
 * the repository's test harness runs Node directly (no tsc step); if the
 * TS helper's semantics change, this test must be updated in lockstep.
 */

import { strict as assert } from "node:assert";
import { describe, it, beforeEach, afterEach } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// ── Local copy of `extractUsageFromSdkMessage` from index.ts ──────────────
// Faithful type-stripped mirror; drift-guarded by copy_sync_guard.test.mjs.
function extractUsageFromSdkMessage(message) {
  const zero = {
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  };
  if (!message || typeof message !== 'object') return zero;
  const msg = message;
  // Assistant messages nest under .message.usage; result messages expose .usage
  // at the top level. Check both and return whichever is populated.
  const nestedMessage = (msg.message && typeof msg.message === 'object')
    ? msg.message
    : null;
  const nestedUsage = nestedMessage && typeof nestedMessage.usage === 'object'
    ? nestedMessage.usage
    : null;
  const topUsage = msg.usage && typeof msg.usage === 'object'
    ? msg.usage
    : null;
  const usage = topUsage ?? nestedUsage;
  if (!usage) return zero;
  const num = (v) => {
    const n = Number(v ?? 0);
    return Number.isFinite(n) && n > 0 ? n : 0;
  };
  return {
    input_tokens: num(usage.input_tokens),
    output_tokens: num(usage.output_tokens),
    cache_creation_input_tokens: num(usage.cache_creation_input_tokens),
    cache_read_input_tokens: num(usage.cache_read_input_tokens),
  };
}

// ── Local copy of `recoverFromSessionLog` from index.ts ───────────────────
// Faithful type-stripped mirror; drift-guarded by copy_sync_guard.test.mjs.
const CONTAINER_CLAUDE_SESSIONS_ROOT = '/home/node/.claude/projects';

function recoverFromSessionLog(
  rootDir = CONTAINER_CLAUDE_SESSIONS_ROOT,
  readFileImpl = (p) => fs.readFileSync(p, 'utf-8'),
) {
  // 1) Locate session log files.
  let candidates = [];
  try {
    if (!fs.existsSync(rootDir)) return null;
    const walk = (dir) => {
      let entries;
      try {
        entries = fs.readdirSync(dir, { withFileTypes: true });
      } catch {
        return;
      }
      for (const ent of entries) {
        const full = path.join(dir, ent.name);
        if (ent.isDirectory()) {
          walk(full);
        } else if (ent.isFile() && ent.name.endsWith('.jsonl')) {
          candidates.push(full);
        }
      }
    };
    walk(rootDir);
  } catch {
    return null;
  }
  if (candidates.length === 0) return null;

  // Pick the most-recently-modified jsonl — that's the log for this session.
  candidates.sort((a, b) => {
    let am = 0;
    let bm = 0;
    try { am = fs.statSync(a).mtimeMs; } catch { /* ignore */ }
    try { bm = fs.statSync(b).mtimeMs; } catch { /* ignore */ }
    return bm - am;
  });
  const sourcePath = candidates[0];

  // 2) Parse the JSONL.
  let raw;
  try {
    raw = readFileImpl(sourcePath);
  } catch {
    return null;
  }
  let turnCount = 0;
  let toolUseCount = 0;
  let lastAssistantText = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheCreationTokens = 0;
  let cacheReadTokens = 0;

  for (const line of raw.split('\n')) {
    if (!line.trim()) continue;
    let entry;
    try {
      entry = JSON.parse(line);
    } catch {
      continue;
    }
    if (!entry || typeof entry !== 'object') continue;
    // Count anything that looks like a turn (assistant / user / tool roles).
    const t = typeof entry.type === 'string' ? entry.type : '';
    if (t === 'assistant' || t === 'user') {
      turnCount += 1;
    }
    // Per-turn usage may be nested under .message.usage (assistant turns).
    const delta = extractUsageFromSdkMessage(entry);
    inputTokens += delta.input_tokens;
    outputTokens += delta.output_tokens;
    cacheCreationTokens += delta.cache_creation_input_tokens;
    cacheReadTokens += delta.cache_read_input_tokens;

    // Collect assistant text + tool-use counts.
    if (t === 'assistant') {
      const inner = (entry.message && typeof entry.message === 'object')
        ? entry.message
        : null;
      const content = inner && Array.isArray(inner.content) ? inner.content : [];
      const textParts = [];
      for (const block of content) {
        if (!block || typeof block !== 'object') continue;
        const b = block;
        if (b.type === 'text' && typeof b.text === 'string') {
          textParts.push(b.text);
        } else if (b.type === 'tool_use') {
          toolUseCount += 1;
        }
      }
      const joined = textParts.join('').trim();
      if (joined.length > 0) {
        lastAssistantText = joined;
      }
    }
  }

  if (lastAssistantText === null && turnCount === 0) {
    // Nothing usable in the log — let the caller fall through to status=error.
    return null;
  }

  return {
    result: lastAssistantText,
    toolUseCount,
    inputTokens,
    outputTokens,
    cacheCreationTokens,
    cacheReadTokens,
    sourcePath,
    turnCount,
  };
}

// ── Local copy of the silent-exit emit logic from index.ts ────────────────
// Simulates the writeOutput call at the silent-exit branch: given a silent
// condition (resultCount=0, no lastAssistantFallback, zero tokens) plus a
// session-log root, produces the envelope the runner would emit.
function simulateSilentExit(sessionsRoot) {
  let recovered = null;
  try {
    recovered = recoverFromSessionLog(sessionsRoot);
  } catch {
    recovered = null;
  }
  if (recovered && (recovered.result || recovered.turnCount > 0)) {
    const tokenTotal =
      recovered.inputTokens + recovered.outputTokens
      + recovered.cacheCreationTokens + recovered.cacheReadTokens;
    return {
      status: "recovered_from_session",
      result: recovered.result,
      input_tokens: recovered.inputTokens,
      output_tokens: recovered.outputTokens,
      cache_creation_input_tokens: recovered.cacheCreationTokens,
      cache_read_input_tokens: recovered.cacheReadTokens,
      tokens_source: "session_recovery",
      recovery_note:
        `Recovered from on-disk session log at ${recovered.sourcePath}: ` +
        `${recovered.turnCount} turns, ${recovered.toolUseCount} tool_use blocks, ` +
        `~${tokenTotal} tokens.`,
    };
  }
  return {
    status: "error",
    result: null,
    input_tokens: 0,
    output_tokens: 0,
    tokens_source: "unavailable",
    error: "agent-runner produced no output",
  };
}

// ── Fixture helpers ────────────────────────────────────────────────────────

function writeJsonl(filePath, entries) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, entries.map((e) => JSON.stringify(e)).join("\n") + "\n");
}

function makeTempRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ksi-sess-recov-"));
}

// ── Tests ─────────────────────────────────────────────────────────────────

describe("recoverFromSessionLog — happy path", () => {
  let tmp;
  beforeEach(() => { tmp = makeTempRoot(); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  it("extracts last assistant message + summed tokens from 3-turn log", () => {
    // Shape matches a real claude-agent-sdk session JSONL: assistant turns
    // have nested .message.usage and .message.content[].text blocks.
    const logPath = path.join(tmp, "projects", "test-slug", "abc-123.jsonl");
    writeJsonl(logPath, [
      { type: "user", message: { role: "user", content: "hi" } },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "Let me think." }],
          usage: { input_tokens: 100, output_tokens: 20, cache_read_input_tokens: 500 },
        },
      },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "The answer is 42." }],
          usage: { input_tokens: 50, output_tokens: 30, cache_read_input_tokens: 600 },
        },
      },
    ]);

    const out = recoverFromSessionLog(tmp);
    assert.ok(out, "recovery must return a result");
    assert.equal(out.result, "The answer is 42.", "must pick LAST assistant text");
    assert.equal(out.turnCount, 3);
    assert.equal(out.toolUseCount, 0);
    assert.equal(out.inputTokens, 150);
    assert.equal(out.outputTokens, 50);
    assert.equal(out.cacheReadTokens, 1100);
    assert.equal(out.cacheCreationTokens, 0);
  });

  it("counts tool_use blocks as toolUseCount", () => {
    const logPath = path.join(tmp, "projects", "slug", "s.jsonl");
    writeJsonl(logPath, [
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "text", text: "Calling a tool." },
            { type: "tool_use", id: "t1", name: "Read", input: {} },
            { type: "tool_use", id: "t2", name: "Bash", input: {} },
          ],
          usage: { input_tokens: 10, output_tokens: 5 },
        },
      },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "tool_use", id: "t3", name: "Edit", input: {} },
            { type: "text", text: "Done." },
          ],
          usage: { input_tokens: 10, output_tokens: 5 },
        },
      },
    ]);

    const out = recoverFromSessionLog(tmp);
    assert.ok(out);
    assert.equal(out.toolUseCount, 3);
    assert.equal(out.result, "Done.", "last assistant text wins even when interleaved with tool_use");
  });

  it("picks the most-recently-modified JSONL when multiple exist", () => {
    const oldPath = path.join(tmp, "projects", "a", "old.jsonl");
    const newPath = path.join(tmp, "projects", "b", "new.jsonl");
    writeJsonl(oldPath, [
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "OLD answer" }],
          usage: { input_tokens: 1, output_tokens: 1 },
        },
      },
    ]);
    // Write the "new" file AFTER, so its mtime is later.
    writeJsonl(newPath, [
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "NEW answer" }],
          usage: { input_tokens: 1, output_tokens: 1 },
        },
      },
    ]);
    // Force "new" mtime to strictly after "old"'s mtime so the test is not
    // subject to coarse filesystem clock resolution.
    const future = new Date(Date.now() + 60_000);
    fs.utimesSync(newPath, future, future);

    const out = recoverFromSessionLog(tmp);
    assert.ok(out);
    assert.equal(out.result, "NEW answer");
    assert.ok(out.sourcePath.endsWith("new.jsonl"));
  });
});

describe("recoverFromSessionLog — fallthrough cases", () => {
  it("returns null when the root does not exist", () => {
    const out = recoverFromSessionLog("/nonexistent/definitely/not/there");
    assert.equal(out, null);
  });

  it("returns null when the root exists but is empty", () => {
    const tmp = makeTempRoot();
    try {
      const out = recoverFromSessionLog(tmp);
      assert.equal(out, null);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("returns null when JSONLs exist but have no assistant turns or valid entries", () => {
    const tmp = makeTempRoot();
    try {
      const logPath = path.join(tmp, "projects", "slug", "s.jsonl");
      // Only junk lines → no turns, no assistant text → returns null
      fs.mkdirSync(path.dirname(logPath), { recursive: true });
      fs.writeFileSync(logPath, "not json\n{broken\n\n");
      const out = recoverFromSessionLog(tmp);
      assert.equal(out, null);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });
});

describe("silent-exit envelope emission", () => {
  it("emits status=recovered_from_session with session_recovery tokens_source on recovery", () => {
    const tmp = makeTempRoot();
    try {
      const logPath = path.join(tmp, "projects", "slug", "s.jsonl");
      writeJsonl(logPath, [
        { type: "user", message: { role: "user", content: "hi" } },
        {
          type: "assistant",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "recovered output" }],
            usage: { input_tokens: 200, output_tokens: 40 },
          },
        },
        {
          type: "assistant",
          message: {
            role: "assistant",
            content: [
              { type: "tool_use", id: "t1", name: "Read", input: {} },
              { type: "text", text: "final answer" },
            ],
            usage: { input_tokens: 100, output_tokens: 20 },
          },
        },
      ]);

      const envelope = simulateSilentExit(tmp);
      assert.equal(envelope.status, "recovered_from_session");
      assert.equal(envelope.result, "final answer", "last assistant text goes to envelope.result");
      assert.equal(envelope.tokens_source, "session_recovery");
      assert.equal(envelope.input_tokens, 300);
      assert.equal(envelope.output_tokens, 60);
      assert.ok(envelope.recovery_note, "recovery_note must be populated");
      assert.ok(envelope.recovery_note.includes("3 turns"));
      assert.ok(envelope.recovery_note.includes("1 tool_use"));
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("TRUE silent-failure with no session memory falls through to status=error", () => {
    // Regression: empty session directory must NOT be treated as a recovery —
    // otherwise we'd turn every silent-exit into a success and hide real
    // startup/auth failures.
    const tmp = makeTempRoot();
    try {
      const envelope = simulateSilentExit(tmp);
      assert.equal(envelope.status, "error");
      assert.equal(envelope.tokens_source, "unavailable");
      assert.equal(envelope.input_tokens, 0);
      assert.equal(envelope.output_tokens, 0);
      assert.equal(envelope.result, null);
      assert.ok(envelope.error);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("TRUE silent-failure when the session-log root doesn't exist falls through to status=error", () => {
    const envelope = simulateSilentExit("/this/path/does/not/exist/ever");
    assert.equal(envelope.status, "error");
    assert.equal(envelope.tokens_source, "unavailable");
  });
});

// ── Issue #1051 ─────────────────────────────────────────────────────────────
// A 2026-04-20 SWE-bench Pro trace was scored `no_patch` despite `model_output`
// containing a real-looking diff, because the 167KB `model_output` was
// suspected to be the *entire raw JSONL session transcript* (JSON-escaped) —
// which src/ksi/eval/patch_extract.py::extract_patch() correctly fails to
// parse a clean diff from, since a raw JSONL dump never has a *line* that
// literally starts with "diff --git " (the diff text is embedded inside an
// escaped JSON string value on one long transcript line).
//
// Forensics (this file) confirmed `recoverFromSessionLog()` already extracts
// only the LAST assistant message's joined `text` blocks — never tool_use
// payloads or other turns' raw JSON — so a diff the agent printed in its
// final answer survives as a clean, `extract_patch()`-parseable string. This
// was true even in the original 2026-04-20 introduction of the recovery path
// (commit 876693c5), so the raw-dump failure mode described in #1051 was not
// reproduced in the current (or that day's) `recoverFromSessionLog`
// implementation. This test pins the contract so it can't regress silently.
describe("recoverFromSessionLog — clean-text contract (issue #1051)", () => {
  it("recovered result is a clean diff line, not a raw JSONL transcript dump", () => {
    const tmp = makeTempRoot();
    try {
      const diffText = [
        "I've made the fix. Here is the diff:",
        "",
        "diff --git a/openlibrary/foo.py b/openlibrary/foo.py",
        "index 1111111..2222222 100644",
        "--- a/openlibrary/foo.py",
        "+++ b/openlibrary/foo.py",
        "@@ -1,3 +1,3 @@",
        " def bar():",
        "-    return 1",
        "+    return 2",
      ].join("\n");

      const logPath = path.join(tmp, "projects", "slug", "s.jsonl");
      writeJsonl(logPath, [
        { type: "user", message: { role: "user", content: "fix the bug" } },
        {
          type: "assistant",
          message: {
            role: "assistant",
            content: [
              { type: "tool_use", id: "t1", name: "Edit", input: { path: "openlibrary/foo.py" } },
            ],
            usage: { input_tokens: 500, output_tokens: 100 },
          },
        },
        {
          type: "assistant",
          message: {
            role: "assistant",
            content: [{ type: "text", text: diffText }],
            usage: { input_tokens: 50, output_tokens: 60 },
          },
        },
      ]);

      const out = recoverFromSessionLog(tmp);
      assert.ok(out, "recovery must return a result");
      assert.equal(out.result, diffText, "result must be exactly the last assistant text, verbatim");

      // Negative assertions: the result must NOT look like a raw JSONL dump —
      // i.e. it must not carry JSONL/session-log framing that would defeat
      // extract_patch()'s line-anchored "diff --git " / "--- a/" scan.
      assert.ok(!out.result.includes('"type":"assistant"'), "must not embed raw JSONL entry framing");
      assert.ok(!out.result.includes('"type": "assistant"'), "must not embed raw JSONL entry framing");
      assert.ok(!out.result.includes("tool_use"), "must not leak tool_use payloads from other turns");
      assert.ok(!out.result.includes('"role":"user"'), "must not include earlier user turns");

      // Positive assertion: a real line in the result starts with "diff --git ",
      // which is exactly what extract_patch()'s raw-unified-diff fallback (case
      // 4) anchors on. A raw JSON-escaped dump would fail this because the
      // diff text would be embedded mid-line inside a JSON string, not at the
      // start of an actual line.
      const hasAnchoredDiffLine = out.result
        .split("\n")
        .some((line) => line.startsWith("diff --git "));
      assert.ok(hasAnchoredDiffLine, "recovered text must contain a line literally starting with 'diff --git '");
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });
});
