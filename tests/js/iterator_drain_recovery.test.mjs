/**
 * Regression test for PR #527 (commit cbb02035): non-strict scheduled tasks
 * (polyglot, swebench_pro) now route through `maybeRecoverFromEmptyScheduledOutcome`
 * with `trigger='iterator_drain_pending_tools'` when the claude-agent-sdk
 * iterator drains mid-conversation with pending tool calls. Previously these
 * tasks emitted `status='error'`, which the orchestrator's retry gate
 * classified as transient → 1-3 retries, each producing the same empty
 * workspace because the SDK race fires deterministically on Haiku.
 *
 * PR #527 was verified to suppress retry storms but the *correctness of the
 * recovered output* — i.e. whether a recovered envelope can actually score >0
 * downstream — was never tested with realistic transcript content.
 *
 * Downstream scoring path:
 *   - src/ksi/benchmarks/polyglot_harness.py:411-413
 *       `extract_solution_files(model_output, language=language)`
 *       parses fenced ``` <lang> blocks (or `// file: <name>` named blocks)
 *   - src/ksi/benchmarks/swebench_pro.py:273-274
 *       `extract_patch(model_output or "")` parses unified-diff blocks
 *
 * So a recovered run can score >0 IF the agent's last assistant text contained
 * a code/patch block. This test pins that contract: a JSONL fragment where the
 * assistant emits a `tool_use` (Edit) but the iterator drains before the
 * `tool_result` is yielded back, and the prior assistant turn carries the
 * canonical fenced ```python block. The recovered envelope must:
 *   1. carry status='recovered_from_session'
 *   2. preserve the fenced ```python block in `result`
 *   3. stamp the recovery_note with the iterator_drain_pending_tools wording
 *   4. be parseable by a `extract_solution_files`-equivalent fence scanner
 *
 * Future changes that drop the prior-turn content (e.g. a turn-pruning
 * optimization) will fail this test instead of silently zeroing scores on
 * recovered runs.
 *
 * The helper implementation is the source of truth in the TypeScript file
 * (`runtime_runner/agent-runner/src/index.ts`). This test duplicates the
 * helper inline because the repo's test harness runs Node directly (no tsc
 * step). If the TS helper's semantics change, this test must be updated in
 * lockstep — same pattern as `session_recovery.test.mjs` and
 * `silent_failure_recovery.test.mjs`.
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

// ── DELIBERATELY TRIMMED copy of `maybeRecoverFromEmptyScheduledOutcome` ──
// Trimmed to the recovery-success path + recovery_note wording for the
// `iterator_drain_pending_tools` trigger; the diagnostic status=error path
// is exercised by silent_failure_recovery.test.mjs's FULL copy instead.
// Source: runtime_runner/agent-runner/src/index.ts:503-637. The trigger
// wordings below are anchor-pinned against the TS source by
// copy_sync_guard.test.mjs (the trim itself is intentional and exempt from
// the strict byte-mirror guard).
function maybeRecoverFromEmptyScheduledOutcome(ctx, recoverImpl) {
  const logLines = [];
  let recovered = null;
  try {
    recovered = recoverImpl();
  } catch (err) {
    logLines.push(
      `Silent-exit recovery threw (trigger=${ctx.trigger}): ${
        err instanceof Error ? err.message : String(err)
      } — falling through to diagnostic status=error.`,
    );
    recovered = null;
  }
  if (recovered && (recovered.result || recovered.turnCount > 0)) {
    const tokenTotal =
      recovered.inputTokens +
      recovered.outputTokens +
      recovered.cacheCreationTokens +
      recovered.cacheReadTokens;
    const recoveryNote =
      `Recovered from on-disk session log at ${recovered.sourcePath}: ` +
      `${recovered.turnCount} turns, ${recovered.toolUseCount} tool_use blocks, ` +
      `~${tokenTotal} tokens (summed from per-turn usage). ` +
      `The claude-agent-sdk iterator ${
        ctx.trigger === "empty_result_event"
          ? "emitted an empty result event with zero tokens and no tool trace"
          : ctx.trigger === "iterator_drain_pending_tools"
            ? "drained mid-conversation with pending tool calls (no tool_result on the wire)"
            : "drained without yielding events to the Node wrapper"
      }, but the underlying CLI subprocess ran to completion. This output ` +
      `was reconstructed from that log so downstream evaluators have something ` +
      `to score; treat the tokens as approximate.`;
    logLines.push(
      `Silent-exit recovery succeeded (trigger=${ctx.trigger}): turns=${recovered.turnCount}, ` +
        `tools=${recovered.toolUseCount}, tokens=${tokenTotal}, ` +
        `resultLen=${recovered.result ? recovered.result.length : 0}.`,
    );
    return {
      recovered,
      logLines,
      envelope: {
        status: "recovered_from_session",
        result: recovered.result,
        newSessionId: ctx.newSessionId,
        toolTrace: ctx.toolTrace.slice(-1000),
        input_tokens: recovered.inputTokens,
        output_tokens: recovered.outputTokens,
        cache_creation_input_tokens: recovered.cacheCreationTokens,
        cache_read_input_tokens: recovered.cacheReadTokens,
        tokens_source: "session_recovery",
        recovery_note: recoveryNote,
      },
    };
  }
  return {
    recovered,
    logLines,
    envelope: {
      status: "error",
      result: null,
      newSessionId: ctx.newSessionId,
      toolTrace: ctx.toolTrace.slice(-1000),
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      tokens_source: "unavailable",
      error: `agent-runner produced no output (trigger=${ctx.trigger})`,
    },
  };
}

// ── JS port of `extract_solution_files` (Pattern-2 fallback) ────────────
// Mirrors the regex in src/ksi/benchmarks/polyglot_harness.py:351-387 for
// language='python' (fence tag is `python`). We intentionally do NOT port
// Pattern-1 (named-file blocks) because the typical Haiku polyglot answer
// uses bare ```python ... ``` blocks. Pattern-2 is what scores on the
// realistic recovered transcript below.
function extractPythonSolutionFromFences(output) {
  const re = /```python\s*\n([\s\S]*?)\n```/g;
  const matches = [];
  let m;
  while ((m = re.exec(output)) !== null) {
    matches.push(m[1]);
  }
  if (matches.length === 0) return {};
  return { "solution.py": matches[matches.length - 1] };
}

// ── Fixtures ─────────────────────────────────────────────────────────────

function writeJsonl(filePath, entries) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(
    filePath,
    entries.map((e) => JSON.stringify(e)).join("\n") + "\n",
  );
}

function makeTempRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ksi-iter-drain-"));
}

// Canonical polyglot Python answer the agent produces in its FINAL assistant
// turn before attempting to write the file via `Edit`. Includes the fenced
// ```python block that `extract_solution_files(language='python')` parses.
const FINAL_ANSWER_TEXT =
  "Here is the implementation for the leap year exercise. " +
  "The function returns True iff the year is divisible by 4 and " +
  "not divisible by 100, unless also divisible by 400.\n\n" +
  "```python\n" +
  "def is_leap_year(year: int) -> bool:\n" +
  "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n" +
  "```\n\n" +
  "I'll now write this to leap.py.";

function baseRecoveryCtx(overrides = {}) {
  return {
    messageCount: 12,
    resultCount: 0,
    lastAssistantFallback: null,
    perTurnInputTokens: 0,
    perTurnOutputTokens: 0,
    perTurnCacheCreationTokens: 0,
    perTurnCacheReadTokens: 0,
    resultInputTokens: 0,
    resultOutputTokens: 0,
    resultCacheCreationTokens: 0,
    resultCacheReadTokens: 0,
    toolTrace: [],
    newSessionId: "sess-iter-drain-1",
    sdkEnv: {
      ANTHROPIC_API_KEY: "sk-ant-api03-" + "a".repeat(50),
      MODEL: "claude-haiku-4-5",
      MODEL_PROVIDER: "anthropic",
    },
    mcpServerConfig: {},
    iteratorError: null,
    trigger: "iterator_drain_pending_tools",
    ...overrides,
  };
}

// ── Tests ───────────────────────────────────────────────────────────────
//
// Note on transcript shape: `recoverFromSessionLog` iterates turns in order
// and OVERWRITES `lastAssistantText` whenever it sees an assistant turn with
// non-empty joined text. So the "result" ends up being the LAST text-bearing
// assistant turn. For polyglot scoring to work post-recovery, the agent's
// canonical fenced ```python answer must be in (or co-located with) that
// last text turn. This is what realistic Haiku polyglot transcripts look
// like — the model restates the solution in-prose right before invoking
// Edit. The first test pins the success case; the second pins the failure
// mode (text not in the last turn) so future contract changes are explicit.

describe("iterator_drain_pending_tools — realistic Haiku polyglot transcript", () => {
  let tmp;
  beforeEach(() => {
    tmp = makeTempRoot();
  });
  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("recovered result has the python fence + is parseable by extract_solution_files-equivalent", () => {
    // Final turn carries BOTH the canonical fenced python block AND the
    // Edit tool_use. This matches actual Haiku polyglot transcripts where
    // the model restates the solution in-prose right before invoking Edit.
    const logPath = path.join(
      tmp,
      "projects",
      "-workspace-task",
      "session-real.jsonl",
    );
    writeJsonl(logPath, [
      {
        type: "user",
        message: {
          role: "user",
          content: "Solve the leap year exercism exercise.",
        },
      },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "text", text: "Reading the spec." },
            {
              type: "tool_use",
              id: "t1",
              name: "Read",
              input: { file_path: "/workspace/task/leap.py" },
            },
          ],
          usage: { input_tokens: 800, output_tokens: 60 },
        },
      },
      {
        type: "user",
        message: {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "t1",
              content: "def is_leap_year(year):\n    pass\n",
            },
          ],
        },
      },
      {
        // Final turn: text (with fenced answer) + Edit tool_use. Iterator
        // drains here — the Edit tool_result never comes back.
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "text", text: FINAL_ANSWER_TEXT },
            {
              type: "tool_use",
              id: "t2",
              name: "Edit",
              input: {
                file_path: "/workspace/task/leap.py",
                old_string: "def is_leap_year(year):\n    pass",
                new_string:
                  "def is_leap_year(year: int) -> bool:\n" +
                  "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
              },
            },
          ],
          usage: { input_tokens: 950, output_tokens: 220 },
        },
      },
    ]);

    const outcome = maybeRecoverFromEmptyScheduledOutcome(
      baseRecoveryCtx(),
      () => recoverFromSessionLog(tmp),
    );

    // ── Envelope contract ────────────────────────────────────────────
    assert.equal(outcome.envelope.status, "recovered_from_session");
    assert.equal(outcome.envelope.tokens_source, "session_recovery");
    assert.ok(
      outcome.envelope.result && outcome.envelope.result.length > 0,
      "result must be non-empty",
    );
    // Recovery note must identify the new trigger introduced in PR #527.
    assert.ok(
      outcome.envelope.recovery_note.includes(
        "drained mid-conversation with pending tool calls",
      ),
      "recovery_note must use iterator_drain_pending_tools wording",
    );
    // Token counts must reflect the on-disk per-turn usage sum.
    assert.equal(outcome.envelope.input_tokens, 1750);
    assert.equal(outcome.envelope.output_tokens, 280);

    // ── Content contract: the fenced python block survived ───────────
    assert.ok(
      outcome.envelope.result.includes("```python"),
      "recovered result must include the fenced python block",
    );
    assert.ok(
      outcome.envelope.result.includes("def is_leap_year"),
      "recovered result must contain the function definition",
    );
    assert.ok(
      outcome.envelope.result.includes("year % 400 == 0"),
      "recovered result must contain the canonical predicate",
    );

    // ── Downstream-scoreable contract ────────────────────────────────
    // Use the JS port of extract_solution_files (Pattern-2, the fallback
    // path) to confirm the recovered text actually parses into a
    // {filename: content} dict — i.e. polyglot_harness will see a
    // non-empty solution and proceed past the no_solution short-circuit
    // at src/ksi/benchmarks/polyglot_harness.py:415-422.
    const solutionFiles = extractPythonSolutionFromFences(
      outcome.envelope.result,
    );
    assert.ok(
      Object.keys(solutionFiles).length > 0,
      "extract_solution_files-equivalent must return a non-empty dict — " +
        "this is what determines whether downstream scoring can score >0",
    );
    assert.ok(
      "solution.py" in solutionFiles,
      "Pattern-2 fallback assigns the default filename",
    );
    assert.ok(
      solutionFiles["solution.py"].includes("year % 4 == 0"),
      "extracted file content must contain the implementation",
    );
    assert.ok(
      solutionFiles["solution.py"].includes("year % 400 == 0"),
      "extracted file content must contain the full predicate",
    );
  });

  it("transcript with NO fenced block in the final assistant text → result recovered but NOT scoreable", () => {
    // Counterexample: iterator drained on a transcript whose final
    // assistant turn was just "Now writing the file." (no fenced block).
    // Recovery still succeeds (tokens + turn count are real), but the
    // downstream extractor returns {} — i.e. polyglot would short-circuit
    // to no_solution. This pins the exact contract: recovery preserves
    // whatever was in the last text-bearing turn, and SCOREABILITY depends
    // on the agent having put the answer there.
    //
    // Future change to (say) prefer the longest text block, or join all
    // assistant turns, would make this test fail — at which point the
    // person making the change must explicitly update the contract.
    const logPath = path.join(
      tmp,
      "projects",
      "-workspace-task",
      "session-noskip.jsonl",
    );
    writeJsonl(logPath, [
      {
        type: "user",
        message: {
          role: "user",
          content: "Solve the leap year exercism exercise.",
        },
      },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: FINAL_ANSWER_TEXT }],
          usage: { input_tokens: 900, output_tokens: 150 },
        },
      },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "text", text: "Now writing the file." },
            {
              type: "tool_use",
              id: "t3",
              name: "Edit",
              input: { file_path: "/workspace/task/leap.py" },
            },
          ],
          usage: { input_tokens: 80, output_tokens: 20 },
        },
      },
    ]);

    const outcome = maybeRecoverFromEmptyScheduledOutcome(
      baseRecoveryCtx(),
      () => recoverFromSessionLog(tmp),
    );
    // Recovery succeeds — turn count and tokens are real.
    assert.equal(outcome.envelope.status, "recovered_from_session");
    // But the LAST text-bearing turn was the preamble, not the answer.
    assert.equal(
      outcome.envelope.result,
      "Now writing the file.",
      "recoverFromSessionLog picks the last text-bearing turn — current contract",
    );
    // Downstream extractor returns {} → polyglot would short-circuit.
    const solutionFiles = extractPythonSolutionFromFences(
      outcome.envelope.result,
    );
    assert.equal(
      Object.keys(solutionFiles).length,
      0,
      "no fenced block in last text turn → no scoreable solution; " +
        "documents the gap for transcripts where the answer is not in the final turn",
    );
  });
});
