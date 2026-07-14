/**
 * Regression test for issue #945: a late SDK iterator error must NOT clobber a
 * run that already produced a result envelope.
 *
 * Flow (runtime_runner/agent-runner/src/query_runner.ts): the scheduled query
 * loop `break`s immediately after the SDK `result` event, at which point
 * `buildScheduledResultOutcome` has already `writeOutput`'d a success envelope.
 * But `break` triggers the async iterator's `.return()` cleanup, which can
 * throw and be caught by the `catch` around the for-await loop — setting
 * `iteratorError` even though `resultCount > 0`. The post-loop
 * `else if (iteratorError)` branch then emitted a follow-up status='error'
 * envelope. Because the host's streaming parser keeps the LAST parsed marker
 * pair (runtime_runner/src/container_output.ts: `lastParsed`), that error
 * overrode the earlier success — clobbering a good run and sending it down the
 * retry path.
 *
 * The fix guards the branch with `&& resultCount === 0`, so a late iterator
 * error only surfaces when no result was produced.
 *
 * agent-runner's SDK deps are not installed at `node --test` time (CI runs
 * `npm ci` only in runtime_runner, not runtime_runner/agent-runner), so this
 * follows the repo's inline-mirror idiom (see iterator_drain_recovery.test.mjs)
 * plus a SOURCE-PIN that anchors the mirror's branch chain to the real
 * query_runner.ts. If the guard is removed from the source, the source-pin
 * fails; if the mirror drifts from the source's branch ordering, the same pin
 * fails.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const QUERY_RUNNER_SRC = path.join(
  repoRoot,
  "runtime_runner",
  "agent-runner",
  "src",
  "query_runner.ts",
);

// ── Local copy of `hasAnyUsageDelta` from usage.ts ────────────────────────
// (a, b, c, d) => any of the four token counts is a positive number.
function hasAnyUsageDelta(a, b, c, d) {
  return [a, b, c, d].some((n) => typeof n === "number" && n > 0);
}

// ── DELIBERATELY TRIMMED mirror of `buildPostLoopOutcome`'s branch chain ──
// Source: runtime_runner/agent-runner/src/query_runner.ts. The helper bodies
// (maybeRecoverFromEmptyScheduledOutcome, buildSilentDiagnostic) are covered
// by silent_failure_recovery / iterator_drain_recovery; here we only need to
// know WHICH branch fires, so each branch returns a sentinel tag instead of
// the full envelope. The branch CONDITIONS (and their ordering) mirror the
// source exactly — that ordering + the `resultCount === 0` guard on the
// iterator-error branch is what this test pins. The source-pin block below
// asserts the real source still matches this chain.
function buildPostLoopOutcomeMirror(args) {
  const {
    resultCount,
    lastAssistantFallback,
    usage,
    iteratorError,
  } = args;

  if (resultCount === 0 && lastAssistantFallback) {
    return { branch: "fallback", envelope: { status: "success" } };
  } else if (
    resultCount === 0 &&
    !lastAssistantFallback &&
    !hasAnyUsageDelta(
      usage.resultInputTokens,
      usage.resultOutputTokens,
      usage.resultCacheCreationTokens,
      usage.resultCacheReadTokens,
    ) &&
    !hasAnyUsageDelta(
      usage.perTurnInputTokens,
      usage.perTurnOutputTokens,
      usage.perTurnCacheCreationTokens,
      usage.perTurnCacheReadTokens,
    )
  ) {
    return { branch: "silent_exit", envelope: { status: "error" } };
  } else if (iteratorError && resultCount === 0) {
    return { branch: "iterator_error", envelope: { status: "error" } };
  }
  return { branch: "fallthrough", envelope: null };
}

function baseUsage(overrides = {}) {
  return {
    perTurnInputTokens: 0,
    perTurnOutputTokens: 0,
    perTurnCacheCreationTokens: 0,
    perTurnCacheReadTokens: 0,
    resultInputTokens: 0,
    resultOutputTokens: 0,
    resultCacheCreationTokens: 0,
    resultCacheReadTokens: 0,
    ...overrides,
  };
}

describe("buildPostLoopOutcome — late iterator error (issue #945)", () => {
  it("resultCount>0 + late iteratorError → fallthrough (envelope:null), NOT a clobbering error", () => {
    // A result was produced; the loop already wrote a success envelope and
    // broke. `.return()` then threw, setting iteratorError. The post-loop
    // outcome must NOT emit anything.
    const outcome = buildPostLoopOutcomeMirror({
      resultCount: 1,
      lastAssistantFallback: "final answer text",
      usage: baseUsage({
        perTurnInputTokens: 1000,
        perTurnOutputTokens: 200,
        resultInputTokens: 1000,
        resultOutputTokens: 200,
      }),
      iteratorError: { name: "AbortError", message: "stream aborted during return()" },
    });
    assert.equal(outcome.branch, "fallthrough");
    assert.equal(
      outcome.envelope,
      null,
      "a late iterator error after a produced result must not write a " +
        "follow-up envelope that clobbers the loop's success write",
    );
  });

  it("resultCount=0 + iteratorError + per-turn tokens (no fallback) → iterator_error envelope (unchanged)", () => {
    // No result, no fallback text, but per-turn usage accumulated → the
    // iterator-error branch is still the correct envelope. The guard does not
    // suppress this case.
    const outcome = buildPostLoopOutcomeMirror({
      resultCount: 0,
      lastAssistantFallback: null,
      usage: baseUsage({ perTurnInputTokens: 500, perTurnOutputTokens: 50 }),
      iteratorError: { name: "AbortError", message: "aborted before any result" },
    });
    assert.equal(outcome.branch, "iterator_error");
    assert.equal(outcome.envelope.status, "error");
  });

  it("resultCount=0 + iteratorError + lastAssistantFallback → fallback branch handles it first (not iterator_error)", () => {
    // When there is fallback text, the first branch owns it (it already
    // threads iteratorError into recovery); the iterator-error branch must not
    // pre-empt it.
    const outcome = buildPostLoopOutcomeMirror({
      resultCount: 0,
      lastAssistantFallback: "some assistant text",
      usage: baseUsage({ perTurnInputTokens: 10, perTurnOutputTokens: 5 }),
      iteratorError: { name: "AbortError", message: "aborted with fallback present" },
    });
    assert.equal(outcome.branch, "fallback");
  });
});

// ── SOURCE-PIN: anchor the mirror to the real query_runner.ts ─────────────
// These assertions read the actual TypeScript source so the mirror above
// cannot silently diverge, and so removing the #945 guard fails CI.
describe("query_runner.ts source-pin (issue #945 guard)", () => {
  const src = fs.readFileSync(QUERY_RUNNER_SRC, "utf-8");

  it("the post-loop iterator-error branch is guarded by `resultCount === 0`", () => {
    // The branch must read `iteratorError && resultCount === 0` (order-
    // insensitive on whitespace). A bare `else if (iteratorError)` would
    // reintroduce the clobber.
    assert.match(
      src,
      /else if\s*\(\s*iteratorError\s*&&\s*resultCount === 0\s*\)/,
      "post-loop iterator-error branch must be guarded with `resultCount === 0` (#945)",
    );
    assert.doesNotMatch(
      src,
      /else if\s*\(\s*iteratorError\s*\)\s*\{/,
      "an unguarded `else if (iteratorError) {` branch reintroduces the #945 clobber",
    );
  });

  it("the branch chain ordering matches the mirror (fallback → silent-exit → iterator-error)", () => {
    const fallbackIdx = src.indexOf("if (resultCount === 0 && lastAssistantFallback)");
    const silentIdx = src.indexOf("// Silent-exit branch.");
    const iterIdx = src.indexOf("} else if (iteratorError && resultCount === 0) {");
    assert.ok(fallbackIdx !== -1, "fallback branch present");
    assert.ok(silentIdx !== -1, "silent-exit branch present");
    assert.ok(iterIdx !== -1, "guarded iterator-error branch present");
    assert.ok(
      fallbackIdx < silentIdx && silentIdx < iterIdx,
      "branch order must be fallback → silent-exit → iterator-error",
    );
  });
});
