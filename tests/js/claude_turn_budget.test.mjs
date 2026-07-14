import { describe, it } from "node:test";
import assert from "node:assert/strict";

// Mirrors runtime_runner/agent-runner/src/index.ts:resolveClaudeMaxTurns().
// Faithful type-stripped mirror (the `_taskSource` parameter is kept for API
// stability with existing call sites and ignored by the unified 150-cap
// policy); drift-guarded by copy_sync_guard.test.mjs.
function resolveClaudeMaxTurns(
  _taskSource,
  envOverride,
) {
  const override = Number(envOverride);
  if (Number.isFinite(override) && override > 0) {
    return Math.floor(override);
  }
  // Universal default: every scheduled task gets 150 turns to match the OpenAI
  // agents-sdk path. Override via KCSI_CLAUDE_MAX_TURNS.
  return 150;
}

describe("resolveClaudeMaxTurns", () => {
  it("returns the unified 150-turn cap across all task sources", () => {
    assert.equal(resolveClaudeMaxTurns("per_task_forum", undefined), 150);
    assert.equal(resolveClaudeMaxTurns("arc", undefined), 150);
    assert.equal(resolveClaudeMaxTurns("polyglot", undefined), 150);
    assert.equal(resolveClaudeMaxTurns("swebench_pro", undefined), 150);
    assert.equal(resolveClaudeMaxTurns("terminal_bench_2", undefined), 150);
    assert.equal(resolveClaudeMaxTurns("", undefined), 150);
    assert.equal(resolveClaudeMaxTurns(undefined, undefined), 150);
  });

  it("honors positive env overrides and ignores invalid overrides", () => {
    assert.equal(resolveClaudeMaxTurns("swebench_pro", "200"), 200);
    assert.equal(resolveClaudeMaxTurns("polyglot", "1"), 1);
    assert.equal(resolveClaudeMaxTurns("arc", "0"), 150);
    assert.equal(resolveClaudeMaxTurns("arc", "-3"), 150);
    assert.equal(resolveClaudeMaxTurns("arc", "abc"), 150);
    assert.equal(resolveClaudeMaxTurns("arc", ""), 150);
  });
});

// Mirrors runtime_runner/agent-runner/src/query_config.ts:resolveTurnBudgets();
// drift-guarded by copy_sync_guard.test.mjs.
function resolveTurnBudgets(taskSource, sdkEnv) {
  const scheduledMaxTurns = resolveClaudeMaxTurns(taskSource, sdkEnv.KCSI_CLAUDE_MAX_TURNS);
  const defaultMaxMessages = 150;
  const messagesOverride = Number(sdkEnv.KCSI_CLAUDE_MAX_MESSAGES);
  const scheduledMaxMessages = Math.max(
    1,
    Number.isFinite(messagesOverride) && messagesOverride > 0
      ? messagesOverride
      : defaultMaxMessages,
  );
  return { scheduledMaxTurns, scheduledMaxMessages };
}

describe("resolveTurnBudgets — message ceiling", () => {
  it("no longer gives ARC a lower message ceiling than its own turn cap", () => {
    // Regression: ARC's defaultMaxMessages was left at 80 when maxTurns was
    // raised 25 -> 150 (native/no-MCP ARC sessions reliably hit 80 raw
    // messages ~25-27 Bash round-trips before 150 turns, forfeiting the
    // task before ever writing a prediction file). Must match maxTurns.
    const { scheduledMaxTurns, scheduledMaxMessages } = resolveTurnBudgets("arc", {});
    assert.equal(scheduledMaxMessages, 150);
    assert.equal(scheduledMaxMessages, scheduledMaxTurns);
  });

  it("no longer gives per_task_forum a lower message ceiling than its own turn cap (#1049)", () => {
    // Regression: per_task_forum's defaultMaxMessages was left at 60 when
    // #1037 raised every OTHER task source's ceiling to 150 to match
    // maxTurns. Same bug shape, same fix.
    const { scheduledMaxTurns, scheduledMaxMessages } = resolveTurnBudgets("per_task_forum", {});
    assert.equal(scheduledMaxMessages, 150);
    assert.equal(scheduledMaxMessages, scheduledMaxTurns);
  });

  it("uses 150 for every other task source", () => {
    for (const src of ["polyglot", "swebench_pro", "terminal_bench_2", "cross_task_forum", "", undefined]) {
      assert.equal(resolveTurnBudgets(src, {}).scheduledMaxMessages, 150);
    }
  });

  it("honors KCSI_CLAUDE_MAX_MESSAGES override and ignores invalid values", () => {
    assert.equal(resolveTurnBudgets("arc", { KCSI_CLAUDE_MAX_MESSAGES: "40" }).scheduledMaxMessages, 40);
    assert.equal(resolveTurnBudgets("arc", { KCSI_CLAUDE_MAX_MESSAGES: "0" }).scheduledMaxMessages, 150);
    assert.equal(resolveTurnBudgets("arc", { KCSI_CLAUDE_MAX_MESSAGES: "abc" }).scheduledMaxMessages, 150);
  });

  it("standing regression: messageCeiling >= turnCap for every registered task source (#1049)", () => {
    // Guards the bug CLASS #1037/#1049 kept rediscovering one task source at
    // a time: a message ceiling tighter than the turn cap silently forces a
    // fallback stop before the agent finishes, regardless of which task
    // source it happens to be set for. Any future task source that adds a
    // per-source defaultMaxMessages below 150 without also raising its
    // maxTurns will fail here.
    for (const src of [
      "arc",
      "polyglot",
      "swebench_pro",
      "terminal_bench_2",
      "per_task_forum",
      "cross_task_forum",
      "",
      undefined,
    ]) {
      const { scheduledMaxTurns, scheduledMaxMessages } = resolveTurnBudgets(src, {});
      assert.ok(
        scheduledMaxMessages >= scheduledMaxTurns,
        `task source ${String(src)}: messageCeiling (${scheduledMaxMessages}) < turnCap (${scheduledMaxTurns})`,
      );
    }
  });
});
