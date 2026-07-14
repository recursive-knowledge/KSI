/**
 * Regression tests for the Anthropic direct forum adapter's prompt-cache
 * architecture.
 *
 * Background: prior to this fix, `anthropic_direct_forum.ts` sent system and
 * the initial user message as plain strings (no `cache_control`) and called
 * `compactConsumedToolResults(messages, ..., 2)` after every turn — which
 * rewrites earlier `tool_result.content` in place. Those two together
 * produced `cache_creation_input_tokens = cache_read_input_tokens = 0`
 * across every forum_round in production (verified across 12+ post-fix
 * audit DBs in the 2026-04-28 audit).
 *
 * The fix (originally shared with the now-removed direct-ARC adapter):
 *   - Block-form system + initial user, both with `cache_control: ephemeral`.
 *   - No mid-loop compaction (caching is the sole cost-control mechanism).
 *   - Rolling `cache_control` marker on the most recent tool_result, with
 *     the previous turn's marker stripped. Text-block markers (the cache
 *     floor) are deliberately spared.
 *
 * These tests pin the architectural invariants so they cannot silently
 * regress (same-shape coverage as `arc_direct_adapter_inline_grids.test.mjs`).
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');

const directForumRunner = fs.readFileSync(
  path.join(repoRoot, 'runtime_runner', 'agent-runner', 'src', 'anthropic_direct_forum.ts'),
  'utf-8',
);

// Drive the real (exported) `buildScheduledGuidancePrefix` through tsx so the
// cross-agent invariance is tested against the shipped TypeScript, not a copy
// (same pattern as openai_forum_mcp_only_tools.test.mjs). Skipped only if
// `runtime_runner/node_modules` (tsx) is not installed; the always-run source
// pins below still guard the invariant in that case.
const tsxBin = path.join(
  repoRoot,
  'runtime_runner',
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'tsx.cmd' : 'tsx',
);
const tsxSkip = !fs.existsSync(tsxBin);

// Return `buildScheduledGuidancePrefix` output for a given task source + agent.
function guidancePrefixFor({ taskSource, agentId, generation = 3 }) {
  const source = `
    import { buildScheduledGuidancePrefix } from "./runtime_runner/agent-runner/src/anthropic_direct_forum.ts";
    const out = buildScheduledGuidancePrefix({
      memoryMcp: {
        taskSource: ${JSON.stringify(taskSource)},
        forumAgentId: ${JSON.stringify(agentId)},
        forumGeneration: ${JSON.stringify(generation)},
        forumTaskIds: ["task-alpha"],
      },
      assistantName: ${JSON.stringify(agentId)},
    });
    console.log(JSON.stringify(out));
  `;
  const result = spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    encoding: 'utf8',
    cwd: repoRoot,
  });
  assert.equal(result.status, 0, `tsx exited nonzero: ${result.status}\n${result.stderr}`);
  return JSON.parse(result.stdout.trim());
}

function sharedContainerMarkerCounts() {
  const source = `
    import { clearRollingCacheControl } from "./runtime_runner/agent-runner/src/anthropic_direct_forum.ts";

    const messages = [{
      role: "user",
      content: [
        { type: "text", text: "INITIAL PREFIX", cache_control: { type: "ephemeral" } },
        { type: "text", text: "INITIAL SUFFIX" },
      ],
    }];

    const countMarkers = () => 1 + messages
      .flatMap((msg) => Array.isArray(msg.content) ? msg.content : [])
      .filter((block) => block && block.cache_control)
      .length;

    const placeRollingMarker = () => {
      const msg = messages[messages.length - 1];
      const block = msg.content[msg.content.length - 1];
      block.cache_control = { type: "ephemeral" };
    };

    clearRollingCacheControl(messages);
    placeRollingMarker();
    const r0Turn1 = countMarkers();

    messages.push({
      role: "user",
      content: [{ type: "text", text: "R1 synthetic prompt" }],
    });
    clearRollingCacheControl(messages);
    placeRollingMarker();
    const r1Turn1 = countMarkers();

    messages.push({
      role: "assistant",
      content: [{ type: "tool_use", id: "toolu_1", name: "forum_post", input: {} }],
    });
    messages.push({
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "toolu_1", content: [{ type: "text", text: "ok" }] }],
    });
    clearRollingCacheControl(messages);
    placeRollingMarker();
    const r1Turn2 = countMarkers();

    console.log(JSON.stringify([r0Turn1, r1Turn1, r1Turn2]));
  `;
  const result = spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    encoding: 'utf8',
    cwd: repoRoot,
  });
  assert.equal(result.status, 0, `tsx exited nonzero: ${result.status}\n${result.stderr}`);
  return JSON.parse(result.stdout.trim());
}

describe('Anthropic direct forum adapter — cached-prefix architecture', () => {
  it('initial user message is block-form with cache_control on a text block', () => {
    // Block form is required so `cache_control` can attach to the text
    // block (Anthropic only honors block-level cache_control on
    // structured content).
    assert.match(
      directForumRunner,
      /const initialUserBlocks(?:\s*:\s*[^=]+)?\s*=\s*\[\s*\{\s*type: 'text' as const,/,
    );
    assert.match(
      directForumRunner,
      /cache_control: \{ type: 'ephemeral' as const \},?\s*\},?\s*\];/,
    );
    // The messages array uses the block form (not the legacy string form).
    assert.match(
      directForumRunner,
      /\{\s*role:\s*'user',\s*content:\s*initialUserBlocks(?:\s+as\s+[^\}]+)?\s*\}/,
    );
  });

  it('system prompt is sent as a cached block array, not a plain string', () => {
    // The system field passed to messages.create must be a block array with
    // cache_control — passing the raw string return of buildSystemPrompt()
    // would not opt into caching.
    assert.match(
      directForumRunner,
      /const cachedSystem = \[\s*\{\s*type: 'text' as const, text: buildSystemPrompt\(\), cache_control: \{ type: 'ephemeral' as const \} \},?\s*\];/,
    );
    assert.match(directForumRunner, /system:\s*(?:args\.)?cachedSystem,/);
    // Defensive: the old `system: buildSystemPrompt(),` string form must
    // not return.
    assert.doesNotMatch(directForumRunner, /system:\s*buildSystemPrompt\(\),/);
  });

  it('clearRollingCacheControl spares only initial-user text-block markers', () => {
    // The initial-message text-block check is the load-bearing invariant:
    // without it, the initial-user cache_control would be stripped on turn 2
    // onward and the cache floor would collapse. Later synthetic text turns
    // must NOT be spared, or shared-container R1 can exceed Anthropic's
    // four-breakpoint cap after the first R1 tool call.
    const fnMatch = directForumRunner.match(
      /export function clearRollingCacheControl\(messages: AnthropicMessage\[\]\)[^]*?\n\}\n/,
    );
    assert.ok(fnMatch, 'clearRollingCacheControl should be defined');
    const body = fnMatch[0];
    assert.match(body, /msgIndex === 0 && blockObj\.type === 'text'/);
    assert.match(body, /continue;/);
  });

  it('shared-container R1 stays within Anthropic four-breakpoint cap', { skip: tsxSkip }, () => {
    // Before the fix, clearRollingCacheControl spared every text block. The R1
    // synthetic text prompt kept its cache_control marker after the first R1
    // tool call, so adding the rolling tool_result marker produced five
    // explicit markers: system + initial prefix + initial suffix + R1 text +
    // tool_result. Anthropic rejects requests with more than four.
    assert.deepEqual(sharedContainerMarkerCounts(), [3, 4, 4]);
  });

  it('does NOT call compactConsumedToolResults in the message loop', () => {
    // Compaction's rewrite-in-place behavior on consumed tool_results was
    // the root cause of cache_read=0 on Haiku ARC (issue #535) and is the
    // same pathology that produced cache_read=0 on forum runs. Removing
    // the call is the architectural fix; prompt caching is now the sole
    // cost-control mechanism.
    assert.doesNotMatch(directForumRunner, /compactConsumedToolResults\(/);
    // The import should also be gone.
    assert.doesNotMatch(
      directForumRunner,
      /import .*compactConsumedToolResults.* from/,
    );
  });

  it('places exactly one rolling cache_control marker per turn', () => {
    // Floor count: cached system + initial-user text + rolling tool_result.
    // System block and initial user message use object-literal form
    // (`cache_control: { type: 'ephemeral' as const }`); the rolling
    // marker uses assignment form (`block.cache_control = ...`). Count
    // both shapes — three placement sites is the invariant.
    const literals = directForumRunner.match(/cache_control:\s*\{\s*type:\s*'ephemeral'/g) || [];
    const assignments = directForumRunner.match(/\.cache_control\s*=\s*\{\s*type:\s*'ephemeral'/g) || [];
    assert.equal(literals.length, 2, 'system + initial-user text block carry literal cache_control');
    assert.equal(assignments.length, 1, 'rolling marker uses assignment form');
  });

  it('releases the R0 suffix cache_control before the R1 synthetic turn (stays under the 4-marker cap)', () => {
    // Regression: the R0 initial-user *suffix* text block accumulates a
    // permanent rolling marker (clearRollingCacheControl spares text blocks).
    // Without clearing it before R1, the R1 prompt's marker makes 5
    // simultaneous cache_control markers on R1's 2nd turn — over Anthropic's
    // cap of 4 — so every multi-turn cross-task R1 round 400s and is silently
    // dropped (r1_captured=false). The fix deletes cache_control from the
    // initial-user NON-prefix blocks (index >= 1, sparing the block-0 prefix
    // floor) BEFORE pushing the r1PromptText message.
    const clearIdx = directForumRunner.indexOf('messages[0]?.content');
    const r1PushIdx = directForumRunner.indexOf('text: r1PromptText');
    assert.ok(clearIdx > 0, 'R1 must inspect messages[0].content to release the stale suffix marker');
    assert.ok(r1PushIdx > 0, 'R1 synthetic prompt push should be present');
    assert.ok(clearIdx < r1PushIdx, 'the suffix marker must be cleared BEFORE the R1 prompt is pushed');
    const region = directForumRunner.slice(clearIdx, r1PushIdx);
    assert.match(region, /for \(let i = 1;/, 'clearing loop must skip block 0 (the prefix cache floor)');
    assert.match(region, /delete block\.cache_control;/, 'clearing loop must delete the stale marker');
  });

  it('R0 turn-1 caches only the shared prefix — the variable suffix is never cache-marked (deep-review #1264 High 1)', () => {
    // Bug: the initial user message is [prefix (cache_control at
    // construction), suffix (no marker)]. The suffix is the VARIABLE,
    // agent/round-specific guidance (the cross-task agent identity lives
    // here per #1259). On R0 turn 1 the last user message IS the initial
    // user message, so the rolling-marker logic in `runForumRound` used to
    // mark its LAST block — the variable suffix — as cacheable. Because it
    // is `type:'text'`, `clearRollingCacheControl` never strips it, so the
    // FIRST turn cache-WROTE agent/round-specific content that no other
    // agent can cross-read — defeating the cross-agent caching intent of
    // #1259 on turn 1. (Cost bug, not correctness: block 0's shared-prefix
    // cache still fires.)
    //
    // The fix restricts the rolling marker to `tool_result` blocks. This
    // test pins BOTH halves of the invariant: (1) the shared prefix (block
    // 0) IS cache-marked at construction while the variable suffix block is
    // pushed with NO marker; (2) the rolling marker assignment is guarded
    // by a `type === 'tool_result'` check, so it can never land on the R0
    // suffix (or the R1 prompt) text block on turn 1.

    // (1) Construction: shared prefix block carries cache_control; the
    //     variable suffix block does NOT.
    assert.match(
      directForumRunner,
      /const initialUserBlocks(?:\s*:\s*[^=]+)?\s*=\s*\[\s*\{\s*type: 'text' as const,\s*text: initialUserPrefix,\s*cache_control: \{ type: 'ephemeral' as const \},/,
      'shared prefix (block 0) must carry cache_control at construction',
    );
    const suffixPushMatch = directForumRunner.match(
      /initialUserBlocks\.push\(\{[^]*?\}\);/,
    );
    assert.ok(suffixPushMatch, 'variable suffix push should be present');
    assert.doesNotMatch(
      suffixPushMatch[0],
      /cache_control/,
      'the variable suffix block must be pushed WITHOUT cache_control',
    );

    // (2) Rolling logic: the marker assignment must be guarded by a
    //     tool_result type check so it never marks a text block.
    const loopMatch = directForumRunner.match(
      /for \(let turn = 1; turn <= (?:args\.)?maxTurns; turn\+\+\) \{[^]*?\n  {0,4}\}/,
    );
    assert.ok(loopMatch, 'turn loop should be present');
    const loopBody = loopMatch[0];
    const guardIdx = loopBody.indexOf("lastBlock.type === 'tool_result'");
    const assignIdx = loopBody.indexOf("lastBlock.cache_control = { type: 'ephemeral' }");
    assert.ok(guardIdx > 0, 'rolling marker must be guarded by a tool_result type check');
    assert.ok(assignIdx > 0, 'rolling marker assignment should be present');
    assert.ok(
      guardIdx < assignIdx,
      'the tool_result guard must precede the rolling marker assignment',
    );
  });

  it('scheduled-guidance prefix excludes the round label / round-conditional hint', () => {
    // PR #572 follow-up: the scheduled guidance was placed entirely
    // inside the cached prefix block, but its first line embedded
    // `round ${round}` and a round-conditional `liveBoardGuidance`
    // line. Both vary per round for a fixed agent/generation, which
    // mutates the cached prefix hash between rounds and defeats
    // cross-round cache reuse.
    //
    // The fix splits the guidance into a stable prefix
    // (`buildScheduledGuidancePrefix`) — agentId + generation only —
    // and a round-specific suffix (`buildScheduledGuidanceSuffix`).
    // The cache_control marker stays on the prefix; the suffix sits
    // in the variable suffix block.
    //
    // These regex-level assertions pin that the prefix function does
    // NOT reference `round` and DOES reference `agentId` /
    // `generation`, while the suffix function DOES reference `round`.
    const prefixFnMatch = directForumRunner.match(
      /export function buildScheduledGuidancePrefix\(containerInput: ContainerInput\): string \{[^]*?\n\}\n/,
    );
    assert.ok(prefixFnMatch, 'buildScheduledGuidancePrefix should be defined');
    const prefixBody = prefixFnMatch[0];
    // Prefix must NOT mention round at all (no template literal, no
    // free-standing token) — that is the load-bearing invariant.
    assert.doesNotMatch(prefixBody, /round/i);
    // Must still anchor the generation scope (agent-invariant, so safe in
    // the cached prefix).
    assert.match(prefixBody, /\$\{generation\}/);

    // Cross-task branch must NOT embed `${agentId}`: that block is
    // concatenated into the cache_control-marked prefix and must be
    // byte-identical across all agents in a (generation, round) for
    // cross-AGENT cache reuse to fire. (Whole-function `${agentId}` checks
    // are useless here — the per-task branch legitimately keeps it, which is
    // exactly how the cross-task leak slipped past review before.)
    const crossTaskBlockMatch = prefixBody.match(
      /if \(taskSource === 'cross_task_forum'\) \{[^]*?\n {2}\}/,
    );
    assert.ok(crossTaskBlockMatch, 'cross_task_forum branch should be present');
    assert.doesNotMatch(
      crossTaskBlockMatch[0],
      /\$\{agentId\}/,
      'cross-task guidance prefix must not embed agentId (breaks cross-agent cache)',
    );
    // The per-task branch DOES keep agentId (within-agent caching by design).
    assert.match(prefixBody, /\$\{agentId\}/);

    const suffixFnMatch = directForumRunner.match(
      /function buildScheduledGuidanceSuffix\(containerInput: ContainerInput\): string \{[^]*?\n\}\n/,
    );
    assert.ok(suffixFnMatch, 'buildScheduledGuidanceSuffix should be defined');
    const suffixBody = suffixFnMatch[0];
    // Suffix must carry the round info.
    assert.match(suffixBody, /\$\{round\}/);
    assert.match(suffixBody, /forumRound/);
  });

  it('buildInitialPrefixText routes guidance suffix into the variable suffix block', () => {
    // The cached `prefix` half of the return value must include only
    // the cache-stable guidance (`guidancePrefix`); the round-varying
    // `guidanceSuffix` must be prepended to the variable suffix block
    // so the cache_control marker (which sits on the prefix block)
    // does not see round mutations.
    const fnMatch = directForumRunner.match(
      /function buildInitialPrefixText\([^]*?\n\}\n/,
    );
    assert.ok(fnMatch, 'buildInitialPrefixText should be defined');
    const body = fnMatch[0];
    // Cached `prefix` array must contain `guidancePrefix`, not
    // `guidanceSuffix`.
    assert.match(body, /const prefix = \[[^]*?guidancePrefix[^]*?\]\.join/);
    assert.doesNotMatch(body, /const prefix = \[[^]*?guidanceSuffix[^]*?\]\.join/);
    // Suffix-side composition must include `guidanceSuffix`.
    assert.match(body, /guidanceSuffix/);
  });

  it('rolls the cache_control marker forward inside the turn loop', () => {
    // The rolling marker placement must happen INSIDE `for (let turn ...)`
    // so each turn strips the previous marker and places a new one on the
    // most-recent tool_result. Placing it outside the loop would only
    // mark the initial user (already covered by the static cache_control)
    // and never advance the rolling breakpoint.
    const loopMatch = directForumRunner.match(
      /for \(let turn = 1; turn <= (?:args\.)?maxTurns; turn\+\+\) \{[^]*?\n  {0,4}\}/,
    );
    assert.ok(loopMatch, 'turn loop should be present');
    const loopBody = loopMatch[0];
    assert.match(loopBody, /clearRollingCacheControl\((?:args\.)?messages\);/);
    assert.match(loopBody, /lastBlock\.cache_control = \{ type: 'ephemeral' \}/);
  });

  it('cross-task guidance prefix is byte-identical across agents (behavioral)', { skip: tsxSkip }, () => {
    // The real payoff test: exercise the shipped `buildScheduledGuidancePrefix`
    // for two different agents and assert the cross-task output is identical.
    // This is what actually fires the cross-agent prompt-cache read — the
    // Python-side `cacheable_prefix` fix (build_cross_task_discussion_parts)
    // is necessary but not sufficient because this guidance block is
    // concatenated INTO the same cache_control-marked prefix.
    const a = guidancePrefixFor({ taskSource: 'cross_task_forum', agentId: 'agent-0' });
    const b = guidancePrefixFor({ taskSource: 'cross_task_forum', agentId: 'agent-7' });
    assert.equal(a, b, 'cross-task guidance prefix must not vary by agentId');
    assert.doesNotMatch(a, /agent-0|agent-7/, 'no agent id should appear in cross-task prefix');
    assert.match(a, /generation 3/, 'generation is retained (agent-invariant)');
  });

  it('per-task guidance prefix DOES vary by agent (within-agent caching, by design)', { skip: tsxSkip }, () => {
    // Contrast case: per-task forum deliberately keeps agentId in the cached
    // prefix (each agent discusses only its own task pages), so it targets
    // within-agent/cross-round reuse, not cross-agent reuse.
    const a = guidancePrefixFor({ taskSource: 'per_task_forum', agentId: 'agent-0' });
    const b = guidancePrefixFor({ taskSource: 'per_task_forum', agentId: 'agent-7' });
    assert.notEqual(a, b, 'per-task guidance prefix is intentionally per-agent');
    assert.match(a, /agent-0/);
  });
});
