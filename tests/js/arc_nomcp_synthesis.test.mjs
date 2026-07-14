/**
 * --arc-no-mcp trace synthesis (issue #694).
 *
 * `runtime_runner/src/arc_nomcp_synth.ts` reads per-test ASCII prediction
 * files from the workspace and appends a synthetic ARC tool trace that the
 * Python scorer reconstructs. Before this fix the synthesizer only handled
 * test 0 (no `arc_next_test_input` event), so multi-test ARC tasks capped at
 * <=0.5 / resolved=False in native (default) mode.
 *
 * We drive the real TypeScript through `tsx` (like
 * session_state_path_guard.test.mjs) so we exercise the shipped code, not a
 * copy. Skipped if `runtime_runner/node_modules` is not installed.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");
const tsxBin = path.join(
  repoRoot,
  "runtime_runner",
  "node_modules",
  ".bin",
  process.platform === "win32" ? "tsx.cmd" : "tsx",
);
const tsxSkip = !fs.existsSync(tsxBin);

function synthesize(wsRoot) {
  const source = `
    import { synthesizeArcSubmitTraceFromWorkspaceDir } from "./runtime_runner/src/arc_nomcp_synth.ts";
    const trace = [];
    synthesizeArcSubmitTraceFromWorkspaceDir(${JSON.stringify(wsRoot)}, trace);
    console.log(JSON.stringify(trace.map((e) => ({
      tool_name: e.tool_name,
      tool_output: e.tool_output,
      grid: e.tool_input && e.tool_input.grid,
    }))));
  `;
  const result = spawnSync(tsxBin, ["--input-type=module", "--eval", source], {
    encoding: "utf8",
    cwd: repoRoot,
  });
  assert.equal(
    result.status,
    0,
    `tsx exited nonzero: ${result.status}\n${result.stderr}`,
  );
  return JSON.parse(result.stdout.trim());
}

describe("arc-no-mcp trace synthesis", () => {
  it(
    "emits set/submit/next_test/set/submit for a 2-test task",
    { skip: tsxSkip },
    () => {
      const ws = fs.mkdtempSync(path.join(os.tmpdir(), "arc-nomcp-multi-"));
      fs.writeFileSync(path.join(ws, "attempt_0_1.txt"), "1\n");
      fs.writeFileSync(path.join(ws, "attempt_0_2.txt"), "1\n");
      fs.writeFileSync(path.join(ws, "attempt_1_1.txt"), "2\n");
      fs.writeFileSync(path.join(ws, "attempt_1_2.txt"), "2\n");

      const trace = synthesize(ws);
      const names = trace.map((e) => e.tool_name);
      // test 0: two trials, then advance, then test 1: two trials.
      assert.deepEqual(names, [
        "arc_set_output_grid",
        "arc_submit_trial",
        "arc_set_output_grid",
        "arc_submit_trial",
        "arc_next_test_input",
        "arc_set_output_grid",
        "arc_submit_trial",
        "arc_set_output_grid",
        "arc_submit_trial",
      ]);

      // First submit carries test_index 0; the next_test announces index 1;
      // submits after it carry test_index 1.
      const firstSubmit = JSON.parse(trace[1].tool_output);
      assert.equal(firstSubmit.test_index, 0);
      const nextTest = JSON.parse(trace[4].tool_output);
      assert.equal(nextTest.status, "ok");
      assert.equal(nextTest.current_test_index, 1);
      const lastSubmit = JSON.parse(trace[8].tool_output);
      assert.equal(lastSubmit.test_index, 1);
    },
  );

  it(
    "legacy attempt_1/attempt_2 files produce a single-test trace (no next_test)",
    { skip: tsxSkip },
    () => {
      const ws = fs.mkdtempSync(path.join(os.tmpdir(), "arc-nomcp-legacy-"));
      fs.writeFileSync(path.join(ws, "attempt_1.txt"), "1 2\n3 4\n");
      fs.writeFileSync(path.join(ws, "attempt_2.txt"), "1 2\n3 4\n");

      const trace = synthesize(ws);
      const names = trace.map((e) => e.tool_name);
      // Byte-for-byte the pre-multi-test shape: two set/submit pairs, no advance.
      assert.deepEqual(names, [
        "arc_set_output_grid",
        "arc_submit_trial",
        "arc_set_output_grid",
        "arc_submit_trial",
      ]);
      assert.ok(!names.includes("arc_next_test_input"));
      for (const entry of trace) {
        if (entry.tool_name === "arc_submit_trial") {
          assert.equal(JSON.parse(entry.tool_output).test_index, 0);
        }
      }
    },
  );

  it(
    "empty/sentinel-only workspace leaves the trace untouched",
    { skip: tsxSkip },
    () => {
      const ws = fs.mkdtempSync(path.join(os.tmpdir(), "arc-nomcp-empty-"));
      fs.writeFileSync(path.join(ws, "attempt_1.txt"), "__NOT_SUBMITTED__\n");
      fs.writeFileSync(path.join(ws, "attempt_2.txt"), "__NOT_SUBMITTED__\n");
      const trace = synthesize(ws);
      assert.deepEqual(trace, []);
    },
  );
});
