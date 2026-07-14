/**
 * Runtime session-id path handling in `runtime_runner/src/sessions.ts` must
 * treat agent IDs as opaque values before touching disk.
 *
 * The implementation is executed through `tsx` so we exercise the real
 * TypeScript behavior in `runtime_runner/src/sessions.ts` (not a handcrafted
 * copy). If `runtime_runner/node_modules` has not been installed, this test
 * is intentionally skipped.
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

function runSessionsFixture(scriptSource, tempRoot) {
  return spawnSync(
    tsxBin,
    ["--input-type=module", "--eval", scriptSource],
    {
      env: {
        ...process.env,
        KSI_SESSION_STATE_ROOT: tempRoot,
      },
      encoding: "utf8",
      cwd: repoRoot,
    },
  );
}

describe("runtime session state path guard", () => {
  it("builds a stable path for safe agent ids", { skip: tsxSkip }, () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "ksi-session-root-"));
    const source = `
      import { sessionStatePath } from "./runtime_runner/src/sessions.ts";
      import path from "node:path";
      const p = sessionStatePath("agent.safe-id_01");
      console.log(JSON.stringify({
        path: p,
        dir: path.basename(path.dirname(p)),
      }));
    `;

    const result = runSessionsFixture(source, tempRoot);
    assert.equal(result.status, 0, `tsx exited nonzero: ${result.status}\n${result.stderr}`);
    const payload = JSON.parse(result.stdout.trim());
    assert.ok(payload.path.startsWith(tempRoot));
    assert.match(payload.dir, /^agent-[0-9a-f]{64}$/);
  });

  it("does not let path syntax alias distinct agent ids", { skip: tsxSkip }, () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "ksi-session-root-"));
    const source = `
      import { sessionStatePath } from "./runtime_runner/src/sessions.ts";
      const root = process.env.KSI_SESSION_STATE_ROOT;
      const ids = [
        "victim",
        "attacker/../victim",
        root + "/victim",
      ];
      console.log(JSON.stringify(ids.map((id) => sessionStatePath(id))));
    `;

    const result = runSessionsFixture(source, tempRoot);
    assert.equal(result.status, 0, `tsx exited nonzero: ${result.status}\n${result.stderr}`);
    const paths = JSON.parse(result.stdout.trim());
    assert.equal(new Set(paths).size, paths.length);
    for (const sessionPath of paths) {
      assert.ok(sessionPath.startsWith(tempRoot));
      assert.equal(path.basename(sessionPath), ".ksi_session.json");
      assert.match(path.basename(path.dirname(sessionPath)), /^agent-[0-9a-f]{64}$/);
    }
  });

  it("keeps session load/save isolated for path-like agent ids", { skip: tsxSkip }, () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "ksi-session-root-"));
    const source = `
      import { loadSessionForAgent, saveSessionForAgent, sessionStatePath } from "./runtime_runner/src/sessions.ts";

      saveSessionForAgent("victim", "session-123");
      saveSessionForAgent("attacker/../victim", "session-alias");

      console.log(JSON.stringify({
        victimSession: loadSessionForAgent("victim"),
        aliasSession: loadSessionForAgent("attacker/../victim"),
        victimPath: sessionStatePath("victim"),
        aliasPath: sessionStatePath("attacker/../victim"),
      }));
    `;

    const result = runSessionsFixture(source, tempRoot);
    assert.equal(result.status, 0, `tsx exited nonzero: ${result.status}\n${result.stderr}`);
    const payload = JSON.parse(result.stdout.trim());
    assert.equal(payload.victimSession, "session-123");
    assert.equal(payload.aliasSession, "session-alias");
    assert.notEqual(payload.victimPath, payload.aliasPath);
  });

  it("ignores blank agent ids", { skip: tsxSkip }, () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "ksi-session-root-"));
    const source = `
      import { loadSessionForAgent, saveSessionForAgent } from "./runtime_runner/src/sessions.ts";
      import fs from "node:fs";

      saveSessionForAgent("   ", "bad-session");
      console.log(JSON.stringify({
        blankLoad: loadSessionForAgent("   "),
        rootEntries: fs.existsSync(process.env.KSI_SESSION_STATE_ROOT)
          ? fs.readdirSync(process.env.KSI_SESSION_STATE_ROOT)
          : [],
      }));
    `;

    const result = runSessionsFixture(source, tempRoot);
    assert.equal(result.status, 0, `tsx exited nonzero: ${result.status}\n${result.stderr}`);
    const payload = JSON.parse(result.stdout.trim());
    assert.equal(payload.blankLoad, undefined);
    assert.deepEqual(payload.rootEntries, []);
  });
});
