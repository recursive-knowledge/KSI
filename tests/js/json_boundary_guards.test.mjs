/**
 * Guards for the JSON process-boundary validators added in #859:
 *   - main.ts          : assertKsiPayload, coerceOptionalNumber, coerceOptionalString
 *   - container_output.ts : isContainerOutput
 *
 * The repo's JS harness runs Node directly (no tsc step), so we keep inline JS
 * copies of these TypeScript helpers and (a) drift-guard each copy against its
 * TS source via _sync_check.mjs, and (b) exercise the runtime behavior. See
 * tests/js/copy_sync_guard.test.mjs for the same pattern.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { compareFunctions } from "./_sync_check.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const readRepo = (...p) => fs.readFileSync(path.join(repoRoot, ...p), "utf-8");

const mainTs = readRepo("runtime_runner", "src", "main.ts");
const containerOutputTs = readRepo("runtime_runner", "src", "container_output.ts");
const selfSource = fs.readFileSync(fileURLToPath(import.meta.url), "utf-8");

// ── Inline JS copies (must mirror the TS sources; the drift guards below
//    enforce it). Types are stripped; bodies are otherwise verbatim. ──────────

function die(msg) {
  throw new Error(msg);
}

const CONTAINER_OUTPUT_STATUSES = new Set([
  "success",
  "error",
  "recovered_from_session",
]);

function assertKsiPayload(value, filePath) {
  function fail(pathName, expectation) {
    die(`payload at ${filePath} ${pathName} must be ${expectation}`);
  }
  function requireString(obj, key, pathName) {
    if (typeof obj[key] !== "string" || !obj[key]) {
      fail(`${pathName}.${key}`, "a non-empty string");
    }
  }
  function optionalString(obj, key, pathName) {
    if (obj[key] !== undefined && typeof obj[key] !== "string") {
      fail(`${pathName}.${key}`, "a string when present");
    }
  }
  function optionalBoolean(obj, key, pathName) {
    if (obj[key] !== undefined && typeof obj[key] !== "boolean") {
      fail(`${pathName}.${key}`, "a boolean when present");
    }
  }
  function optionalNumber(obj, key, pathName) {
    if (obj[key] !== undefined || Object.prototype.hasOwnProperty.call(obj, key)) {
      if (typeof obj[key] !== "number" || !Number.isFinite(obj[key])) {
        fail(`${pathName}.${key}`, "a finite number when present");
      }
    }
  }
  function requireNumber(obj, key, pathName) {
    if (typeof obj[key] !== "number" || !Number.isFinite(obj[key])) {
      fail(`${pathName}.${key}`, "a finite number");
    }
  }
  function objectAt(obj, key, pathName) {
    const child = obj[key];
    if (child === undefined) return undefined;
    if (child === null || typeof child !== "object" || Array.isArray(child)) {
      fail(`${pathName}.${key}`, "a JSON object when present");
    }
    return child;
  }
  function stringRecord(obj, key, pathName) {
    const record = objectAt(obj, key, pathName);
    if (!record) return;
    for (const [childKey, childValue] of Object.entries(record)) {
      if (typeof childValue !== "string") {
        fail(`${pathName}.${key}.${childKey}`, "a string");
      }
    }
  }
  function requireStringArray(obj, key, pathName) {
    const valueAtKey = obj[key];
    if (!Array.isArray(valueAtKey) || valueAtKey.some((item) => typeof item !== "string")) {
      fail(`${pathName}.${key}`, "an array of strings");
    }
  }

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    die(`payload at ${filePath} is not a JSON object`);
  }
  const obj = value;
  requireNumber(obj, "generation", "root");
  if (typeof obj.agent_id !== "string" || !obj.agent_id) {
    die(`payload at ${filePath} is missing required string field "agent_id"`);
  }
  optionalString(obj, "experiment_name", "root");
  optionalString(obj, "execution_prompt", "root");
  optionalBoolean(obj, "arc_no_mcp", "root");
  if (
    obj.task === null ||
    typeof obj.task !== "object" ||
    Array.isArray(obj.task)
  ) {
    die(`payload at ${filePath} is missing required object field "task"`);
  }
  const task = obj.task;
  if (typeof task.id !== "string" || !task.id) {
    die(`payload at ${filePath} task is missing required string field "id"`);
  }
  optionalString(task, "repo", "task");
  optionalString(task, "prompt", "task");
  objectAt(task, "metadata", "task");

  const workspaceSeed = objectAt(obj, "workspace_seed", "root");
  if (workspaceSeed) {
    for (const key of ["instruction_md", "memory_md", "task_md", "tools_md", "repo_source_path"]) {
      optionalString(workspaceSeed, key, "workspace_seed");
    }
    stringRecord(workspaceSeed, "task_files", "workspace_seed");
  }

  const runtime = objectAt(obj, "runtime", "root");
  if (runtime) {
    const sessionScope = runtime.session_scope;
    if (sessionScope !== undefined && sessionScope !== "task" && sessionScope !== "agent") {
      fail("runtime.session_scope", '"task" or "agent" when present');
    }
    optionalBoolean(runtime, "wipe_workspace_per_task", "runtime");
    for (const key of [
      "container_image",
      "official_container_image",
      "runner_image",
      "repo_container_path",
      "official_repo_container_path",
      "runner_root",
    ]) {
      optionalString(runtime, key, "runtime");
    }
  }

  const knowledge = objectAt(obj, "knowledge", "root");
  if (knowledge) {
    requireString(knowledge, "db_path", "knowledge");
    requireString(knowledge, "mcp_server_dir", "knowledge");
    optionalString(knowledge, "snapshot_path", "knowledge");
    optionalBoolean(knowledge, "disable_memory_tools", "knowledge");
    optionalNumber(knowledge, "forum_generation", "knowledge");
    optionalString(knowledge, "experiment_name", "knowledge");
  }

  const runtimeAudit = objectAt(obj, "runtime_audit", "root");
  if (runtimeAudit) {
    requireString(runtimeAudit, "db_path", "runtime_audit");
  }

  const arcTools = objectAt(obj, "arc_tools", "root");
  if (arcTools) {
    if (typeof arcTools.enable !== "boolean") {
      fail("arc_tools.enable", "a boolean");
    }
    requireString(arcTools, "mcp_server_dir", "arc_tools");
    optionalString(arcTools, "task_source", "arc_tools");
    optionalString(arcTools, "task_id", "arc_tools");
    optionalString(arcTools, "snapshot_path", "arc_tools");
  }

  const phase1Reflection = objectAt(obj, "phase1_reflection", "root");
  if (phase1Reflection) {
    if (typeof phase1Reflection.enabled !== "boolean") {
      fail("phase1_reflection.enabled", "a boolean");
    }
    optionalNumber(phase1Reflection, "eval_result_poll_timeout_ms", "phase1_reflection");
  }

  const crossTaskSharedContainer = objectAt(obj, "cross_task_shared_container", "root");
  if (crossTaskSharedContainer) {
    if (typeof crossTaskSharedContainer.enabled !== "boolean") {
      fail("cross_task_shared_container.enabled", "a boolean");
    }
    optionalString(crossTaskSharedContainer, "barrier_name", "cross_task_shared_container");
    optionalNumber(crossTaskSharedContainer, "response_poll_timeout_ms", "cross_task_shared_container");
  }

  const polyglotFeedback = objectAt(obj, "polyglot_test_feedback", "root");
  if (polyglotFeedback) {
    if (typeof polyglotFeedback.enabled !== "boolean") {
      fail("polyglot_test_feedback.enabled", "a boolean");
    }
    requireString(polyglotFeedback, "agentId", "polyglot_test_feedback");
    requireNumber(polyglotFeedback, "triesRemaining", "polyglot_test_feedback");
    requireNumber(polyglotFeedback, "maxLines", "polyglot_test_feedback");
    requireString(polyglotFeedback, "fileList", "polyglot_test_feedback");
    requireStringArray(polyglotFeedback, "allowedTools", "polyglot_test_feedback");
    if (!objectAt(polyglotFeedback, "mcpServers", "polyglot_test_feedback")) {
      fail("polyglot_test_feedback.mcpServers", "a JSON object");
    }
    requireNumber(polyglotFeedback, "maxTurnsPerRound", "polyglot_test_feedback");
    optionalNumber(polyglotFeedback, "evalResultPollTimeoutMs", "polyglot_test_feedback");
  }
}

function coerceOptionalNumber(value) {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

function coerceOptionalString(value) {
  if (value === undefined || value === null) {
    return undefined;
  }
  const s = String(value);
  return s.length > 0 ? s : undefined;
}

function isContainerOutput(value) {
  if (value === null || typeof value !== "object") {
    return false;
  }
  const status = value.status;
  if (typeof status !== "string" || !CONTAINER_OUTPUT_STATUSES.has(status)) {
    return false;
  }
  const result = value.result;
  return result === null || typeof result === "string";
}

// ── Drift guards: each inline copy must mirror its TS source ──────────────────

describe("json-boundary guards: copies mirror the TS source", () => {
  for (const [name, tsSource, label] of [
    ["assertKsiPayload", mainTs, "main.ts"],
    ["coerceOptionalNumber", mainTs, "main.ts"],
    ["coerceOptionalString", mainTs, "main.ts"],
    ["isContainerOutput", containerOutputTs, "container_output.ts"],
  ]) {
    it(`${name} mirrors ${label}`, () => {
      const r = compareFunctions(tsSource, selfSource, name);
      assert.ok(r.ok, r.ok ? "" : r.message);
    });
  }

  it("the drift guard is not vacuous (detects an injected change)", () => {
    const mutated = selfSource.replace(
      "return Number.isFinite(n) ? n : undefined;",
      "return Number.isFinite(n) ? n + 1 : undefined;",
    );
    assert.notEqual(mutated, selfSource, "mutation must apply");
    const r = compareFunctions(mainTs, mutated, "coerceOptionalNumber");
    assert.equal(r.ok, false, "guard must flag the mutated copy");
  });
});

// ── Behavior: assertKsiPayload ──────────────────────────────────────────────

describe("assertKsiPayload", () => {
  it("accepts a well-formed payload", () => {
    assert.doesNotThrow(() =>
      assertKsiPayload(
        {
          generation: 1,
          agent_id: "a1",
          experiment_name: "exp",
          execution_prompt: "Solve.",
          task: { id: "t1", repo: "org/repo", prompt: "Task", metadata: { task_source: "polyglot" } },
          workspace_seed: {
            instruction_md: "Instruction",
            memory_md: "",
            task_md: "Task",
            tools_md: "Tools",
            task_files: { "payload.json": "{}" },
            repo_source_path: "/tmp/repo",
          },
          runtime: {
            session_scope: "agent",
            wipe_workspace_per_task: false,
            container_image: "img",
            official_container_image: "official",
            runner_image: "runner",
            repo_container_path: "/workspace/repo",
            official_repo_container_path: "/official/repo",
            runner_root: "/app",
          },
          knowledge: {
            db_path: "/tmp/knowledge.sqlite",
            mcp_server_dir: "/app/memory",
            snapshot_path: "/tmp/snapshot.json",
            disable_memory_tools: false,
            forum_generation: 1,
            experiment_name: "exp",
          },
          runtime_audit: { db_path: "/tmp/runtime.sqlite" },
          arc_tools: {
            enable: false,
            mcp_server_dir: "/app/memory",
            task_source: "arc",
            task_id: "t1",
            snapshot_path: "/tmp/arc.json",
          },
          arc_no_mcp: true,
          phase1_reflection: { enabled: true, eval_result_poll_timeout_ms: 5000 },
          cross_task_shared_container: {
            enabled: true,
            barrier_name: "r1",
            response_poll_timeout_ms: 5000,
          },
          polyglot_test_feedback: {
            enabled: true,
            agentId: "a1",
            triesRemaining: 1,
            maxLines: 20,
            fileList: "solution.py",
            allowedTools: ["Read"],
            mcpServers: {},
            maxTurnsPerRound: 2,
            evalResultPollTimeoutMs: 5000,
          },
        },
        "p.json",
      ),
    );
  });

  for (const [label, payload, needle] of [
    ["null", null, "not a JSON object"],
    ["a non-object", 42, "not a JSON object"],
    ["an array", [], "not a JSON object"],
    ["missing generation", { agent_id: "a1", task: { id: "t1" } }, "root.generation"],
    ["missing agent_id", { generation: 1, task: { id: "t1" } }, '"agent_id"'],
    ["empty agent_id", { generation: 1, agent_id: "", task: { id: "t1" } }, '"agent_id"'],
    ["missing task", { generation: 1, agent_id: "a1" }, '"task"'],
    ["non-object task", { generation: 1, agent_id: "a1", task: 5 }, '"task"'],
    ["missing task.id", { generation: 1, agent_id: "a1", task: {} }, '"id"'],
    ["empty task.id", { generation: 1, agent_id: "a1", task: { id: "" } }, '"id"'],
    [
      "task metadata array",
      { generation: 1, agent_id: "a1", task: { id: "t1", metadata: [] } },
      "task.metadata",
    ],
    [
      "task metadata null",
      { generation: 1, agent_id: "a1", task: { id: "t1", metadata: null } },
      "task.metadata",
    ],
    ["runtime null", { generation: 1, agent_id: "a1", task: { id: "t1" }, runtime: null }, "root.runtime"],
    ["knowledge null", { generation: 1, agent_id: "a1", task: { id: "t1" }, knowledge: null }, "root.knowledge"],
    [
      "workspace task_files non-string",
      { generation: 1, agent_id: "a1", task: { id: "t1" }, workspace_seed: { task_files: { a: 1 } } },
      "workspace_seed.task_files.a",
    ],
    [
      "bad runtime session_scope",
      { generation: 1, agent_id: "a1", task: { id: "t1" }, runtime: { session_scope: "global" } },
      "runtime.session_scope",
    ],
    [
      "knowledge missing db_path",
      { generation: 1, agent_id: "a1", task: { id: "t1" }, knowledge: { mcp_server_dir: "/app/memory" } },
      "knowledge.db_path",
    ],
    [
      "arc_tools bad enable",
      {
        generation: 1,
        agent_id: "a1",
        task: { id: "t1" },
        arc_tools: { enable: "yes", mcp_server_dir: "/app/memory" },
      },
      "arc_tools.enable",
    ],
    [
      "phase1 bad enabled",
      { generation: 1, agent_id: "a1", task: { id: "t1" }, phase1_reflection: { enabled: "yes" } },
      "phase1_reflection.enabled",
    ],
    [
      "cross-task bad timeout",
      {
        generation: 1,
        agent_id: "a1",
        task: { id: "t1" },
        cross_task_shared_container: { enabled: true, response_poll_timeout_ms: "slow" },
      },
      "cross_task_shared_container.response_poll_timeout_ms",
    ],
    [
      "polyglot feedback bad tools",
      {
        generation: 1,
        agent_id: "a1",
        task: { id: "t1" },
        polyglot_test_feedback: {
          enabled: true,
          agentId: "a1",
          triesRemaining: 1,
          maxLines: 20,
          fileList: "solution.py",
          allowedTools: ["Read", 7],
          mcpServers: {},
          maxTurnsPerRound: 2,
        },
      },
      "polyglot_test_feedback.allowedTools",
    ],
  ]) {
    it(`rejects ${label}`, () => {
      assert.throws(
        () => assertKsiPayload(payload, "p.json"),
        (err) => err instanceof Error && err.message.includes(needle),
      );
    });
  }
});

// ── Behavior: coerceOptionalNumber ───────────────────────────────────────────

describe("coerceOptionalNumber", () => {
  it("returns undefined for nullish / empty", () => {
    assert.equal(coerceOptionalNumber(undefined), undefined);
    assert.equal(coerceOptionalNumber(null), undefined);
    assert.equal(coerceOptionalNumber(""), undefined);
  });
  it("coerces finite numbers and numeric strings", () => {
    assert.equal(coerceOptionalNumber(42), 42);
    assert.equal(coerceOptionalNumber("42"), 42);
    assert.equal(coerceOptionalNumber("3.14"), 3.14);
    assert.equal(coerceOptionalNumber(0), 0);
  });
  it("returns undefined for non-finite / non-numeric", () => {
    assert.equal(coerceOptionalNumber("abc"), undefined);
    assert.equal(coerceOptionalNumber(NaN), undefined);
    assert.equal(coerceOptionalNumber(Infinity), undefined);
  });
});

// ── Behavior: coerceOptionalString ───────────────────────────────────────────

describe("coerceOptionalString", () => {
  it("returns undefined for nullish and empty result", () => {
    assert.equal(coerceOptionalString(undefined), undefined);
    assert.equal(coerceOptionalString(null), undefined);
    assert.equal(coerceOptionalString(""), undefined);
  });
  it("stringifies non-empty values", () => {
    assert.equal(coerceOptionalString("x"), "x");
    assert.equal(coerceOptionalString(0), "0");
    assert.equal(coerceOptionalString(false), "false");
  });
});

// ── Behavior: isContainerOutput ──────────────────────────────────────────────

describe("isContainerOutput", () => {
  for (const status of ["success", "error", "recovered_from_session"]) {
    it(`accepts a real envelope with status="${status}" and result`, () => {
      assert.equal(isContainerOutput({ status, result: "x" }), true);
      assert.equal(isContainerOutput({ status, result: null }), true);
    });
  }
  for (const [label, value] of [
    ["null", null],
    ["a bare number", 42],
    ["an object with no status", {}],
    ["an unknown status", { status: "bogus", result: null }],
    ["a non-string status", { status: 42, result: null }],
    ["an array", []],
    ["a known status but missing result", { status: "success" }],
    ["a result of the wrong type", { status: "success", result: 42 }],
  ]) {
    it(`rejects ${label}`, () => {
      assert.equal(isContainerOutput(value), false);
    });
  }
});
