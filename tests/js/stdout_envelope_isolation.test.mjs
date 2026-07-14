/**
 * Regression / hypothesis-locking tests for the stdout envelope isolation
 * fix (Option A: route all pino output to stderr).
 *
 * Background
 * ----------
 * Forensics on ``arc2_post_fix_v3_20260420_memory.sqlite`` showed 36/60
 * attempts (60%) had a pino log line injected MID-STRING inside the JSON
 * envelope emitted by ``runtime_runner/src/main.ts``. The host's Python
 * parser (``src/kcsi/runtime/normalize.py::parse_runner_stdout``) could not
 * read the envelope and fell back to the raw-text path, hiding real work.
 *
 * Root cause: pino's default destination is stdout (fd 1). Node's
 * ``process.stdout.write`` is not atomic across concurrent async writers
 * on a pipe once the write exceeds PIPE_BUF (4096 bytes). Pino log lines
 * interleaved byte-for-byte with the single envelope emission.
 *
 * Fix: ``runtime_runner/src/logger.ts`` constructs pino with an
 * explicit destination of fd 2 (stderr). The only writer on stdout is
 * the single ``process.stdout.write(JSON.stringify(output) + '\n')`` call
 * at ``main.ts:361``.
 *
 * These tests pin both halves of the fix:
 *   1. Static scan: no stray ``console.log(`` writes outside the one
 *      envelope emission point.
 *   2. Dynamic spawn: loading ``logger.ts`` and logging at info level
 *      produces output on stderr with ZERO bytes on stdout.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, "..", "..");
const SRC_DIR = join(REPO_ROOT, "runtime_runner", "src");

// ── Static scan tests ────────────────────────────────────────────────────

function listTsFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      out.push(...listTsFiles(join(dir, entry.name)));
    } else if (entry.isFile() && entry.name.endsWith(".ts")) {
      out.push(join(dir, entry.name));
    }
  }
  return out;
}

describe("stdout writer isolation (static)", () => {
  it("no console.log(...) in runtime_runner/src", () => {
    const files = listTsFiles(SRC_DIR);
    assert.ok(files.length > 0, `no .ts files found under ${SRC_DIR}`);
    const offenders = [];
    for (const f of files) {
      const lines = readFileSync(f, "utf8").split("\n");
      lines.forEach((line, idx) => {
        // Strip inline comment text so a mention inside a comment doesn't
        // trip the check; we still flag real ``console.log(`` call sites.
        const code = line.replace(/\/\/.*$/, "");
        if (/\bconsole\.log\s*\(/.test(code)) {
          offenders.push(`${f}:${idx + 1}: ${line.trim()}`);
        }
      });
    }
    assert.deepEqual(
      offenders,
      [],
      `console.log() is forbidden in runtime_runner/src (interleaves with envelope):\n` +
        offenders.join("\n"),
    );
  });

  it("process.stdout.write is only used for the envelope emission", () => {
    const files = listTsFiles(SRC_DIR);
    const hits = [];
    for (const f of files) {
      const lines = readFileSync(f, "utf8").split("\n");
      lines.forEach((line, idx) => {
        const code = line.replace(/\/\/.*$/, "");
        if (/\bprocess\.stdout\.write\s*\(/.test(code)) {
          hits.push({ file: f, line: idx + 1, text: line.trim() });
        }
      });
    }
    // Expect exactly one hit and it MUST be the envelope emission. The
    // whitelist check is structural — ``JSON.stringify(output)`` is the
    // tell that this is the main.ts final write, not some new log call.
    assert.equal(
      hits.length,
      1,
      `expected exactly 1 process.stdout.write() in runtime_runner/src; found ${hits.length}:\n` +
        hits.map((h) => `  ${h.file}:${h.line}: ${h.text}`).join("\n"),
    );
    assert.ok(
      hits[0].file.endsWith("main.ts"),
      `the one allowed process.stdout.write must be in main.ts, got ${hits[0].file}`,
    );
    assert.ok(
      /JSON\.stringify\(output\)/.test(hits[0].text),
      `the main.ts write must be the envelope (JSON.stringify(output)), got: ${hits[0].text}`,
    );
  });
});

// ── Dynamic spawn tests ──────────────────────────────────────────────────

describe("pino logger destination (dynamic)", () => {
  const tsxBin = join(REPO_ROOT, "runtime_runner", "node_modules", ".bin", "tsx");
  const tsxAvailable = existsSync(tsxBin);

  const dynamicTest = () => {
    // Spawn a node subprocess that imports the real logger.ts via tsx and
    // emits an info-level log line. Verify all bytes land on stderr and
    // stdout receives zero bytes.
    const loggerPath = join(SRC_DIR, "logger.ts");
    assert.ok(existsSync(loggerPath), `logger.ts missing at ${loggerPath}`);
    const script = `
import { logger } from ${JSON.stringify(loggerPath)};
logger.info({ check: 'hello' }, 'dynamic test message');
// Give pino's SonicBoom a tick to flush before the process exits.
setTimeout(() => {}, 50);
`;
    const res = spawnSync(
      tsxBin,
      ["--eval", script, "--conditions=node"],
      {
        cwd: join(REPO_ROOT, "runtime_runner"),
        encoding: "utf8",
        env: {
          ...process.env,
          LOG_LEVEL: "info",
          // Disable pretty print for deterministic assertion on raw JSON.
          KCSI_PRETTY_LOGS: "0",
        },
      },
    );
    assert.equal(
      res.status,
      0,
      `tsx exit nonzero: status=${res.status}\nstderr=${res.stderr}\nstdout=${res.stdout}`,
    );
    assert.equal(
      res.stdout.length,
      0,
      `stdout must be empty (envelope channel reserved); got ${res.stdout.length} bytes: ${JSON.stringify(res.stdout.slice(0, 200))}`,
    );
    assert.ok(
      res.stderr.includes("dynamic test message"),
      `stderr must contain the log line; got: ${JSON.stringify(res.stderr.slice(0, 400))}`,
    );
    // Verify the pino JSON shape — proves the real pino logger ran, not a
    // fallback like console.error.
    assert.ok(
      /"level":30/.test(res.stderr),
      `stderr must contain pino-shaped JSON (level=30 for info); got: ${JSON.stringify(res.stderr.slice(0, 400))}`,
    );
  };

  if (!tsxAvailable) {
    // Register as SKIPPED (not a silent pass) when node_modules isn't
    // installed; run `npm install` in runtime_runner/ to enable it.
    it.skip("logger.ts routes output to stderr, not stdout (tsx not installed)");
  } else {
    it("logger.ts routes output to stderr, not stdout", dynamicTest);
  }

  it("logger.ts owns the only pino() construction in runtime_runner/src", () => {
    // The 2026-era mount_security.ts had its OWN pino() call that also had
    // to be pinned to stderr. That module is gone; keep stdout isolation
    // auditable by asserting no other module constructs a pino instance.
    const offenders = [];
    for (const f of listTsFiles(SRC_DIR)) {
      if (f.endsWith("logger.ts")) continue;
      if (/\bpino\(/.test(readFileSync(f, "utf8"))) {
        offenders.push(f);
      }
    }
    assert.deepEqual(
      offenders,
      [],
      `unexpected pino() constructions outside logger.ts: ${offenders.join(", ")}`,
    );
  });
});
