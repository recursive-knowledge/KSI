import { describe, it } from "node:test";
import assert from "node:assert/strict";

function prefixRawCreateContent(text) {
  if (!text) return "";
  const hasTrailingNewline = text.endsWith("\n");
  const body = hasTrailingNewline ? text.slice(0, -1) : text;
  const lines = body ? body.split("\n") : [""];
  const prefixed = lines.map((line) => `+${line}`).join("\n");
  return hasTrailingNewline ? `${prefixed}\n` : prefixed;
}

function isV4ACreateBody(text) {
  if (!text || text.includes("*** Begin Patch")) return false;
  const body = text.endsWith("\n") ? text.slice(0, -1) : text;
  if (!body) return false;
  const lines = body.split("\n");
  if (lines.length === 1 && lines[0].startsWith("+") && !lines[0].startsWith("++")) {
    return false;
  }
  return lines.every((line) => line.startsWith("+"));
}

function extractUnifiedCreateBody(lines) {
  const oldPathIndex = lines.findIndex((line) => line === "--- /dev/null");
  if (oldPathIndex < 0 || !lines[oldPathIndex + 1]?.startsWith("+++ ")) {
    return null;
  }
  const body = [];
  let inHunk = false;
  for (const line of lines.slice(oldPathIndex + 2)) {
    if (line.startsWith("diff --git ")) break;
    if (line.startsWith("@@ ")) {
      inHunk = true;
      continue;
    }
    if (!inHunk || line === "\\ No newline at end of file") continue;
    if (line.startsWith("+")) {
      body.push(line);
    }
  }
  return inHunk ? body.join("\n") : null;
}

function toV4ACreate(diff) {
  const text = String(diff ?? "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!text) return "";
  if (isV4ACreateBody(text)) return text;

  const lines = text.split("\n");
  const addFileIndex = lines.findIndex((line) => line.startsWith("*** Add File:"));
  if (addFileIndex >= 0 && lines.some((line) => line.startsWith("*** Begin Patch"))) {
    const body = [];
    for (const line of lines.slice(addFileIndex + 1)) {
      if (
        line.startsWith("*** End Patch") ||
        line.startsWith("*** Update File:") ||
        line.startsWith("*** Delete File:")
      ) {
        break;
      }
      body.push(line.startsWith("+") ? line : `+${line}`);
    }
    return body.join("\n");
  }

  const unifiedCreateBody = extractUnifiedCreateBody(lines);
  if (unifiedCreateBody != null) {
    return unifiedCreateBody;
  }

  return prefixRawCreateContent(text);
}

describe("toV4ACreate", () => {
  it("prefixes raw create content with V4A plus lines", () => {
    assert.equal(toV4ACreate("hello\nworld\n"), "+hello\n+world\n");
  });

  it("leaves already-V4A create bodies unchanged", () => {
    assert.equal(toV4ACreate("+hello\n+world\n"), "+hello\n+world\n");
  });

  it("preserves single-line raw content with a literal leading plus", () => {
    assert.equal(toV4ACreate("+literal\n"), "++literal\n");
  });

  it("extracts plus lines from a full Add File envelope", () => {
    const fullPatch = [
      "*** Begin Patch",
      "*** Add File: src/new.ts",
      "+export const x = 1;",
      "+",
      "+export const y = 2;",
      "*** End Patch",
      "",
    ].join("\n");

    assert.equal(toV4ACreate(fullPatch), "+export const x = 1;\n+\n+export const y = 2;");
  });

  it("extracts plus lines from a standard unified create diff", () => {
    const unifiedCreate = [
      "diff --git a/src/new.ts b/src/new.ts",
      "new file mode 100644",
      "--- /dev/null",
      "+++ b/src/new.ts",
      "@@ -0,0 +1,3 @@",
      "+export const x = 1;",
      "+++literalPlus",
      "+export const y = 2;",
      "",
    ].join("\n");

    assert.equal(toV4ACreate(unifiedCreate), "+export const x = 1;\n+++literalPlus\n+export const y = 2;");
  });
});
