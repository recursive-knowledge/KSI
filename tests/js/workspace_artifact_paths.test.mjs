import { describe, it } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";

function resolveRepoRelativePath(repoDir, relPath) {
  if (!relPath || path.isAbsolute(relPath)) return null;
  const repoRoot = path.resolve(repoDir);
  const fullPath = path.resolve(repoRoot, relPath);
  if (!fullPath.startsWith(repoRoot + path.sep)) return null;
  return fullPath;
}

describe("resolveRepoRelativePath", () => {
  it("accepts normal relative paths inside the repo", () => {
    const repoDir = path.resolve("/tmp/workspace/repo");

    assert.equal(resolveRepoRelativePath(repoDir, "src/solution.py"), path.join(repoDir, "src/solution.py"));
  });

  it("rejects absolute and parent-traversal paths", () => {
    const repoDir = path.resolve("/tmp/workspace/repo");

    assert.equal(resolveRepoRelativePath(repoDir, "/etc/passwd"), null);
    assert.equal(resolveRepoRelativePath(repoDir, "../secret.txt"), null);
    assert.equal(resolveRepoRelativePath(repoDir, "src/../../secret.txt"), null);
  });
});
