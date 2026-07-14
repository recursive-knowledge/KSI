/**
 * Token-shape mapping regression test for the OpenAI agent-runner.
 *
 * Context
 * -------
 * `runtime_runner/agent-runner/src/openai_usage.ts::usageFromResult` historically
 * mapped OpenAI's per-response usage into Claude-shaped buckets wrongly:
 *
 *   input_tokens         ← usage.inputTokens           (TOTAL input, INCLUDES cached)
 *   output_tokens        ← usage.outputTokens          (TOTAL output, includes reasoning)
 *   cache_read_input     ← inputDetails.cachedTokens   (correct)
 *   cache_creation_input ← outputDetails.reasoningTokens (WRONG — reasoning is output)
 *
 * Downstream, `kcsi/tokens.py::TokenUsage.total` sums all four fields:
 *   total = input + output + cache_read + cache_creation
 *
 * With the old mapping:
 *   - reasoning tokens double-counted (in output AND cache_creation)
 *   - cached tokens double-counted (in input AND cache_read)
 *
 * Correct Claude-aligned mapping for OpenAI:
 *   input_tokens          = inputTotal - cachedTokens   (fresh input only)
 *   output_tokens         = outputTotal                 (includes reasoning, that's fine)
 *   cache_read_input      = cachedTokens
 *   cache_creation_input  = 0                           (OpenAI has no separate create charge)
 *
 * This yields a faithful `TokenUsage.total` equal to
 * `(inputTotal - cached) + outputTotal + cached + 0 == inputTotal + outputTotal`,
 * with cache visibility preserved.
 */

import { strict as assert } from "node:assert";
import { spawnSync } from "node:child_process";
import { describe, it } from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const tsxBin = path.join(repoRoot, "runtime_runner", "node_modules", ".bin", "tsx");
const usageTs = path.join(
  repoRoot,
  "runtime_runner",
  "agent-runner",
  "src",
  "openai_usage.ts",
);
const tsxSkip = !fs.existsSync(tsxBin);

function usageFromResult(result) {
  const source = `
    import { usageFromResult } from ${JSON.stringify(usageTs)};
    process.stdout.write(JSON.stringify(usageFromResult(${JSON.stringify(result)})));
  `;
  const proc = spawnSync(tsxBin, ["--input-type=module", "--eval", source], {
    cwd: repoRoot,
    encoding: "utf8",
  });
  assert.equal(proc.status, 0, `tsx failed: status=${proc.status}\nstdout=${proc.stdout}\nstderr=${proc.stderr}`);
  return JSON.parse(proc.stdout);
}

describe("usageFromResult — OpenAI token-shape mapping", () => {
  if (tsxSkip) {
    it("tsx not installed; run npm install in runtime_runner/", { skip: true }, () => {});
    return;
  }

  it("maps a single response without cache or reasoning correctly", () => {
    const result = {
      rawResponses: [
        {
          usage: {
            inputTokens: 100,
            outputTokens: 50,
          },
        },
      ],
    };
    const usage = usageFromResult(result);
    assert.equal(usage.input_tokens, 100);
    assert.equal(usage.output_tokens, 50);
    assert.equal(usage.cache_read_input_tokens, 0);
    assert.equal(usage.cache_creation_input_tokens, 0);
  });

  it("separates cached input from fresh input (no double-count)", () => {
    // OpenAI reports inputTokens as the TOTAL (cached + fresh). The cached
    // portion appears in inputTokensDetails[*].cachedTokens.
    const result = {
      rawResponses: [
        {
          usage: {
            inputTokens: 1000,
            outputTokens: 200,
            inputTokensDetails: [{ cachedTokens: 800 }],
          },
        },
      ],
    };
    const usage = usageFromResult(result);
    assert.equal(usage.input_tokens, 200, "fresh input = total - cached");
    assert.equal(usage.cache_read_input_tokens, 800, "cache_read = cached");
    assert.equal(usage.output_tokens, 200);
    assert.equal(usage.cache_creation_input_tokens, 0);

    // Downstream total must equal inputTotal + outputTotal exactly
    // (no double-count via cache_read or cache_creation).
    const total =
      usage.input_tokens +
      usage.output_tokens +
      usage.cache_read_input_tokens +
      usage.cache_creation_input_tokens;
    assert.equal(total, 1200, "input_total (1000) + output_total (200)");
  });

  it("does NOT push reasoning tokens into cache_creation_input_tokens", () => {
    // Reasoning tokens are ALREADY counted in outputTokens. Mapping them
    // into cache_creation_input_tokens would double-count them in
    // TokenUsage.total (which sums all four fields).
    const result = {
      rawResponses: [
        {
          usage: {
            inputTokens: 100,
            outputTokens: 500, // includes 400 reasoning
            outputTokensDetails: [{ reasoningTokens: 400 }],
          },
        },
      ],
    };
    const usage = usageFromResult(result);
    assert.equal(usage.output_tokens, 500, "reasoning stays inside output_tokens");
    assert.equal(
      usage.cache_creation_input_tokens,
      0,
      "reasoning MUST NOT leak into cache_creation_input_tokens",
    );
  });

  it("sums across multiple rawResponses", () => {
    const result = {
      rawResponses: [
        {
          usage: {
            inputTokens: 100,
            outputTokens: 20,
            inputTokensDetails: [{ cachedTokens: 60 }],
          },
        },
        {
          usage: {
            inputTokens: 200,
            outputTokens: 50,
            inputTokensDetails: [{ cachedTokens: 150 }],
          },
        },
      ],
    };
    const usage = usageFromResult(result);
    assert.equal(usage.input_tokens, (100 - 60) + (200 - 150));
    assert.equal(usage.output_tokens, 20 + 50);
    assert.equal(usage.cache_read_input_tokens, 60 + 150);
    assert.equal(usage.cache_creation_input_tokens, 0);
  });

  it("handles snake_case field aliases", () => {
    const result = {
      rawResponses: [
        {
          usage: {
            input_tokens: 1000,
            output_tokens: 200,
            input_tokens_details: [{ cached_tokens: 800 }],
          },
        },
      ],
    };
    const usage = usageFromResult(result);
    assert.equal(usage.input_tokens, 200);
    assert.equal(usage.cache_read_input_tokens, 800);
  });

  it("returns zeros when no rawResponses or usage present", () => {
    assert.deepEqual(usageFromResult({}), {
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    });
    assert.deepEqual(usageFromResult({ rawResponses: [] }), {
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    });
    assert.deepEqual(usageFromResult({ rawResponses: [{}] }), {
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    });
  });

  it("clamps input_tokens to zero if cached > reported input total", () => {
    // Defensive: some SDK versions briefly reported cached > input due to
    // an off-by-one. Don't produce negative input_tokens in that case.
    const result = {
      rawResponses: [
        {
          usage: {
            inputTokens: 100,
            outputTokens: 20,
            inputTokensDetails: [{ cachedTokens: 150 }],
          },
        },
      ],
    };
    const usage = usageFromResult(result);
    assert.equal(usage.input_tokens, 0);
    assert.equal(usage.cache_read_input_tokens, 150);
  });
});
