import fs from 'fs';
import path from 'path';

import { resolveWorkspaceRootPath } from './workspace_scope.js';

/**
 * --arc-no-mcp synthesis: read per-test ASCII grids from the workspace and
 * append synthetic `arc_set_output_grid` + `arc_submit_trial` (and, for
 * multi-test tasks, `arc_next_test_input`) tool_call entries to rawTrace. The
 * downstream collection loop in main() then picks up the synthesized
 * arc_submit_trial outputs into runtime_meta.arc_submit_trial_results, which
 * the Python scorer
 * (`src/ksi/eval/arc_session.py::_reconstruct_submissions_from_trace`)
 * consumes unchanged.
 *
 * Prediction files are per-test, named `attempt_<k>_<t>.txt` where ``k`` is the
 * 0-based test index and ``t`` is the 1-based trial number (1 or 2). For
 * backward compatibility, legacy `attempt_1.txt` / `attempt_2.txt` (no test
 * index) are treated as test 0, trials 1 and 2. A single-test task therefore
 * produces a trace byte-identical to the pre-multi-test behavior (zero
 * `arc_next_test_input` events).
 *
 * Each attempt file's content is rows of space-separated digits 0-9, one row
 * per line. Stub files (`__NOT_SUBMITTED__\n` sentinel) are pre-populated by
 * the host Python so the runtime always finds a file; `parseAsciiGrid` rejects
 * the sentinel (`Number("__NOT_SUBMITTED__")` → NaN), which keeps the
 * synthesizer from crediting an un-submitted attempt as a real grid. Submission
 * rate is therefore the rate at which the agent actually overwrote the stub,
 * not 100% by construction.
 */
export function parseAsciiGrid(text: string): number[][] | null {
  const rows: number[][] = [];
  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    const cells = line.split(/\s+/);
    const row: number[] = [];
    for (const c of cells) {
      const v = Number(c);
      if (!Number.isInteger(v) || v < 0 || v > 9) return null;
      row.push(v);
    }
    rows.push(row);
  }
  if (rows.length === 0) return null;
  const width = rows[0].length;
  if (width === 0 || width > 30 || rows.length > 30) return null;
  for (const r of rows) if (r.length !== width) return null;
  return rows;
}

function readGridFile(wsRoot: string, fname: string): number[][] | null {
  const fpath = path.join(wsRoot, fname);
  if (!fs.existsSync(fpath)) return null;
  let text: string;
  try {
    text = fs.readFileSync(fpath, 'utf-8');
  } catch {
    return null;
  }
  return parseAsciiGrid(text);
}

/**
 * Discover the highest per-test index ``k`` for which any `attempt_<k>_*.txt`
 * file exists in ``wsRoot``. Returns -1 when no per-test files are present
 * (i.e. only legacy `attempt_1.txt`/`attempt_2.txt`, or nothing).
 */
function maxPerTestIndex(wsRoot: string): number {
  let maxIdx = -1;
  let entries: string[];
  try {
    entries = fs.readdirSync(wsRoot);
  } catch {
    return -1;
  }
  const re = /^attempt_(\d+)_(\d+)\.txt$/;
  for (const name of entries) {
    const m = re.exec(name);
    if (!m) continue;
    const k = Number(m[1]);
    if (Number.isInteger(k) && k >= 0 && k > maxIdx) maxIdx = k;
  }
  return maxIdx;
}

/**
 * Core synthesizer operating on a resolved workspace directory. Exposed for
 * unit testing (the js guard test writes per-test files into a temp dir).
 */
export function synthesizeArcSubmitTraceFromWorkspaceDir(
  wsRoot: string,
  rawTrace: Array<Record<string, unknown>>,
): void {
  const maxIdx = maxPerTestIndex(wsRoot);

  // Build a per-test list of trial grids. Tests with no parseable grid get an
  // empty list (no set/submit), but the cursor still advances via next_test so
  // later submissions are scored against the right test index.
  const perTest: number[][][][] = [];

  if (maxIdx < 0) {
    // Legacy / single-test: attempt_1.txt + attempt_2.txt are test 0 trials.
    const grids: number[][][] = [];
    for (const fname of ['attempt_1.txt', 'attempt_2.txt']) {
      const grid = readGridFile(wsRoot, fname);
      if (grid) grids.push(grid);
    }
    perTest.push(grids);
  } else {
    for (let k = 0; k <= maxIdx; k++) {
      const grids: number[][][] = [];
      for (let t = 1; t <= 2; t++) {
        const grid = readGridFile(wsRoot, `attempt_${k}_${t}.txt`);
        if (grid) grids.push(grid);
      }
      perTest.push(grids);
    }
  }

  const testCount = perTest.length;
  // Nothing parseable anywhere → leave rawTrace untouched; the Python scorer
  // then finds no submission and scores 0 (canonical ARC; no text fallback as
  // of #944). Preserves the pre-multi-test no-op behavior.
  if (perTest.every((grids) => grids.length === 0)) {
    return;
  }

  for (let k = 0; k < testCount; k++) {
    if (k > 0) {
      // Advance the scorer's test cursor before emitting this test's trials.
      rawTrace.push({
        type: 'tool_call',
        tool_name: 'arc_next_test_input',
        tool_input: {},
        tool_output: JSON.stringify({
          status: 'ok',
          current_test_index: k,
          test_count: testCount,
        }),
      });
    }
    const grids = perTest[k];
    grids.forEach((grid, idx) => {
      rawTrace.push({
        type: 'tool_call',
        tool_name: 'arc_set_output_grid',
        tool_input: { grid },
        tool_output: JSON.stringify({ status: 'ok' }),
      });
      rawTrace.push({
        type: 'tool_call',
        tool_name: 'arc_submit_trial',
        tool_input: {},
        tool_output: JSON.stringify({
          status: 'ok',
          trial_count: idx + 1,
          trials_remaining: grids.length - 1 - idx,
          test_index: k,
        }),
      });
    });
  }
}

export function synthesizeArcSubmitTraceFromPrediction(
  workspaceKey: string,
  rawTrace: Array<Record<string, unknown>>,
): void {
  const wsRoot = path.join(resolveWorkspaceRootPath(workspaceKey), 'workspace');
  synthesizeArcSubmitTraceFromWorkspaceDir(wsRoot, rawTrace);
}
