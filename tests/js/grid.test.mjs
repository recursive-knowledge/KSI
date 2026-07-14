import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  MAX_GRID_SIDE,
  SYMBOL_COUNT,
  assertGrid,
  cloneGrid,
  zeroGrid,
} from "../../benchmarks/arc/workspace_ui/js/grid.js";

describe("grid.js constants", () => {
  it("exposes expected grid side and symbol count", () => {
    assert.equal(MAX_GRID_SIDE, 30);
    assert.equal(SYMBOL_COUNT, 10);
  });
});

describe("cloneGrid", () => {
  it("produces a deep copy of rows so edits do not alias", () => {
    const original = [
      [1, 2],
      [3, 4],
    ];
    const copy = cloneGrid(original);
    assert.deepEqual(copy, original);
    copy[0][0] = 9;
    assert.equal(original[0][0], 1, "mutating copy must not touch original");
  });
});

describe("zeroGrid", () => {
  it("creates a rows x cols grid filled with zeros", () => {
    const grid = zeroGrid(2, 3);
    assert.deepEqual(grid, [
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it("does not alias rows", () => {
    const grid = zeroGrid(2, 2);
    grid[0][0] = 5;
    assert.equal(grid[1][0], 0);
  });
});

describe("assertGrid", () => {
  it("accepts a valid grid", () => {
    assert.doesNotThrow(() => assertGrid([[0, 5, 9]]));
    assert.doesNotThrow(() =>
      assertGrid([
        [1, 2, 3],
        [4, 5, 6],
      ]),
    );
  });

  it("rejects non-array input", () => {
    assert.throws(() => assertGrid("not-a-grid"), /non-empty 2D array/);
    assert.throws(() => assertGrid(null), /non-empty 2D array/);
    assert.throws(() => assertGrid(undefined), /non-empty 2D array/);
  });

  it("rejects an empty grid", () => {
    assert.throws(() => assertGrid([]), /non-empty 2D array/);
  });

  it("rejects a grid whose first row is not an array", () => {
    assert.throws(() => assertGrid([42]), /non-empty 2D array/);
  });

  it("rejects a grid whose first row is empty", () => {
    assert.throws(() => assertGrid([[]]), /at least one cell/);
  });

  it("rejects ragged rows", () => {
    assert.throws(
      () =>
        assertGrid([
          [1, 2],
          [3],
        ]),
      /has length 1; expected 2/,
    );
  });

  it("rejects a non-array later row", () => {
    assert.throws(() => assertGrid([[1, 2], "oops"]), /row 1 is not an array/);
  });

  it("rejects non-integer cells", () => {
    assert.throws(() => assertGrid([[1.5]]), /integer in 0-9/);
    assert.throws(() => assertGrid([["x"]]), /integer in 0-9/);
    assert.throws(() => assertGrid([[null]]), /integer in 0-9/);
  });

  it("rejects cells out of [0, 9]", () => {
    assert.throws(() => assertGrid([[10]]), /integer in 0-9/);
    assert.throws(() => assertGrid([[-1]]), /integer in 0-9/);
  });

  it("rejects booleans (they are not type 'number' so Number.isInteger rejects them)", () => {
    // Pins the parity with Python's validate_grid, which also rejects booleans.
    assert.throws(() => assertGrid([[true, false]]), /integer in 0-9/);
  });
});
