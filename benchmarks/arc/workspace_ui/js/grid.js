// Pure grid helpers for the ARC workspace UI.
//
// Extracted from workspace.js so they can be unit-tested under node without
// pulling in DOM-dependent code. Keep this file side-effect free — no
// document or window references, no module-level state.

export const MAX_GRID_SIDE = 30;
export const SYMBOL_COUNT = 10;

export function cloneGrid(values) {
  return values.map((row) => row.slice());
}

export function zeroGrid(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

export function assertGrid(grid) {
  if (!Array.isArray(grid) || grid.length === 0 || !Array.isArray(grid[0])) {
    throw new Error("Grid must be a non-empty 2D array.");
  }
  const cols = grid[0].length;
  if (cols === 0) {
    throw new Error("Grid row 0 must have at least one cell.");
  }
  for (let r = 0; r < grid.length; r += 1) {
    const row = grid[r];
    if (!Array.isArray(row)) {
      throw new Error(`Grid row ${r} is not an array.`);
    }
    if (row.length !== cols) {
      throw new Error(`Grid row ${r} has length ${row.length}; expected ${cols}.`);
    }
    for (let c = 0; c < row.length; c += 1) {
      const v = row[c];
      if (!Number.isInteger(v) || v < 0 || v > 9) {
        throw new Error(`Grid cell [${r}][${c}] = ${v} is not an integer in 0-9.`);
      }
    }
  }
}
