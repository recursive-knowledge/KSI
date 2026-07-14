import {
  MAX_GRID_SIDE,
  SYMBOL_COUNT,
  assertGrid,
  cloneGrid,
  zeroGrid,
} from "./grid.js";

// Must stay in sync with sanitize_task_id() in scripts/arc_prep/_common.py.
// Exported prediction file names are built from task_id, so any character the
// converter's sanitize_task_id rejects here will fail later at conversion time.
const TASK_ID_PATTERN = /^[A-Za-z0-9_\-]+$/;

const state = {
  payload: null,
  outputGrid: null,
  selectedSymbol: 0,
  toolMode: "edit",
  showNumbers: false,
  attemptIndex: 1,
};

function setMessage(text, isError = false) {
  const el = document.getElementById("message");
  el.textContent = text;
  el.style.color = isError ? "#b91c1c" : "#5b6473";
}

function computeCellSize(rows, cols) {
  // Default cell size is 28px. For large grids (up to 30x30) shrink cells so
  // the grid fits inside a typical host width (~420px) without overflow.
  const maxCells = Math.max(rows, cols);
  if (maxCells <= 15) {
    return 28;
  }
  return Math.max(12, Math.floor(420 / maxCells));
}

function renderGrid(host, grid, options = {}) {
  host.innerHTML = "";
  assertGrid(grid);
  const rows = grid.length;
  const cols = grid[0].length;
  const cellSize = computeCellSize(rows, cols);
  const gridEl = document.createElement("div");
  gridEl.className = "grid";
  gridEl.style.setProperty("--cell-size", `${cellSize}px`);
  gridEl.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = `cell symbol_${grid[r][c]}${options.editable ? " editable" : ""}`;
      cell.dataset.row = String(r);
      cell.dataset.col = String(c);
      cell.textContent = state.showNumbers ? String(grid[r][c]) : "";
      if (options.editable) {
        cell.addEventListener("click", () => editCell(r, c));
      } else {
        cell.disabled = true;
      }
      gridEl.appendChild(cell);
    }
  }

  host.appendChild(gridEl);
}

function floodFill(grid, row, col, newValue) {
  const target = grid[row][col];
  if (target === newValue) {
    return;
  }
  const stack = [[row, col]];
  while (stack.length > 0) {
    const [r, c] = stack.pop();
    if (r < 0 || c < 0 || r >= grid.length || c >= grid[0].length) {
      continue;
    }
    if (grid[r][c] !== target) {
      continue;
    }
    grid[r][c] = newValue;
    stack.push([r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]);
  }
}

function editCell(row, col) {
  if (!state.outputGrid) {
    return;
  }
  if (state.toolMode === "floodfill") {
    floodFill(state.outputGrid, row, col, state.selectedSymbol);
  } else {
    state.outputGrid[row][col] = state.selectedSymbol;
  }
  refreshOutputGrid();
}

function renderTrainExamples(train) {
  const host = document.getElementById("train_examples");
  host.innerHTML = "";
  if (!train.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No training examples in payload.";
    host.appendChild(empty);
    return;
  }
  train.forEach((pair, index) => {
    assertGrid(pair.input);
    assertGrid(pair.output);

    const card = document.createElement("div");
    card.className = "example-card";

    const title = document.createElement("div");
    title.className = "example-title";
    title.textContent = `Example ${index}`;
    card.appendChild(title);

    const grids = document.createElement("div");
    grids.className = "example-grids";

    const inputCol = document.createElement("div");
    const inputLabel = document.createElement("div");
    inputLabel.className = "grid-card-label";
    inputLabel.textContent = "Input";
    const inputHost = document.createElement("div");
    inputHost.className = "grid-host";
    inputCol.appendChild(inputLabel);
    inputCol.appendChild(inputHost);

    const outputCol = document.createElement("div");
    const outputLabel = document.createElement("div");
    outputLabel.className = "grid-card-label";
    outputLabel.textContent = "Output";
    const outputHost = document.createElement("div");
    outputHost.className = "grid-host";
    outputCol.appendChild(outputLabel);
    outputCol.appendChild(outputHost);

    grids.appendChild(inputCol);
    grids.appendChild(outputCol);
    card.appendChild(grids);

    renderGrid(inputHost, pair.input);
    renderGrid(outputHost, pair.output);
    host.appendChild(card);
  });
}

function refreshOutputGrid() {
  const host = document.getElementById("output_grid");
  renderGrid(host, state.outputGrid, { editable: true });
  document.getElementById("output_rows").value = String(state.outputGrid.length);
  document.getElementById("output_cols").value = String(state.outputGrid[0].length);
}

function loadPayloadObject(payload) {
  if (!payload || !Array.isArray(payload.train) || !Array.isArray(payload.test_input)) {
    throw new Error("Payload must contain train and test_input fields.");
  }
  if (typeof payload.task_id !== "string" || !TASK_ID_PATTERN.test(payload.task_id)) {
    throw new Error(
      `Payload task_id ${JSON.stringify(payload.task_id)} must match [A-Za-z0-9_-]+ ` +
        "so exported prediction filenames are safe for the converter.",
    );
  }
  if (!Number.isInteger(payload.pair_index) || payload.pair_index < 0) {
    throw new Error(
      `Payload pair_index ${JSON.stringify(payload.pair_index)} must be a non-negative integer.`,
    );
  }
  assertGrid(payload.test_input);
  payload.train.forEach((pair, index) => {
    if (!pair || !Array.isArray(pair.input) || !Array.isArray(pair.output)) {
      throw new Error(`Training pair ${index} must contain input and output grids.`);
    }
    assertGrid(pair.input);
    assertGrid(pair.output);
  });
  state.attemptIndex = 1;
  renderAttemptPicker();
  state.payload = payload;
  state.outputGrid = zeroGrid(payload.test_input.length, payload.test_input[0].length);
  document.getElementById("task_label").textContent =
    `${payload.benchmark} :: ${payload.task_id} :: pair ${payload.pair_index}`;
  document.getElementById("test_input_meta").textContent =
    `${payload.test_input.length} rows x ${payload.test_input[0].length} cols`;
  renderTrainExamples(payload.train);
  renderGrid(document.getElementById("test_input_grid"), payload.test_input);
  refreshOutputGrid();
  updatePredictionPreview();
  setMessage("Payload loaded.");
}

function updatePredictionPreview() {
  const text = state.payload
    ? JSON.stringify(
        {
          benchmark: state.payload.benchmark,
          task_id: state.payload.task_id,
          pair_index: state.payload.pair_index,
          attempt_index: state.attemptIndex,
          prediction: state.outputGrid,
        },
        null,
        2,
      )
    : "";
  document.getElementById("prediction_json").value = text;
}

function downloadPrediction() {
  if (!state.payload) {
    setMessage("Load a payload before exporting a prediction.", true);
    return;
  }
  updatePredictionPreview();
  const blob = new Blob([document.getElementById("prediction_json").value + "\n"], { type: "application/json" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `${state.payload.task_id}_pair${state.payload.pair_index}_attempt${state.attemptIndex}_prediction.json`;
  document.body.appendChild(anchor);
  const blobUrl = anchor.href;
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(blobUrl), 0);
}

function resizeOutput() {
  if (!state.outputGrid) {
    return;
  }
  const rows = Number(document.getElementById("output_rows").value);
  const cols = Number(document.getElementById("output_cols").value);
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 1 || cols < 1 || rows > MAX_GRID_SIDE || cols > MAX_GRID_SIDE) {
    setMessage(`Grid size must be between 1 and ${MAX_GRID_SIDE}.`, true);
    return;
  }

  const next = zeroGrid(rows, cols);
  for (let r = 0; r < Math.min(rows, state.outputGrid.length); r += 1) {
    for (let c = 0; c < Math.min(cols, state.outputGrid[0].length); c += 1) {
      next[r][c] = state.outputGrid[r][c];
    }
  }
  state.outputGrid = next;
  refreshOutputGrid();
  updatePredictionPreview();
}

function copyInput() {
  if (!state.payload) {
    return;
  }
  state.outputGrid = cloneGrid(state.payload.test_input);
  refreshOutputGrid();
  updatePredictionPreview();
}

function resetOutput() {
  if (!state.outputGrid) {
    return;
  }
  state.outputGrid = zeroGrid(state.outputGrid.length, state.outputGrid[0].length);
  refreshOutputGrid();
  updatePredictionPreview();
}

function renderAttemptPicker() {
  const host = document.getElementById("attempt_picker");
  host.innerHTML = "";
  const label = document.createElement("span");
  label.textContent = "Attempt";
  host.appendChild(label);
  for (const value of [1, 2]) {
    const wrap = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "attempt_index";
    input.value = String(value);
    input.checked = state.attemptIndex === value;
    input.addEventListener("change", () => {
      if (input.checked) {
        state.attemptIndex = value;
        updatePredictionPreview();
      }
    });
    const text = document.createTextNode(` ${value}`);
    wrap.appendChild(input);
    wrap.appendChild(text);
    host.appendChild(wrap);
  }
}

function renderSymbolPicker() {
  const host = document.getElementById("symbol_picker");
  host.innerHTML = "";
  for (let value = 0; value < SYMBOL_COUNT; value += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `symbol-button symbol_${value}${value === state.selectedSymbol ? " selected" : ""}`;
    button.title = `Symbol ${value}`;
    button.addEventListener("click", () => {
      state.selectedSymbol = value;
      renderSymbolPicker();
    });
    host.appendChild(button);
  }
}

async function loadPayloadFromPath() {
  const path = document.getElementById("payload_path").value.trim();
  if (!path) {
    setMessage("Provide a relative payload path.", true);
    return;
  }
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    loadPayloadObject(await response.json());
  } catch (error) {
    setMessage(`Failed to load payload path: ${error.message}`, true);
  }
}

async function handleFileInput(event) {
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  try {
    loadPayloadObject(JSON.parse(await file.text()));
  } catch (error) {
    setMessage(`Failed to load payload file: ${error.message}`, true);
  } finally {
    event.target.value = "";
  }
}

function bindEvents() {
  document.getElementById("payload_file").addEventListener("change", handleFileInput);
  document.getElementById("load_path_btn").addEventListener("click", loadPayloadFromPath);
  document.getElementById("resize_btn").addEventListener("click", resizeOutput);
  document.getElementById("copy_btn").addEventListener("click", copyInput);
  document.getElementById("reset_btn").addEventListener("click", resetOutput);
  document.getElementById("preview_btn").addEventListener("click", updatePredictionPreview);
  document.getElementById("download_btn").addEventListener("click", downloadPrediction);
  document.getElementById("show_numbers").addEventListener("change", (event) => {
    state.showNumbers = event.target.checked;
    if (state.payload) {
      renderTrainExamples(state.payload.train);
      renderGrid(document.getElementById("test_input_grid"), state.payload.test_input);
      refreshOutputGrid();
    }
  });
  document.querySelectorAll('input[name="tool_mode"]').forEach((input) => {
    input.addEventListener("change", (event) => {
      state.toolMode = event.target.value;
    });
  });
}

function init() {
  renderSymbolPicker();
  renderAttemptPicker();
  bindEvents();
  document.getElementById("train_examples").innerHTML = '<div class="empty-state">Load a redacted payload to begin.</div>';
  document.getElementById("test_input_grid").innerHTML = '<div class="empty-state">No payload loaded.</div>';
  document.getElementById("output_grid").innerHTML = '<div class="empty-state">No payload loaded.</div>';
}

init();
