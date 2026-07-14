#!/bin/bash
# run_kcsi.sh — single-entry launcher that auto-detects the right harness
# from a dataset path.
#
# Usage:
#   bash scripts/run_kcsi.sh \
#     --data data/polyglot_medium.json \
#     --model haiku \
#     --name arc2_main \
#     [--preset smoke|main] [--max-tasks 6] [--gens 10] [--seed 42] [--no-drop-solved] [--no-memory]
#
# Defaults follow the paper's MAIN protocol (10 generations, drop-solved on) so
# a bare run is comparable to the other run presets. For a quick single-shot
# smoke, pass `--gens 1` (and `--no-drop-solved` if you want every task retried).
#
# What it does:
#   1. Sniffs the data file's first record to infer task-source
#      (polyglot / swebench_pro / arc).
#   2. Selects the matching evaluator and (where applicable) verifies the
#      required Docker image is present.
#   3. Resolves --model (haiku, sonnet, opus, openai, gpt4o-mini, …) to a
#      provider profile under configs/kcsi/.
#   4. Runs the kcsi CLI with the right flags. Stores artifacts under
#      results/<name>/ for easy follow-up analysis.
#
# Why: pre-fix users had to memorize --task-source / --evaluator / --runtime
# triples per benchmark. A wrong combination (e.g. polyglot data with
# swebench_pro evaluator) silently scored 0 across the board.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../benchmarks/common.sh"

usage() {
  sed -n '2,28p' "$0"
}

require_value_arg() {
  local flag="$1"
  local value="${2-}"
  if [[ -z "$value" || "$value" == -* ]]; then
    echo "Missing value for $flag" >&2
    usage >&2
    exit 2
  fi
}

# ---------- Parse args ----------
DATA=""
MODEL=""
NAME=""
MAX_TASKS=""
PRESET="${PRESET:-main}"
GENERATIONS="${GENERATIONS:-}"
SEED="${SEED:-42}"
DROP_SOLVED_FLAG=""
TASK_SOURCE_OVERRIDE=""
EVALUATOR_OVERRIDE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) require_value_arg "$1" "${2-}"; DATA="$2"; shift 2;;
    --model) require_value_arg "$1" "${2-}"; MODEL="$2"; shift 2;;
    --name) require_value_arg "$1" "${2-}"; NAME="$2"; shift 2;;
    --preset) require_value_arg "$1" "${2-}"; PRESET="$2"; shift 2;;
    --max-tasks) require_value_arg "$1" "${2-}"; MAX_TASKS="$2"; shift 2;;
    --gens|--generations) require_value_arg "$1" "${2-}"; GENERATIONS="$2"; shift 2;;
    --seed) require_value_arg "$1" "${2-}"; SEED="$2"; shift 2;;
    --drop-solved) DROP_SOLVED_FLAG="--drop-solved"; shift;;
    --no-drop-solved) DROP_SOLVED_FLAG="--no-drop-solved"; shift;;
    --task-source) require_value_arg "$1" "${2-}"; TASK_SOURCE_OVERRIDE="$2"; shift 2;;
    --evaluator) require_value_arg "$1" "${2-}"; EVALUATOR_OVERRIDE="$2"; shift 2;;
    -h|--help)
      usage
      exit 0
      ;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "$DATA" || -z "$MODEL" || -z "$NAME" ]]; then
  echo "ERROR: --data, --model, and --name are required." >&2
  echo "Run: $0 --help" >&2
  exit 2
fi

case "$PRESET" in
  main)
    GENERATIONS="${GENERATIONS:-10}"
    if [[ -z "$DROP_SOLVED_FLAG" ]]; then
      DROP_SOLVED_FLAG="--drop-solved"
    fi
    ;;
  smoke)
    GENERATIONS="${GENERATIONS:-1}"
    if [[ -z "$DROP_SOLVED_FLAG" ]]; then
      DROP_SOLVED_FLAG="--no-drop-solved"
    fi
    ;;
  *) echo "ERROR: --preset must be smoke or main." >&2; exit 2 ;;
esac

if [[ -n "$TASK_SOURCE_OVERRIDE" ]]; then
  maybe_validate_file "$DATA" "Dataset"
else
  validate_file "$DATA" "Dataset"
fi

# ---------- Detect task source ----------
detect_task_source() {
  local data_path="$1"
  uv run python - "$data_path" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
# Try JSON first; fall back to JSONL.
record = None
try:
    payload = json.loads(text)
    if isinstance(payload, list) and payload:
        record = payload[0]
    elif isinstance(payload, dict):
        record = payload
except json.JSONDecodeError:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
if not isinstance(record, dict):
    print("unknown")
    raise SystemExit(0)

# Polyglot signature: language + exercise_name + test_files
if {"language", "exercise_name"}.issubset(record.keys()):
    print("polyglot")
elif "task_source" in record:
    print(str(record["task_source"]).lower())
elif {"train_pairs", "test_inputs"}.issubset(record.keys()) or "arc_split" in record:
    print("arc")
elif {"instance_id", "repo", "base_commit"}.issubset(record.keys()) or "selected_test_files_to_run" in record:
    print("swebench_pro")
else:
    print("unknown")
PY
}

if [[ -n "$TASK_SOURCE_OVERRIDE" ]]; then
  TASK_SOURCE="$TASK_SOURCE_OVERRIDE"
else
  TASK_SOURCE="$(detect_task_source "$DATA")"
fi
echo "[run_kcsi] detected task_source=$TASK_SOURCE for $DATA"

case "$TASK_SOURCE" in
  polyglot)
    EVALUATOR="${EVALUATOR_OVERRIDE:-polyglot_harness}"
    REQUIRED_IMAGE="${POLYGLOT_DOCKER_IMAGE:-kcsi-polyglot-eval:latest}"
    ;;
  swebench_pro)
    EVALUATOR="${EVALUATOR_OVERRIDE:-swebench_pro}"
    REQUIRED_IMAGE="$CONTAINER_IMAGE"
    ;;
  arc)
    EVALUATOR="${EVALUATOR_OVERRIDE:-arc_session}"
    REQUIRED_IMAGE="$CONTAINER_IMAGE"
    ;;
  *)
    echo "ERROR: could not infer task source from $DATA. Pass --task-source to this wrapper or use uv run python -m kcsi.cli directly." >&2
    exit 3
    ;;
esac

# ---------- Resolve provider profile ----------
resolve_profile() {
  local model="$1"
  case "$model" in
    haiku) echo "configs/kcsi/.env.haiku";;
    sonnet) echo "configs/kcsi/.env.sonnet";;
    sonnet35) echo "configs/kcsi/.env.sonnet35";;
    opus) echo "configs/kcsi/.env.opus";;
    openai|gpt4o-mini|openai-mini) echo "${OPENAI_PROFILE:-configs/kcsi/.env.openai}";;
    *)
      # Treat as literal path to a profile.
      echo "$model";;
  esac
}

PROFILE="$(resolve_profile "$MODEL")"
maybe_validate_file "$PROFILE" "Provider profile"

# ---------- Verify required Docker image ----------
# Under DRY_RUN we only print the command below, so the image check is skipped
# entirely — a dry run must never require Docker to be set up.
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[run_kcsi] (dry-run) skipping required Docker image check for '$REQUIRED_IMAGE'"
elif ! docker image inspect "$REQUIRED_IMAGE" >/dev/null 2>&1; then
  echo "ERROR: Required Docker image '$REQUIRED_IMAGE' is missing." >&2
  case "$TASK_SOURCE" in
    polyglot)
      echo "  Build it with:" >&2
      echo "    uv run python -c \"from kcsi.benchmarks.polyglot_docker import build_image; build_image()\"" >&2
      ;;
    swebench_pro|arc)
      echo "  Build the kcsi agent image with:" >&2
      echo "    bash container/build.sh --bench" >&2
      ;;
  esac
  exit 4
else
  echo "[run_kcsi] image '$REQUIRED_IMAGE' present"
fi

# ---------- Output paths ----------
OUTPUT_DIR="$LOG_DIR/$NAME"
mkdir -p "$OUTPUT_DIR"
KNOWLEDGE_DB="$OUTPUT_DIR/knowledge.sqlite"
RUNTIME_DB="$OUTPUT_DIR/runtime.sqlite"
RESULT_JSON="$OUTPUT_DIR/result.json"
LAUNCH_LOG="$OUTPUT_DIR/launch.log"

# ---------- Build the CLI invocation ----------
CMD=(
  uv run python -m kcsi.cli
  --task-source "$TASK_SOURCE"
  --tasks-path "$DATA"
  --evaluator "$EVALUATOR"
  --runtime container
  --provider-profile "$PROFILE"
  --generations "$GENERATIONS"
  --seed "$SEED"
  "$DROP_SOLVED_FLAG"
  --knowledge-db-path "$KNOWLEDGE_DB"
  --runtime-db-path "$RUNTIME_DB"
  --experiment-name "$NAME"
  --output-json "$RESULT_JSON"
)

if [[ -n "$MAX_TASKS" ]]; then
  CMD+=(--max-tasks "$MAX_TASKS")
fi

# Forward any extra flags the user passed (e.g. --no-memory, --per-task-forum-rounds).
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[DRY RUN] run_kcsi would launch (preset=$PRESET, generations=$GENERATIONS, drop_solved=$DROP_SOLVED_FLAG):"
  printf '   %q ' "${CMD[@]}"
  echo
  exit 0
fi

echo "[run_kcsi] preset=$PRESET generations=$GENERATIONS drop_solved=$DROP_SOLVED_FLAG"
echo "[run_kcsi] launching:"
printf '   %q ' "${CMD[@]}"
echo
echo "[run_kcsi] log: $LAUNCH_LOG"

run_logged_cmd "$LAUNCH_LOG" "${CMD[@]}"
