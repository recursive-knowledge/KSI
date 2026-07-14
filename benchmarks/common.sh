#!/bin/bash
# Shared helpers for experiment scripts.
# Source this file — do not execute directly.
#
# Provides:
#   run_experiment NAME [cli args...]   — run ksi CLI with banner/timing/logging
#   validate_file PATH LABEL            — exit 1 if file missing
#   validate_dir  PATH LABEL            — exit 1 if dir  missing
#   validate_arc_task_map MAP DATA LABEL - fail closed on ARC task-map provenance / ID drift
#   load_task_ids TASK_MAP_JSON         — prints comma-separated IDs from a task map
#   arc_default_path BENCH FIELD        — prints an absolute ARC default path from configs/benchmarks/arc_defaults.json
#   KSI_ROOT                         — auto-detected repo root
#   SONNET_PROFILE / HAIKU_PROFILE      — provider profile paths (env-overridable)

set -euo pipefail


# ---------- Repo root ----------
_COMMON_CALLER_SOURCE="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
SCRIPT_DIR="$(cd "$(dirname "$_COMMON_CALLER_SOURCE")" && pwd)"  # caller's dir, or this file when sourced directly
KSI_ROOT="${KSI_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$KSI_ROOT"

# ---------- Provider profiles ----------
SONNET_PROFILE="${SONNET_PROFILE:-configs/ksi/.env.sonnet}"
HAIKU_PROFILE="${HAIKU_PROFILE:-configs/ksi/.env.haiku}"
OPENAI_PROFILE="${OPENAI_PROFILE:-configs/ksi/.env.openai}"
ARC_DEFAULTS_CONFIG="${ARC_DEFAULTS_CONFIG:-$KSI_ROOT/configs/benchmarks/arc_defaults.json}"

# ---------- Container image ----------
# Default to :bench (lighter) for benchmark experiments.
# Override with CONTAINER_IMAGE=ksi-agent:latest for interactive tasks.
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-ksi-agent:bench}"

# ---------- Results & logs dirs ----------
LOG_DIR="${KSI_ROOT}/results"
EXPERIMENT_LOG_DIR="/tmp/ksi-experiments"
mkdir -p "$LOG_DIR" "$EXPERIMENT_LOG_DIR"

# ---------- Helpers ----------

validate_file() {
  local path="$1" label="$2"
  if [ ! -f "$path" ]; then
    echo "ERROR: $label not found at $path" >&2
    exit 1
  fi
}

validate_dir() {
  local path="$1" label="$2"
  if [ ! -d "$path" ]; then
    echo "ERROR: $label not found at $path" >&2
    exit 1
  fi
}

validate_profiles() {
  for profile in "$@"; do
    validate_file "$profile" "Provider profile"
  done
}

is_dry_run() {
  [[ "${DRY_RUN:-false}" == "true" ]]
}

maybe_validate_file() {
  local path="$1" label="$2"
  if is_dry_run && [[ ! -f "$path" ]]; then
    echo "WARNING (dry-run): $label not found at $path — previewing command anyway" >&2
    return 0
  fi
  validate_file "$path" "$label"
}

maybe_validate_dir() {
  local path="$1" label="$2"
  if is_dry_run && [[ ! -d "$path" ]]; then
    echo "WARNING (dry-run): $label not found at $path — previewing command anyway" >&2
    return 0
  fi
  validate_dir "$path" "$label"
}

maybe_validate_profiles() {
  if is_dry_run; then
    for profile in "$@"; do
      if [[ ! -f "$profile" ]]; then
        echo "WARNING (dry-run): Provider profile not found at $profile — previewing command anyway" >&2
      fi
    done
    return 0
  fi
  validate_profiles "$@"
}

validate_arc_task_map() {
  local task_map="$1" tasks_path="$2" label="${3:-ARC task map}"
  validate_file "$task_map" "$label"
  if [ ! -e "$tasks_path" ]; then
    echo "ERROR: ARC tasks path not found at $tasks_path" >&2
    exit 1
  fi
  uv run python benchmarks/scripts/dataprep/validate_task_map.py \
    --task-map "$task_map" \
    --task-source arc \
    --tasks-path "$tasks_path" \
    --require-provenance >/dev/null
}

maybe_validate_arc_task_map() {
  local task_map="$1" tasks_path="$2" label="${3:-ARC task map}"
  if is_dry_run && { [[ ! -f "$task_map" ]] || [[ ! -e "$tasks_path" ]]; }; then
    echo "WARNING (dry-run): cannot validate $label at $task_map against $tasks_path - previewing command anyway" >&2
    return 0
  fi
  validate_arc_task_map "$task_map" "$tasks_path" "$label"
}

run_logged_cmd() {
  local logfile="$1"
  shift
  local cmd_rc=0 tee_rc=0
  local -a rc=()
  set +e
  set +o pipefail
  "$@" 2>&1 | tee "$logfile"
  # Capture the whole array in one expansion: any command, including a plain
  # assignment, resets PIPESTATUS.
  rc=("${PIPESTATUS[@]}")
  cmd_rc=${rc[0]}
  tee_rc=${rc[1]}
  set -e
  set -o pipefail
  if [[ "$tee_rc" -ne 0 ]]; then
    echo "ERROR: failed to write log $logfile (tee exit=$tee_rc)" >&2
    return "$tee_rc"
  fi
  return "$cmd_rc"
}

load_task_ids() {
  local task_map="$1"
  validate_file "$task_map" "Task map"
  uv run python - "$task_map" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if "task_ids" in payload:
    ids = payload["task_ids"]
else:
    ids = [item["task_id"] for item in payload.get("tasks", [])]
print(",".join(ids))
PY
}

arc_default_path() {
  local bench="$1" field="$2"
  validate_file "$ARC_DEFAULTS_CONFIG" "ARC defaults config"
  uv run python - "$KSI_ROOT" "$ARC_DEFAULTS_CONFIG" "$bench" "$field" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
bench = sys.argv[3]
field = sys.argv[4]

payload = json.loads(config_path.read_text(encoding="utf-8"))
try:
    value = payload[bench][field]
except KeyError as exc:
    raise SystemExit(f"missing ARC default {bench}.{field} in {config_path}") from exc

path = Path(value)
if not path.is_absolute():
    path = root / path
print(path)
PY
}

# ---------- Model selection (paper backbone matrix: haiku | openai) ----------
# Resolve a run model argument into MODEL_PROFILE / MODEL_TAG / MODEL_LABEL.
# The paper's OURS rows use exactly two backbones: Haiku 4.5 and GPT-5.4-mini.
resolve_model_profile() {
  local model="${1:-}"
  case "$model" in
    haiku)
      MODEL_PROFILE="$HAIKU_PROFILE"; MODEL_TAG="haiku"; MODEL_LABEL="Haiku 4.5" ;;
    openai)
      MODEL_PROFILE="$OPENAI_PROFILE"; MODEL_TAG="openai"; MODEL_LABEL="GPT-5.4-mini" ;;
    "")
      echo "ERROR: model argument required (haiku | openai)" >&2
      exit 1 ;;
    *)
      echo "ERROR: unknown model '$model' (valid: haiku | openai)" >&2
      exit 1 ;;
  esac
}

# ---------- Seed resolution ----------
# Fills the global SEEDS_LIST array from the SEEDS env var.
#   (unset)            -> a single seed: 1   (default; prompt says how to expand)
#   SEEDS="1 2 3 4 5"  -> that explicit list  (space- or comma-separated)
#   SEEDS=random       -> N_SEEDS distinct random seeds (N_SEEDS default 5)
# Legacy single-seed SEED=<n> is honored when SEEDS is unset.
resolve_seeds() {
  local raw="${SEEDS:-}"
  if [[ -z "$raw" && -n "${SEED:-}" ]]; then
    raw="$SEED"
  fi
  raw="${raw:-1}"
  SEEDS_LIST=()
  if [[ "$raw" == "random" ]]; then
    local n="${N_SEEDS:-5}" i s
    for ((i = 0; i < n; i++)); do
      s=$((RANDOM % 90000 + 10000))
      SEEDS_LIST+=("$s")
    done
    echo "[seeds] random mode: drew ${SEEDS_LIST[*]} (N_SEEDS=$n)" >&2
  else
    read -r -a SEEDS_LIST <<< "${raw//,/ }"
  fi
  echo "[seeds] running seed(s): ${SEEDS_LIST[*]}  (default is 1; set SEEDS=\"1 2 3 4 5\" to sweep, or SEEDS=random N_SEEDS=5)" >&2
}

run_experiment() {
  local NAME="$1"
  shift

  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    local knowledge_db="" runtime_db="" no_runtime_db=0
    local _prev="" _arg=""
    for _arg in "$@"; do
      if [[ "$_prev" == "--knowledge-db-path" ]]; then
        knowledge_db="$_arg"
      elif [[ "$_prev" == "--runtime-db-path" ]]; then
        runtime_db="$_arg"
      elif [[ "$_arg" == "--no-runtime-db" ]]; then
        no_runtime_db=1
      fi
      _prev="$_arg"
    done
    echo "[DRY RUN] $NAME"
    # %q preserves per-arg quoting so args with spaces/globs render copy-pastably.
    printf '  Command: uv run python -m ksi.cli'
    printf ' %q' "$@"
    printf '\n\n'
    if [[ "$no_runtime_db" == "0" ]]; then
      if [[ -n "$runtime_db" ]]; then
        echo "  Runtime DB: $runtime_db"
      elif [[ -n "$knowledge_db" ]]; then
        uv run python - "$knowledge_db" <<'PY'
import sys
from ksi.layout import derive_runtime_sibling

print(f"  Runtime DB: {derive_runtime_sibling(sys.argv[1])}")
PY
      fi
    fi
    return 0
  fi

  # Auto-generate log path from experiment-name arg
  local LOG_FILE=""
  local _prev=""
  for i in "$@"; do
    if [[ "$_prev" == "--experiment-name" ]]; then
      LOG_FILE="${EXPERIMENT_LOG_DIR}/${i}.log"
      break
    fi
    _prev="$i"
  done
  if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="${EXPERIMENT_LOG_DIR}/experiment-$(date +%Y%m%d-%H%M%S).log"
  fi

  echo ""
  echo "=========================================="
  echo "$NAME"
  echo "Started: $(date)"
  echo "Log: $LOG_FILE"
  echo "=========================================="

  # Run with tee to both stdout and log file.
  run_logged_cmd "$LOG_FILE" uv run python -m ksi.cli "$@"

  echo "=========================================="
  echo "$NAME finished at $(date)"
  echo "Log saved: $LOG_FILE"
  echo "=========================================="
}
