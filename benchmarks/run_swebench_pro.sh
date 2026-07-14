#!/bin/bash
# SWE-bench Pro run — one LLM per invocation (paper OURS matrix).
#
# Usage:
#   bash benchmarks/run_swebench_pro.sh <haiku|openai> [swarm|noforum ...]
#
# Model (required): haiku = Haiku 4.5, openai = GPT-5.4-mini (the paper's two backbones).
# Condition (optional, default: swarm):
#   swarm    — full knowledge-sharing pipeline ON (paper main OURS)
#   noforum  — discussion + distillation OFF, memory ON (ablation)
#
# Seeds: default a single seed (1). To sweep, set SEEDS="1 2 3 4 5"; to draw
#        random seeds, set SEEDS=random N_SEEDS=5. (See resolve_seeds in common.sh.)
#
# Prerequisites:
#   - SWE-bench Pro JSONL file (default: benchmarks/swebench_pro/dataset/test.jsonl)
#   - Task map (default: benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed0_v1.json)
#   - SWE-bench Docker images
#   - uv sync --extra swebench-pro
#
# Environment variables:
#   SWEBENCH_PRO_JSONL    (default: $KSI_ROOT/benchmarks/swebench_pro/dataset/test.jsonl)
#   SWEBENCH_PRO_TASK_MAP (default: $KSI_ROOT/benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed0_v1.json)
#   HAIKU_PROFILE / OPENAI_PROFILE, GENERATIONS (default 10),
#   SEEDS (default "1"), RUNTIME_TIMEOUT (default 1800), SWEBENCH_TIMEOUT (default 3600)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL="${1:-}"
if [ "$#" -gt 0 ]; then shift; fi
CONDITIONS=("$@")
[ ${#CONDITIONS[@]} -eq 0 ] && CONDITIONS=(swarm)

resolve_model_profile "$MODEL"   # sets MODEL_PROFILE / MODEL_TAG / MODEL_LABEL
resolve_seeds                    # sets SEEDS_LIST

GENERATIONS="${GENERATIONS:-10}"
# Per-task AGENT runtime cap (distinct from the SWE-bench grader timeout below).
# Paper MAIN protocol = 1800s (frozen baseline sweeps raise this to 3600s).
# Pinned explicitly so it does not silently inherit a stray
# CROSS_RUNNER_AGENT_TIMEOUT_SEC from the shell (cli.py reads that env var as the
# --runtime-timeout-sec default).
RUNTIME_TIMEOUT="${RUNTIME_TIMEOUT:-1800}"
SWEBENCH_PRO_JSONL="${SWEBENCH_PRO_JSONL:-$KSI_ROOT/benchmarks/swebench_pro/dataset/test.jsonl}"
SWEBENCH_PRO_TASK_MAP="${SWEBENCH_PRO_TASK_MAP:-$KSI_ROOT/benchmarks/swebench_pro/task_maps/swebench_pro_test_50_seed0_v1.json}"
SWEBENCH_TIMEOUT="${SWEBENCH_TIMEOUT:-3600}"

maybe_validate_file "$SWEBENCH_PRO_JSONL" "SWE-bench Pro JSONL"
maybe_validate_file "$SWEBENCH_PRO_TASK_MAP" "SWE-bench Pro task map"

if is_dry_run && [[ ! -f "$SWEBENCH_PRO_TASK_MAP" ]]; then
  TASK_COUNT="unknown"
else
  TASK_COUNT=$(uv run python - "$SWEBENCH_PRO_TASK_MAP" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("task_count", len(payload.get("tasks", payload.get("task_ids", [])))))
PY
)
fi
echo "SWE-bench Pro run: $TASK_COUNT tasks | Model: $MODEL_TAG | Gens: $GENERATIONS | Seeds: ${SEEDS_LIST[*]}"

COMMON_ARGS=(
  --task-source swebench_pro
  --tasks-path "$SWEBENCH_PRO_JSONL"
  --task-ids-file "$SWEBENCH_PRO_TASK_MAP"
  --evaluator swebench_pro
  --runtime container
  --generations "$GENERATIONS"
  --drop-solved
  --runtime-timeout-sec "$RUNTIME_TIMEOUT"
  --swebench-timeout-sec "$SWEBENCH_TIMEOUT"
)

run_condition() {
  local condition="$1" seed="$2"
  local extra=() label_cond=""
  case "$condition" in
    swarm)   label_cond="Knowledge-Sharing" ;;
    noforum) label_cond="No-Discussion"
             extra=(--per-task-forum-rounds 0 --cross-task-forum-rounds 0 --distill-enabled false) ;;
    *) echo "Unknown condition: $condition (valid: swarm noforum)" >&2; exit 1 ;;
  esac
  local artifact_name="swebench_${MODEL_TAG}_${condition}_seed${seed}"
  maybe_validate_profiles "$MODEL_PROFILE"
  run_experiment "SWE-bench Pro: ${MODEL_LABEL} ${label_cond} (seed ${seed})" \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    --provider-profile "$MODEL_PROFILE" \
    "${extra[@]}" \
    --knowledge-db-path "./${artifact_name}_knowledge.sqlite" \
    --experiment-name "$artifact_name" \
    --output-json "./results/${artifact_name}.json"
}

for seed in "${SEEDS_LIST[@]}"; do
  for c in "${CONDITIONS[@]}"; do
    run_condition "$c" "$seed"
  done
done

echo ""
echo "SWE-BENCH PRO RUN COMPLETE (${MODEL_TAG}, conditions: ${CONDITIONS[*]}, seeds: ${SEEDS_LIST[*]}) at $(date)"
