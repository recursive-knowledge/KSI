#!/bin/bash
# Polyglot benchmark run — one LLM per invocation (paper OURS matrix).
#
# Usage:
#   bash benchmarks/run_polyglot.sh <haiku|openai> [swarm|noforum ...]
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
#   - data/polyglot_medium.json (uv run python benchmarks/scripts/dataprep/prepare_polyglot_dataset.py)
#   - ksi-polyglot-eval:latest Docker image
#
# Environment variables:
#   HAIKU_PROFILE / OPENAI_PROFILE, GENERATIONS (default 10),
#   SEEDS (default "1"), RUNTIME_TIMEOUT (default 1800)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL="${1:-}"
if [ "$#" -gt 0 ]; then shift; fi
CONDITIONS=("$@")
[ ${#CONDITIONS[@]} -eq 0 ] && CONDITIONS=(swarm)

resolve_model_profile "$MODEL"   # sets MODEL_PROFILE / MODEL_TAG / MODEL_LABEL
resolve_seeds                    # sets SEEDS_LIST

GENERATIONS="${GENERATIONS:-10}"
# Per-task agent runtime cap. Paper MAIN protocol = 1800s (the frozen baseline
# sweeps raise this to 3600s). Pinned explicitly so it does not silently inherit
# a stray CROSS_RUNNER_AGENT_TIMEOUT_SEC from the shell (cli.py reads that env
# var as the --runtime-timeout-sec default).
RUNTIME_TIMEOUT="${RUNTIME_TIMEOUT:-1800}"
TASKS_PATH="$KSI_ROOT/data/polyglot_medium.json"
maybe_validate_file "$TASKS_PATH" "Polyglot dataset"

COMMON_ARGS=(
  --task-source polyglot
  --tasks-path "$TASKS_PATH"
  --evaluator polyglot_harness
  --runtime container
  --generations "$GENERATIONS"
  --drop-solved
  --runtime-timeout-sec "$RUNTIME_TIMEOUT"
  # Pin the per-test execution budget explicitly to the value that produced the
  # reported numbers (== DEFAULT_POLYGLOT_TIMEOUT_SEC=180) rather than riding the
  # mutable code default. Reconciles paper Appendix F (issue #1141).
  --polyglot-timeout-sec 180
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
  local artifact_name="polyglot_${MODEL_TAG}_${condition}_seed${seed}"
  maybe_validate_profiles "$MODEL_PROFILE"
  run_experiment "Polyglot: ${MODEL_LABEL} ${label_cond} (seed ${seed})" \
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
echo "POLYGLOT RUN COMPLETE (${MODEL_TAG}, conditions: ${CONDITIONS[*]}, seeds: ${SEEDS_LIST[*]}) at $(date)"
