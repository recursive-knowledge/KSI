#!/bin/bash
# ARC-AGI run — one LLM per invocation (paper OURS matrix).
#
# Usage:
#   bash benchmarks/run_arc.sh <1|2> <haiku|openai> [swarm|noforum ...]
#   bash benchmarks/run_arc.sh 2 haiku            # ARC2, Haiku, swarm
#   bash benchmarks/run_arc.sh 1 openai swarm noforum
#
# Version (required): 1 = ARC-AGI-1, 2 = ARC-AGI-2.
# Model (required):   haiku = Haiku 4.5, openai = GPT-5.4-mini (the paper's two backbones).
# Condition (optional, default: swarm):
#   swarm    — full knowledge-sharing pipeline ON (paper main OURS)
#   noforum  — discussion + distillation OFF, memory ON (ablation)
#
# Seeds: default a single seed (1). To sweep, set SEEDS="1 2 3 4 5"; to draw
#        random seeds, set SEEDS=random N_SEEDS=5. (See resolve_seeds in common.sh.)
#
# Environment variables:
#   ARC1_DATA_DIR / ARC2_DATA_DIR, HAIKU_PROFILE / OPENAI_PROFILE,
#   GENERATIONS (default 10), SEEDS (default "1"), RUNTIME_TIMEOUT (default 1800)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ARC1_DATA_DIR="${ARC1_DATA_DIR:-$(arc_default_path arc1 task_dir)}"
ARC2_DATA_DIR="${ARC2_DATA_DIR:-$(arc_default_path arc2 task_dir)}"

VERSION="${1:-}"
if [ "$#" -gt 0 ]; then shift; fi
MODEL="${1:-}"
if [ "$#" -gt 0 ]; then shift; fi
CONDITIONS=("$@")
[ ${#CONDITIONS[@]} -eq 0 ] && CONDITIONS=(swarm)

case "$VERSION" in
  1) DATA_DIR="$ARC1_DATA_DIR"; TASK_MAP="$(arc_default_path arc1 task_map)"; PREFIX="arc1" ;;
  2) DATA_DIR="$ARC2_DATA_DIR"; TASK_MAP="$(arc_default_path arc2 task_map)"; PREFIX="arc2" ;;
  *) echo "Usage: $0 <1|2> <haiku|openai> [swarm|noforum ...]" >&2; exit 1 ;;
esac

resolve_model_profile "$MODEL"   # sets MODEL_PROFILE / MODEL_TAG / MODEL_LABEL
resolve_seeds                    # sets SEEDS_LIST

# Generation count is overridable via GENERATIONS; the ARC protocol default is 10.
GENERATIONS="${GENERATIONS:-10}"
# Per-task agent runtime cap. Paper MAIN protocol = 1800s (the frozen baseline
# sweeps raise this to 3600s). Pinned explicitly so it does not silently inherit
# a stray CROSS_RUNNER_AGENT_TIMEOUT_SEC from the shell (cli.py reads that env
# var as the --runtime-timeout-sec default).
RUNTIME_TIMEOUT="${RUNTIME_TIMEOUT:-1800}"

maybe_validate_dir "$DATA_DIR" "ARC-AGI $VERSION data"
maybe_validate_arc_task_map "$TASK_MAP" "$DATA_DIR" "ARC-AGI $VERSION task map"

if is_dry_run && [[ ! -f "$TASK_MAP" ]]; then
  TASK_COUNT="unknown"
else
  TASK_IDS=$(load_task_ids "$TASK_MAP")
  TASK_COUNT=$(echo "$TASK_IDS" | tr ',' '\n' | wc -l)
fi
echo "ARC$VERSION run: $TASK_COUNT tasks | Model: $MODEL_TAG | Gens: $GENERATIONS | Seeds: ${SEEDS_LIST[*]}"

run_condition() {
  local condition="$1" seed="$2"
  local extra=() label_cond=""
  case "$condition" in
    swarm)   label_cond="Knowledge-Sharing" ;;
    noforum) label_cond="No-Discussion"
             extra=(--per-task-forum-rounds 0 --cross-task-forum-rounds 0 --distill-enabled false) ;;
    *) echo "Unknown condition: $condition (valid: swarm noforum)" >&2; exit 1 ;;
  esac
  local artifact_name="${PREFIX}_${MODEL_TAG}_${condition}_seed${seed}"
  maybe_validate_profiles "$MODEL_PROFILE"
  run_experiment "ARC$VERSION: ${MODEL_LABEL} ${label_cond} ($TASK_COUNT tasks, seed ${seed})" \
    --task-source arc \
    --tasks-path "$DATA_DIR" \
    --evaluator arc_session \
    --arc-max-trials 2 \
    --runtime container \
    --task-ids-file "$TASK_MAP" \
    --task-map-path "$TASK_MAP" \
    --generations "$GENERATIONS" \
    --seed "$seed" \
    --drop-solved \
    --runtime-timeout-sec "$RUNTIME_TIMEOUT" \
    --provider-profile "$MODEL_PROFILE" \
    "${extra[@]}" \
    --knowledge-db-path "./${artifact_name}_knowledge.sqlite" \
    --experiment-name "$artifact_name" \
    --output-json "./results/${artifact_name}.json"
}

for seed in "${SEEDS_LIST[@]}"; do
  echo "---- Seed $seed ----"
  for c in "${CONDITIONS[@]}"; do
    run_condition "$c" "$seed"
  done
done

echo ""
echo "ARC$VERSION RUN COMPLETE (${MODEL_TAG}, conditions: ${CONDITIONS[*]}, seeds: ${SEEDS_LIST[*]}) at $(date)"
