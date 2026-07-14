#!/bin/bash
# Terminal-Bench 2 run — one LLM per invocation (paper OURS matrix).
#
# Usage:
#   bash benchmarks/run_terminal_bench_2.sh <haiku|openai> [swarm|noforum ...] [--no-drop-solved]
#
# Model (required): haiku = Haiku 4.5, openai = GPT-5.4-mini (the paper's two backbones).
# Condition (optional, default: swarm):
#   swarm    — full knowledge-sharing pipeline ON (paper main OURS)
#   noforum  — discussion + distillation OFF, memory ON (ablation)
#
# Seeds: default a single seed (1). To sweep, set SEEDS="1 2 3 4 5"; to draw
#        random seeds, set SEEDS=random N_SEEDS=5. (See resolve_seeds in common.sh.)
#
# NOTE ON THE PUBLISHED 47.2% TB2 NUMBER:
#   The exact configuration behind the previously reported 47.2% TB2 result
#   (generations, backbone model, concurrency) is NOT recoverable from code —
#   no wrapper encoded it and it lives only in that run's logs. It must be
#   backfilled from the run logs by the maintainer; do NOT infer it from this
#   file. This wrapper is the canonical GO-FORWARD TB2 config, grounded in the
#   fairness-mode recipe documented in
#   benchmarks/terminal_bench_2/INTEGRATION.md — it is not a reconstruction of
#   the 47.2% run.
#
# PROTOCOL + HARBOR KNOBS (see INTEGRATION.md):
#   Default config follows the paper's main protocol (10 generations, drop-solved).
#   The image-pull / step-cap / wall-time knobs below remain Harbor-aligned, but
#   drop-solved + 10 generations depart from Harbor's >=5-attempts-per-task rule.
#   - KCSI_TB2_REQUIRE_PULL=1 is exported below: a task whose canonical image
#     cannot be pulled fails that trial rather than silently rebuilding locally;
#     healthy siblings continue unless every dispatched trial has a registry
#     acquisition failure, which aborts after their traces are persisted.
#   - For publishable image-byte reproducibility, set
#     KCSI_TB2_IMAGE_DIGEST_MANIFEST=/path/to/image_digests.json. When set, the
#     runtime aborts before the task starts if the pulled registry digest is
#     missing from the manifest or differs from it.
#   - KCSI_TB2_MAX_STEPS is left UNSET (no kcsi-side step cap); the per-task
#     [agent].timeout_sec from task.toml is the sole wall-time bound.
#   - The runtime timeout is FIXED at a negative --runtime-timeout-sec (no kcsi
#     hard container cap). NOT configurable: setting TIMEOUT aborts the run.
#   - drop-solved defaults ON and GENERATIONS defaults to 10 (paper MAIN
#     protocol). For a Harbor submission pass `--no-drop-solved` and keep
#     GENERATIONS>=5 so every task is attempted every generation.
#   - --runtime-db-path is set per condition so the runtime audit DB exists for
#     kcsi.benchmarks.tb2_submission (Harbor submission needs the runtime sidecar).
#
# Environment variables:
#   TB2_TASK_MAP    (default: $KCSI_ROOT/benchmarks/terminal_bench_2/task_maps/terminal_bench_2_all.json)
#   HAIKU_PROFILE / OPENAI_PROFILE, GENERATIONS (default 10), SEEDS (default "1"),
#   MAX_CONCURRENT (default 25). TIMEOUT is NOT configurable (aborts the run).

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# TB2's per-task task.toml [agent].timeout_sec is the AUTHORITATIVE wall-time
# bound (Harbor parity; e.g. build-pov-ray = 12000s). The KCSI-side runtime hard
# cap is deliberately NOT configurable for TB2: the run always passes a
# negative --runtime-timeout-sec (no hard cap) so the per-task timeout binds.
if [[ -n "${TIMEOUT:-}" ]]; then
  echo "ERROR: TIMEOUT is not configurable for the Terminal-Bench 2 run." >&2
  echo "       The per-task task.toml [agent].timeout_sec is the authoritative" >&2
  echo "       wall-time bound (Harbor parity). Unset TIMEOUT and re-run." >&2
  exit 1
fi

# ---- Parse args: <model> [conditions...] [--(no-)drop-solved] in any order ----
MODEL=""
CONDITIONS=()
DROP_SOLVED_FLAG="${DROP_SOLVED_FLAG:---drop-solved}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --drop-solved)    DROP_SOLVED_FLAG="--drop-solved" ;;
    --no-drop-solved) DROP_SOLVED_FLAG="--no-drop-solved" ;;
    *)
      if [[ -z "$MODEL" ]]; then MODEL="$1"; else CONDITIONS+=("$1"); fi ;;
  esac
  shift
done
[ ${#CONDITIONS[@]} -eq 0 ] && CONDITIONS=(swarm)

resolve_model_profile "$MODEL"   # sets MODEL_PROFILE / MODEL_TAG / MODEL_LABEL
resolve_seeds                    # sets SEEDS_LIST

GENERATIONS="${GENERATIONS:-10}"
MAX_CONCURRENT="${MAX_CONCURRENT:-25}"
TB2_TASK_MAP="${TB2_TASK_MAP:-$KCSI_ROOT/benchmarks/terminal_bench_2/task_maps/terminal_bench_2_all.json}"

# Fairness knobs (see INTEGRATION.md). REQUIRE_PULL on; MAX_STEPS deliberately unset.
export KCSI_TB2_REQUIRE_PULL="${KCSI_TB2_REQUIRE_PULL:-1}"
unset KCSI_TB2_MAX_STEPS

maybe_validate_file "$TB2_TASK_MAP" "Terminal-Bench 2 task map"

if is_dry_run && [[ ! -f "$TB2_TASK_MAP" ]]; then
  TASK_COUNT="unknown"
else
  TASK_COUNT=$(uv run python - "$TB2_TASK_MAP" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("task_count", len(payload.get("tasks", payload.get("task_ids", [])))))
PY
)
fi
echo "Terminal-Bench 2 run: $TASK_COUNT tasks | Model: $MODEL_TAG | Gens: $GENERATIONS | Seeds: ${SEEDS_LIST[*]} | drop-solved: $DROP_SOLVED_FLAG"

COMMON_ARGS=(
  --task-source terminal_bench_2
  --tasks-path "$TB2_TASK_MAP"
  --evaluator terminal_bench_2
  --runtime container
  --generations "$GENERATIONS"
  --max-concurrent-tasks "$MAX_CONCURRENT"
  # Negative = no KCSI hard container cap; the per-task task.toml
  # [agent].timeout_sec is the sole wall-time bound (Harbor parity). Fixed, not
  # configurable — see the TIMEOUT guard above.
  --runtime-timeout-sec -1
  "$DROP_SOLVED_FLAG"
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
  local artifact_name="tb2_${MODEL_TAG}_${condition}_seed${seed}"
  maybe_validate_profiles "$MODEL_PROFILE"
  run_experiment "Terminal-Bench 2: ${MODEL_LABEL} ${label_cond} (seed ${seed})" \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    --provider-profile "$MODEL_PROFILE" \
    "${extra[@]}" \
    --knowledge-db-path "./${artifact_name}_knowledge.sqlite" \
    --runtime-db-path "./${artifact_name}_runtime.sqlite" \
    --experiment-name "$artifact_name" \
    --output-json "./results/${artifact_name}.json"
}

for seed in "${SEEDS_LIST[@]}"; do
  for c in "${CONDITIONS[@]}"; do
    run_condition "$c" "$seed"
  done
done

echo ""
echo "TERMINAL-BENCH 2 RUN COMPLETE (${MODEL_TAG}, conditions: ${CONDITIONS[*]}, seeds: ${SEEDS_LIST[*]}) at $(date)"
