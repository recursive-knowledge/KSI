#!/usr/bin/env bash
# Live end-to-end smoke for the polyglot test-feedback retry loop.
# Proves the whole pipeline runs against a real provider in Docker: on a
# failing attempt 1, the solver sees its own capped test-runner output and
# gets a second try in the same SDK session before scoring.
#
# Chosen instance: java__bowling (edge-case-heavy: strikes/spares/the special
# tenth-frame fill-ball rules are easy to get subtly wrong on a naive first
# pass, and a small model can plausibly self-correct once it sees a concrete
# assertion failure). Pick per Step 1 of the task brief:
#   grep -o '"[a-z+]*__bowling"' benchmarks/polyglot/task_maps/*.json
#
# Prerequisites:
#   - kcsi-agent:bench and kcsi-polyglot-eval:latest Docker images built
#   - data/polyglot_medium.json containing the chosen instance_id, e.g.:
#       echo '["java__bowling"]' > /tmp/subset.json
#       uv run python benchmarks/scripts/dataprep/prepare_polyglot_dataset.py \
#         --subset-url /tmp/subset.json --output data/polyglot_medium.json
#   - a live provider profile (see configs/kcsi/*.env.*.template)
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-false}"
TASK_ID="${1:?usage: polyglot_test_feedback_smoke.sh <instance_id>}"
TASKS_PATH="${TASKS_PATH:-data/polyglot_medium.json}"
PROVIDER_PROFILE="${PROVIDER_PROFILE:-configs/kcsi/.env.haiku}"

CMD=(uv run python -m kcsi.cli
  --task-source polyglot
  --tasks-path "${TASKS_PATH}"
  --task-ids "${TASK_ID}"
  --evaluator polyglot_harness
  --polyglot-timeout-sec 180
  --runtime container
  --provider-profile "${PROVIDER_PROFILE}"
  --polyglot-test-feedback-tries 2
  --polyglot-test-feedback-max-lines 50
  --no-memory
  --max-concurrent-tasks 1
  --generations 1
  --experiment-name polyglot-tf-smoke
)

if [ "${DRY_RUN}" = "true" ]; then
  printf '%s\n' "${CMD[@]}"
  exit 0
fi

"${CMD[@]}"
