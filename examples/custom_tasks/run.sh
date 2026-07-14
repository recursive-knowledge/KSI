#!/usr/bin/env bash
# Run KCSI on the bundled custom-tasks demo (CLI form).
# Requires: docker, node, a provider profile (see README).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python -m kcsi.cli \
  --task-source custom \
  --tasks-path examples/custom_tasks/tasks.jsonl \
  --evaluator command \
  --generations 1 \
  --provider-profile "${PROVIDER_PROFILE:-configs/kcsi/.env.haiku}" \
  --experiment-name custom_demo
