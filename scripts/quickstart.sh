#!/usr/bin/env bash
# One command, fresh clone -> the full knowledge loop on 5 hard ARC-AGI-1 tasks.
#
# No external dataset download and no prior setup_all.sh run needed. Given
# Docker and Node.js 22.16.0 installed (and either uv or a local
# `pip install -e .`),
# this self-bootstraps everything it needs:
#   - a provider profile, synthesized from an ambient ANTHROPIC_API_KEY /
#     OPENAI_API_KEY if you don't already have one (like GEPA reads the key
#     from the environment);
#   - the ksi-agent:bench Docker image (built on first run if missing);
#   - the host runtime_runner Node dependencies.
# Then it runs 3 generations with the forums on over 3 hard ARC-AGI-1 tasks
# (bundled under examples/quickstart/arc1_hard/, no download). ARC tasks are hard
# for every current model, so they don't all solve on generation 1 — unsolved
# tasks carry forward under the default --drop-solved and the full knowledge loop
# (execute -> forum -> distill -> seed) fires end to end across generations.
#
# Usage:
#   ANTHROPIC_API_KEY=sk-ant-... bash scripts/quickstart.sh
#   OPENAI_API_KEY=sk-...        bash scripts/quickstart.sh
#   PROFILE=configs/ksi/.env.openai bash scripts/quickstart.sh
#
# Env knobs:
#   TASKS_PATH=<path>  run your own custom tasks .jsonl/.json (command evaluator)
#                      instead of the default ARC-AGI-1 demo
#   ARC_DATA_DIR=<d>   directory of ARC task JSONs (default: the bundled 3 tasks)
#   ARC_TASK_MAP=<p>   optional ARC task-map JSON to filter/pin the selection
#   EXPERIMENT_NAME=x  name the run (default: quickstart_demo)
#   PROFILE=<path>     provider profile to use (default: configs/ksi/.env.haiku)
#   SKIP_BOOTSTRAP=1   don't build the image / install deps / synthesize a profile
#   SKIP_DOCTOR=1      skip the readiness check
#   DRY_RUN=true       print the run command (skips the image build / npm install,
#                      still synthesizes a profile), don't execute

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KSI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KSI_ROOT"

PROFILE="${PROFILE:-configs/ksi/.env.haiku}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-quickstart_demo}"
AGENT_IMAGE="ksi-agent:bench"

# Demo task selection. Default: 3 hard ARC-AGI-1 tasks bundled under
# examples/quickstart/arc1_hard/ (no download). Set TASKS_PATH to a custom
# .jsonl/.json to run your own tasks through the command evaluator instead.
TASKS_PATH="${TASKS_PATH:-}"
ARC_DATA_DIR="${ARC_DATA_DIR:-examples/quickstart/arc1_hard}"
ARC_TASK_MAP="${ARC_TASK_MAP:-}"

# Prefer uv, but fall back to a plain interpreter so a `pip install`d package
# (no uv) still works. uv is a convenience here, not a hard requirement.
if command -v uv >/dev/null 2>&1; then
  PYRUN=(uv run python)
  DOCTOR=(uv run ksi-doctor)
elif command -v python >/dev/null 2>&1; then
  PYRUN=(python)
  DOCTOR=(python -m ksi.doctor)
elif command -v python3 >/dev/null 2>&1; then
  PYRUN=(python3)
  DOCTOR=(python3 -m ksi.doctor)
else
  echo "ERROR: no uv or python found on PATH." >&2
  exit 1
fi

# --- Bootstrap: make the common "fresh clone" failures self-healing ---------
synthesize_profile() {
  # Write a minimal provider profile from an ambient key if none usable exists.
  [[ -f "$PROFILE" ]] && return 0
  local out="${QUICKSTART_PROFILE_OUT:-configs/ksi/.env.quickstart}"
  mkdir -p "$(dirname "$out")"
  if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    cat > "$out" <<EOF
MODEL_PROVIDER=anthropic
MODEL=claude-haiku-4-5-20251001
MODEL_AUTH_MODE=api
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
EOF
    PROFILE="$out"
    echo "    synthesized $out from ANTHROPIC_API_KEY"
  elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
    cat > "$out" <<EOF
MODEL_PROVIDER=openai
MODEL=gpt-5.4-mini
MODEL_AUTH_MODE=api
OPENAI_API_KEY=${OPENAI_API_KEY}
REASONING_EFFORT=medium
EOF
    PROFILE="$out"
    echo "    synthesized $out from OPENAI_API_KEY"
  fi
}

bootstrap() {
  echo "==> Bootstrapping (set SKIP_BOOTSTRAP=1 to skip)"
  synthesize_profile

  # The image build and npm install are heavy; skip them on a dry run.
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo
    return 0
  fi

  if [[ -d runtime_runner ]] && [[ ! -d runtime_runner/node_modules ]]; then
    if command -v npm >/dev/null 2>&1; then
      echo "    installing runtime_runner dependencies"
      npm --prefix runtime_runner install --legacy-peer-deps --silent
    else
      echo "ERROR: Node.js/npm not found — install Node.js 22.16.0 and re-run." >&2
      exit 1
    fi
  fi

  if command -v docker >/dev/null 2>&1; then
    if [[ -z "$(docker images -q "$AGENT_IMAGE" 2>/dev/null)" ]]; then
      echo "    building $AGENT_IMAGE (first run only; this takes a few minutes)"
      bash container/build.sh --bench
    fi
  else
    echo "ERROR: Docker not found — install Docker and start the daemon." >&2
    exit 1
  fi
  echo
}

if [[ "${SKIP_BOOTSTRAP:-0}" != "1" ]]; then
  bootstrap
fi

if [[ "${SKIP_DOCTOR:-0}" != "1" ]]; then
  echo "==> Checking setup (ksi-doctor)"
  "${DOCTOR[@]}"
  echo
fi

if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: no provider profile and no API key in the environment." >&2
  echo "       Set ANTHROPIC_API_KEY (or OPENAI_API_KEY), or run scripts/setup_all.sh," >&2
  echo "       or pass PROFILE=<path>." >&2
  exit 1
fi

# Full-loop demo: 3 generations with the per-task and cross-task forums on. The
# default task set is 3 hard ARC-AGI-1 tasks — hard enough for any current model
# that they don't all solve on generation 1, so unsolved tasks carry forward
# under the default --drop-solved and every phase fires across generations
# (execute -> forum -> distill -> seed -> next generation). Setting TASKS_PATH
# switches to a custom .jsonl/.json graded by the command evaluator.
if [[ -n "$TASKS_PATH" ]]; then
  TASK_FLAGS=(--task-source custom --tasks-path "$TASKS_PATH" --evaluator command)
  TASK_BANNER="custom tasks from $TASKS_PATH"
  CONCURRENCY=4
else
  TASK_FLAGS=(
    --task-source arc
    --tasks-path "$ARC_DATA_DIR"
    --evaluator arc_session
    --arc-max-trials 2
  )
  [[ -n "$ARC_TASK_MAP" ]] && TASK_FLAGS+=(--task-map-path "$ARC_TASK_MAP")
  TASK_BANNER="3 hard ARC-AGI-1 tasks"
  CONCURRENCY=3
fi

CMD=(
  "${PYRUN[@]}" -m ksi.cli
  "${TASK_FLAGS[@]}"
  --provider-profile "$PROFILE"
  --generations 3
  --per-task-forum-rounds 1
  --cross-task-forum-rounds 1
  --max-concurrent-tasks "$CONCURRENCY"
  --experiment-name "$EXPERIMENT_NAME"
)

echo "==> Running quickstart demo ($TASK_BANNER)"
echo "    ${CMD[*]}"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[DRY RUN] command not executed"
  exit 0
fi

"${CMD[@]}"

echo
echo "==> Done. Inspect results:"
echo "    runtime_state/knowledge/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}_knowledge.sqlite"
echo "    /tmp/ksi-experiments/${EXPERIMENT_NAME}.log"
