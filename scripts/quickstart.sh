#!/usr/bin/env bash
# One command, fresh clone -> solved task on the bundled custom-tasks demo.
#
# No external dataset download and no prior setup_all.sh run needed. Given
# Docker and Node 20+ installed (and either uv or a local `pip install -e .`),
# this self-bootstraps everything it needs:
#   - a provider profile, synthesized from an ambient ANTHROPIC_API_KEY /
#     OPENAI_API_KEY if you don't already have one (like GEPA reads the key
#     from the environment);
#   - the kcsi-agent:bench Docker image (built on first run if missing);
#   - the host runtime_runner Node dependencies.
# Then it runs a single minimal generation so the demo finishes in a few minutes.
#
# Usage:
#   ANTHROPIC_API_KEY=sk-ant-... bash scripts/quickstart.sh
#   OPENAI_API_KEY=sk-...        bash scripts/quickstart.sh
#   PROFILE=configs/kcsi/.env.openai bash scripts/quickstart.sh
#
# Env knobs:
#   SKIP_BOOTSTRAP=1   don't build the image / install deps / synthesize a profile
#   SKIP_DOCTOR=1      skip the readiness check
#   DRY_RUN=true       print the run command (skips the image build / npm install,
#                      still synthesizes a profile), don't execute

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KCSI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KCSI_ROOT"

PROFILE="${PROFILE:-configs/kcsi/.env.haiku}"
TASKS_PATH="examples/custom_tasks/tasks.jsonl"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-quickstart_demo}"
AGENT_IMAGE="kcsi-agent:bench"

# Prefer uv, but fall back to a plain interpreter so a `pip install`d package
# (no uv) still works. uv is a convenience here, not a hard requirement.
if command -v uv >/dev/null 2>&1; then
  PYRUN=(uv run python)
  DOCTOR=(uv run kcsi-doctor)
elif command -v python >/dev/null 2>&1; then
  PYRUN=(python)
  DOCTOR=(python -m kcsi.doctor)
elif command -v python3 >/dev/null 2>&1; then
  PYRUN=(python3)
  DOCTOR=(python3 -m kcsi.doctor)
else
  echo "ERROR: no uv or python found on PATH." >&2
  exit 1
fi

# --- Bootstrap: make the common "fresh clone" failures self-healing ---------
synthesize_profile() {
  # Write a minimal provider profile from an ambient key if none usable exists.
  [[ -f "$PROFILE" ]] && return 0
  local out="${QUICKSTART_PROFILE_OUT:-configs/kcsi/.env.quickstart}"
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
      echo "ERROR: Node.js/npm not found — install Node 20+ and re-run." >&2
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
  echo "==> Checking setup (kcsi-doctor)"
  "${DOCTOR[@]}"
  echo
fi

if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: no provider profile and no API key in the environment." >&2
  echo "       Set ANTHROPIC_API_KEY (or OPENAI_API_KEY), or run scripts/setup_all.sh," >&2
  echo "       or pass PROFILE=<path>." >&2
  exit 1
fi

# Minimal canonical run: only the required flags plus one fast generation with
# the discussion/distillation phases off, so the demo finishes quickly.
CMD=(
  "${PYRUN[@]}" -m kcsi.cli
  --task-source custom
  --tasks-path "$TASKS_PATH"
  --evaluator command
  --provider-profile "$PROFILE"
  --generations 1
  --per-task-forum-rounds 0
  --cross-task-forum-rounds 0
  --max-concurrent-tasks 3
  --experiment-name "$EXPERIMENT_NAME"
)

echo "==> Running quickstart demo (3 custom Python tasks: fizzbuzz, reverse-words, anagram-groups)"
echo "    ${CMD[*]}"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[DRY RUN] command not executed"
  exit 0
fi

"${CMD[@]}"

echo
echo "==> Done. Inspect results:"
echo "    runtime_state/knowledge/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}_knowledge.sqlite"
echo "    /tmp/kcsi-experiments/${EXPERIMENT_NAME}.log"
