#!/usr/bin/env bash
# setup_all.sh — One-command setup for Knowledge-centric Self-Improvement
#
# Downloads datasets, installs dependencies, and runs smoke tests for
# all supported benchmarks.
#
# Usage:
#   bash scripts/setup_all.sh          # full setup + smoke tests
#   bash scripts/setup_all.sh --no-test # skip smoke tests
#
# Environment overrides:
#   SKIP_SWEBENCH_PRO_EVALUATOR=1
#                               skip cloning/verifying the external
#                               SWE-bench Pro evaluator checkout
#   SKIP_TERMINAL_BENCH_2=1     skip the Terminal-Bench 2 corpus submodule init
#   SKIP_CONTAINER_REBUILD=1    disable the .githooks auto-rebuild path (passed
#                               through to the post-merge / post-checkout hooks)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export NPM_CONFIG_CACHE="${NPM_CONFIG_CACHE:-$REPO_ROOT/.npm-cache}"
mkdir -p "$NPM_CONFIG_CACHE"

SKIP_TESTS=false
if [[ "${1:-}" == "--no-test" ]]; then
    SKIP_TESTS=true
fi

section() { echo -e "\n==> $1"; }
ok()      { echo "    OK: $1"; }
skip()    { echo "    SKIP: $1"; }

ensure_default_env_value() {
    local file="$1" key="$2" value="$3" stale_regex="${4:-^$}"
    local current=""
    if [[ -f "$file" ]] && grep -qE "^${key}=" "$file"; then
        current="$(grep -m1 -E "^${key}=" "$file" | cut -d= -f2-)"
        if [[ -z "$current" || "$current" =~ $stale_regex ]]; then
            local tmp
            tmp="$(mktemp "${file}.XXXXXX")"
            awk -v key="$key" -v value="$value" '
                BEGIN { prefix = key "=" }
                index($0, prefix) == 1 { $0 = prefix value }
                { print }
            ' "$file" > "$tmp"
            mv "$tmp" "$file"
            ok "Updated $key in ${file#$REPO_ROOT/}"
        else
            ok "$key already set in ${file#$REPO_ROOT/}"
        fi
    else
        if [[ -f "$file" && -s "$file" && -n "$(tail -c 1 "$file")" ]]; then
            printf '\n' >> "$file"
        fi
        printf '%s=%s\n' "$key" "$value" >> "$file"
        ok "Added $key to ${file#$REPO_ROOT/}"
    fi
}

env_file_has_secret() {
    local file="$1" key="$2"
    local value=""
    if [[ ! -f "$file" ]]; then
        return 1
    fi
    value="$(grep -m1 -E "^${key}=" "$file" | cut -d= -f2- || true)"
    [[ -n "$value" && "$value" != "<your-api-key>" && "$value" != "sk-ant-placeholder" ]]
}

# ── 0a. Git hooks (idempotent; safe to re-run) ──────────────────────────
#
# Enables the repo-local .githooks/ directory so post-merge / post-checkout
# can auto-rebuild the ksi-agent:bench container when TypeScript source
# drifts from the baked image. Contributors can disable with
# SKIP_CONTAINER_REBUILD=1 or by unsetting core.hooksPath.

section "Configuring git hooks"

if [[ -d "$(git rev-parse --show-toplevel 2>/dev/null)/.githooks" ]]; then
    current_hooks_path="$(git config --get core.hooksPath || true)"
    if [[ "$current_hooks_path" != ".githooks" ]]; then
        git config core.hooksPath .githooks
        ok "set core.hooksPath=.githooks"
    else
        ok "core.hooksPath already .githooks"
    fi
else
    skip ".githooks directory not present"
fi

# ── 0. Node.js (required by the KSI runtime-runner) ──────────────────

section "Checking Node.js"

NODE_VERSION_FILE="$REPO_ROOT/.nvmrc"
NODE_VERSION_PIN="$(tr -d '[:space:]' < "$NODE_VERSION_FILE")"
NODE_MAX_MAJOR_EXCLUSIVE=23

node_version_supported() {
    local current="${1#v}"
    python3 - "$current" "$NODE_VERSION_PIN" "$NODE_MAX_MAJOR_EXCLUSIVE" <<'PY'
import sys


def parse(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


current = parse(sys.argv[1])
minimum = parse(sys.argv[2])
max_major_exclusive = int(sys.argv[3])
raise SystemExit(0 if current >= minimum and current[0] < max_major_exclusive else 1)
PY
}

install_node_with_nvm() {
    export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
        python3 -c "
import urllib.request, os, stat
url = 'https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh'
dest = '/tmp/nvm-install.sh'
urllib.request.urlretrieve(url, dest)
os.chmod(dest, os.stat(dest).st_mode | stat.S_IEXEC)
" && bash /tmp/nvm-install.sh 2>&1 | tail -3
    fi
    # shellcheck source=/dev/null
    source "$NVM_DIR/nvm.sh"
    nvm install "$NODE_VERSION_PIN"
    ok "Installed node $(node --version), npm $(npm --version)"
}

if command -v node &>/dev/null && command -v npm &>/dev/null; then
    NODE_CURRENT="$(node --version)"
    if node_version_supported "$NODE_CURRENT"; then
        ok "node $NODE_CURRENT, npm $(npm --version) (required: >=$NODE_VERSION_PIN <$NODE_MAX_MAJOR_EXCLUSIVE)"
    else
        echo "    Node.js $NODE_CURRENT unsupported (need >=$NODE_VERSION_PIN <$NODE_MAX_MAJOR_EXCLUSIVE) - installing via nvm..."
        install_node_with_nvm
    fi
else
    echo "    Node.js/npm not found - installing Node.js $NODE_VERSION_PIN via nvm..."
    install_node_with_nvm
fi

NPM_BIN="$(command -v npm)"
ok "Using npm binary: $NPM_BIN"

install_npm_project() {
    local project_dir="$1"
    local label="$2"

    if [[ -f "$project_dir/package-lock.json" || -f "$project_dir/npm-shrinkwrap.json" ]]; then
        "$NPM_BIN" --prefix "$project_dir" ci --silent && ok "$label npm ci"
    else
        "$NPM_BIN" --prefix "$project_dir" install --silent && ok "$label npm install"
    fi
}

# ── 1. Runtime-runner setup ──────────────────────────────────────────────
#
# Install runtime_runner's node_modules up front. Without this, the first
# `swarm` invocation triggers `npx --yes --prefix runtime_runner tsx ...`
# which lazily populates node_modules/ concurrently with gen-1 task dispatch
# (e.g. 50 parallel containers on ARC). tsx resolves as soon as its own
# binary is on disk, but peer dependencies like `pino` may still be
# extracting — containers then crash with
# `ERR_MODULE_NOT_FOUND: Cannot find package 'pino'`.

section "Setting up runtime_runner (host-side launcher)"

if [[ -f "$REPO_ROOT/runtime_runner/package.json" ]]; then
    install_npm_project "$REPO_ROOT/runtime_runner" "runtime_runner"
else
    skip "runtime_runner/package.json not found"
fi

# ── 2c. Frontend setup ─────────────────────────────────────────────────────

section "Setting up frontend"

if [[ -f "$REPO_ROOT/frontend/package.json" ]]; then
    install_npm_project "$REPO_ROOT/frontend" "frontend"
else
    skip "frontend/package.json not found"
fi

# ── 2b. KSI agent Docker images ───────────────────────────────────────

section "Building KSI agent Docker images"

if command -v docker &>/dev/null; then
    # Build :bench first (lighter base — used by benchmark experiments)
    if docker image inspect ksi-agent:bench &>/dev/null; then
        ok "ksi-agent:bench already exists"
    else
        bash container/build.sh --bench && ok "ksi-agent:bench built"
    fi

    # Build :latest (full image — used by interactive tasks)
    if docker image inspect ksi-agent:latest &>/dev/null; then
        ok "ksi-agent:latest already exists"
    else
        bash container/build.sh && ok "ksi-agent:latest built"
    fi
else
    echo "    WARN: Docker not available — skipping image builds"
    echo "    Run 'bash container/build.sh --bench' and 'bash container/build.sh' when Docker is available"
fi

# ── 3. Python dependencies ────────────────────────────────────────────────

section "Installing Python dependencies (uv sync)"

# uv rejects `--locked` outright (exit 2) when UV_FROZEN/UV_LOCKED also pin
# resolution from the environment, which under `set -e` aborts setup before any
# benchmark prep runs. Clear those for this one call rather than dropping
# `--locked`: the flag verifies the lockfile is current instead of merely assuming
# it, so it is the stronger guarantee and the one tests pin.
env -u UV_FROZEN -u UV_LOCKED uv sync --locked --extra memory --extra swebench-pro && ok "uv sync complete"

# ── 3a. SWE-bench Pro dataset export ────────────────────────────────────
# The raw test split is a third-party dataset (ScaleAI SWE-bench Pro) that
# embeds upstream repo code, so it is gitignored rather than committed (#874).
# Export it from Hugging Face here; the evaluator setup below needs it present.

section "Preparing SWE-bench Pro dataset"

SWEBENCH_PRO_JSONL="$REPO_ROOT/benchmarks/swebench_pro/dataset/test.jsonl"
SWEBENCH_PRO_DATASET_REVISION="${SWEBENCH_PRO_DATASET_REVISION:-7ab5114912baf22bb098818e604c02fe7ad2c11f}"
if [[ "${SKIP_SWEBENCH_PRO_EVALUATOR:-0}" == "1" ]]; then
    skip "SKIP_SWEBENCH_PRO_EVALUATOR=1"
elif [[ -s "$SWEBENCH_PRO_JSONL" ]]; then
    ok "SWE-bench Pro dataset already exists at $SWEBENCH_PRO_JSONL"
else
    echo "    Exporting SWE-bench Pro test split from Hugging Face (ScaleAI/SWE-bench_Pro@$SWEBENCH_PRO_DATASET_REVISION)..."
    if uv run python benchmarks/scripts/dataprep/export_swebench_pro_dataset.py \
        --split test --format jsonl --revision "$SWEBENCH_PRO_DATASET_REVISION" \
        --output "$SWEBENCH_PRO_JSONL" 2>&1 | tail -3; then
        ok "SWE-bench Pro dataset exported: $SWEBENCH_PRO_JSONL"
    else
        echo "    WARN: Failed to export SWE-bench Pro dataset (needs HF access) — run manually:"
        echo "    uv run python benchmarks/scripts/dataprep/export_swebench_pro_dataset.py --split test --format jsonl --revision $SWEBENCH_PRO_DATASET_REVISION --output benchmarks/swebench_pro/dataset/test.jsonl"
    fi
fi

# ── 3b. External SWE-bench Pro evaluator ────────────────────────────────

section "Preparing SWE-bench Pro evaluator"

if [[ "${SKIP_SWEBENCH_PRO_EVALUATOR:-0}" == "1" ]]; then
    skip "SKIP_SWEBENCH_PRO_EVALUATOR=1"
elif command -v git &>/dev/null; then
    if uv run python benchmarks/scripts/dataprep/setup_swebench_pro_evaluator.py 2>&1; then
        ok "SWE-bench Pro evaluator ready"
    else
        echo "    WARN: SWE-bench Pro evaluator setup failed — continuing without SWE-bench Pro eval"
        echo "    Re-run: uv run python benchmarks/scripts/dataprep/setup_swebench_pro_evaluator.py"
    fi
else
    echo "    WARN: git not available — skipping SWE-bench Pro evaluator setup"
fi

# ── 3c. Provider profile ────────────────────────────────────────────────

section "Checking provider profiles"

PROVIDERS_DIR="$REPO_ROOT/configs/ksi"
mkdir -p "$PROVIDERS_DIR"

# Adopt profiles left behind at the pre-reorg location. Mirrors _PROFILE_MOVES in
# src/ksi/providers.py. Copies rather than moves so an older checkout still resolves
# the old path. Without this, a user whose real-key profile predates the reorg gets
# a freshly generated key-less profile here and the run fails at provider load.
LEGACY_PROVIDERS_DIR="$REPO_ROOT/configs/providers"
for legacy_profile in "$LEGACY_PROVIDERS_DIR"/.env.*; do
    [[ -f "$legacy_profile" ]] || continue
    case "$legacy_profile" in *.template | *.example) continue ;; esac
    legacy_name="$(basename "$legacy_profile")"
    if [[ -f "$PROVIDERS_DIR/$legacy_name" ]]; then
        ok "$legacy_name already present in configs/ksi"
    else
        cp "$legacy_profile" "$PROVIDERS_DIR/$legacy_name"
        ok "Adopted $legacy_name from configs/providers (pre-reorg location)"
    fi
done

# Auto-detect auth tokens once, reuse for all profiles
OAUTH_TOKEN="${CLAUDE_CODE_OAUTH_TOKEN:-}"
API_KEY="${ANTHROPIC_API_KEY:-}"
if [[ -z "$OAUTH_TOKEN" && -f "$REPO_ROOT/.env" ]]; then
    OAUTH_TOKEN="$(grep -oP '^\s*CLAUDE_CODE_OAUTH_TOKEN=\K.*' "$REPO_ROOT/.env" 2>/dev/null || true)"
fi
if [[ -z "$API_KEY" && -f "$REPO_ROOT/.env" ]]; then
    API_KEY="$(grep -oP '^\s*ANTHROPIC_API_KEY=\K.*' "$REPO_ROOT/.env" 2>/dev/null || true)"
fi

# Interactive prompt for missing tokens (only when running in a terminal)
if [[ -t 0 && -z "$API_KEY" ]]; then
    echo ""
    echo "    Anthropic API key not found in environment or .env file."
    read -rp "    Paste your ANTHROPIC_API_KEY (or press Enter to skip): " API_KEY
    if [[ -n "$API_KEY" ]]; then
        ok "ANTHROPIC_API_KEY set"
    else
        echo "    Skipped — you can add it later to configs/ksi/.env.haiku"
    fi
fi

# Generate .env.haiku
PROFILE_HAIKU="$PROVIDERS_DIR/.env.haiku"
if [[ -f "$PROFILE_HAIKU" ]]; then
    ok "Provider profile exists at $PROFILE_HAIKU"
else
    cat > "$PROFILE_HAIKU" <<ENVEOF
MODEL_PROVIDER=anthropic
MODEL=claude-haiku-4-5-20251001
MODEL_AUTH_MODE=api
ANTHROPIC_API_KEY=${API_KEY:-}
ENVEOF
    ok "Created $PROFILE_HAIKU"
fi

# Generate .env.sonnet
PROFILE_SONNET="$PROVIDERS_DIR/.env.sonnet"
if [[ -f "$PROFILE_SONNET" ]]; then
    ok "Provider profile exists at $PROFILE_SONNET"
else
    cat > "$PROFILE_SONNET" <<ENVEOF
MODEL_PROVIDER=anthropic
MODEL=claude-sonnet-4-6
MODEL_AUTH_MODE=api
ANTHROPIC_API_KEY=${API_KEY:-}
ENVEOF
    ok "Created $PROFILE_SONNET"
fi

# Generate .env.openai (default OpenAI model: gpt-5.4-mini, medium reasoning)
OPENAI_KEY="${OPENAI_API_KEY:-}"
if [[ -z "$OPENAI_KEY" && -f "$REPO_ROOT/.env" ]]; then
    OPENAI_KEY="$(grep -oP '^\s*OPENAI_API_KEY=\K.*' "$REPO_ROOT/.env" 2>/dev/null || true)"
fi
PROFILE_OPENAI="$PROVIDERS_DIR/.env.openai"
if [[ -f "$PROFILE_OPENAI" ]]; then
    ok "Provider profile exists at $PROFILE_OPENAI"
    ensure_default_env_value "$PROFILE_OPENAI" MODEL "gpt-5.4-mini" '^(gpt-4o-mini|gpt-4o|o3-mini|o3-mini-2025-01-31)$'
    ensure_default_env_value "$PROFILE_OPENAI" REASONING_EFFORT "medium" '^$'
else
    cat > "$PROFILE_OPENAI" <<ENVEOF
MODEL_PROVIDER=openai
MODEL=gpt-5.4-mini
MODEL_AUTH_MODE=api
OPENAI_API_KEY=${OPENAI_KEY:-}
REASONING_EFFORT=medium
ENVEOF
    ok "Created $PROFILE_OPENAI"
fi

if env_file_has_secret "$PROFILE_HAIKU" ANTHROPIC_API_KEY \
    || env_file_has_secret "$PROFILE_SONNET" ANTHROPIC_API_KEY \
    || env_file_has_secret "$PROFILE_OPENAI" OPENAI_API_KEY; then
    ok "Provider profiles contain at least one non-placeholder API key"
else
    echo "    WARN: Provider profiles created with empty API keys — edit before running"
fi

# ── 5. Datasets ──────────────────────────────────────────────────────────

# ── 5a. ARC-AGI 1.0 data ────────────────────────────────────────────────
#
# Canonical location: benchmarks/arc1/source (matches arc_defaults.json and
# every task_map's source_file field). Pinned commits keep task_maps/*.json
# reproducible across fresh clones.

section "Downloading ARC-AGI 1.0 data"

ARC1_DIR="$REPO_ROOT/benchmarks/arc1/source"
ARC1_COMMIT="399030444e0ab0cc8b4e199870fb20b863846f34"
if [[ -d "$ARC1_DIR/data/training" ]]; then
    ARC1_COUNT=$(find "$ARC1_DIR/data/training" -name '*.json' | wc -l)
    ok "ARC-AGI 1.0 already exists at $ARC1_DIR ($ARC1_COUNT training tasks)"
else
    mkdir -p "$(dirname "$ARC1_DIR")"
    echo "    Cloning fchollet/ARC-AGI and pinning to $ARC1_COMMIT..."
    git clone https://github.com/fchollet/ARC-AGI.git "$ARC1_DIR" 2>&1 | tail -2
    git -C "$ARC1_DIR" checkout --quiet "$ARC1_COMMIT"
    ARC1_COUNT=$(find "$ARC1_DIR/data/training" -name '*.json' | wc -l)
    ok "ARC-AGI 1.0 cloned to $ARC1_DIR ($ARC1_COUNT training tasks)"
fi

# ── 5c. ARC-AGI 2.0 data ────────────────────────────────────────────────

section "Downloading ARC-AGI 2.0 data"

ARC2_DIR="$REPO_ROOT/benchmarks/arc2/source"
ARC2_COMMIT="f3283f727488ad98fe575ea6a5ac981e4a188e49"
if [[ -d "$ARC2_DIR/data/training" ]]; then
    ARC2_COUNT=$(find "$ARC2_DIR/data/training" -name '*.json' | wc -l)
    ok "ARC-AGI 2.0 already exists at $ARC2_DIR ($ARC2_COUNT training tasks)"
else
    mkdir -p "$(dirname "$ARC2_DIR")"
    echo "    Cloning arcprize/ARC-AGI-2 and pinning to $ARC2_COMMIT..."
    git clone https://github.com/arcprize/ARC-AGI-2.git "$ARC2_DIR" 2>&1 | tail -2
    git -C "$ARC2_DIR" checkout --quiet "$ARC2_COMMIT"
    ARC2_COUNT=$(find "$ARC2_DIR/data/training" -name '*.json' | wc -l)
    ok "ARC-AGI 2.0 cloned to $ARC2_DIR ($ARC2_COUNT training tasks)"
fi

# ── 5e. ARC workspace payload manifests ─────────────────────────────────
#
# ARC native/UI workspace mode (benchmarks/arc/native_mode/,
# benchmarks/arc/workspace_ui/) looks up payload manifests via
# `ARC_DEFAULT_MANIFESTS`, which defaults to
# `benchmarks/arc/workspace_payloads/<bench>/<selection>/manifest.json` as
# declared in `configs/benchmarks/arc_defaults.json`. These are generated
# lazily from each benchmark's task map + source data; committing the redacted
# payload files would balloon the repo, so we generate them here after the
# upstream ARC data is on disk.

section "Generating ARC workspace payload manifests"

generate_arc_manifests() {
    local bench="$1" task_map="$2"
    local manifest_dir="$REPO_ROOT/benchmarks/arc/workspace_payloads/${bench}/$(basename "${task_map%.json}")"
    if [[ -f "$manifest_dir/manifest.json" ]]; then
        ok "$bench payload manifest already at $manifest_dir/manifest.json"
        return 0
    fi
    if [[ ! -f "$REPO_ROOT/$task_map" ]]; then
        skip "$bench task map not found at $task_map"
        return 0
    fi
    if uv run python benchmarks/scripts/arc_prep/prepare_arc_workspace_payloads.py \
        --task-map "$REPO_ROOT/$task_map" 2>&1 | tail -1; then
        ok "$bench payloads generated at $manifest_dir"
    else
        echo "    WARN: failed to generate $bench payloads — ARC native/UI workspace mode will be unavailable"
    fi
}

generate_arc_manifests arc1 benchmarks/arc1/task_maps/arc1_train_50_seed0.json
generate_arc_manifests arc2 benchmarks/arc2/task_maps/arc2_train_50_seed0.json

# ── 5f. Polyglot benchmark data ──────────────────────────────────────────

section "Preparing Polyglot benchmark data"

POLYGLOT_JSON="$REPO_ROOT/data/polyglot_medium.json"
if [[ -f "$POLYGLOT_JSON" ]]; then
    POLYGLOT_COUNT=$(uv run python -c "import json; print(len(json.loads(open('$POLYGLOT_JSON').read())))" 2>/dev/null || echo "?")
    ok "Polyglot dataset already exists at $POLYGLOT_JSON ($POLYGLOT_COUNT tasks)"
else
    echo "    Preparing polyglot dataset (clones Aider-AI/polyglot-benchmark)..."
    if uv run python benchmarks/scripts/dataprep/prepare_polyglot_dataset.py --output "$POLYGLOT_JSON" 2>&1 | tail -3; then
        POLYGLOT_COUNT=$(uv run python -c "import json; print(len(json.loads(open('$POLYGLOT_JSON').read())))" 2>/dev/null || echo "?")
        ok "Polyglot dataset prepared: $POLYGLOT_JSON ($POLYGLOT_COUNT tasks)"
    else
        echo "    WARN: Failed to prepare polyglot dataset — run manually:"
        echo "    uv run python benchmarks/scripts/dataprep/prepare_polyglot_dataset.py --output data/polyglot_medium.json"
    fi
fi

# ── 5g. Polyglot Docker image ───────────────────────────────────────────
#
# `ksi-polyglot-eval:latest` is used by the ksi-side polyglot evaluator
# (`ksi.benchmarks.polyglot_docker`). DGM and HyperAgents polyglot use their own
# evaluator images (`pb.{base,env,eval}.*`) and do NOT need this build.
#
# The build's apt-get RUN step (openjdk-21-jdk + libboost-all-dev + python3.11
# from deadsnakes) is heavy enough to deadlock or OOM under concurrent Docker
# activity (observed: 11+ min stalls, OOM-kill exit 137 during the
# 2026-05-01 audit-sweep when other polyglot harnesses were running). Set
# `KSI_SKIP_POLYGLOT_EVAL_IMAGE=1` to skip when only running DGM/HA polyglot
# baselines (which do not need this image).
section "Building Polyglot evaluation Docker image"

if [[ "${KSI_SKIP_POLYGLOT_EVAL_IMAGE:-0}" == "1" ]]; then
    skip "KSI_SKIP_POLYGLOT_EVAL_IMAGE=1 — skipping ksi-polyglot-eval build"
elif command -v docker &>/dev/null; then
    echo "    Ensuring ksi-polyglot-eval:latest matches the maintained recipe..."
    if uv run python -c "from ksi.benchmarks.polyglot_docker import build_image; build_image()" 2>&1 | tail -3; then
        ok "ksi-polyglot-eval:latest ready"
    else
        echo "    WARN: Failed to build polyglot eval image — run manually:"
        echo "    uv run python -c \"from ksi.benchmarks.polyglot_docker import build_image; build_image()\""
    fi
else
    skip "Docker not available — polyglot evaluation image not built"
fi

# ── 5h. Terminal-Bench 2 corpus ─────────────────────────────────────────
#
# TB2 ships as a git submodule rather than a download (see
# benchmarks/docs/BENCHMARK_PREPARE.md) — a benchmark task corpus that the
# terminal_bench_2 task source and evaluator read directly.

section "Initializing Terminal-Bench 2 corpus"

TB2_DIR="$REPO_ROOT/benchmarks/terminal_bench_2/source"
if [[ "${SKIP_TERMINAL_BENCH_2:-0}" == "1" ]]; then
    skip "SKIP_TERMINAL_BENCH_2=1"
elif [[ -n "$(ls -A "$TB2_DIR" 2>/dev/null)" ]]; then
    ok "Terminal-Bench 2 corpus already checked out at $TB2_DIR"
elif [[ -f "$REPO_ROOT/.gitmodules" ]]; then
    # Soft-fail: a network outage should not abort the rest of setup, it just
    # leaves the terminal_bench_2 arm unavailable.
    if git submodule update --init --recursive benchmarks/terminal_bench_2/source 2>&1; then
        ok "Terminal-Bench 2 corpus initialized"
    else
        echo "    WARN: Terminal-Bench 2 submodule init failed — the terminal_bench_2 arm will be unavailable"
        echo "    Re-run: git submodule update --init --recursive benchmarks/terminal_bench_2/source"
    fi
else
    skip ".gitmodules not present"
fi

# ── 6. Preflight checks ──────────────────────────────────────────────────

section "Preflight checks"

# Source .env if HF_TOKEN not already in environment
if [[ -z "${HF_TOKEN:-}" && -f "$REPO_ROOT/.env" ]]; then
    HF_TOKEN="$(grep -oP '^\s*HF_TOKEN=\K.*' "$REPO_ROOT/.env" 2>/dev/null || true)"
    [[ -n "$HF_TOKEN" ]] && export HF_TOKEN
fi

# HuggingFace token (expected for the default EmbeddingGemma model)
if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN is set"
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    ok "HuggingFace token found via huggingface-cli login"
else
    EMBEDDING_MODEL_FOR_SETUP="${KSI_EMBEDDING_MODEL:-google/embeddinggemma-300m}"
    if [[ -t 0 && "${EMBEDDING_MODEL_FOR_SETUP}" == google/* ]]; then
        echo "    HF_TOKEN not set — only needed for semantic vector search (--require-vector); default retrieval is FTS5 and needs no token. Embedding model: ${EMBEDDING_MODEL_FOR_SETUP}."
        read -rp "    Paste your HF_TOKEN (or press Enter to skip): " HF_TOKEN_INPUT
        if [[ -n "$HF_TOKEN_INPUT" ]]; then
            export HF_TOKEN="$HF_TOKEN_INPUT"
            # Persist to .env so subsequent runs (including this script re-run) pick it up.
            ENV_FILE="$REPO_ROOT/.env"
            touch "$ENV_FILE"
            if grep -qE '^\s*HF_TOKEN=' "$ENV_FILE"; then
                # Replace existing line in-place; POSIX-safe sed avoids GNU -i.bak quirks.
                tmp="$(mktemp)"
                awk -v val="$HF_TOKEN_INPUT" '
                    /^[[:space:]]*HF_TOKEN=/ { print "HF_TOKEN=" val; next }
                    { print }
                ' "$ENV_FILE" > "$tmp" && mv "$tmp" "$ENV_FILE"
            else
                printf 'HF_TOKEN=%s\n' "$HF_TOKEN_INPUT" >> "$ENV_FILE"
            fi
            ok "HF_TOKEN set and persisted to .env"
        else
            echo "    Skipped — agent retrieval (MCP query) will use the FTS-only (lexical FTS5) fallback"
        fi
    else
        echo "    HF_TOKEN not set — OK only if ${EMBEDDING_MODEL_FOR_SETUP} is public or already cached."
        echo "    Default embedding model is google/embeddinggemma-300m; set HF_TOKEN or run huggingface-cli login."
    fi
fi

# Templates required by the KSI runtime
if [[ -f "$REPO_ROOT/templates/INSTRUCTION.md" ]]; then
    ok "templates/INSTRUCTION.md exists"
else
    echo "    WARN: templates/INSTRUCTION.md missing — task execution will fail"
fi

# File descriptor limit (swebench_pro can exhaust low limits)
FD_LIMIT=$(ulimit -n 2>/dev/null || echo "unknown")
if [[ "$FD_LIMIT" != "unknown" && "$FD_LIMIT" -lt 8192 ]]; then
    echo "    WARN: ulimit -n is $FD_LIMIT (recommend >= 65536 for SWE-bench harness)"
    echo "    Run: ulimit -n 65536"
else
    ok "File descriptor limit: $FD_LIMIT"
fi

# Docker daemon
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    ok "Docker daemon is running"
else
    echo "    WARN: Docker not available or not running — container-based tasks will fail"
fi

# ── 7. Smoke tests ────────────────────────────────────────────────────────

if $SKIP_TESTS; then
    section "Skipping smoke tests (--no-test)"
else
    section "Running smoke tests (pytest, no containers)"

    echo -n "    Import check ... "
    if uv run python -c "from ksi.cli import build_parser; build_parser()" 2>/dev/null; then
        echo "PASS"
    else
        echo "FAIL — ksi package not importable"
        exit 1
    fi

    echo -n "    Quick pytest ... "
    if uv run pytest tests/ -x -q --timeout=60 -k "not parallel and not slow" --ignore=tests/orchestrator/test_parallel_execution.py 2>&1 | tail -1; then
        true
    else
        echo "    Some tests failed — run 'uv run pytest tests/ -x' for details"
        exit 1
    fi
fi

section "Setup complete"

echo ""
REMAINING=()
if ! env_file_has_secret "$PROFILE_HAIKU" ANTHROPIC_API_KEY \
    && ! env_file_has_secret "$PROFILE_SONNET" ANTHROPIC_API_KEY; then
    REMAINING+=("  - Add your Anthropic API key to the provider profiles:")
    REMAINING+=("      vim configs/ksi/.env.haiku       # ANTHROPIC_API_KEY=sk-ant-...")
    REMAINING+=("      vim configs/ksi/.env.sonnet      # ANTHROPIC_API_KEY=sk-ant-...")
fi
if ! env_file_has_secret "$PROFILE_OPENAI" OPENAI_API_KEY; then
    REMAINING+=("  - Add your OpenAI API key for gpt-5.4-mini experiments:")
    REMAINING+=("      vim configs/ksi/.env.openai      # OPENAI_API_KEY=sk-...")
fi
EMBEDDING_MODEL_FOR_SETUP="${KSI_EMBEDDING_MODEL:-google/embeddinggemma-300m}"
if [[ "${EMBEDDING_MODEL_FOR_SETUP}" == google/* && -z "${HF_TOKEN:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
    REMAINING+=("  - Set HuggingFace token for default memory embedding model (${EMBEDDING_MODEL_FOR_SETUP}):")
    REMAINING+=("      export HF_TOKEN=hf_...   # or: huggingface-cli login")
fi

if [[ ${#REMAINING[@]} -gt 0 ]]; then
    echo "  REMAINING SETUP:"
    echo ""
    for line in "${REMAINING[@]}"; do
        echo "$line"
    done
    echo ""
fi

echo "  RUN AN EXPERIMENT:"
echo "    bash benchmarks/run_arc.sh 2 haiku"
echo ""
