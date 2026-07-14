#!/bin/bash
# Active build entrypoint for KSI container images.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKERFILES_DIR="$REPO_ROOT/container"

PRIMARY_IMAGE_NAME="ksi-agent"
DOCKERFILE="$DOCKERFILES_DIR/Dockerfile"
TAG="latest"

# Unknown args starting with `-` are forwarded to `docker build`
# (e.g. --no-cache, --pull, --progress=plain). Previously every non-`--bench`
# arg was treated as the tag, so `bash container/build.sh --no-cache` produced
# `ksi-agent:--no-cache` and docker rejected the reference.
EXTRA_BUILD_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --bench) DOCKERFILE="$DOCKERFILES_DIR/Dockerfile.bench"; TAG="bench" ;;
        -*) EXTRA_BUILD_ARGS+=("$arg") ;;
        *) TAG="$arg" ;;
    esac
done

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"

echo "Building KSI agent container image..."
echo "Primary image: ${PRIMARY_IMAGE_NAME}:${TAG}"
echo "Dockerfile: ${DOCKERFILE#$REPO_ROOT/}"
if ((${#EXTRA_BUILD_ARGS[@]} > 0)); then
    echo "Extra docker args: ${EXTRA_BUILD_ARGS[*]}"
fi

"${CONTAINER_RUNTIME}" build "${EXTRA_BUILD_ARGS[@]}" -f "$DOCKERFILE" -t "${PRIMARY_IMAGE_NAME}:${TAG}" "$REPO_ROOT"

echo ""
echo "Build complete!"
echo "Images:"
echo "  ${PRIMARY_IMAGE_NAME}:${TAG}"
echo ""
echo "Test with:"
echo "  echo '{\"prompt\":\"What is 2+2?\",\"workspaceKey\":\"task__test\"}' | ${CONTAINER_RUNTIME} run -i ${PRIMARY_IMAGE_NAME}:${TAG}"
