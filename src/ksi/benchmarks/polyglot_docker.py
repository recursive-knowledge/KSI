"""Docker image builder for polyglot benchmark evaluation.

Builds a multi-language Docker image with toolchains for:
Python 3.11, Rust 1.85.1, Go 1.21.5, Node.js 20, Java 21,
C++/CMake, and Miniconda. The base/toolchain recipe is the HyperAgents
Polyglot ``pb.base`` recipe with a small KSI harness compatibility layer on
top. The Rust toolchain is pinned (a deliberate deviation from the pristine
pb.base recipe, which tracks rustup's moving "latest stable") so the grader
matches the bench agent image (container/Dockerfile.bench) it grades against.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import tempfile

from dotenv import load_dotenv

# Load .env so the polyglot eval Docker image name can be overridden without
# editing source. load_dotenv() is idempotent.
load_dotenv()

log = logging.getLogger(__name__)

IMAGE_NAME = os.environ.get("POLYGLOT_DOCKER_IMAGE", "ksi-polyglot-eval:latest")
POLYGLOT_RECIPE_LABEL = "org.knowledgecentric.polyglot.recipe"
POLYGLOT_RECIPE_VERSION = "polyglot-hyperagents-base-20260422-rust-pin"
POLYGLOT_RECIPE_BASE_IMAGE_LABEL = "org.knowledgecentric.polyglot.recipe_base_image"
POLYGLOT_RECIPE_BASE_IMAGE = "buildpack-deps:jammy"
POLYGLOT_RECIPE_SOURCE_LABEL = "org.knowledgecentric.polyglot.recipe_source"
POLYGLOT_RECIPE_SOURCE = "baselines/hyperagents/domains/polyglot/dockerfiles.py:_DOCKERFILE_BASE"
POLYGLOT_RECIPE_TARGET_ARCH_LABEL = "org.knowledgecentric.polyglot.recipe_target_arch"

# Preserve the historical MockingJay labels for already-built images and
# downstream tooling that still keys off the older namespace.
POLYGLOT_RECIPE_LABEL_ALIAS = "org.mockingjay.polyglot.recipe"
POLYGLOT_RECIPE_BASE_IMAGE_LABEL_ALIAS = "org.mockingjay.polyglot.recipe_base_image"
POLYGLOT_RECIPE_SOURCE_LABEL_ALIAS = "org.mockingjay.polyglot.recipe_source"
POLYGLOT_RECIPE_TARGET_ARCH_LABEL_ALIAS = "org.mockingjay.polyglot.recipe_target_arch"


def _normalize_target_arch(arch: str) -> str:
    value = (arch or "").strip().lower()
    if value in {"amd64", "x86_64"}:
        return "x86_64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    raise ValueError(f"Unsupported Polyglot Docker target architecture: {arch!r}")


def _host_target_arch() -> str:
    return _normalize_target_arch(platform.machine())


def _target_arch_from_platform(target_platform: str) -> str:
    parts = [part for part in target_platform.strip().lower().split("/") if part]
    if len(parts) < 2 or parts[0] != "linux":
        raise ValueError(f"Unsupported Polyglot Docker target platform: {target_platform!r}")
    return _normalize_target_arch(parts[1])


def _resolve_target_platform(platform_name: str | None = None) -> str | None:
    return (platform_name or os.environ.get("DOCKER_DEFAULT_PLATFORM") or "").strip() or None


def _resolve_target_arch(*, platform_name: str | None = None, arch: str | None = None) -> str:
    if arch:
        return _normalize_target_arch(arch)
    if platform_name:
        return _target_arch_from_platform(platform_name)
    return _host_target_arch()


def _conda_arch_for_target(target_arch: str) -> str:
    return "aarch64" if _normalize_target_arch(target_arch) == "arm64" else "x86_64"


_HYPERAGENTS_BASE_DOCKERFILE_TEMPLATE = r"""
FROM buildpack-deps:jammy

# Install base build/runtime deps. Python 3.11 comes from Miniconda below,
# which avoids a fragile Launchpad PPA dependency during Docker build.
RUN set -eux; \
    retry() {{ \
      n=0; \
      until "$@"; do \
        n=$((n + 1)); \
        if [ "$n" -ge 5 ]; then \
          return 1; \
        fi; \
        sleep $((5 * n)); \
      done; \
    }}; \
    export DEBIAN_FRONTEND=noninteractive; \
    retry apt-get update; \
    retry apt-get install -y \
      cmake \
      libboost-all-dev \
      python3-pip \
      ca-certificates-java \
      openjdk-21-jdk \
      libtbb-dev; \
    rm -rf /var/lib/apt/lists/*

# Install Go with architecture detection
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        GOARCH="amd64"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        GOARCH="arm64"; \
    else \
        false; \
    fi && \
    curl -L "https://golang.org/dl/go1.21.5.linux-$GOARCH.tar.gz" -o go.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && \
    rm go.tar.gz
ENV PATH="/usr/local/go/bin:${{PATH}}"

# Install Rust. Pin the default toolchain to match the bench agent image
# (container/Dockerfile.bench ARG RUST_TOOLCHAIN=1.85.1) so the grader's
# rustc/cargo reproduce what the agent built against instead of tracking
# rustup's moving "latest stable". Deliberate KSI deviation from the
# pristine HyperAgents pb.base recipe (which omits --default-toolchain). The
# rustup installer itself is still the upstream convenience script; only the
# compiling toolchain -- the reproducibility-relevant part -- is pinned.
ADD https://sh.rustup.rs /tmp/rustup.sh
RUN chmod +x /tmp/rustup.sh && /tmp/rustup.sh -y --default-toolchain 1.85.1 && rm /tmp/rustup.sh
ENV PATH="/root/.cargo/bin:${{PATH}}"

# Install Node.js and dependencies
RUN set -eux; \
    retry() {{ \
      n=0; \
      until "$@"; do \
        n=$((n + 1)); \
        if [ "$n" -ge 5 ]; then \
          return 1; \
        fi; \
        sleep $((5 * n)); \
      done; \
    }}; \
    retry bash -c "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -" && \
    retry apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /npm-install && \
    cd /npm-install && \
    npm init -y && \
    npm install \
    jest@29.7.0 \
    @babel/core@7.25.2 \
    @exercism/babel-preset-javascript@0.2.1 \
    @exercism/eslint-config-javascript@0.6.0 \
    @types/jest@29.5.12 \
    @types/node@20.12.12 \
    babel-jest@29.6.4 \
    core-js@3.37.1 \
    eslint@8.49.0

# Download and install conda
RUN wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-{conda_arch}.sh' -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda3
# Add conda to PATH
ENV PATH=/opt/miniconda3/bin:$PATH
# Add conda to shell startup scripts like .bashrc (DO NOT REMOVE THIS)
RUN conda init --all
RUN conda config --append channels conda-forge

RUN adduser --disabled-password --gecos 'dog' nonroot
"""


def hyperagents_base_dockerfile(*, arch: str | None = None, platform_name: str | None = None) -> str:
    target_arch = _resolve_target_arch(platform_name=platform_name, arch=arch)
    return _HYPERAGENTS_BASE_DOCKERFILE_TEMPLATE.format(conda_arch=_conda_arch_for_target(target_arch)).strip()


HYPERAGENTS_BASE_DOCKERFILE = hyperagents_base_dockerfile()


def ksi_compatibility_dockerfile(*, target_arch: str) -> str:
    return f"""
LABEL {POLYGLOT_RECIPE_LABEL}="{POLYGLOT_RECIPE_VERSION}" \\
      {POLYGLOT_RECIPE_LABEL_ALIAS}="{POLYGLOT_RECIPE_VERSION}" \\
      {POLYGLOT_RECIPE_BASE_IMAGE_LABEL}="{POLYGLOT_RECIPE_BASE_IMAGE}" \\
      {POLYGLOT_RECIPE_BASE_IMAGE_LABEL_ALIAS}="{POLYGLOT_RECIPE_BASE_IMAGE}" \\
      {POLYGLOT_RECIPE_SOURCE_LABEL}="{POLYGLOT_RECIPE_SOURCE}" \\
      {POLYGLOT_RECIPE_SOURCE_LABEL_ALIAS}="{POLYGLOT_RECIPE_SOURCE}" \\
      {POLYGLOT_RECIPE_TARGET_ARCH_LABEL}="{target_arch}" \\
      {POLYGLOT_RECIPE_TARGET_ARCH_LABEL_ALIAS}="{target_arch}"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PATH="/usr/local/bin:${{PATH}}"

# The block above is the HyperAgents pb.base recipe. KSI adds only the
# direct-harness tools it needs because it mounts generated task files into one
# shared eval image instead of building per-task pb.env/pb.eval images.
RUN python -m pip install --no-cache-dir --upgrade pip pytest
RUN apt-get update && apt-get install -y \\
    catch2 \\
    zip \\
    unzip \\
    && rm -rf /var/lib/apt/lists/*

# Jammy's catch2 package provides catch.hpp; newer packages provide
# catch_all.hpp. Keep the compatibility shim conditional so either layout works.
RUN if [ ! -f /usr/include/catch2/catch.hpp ] && [ -f /usr/include/catch2/catch_all.hpp ]; then \\
        echo '#include <catch2/catch_all.hpp>' > /usr/include/catch2/catch.hpp; \\
    fi

# Gradle 8.x (Ubuntu's gradle is too old for Exercism's build.gradle files)
RUN curl -fsSL "https://services.gradle.org/distributions/gradle-8.5-bin.zip" -o /tmp/gradle.zip \\
    && unzip -q /tmp/gradle.zip -d /opt \\
    && ln -s /opt/gradle-8.5/bin/gradle /usr/local/bin/gradle \\
    && rm /tmp/gradle.zip

# Working directory
WORKDIR /exercise
"""


def dockerfile_for_target(*, platform_name: str | None = None, arch: str | None = None) -> str:
    target_arch = _resolve_target_arch(platform_name=platform_name, arch=arch)
    base = hyperagents_base_dockerfile(arch=target_arch)
    compat = ksi_compatibility_dockerfile(target_arch=target_arch)
    return f"{base}\n\n{compat.lstrip()}"


DOCKERFILE = dockerfile_for_target()


def image_matches_recipe(
    image_name: str = IMAGE_NAME,
    *,
    platform_name: str | None = None,
    arch: str | None = None,
) -> bool:
    """Return whether *image_name* exists and was built from this recipe."""
    target_arch = _resolve_target_arch(
        platform_name=_resolve_target_platform(platform_name),
        arch=arch,
    )
    result = subprocess.run(
        ["docker", "image", "inspect", image_name, "--format", "{{json .Config.Labels}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    try:
        labels = json.loads(result.stdout.strip() or "null") or {}
    except json.JSONDecodeError:
        log.warning("Could not parse Docker labels for %s: %s", image_name, result.stdout)
        return False
    recipe = labels.get(POLYGLOT_RECIPE_LABEL) or labels.get(POLYGLOT_RECIPE_LABEL_ALIAS)
    target_arch_label = labels.get(POLYGLOT_RECIPE_TARGET_ARCH_LABEL) or labels.get(
        POLYGLOT_RECIPE_TARGET_ARCH_LABEL_ALIAS
    )
    return recipe == POLYGLOT_RECIPE_VERSION and target_arch_label == target_arch


def build_image(*, force: bool = False, platform_name: str | None = None, arch: str | None = None) -> str:
    """Build the polyglot evaluation Docker image if not already present.

    Returns the image name.
    """
    target_platform = _resolve_target_platform(platform_name)
    target_arch = _resolve_target_arch(platform_name=target_platform, arch=arch)
    dockerfile = dockerfile_for_target(arch=target_arch)
    if not force and image_matches_recipe(IMAGE_NAME, platform_name=target_platform, arch=target_arch):
        log.info("Polyglot eval image already matches recipe: %s", IMAGE_NAME)
        return IMAGE_NAME

    log.info("Building polyglot eval image: %s", IMAGE_NAME)
    # Use an empty temp dir as build context to avoid scanning unrelated files
    cmd = ["docker", "build", "--network", "host"]
    if target_platform:
        cmd.extend(["--platform", target_platform])
    cmd.extend(["-t", IMAGE_NAME, "-f", "-", "."])
    with tempfile.TemporaryDirectory(prefix="polyglot-build-") as build_ctx:
        proc = subprocess.run(
            cmd,
            input=dockerfile,
            capture_output=True,
            text=True,
            cwd=build_ctx,
        )
    if proc.returncode != 0:
        log.error("Docker build failed:\n%s", proc.stderr[-2000:])
        raise RuntimeError(f"Failed to build polyglot eval image: {proc.stderr[-500:]}")

    log.info("Built polyglot eval image: %s", IMAGE_NAME)
    return IMAGE_NAME
