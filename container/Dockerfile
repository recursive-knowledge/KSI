# KSI agent container
# Full variant (local embedding/memory extras)

# Base image pinned by digest for reproducible builds (tag kept for readability).
# The digest is the multi-arch manifest list, so it resolves on amd64/arm64 alike.
# To bump: `docker pull node:<tag> && docker inspect --format='{{index .RepoDigests 0}}' node:<tag>`
# then update BOTH container/Dockerfile and container/Dockerfile.bench to match.
FROM node:22.16-slim@sha256:048ed02c5fd52e86fda6fbd2f6a76cf0d4492fd6c6fee9e2c463ed5108da0e34
ARG TARGETARCH

RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir \
      'filelock==3.29.0' \
      'fsspec==2026.2.0' \
      'jinja2==3.1.6' \
      'networkx==3.6.1' \
      'sympy==1.14.0' \
      'typing-extensions==4.15.0' \
    && if [ "${TARGETARCH:-}" = "amd64" ]; then \
      pip3 install --break-system-packages --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/cpu \
        'torch==2.7.1+cpu'; \
    else \
      pip3 install --break-system-packages --no-cache-dir \
        'torch==2.6.0'; \
    fi \
    && pip3 install --break-system-packages --no-cache-dir \
      'sqlite-vec==0.1.9' \
      'sentence-transformers==3.2.1' \
      'transformers>=4.41,<5.0'

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

ENV USE_TF=0
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/home/node/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/home/node/.cache/sentence-transformers

WORKDIR /app

COPY runtime_runner/agent-runner/package.json ./package.json
COPY runtime_runner/agent-runner/package-lock.json ./package-lock.json
COPY runtime_runner/agent-runner/tsconfig.json ./tsconfig.json
RUN npm ci --legacy-peer-deps

COPY runtime_runner/agent-runner/src ./src

RUN npx tsc --outDir /tmp/dist \
 && SRC_HASH=$(find src -type f -name "*.ts" -exec md5sum {} + 2>/dev/null | sort | md5sum | cut -d" " -f1) \
 && echo "$SRC_HASH" > /tmp/dist/.src_hash \
 && ln -sfn /app/node_modules /tmp/dist/node_modules

# Single source of truth for retryable-error markers (shared with the Python
# orchestrator via src/ksi/errors.py). retryable_markers.ts reads this at
# runtime from the compiled module's dir (/tmp/dist; tsc does not copy .json
# into outDir), else it silently uses its vendored fallback. Keep in lockstep
# with container/Dockerfile.bench. See issue #648.
COPY runtime_runner/shared/retryable_markers.json /tmp/dist/retryable_markers.json

RUN mkdir -p /workspace/task /workspace/global /workspace/extra /workspace/ipc/input /app/memory-db

# Entrypoint lives in container/entrypoint.sh so it can be version-controlled
# and unit-tested. See tests/test_tsc_error_surfacing.py for the framing
# contract (====TSC_COMPILE_FAILED==== block on tsc failure).
COPY container/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN chown -R node:node /workspace /app/memory-db && chmod -R 777 /tmp/dist && chmod 777 /home/node

USER node
WORKDIR /workspace/task
ENTRYPOINT ["/app/entrypoint.sh"]
