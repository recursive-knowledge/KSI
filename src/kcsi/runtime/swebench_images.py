from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from ..models import TaskSpec
from ..tasks.registry import resolve_source

log = logging.getLogger(__name__)

# --- Shared low-level helpers ------------------------------------------------
# These small, dependency-free utilities are used both by the SWE-bench image
# helpers below and by ``container_host`` elsewhere. They live here (rather than
# in ``container_host``) so the SWE-bench image helpers can import them without
# creating an import cycle; ``container_host`` re-imports them to preserve the
# ``container_host.<name>`` access path.

_CREDENTIAL_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9\-]{10,}"  # Anthropic API keys
    r"|sk-[A-Za-z0-9_-]{20,}"  # OpenAI API keys (classic and sk-proj-)
    r"|hf_[A-Za-z0-9]{20,}"  # Hugging Face tokens
    r"|gh[pousr]_[A-Za-z0-9]{20,}"  # GitHub tokens (ghp_/gho_/ghu_/ghs_/ghr_)
    r"|Bearer\s+[A-Za-z0-9._\-]{20,})",  # raw bearer tokens
    re.IGNORECASE,
)


def _scrub_credentials(text: str) -> str:
    """Remove API keys and bearer tokens from error text."""
    return _CREDENTIAL_RE.sub("[REDACTED]", text)


def _tail(value: str, max_chars: int) -> str:
    value = value or ""
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


_SAFE_DOCKER_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,511}$")
_SAFE_DOCKER_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_SAFE_DOCKER_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_SAFE_SWEBENCH_INSTANCE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_MISSING_METADATA_VALUES = {"", "none", "null", "nan", "na", "n/a", "<na>"}
_DEFAULT_SWEBENCH_DOCKERHUB_USERNAME = "jefzda"
_DEFAULT_KCSI_AGENT_IMAGE = "kcsi-agent:bench"
_SWE_WORKSPACE_REPO_CONTAINER_PATH = "/workspace/task/workspace/repo"
_SWE_OFFICIAL_REPO_ALIAS_PATH = "/app"
_SWE_OFFICIAL_RUNNER_ROOT = "/kcsi-runner"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _metadata_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in _MISSING_METADATA_VALUES:
        return ""
    return text


def _is_enabled_env(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _safe_docker_image_ref(value: Any) -> str:
    image = _metadata_text(value)
    if not image or image.startswith("-") or any(ch.isspace() for ch in image):
        return ""
    if not _SAFE_DOCKER_IMAGE_RE.fullmatch(image):
        return ""
    return image


def _safe_docker_tag(value: Any) -> str:
    tag = _metadata_text(value)
    if not tag or not _SAFE_DOCKER_TAG_RE.fullmatch(tag):
        return ""
    return tag


def _safe_docker_repository(value: Any, *, fallback: str) -> str:
    repo = _metadata_text(value)
    if repo and _SAFE_DOCKER_REPOSITORY_RE.fullmatch(repo) and not repo.startswith("-"):
        return repo.rstrip("/")
    return fallback


def _swebench_dockerhub_username(metadata: dict[str, Any], env: dict[str, str]) -> str:
    return _safe_docker_repository(
        metadata.get("dockerhub_username")
        or metadata.get("swebench_pro_dockerhub_username")
        or env.get("SWEBENCH_PRO_DOCKERHUB_USERNAME")
        or env.get("SWE_BENCH_PRO_DOCKERHUB_USERNAME"),
        fallback=_DEFAULT_SWEBENCH_DOCKERHUB_USERNAME,
    )


def _fallback_swebench_dockerhub_tag(task: TaskSpec) -> str:
    metadata = task.metadata or {}
    instance_id = _metadata_text(metadata.get("instance_id") or task.id)
    if not instance_id or not _SAFE_SWEBENCH_INSTANCE_RE.fullmatch(instance_id):
        return ""
    repo_name = _metadata_text(metadata.get("repo") or task.repo).lower()
    if "/" not in repo_name:
        return ""
    repo_base, repo_name_only = repo_name.split("/", 1)
    if not repo_base or not repo_name_only:
        return ""
    hsh = instance_id.replace("instance_", "")

    if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name and "element-web" in repo_name:
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    candidate = f"{repo_base}.{repo_name_only}-{hsh}"[:128]
    # Docker tags must start with [A-Za-z0-9_] and contain only [A-Za-z0-9_.-].
    # `_SAFE_SWEBENCH_INSTANCE_RE` permits values starting with `.` or `-`,
    # which would produce a tag Docker rejects at build/inspect time. Validate
    # the constructed tag before returning so callers fall through to the
    # explicit `dockerhub_tag`/`image_name` path or no-op cleanly.
    if not _SAFE_DOCKER_TAG_RE.fullmatch(candidate):
        return ""
    return candidate


def _swebench_official_base_image(task: TaskSpec, env: dict[str, str]) -> str:
    metadata = task.metadata or {}
    _img_spec = resolve_source(metadata.get("task_source"))
    if _img_spec is None or not _img_spec.uses_repo_snapshots:
        return ""
    for key in (
        "official_container_image",
        "image_name",
        "docker_image",
        "swebench_image",
        "swebench_pro_image",
    ):
        image = _safe_docker_image_ref(metadata.get(key))
        if image:
            return image
    tag = _safe_docker_tag(metadata.get("dockerhub_tag"))
    if not tag:
        tag = _fallback_swebench_dockerhub_tag(task)
    if not tag:
        return ""
    username = _swebench_dockerhub_username(metadata, env)
    return f"{username}/sweap-images:{tag}"


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _swebench_runner_image(env: dict[str, str]) -> str:
    return (
        _safe_docker_image_ref(env.get("KCSI_SWEBENCH_PRO_RUNNER_IMAGE"))
        or _safe_docker_image_ref(env.get("KCSI_AGENT_RUNNER_IMAGE"))
        or _safe_docker_image_ref(env.get("CONTAINER_IMAGE"))
        or _safe_docker_image_ref(env.get("KCSI_CONTAINER_IMAGE"))
        or _DEFAULT_KCSI_AGENT_IMAGE
    )


_DOCKER_IMAGE_ID_CACHE: dict[str, str] = {}


def _docker_image_id(image: str) -> str:
    """Return the docker image content ID, or empty string on lookup failure.

    Cached per image tag (for the process lifetime) because a blocking
    ``docker image inspect`` runs once per task for the same image (e.g. the
    kcsi-agent runner). This is safe because the runner image is built
    out-of-process (``container/build.sh`` / ``scripts/setup_all.sh``) before a
    run starts, so its content id is stable within a process and a fresh run is
    a fresh process with an empty cache. A rebuild of the same tag *while the
    process is running* would not be picked up — there is no in-process cache
    reset; the out-of-process build ordering is the invariant that makes this
    sound. (Lookup failures are intentionally not cached; see below.)
    """
    if image in _DOCKER_IMAGE_ID_CACHE:
        return _DOCKER_IMAGE_ID_CACHE[image]
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        # Do NOT cache transient lookup failures: a missing image may appear
        # after a build, and caching "" here would pin the stale miss.
        return ""
    if proc.returncode != 0:
        return ""
    image_id = (proc.stdout or "").strip()
    _DOCKER_IMAGE_ID_CACHE[image] = image_id
    return image_id


def _agent_runner_manifest_dir(root: Path) -> Path:
    # The agent-runner npm manifest (package.json/tsconfig.json) is being
    # co-located under runtime_runner/agent-runner/; older trees keep
    # it under container/agent-runner/. Resolve whichever exists so the overlay
    # tag keeps rotating on a manifest change regardless of which layout is
    # checked out — otherwise a digest input pointing at the moved-away path
    # silently becomes "" (missing-file digest) and a stale overlay is reused.
    relocated = root / "runtime_runner" / "agent-runner"
    if (relocated / "package.json").exists():
        return relocated
    return root / "container" / "agent-runner"


def _swebench_agent_image_tag(base_image: str, runner_image: str, env: dict[str, str]) -> str:
    root = _repo_root()
    manifest_dir = _agent_runner_manifest_dir(root)
    # Include the runner image's CONTENT id (not just its tag string) so that a
    # rebuild of kcsi-agent:bench under the same tag invalidates the cached
    # agent_image. Without this, a stale overlay is reused after a runner
    # rebuild and the new TypeScript never reaches the container. Falls back
    # to the tag string when docker isn't reachable so unit tests stay
    # deterministic.
    runner_id = _docker_image_id(runner_image) or runner_image
    digest_input = "\n".join(
        [
            base_image,
            runner_image,
            runner_id,
            _file_digest(root / "container" / "entrypoint.sh"),
            _file_digest(manifest_dir / "package.json"),
            _file_digest(manifest_dir / "tsconfig.json"),
            _file_digest(Path(__file__)),
        ]
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:20]
    prefix = _safe_docker_repository(
        env.get("KCSI_SWEBENCH_PRO_AGENT_IMAGE_PREFIX"),
        fallback="kcsi-swebench-pro-agent",
    )
    return f"{prefix}:{digest}"


def _swebench_agent_overlay_dockerfile(*, base_image: str, runner_image: str) -> str:
    # Why we install npm/npx as symlinks instead of `COPY /usr/local/bin/npm`:
    # in the runner image those paths are symlinks to
    # ../lib/node_modules/npm/bin/{npm,npx}-cli.js. Docker's COPY follows
    # symlinks and writes the target's content into /usr/local/bin/. The
    # extracted *-cli.js scripts then do `require('../lib/cli.js')` resolved
    # from their new location (/usr/local/bin/) — which lands on /usr/lib/...
    # instead of /usr/local/lib/node_modules/npm/lib/. Result: every overlay
    # container exits at startup with "Cannot find module '../lib/cli.js'"
    # the moment the entrypoint runs `npx tsc` to recompile mounted TypeScript.
    # Recreating the symlinks at the destination preserves the relative-path
    # resolution that npm-cli.js / npx-cli.js depend on.
    return f"""\
FROM {runner_image} AS kcsi_runner
FROM {base_image}
USER root
COPY --from=kcsi_runner /usr/local/bin/node /usr/local/bin/node
COPY --from=kcsi_runner /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=kcsi_runner /app {_SWE_OFFICIAL_RUNNER_ROOT}
COPY --from=kcsi_runner /tmp/dist /tmp/dist
RUN set -eux; \\
    if command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then \\
      ln -s "$(command -v python3)" /usr/local/bin/python; \\
    fi; \\
    rm -f /usr/local/bin/npm /usr/local/bin/npx; \\
    ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm; \\
    ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx; \\
    ln -sfn {_SWE_OFFICIAL_RUNNER_ROOT}/node_modules /tmp/dist/node_modules; \\
    chmod +x {_SWE_OFFICIAL_RUNNER_ROOT}/entrypoint.sh; \\
    mkdir -p /workspace/task /workspace/global /workspace/extra \\
      /workspace/ipc/input {_SWE_OFFICIAL_RUNNER_ROOT}/memory-db \\
      /home/node/.cache/claude-cli-nodejs /home/node/.cache/huggingface \\
      /home/node/.cache/sentence-transformers; \\
    chmod -R a+rwX /workspace /home/node {_SWE_OFFICIAL_RUNNER_ROOT}/memory-db /tmp/dist
ENV KCSI_RUNNER_ROOT={_SWE_OFFICIAL_RUNNER_ROOT}
ENV HOME=/home/node
ENV HF_HOME=/home/node/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/home/node/.cache/sentence-transformers
ENV USE_TF=0
ENV TOKENIZERS_PARALLELISM=false
WORKDIR /workspace/task
USER 1000
ENTRYPOINT ["{_SWE_OFFICIAL_RUNNER_ROOT}/entrypoint.sh"]
"""


_AGENT_IMAGE_INPROC_LOCK = threading.Lock()
_AGENT_IMAGE_INPROC_BUILDS: dict[str, threading.Lock] = {}


def _agent_image_lock_path(agent_image: str) -> Path:
    digest = hashlib.sha256(agent_image.encode("utf-8")).hexdigest()[:32]
    lock_dir = Path(tempfile.gettempdir()) / "kcsi-swebench-agent-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"{digest}.lock"


@contextlib.contextmanager
def _agent_image_build_lock(agent_image: str):
    """Serialize concurrent ensure-image calls for the same agent_image.

    Uses an in-process threading.Lock + an fcntl.flock on a tempfile so that
    parallel orchestrator workers (in-process via ThreadPoolExecutor) and
    parallel processes (e.g. multiple campaign runners) both wait for an
    in-flight `docker build` instead of duplicating it. On platforms without
    fcntl (e.g. Windows), the threading.Lock alone applies.
    """
    with _AGENT_IMAGE_INPROC_LOCK:
        per_image_lock = _AGENT_IMAGE_INPROC_BUILDS.setdefault(agent_image, threading.Lock())
    per_image_lock.acquire()
    try:
        try:
            import fcntl
        except ImportError:
            yield
            return
        lock_path = _agent_image_lock_path(agent_image)
        with open(lock_path, "a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        per_image_lock.release()


def _agent_image_present(agent_image: str) -> bool:
    inspect = subprocess.run(
        ["docker", "image", "inspect", agent_image],
        cwd=str(_repo_root()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=15,
        check=False,
    )
    return inspect.returncode == 0


def _ensure_swebench_agent_image(base_image: str, env: dict[str, str]) -> tuple[str, str]:
    runner_image = _swebench_runner_image(env)
    agent_image = _swebench_agent_image_tag(base_image, runner_image, env)
    rebuild = _is_enabled_env(env.get("KCSI_SWEBENCH_PRO_REBUILD_AGENT_IMAGE"), default=False)
    if not rebuild and _agent_image_present(agent_image):
        return agent_image, runner_image

    with _agent_image_build_lock(agent_image):
        # Re-check after acquiring the lock — another worker may have just
        # finished building the same image.
        if not rebuild and _agent_image_present(agent_image):
            return agent_image, runner_image

        dockerfile = _swebench_agent_overlay_dockerfile(base_image=base_image, runner_image=runner_image)
        build_cmd = ["docker", "build", "-t", agent_image, "-f", "-"]
        if _is_enabled_env(env.get("KCSI_SWEBENCH_PRO_DOCKER_PULL"), default=False):
            build_cmd.append("--pull")
        with tempfile.TemporaryDirectory(prefix="kcsi-swebench-image-") as td:
            proc = subprocess.run(
                [*build_cmd, td],
                input=dockerfile,
                cwd=str(_repo_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(env.get("KCSI_SWEBENCH_PRO_IMAGE_BUILD_TIMEOUT_SEC") or "1800"),
                check=False,
            )
        if proc.returncode != 0:
            stderr_tail = _scrub_credentials(_tail(proc.stderr or proc.stdout or "", 4000))
            raise RuntimeError(
                "Failed to build SWE-bench Pro official runner image "
                f"{agent_image!r} from {base_image!r}: {stderr_tail}"
            )
    return agent_image, runner_image


_BASE_IMAGE_LIBC_CACHE: dict[str, str] = {}


def _ensure_base_image_present(base_image: str, env: dict[str, str] | None = None) -> None:
    """Best-effort pre-pull of ``base_image`` so a subsequent libc probe runs
    warm.

    The probe in :func:`_detect_base_image_libc` runs ``docker run`` under a
    short timeout; on a not-yet-present image that ``docker run`` triggers a
    cold pull, and a ~5GB base under concurrent pulls blows the timeout →
    the probe returns '' (inconclusive) → the caller fails closed and a genuine
    glibc base needlessly loses the overlay. Pulling first (with a generous,
    image-pull-sized timeout) makes the probe fast and reliable. All failures
    are swallowed: the probe still runs afterwards, just possibly cold.

    The pull uses the Docker daemon's own network, not the probe container's
    ``--network none`` namespace, and downloads layers without executing any
    image code — consistent with the probe's tampered-image threat model.
    """
    env = env or {}
    try:
        present = subprocess.run(
            ["docker", "image", "inspect", base_image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if present.returncode == 0:
            return
        timeout = int(env.get("KCSI_SWEBENCH_PRO_IMAGE_PULL_TIMEOUT_SEC") or "1800")
        subprocess.run(
            ["docker", "pull", base_image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except Exception:
        # Pre-pull is an optimization; the probe still runs (possibly cold).
        return


def _detect_base_image_libc(base_image: str, env: dict[str, str] | None = None) -> str:
    """Probe the official base image for its libc. Returns 'glibc', 'musl',
    or '' on detection failure. A positive 'glibc'/'musl' is cached per-image
    (each probe spins up a short-lived container); an inconclusive '' is NOT
    cached, so a later call after the image finishes pulling can re-probe.

    Why we care: the kcsi-agent:bench runner ships a glibc-built `node`
    binary. The overlay design copies that binary into the official
    SWE-bench Pro base image. If the base is Alpine (musl), node hits
    `Error relocating /usr/local/bin/node: fcntl64: symbol not found` at
    container start — every task on that image errors before any tool call.
    Detecting libc lets the caller keep the overlay only for confirmed-glibc
    bases and fall back to the legacy shared-runner path otherwise.
    """
    if base_image in _BASE_IMAGE_LIBC_CACHE:
        return _BASE_IMAGE_LIBC_CACHE[base_image]
    # Pre-pull so the probe below runs against a warm image. Otherwise the
    # probe's `docker run` triggers a cold pull inside its short `timeout`; a
    # ~5GB base under concurrent pulls blows the timeout, the probe returns ''
    # (inconclusive), and the caller fails closed even for glibc bases that
    # would have supported the overlay. A warm probe returns in ~0.1s.
    _ensure_base_image_present(base_image, env)
    try:
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                # The base image is dataset-controlled (metadata image_name /
                # dockerhub_tag), and this probe runs the image's OWN sh/binaries
                # on the host before the task's isolated agent container exists.
                # The probe only inspects local files (libc markers), so give it
                # no network — defense-in-depth against a tampered-dataset image
                # phoning out from the non-isolated host bridge. Any image pull
                # is performed by the Docker daemon (its own host network) and is
                # unaffected by the container's `--network none` namespace.
                "--network",
                "none",
                "--entrypoint",
                "sh",
                base_image,
                "-c",
                "if [ -f /etc/alpine-release ] || ls /lib/ld-musl-* >/dev/null 2>&1; then "
                "echo musl; "
                "elif ls /lib*/ld-linux* >/dev/null 2>&1 || command -v ldd >/dev/null 2>&1; then "
                "echo glibc; "
                "else echo unknown; fi",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception:
        # Do NOT cache an inconclusive result — a later call, after the image
        # has finished pulling, must be able to re-probe and get the real libc.
        return ""
    out = (proc.stdout or "").strip().splitlines()
    libc = out[-1].strip() if out else ""
    if libc not in {"glibc", "musl"}:
        # Inconclusive — do not cache; allow a warm re-probe to succeed later.
        return ""
    _BASE_IMAGE_LIBC_CACHE[base_image] = libc
    return libc


def _swebench_pro_container_images(task: TaskSpec, env: dict[str, str]) -> dict[str, str]:
    if not _is_enabled_env(env.get("KCSI_SWEBENCH_PRO_OFFICIAL_CONTAINERS"), default=True):
        return {}
    base_image = _swebench_official_base_image(task, env)
    if not base_image:
        return {}
    libc = _detect_base_image_libc(base_image, env)
    if libc != "glibc":
        # Fail CLOSED: only a positively-detected glibc base can host the
        # runner's glibc `node` binary that the overlay copies in. On a musl
        # (Alpine) base that node exit-127s at container start
        # (`fcntl64: symbol not found`) and the task fails non-retryably. On an
        # inconclusive probe ('') we cannot prove glibc — and the probe returns
        # '' precisely when it times out on a cold multi-GB pull, exactly when a
        # musl base is most likely to slip through — so we must not risk the
        # overlay. Skip it; the caller falls back to the legacy shared-runner
        # path (no in-image build toolchain, but the agent still runs and
        # produces a gradable workspace_diff). Better than a non-retryable
        # exit-127. `_ensure_base_image_present` (in _detect_base_image_libc)
        # pre-pulls so genuine glibc bases are still detected and keep the overlay.
        reason = "is musl-based" if libc == "musl" else f"has unconfirmed libc (detected {libc!r})"
        log.warning(
            "Skipping SWE-bench Pro official-container overlay for %s: base image %r %s "
            "(set KCSI_SWEBENCH_PRO_OFFICIAL_CONTAINERS=0 to silence this).",
            task.id,
            base_image,
            reason,
        )
        return {}
    agent_image, runner_image = _ensure_swebench_agent_image(base_image, env)
    return {
        "official_container_image": base_image,
        "container_image": agent_image,
        "runner_image": runner_image,
        "repo_container_path": _SWE_WORKSPACE_REPO_CONTAINER_PATH,
        "official_repo_container_path": _SWE_OFFICIAL_REPO_ALIAS_PATH,
        "runner_root": _SWE_OFFICIAL_RUNNER_ROOT,
    }
