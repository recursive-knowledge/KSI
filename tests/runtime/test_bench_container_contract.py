import re
import tomllib

from conftest import REPO_ROOT

DOCKERFILE = REPO_ROOT / "container" / "Dockerfile"
DOCKERFILE_BENCH = REPO_ROOT / "container" / "Dockerfile.bench"
UV_LOCK = REPO_ROOT / "uv.lock"

# Auxiliary Python packages pinned (issue #983) in BOTH Dockerfiles so a Docker
# rebuild can't silently pull a newer transitive release than the one uv.lock
# has already validated.
PINNED_IN_BOTH = {
    "filelock": "3.29.0",
    "fsspec": "2026.2.0",
    "jinja2": "3.1.6",
    "networkx": "3.6.1",
    "sympy": "1.14.0",
    "typing-extensions": "4.15.0",
}

# pytest is only installed in the benchmark variant (it drives the Polyglot
# Python suites in-container); the full variant has no test runner.
PYTEST_PIN = ("pytest", "9.0.3")

_PIP_PIN_RE = re.compile(r"'([A-Za-z0-9_.-]+)==([^']+)'")


def _uv_lock_version(name: str) -> str:
    data = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    matches = [p for p in data.get("package", []) if p.get("name") == name]
    assert matches, f"{name} not found in {UV_LOCK} — did the dependency get removed?"
    assert len(matches) == 1, f"{name} appears {len(matches)} times in {UV_LOCK}"
    version = matches[0].get("version")
    assert version, f"{name} entry in {UV_LOCK} has no version"
    return version


def _dockerfile_pip_pins(path) -> dict[str, str]:
    return dict(_PIP_PIN_RE.findall(path.read_text(encoding="utf-8")))


def test_dockerfiles_pin_exact_aux_dependency_versions():
    """A bare `"pytest" in dockerfile` substring check (the previous form of

    this guard) passes identically whether a package is pinned, unpinned, or
    pinned to the WRONG version. Assert the literal `'pkg==version'` pip
    argument instead, so a typo'd or reverted pin fails CI.
    """
    for path in (DOCKERFILE, DOCKERFILE_BENCH):
        text = path.read_text(encoding="utf-8")
        for name, version in PINNED_IN_BOTH.items():
            pin = f"'{name}=={version}'"
            assert pin in text, f"{path.name} is missing exact pin {pin}"

    bench_text = DOCKERFILE_BENCH.read_text(encoding="utf-8")
    name, version = PYTEST_PIN
    pin = f"'{name}=={version}'"
    assert pin in bench_text, f"{DOCKERFILE_BENCH.name} is missing exact pin {pin}"


def test_dockerfile_pins_match_uv_lock_resolved_versions():
    """Guard against drift: a future `uv lock` bump that resolves one of these

    packages to a newer version must be mirrored into both Dockerfiles, or
    this fails loudly instead of the Dockerfile pin silently going stale.
    Parses each Dockerfile's ACTUAL pinned version (not a hardcoded
    expectation) so it also catches an edit to one file that isn't mirrored
    to the other or to uv.lock.
    """
    for path in (DOCKERFILE, DOCKERFILE_BENCH):
        pins = _dockerfile_pip_pins(path)
        for name in PINNED_IN_BOTH:
            pinned = pins.get(name)
            assert pinned, f"{path.name} has no exact pin for {name}"
            locked = _uv_lock_version(name)
            assert pinned == locked, (
                f"{path.name} pin for {name} ({pinned}) has drifted from uv.lock's "
                f"resolved version ({locked}); update the pin or re-run `uv lock`."
            )

    bench_pins = _dockerfile_pip_pins(DOCKERFILE_BENCH)
    name, _ = PYTEST_PIN
    pinned = bench_pins.get(name)
    assert pinned, f"{DOCKERFILE_BENCH.name} has no exact pin for {name}"
    locked = _uv_lock_version(name)
    assert pinned == locked, (
        f"{DOCKERFILE_BENCH.name} pin for {name} ({pinned}) has drifted from uv.lock's "
        f"resolved version ({locked}); update the pin or re-run `uv lock`."
    )


def test_benchmark_runs_default_to_bench_image():
    config_ts = (REPO_ROOT / "runtime_runner" / "src" / "config.ts").read_text(encoding="utf-8")

    match = re.search(
        r"export const CONTAINER_IMAGE\s*=\s*(?:.|\n)*?'([^']+)';",
        config_ts,
    )

    assert match is not None
    assert match.group(1) == "ksi-agent:bench"


def test_bench_dockerfile_verifies_go_and_ripgrep_contract():
    dockerfile = (REPO_ROOT / "container" / "Dockerfile.bench").read_text(encoding="utf-8")

    assert "ripgrep" in dockerfile
    assert (
        'amd64) GOARCH="amd64"; GOSHA256="e2bc0b3e4b64111ec117295c088bde5f00eeed1567999ff77bc859d7df70078e"'
        in dockerfile
    )
    assert (
        'arm64) GOARCH="arm64"; GOSHA256="841cced7ecda9b2014f139f5bab5ae31785f35399f236b8b3e75dff2a2978d96"'
        in dockerfile
    )
    assert "go${GO_VERSION}.linux-${GOARCH}.tar.gz" in dockerfile
    assert 'echo "${GOSHA256}  /tmp/go.tar.gz" | sha256sum -c -' in dockerfile
    assert "/usr/local/go/bin/go version" in dockerfile
    assert "/usr/local/go/bin/gofmt -h >/dev/null 2>&1" in dockerfile
    assert "rg --version >/dev/null" in dockerfile
    assert "ENV PATH=/usr/local/go/bin:/usr/local/cargo/bin:${JAVA_HOME}/bin:${PATH}" in dockerfile
    assert "golang-go" not in dockerfile


def test_bench_dockerfile_carries_polyglot_toolchains():
    """The agent's own execution image must be able to compile/test the

    non-Go/Python Polyglot languages itself (rust, java, cpp, javascript),
    not just the separate ksi-polyglot-eval grading image: the agent can
    otherwise hit `cargo: command not found` mid-task on rust tasks, and the
    same gap affects JavaScript's `npm test`/jest under egress isolation.
    """
    dockerfile = (REPO_ROOT / "container" / "Dockerfile.bench").read_text(encoding="utf-8")

    assert "'pytest==9.0.3'" in dockerfile
    assert "ARG RUSTUP_VERSION=1.28.2" in dockerfile
    assert "ARG RUST_TOOLCHAIN=1.85.1" in dockerfile
    assert "20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c" in dockerfile
    assert "e3853c5a252fca15252d07cb23a1bdd9377a8c6f3efa01531109281ae47f841c" in dockerfile
    assert "sh.rustup.rs" not in dockerfile
    # Pin the full verification command, not just the bare hash literal, so a
    # future edit can't drop the `| sha256sum -c -` pipe while leaving the
    # hash string sitting unused elsewhere in the file.
    assert 'echo "${RUSTUP_SHA256}  /tmp/rustup-init" | sha256sum -c -' in dockerfile
    assert 'chmod -R a+rX "${RUSTUP_HOME}" "${CARGO_HOME}"' in dockerfile
    assert "chmod -R a+rwX" not in dockerfile
    assert "ARG TEMURIN_VERSION=21.0.11_10" in dockerfile
    assert "4b2220e232a97997b436ca6ab15cbf70171ecff52958a46159dfa5a8c44ca4de" in dockerfile
    assert "8d498ec88e1c1989fab95c6784240ab92d011e29c54d20a3f9c324b13476f9ad" in dockerfile
    assert 'echo "${TEMURIN_SHA256}  /tmp/temurin.tar.gz" | sha256sum -c -' in dockerfile
    assert 'chmod -R a+rX "${JAVA_HOME}"' in dockerfile
    assert "temurin-21-jdk" not in dockerfile
    assert "packages.adoptium.net" not in dockerfile
    assert "OpenJDK21U-jdk_${TEMURIN_ARCH}_linux_hotspot_${TEMURIN_VERSION}.tar.gz" in dockerfile
    assert "ARG GRADLE_VERSION=8.5" in dockerfile
    assert "ARG GRADLE_SHA256=9d926787066a081739e8200858338b4a69e837c3a821a33aca9db09dd4a41026" in dockerfile
    assert "gradle-${GRADLE_VERSION}-bin.zip" in dockerfile
    assert 'echo "${GRADLE_SHA256}  /tmp/gradle.zip" | sha256sum -c -' in dockerfile
    assert "g++" in dockerfile
    assert "cmake" in dockerfile
    # Arbitrary --user <uid>:<gid> (container_args.ts) has no /etc/passwd
    # entry; the JDK resolves user.home via getpwuid and silently breaks
    # Gradle unless this is forced.
    assert "JAVA_TOOL_OPTIONS=-Duser.home=/home/node" in dockerfile
    # jest/babel, version-pinned to match the ksi-polyglot-eval grading
    # image's recipe (polyglot_docker.py's HyperAgents pb.base template).
    assert "jest@29.7.0" in dockerfile
    assert "@babel/core@7.25.2" in dockerfile
    assert "@exercism/babel-preset-javascript@0.2.1" in dockerfile
    assert "babel-jest@29.6.4" in dockerfile
    assert "core-js@3.37.1" in dockerfile
    # Global npm installs aren't on Node's require() resolution path by
    # default (only the bin symlinks are, via PATH) — without this,
    # babel-jest's require() of the preset fails with MODULE_NOT_FOUND in
    # any workspace that has no local node_modules. Verified empirically
    # against a real exercism-style (import/export) test file.
    assert "NODE_PATH=/usr/local/lib/node_modules" in dockerfile
    # None of the four toolchain checksum verifications may be neutered with
    # a non-fatal suffix — a plain substring check on the pinned command
    # above would still pass even if `|| true` were appended after it.
    assert "sha256sum -c - || true" not in dockerfile


def test_bench_and_grader_images_pin_identical_js_toolchain_versions():
    """The agent's bench image and the ksi-polyglot-eval grading image must

    pin the SAME jest/babel versions, or the agent's in-container `npm test`
    smoke pass and the grader's verdict can silently diverge (deep review
    2026-07-03, PR #1091 M1: the grading recipe previously left `jest`
    unpinned while the bench image pinned 29.7.0). Assert each version token
    appears in BOTH container/Dockerfile.bench and the grading recipe in
    src/ksi/benchmarks/polyglot_docker.py, so a bump to one that isn't mirrored to
    the other fails loudly.
    """
    dockerfile_bench = DOCKERFILE_BENCH.read_text(encoding="utf-8")
    grader_recipe = (REPO_ROOT / "src" / "ksi" / "benchmarks" / "polyglot_docker.py").read_text(encoding="utf-8")

    for token in (
        "jest@29.7.0",
        "@babel/core@7.25.2",
        "@exercism/babel-preset-javascript@0.2.1",
        "babel-jest@29.6.4",
        "core-js@3.37.1",
    ):
        assert token in dockerfile_bench, f"Dockerfile.bench is missing JS toolchain pin {token}"
        assert token in grader_recipe, f"polyglot_docker.py is missing JS toolchain pin {token}"
