"""Tests for the ksi Polyglot Docker image recipe."""

from __future__ import annotations

import pytest

from ksi.benchmarks import polyglot_docker


def test_polyglot_dockerfile_tracks_hyperagents_base_toolchain() -> None:
    dockerfile = polyglot_docker.DOCKERFILE

    assert "FROM buildpack-deps:jammy" in dockerfile
    assert 'LABEL org.knowledgecentric.polyglot.recipe="polyglot-hyperagents-base-20260422-rust-pin"' in dockerfile
    assert 'org.knowledgecentric.polyglot.recipe_target_arch="' in dockerfile
    assert (
        'org.knowledgecentric.polyglot.recipe_source="baselines/hyperagents/domains/polyglot/dockerfiles.py:_DOCKERFILE_BASE"'
        in dockerfile
    )
    assert 'org.mockingjay.polyglot.recipe="polyglot-hyperagents-base-20260422-rust-pin"' in dockerfile
    assert 'org.mockingjay.polyglot.recipe_target_arch="' in dockerfile
    assert "Python 3.11 comes from Miniconda below" in dockerfile
    assert "ppa:deadsnakes/ppa" not in dockerfile
    assert "python3.11-venv" not in dockerfile
    assert "python3.11-dev" not in dockerfile
    assert "go1.21.5.linux-$GOARCH.tar.gz" in polyglot_docker.HYPERAGENTS_BASE_DOCKERFILE
    assert "https://sh.rustup.rs" in dockerfile
    # Rust toolchain pinned to the bench image's RUST_TOOLCHAIN (#1079).
    assert "/tmp/rustup.sh -y --default-toolchain 1.85.1" in dockerfile
    assert "https://deb.nodesource.com/setup_20.x" in dockerfile
    assert "openjdk-21-jdk" in dockerfile
    assert "Miniconda3-py311_23.11.0-2-Linux-" in dockerfile
    assert "@exercism/babel-preset-javascript@0.2.1" in dockerfile
    assert 'ENV PATH="/usr/local/go/bin:${PATH}"' in dockerfile
    assert 'ENV PATH="/root/.cargo/bin:${PATH}"' in dockerfile
    assert "ENV PATH=/opt/miniconda3/bin:$PATH" in dockerfile

    assert "FROM ubuntu:24.04" not in dockerfile
    assert "golang-go" not in dockerfile
    assert "/opt/python311" not in dockerfile


@pytest.mark.parametrize(
    ("platform_name", "expected_arch", "expected_conda"),
    [
        ("linux/amd64", "x86_64", "Linux-x86_64.sh"),
        ("linux/x86_64", "x86_64", "Linux-x86_64.sh"),
        ("linux/arm64/v8", "arm64", "Linux-aarch64.sh"),
    ],
)
def test_polyglot_dockerfile_uses_target_arch_for_conda(platform_name, expected_arch, expected_conda) -> None:
    dockerfile = polyglot_docker.dockerfile_for_target(platform_name=platform_name)

    assert expected_conda in dockerfile
    assert f'{polyglot_docker.POLYGLOT_RECIPE_TARGET_ARCH_LABEL}="{expected_arch}"' in dockerfile


def test_build_image_uses_docker_default_platform_for_recipe(monkeypatch) -> None:
    calls = []

    def fake_matches(*args, **kwargs):
        calls.append(("matches", args, kwargs))
        return False

    def fake_run(cmd: list[str], **kwargs):
        calls.append(("run", cmd, kwargs))
        return polyglot_docker.subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("DOCKER_DEFAULT_PLATFORM", "linux/arm64/v8")
    monkeypatch.setattr(polyglot_docker, "image_matches_recipe", fake_matches)
    monkeypatch.setattr(polyglot_docker.subprocess, "run", fake_run)

    polyglot_docker.build_image(force=False)

    assert calls[0] == (
        "matches",
        (polyglot_docker.IMAGE_NAME,),
        {"platform_name": "linux/arm64/v8", "arch": "arm64"},
    )
    build_call = calls[1]
    assert build_call[0] == "run"
    assert build_call[1][:5] == ["docker", "build", "--network", "host", "--platform"]
    assert build_call[1][5] == "linux/arm64/v8"
    assert "Linux-aarch64.sh" in build_call[2]["input"]


def test_polyglot_image_recipe_match_checks_recipe_label(monkeypatch) -> None:
    def fake_run(cmd: list[str], **kwargs):
        assert cmd == [
            "docker",
            "image",
            "inspect",
            "poly-img",
            "--format",
            "{{json .Config.Labels}}",
        ]
        return polyglot_docker.subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                '{"org.knowledgecentric.polyglot.recipe":"polyglot-hyperagents-base-20260422-rust-pin",'
                '"org.knowledgecentric.polyglot.recipe_target_arch":"x86_64"}\n'
            ),
            stderr="",
        )

    monkeypatch.delenv("DOCKER_DEFAULT_PLATFORM", raising=False)
    monkeypatch.setattr(polyglot_docker, "platform", type("P", (), {"machine": staticmethod(lambda: "x86_64")}))
    monkeypatch.setattr(polyglot_docker.subprocess, "run", fake_run)

    assert polyglot_docker.image_matches_recipe("poly-img") is True


def test_polyglot_image_recipe_match_accepts_alias_labels(monkeypatch) -> None:
    def fake_run(cmd: list[str], **kwargs):
        assert cmd == [
            "docker",
            "image",
            "inspect",
            "poly-img",
            "--format",
            "{{json .Config.Labels}}",
        ]
        return polyglot_docker.subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                '{"org.mockingjay.polyglot.recipe":"polyglot-hyperagents-base-20260422-rust-pin",'
                '"org.mockingjay.polyglot.recipe_target_arch":"x86_64"}\n'
            ),
            stderr="",
        )

    monkeypatch.delenv("DOCKER_DEFAULT_PLATFORM", raising=False)
    monkeypatch.setattr(polyglot_docker, "platform", type("P", (), {"machine": staticmethod(lambda: "x86_64")}))
    monkeypatch.setattr(polyglot_docker.subprocess, "run", fake_run)

    assert polyglot_docker.image_matches_recipe("poly-img") is True


def test_polyglot_dockerfile_keeps_ksi_harness_compatibility() -> None:
    dockerfile = polyglot_docker.DOCKERFILE

    assert "python -m pip install --no-cache-dir --upgrade pip pytest" in dockerfile
    assert "gradle-8.5-bin.zip" in dockerfile
    assert "catch2" in dockerfile
    assert "WORKDIR /exercise" in dockerfile
