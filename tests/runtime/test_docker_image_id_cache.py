"""Unit tests for the per-tag _docker_image_id cache.

A blocking ``docker image inspect`` ran once per task for the same image. The
result is cached per image tag so repeated lookups avoid the subprocess.
"""

from unittest.mock import MagicMock, patch

import kcsi.runtime.container_host as ch


def _proc(stdout: str, returncode: int = 0) -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    return m


def test_successful_lookup_is_cached(monkeypatch):
    ch._DOCKER_IMAGE_ID_CACHE.clear()
    with patch.object(ch.subprocess, "run", return_value=_proc("sha256:abc\n")) as run:
        first = ch._docker_image_id("kcsi-agent:bench")
        second = ch._docker_image_id("kcsi-agent:bench")
    assert first == "sha256:abc"
    assert second == "sha256:abc"
    # Second lookup served from cache -> no second subprocess.
    assert run.call_count == 1
    ch._DOCKER_IMAGE_ID_CACHE.clear()


def test_distinct_tags_probed_separately(monkeypatch):
    ch._DOCKER_IMAGE_ID_CACHE.clear()
    outputs = {"img:a": "sha256:aaa\n", "img:b": "sha256:bbb\n"}

    def fake_run(cmd, **kwargs):
        return _proc(outputs[cmd[-1]])

    with patch.object(ch.subprocess, "run", side_effect=fake_run) as run:
        assert ch._docker_image_id("img:a") == "sha256:aaa"
        assert ch._docker_image_id("img:b") == "sha256:bbb"
        # Re-lookup both: still cached.
        assert ch._docker_image_id("img:a") == "sha256:aaa"
        assert ch._docker_image_id("img:b") == "sha256:bbb"
    assert run.call_count == 2
    ch._DOCKER_IMAGE_ID_CACHE.clear()


def test_failed_lookup_not_cached(monkeypatch):
    """A miss must NOT pin: an image absent now may appear after a build."""
    ch._DOCKER_IMAGE_ID_CACHE.clear()
    with patch.object(ch.subprocess, "run", return_value=_proc("", returncode=1)) as run:
        assert ch._docker_image_id("img:missing") == ""
        assert ch._docker_image_id("img:missing") == ""
    # Re-probed both times because the failure was not cached.
    assert run.call_count == 2
    assert "img:missing" not in ch._DOCKER_IMAGE_ID_CACHE
    ch._DOCKER_IMAGE_ID_CACHE.clear()
