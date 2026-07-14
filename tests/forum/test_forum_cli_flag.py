"""Tests for curation pipeline CLI flags."""

from ksi.cli import build_parser


def test_curation_timeout_flag():
    """--forum-timeout-sec should be parsed."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--runtime",
            "container",
            "--provider-profile",
            "/tmp/profile.env",
            "--max-concurrent-tasks",
            "8",
            "--max-concurrent-forum-tasks",
            "4",
            "--forum-timeout-sec",
            "900",
        ]
    )
    assert args.forum_timeout_sec == 900


def test_curation_timeout_default():
    """--forum-timeout-sec should default to 900."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--runtime",
            "container",
            "--provider-profile",
            "/tmp/profile.env",
            "--max-concurrent-tasks",
            "8",
            "--max-concurrent-forum-tasks",
            "4",
        ]
    )
    assert args.forum_timeout_sec == 900


def test_curation_concurrency_default():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--runtime",
            "container",
            "--provider-profile",
            "/tmp/profile.env",
        ]
    )
    # 0 = follow --max-concurrent-tasks (see tests/forum/test_forum_worker_cap.py)
    assert args.max_concurrent_forum_tasks == 0


def test_backward_compat_forum_timeout_alias():
    """--forum-timeout-sec should parse to forum_timeout_sec."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--runtime",
            "container",
            "--provider-profile",
            "/tmp/profile.env",
            "--forum-timeout-sec",
            "1200",
        ]
    )
    assert args.forum_timeout_sec == 1200


def test_backward_compat_forum_concurrent_alias():
    """--max-concurrent-forum-tasks should parse to max_concurrent_forum_tasks."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--runtime",
            "container",
            "--provider-profile",
            "/tmp/profile.env",
            "--max-concurrent-forum-tasks",
            "20",
        ]
    )
    assert args.max_concurrent_forum_tasks == 20
