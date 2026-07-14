"""Test CLI flag parsing for knowledge DB path."""

from kcsi.cli import build_parser


def test_knowledge_db_path_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--provider-profile",
            "configs/kcsi/.env.haiku",
        ]
    )
    assert args.knowledge_db_path == "/tmp/memory.sqlite"


def test_runtime_db_path_accepts_explicit_path():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--runtime-db-path",
            "/tmp/memory.sqlite",
            "--provider-profile",
            "configs/kcsi/.env.haiku",
            "--max-concurrent-tasks",
            "8",
            "--max-concurrent-forum-tasks",
            "4",
        ]
    )
    assert args.runtime_db_path == "/tmp/memory.sqlite"


def test_runtime_db_path_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--runtime-db-path",
            "/tmp/runtime.sqlite",
            "--provider-profile",
            "configs/kcsi/.env.haiku",
            "--max-concurrent-tasks",
            "8",
            "--max-concurrent-forum-tasks",
            "4",
        ]
    )
    assert args.runtime_db_path == "/tmp/runtime.sqlite"


def test_knowledge_db_path_optional_with_auto_generation():
    """Knowledge DB path is optional; omitting it is auto-generated at runtime."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--provider-profile",
            "configs/kcsi/.env.haiku",
            "--max-concurrent-tasks",
            "8",
            "--max-concurrent-forum-tasks",
            "4",
        ]
    )
    assert args.knowledge_db_path == ""


def test_capture_limits_defaults():
    """Default values for capture-limit flags match hardcoded defaults."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--provider-profile",
            "configs/kcsi/.env.haiku",
            "--max-concurrent-tasks",
            "8",
            "--max-concurrent-forum-tasks",
            "4",
        ]
    )
    assert args.native_memory_max_chars == 240_000
    assert args.native_memory_max_files == 8
    assert args.native_memory_max_chars_per_file == 60_000


def test_capture_limits_custom_values():
    """Custom values for capture-limit flags parse correctly."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--provider-profile",
            "configs/kcsi/.env.haiku",
            "--max-concurrent-tasks",
            "8",
            "--max-concurrent-forum-tasks",
            "4",
            "--native-memory-max-chars",
            "500000",
            "--native-memory-max-files",
            "16",
            "--native-memory-max-chars-per-file",
            "80000",
        ]
    )
    assert args.native_memory_max_chars == 500_000
    assert args.native_memory_max_files == 16
    assert args.native_memory_max_chars_per_file == 80_000


def test_capture_limits_zero_means_disabled():
    """Setting --native-memory-max-chars 0 parses to 0 (disabled behavior)."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--provider-profile",
            "configs/kcsi/.env.haiku",
            "--max-concurrent-tasks",
            "8",
            "--max-concurrent-forum-tasks",
            "4",
            "--native-memory-max-chars",
            "0",
        ]
    )
    assert args.native_memory_max_chars == 0


def test_native_memory_file_limits_defaults():
    """Default values for native memory file-level limits."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "swebench_pro",
            "--tasks-path",
            "/tmp/tasks.parquet",
            "--knowledge-db-path",
            "/tmp/memory.sqlite",
            "--provider-profile",
            "configs/kcsi/.env.haiku",
            "--max-concurrent-tasks",
            "8",
            "--max-concurrent-forum-tasks",
            "4",
        ]
    )
    assert args.native_memory_max_files == 8
    assert args.native_memory_max_chars_per_file == 60_000
