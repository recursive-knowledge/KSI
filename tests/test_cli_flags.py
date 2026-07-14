"""Tests for new three-phase CLI flags and removed-flag tombstones."""

from __future__ import annotations

import subprocess
import sys

from ksi.cli import _build_parser

_BASE_ARGV = ["--task-source", "polyglot", "--tasks-path", "dummy.json"]


def test_new_flags_in_help():
    """The new three-phase flags should appear in --help output."""
    result = subprocess.run(
        [sys.executable, "-m", "ksi.cli", "--help"],
        capture_output=True,
        text=True,
    )
    out = result.stdout + result.stderr
    for f in [
        "--per-task-forum-rounds",
        "--cross-task-forum-rounds",
        "--cross-task-forum-timeout-sec",
        "--distill-enabled",
    ]:
        assert f in out, f"missing flag: {f}"


def test_three_phase_flag_defaults():
    """Without the three-phase flags, the argparse defaults apply."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "polyglot",
            "--tasks-path",
            "dummy.json",
        ]
    )
    assert args.per_task_forum_rounds == 1
    assert args.cross_task_forum_rounds == 2  # V2: default raised 1→2


def test_distill_enabled_bare_flag():
    """Bare --distill-enabled (no value) must parse as True."""
    from ksi.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "polyglot",
            "--tasks-path",
            "dummy.json",
            "--distill-enabled",
        ]
    )
    assert args.distill_enabled is True


def test_distill_enabled_explicit_values():
    """--distill-enabled accepts true/false/yes/no/1/0 variants."""
    from ksi.cli import _build_parser

    base = [
        "--task-source",
        "polyglot",
        "--tasks-path",
        "dummy.json",
    ]

    for raw, expected in [
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
    ]:
        parser = _build_parser()
        args = parser.parse_args([*base, "--distill-enabled", raw])
        assert args.distill_enabled is expected, (raw, expected)

    # = syntax should also work
    parser = _build_parser()
    args = parser.parse_args([*base, "--distill-enabled=false"])
    assert args.distill_enabled is False


def test_distill_enabled_default_is_true():
    """When --distill-enabled is not passed, default is True."""
    from ksi.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "polyglot",
            "--tasks-path",
            "dummy.json",
        ]
    )
    assert args.distill_enabled is True


# ---------------------------------------------------------------------------
# --per-task-forum-skip-when-monologue removed in V2 design (Phase 2 always
# runs). Tests for the removed flag are deleted; the new contract is that
# the flag should NOT appear in --help.
# ---------------------------------------------------------------------------


def test_per_task_forum_skip_when_monologue_flag_removed():
    """V2 removes the monologue-skip escape hatch; flag must NOT appear."""
    result = subprocess.run(
        [sys.executable, "-m", "ksi.cli", "--help"],
        capture_output=True,
        text=True,
    )
    out = result.stdout + result.stderr
    assert "--per-task-forum-skip-when-monologue" not in out


def test_phase1_reflection_flag_default_off():
    """Without --phase1-reflection-enabled the attribute must default to False."""
    from ksi.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "polyglot",
            "--tasks-path",
            "dummy.json",
        ]
    )
    assert args.phase1_reflection_enabled is False


def test_phase1_reflection_flag_bare_enables():
    """Bare --phase1-reflection-enabled (no value) must parse as True."""
    from ksi.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--task-source",
            "polyglot",
            "--tasks-path",
            "dummy.json",
            "--phase1-reflection-enabled",
        ]
    )
    assert args.phase1_reflection_enabled is True


def test_phase1_reflection_flag_accepts_bool_words():
    """The flag accepts true/false/0/1/yes/no like the other bool-flag forms."""
    from ksi.cli import _build_parser

    base = [
        "--task-source",
        "polyglot",
        "--tasks-path",
        "dummy.json",
    ]
    for raw, expected in [
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
    ]:
        parser = _build_parser()
        args = parser.parse_args([*base, "--phase1-reflection-enabled", raw])
        assert args.phase1_reflection_enabled is expected, (raw, expected)


def test_phase1_reflection_flag_in_help():
    """The flag must show up in --help so users can discover it."""
    result = subprocess.run(
        [sys.executable, "-m", "ksi.cli", "--help"],
        capture_output=True,
        text=True,
    )
    out = result.stdout + result.stderr
    assert "--phase1-reflection-enabled" in out


# ---------------------------------------------------------------------------
# Removed compat flags (--forum-rounds / --forum-mode / --forum-ablate-r3 and
# the --runtime openai alias). They were deprecated aliases for the three-phase
# flags; the flags now hard-error with guidance naming the replacements
# (same tombstone pattern as --memory-db-path below).
# ---------------------------------------------------------------------------


def test_removed_forum_flags_are_rejected_with_guidance(capsys):
    """The removed forum compat flags must hard-error and name the replacement."""
    import pytest

    for argv, replacement in [
        (["--forum-rounds", "5"], "--per-task-forum-rounds"),
        (["--forum-mode", "off"], "--per-task-forum-rounds"),
        (["--forum-ablate-r3"], "--distill-enabled"),
    ]:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([*_BASE_ARGV, *argv])
        err = capsys.readouterr().err
        assert "was removed" in err and replacement in err, (argv, err)


def test_runtime_openai_alias_is_rejected():
    """--runtime openai (removed compat alias) fails argparse choices validation."""
    import pytest

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*_BASE_ARGV, "--runtime", "openai"])


# ---------------------------------------------------------------------------
# Removed --memory-db-path flag. The hydra-layer rejection (in the deleted
# tests/test_experiment_entry.py::test_legacy_memory_db_path_is_rejected) is
# gone with the Hydra PoC; the user-facing CLI rejection in cli.py
# (`_RemovedMemoryDbPathAction`) survives and is what actually guards real
# invocations. Pin it here so the removal stays enforced.
# ---------------------------------------------------------------------------


def test_memory_db_path_flag_is_rejected_with_guidance():
    """--memory-db-path must hard-error (SystemExit) and name the replacements."""
    import pytest

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*_BASE_ARGV, "--memory-db-path", "/tmp/legacy.sqlite"])


def test_memory_db_path_rejection_message_via_subprocess():
    """End-to-end: the CLI exits non-zero and the stderr message points at
    --knowledge-db-path / --runtime-db-path."""
    result = subprocess.run(
        [sys.executable, "-m", "ksi.cli", "--memory-db-path", "/tmp/legacy.sqlite"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    out = result.stdout + result.stderr
    assert "--memory-db-path was removed" in out
    assert "--knowledge-db-path" in out


def test_log_level_flag_parses():
    """--log-level sets args.log_level (case-insensitive)."""
    parser = _build_parser()
    args = parser.parse_args([*_BASE_ARGV, "--log-level", "debug"])
    assert args.log_level == "DEBUG"


def test_log_level_defaults_none():
    """Without --log-level, args.log_level is None (env/default wins)."""
    parser = _build_parser()
    args = parser.parse_args([*_BASE_ARGV])
    assert args.log_level is None


def test_verbose_shortcut_sets_debug():
    """-v / --verbose is a shortcut for --log-level DEBUG."""
    parser = _build_parser()
    assert parser.parse_args([*_BASE_ARGV, "-v"]).log_level == "DEBUG"
    assert parser.parse_args([*_BASE_ARGV, "--verbose"]).log_level == "DEBUG"


def test_log_level_in_help():
    """--log-level should appear in --help output."""
    result = subprocess.run(
        [sys.executable, "-m", "ksi.cli", "--help"],
        capture_output=True,
        text=True,
    )
    out = result.stdout + result.stderr
    assert "--log-level" in out


# ---------------------------------------------------------------------------
# Removed --agents flag. In decentralized task mode the agent count derives
# from the filtered task pool, so the old `--agents N` knob was silently
# ignored. It now hard-errors with guidance (same tombstone pattern as
# --memory-db-path / the removed forum flags) and names --max-concurrent-tasks.
# ---------------------------------------------------------------------------


def test_agents_flag_is_rejected_with_guidance(capsys):
    """--agents must hard-error (SystemExit) and name --max-concurrent-tasks."""
    import pytest

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*_BASE_ARGV, "--agents", "5"])
    err = capsys.readouterr().err
    assert "--agents was removed" in err
    assert "--max-concurrent-tasks" in err


def test_agents_rejection_message_via_subprocess():
    """End-to-end: the CLI exits non-zero and the stderr message points at
    --max-concurrent-tasks."""
    result = subprocess.run(
        [sys.executable, "-m", "ksi.cli", "--agents", "5"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    out = result.stdout + result.stderr
    assert "--agents was removed" in out
    assert "--max-concurrent-tasks" in out
