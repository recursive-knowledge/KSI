"""Tests for seed_package_to_memory_md rendering of scoped bundles (Plan Task 16)."""

from __future__ import annotations

from ksi.runtime.seeding import _EXTERNAL_BUNDLE_TRUNCATION_MARKER, seed_package_to_memory_md


def test_renders_per_task_bundle_section():
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {
            "transferable_insights": ["Use lib X"],
            "pitfalls": ["Don't use Y"],
            "checks": ["Run tests first"],
            "evidence_post_ids": [12],
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert "Task-specific guidance" in md
    assert "Use lib X" in md
    assert "Don't use Y" in md
    assert "Run tests first" in md
    assert "Evidence posts: #12" in md


def test_arc_renders_task_specific_guidance_in_compact_structure():
    pkg = {
        "workstream_name": "arc_solver",
        "per_task_bundle": {
            "transferable_insights": ["Color 9 appears as a stripe feature, not always background."],
            "confirmed_constraints": ["All train/test grids are 30x30."],
            "rejected_hypotheses": ["Coordinate-pair output is invalid for this task."],
            "pitfalls": ["Do not emit sparse point lists."],
            "checks": ["Verify output is a rectangular raster grid."],
            "next_steps": ["Derive a raster-to-raster rule over the full 30x30 grid."],
            "evidence_post_ids": [4, 6],
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="arc_task", task_source="arc")
    assert "Task-specific guidance" in md
    assert "### Hard constraints" in md
    assert "### Rejected rules" in md
    assert "### Do not repeat" in md
    assert "### Verification focus" in md
    assert "### Next hypotheses" in md
    assert "### Observed task insights" in md
    assert "Coordinate-pair output is invalid" in md
    assert "Evidence posts: #4, #6" in md
    assert "### Checks" not in md
    assert "### Insights" not in md


def test_external_per_task_bundle_renders_honest_kt_header():
    """Externally injected bundles (--seed-per-task-bundles-path) must not be
    framed as this task's own distilled history (arm B/C, 2026-07-04)."""
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {
            "transferable_insights": ["Use lib X"],
            "pitfalls": ["Don't use Y"],
            "checks": [],
            "evidence_post_ids": [12],
            "_external_seed_source": "/tmp/bundles.json",
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert "## Possibly-relevant patterns from OTHER tasks (external knowledge transfer)" in md
    assert (
        "These items were distilled from DIFFERENT exercises, not from prior work on this task. "
        "Treat them as optional background patterns." in md
    )
    assert (
        "If anything here conflicts with this task's statement, your own observations, "
        "or official test output, IGNORE the pattern and trust the task evidence." in md
    )
    assert "Task-specific guidance" not in md
    # Item sections render identically to the internal path.
    assert "### Insights" in md
    assert "- Use lib X" in md
    assert "### Pitfalls (do not re-attempt)" in md
    assert "- Don't use Y" in md
    assert "Evidence posts: #12" in md


def test_external_seed_source_path_never_leaks_into_rendered_md():
    """The `_external_seed_source` marker is used only as a truthiness switch
    for header selection — its path VALUE (which can name a host filesystem
    path) must never appear in the rendered MEMORY.md the solver sees."""
    marker = "/private/donor/bundles-2026-07-04.json"
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {
            "transferable_insights": ["Use lib X"],
            "pitfalls": ["Don't use Y"],
            "checks": [],
            "evidence_post_ids": [12],
            "_external_seed_source": marker,
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert marker not in md
    assert "_external_seed_source" not in md


def test_internal_per_task_bundle_section_byte_identical():
    """Pin the internal-path section verbatim: no `_external_seed_source` key
    must keep the pre-existing header/output byte-identical."""
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {
            "transferable_insights": ["Use lib X"],
            "pitfalls": ["Don't use Y"],
            "checks": [],
            "evidence_post_ids": [12],
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    expected_section = (
        "\n"
        "## Task-specific guidance (distilled from previous attempts and forum evidence on this task)\n"
        "Evidence posts: #12\n"
        "### Insights\n"
        "- Use lib X\n"
        "### Pitfalls (do not re-attempt)\n"
        "- Don't use Y"
    )
    assert expected_section in md
    assert "OTHER tasks" not in md
    assert "external knowledge transfer" not in md
    assert "DIFFERENT exercises" not in md


def test_renders_cross_task_bundle_section():
    pkg = {
        "workstream_name": "solver",
        "cross_task_bundle": {
            "transferable_insights": ["Pattern P"],
            "pitfalls": [],
            "checks": [],
            "evidence_post_ids": [7, 9],
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert "Cross-task patterns" in md
    assert "Pattern P" in md
    assert "Evidence posts: #7, #9" in md


def test_arc_renders_cross_task_guidance_as_reusable_heuristics():
    pkg = {
        "workstream_name": "arc_solver",
        "cross_task_bundle": {
            "transferable_insights": ["Treat sparse colors as anchors relative to the dominant background."],
            "confirmed_constraints": ["Square grids can still change size between train and test."],
            "pitfalls": ["Do not reread every train pair without a new hypothesis."],
            "checks": ["Compare connected components before submitting."],
            "next_steps": ["Try stripe/border interpretations before brute-force color replacement."],
            "evidence_post_ids": [9, 10],
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="arc_task", task_source="arc")
    assert "Cross-task patterns" in md
    assert "### Reusable heuristics" in md
    assert "### Shared constraints" in md
    assert "### Shared pitfalls" in md
    assert "### Shared checks" in md
    assert "### Shared next moves" in md
    assert "### Patterns" not in md


def test_evidence_post_ids_skip_malformed_values():
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {
            "transferable_insights": ["Use lib X"],
            "evidence_post_ids": [None, "#", "abc", "2, #999", -1, 0, True, "4", 5],
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert "Evidence posts: #4, #5" in md
    assert "#999" not in md
    assert "#abc" not in md
    assert "#None" not in md


_SENT = "Reflect each panel across the divider rows and merge the resulting colors."


def test_bundle_item_at_480_renders_uncut():
    """#690 follow-up: render caps must match the 480-char distill item cap —
    a ~470-char insight must reach MEMORY.md whole, not sliced at 300."""
    text = " ".join([_SENT] * 6) + " Check panel alignment."
    assert 460 <= len(text) <= 480  # fixture sanity
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {"transferable_insights": [{"text": text}]},
        "cross_task_bundle": {"transferable_insights": [{"text": text}]},
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert md.count(text) == 2  # per-task AND cross-task sections, uncut


def test_bundle_item_above_cap_truncates_at_boundary():
    text = " ".join([_SENT] * 8)
    assert len(text) > 520  # fixture sanity
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {"transferable_insights": [{"text": text}]},
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    [line] = [ln for ln in md.splitlines() if ln.endswith("...(truncated)")]
    body = line[len("- ") : -len("...(truncated)")]
    assert body.endswith(".")  # sentence boundary, not a mid-word slice
    assert len(body) + len("...(truncated)") <= 480


def test_external_bundle_item_2000_chars_renders_in_full():
    """arm-G pilot (2026-07-04): a ~2000-char mechanical-excerpts item in an
    externally injected KT bundle (`--seed-per-task-bundles-path`) was cut at
    the 480-char delivery cap — "...(truncated)" landed before any payload
    line rendered, so the agent never saw the load-bearing content even
    though the bundle JSON was verified. External items are the transfer
    treatment itself and must carry their body into MEMORY.md."""
    text = " ".join([_SENT] * 26)
    assert 1800 <= len(text) <= 2100  # fixture sanity: the incident's scale
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {
            "transferable_insights": [{"text": text}],
            "_external_seed_source": "/tmp/bundles.json",
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert text in md
    assert "...(truncated)" not in md


def test_external_bundle_pathological_total_is_bounded_with_marker():
    """A pathological external bundle (10 near-cap items, ~36k chars total)
    must be bounded by the whole-section budget at item granularity, with an
    explicit truncation marker — never silently dropped or mid-item sliced."""
    item_text = " ".join([_SENT] * 48)  # ~3.6k chars, under the per-item cap
    items = [{"text": f"Variant {i}: {item_text}"} for i in range(10)]
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {
            "transferable_insights": items,
            "_external_seed_source": "/tmp/bundles.json",
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    rendered = [i for i in range(10) if f"Variant {i}:" in md]
    assert rendered, "budget must still deliver at least one item"
    assert len(rendered) < 10, "pathological bundle must be bounded"
    assert rendered == list(range(len(rendered))), "items drop from the tail, not the middle"
    assert _EXTERNAL_BUNDLE_TRUNCATION_MARKER in md
    assert "...(truncated)" not in md  # item granularity: no mid-item slice
    assert len(md) < 20_000


def test_internal_bundle_item_2000_chars_still_truncates_at_480():
    """Characterization pin: the INTERNAL per-task path and the cross-task
    path keep the historical 480-char item cap byte-for-byte — the external
    raise must not leak into them."""
    text = " ".join([_SENT] * 26)
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {"transferable_insights": [{"text": text}]},
        "cross_task_bundle": {"transferable_insights": [{"text": text}]},
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert text not in md
    truncated = [ln for ln in md.splitlines() if ln.endswith("...(truncated)")]
    assert len(truncated) == 2  # per-task AND cross-task both still capped
    for ln in truncated:
        assert len(ln) <= len("- ") + 480


def test_external_cross_task_bundle_item_2000_chars_renders_in_full():
    """Same latent gap one flag over from the arm-G incident: bundles injected
    via `--seed-bundle-path` (cross-task external) rendered through the
    cross-task loop's hardcoded 480-char cap. External cross-task items are
    equally the transfer treatment itself and must carry their body into
    MEMORY.md. (Measured: the tb1-kt-ab donor bundle's max item was 240 chars,
    so no past run was bitten — this closes the gap before one is.)"""
    text = " ".join([_SENT] * 26)
    assert 1800 <= len(text) <= 2100
    pkg = {
        "workstream_name": "solver",
        "cross_task_bundle": {
            "transferable_insights": [{"text": text}],
            "_external_seed_source": "/tmp/bundle.json",
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert text in md
    assert "...(truncated)" not in md


def test_external_cross_task_bundle_pathological_total_is_bounded_with_marker():
    """A pathological external cross-task bundle must be bounded by the same
    whole-section budget at item granularity, with the explicit marker."""
    item_text = " ".join([_SENT] * 48)
    items = [{"text": f"Variant {i}: {item_text}"} for i in range(10)]
    pkg = {
        "workstream_name": "solver",
        "cross_task_bundle": {
            "transferable_insights": items,
            "_external_seed_source": "/tmp/bundle.json",
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    rendered = [i for i in range(10) if f"Variant {i}:" in md]
    assert rendered, "budget must still deliver at least one item"
    assert len(rendered) < 10, "pathological bundle must be bounded"
    assert rendered == list(range(len(rendered))), "items drop from the tail, not the middle"
    assert _EXTERNAL_BUNDLE_TRUNCATION_MARKER in md
    assert "...(truncated)" not in md
    assert len(md) < 20_000


def test_external_marker_key_never_renders_as_content():
    """The `_external_seed_source` bookkeeping key must not leak into the
    rendered MEMORY.md on either external path."""
    pkg = {
        "workstream_name": "solver",
        "per_task_bundle": {
            "transferable_insights": [{"text": "pt"}],
            "_external_seed_source": "/tmp/secret-path/bundles.json",
        },
        "cross_task_bundle": {
            "transferable_insights": [{"text": "ct"}],
            "_external_seed_source": "/tmp/secret-path/bundle.json",
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert "/tmp/secret-path" not in md
    assert "_external_seed_source" not in md


def test_missing_bundles_render_none_available():
    pkg = {"workstream_name": "solver"}
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    # Either both sections with "None available" OR both sections absent.
    # We accept either, but if the sections are rendered, they must say
    # "None available".
    if "Task-specific guidance" in md:
        assert "None available" in md


def test_related_summaries_redact_hidden_verifier_trace_fragments():
    pkg = {
        "related_summaries": [
            {
                "task_id": "old",
                "approach": (
                    "TB2 attempt summary: reward=0.0; "
                    "failure_signature=secretcanarytoken_old_failure; "
                    "verifier_clues=['secretcanarytoken_old_clue']; "
                    "tool_count=3"
                ),
                "outcome": "unresolved",
                "score": 0.0,
                "lessons": "keep reward=0.0; verifier_stdout_tail=secretcanarytoken_tail",
            }
        ]
    }

    md = seed_package_to_memory_md(pkg, current_task_id="t1")

    assert "secretcanarytoken" not in md
    assert "failure_signature" not in md
    assert "verifier_clues" not in md
    assert "verifier_stdout_tail" not in md
    assert "reward=0.0" in md
    assert "tool_count=3" in md


def test_prior_attempts_render_best_attempt_and_stagnation_warning():
    pkg = {
        "workstream_name": "solver",
        "prior_attempts": [
            {
                "gen": 3,
                "eval_results": {
                    "native_score": 0.0,
                    "resolved": False,
                    "instance_report": {
                        "tests_status": {
                            "FAIL_TO_PASS": {
                                "success": ["test_a"],
                                "failure": ["test_b", "test_c"],
                            },
                            "PASS_TO_PASS": {"failure": ["test_existing"]},
                        }
                    },
                },
                "final_model_output": "diff --git a/src/core.py b/src/core.py\n",
                "full_memory_trace_condensed": "Outcome: partial.",
                "task_specific_insights": ["src/core.py path still misses test_b"],
            },
            {
                "gen": 2,
                "eval_results": {
                    "native_score": 0.0,
                    "resolved": False,
                    "instance_report": {
                        "tests_status": {
                            "FAIL_TO_PASS": {
                                "success": [],
                                "failure": ["test_a", "test_b", "test_c"],
                            },
                            "PASS_TO_PASS": {"failure": []},
                        }
                    },
                },
                "final_model_output": "diff --git a/src/core.py b/src/core.py\n",
                "full_memory_trace_condensed": "Outcome: unresolved.",
            },
        ],
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert "Best attempt so far" in md
    assert "passed 1/3" in md
    # Anonymized counts — exact test identifiers must NOT appear in the
    # structured seeding fields (leak-fix D).  Note: "test_b" appears in the
    # task_specific_insights freetext ("misses test_b") which is agent-authored
    # prose, not eval-signal emission — we only gate the structured list output.
    assert "target tests to preserve: test_a" not in md
    assert "remaining failing tests: test_b; test_c" not in md
    assert "regressed tests to preserve: test_existing" not in md
    assert "target tests to preserve" in md
    assert "previously-passing target test" in md
    assert "remaining failing tests" in md
    assert "target test(s) still failing" in md
    assert "regressed tests to preserve" in md
    assert "previously-passing test(s) now failing" in md
    assert "Preserve-best gate" in md
    assert "smallest targeted change" in md
    assert "Stagnation warning" in md
    assert "src/core.py" in md
    assert "Anti-stagnation gate" in md
    assert "earliest live branch" in md


def test_prior_attempts_surface_zero_observed_swebench_tests():
    pkg = {
        "workstream_name": "solver",
        "prior_attempts": [
            {
                "gen": 1,
                "eval_results": {
                    "native_score": 0.0,
                    "resolved": False,
                    "instance_report": {
                        "tests_status": {
                            "observed_count": 0,
                            "FAIL_TO_PASS": {"success": [], "failure": []},
                            "PASS_TO_PASS": {"failure": []},
                        }
                    },
                },
            }
        ],
    }

    md = seed_package_to_memory_md(pkg, current_task_id="t1")

    assert "no benchmark test rows were observed" in md
    assert "build/parser output" in md


def test_prior_attempt_touched_files_normalize_workspace_repo_prefix():
    pkg = {
        "workstream_name": "solver",
        "prior_attempts": [
            {
                "gen": 2,
                "eval_results": {"native_score": 0.0, "resolved": False},
                "final_model_output": "diff --git a/repo/src/core.py b/repo/src/core.py\n",
            },
            {
                "gen": 1,
                "eval_results": {"native_score": 0.0, "resolved": False},
                "final_model_output": "diff --git a/src/core.py b/src/core.py\n",
            },
        ],
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert "Recent unresolved attempts touched the same file(s): src/core.py" in md
    assert "repo/src/core.py" not in md


def test_standard_per_task_bundle_renders_guidance_section():
    pkg = {
        "per_task_bundle": {
            "transferable_insights": ["Use lib X"],
            "pitfalls": [],
            "checks": [],
            "evidence_post_ids": [],
        },
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert "Task-specific guidance" in md


def test_removed_alt_format_bundles_do_not_render_empty_sections():
    pkg = {
        "per_task_bundle": {"format": "ledger", "task_facts": [{"text": "legacy fact"}]},
        "cross_task_bundle": {"format": "motifs", "motifs": [{"name": "legacy"}]},
    }
    md = seed_package_to_memory_md(pkg, current_task_id="t1")
    assert "Task-specific guidance" not in md
    assert "Cross-task patterns" not in md
    assert "legacy fact" not in md
