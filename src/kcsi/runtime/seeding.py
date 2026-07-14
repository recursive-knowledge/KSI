from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..distillation.per_task import truncate_at_boundary
from ..memory.parity import redact_solver_hidden_text
from ..models import TaskSpec
from ..tasks.registry import resolve_source

# Generous runaway backstop on insight text rendered into the seed prompt. The
# prompt-level guidance targets ~2000 chars, so 8000 is 4x that — a well-formed
# insight is never clipped, but a single insight that ignores the prompt cannot
# flood the seed (this body is rendered for up to 12 insights, per recipient
# agent, every generation). Storage stays verbatim (DB rows are cheap); only the
# prompt-injection surface is bounded.
_INSIGHT_SEED_MAX_CHARS = 8000

# External per-task KT bundles (--seed-per-task-bundles-path) are deliberately
# injected experiment payloads: their items ARE the transfer treatment, so the
# per-item cap is a runaway backstop, not an editorial trim — the same policy
# as _INSIGHT_SEED_MAX_CHARS above. 4000 = 2x a ~2000-char item (the delivery
# cap must not silently truncate load-bearing content), and 2x the ~2000-char
# prompt-level guidance target, while still letting >=4
# max-size items fit under the whole-section budget below. Applies to BOTH
# external channels (--seed-per-task-bundles-path per-task and
# --seed-bundle-path cross-task, each stamped `_external_seed_source` at
# inject time). Internal renders keep the historical 480-char cap.
_EXTERNAL_BUNDLE_ITEM_MAX_CHARS = 4000
# Whole-section ceiling across all rendered external per-task items so a
# pathological bundle cannot flood the seed context. Enforced at item
# granularity (whole items dropped from the tail, never mid-item slices) with
# an explicit marker when anything is dropped.
_EXTERNAL_BUNDLE_TOTAL_MAX_CHARS = 16000
_EXTERNAL_BUNDLE_TRUNCATION_MARKER = (
    "[external KT bundle truncated: section render budget reached; remaining items omitted]"
)


def is_canonical_distillation_bundle(bundle: Any) -> bool:
    """Return whether a distillation bundle uses the surviving bundle schema."""
    if not isinstance(bundle, dict) or not bundle:
        return False
    fmt = str(bundle.get("format") or "").strip().lower()
    if fmt and fmt != "bundle":
        return False
    return True


def _extract_approach_excerpt(text: str, max_chars: int = 300) -> str:
    """Extract a substantive excerpt from model output, skipping preamble.

    Mirrors ``kcsi.orchestrator.engine._extract_approach_excerpt``. Duplicated
    here to avoid an engine→runtime circular import.
    """
    if not text:
        return ""
    lines = text.split("\n")
    skip_patterns = re.compile(
        r"^(I'll|I'm |Let me |I need to |I should |I want to |I have |"
        r"Now |OK|Alright|First|Here's|Looking at|Container exited|"
        r"\s*$)",
        re.IGNORECASE,
    )
    start = 0
    for i, line in enumerate(lines):
        if not skip_patterns.match(line.strip()):
            start = i
            break
    excerpt = " ".join(l.strip() for l in lines[start:] if l.strip())
    if not excerpt:
        excerpt = " ".join(l.strip() for l in lines if l.strip())
    return excerpt[:max_chars].strip()


def safe_read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def repo_source_path(task: TaskSpec) -> str:
    metadata = task.metadata or {}
    value = metadata.get("repo_path")
    if isinstance(value, str):
        return str(Path(value).resolve())
    return ""


_SAFE_SWEBENCH_INSTANCE_RE = re.compile(r"[A-Za-z0-9_.-]+")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_swebench_instance_id(value: Any) -> str:
    instance_id = str(value or "").strip()
    if not instance_id or instance_id in {".", ".."}:
        return ""
    if not _SAFE_SWEBENCH_INSTANCE_RE.fullmatch(instance_id):
        return ""
    return instance_id


def _swebench_scripts_dir(metadata: dict[str, Any]) -> Path:
    # Only the operator's environment may override the scripts root.
    # Dataset-controlled metadata keys (`swebench_scripts_dir`, `scripts_dir`)
    # are intentionally NOT honored — accepting them lets a malicious dataset
    # row point the seeder at any path the kcsi user can read, and the
    # contents of <root>/<instance_id>/run_script.sh would then be embedded
    # into the agent's TASK.md and executed inside the container.
    configured = os.environ.get("SWEBENCH_PRO_SCRIPTS_DIR")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser().resolve()
    return _repo_root() / "benchmarks" / "swebench_pro" / "evaluator" / "run_scripts"


def _swebench_task_files(task: TaskSpec) -> dict[str, str]:
    metadata = task.metadata or {}
    source = str(metadata.get("task_source") or "").strip().lower()
    if source != "swebench_pro":
        return {}

    # In upstream-strict mode (swebench_pro_seed_tests=False, the default),
    # do NOT copy run_script.sh into the agent's workspace.  The script embeds
    # the exact test names that define the eval signal, which would let the
    # agent read them out of /workspace/task/workspace/run_script.sh.
    # Only copy it when the caller has opted into DGM-equivalent seeded mode.
    seed_test_files: bool = bool(metadata.get("swebench_pro_seed_tests", False))
    if not seed_test_files:
        return {}

    instance_id = _safe_swebench_instance_id(metadata.get("instance_id") or task.id)
    if not instance_id:
        return {}

    scripts_dir = _swebench_scripts_dir(metadata)
    instance_dir = scripts_dir / instance_id
    out: dict[str, str] = {}
    for filename in ("run_script.sh", "parser.py"):
        path = instance_dir / filename
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if content.strip():
            out[filename] = content
    return out


def workspace_task_files(task: TaskSpec) -> dict[str, str]:
    metadata = task.metadata or {}
    raw = metadata.get("task_files")
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            name = str(k or "").strip()
            if not name:
                continue
            if not isinstance(v, str):
                continue
            content = v.strip()
            if not content:
                continue
            out[name] = content
    for name, content in _swebench_task_files(task).items():
        out.setdefault(name, content)
    return out


def format_query_records_md(
    records: list[dict],
    *,
    task_id: str = "",
    raw_mode: bool = False,
) -> str:
    if not records:
        return ""
    lines = [f"\n\n## Prior Attempts on {task_id or 'this task'}"]
    n_attempts = sum(1 for rec in records if not rec.get("insight_only"))
    lines.append(f"({n_attempts} prior attempt(s), most recent first)\n")

    if raw_mode:
        # Minimal-signal channel for the dumb-distiller ablation: render only
        # gen + score + resolved + a narration-stripped approach_excerpt
        # (1000-char slice of the agent's output, with reasoning/preamble lines
        # skipped). No stderr, no insights, no heuristic best/stagnation gates,
        # no distilled bundles. This is what we want the next-gen agent to see
        # of prior attempts when we've turned off every model-generated
        # reflection layer.
        per_attempt_approach = 1000
        for rec in records[:8]:
            if rec.get("insight_only"):
                continue
            gen = rec.get("gen", "?")
            er = rec.get("eval_results") if isinstance(rec.get("eval_results"), dict) else {}
            score = er.get("native_score")
            resolved = bool(er.get("resolved", False))
            status = "SOLVED" if resolved else f"score={score}"
            lines.append(f"### Gen {gen} ({status})")
            mo = str(rec.get("model_output") or rec.get("final_model_output") or "")
            approach = _extract_approach_excerpt(mo, max_chars=per_attempt_approach)
            if approach:
                lines.append("Approach excerpt:")
                lines.append("```")
                lines.append(approach)
                lines.append("```")
        return "\n".join(lines)

    best = _best_attempt_summary(records)
    if best:
        lines.append("### Best attempt so far")
        lines.extend(best)

    stagnation = _stagnation_summary(records)
    if stagnation:
        lines.append("\n### Stagnation warning")
        lines.extend(stagnation)

    max_attempt_chars = 1_800
    used_attempt_chars = 0

    seen_insight_words: list[set[str]] = []
    unique_insights: list[str] = []

    for rec in records:
        # Insight-only records have no attempt body; render no empty gen header,
        # but still surface their insight text in the Key Insights section below.
        if not rec.get("insight_only"):
            gen = rec.get("gen", "?")
            score = rec.get("eval_results", {}).get("native_score")
            resolved = rec.get("eval_results", {}).get("resolved", False)
            # Older DB rows may have baked hidden-output fragments
            # (verifier_clues=/failure_signature=/...) into the condensed trace.
            # Redact at the seed read boundary like the distillation path does
            # (prompts._fmt_attempts) before it reaches the next-gen agent.
            condensed = _sanitize_seed_excerpt(
                redact_solver_hidden_text(rec.get("full_memory_trace_condensed", "")),
                max_chars=280,
            )
            status = "SOLVED" if resolved else f"score={score}"

            lines.append(f"### Gen {gen} ({status})")
            if condensed and used_attempt_chars < max_attempt_chars:
                remaining = max_attempt_chars - used_attempt_chars
                if len(condensed) > remaining:
                    # This slice is re-cutting an already boundary-truncated
                    # excerpt against the shared cross-generation budget, so it
                    # can chop mid-word/mid-sentence with no marker of its own.
                    # Reuse the same "...(truncated)" suffix as
                    # _sanitize_seed_excerpt for a consistent signal, budgeted
                    # so the total never exceeds ``remaining``. If there isn't
                    # even enough headroom to fit the marker, omit the excerpt
                    # entirely rather than emit an unmarked silent fragment
                    # (mirrors the exhausted-budget omission above).
                    suffix = "...(truncated)"
                    if remaining > len(suffix):
                        excerpt = condensed[: remaining - len(suffix)].rstrip() + suffix
                    else:
                        excerpt = ""
                else:
                    excerpt = condensed
                excerpt = excerpt.strip()
                if excerpt:
                    lines.append(excerpt)
                    used_attempt_chars += len(excerpt)

        for ins in rec.get("task_specific_insights") or []:
            text = (
                str(ins).strip()
                if isinstance(ins, str)
                else str(ins.get("text", "")).strip()
                if isinstance(ins, dict)
                else ""
            )
            # Insights are the primary transfer signal — carry the stored body
            # into the seed context (scrub + a generous runaway backstop that
            # never touches a well-formed insight; see _INSIGHT_SEED_MAX_CHARS).
            text = _sanitize_seed_excerpt(text, max_chars=_INSIGHT_SEED_MAX_CHARS)
            if not text:
                continue
            words = set(text.lower().split())
            is_dup = False
            for existing in seen_insight_words:
                overlap = len(words & existing)
                smaller = min(len(words), len(existing))
                if smaller > 0 and overlap / smaller >= 0.6:
                    is_dup = True
                    break
            if not is_dup:
                seen_insight_words.append(words)
                unique_insights.append(text)

    if unique_insights:
        lines.append("\n### Key Insights (deduplicated)")
        # Keep one generation's worth of bullets intact (12 ≈ two distilled
        # bundles); insights are the primary transfer signal so each carries its
        # full body (already scrubbed + backstopped above).
        for i, ins in enumerate(unique_insights[:12], 1):
            lines.append(f"{i}. {ins}")

    return "\n".join(lines)


def _best_attempt_summary(records: list[dict]) -> list[str]:
    attempts = [_attempt_summary(rec) for rec in records]
    attempts = [a for a in attempts if a]
    if not attempts:
        return []

    def rank(item: dict[str, Any]) -> tuple[float, int, float, int, int]:
        total = int(item.get("f2p_total") or 0)
        passed = int(item.get("f2p_passed") or 0)
        ratio = passed / total if total else 0.0
        p2p_failed = int(item.get("p2p_failed") or 0)
        score = float(item.get("score") or 0.0)
        resolved = 1 if item.get("resolved") else 0
        generation = int(item.get("gen") or 0)
        return (resolved, score, ratio, passed, -p2p_failed, generation)

    best = max(attempts, key=rank)
    lines = [
        f"- gen={best.get('gen')} score={best.get('score')} resolved={bool(best.get('resolved'))}",
    ]
    total = int(best.get("f2p_total") or 0)
    if total:
        lines.append(f"- target tests: passed {best.get('f2p_passed')}/{total}; remaining {best.get('f2p_failed')}")
    observed_count = best.get("observed_count")
    if observed_count == 0:
        lines.append(
            "- verification signal: no benchmark test rows were observed; "
            "treat the build/parser output as the next thing to inspect."
        )
    # Do NOT surface test names — they are evaluation signal that must not be
    # forwarded to the agent.  Emit anonymized counts only so the swarm still
    # knows whether progress was made without leaking F2P/P2P test identifiers.
    passed_count = len(best.get("passed_tests") or [])
    if passed_count:
        lines.append(f"- target tests to preserve: {passed_count} previously-passing target test(s)")
    if int(best.get("p2p_failed") or 0):
        lines.append(f"- regression risk: {best.get('p2p_failed')} previously passing test(s) failed")
    files = best.get("files") or []
    if files:
        lines.append("- touched files: " + ", ".join(files[:5]))
    remaining_count = len(best.get("remaining_tests") or [])
    if remaining_count:
        lines.append(f"- remaining failing tests: {remaining_count} target test(s) still failing")
    regressions_count = len(best.get("p2p_failure_tests") or [])
    if regressions_count:
        lines.append(f"- regressed tests to preserve: {regressions_count} previously-passing test(s) now failing")
    lines.extend(
        [
            "- Preserve-best gate: treat this as the baseline to improve, not a patch to discard.",
            "- Next attempt should preserve the passed behavior above, then make the smallest targeted change for the remaining failure.",
            "- If you replace the approach, first explain why it preserves the same passed behavior or why the prior apparent pass was misleading.",
        ]
    )
    return lines


def _stagnation_summary(records: list[dict]) -> list[str]:
    attempts = [_attempt_summary(rec) for rec in records]
    attempts = [a for a in attempts if a]
    if len(attempts) < 2:
        return []
    recent = attempts[:3]
    unresolved_recent = [a for a in recent if not a.get("resolved") and float(a.get("score") or 0.0) < 1.0]
    if len(unresolved_recent) < 2:
        return []
    file_sets = [set(a.get("files") or []) for a in unresolved_recent if a.get("files")]
    if len(file_sets) < 2:
        return []
    common = set.intersection(*file_sets[:2])
    if not common:
        return []
    common_list = sorted(common)[:5]
    return [
        "- Recent unresolved attempts touched the same file(s): " + ", ".join(common_list),
        "- Anti-stagnation gate: before editing those files again, inspect or run one exact remaining failing test and map it to the earliest live branch.",
        "- Do not submit another same-file patch unless you can name the precise branch, invariant, or data shape the prior patch missed.",
        "- If that evidence is not clear, switch to a different layer, entry point, or invariant check before patching.",
    ]


def _attempt_summary(rec: dict) -> dict[str, Any]:
    # Insight-only records carry standalone insight text, not a real attempt;
    # they must not be ranked as a "best attempt" or counted toward stagnation.
    if rec.get("insight_only"):
        return {}
    eval_results = rec.get("eval_results") if isinstance(rec.get("eval_results"), dict) else {}
    score = eval_results.get("native_score")
    if score is None:
        score = rec.get("score")
    resolved = bool(eval_results.get("resolved"))
    report = eval_results.get("instance_report") if isinstance(eval_results.get("instance_report"), dict) else {}
    tests_status = report.get("tests_status") if isinstance(report.get("tests_status"), dict) else {}
    f2p = tests_status.get("FAIL_TO_PASS") if isinstance(tests_status.get("FAIL_TO_PASS"), dict) else {}
    p2p = tests_status.get("PASS_TO_PASS") if isinstance(tests_status.get("PASS_TO_PASS"), dict) else {}
    f2p_success = [str(x) for x in (f2p.get("success") or [])]
    f2p_failure = [str(x) for x in (f2p.get("failure") or [])]
    p2p_failure = [str(x) for x in (p2p.get("failure") or [])]
    observed_count = tests_status.get("observed_count")
    model_output = str(rec.get("final_model_output") or rec.get("model_output") or "")
    files = _diff_files(model_output)
    return {
        "gen": int(rec.get("gen") or rec.get("generation") or 0),
        "score": score,
        "resolved": resolved,
        "f2p_passed": len(f2p_success),
        "f2p_total": len(f2p_success) + len(f2p_failure),
        "f2p_failed": len(f2p_failure),
        "p2p_failed": len(p2p_failure),
        "observed_count": observed_count,
        "passed_tests": [_sanitize_seed_excerpt(x, max_chars=140) for x in f2p_success],
        "remaining_tests": [_sanitize_seed_excerpt(x, max_chars=140) for x in f2p_failure],
        "p2p_failure_tests": [_sanitize_seed_excerpt(x, max_chars=140) for x in p2p_failure],
        "files": files,
    }


def _diff_files(model_output: str) -> list[str]:
    seen: set[str] = set()
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/", model_output or "", flags=re.M):
        path = match.group(1).strip()
        if path.startswith("repo/"):
            path = path[len("repo/") :]
        if path and path not in seen:
            seen.add(path)
            files.append(path)
    return files


def _format_evidence_post_ids(bundle: dict[str, Any]) -> str:
    raw = bundle.get("evidence_post_ids") or []
    if not isinstance(raw, list):
        return ""
    out: list[str] = []
    for item in raw[:20]:
        if isinstance(item, bool):
            continue
        try:
            post_id = int(item)
        except (TypeError, ValueError):
            continue
        if post_id > 0:
            out.append(f"#{post_id}")
    if len(out) > 12:
        shown = ", ".join(out[:12])
        return f"{shown}, ... (+{len(out) - 12} more)"
    return ", ".join(out)


def seed_package_to_memory_md(
    seed_package: Any,
    *,
    current_task_id: str = "",
    task_source: str = "",
    raw_mode: bool = False,
) -> str:
    if not isinstance(seed_package, dict) or not seed_package:
        return ""

    lines = ["# MEMORY Seed", ""]
    name = str(seed_package.get("workstream_name") or "").strip()
    desc = _sanitize_seed_excerpt(seed_package.get("workstream_description") or "", max_chars=420)
    if name:
        lines.append(f"- focus: {name}")
    if desc:
        lines.append(f"- description: {desc}")

    if raw_mode:
        # Raw-attempts ablation: show ONLY the prior raw attempts on this task.
        # Skip insight_bundle (cross-task lessons), per_task_bundle, cross_task_bundle,
        # and related_summaries — every distillation/reflection layer is bypassed.
        cur = str(current_task_id or "").strip()
        prior = seed_package.get("prior_attempts")
        if isinstance(prior, list) and prior:
            lines.append(format_query_records_md(prior, task_id=cur or "this task", raw_mode=True))
        return "\n".join(lines).strip()

    bundle = seed_package.get("insight_bundle")
    related_task_ids: list[str] = []
    if isinstance(bundle, list) and bundle:
        lines.append("- insight_bundle:")
        for item in bundle[:10]:
            if not isinstance(item, dict):
                continue
            text = _sanitize_seed_excerpt(item.get("text") or "", max_chars=260)
            title = _sanitize_seed_excerpt(item.get("title") or "", max_chars=80)
            asset_type = str(item.get("asset_type") or "").strip()
            conf = str(item.get("confidence") or "medium").strip()
            source_task = str(item.get("source_task_id") or "").strip()
            if source_task and source_task not in related_task_ids:
                related_task_ids.append(source_task)
            if text:
                prefix_parts = [f"[{conf}]"]
                if asset_type:
                    prefix_parts.append(asset_type)
                if title:
                    prefix_parts.append(title)
                prefix = " ".join(prefix_parts)
                lines.append(f"  - {prefix}: {text}")
    cur = str(current_task_id or "").strip()
    if cur:
        related_task_ids = [tid for tid in related_task_ids if tid != cur]
        lines.extend(
            [
                "",
                "## Task ID Reference",
                f"- current: {cur}",
                "- related:",
            ]
        )
        if related_task_ids:
            lines.extend([f"  - {tid}" for tid in related_task_ids[:5]])
        else:
            lines.append("  - (none)")

    # Format prior attempts from enriched seed package
    prior = seed_package.get("prior_attempts")
    if isinstance(prior, list) and prior:
        lines.append(format_query_records_md(prior, task_id=cur or "this task"))

    # Format related summaries from enriched seed package
    related = seed_package.get("related_summaries")
    if isinstance(related, list) and related:
        lines.append("\n## Related Tasks")
        for row in related:
            row_tid = str(row.get("task_id") or "unknown")
            approach = _sanitize_seed_excerpt(redact_solver_hidden_text(row.get("approach") or ""), max_chars=200)
            outcome = _sanitize_seed_excerpt(redact_solver_hidden_text(row.get("outcome") or ""), max_chars=200)
            score = row.get("score", "?")
            lessons = _sanitize_seed_excerpt(redact_solver_hidden_text(row.get("lessons") or ""), max_chars=300)
            lines.append(f"- **{row_tid}**: approach={approach}, outcome={outcome}, score={score}")
            if lessons and lessons != "[]":
                lines.append(f"  lessons: {lessons}")

    _seed_spec = resolve_source(task_source)
    is_arc = _seed_spec is not None and _seed_spec.prompt_kind == "arc"

    # Task-specific guidance (distilled per-task bundle for this task).
    per_task = seed_package.get("per_task_bundle")
    if is_canonical_distillation_bundle(per_task):
        lines.append("")
        if per_task.get("_external_seed_source"):
            # Externally injected KT bundle (--seed-per-task-bundles-path):
            # distilled from OTHER donor tasks, so the same-task framing below
            # would be epistemically false. Mislabeled foreign hints degrade
            # the test-feedback repair loop via over-trust.
            lines.extend(
                [
                    "## Possibly-relevant patterns from OTHER tasks (external knowledge transfer)",
                    "These items were distilled from DIFFERENT exercises, not from prior work on this task. "
                    "Treat them as optional background patterns.",
                    "If anything here conflicts with this task's statement, your own observations, "
                    "or official test output, IGNORE the pattern and trust the task evidence.",
                ]
            )
        else:
            lines.append("## Task-specific guidance (distilled from previous attempts and forum evidence on this task)")
        evidence = _format_evidence_post_ids(per_task)
        if evidence:
            lines.append(f"Evidence posts: {evidence}")
        if is_arc:
            arc_per_task_fields = [
                ("confirmed_constraints", "Hard constraints"),
                ("rejected_hypotheses", "Rejected rules"),
                ("pitfalls", "Do not repeat"),
                ("checks", "Verification focus"),
                ("next_steps", "Next hypotheses"),
                ("transferable_insights", "Observed task insights"),
            ]
        else:
            arc_per_task_fields = [
                ("transferable_insights", "Insights"),
                ("confirmed_constraints", "Confirmed constraints"),
                ("rejected_hypotheses", "Rejected hypotheses"),
                ("pitfalls", "Pitfalls (do not re-attempt)"),
                ("checks", "Checks"),
                ("next_steps", "Next steps"),
            ]
        # External bundles carry the transfer treatment itself: give each item
        # a generous backstop plus a whole-section budget, so a large item is
        # not silently cut at the internal cap and hidden from the agent. The
        # internal same-task path stays byte-identical at the historical
        # 480-char cap.
        is_external = bool(per_task.get("_external_seed_source"))
        item_max_chars = _EXTERNAL_BUNDLE_ITEM_MAX_CHARS if is_external else 480
        remaining = _EXTERNAL_BUNDLE_TOTAL_MAX_CHARS if is_external else None
        budget_exhausted = False
        for k, label in arc_per_task_fields:
            if budget_exhausted:
                break
            items = per_task.get(k) or []
            if items:
                header_idx = len(lines)
                lines.append(f"### {label}")
                rendered_any = False
                for item in items[:10]:
                    rendered = _render_bundle_item(item, max_chars=item_max_chars)
                    if remaining is not None:
                        cost = sum(len(ln) + 1 for ln in rendered)
                        if cost > remaining:
                            budget_exhausted = True
                            break
                        remaining -= cost
                    lines.extend(rendered)
                    rendered_any = True
                if budget_exhausted and not rendered_any:
                    del lines[header_idx]
        if budget_exhausted:
            lines.append(_EXTERNAL_BUNDLE_TRUNCATION_MARKER)

    # Cross-task patterns (distilled shared-room discussion from last generation)
    cross = seed_package.get("cross_task_bundle")
    if is_canonical_distillation_bundle(cross):
        lines.append("")
        kt_mode = str(seed_package.get("_kt_mode") or "").strip().lower()
        kt_task_source = str(seed_package.get("_kt_task_source") or "").strip().lower()
        _kt_spec = resolve_source(kt_task_source)
        kt_is_polyglot = _kt_spec is not None and _kt_spec.prompt_kind == "polyglot"
        adapter_memo = seed_package.get("kt_adapter_memo")
        if kt_mode == "adapter_transfer" and isinstance(adapter_memo, dict) and adapter_memo:
            if kt_is_polyglot:
                lines.extend(
                    [
                        "## Task-conditioned transfer memo",
                        "This memo was derived from prior distilled coding-task knowledge for the current task.",
                        "Use it actively before you choose an implementation direction.",
                        "This memo was written WITHOUT access to this task's tests, so it does not know the exact "
                        "function signature, parameter type, or input/output representation. Derive those yourself "
                        "from the starter file and test stubs before coding — never assume them from this memo.",
                        "Process:",
                        "1. Use the candidate heuristics below as possible implementation strategies, not guarantees.",
                        "2. Use pitfalls to reject risky or incompatible approaches early.",
                        "3. Use the checks before you finalize code changes or submit an answer.",
                        "4. If the memo conflicts with the current task statement, tests, or repository evidence, trust the current task evidence.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "## Task-conditioned transfer memo",
                        "This memo was derived from prior distilled ARC knowledge for the current task.",
                        "Use it actively before you choose a transformation.",
                        "Process:",
                        "1. Use the candidate heuristics below as possible rules, not guarantees.",
                        "2. Use pitfalls to reject bad hypotheses early.",
                        "3. Use the checks before you finalize the grid.",
                        "4. If the memo conflicts with the current train pairs, trust the current train pairs.",
                    ]
                )
            memo_source = str(adapter_memo.get("_memo_source") or "").strip()
            if memo_source:
                lines.append(f"- memo source: {memo_source}")
            constraints = adapter_memo.get("relevant_constraints") or []
            if constraints:
                lines.append("")
                if kt_is_polyglot:
                    lines.append("### Constraints or contracts to respect")
                else:
                    lines.append("### Constraints to respect")
                for item in constraints[:3]:
                    lines.append(f"- {str(item).strip()[:220]}")
            heuristics = adapter_memo.get("relevant_heuristics") or []
            if heuristics:
                lines.append("")
                if kt_is_polyglot:
                    lines.append("### Candidate implementation heuristics")
                else:
                    lines.append("### Candidate heuristics")
                for item in heuristics[:4]:
                    lines.append(f"- {str(item).strip()[:240]}")
            pitfalls = adapter_memo.get("pitfalls_to_avoid") or []
            if pitfalls:
                lines.append("")
                lines.append("### Pitfalls to avoid")
                for item in pitfalls[:3]:
                    lines.append(f"- {str(item).strip()[:220]}")
            checks = adapter_memo.get("checks_before_submit") or []
            if checks:
                lines.append("")
                lines.append("### Checks before submission")
                for item in checks[:3]:
                    lines.append(f"- {str(item).strip()[:220]}")
            candidate_plan = str(adapter_memo.get("candidate_plan") or "").strip()
            if candidate_plan:
                lines.append("")
                lines.append("### Candidate plan")
                lines.append(candidate_plan[:500])
            rationale = str(adapter_memo.get("knowledge_use_rationale") or "").strip()
            if rationale:
                lines.append("")
                lines.append("### Why this prior is relevant")
                lines.append(rationale[:500])
        else:
            lines.append("## Cross-task patterns (distilled from agent discussions last generation)")
            evidence = _format_evidence_post_ids(cross)
            if evidence:
                lines.append(f"Evidence posts: {evidence}")
            if is_arc:
                arc_cross_fields = [
                    ("transferable_insights", "Reusable heuristics"),
                    ("confirmed_constraints", "Shared constraints"),
                    ("rejected_hypotheses", "Rejected hypotheses"),
                    ("pitfalls", "Shared pitfalls"),
                    ("checks", "Shared checks"),
                    ("next_steps", "Shared next moves"),
                ]
            else:
                arc_cross_fields = [
                    ("transferable_insights", "Patterns"),
                    ("confirmed_constraints", "Shared constraints"),
                    ("rejected_hypotheses", "Rejected hypotheses"),
                    ("pitfalls", "General pitfalls"),
                    ("checks", "General checks"),
                    ("next_steps", "Next steps"),
                ]
            # External cross-task bundles (--seed-bundle-path) carry the
            # transfer treatment itself: same generous caps as the external
            # per-task path above. Internal cross-task renders stay
            # byte-identical at the historical 480-char cap.
            cross_is_external = bool(cross.get("_external_seed_source"))
            cross_item_max_chars = _EXTERNAL_BUNDLE_ITEM_MAX_CHARS if cross_is_external else 480
            cross_remaining = _EXTERNAL_BUNDLE_TOTAL_MAX_CHARS if cross_is_external else None
            cross_budget_exhausted = False
            for k, label in arc_cross_fields:
                if cross_budget_exhausted:
                    break
                items = cross.get(k) or []
                if items:
                    header_idx = len(lines)
                    lines.append(f"### {label}")
                    rendered_any = False
                    for item in items[:10]:
                        rendered = _render_bundle_item(item, max_chars=cross_item_max_chars)
                        if cross_remaining is not None:
                            cost = sum(len(ln) + 1 for ln in rendered)
                            if cost > cross_remaining:
                                cross_budget_exhausted = True
                                break
                            cross_remaining -= cost
                        lines.extend(rendered)
                        rendered_any = True
                    if cross_budget_exhausted and not rendered_any:
                        del lines[header_idx]
            if cross_budget_exhausted:
                lines.append(_EXTERNAL_BUNDLE_TRUNCATION_MARKER)

    return "\n".join(lines).strip()


_POLICY_ERROR_PATTERNS = (
    re.compile(r"container exited with code 1:\s*400 invalid prompt:.*", re.IGNORECASE),
    re.compile(r"invalid prompt:\s*your prompt was flagged.*", re.IGNORECASE),
    re.compile(r"https?://\S+", re.IGNORECASE),
)


def _render_bundle_item(item: Any, *, max_chars: int = 300) -> list[str]:
    """Render a bundle item as one or more markdown lines.

    Handles two shapes:
    - Legacy: bare string → ``- <text>`` (single line).
    - Structured Insight dict → ``- (<confidence>) <text>`` plus optional
      ``Applies when:``, ``NOT when:``, and ``Evidence:`` lines.

    Empty / whitespace-only items return an empty list (caller should skip).
    """
    if isinstance(item, dict):
        text = _sanitize_seed_excerpt(item.get("text", ""), max_chars=max_chars)
        if not text:
            return []
        confidence = str(item.get("confidence") or "").strip().lower()
        prefix = f"({confidence}) " if confidence in ("high", "medium", "low") else ""
        lines = [f"- {prefix}{text}"]
        applies = _sanitize_seed_excerpt(item.get("applies_when", ""), max_chars=200)
        if applies:
            lines.append(f"  Applies when: {applies}")
        not_applies = _sanitize_seed_excerpt(item.get("does_not_apply_when", ""), max_chars=200)
        if not_applies:
            lines.append(f"  NOT when: {not_applies}")
        evidence = item.get("evidence") or []
        if isinstance(evidence, list) and evidence:
            chunks: list[str] = []
            for ev in evidence[:3]:
                if not isinstance(ev, dict):
                    continue
                tid = str(ev.get("task_id") or "").strip()
                pid = ev.get("post_id")
                quote = _sanitize_seed_excerpt(ev.get("quote", ""), max_chars=140)
                pid_part = f"post#{pid}" if pid is not None else "post#?"
                tid_part = f" ({tid})" if tid else ""
                quote_part = f': "{quote}"' if quote else ""
                chunks.append(f"{pid_part}{tid_part}{quote_part}")
            if chunks:
                lines.append("  Evidence: " + "; ".join(chunks))
        return lines
    text = _sanitize_seed_excerpt(str(item), max_chars=max_chars)
    return [f"- {text}"] if text else []


def _sanitize_seed_excerpt(value: Any, *, max_chars: int | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for pattern in _POLICY_ERROR_PATTERNS:
        text = pattern.sub(
            "Previous run was blocked before model execution; avoid repeating the flagged wording verbatim.",
            text,
        )
    text = re.sub(r"\s+", " ", text).strip()
    # ``max_chars=None`` means scrub-and-collapse only, no length cap. The
    # production insight render passes _INSIGHT_SEED_MAX_CHARS instead, keeping
    # well-formed insights intact while bounding runaway payloads.
    if max_chars is None or len(text) <= max_chars:
        return text
    suffix = "...(truncated)"
    # Boundary-aware cut (sentence end / word boundary) so the rendered item
    # never ends mid-word — matches the distill-side item cap.
    return truncate_at_boundary(text, max_chars - len(suffix)).rstrip() + suffix
