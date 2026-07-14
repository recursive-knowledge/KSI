"""Prompt builders for the 3-round forum flow.

Round 1: Agent-local synthesis (task/meta insights)
Round 2: Social commenting on insight pages
Round 3: Task-mode asset condensation or workstream clustering
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..discussion.concreteness import ANTI_META_BLOCK
from ..models import TaskTrace


@dataclass(frozen=True)
class ForumPromptParts:
    """Cache-stable split of a forum prompt body.

    The V2 forum embedded per-agent / per-generation content
    (prior_posts, native_memory, peer_posts) into the prompt text body
    that downstream consumers attach `cache_control` to. Because the
    Anthropic prompt cache keys on the content hash up to and including
    the cache marker, mutating that body invalidates the cache every
    call (cache_read=0). See the cache_stability invariant.

    The fix: split the prompt into a ``cacheable_prefix`` (agent- and
    generation-invariant content — task IDs, descriptions, round
    instructions, output schemas, MCP tool list) and a
    ``variable_suffix`` (per-agent / per-generation content). Adapters
    that support cache-aware delivery (e.g. ``anthropic_direct_forum``)
    place ``cache_control`` only on a block containing the prefix and
    append the suffix as a separate block.

    Adapters that don't support a split delivery (file-using runners
    that surface the body via TASK.md) can still get the full body via
    :meth:`as_text`.
    """

    cacheable_prefix: str
    variable_suffix: str

    def as_text(self) -> str:
        """Return the full prompt body (prefix + suffix) as a single string.

        Concatenation is the inverse of the split — callers that want the
        full prompt as a single string call ``build_*_discussion_parts(...)
        .as_text()``.
        """
        if not self.variable_suffix:
            return self.cacheable_prefix
        if not self.cacheable_prefix:
            return self.variable_suffix
        return self.cacheable_prefix + self.variable_suffix


# ---------------------------------------------------------------------------
# Prompt-render excerpt caps.
#
# Before: model_output excerpts clipped at 220 chars, task descriptions at 300,
# prior-post snippets at 300, native_memory injection at 8000 (even when the
# user set --native-memory-max-chars=240000), and the cross-task bundle JSON
# clipped to 2000 chars mid-structure.
#
# After: raised to sizes that keep actionable reasoning intact and let CLI
# flags like --native-memory-max-chars actually reach the forum prompt. Keep
# these in sync with distillation/prompts.py input caps — the three together
# form the KT-sweep signal chain.
# ---------------------------------------------------------------------------
_TASK_DESCRIPTION_PREVIEW_CHARS = 1200
# Bumped 1200 → 2000 for V2: posts are now structured JSON post-mortems
# (load_bearing_assumption + evidence + proposed_change + predicted_outcome
# + confidence) that routinely run 1.5-2KB. 1200 was clipping the proposal
# section. Cache-safe (per-item bound, deterministic).
_POST_SNIPPET_CHARS = 2000
_NATIVE_MEMORY_FORUM_INJECT_CHARS = 32000
# Bumped 8000 → 20000 for V2: bundle items are now structured dicts
# (text + applies_when + does_not_apply_when + evidence + confidence)
# at ~500 chars/item × 30 items ≈ 15KB. 8KB was clipping most of the
# bundle JSON before agents could read it.


def _sanitize_agent_output(value: str) -> str:
    """Neutralize forum-protocol keywords in agent-generated text.

    Prevents prompt injection where task descriptions, model outputs, forum
    post bodies, or provenance fields containing literal ``INSIGHT`` or
    ``COMMENT`` on their own line would be parsed as valid forum blocks.
    """
    text = str(value or "")
    # Replace standalone protocol keywords with bracketed versions so the
    # forum block parser will not match them.
    text = re.sub(r"(?m)^(\s*)(INSIGHT|COMMENT)(\s*)$", r"\1[\2]\3", text)
    return text


def _sanitize_inline_value(value: Any, *, max_chars: int = 160) -> str:
    text = _sanitize_agent_output(str(value or "?"))
    text = " ".join(text.splitlines()).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text or "?"


def _build_outcomes_section(
    traces: list[TaskTrace],
) -> str:
    """Return outcomes_text for prompt injection."""
    outcome_lines: list[str] = []
    for t in traces:
        status = t.eval_result.get("status") or t.eval_result.get("swebench_status") or "n/a"
        summary = _sanitize_agent_output((t.model_output or "")[-220:].replace("\n", " "))
        outcome_lines.append(f"- task={t.task_id} score={t.native_score} status={status} summary={summary}")
    return "\n".join(outcome_lines) if outcome_lines else "No tasks completed."


def _build_task_descriptions_section(
    task_descriptions: dict[str, str] | None,
) -> str:
    """Build section with task prompts (problem descriptions) keyed by task_id."""
    if not task_descriptions:
        return ""
    lines: list[str] = []
    for task_id, desc in task_descriptions.items():
        preview = _sanitize_agent_output((desc or "").strip().replace("\n", " "))
        # SWE-bench Pro tickets routinely exceed 300 chars; 1200 keeps the
        # bug-report body visible without ballooning prompt tokens.
        if len(preview) > _TASK_DESCRIPTION_PREVIEW_CHARS:
            preview = preview[:_TASK_DESCRIPTION_PREVIEW_CHARS] + "..."
        lines.extend(
            [
                f"### {task_id}",
                "<task_description>",
                f"{preview}",
                "</task_description>",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_per_task_discussion_parts(
    *,
    agent_id: str,
    generation: int,
    traces: list[TaskTrace],
    task_ids: list[str],
    task_descriptions: dict[str, str] | None = None,
    prior_gen_posts: list[dict] | None = None,
    native_memory: str | None = None,
    round_num: int = 0,
    peer_posts_this_gen: list[dict] | None = None,
) -> ForumPromptParts:
    """Build the per-task post-mortem prompt as a cache-stable split.

    Returns a :class:`ForumPromptParts` whose ``cacheable_prefix`` holds
    agent- and generation-stable content (header, round-specific
    instructions, task ID list, task descriptions, gen-1 disclaimer if
    applicable, instruction body) and whose ``variable_suffix`` holds the
    per-agent / per-round sections (this generation's outcomes,
    prior-generation posts, same-generation peer posts from earlier
    rounds, native memory). Adapters that attach ``cache_control`` to a
    prompt block must place the marker on the prefix block — see the
    cache_stability invariant. Call ``.as_text()`` on the result for the
    full prompt as a single string.

    ``round_num`` and ``peer_posts_this_gen`` mirror
    :func:`build_cross_task_discussion_parts`'s multi-round peer-context
    handling: without them, round 1+ of ``--per-task-forum-rounds``
    reproduced the exact same prompt as round 0, since agents had no way
    to see what their same-generation peers posted in earlier rounds.
    """
    outcomes_text = _build_outcomes_section(traces)

    # Build per-task summary from traces. The task-page list (task_id +
    # this-agent's score) is per-agent but call-stable: it doesn't change
    # between turns within a session, so it can sit safely in the
    # cacheable prefix and still produce a within-session cache hit.
    task_score_map: dict[str, float | None] = {}
    for t in traces:
        task_score_map[t.task_id] = t.native_score

    task_lines: list[str] = []
    for tid in task_ids:
        score = task_score_map.get(tid)
        score_str = "n/a" if score is None else f"{score}"
        desc_line = ""
        if task_descriptions and tid in task_descriptions:
            preview = _sanitize_agent_output((task_descriptions[tid] or "").strip().replace("\n", " "))
            if len(preview) > _TASK_DESCRIPTION_PREVIEW_CHARS:
                preview = preview[:_TASK_DESCRIPTION_PREVIEW_CHARS] + "..."
            desc_line = f"  description: {preview}"
        task_lines.append(f"- {tid} (score: {score_str})")
        if desc_line:
            task_lines.append(desc_line)
    task_pages_text = "\n".join(task_lines) if task_lines else "No tasks to discuss."

    task_desc_section = ""
    if task_descriptions:
        task_desc_text = _build_task_descriptions_section(task_descriptions)
        if task_desc_text:
            task_desc_section = f"""
## Task Descriptions
{task_desc_text}
"""

    prior_posts_section = ""
    posts = prior_gen_posts or []
    if posts:
        prior_lines: list[str] = []
        # Chronological order (ASC by id) — append-only across generations
        # keeps the prompt prefix byte-stable so prompt caching can hit on
        # gen N+1 the same prefix that gen N saw, plus newly-appended posts.
        for post in posts:
            pid = _sanitize_inline_value(post.get("id", "?"))
            aid = _sanitize_inline_value(post.get("agent_id", "?"))
            gen_tag = post.get("generation")
            gen_str = f"gen={gen_tag} " if gen_tag is not None else ""
            raw_text = post.get("text") or post.get("content") or ""
            if isinstance(raw_text, dict):
                raw_text = raw_text.get("text", "") if isinstance(raw_text.get("text", ""), str) else ""
            snippet = _sanitize_agent_output(str(raw_text).strip().replace("\n", " "))
            if len(snippet) > _POST_SNIPPET_CHARS:
                snippet = snippet[:_POST_SNIPPET_CHARS] + "..."
            parent = post.get("parent_id") or post.get("reply_to")
            parent_hint = f" reply_to={_sanitize_inline_value(parent)}" if parent else ""
            prior_lines.append(f"- [id={pid} {gen_str}agent={aid}{parent_hint}] {snippet}")
        prior_posts_section = (
            "\n## Per-task forum posts from ALL prior generations on these task(s)\n"
            "(chronological, oldest → newest. Use `gen=N` to weight recency.)\n" + "\n".join(prior_lines) + "\n"
        )

    # Same-generation peer posts from earlier rounds this generation (round
    # 1+ only). Mirrors build_cross_task_discussion_parts's
    # peer_posts_this_gen: agents in round N can see and reply to what
    # their peers posted in rounds < N of the same generation. Kept out of
    # prior_posts_section (cross-generation only) so the two provenances
    # stay visually distinct.
    peer_posts_section = ""
    peers = peer_posts_this_gen or []
    if peers:
        peer_lines: list[str] = []
        for post in peers:
            pid = _sanitize_inline_value(post.get("id", "?"))
            aid = _sanitize_inline_value(post.get("agent_id", "?"))
            r_tag = post.get("round_num")
            r_str = f" round={r_tag}" if r_tag is not None else ""
            raw_text = post.get("text") or post.get("content") or ""
            if isinstance(raw_text, dict):
                raw_text = raw_text.get("text", "") if isinstance(raw_text.get("text", ""), str) else ""
            snippet = _sanitize_agent_output(str(raw_text).strip().replace("\n", " "))
            if len(snippet) > _POST_SNIPPET_CHARS:
                snippet = snippet[:_POST_SNIPPET_CHARS] + "..."
            peer_lines.append(f"- [id={pid}{r_str} agent={aid}] {snippet}")
        peer_posts_section = (
            f"\n## Peer posts from generation {generation}, round(s) < {round_num}\n"
            "(same-generation peers who posted in an earlier round on these "
            "task(s). Cite via parent_post_id to agree/disagree/extend.)\n" + "\n".join(peer_lines) + "\n"
        )

    native_memory_section = ""
    if native_memory and native_memory.strip():
        # --native-memory-max-chars defaults to 240000, but this render-time
        # clip re-caps it much lower. Raised 8000 → 32000 so users who bump
        # the CLI flag actually see the effect in forum prompts. Still well
        # below context-window budget at current per-turn token rates.
        cleaned = native_memory.strip()[:_NATIVE_MEMORY_FORUM_INJECT_CHARS]
        native_memory_section = f"\n## Native memory notes from your Phase 1 attempt\n{cleaned}\n"

    # Gen-1 hardening: when there are genuinely no prior posts (first
    # generation or an empty prior-posts list), agents have been observed
    # hallucinating citations like "Re: post #1809" and "insights from 50+
    # prior attempts". These hallucinations then get distilled into real
    # seed bundles, amplifying bogus claims across generations. Same-gen
    # peer posts (round 1+) count as real history too — the disclaimer must
    # not tell agents to avoid citing IDs that legitimately exist.
    has_prior_posts = bool(posts) or bool(peers)
    gen1_disclaimer = ""
    if int(generation) <= 1 and not has_prior_posts:
        gen1_disclaimer = (
            "\n## IMPORTANT: First generation — no prior posts exist\n"
            "This is generation 1 and there are NO prior posts on any task.\n"
            '- Do NOT cite post IDs (e.g. do NOT write "Re: post #N" or'
            ' "parent_post_id=N" for any N).\n'
            '- Do NOT reference "insights from prior attempts" or "50+'
            ' prior posts" — no such history exists yet.\n'
            "- Form your own hypotheses from your own Phase 1 attempt and"
            " the notes captured above.\n"
            "- `knowledge(task_id=...)` will return an empty or minimal"
            " page; do not invent content that is not there.\n"
        )

    # Stable instruction body — agent-invariant, generation-invariant.
    # Keeping the JSON schema, MCP tool list, and protocol rules in the
    # prefix maximizes within-session and cross-session cache hit rate
    # (Anthropic prompt cache keys on bytes up to and including the
    # cache_control marker; only stable content can hit on subsequent
    # turns or sibling agents).
    instruction_body = """\
## Available MCP Tools
- knowledge(task_id="...", limit=20) — read prior attempts, prior-gen posts,
  and the per-task distilled bundle. Call this BEFORE posting on a task.
- query(task_id="...", query="...", max_records=8) — semantic search on the
  task page. Call this BEFORE posting on a task.
- forum_post(task_id="...", text="...", parent_post_id=N) — post on a task
  page. Pass parent_post_id=<integer id> to thread a reply to a prior post.
- forum_signal_done() — call exactly once at the end so the phase can exit.

## Required protocol

1. For each of your tasks you MUST call `knowledge(task_id="<task>", limit=20)`
   AND MUST call `query(task_id="<task>", query="<specific evidence>", max_records=8)`
   BEFORE posting. The server rejects forum_post until both are called.
2. Post ONE post-mortem per task you attempted, using the JSON template below.
   `forum_post(task_id="<that task>", text=<JSON>)`.
3. If you are extending or contradicting a specific prior post, you MUST call `forum_post(parent_post_id=<integer id>, task_id="<task>", text=<JSON>)` with the integer id from the "Prior generation posts" or "Peer posts" section above. A prose reference like "Re: post #42" is NOT a threaded reply. Do NOT cite a post id that does not appear in either section.
4. After all post-mortems are posted, you MUST call `forum_signal_done()`
   exactly once.

## Required post-mortem JSON

For each task you attempted, the body of `forum_post(text=...)` MUST be a
single JSON object with exactly these fields:

{
  "load_bearing_assumption": "<the ONE assumption your approach relied on. If you failed, name what was wrong with it. If you succeeded, name why it worked when prior attempts didn't. MUST be concrete — name a specific tool, API, file, data shape, or invariant. Reject framings like 'read more carefully' or 'verify assumptions'.>",
  "evidence": "<a verbatim 1-3 sentence excerpt from your own Phase-1 notes above OR from a prior post that supports the claim. Do not paraphrase.>",
  "evidence_post_id": <integer id from the prior-posts list above, or null if the evidence is from your own Phase-1 notes>,
  "proposed_change_for_next_gen": "<ONE concrete change a next-gen agent should try on THIS task. Must name a file/tool/API/decision-point, not a vibe. Must differ from anything tried in prior attempts. If the search space is genuinely exhausted, set this to 'EXHAUSTED' and explain why in evidence.>",
  "predicted_outcome": "<what you predict happens if the change is made — a falsifiable prediction (e.g. 'tests X and Y will pass; test Z still fails because ...').>",
  "confidence": "high" | "medium" | "low"
}

Rules:
- Exactly one post-mortem per task. Do NOT post multiple per task.
- The whole `forum_post(text=...)` body is the JSON object — no preamble,
  no surrounding prose, just `{ ... }`.
- A vague `load_bearing_assumption` (e.g. "the approach was wrong",
  "needed more testing") will be discarded by the distiller.
- Cite by post_id when referring to prior posts; do not paraphrase quotes.
- Call `forum_signal_done()` when finished and produce no further output.
"""

    # Round-specific directive, mirroring build_cross_task_discussion_parts's
    # round_directive: round 0 is the initial post-mortem; round 1+ points
    # agents at the "Peer posts" suffix section below. round_num is
    # call-stable (fixed for the duration of this call) and identical
    # across every agent dispatched in the same round, so it belongs in
    # the cacheable prefix — it legitimately differs between rounds
    # (cache misses across rounds are expected, same as cross-task).
    if int(round_num) <= 0:
        round_directive = "\n## Your task this round (round 0 — initial post-mortem)\nPost your post-mortem(s) as usual (see the JSON template below).\n"
    else:
        round_directive = (
            f"\n## Your task this round (round {round_num} — respond to peers)\n"
            "Same-generation peer posts from earlier rounds on these task(s) are\n"
            "listed below. If a peer's post-mortem conflicts with or extends your\n"
            "own reasoning, cite it via parent_post_id when you post. Otherwise,\n"
            "post your post-mortem as usual.\n"
        )

    # Cacheable prefix: the agent identity in the header is needed for
    # protocol clarity and is stable per-call (the agent does not change
    # mid-session). Keep it here. Per-generation content is `generation`
    # itself, which is also call-stable, as is `round_num`. The variable
    # suffix below carries all genuinely per-agent / per-round content
    # (this generation's outcomes, prior-gen posts, same-gen peer posts,
    # native memory).
    prefix_parts = [
        "# PER-TASK POST-MORTEM (Phase 2)\n",
        "\n",
        f"You are agent {agent_id} in generation {generation}, per-task forum round {round_num},\n",
        "reflecting on the task(s) you just attempted. Your job is to produce ONE structured\n",
        "post-mortem PER TASK that next-generation agents can act on. This is not a summary\n",
        "of what you did — it is a falsifiable claim about what the load-bearing assumption\n",
        "was, what actually happened, and what to try differently.\n",
        "\n",
        "## Task page(s) you should post on\n",
        "\n",
        f"{task_pages_text}\n",
        f"{task_desc_section}",
        f"{gen1_disclaimer}",
        f"{round_directive}",
        "\n",
        instruction_body,
    ]
    cacheable_prefix = "".join(prefix_parts)

    suffix_parts: list[str] = []
    # Per-agent outcomes (scores, summaries) — depend on this agent's traces.
    suffix_parts.append("## Your task outcomes this generation\n\n")
    suffix_parts.append(f"{outcomes_text}\n")
    if prior_posts_section:
        suffix_parts.append(prior_posts_section)
    if peer_posts_section:
        suffix_parts.append(peer_posts_section)
    if native_memory_section:
        suffix_parts.append(native_memory_section)
    variable_suffix = "".join(suffix_parts)

    return ForumPromptParts(
        cacheable_prefix=cacheable_prefix,
        variable_suffix=variable_suffix,
    )


def build_cross_task_discussion_parts(
    *,
    agent_id: str,
    generation: int,
    round_num: int = 0,
    phase1_context: dict | None = None,
    cross_task_history: list[dict] | None = None,
    peer_posts_this_gen: list[dict] | None = None,
) -> ForumPromptParts:
    """Build the cross-task discussion prompt as a cache-stable split.

    Returns a :class:`ForumPromptParts` whose ``cacheable_prefix`` holds
    content that's stable across agents within a generation/round
    (header, cross-task history, round-specific instructions, MCP tool
    list, protocol) and whose ``variable_suffix`` holds the per-agent /
    per-round-per-agent sections (this agent's Phase-1 context, peer
    posts from earlier rounds in this same generation). Call ``.as_text()``
    on the result for the full prompt as a single string.
    """
    # ---- Prefix (stable across agents within a (generation, round)) ----
    prefix_lines: list[str] = [
        "# CROSS-TASK FORUM (Phase 3)",
        "",
        "This is a multi-generation, multi-round conversation among agents who "
        "each just attempted a different task. Your goal: contribute to the "
        "running discussion of what transfers across tasks. Be concrete, "
        "ground in evidence from your own task or from cited prior posts.",
        "",
    ]

    # Cross-task forum history across prior generations. Rendered at a stable
    # byte position so its prefix is byte-identical across the N agents WITHIN a
    # generation (the caller selects it once per generation and shares it), which
    # is the cache win that fires in practice.
    # NOTE: the forum caller passes an EMPTY cross_task_history —
    # cross-task forum agents see only the current generation's posts and
    # cross-generation knowledge flows forward through distillation → seeding,
    # not raw forum history. So on the live forum path this block renders
    # nothing; it is retained for callers (e.g. tests) that still supply history,
    # for whom the within-generation byte-stability above is what matters.
    history = cross_task_history or []
    if history:
        prefix_lines.append("## Cross-task forum history (ALL prior generations)")
        prefix_lines.append("(chronological, oldest → newest. Use `gen=N` to weight recency.)")
        for p in history:
            pid = _sanitize_inline_value(p.get("id", "?"))
            aid = _sanitize_inline_value(p.get("agent_id", "?"))
            gen_tag = p.get("generation")
            r_tag = p.get("round_num")
            tags = []
            if gen_tag is not None:
                tags.append(f"gen={gen_tag}")
            if r_tag is not None:
                tags.append(f"round={r_tag}")
            tag_str = (" " + " ".join(tags)) if tags else ""
            text = _sanitize_agent_output(str(p.get("text") or "").strip()).replace("\n", " ")
            if len(text) > _POST_SNIPPET_CHARS:
                text = text[:_POST_SNIPPET_CHARS] + "..."
            prefix_lines.append(f"- [id={pid}{tag_str} agent={aid}] {text}")
        prefix_lines.append("")
    else:
        prefix_lines.append("## Cross-task forum history")
        if int(generation) <= 1:
            prefix_lines.append("- (none — no cross-task posts have been written in any prior generation yet)")
        else:
            # History is intentionally empty on the live forum path for ALL
            # generations: agents see only this generation's posts.
            # In generation 2+, prior-generation posts DO exist — do not tell the
            # agent none were ever written (that would be false and could mislead
            # its grounding). Cross-generation knowledge reaches the agent through
            # distilled seed memory, not raw forum history.
            prefix_lines.append(
                "- (prior-generation posts are not shown here — this forum shows "
                "only the current generation's posts. Cross-generation knowledge "
                "reaches you via distilled seed memory, not raw forum history.)"
            )
        prefix_lines.append("")

    # Gen-1 hardening (mirrors build_per_task_discussion_parts' gen1_disclaimer
    # above): when there is genuinely no cross-task history yet, agents have
    # been observed gaming the `where_it_appeared` concreteness check by
    # quoting this prompt's OWN MCP tool-call syntax (e.g. `forum_post(...)`)
    # as their "verbatim quote" — it satisfies the >=40-char structural check
    # while carrying zero real task signal. This self-corrects by generation
    # 2+ once real history exists, so the disclaimer only needs to fire here.
    if int(generation) <= 1 and not history:
        prefix_lines.append(
            "## IMPORTANT: First generation — no cross-task history exists\n"
            "This is generation 1 and there is NO cross-task forum history yet.\n"
            "- Your `concrete_primitive` and `where_it_appeared` MUST come from "
            "your own just-attempted task (Phase 1) below — not from this "
            "prompt's own MCP tool-call syntax, JSON schema text, or protocol "
            "instructions.\n"
            "- Quoting `forum_post(...)`, `query(...)`, or any of the "
            'instructions above as your "verbatim quote" is NOT a real '
            "concrete primitive and will be discarded by the distiller.\n"
            "- If your own task genuinely gave you nothing concrete to report, "
            "post the narrowest real primitive from your task rather than "
            "fabricating one from this prompt's own text."
        )
        prefix_lines.append("")

    # Round-specific instructions (structured JSON discipline) ----------
    # Plan A (cross-task forum concreteness): each agent emits exactly ONE
    # JSON post per round with a mandatory `concrete_primitive` slot. The
    # structural slot is what blocks meta-advice at write time — agents
    # cannot fill `concrete_primitive` with "separation of concerns"
    # without the `where_it_appeared` verbatim quote contradicting them.
    # Free-form prose is REJECTED.
    if int(round_num) <= 0:
        round_directive = (
            "## Your task this round (round 0 — initial opinion)\n"
            "Post EXACTLY ONE message contributing one concrete primitive "
            "from your just-attempted task that another agent could fail "
            "without. Free-form prose is REJECTED — your message body MUST "
            "be a single JSON object matching the schema below."
        )
    else:
        round_directive = (
            f"## Your task this round (round {round_num} — respond to peers)\n"
            "Read the peer posts below (from agents in your generation in "
            "earlier rounds). Post EXACTLY ONE message responding to one of "
            "them. Your message body MUST be a single JSON object matching "
            "the schema below. The `transfer_claim` field names which peer "
            "post you AGREE WITH (citing additional evidence), DISAGREE WITH "
            "(citing a counterexample), or SYNTHESIZE (combining two peer "
            "primitives into a refined rule)."
        )

    schema_block = (
        "## Required JSON schema (the body of `forum_post(text=...)`)\n"
        "\n"
        "Your post body MUST be a single JSON object with exactly these fields:\n"
        "\n"
        "{\n"
        '  "concrete_primitive": "<a single named operation, API call, '
        "function/class, error type, file path, language feature, test-runner "
        "flag, or numeric invariant — verbatim from your Phase-1 task or a "
        "cited prior post. e.g. 'rust .iter().map(|c| c.to_digit(10)).collect"
        "::<Option<Vec<_>>>()' or '|Δrow| == |Δcol| diagonal invariant' or "
        "'cargo test -- --nocapture for stdout in failed assertions'. "
        "REJECT framings like 'separation of concerns', 'two-phase pipeline', "
        "'pattern', 'approach', 'strategy', 'architecture'. If you cannot "
        'name a code-shaped token, you have nothing concrete to post — drop>",\n'
        '  "task_grounding": {\n'
        '    "task_id": "<your Phase-1 task_id, OR a prior-post-cited task_id>",\n'
        '    "where_it_appeared": "<verbatim 1-2 sentence quote from your '
        "task description, your reflection, your model_output, an error "
        'message, or a cited prior post — names the primitive in context>",\n'
        '    "evidence_post_id": <integer id of the cited prior post, or null '
        "if grounding is from your own Phase-1 task>\n"
        "  },\n"
        '  "transfer_claim": "<one sentence: which OTHER task in this '
        "generation's forum history would benefit from this primitive, and "
        "how it would change the approach. For round 1+, this field MUST "
        'name the AGREE/DISAGREE/SYNTHESIZE relation to a peer post.>",\n'
        '  "anti_meta_self_check": "<one sentence describing why your '
        "concrete_primitive is NOT lifted-from-tutorial advice. If you can't "
        "fill this without re-stating the primitive concretely, the post is "
        'meta — drop it and find a different primitive.>",\n'
        '  "evidence_task_ids": ["<task_id from the Task Evidence Map that '
        'grounds this primitive>", ...]\n'
        "}\n"
        "\n"
        "Rules:\n"
        "- Exactly ONE post per agent per round. The whole `forum_post(text=...)` "
        "body is the JSON object — no preamble, no surrounding prose.\n"
        '- A vague `concrete_primitive` (e.g. "separation of concerns", '
        '"two-phase pipeline", "defensive coding") will be discarded by '
        "the distiller. NOT 'separation of concerns' — name the actual "
        "function/operator/error/file path.\n"
        "- `where_it_appeared` MUST be a verbatim quote (≥40 chars) that "
        "contains a non-stopword from `concrete_primitive`. Paraphrases are "
        "rejected.\n"
        "- `transfer_claim` for round 0 names a candidate target task; for "
        "round 1+ MUST cite a peer post by id and use one of "
        "AGREE/DISAGREE/SYNTHESIZE.\n"
        "- `evidence_task_ids` (round 1+) MUST be a non-empty list of task ids "
        "drawn only from the Task Evidence Map. The server rejects a round-1 "
        "post that omits it or cites an unknown task id.\n"
    )

    protocol_block = (
        "## Available MCP Tools\n"
        '- query(task_id="__cross_task__", query="...", max_records=8) — required before posting.\n'
        '- knowledge(task_id="__cross_task__", limit=20) — read cross-task page in store.\n'
        '- forum_post(task_id="__cross_task__", text="<JSON>", parent_post_id=N) — post the JSON object.\n'
        "- forum_signal_done() — call exactly once when finished this round.\n"
        "\n"
        "## Required protocol\n"
        '1. You MUST call `query(task_id="__cross_task__", query="<your topic>", max_records=8)` BEFORE posting.\n'
        "   The server rejects forum_post until query has been called.\n"
        '2. Post EXACTLY ONE message via `forum_post(task_id="__cross_task__", text="<JSON>")`. The text argument MUST be the JSON object above as a string.\n'
        '3. When responding to a specific prior post (history or peer), you MUST call `forum_post(parent_post_id=<integer id>, ...)` with the integer id from the lists above. Prose references like "Re: post #N" are NOT threaded replies.\n'
        "4. When done, you MUST call `forum_signal_done()` exactly once.\n"
        "5. NEVER cite a post_id that does not appear in the lists above.\n"
    )

    prefix_lines.extend([round_directive, "", ANTI_META_BLOCK, "", schema_block, "", protocol_block, ""])
    cacheable_prefix = "\n".join(prefix_lines)

    # ---- Suffix (per-agent / per-round-per-agent variable content) ----
    # Identity line moved here (out of the cacheable prefix) so the prefix
    # is byte-identical across agents within a (generation, round) and
    # cross-agent prompt-cache reuse can fire.
    suffix_lines: list[str] = [
        f"You are agent {agent_id} in generation {generation}, cross-task forum round {round_num}.",
        "",
    ]

    # Agent's just-attempted task (Phase 1 context — Path A simulation
    # of container persistence: gives the agent its task experience back).
    # Per-agent: each agent attempted a different task → suffix.
    if phase1_context:
        suffix_lines.append("## Your just-attempted task (Phase 1)")
        ctx_tid = _sanitize_inline_value(phase1_context.get("task_id", "?"))
        ctx_score = phase1_context.get("native_score")
        eval_res = phase1_context.get("eval_result") or {}
        resolved = eval_res.get("resolved") if isinstance(eval_res, dict) else None
        reflection = _sanitize_agent_output(str(phase1_context.get("reflection") or "").strip().replace("\n", " "))
        if len(reflection) > _NATIVE_MEMORY_FORUM_INJECT_CHARS:
            reflection = reflection[:_NATIVE_MEMORY_FORUM_INJECT_CHARS] + "..."
        suffix_lines.append(f"- task_id: `{ctx_tid}`")
        suffix_lines.append(f"- score: {ctx_score}")
        if resolved is not None:
            suffix_lines.append(f"- resolved: {resolved}")
        if reflection:
            suffix_lines.append(f"- your reflection (post-eval): {reflection}")
        suffix_lines.append("")

    # Peer posts from THIS generation's prior rounds (only present at
    # round_num > 0). These are your same-gen peers responding to the
    # history above; round 1+ asks you to respond to them. Varies per
    # agent per round → suffix.
    peers = peer_posts_this_gen or []
    if peers:
        suffix_lines.append(f"## Peer posts from generation {generation}, round(s) < {round_num}")
        for p in peers:
            pid = _sanitize_inline_value(p.get("id", "?"))
            aid = _sanitize_inline_value(p.get("agent_id", "?"))
            r_tag = p.get("round_num")
            r_str = f" round={r_tag}" if r_tag is not None else ""
            text = _sanitize_agent_output(str(p.get("text") or "").strip()).replace("\n", " ")
            if len(text) > _POST_SNIPPET_CHARS:
                text = text[:_POST_SNIPPET_CHARS] + "..."
            suffix_lines.append(f"- [id={pid}{r_str} agent={aid}] {text}")
        suffix_lines.append("")

    variable_suffix = "\n".join(suffix_lines) if suffix_lines else ""

    return ForumPromptParts(
        cacheable_prefix=cacheable_prefix,
        variable_suffix=variable_suffix,
    )
