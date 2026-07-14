import threading
import time

from ksi.tokens import PRICES_AS_OF, PRICING, TokenAccumulator, TokenUsage, TokenUsageDict, pricing_for_model


def test_token_usage_dict_keys_match_to_dict_output():
    # Contract: TokenUsageDict documents the exact shape of TokenUsage.to_dict().
    # If a bucket is added/removed in to_dict() without updating the TypedDict
    # (or vice versa), this fails — keeping the protocol-boundary type honest.
    produced = TokenUsage(input_tokens=1, output_tokens=2).to_dict()
    assert set(produced) == set(TokenUsageDict.__annotations__)


def test_prices_as_of_constant_present():
    # The audit stamp must exist and point sources/reviewers at a concrete date.
    assert PRICES_AS_OF == "2026-06-05"


def test_anthropic_cache_read_is_10pct_of_input():
    # Audit invariant (issue #646): Anthropic cache-HIT rate is 0.1x base input.
    # Verified 2026-06-05 against platform.claude.com pricing.
    for key in ("haiku", "sonnet", "opus"):
        p = PRICING[key]
        assert p["cache_read"] == round(p["input"] * 0.10, 4), key


def test_anthropic_cache_create_is_125pct_of_input():
    # Audit invariant: Anthropic 5-minute cache-WRITE rate is 1.25x base input.
    for key in ("haiku", "sonnet", "opus"):
        p = PRICING[key]
        assert p["cache_create"] == round(p["input"] * 1.25, 4), key


def test_cost_applies_cache_read_discount_not_full_input():
    # A cache_read token must be billed at the discounted cache_read rate, NOT at
    # the full uncached-input rate. Haiku 4.5: input $1.00/MTok, cache_read
    # $0.10/MTok. 1M cache-read tokens => $0.10, not $1.00.
    u = TokenUsage(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000)
    assert u.cost_usd("claude-haiku-4-5") == 0.10
    # Contrast: the same volume as fresh input costs the full $1.00.
    u_fresh = TokenUsage(input_tokens=1_000_000)
    assert u_fresh.cost_usd("claude-haiku-4-5") == 1.00


def test_cost_applies_cache_write_premium_on_anthropic():
    # cache_creation tokens are billed at the 1.25x write premium (Haiku $1.25/MTok).
    u = TokenUsage(cache_creation_input_tokens=1_000_000)
    assert u.cost_usd("claude-haiku-4-5") == 1.25


def test_cost_openai_cache_write_is_free():
    # OpenAI auto-caches server-side: cache_creation must contribute $0.
    u = TokenUsage(cache_creation_input_tokens=1_000_000)
    assert u.cost_usd("gpt-5.4-mini") == 0.0


def test_cost_combined_buckets_anthropic_haiku():
    # End-to-end cache-aware cost across all four buckets for the project's
    # default Anthropic model. Haiku 4.5: in $1.00, out $5.00, read $0.10,
    # write $1.25 per MTok. 1M of each => 1.00 + 5.00 + 0.10 + 1.25 = $7.35.
    u = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    assert u.cost_usd("claude-haiku-4-5") == 7.35


def test_unknown_model_fallback_unchanged():
    # Audit must not regress the "unknown model -> $0.00" contract.
    assert pricing_for_model("vendor-x-fictional-1") is None
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert u.cost_usd("vendor-x-fictional-1") == 0.0
    assert u.cost_usd("gpt-99-future") == 0.0


def test_reasoning_tokens_counted_as_output_via_output_bucket():
    # OpenAI reasoning tokens arrive already folded into output_tokens (see
    # runtime_runner/agent-runner/src/openai.ts: "output_tokens = outputTotal
    # (reasoning stays inside)"). So they are billed at the OUTPUT rate, which is
    # exactly what a cost over output_tokens does. gpt-5.4-mini output $4.50/MTok.
    u = TokenUsage(input_tokens=0, output_tokens=1_000_000)  # includes reasoning
    assert u.cost_usd("gpt-5.4-mini") == 4.50


def test_token_usage_total():
    # No cache tokens → total == input + output.
    u = TokenUsage(input_tokens=100, output_tokens=40)
    assert u.total == 140
    assert u.input_tokens + u.output_tokens == 140
    assert u.uncached_input_tokens == 100


def test_token_usage_total_includes_cache():
    # Cache-aware total: input + output + cache_read + cache_creation.
    # Regression: prior to the cache-inclusive fix, `total` returned 140 on a
    # fixture that billed 640 tokens, losing ~78% of the volume.
    u = TokenUsage(
        input_tokens=100,
        output_tokens=40,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=300,
    )
    assert u.total == 640
    assert u.input_tokens + u.output_tokens == 140
    assert u.uncached_input_tokens == 100


def test_token_usage_total_semantics_contract():
    """Pin the semantic contract for both totals to prevent accidental
    regression or silent reintroduction of the pre-fix behavior.

    For a TokenUsage with non-zero cache fields:
      - ``input_tokens + output_tokens`` is the legacy input+output figure
      - ``total`` == input + output + cache_read + cache_creation  (new/correct)
    """
    u = TokenUsage(
        input_tokens=1_000,
        output_tokens=500,
        cache_creation_input_tokens=10_000,
        cache_read_input_tokens=50_000,
    )
    assert u.uncached_input_tokens == 1_000
    assert u.total == (u.input_tokens + u.output_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens)
    # The gap between total and the legacy input+output figure is exactly
    # the cache volume.
    assert u.total - (u.input_tokens + u.output_tokens) == (u.cache_read_input_tokens + u.cache_creation_input_tokens)


def test_token_usage_to_dict_breaking_semantics_noted():
    """``to_dict()['total_tokens']`` now includes cache buckets.

    This is the breaking change for JSON consumers. Any script that
    computed ``total - (input + output)`` expecting 0 will get the
    cache total after this fix. This test pins that new contract so
    future changes don't silently revert it.
    """
    u = TokenUsage(
        input_tokens=1_000,
        output_tokens=500,
        cache_creation_input_tokens=10_000,
        cache_read_input_tokens=50_000,
    )
    d = u.to_dict()
    # Breaking change: total_tokens is no longer just input+output.
    assert d["total_tokens"] != d["input_tokens"] + d["output_tokens"]
    assert d["total_tokens"] == 61_500
    # Cache buckets are surfaced so consumers can recover either value.
    assert d["cache_read_input_tokens"] == 50_000
    assert d["cache_creation_input_tokens"] == 10_000
    assert d["uncached_input_tokens"] == 1_000
    # Legacy figure is recoverable via subtraction or as input + output.
    legacy = d["total_tokens"] - d["cache_read_input_tokens"] - d["cache_creation_input_tokens"]
    assert legacy == u.input_tokens + u.output_tokens == 1_500


def test_token_usage_cost_unknown_model_is_not_mispriced_as_haiku():
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    # Truly fictional model names must return 0.0, NOT silently fall through
    # via substring leakage (e.g. "haiku" appearing inside another name).
    assert u.cost_usd("vendor-x-fictional-1") == 0.0
    assert u.cost_usd("gpt-99-future") == 0.0
    # Haiku 3.5 (legacy) priced via "3-5-haiku" version-specific entry.
    assert u.cost_usd("claude-3-5-haiku") == 4.8


def test_token_usage_cost_anthropic_version_disambiguation():
    # Anthropic split pricing tiers between Haiku 3.5 and 4.5, and between
    # Opus 4.1 and 4.5. The matcher must route each named version to the
    # correct tier — bare keys represent current-generation pricing, and
    # version-specific keys override via longest-match-first.
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    # Haiku 4.5 (current default):
    # $1.00 in + $5.00 out → $6.00. Falls through to bare "haiku".
    assert u.cost_usd("claude-haiku-4-5") == 6.0
    assert u.cost_usd("claude-haiku-4-5-20251001") == 6.0
    assert u.cost_usd("anthropic/claude-haiku-4-5-20251001") == 6.0
    # Haiku 3.5 (legacy): $0.80 in + $4.00 out → $4.80. Both naming forms.
    assert u.cost_usd("claude-3-5-haiku") == 4.8
    assert u.cost_usd("claude-3-5-haiku-20241022") == 4.8
    assert u.cost_usd("claude-haiku-3-5") == 4.8
    # Haiku 3 (oldest): $0.25 in + $1.25 out → $1.50.
    assert u.cost_usd("claude-haiku-3") == 1.5
    assert u.cost_usd("claude-3-haiku") == 1.5
    # Sonnet — same $3/$15 across 3.5/4/4.5/4.6 (no version-specific keys).
    assert u.cost_usd("claude-sonnet-4-6") == 18.0
    assert u.cost_usd("claude-sonnet-4-20250514") == 18.0
    assert u.cost_usd("claude-3-5-sonnet") == 18.0
    assert u.cost_usd("bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0") == 18.0
    # Opus 4.5+ (current $5/$25 tier): falls through to bare "opus".
    assert u.cost_usd("claude-opus-4-7") == 30.0
    assert u.cost_usd("claude-opus-4-6") == 30.0
    assert u.cost_usd("claude-opus-4-5") == 30.0
    # Opus 4.1 / 4 / 3 (legacy $15/$75 tier): version-specific keys.
    assert u.cost_usd("claude-opus-4-1") == 90.0
    assert u.cost_usd("claude-opus-4") == 90.0
    assert u.cost_usd("claude-opus-3") == 90.0
    assert u.cost_usd("claude-3-opus") == 90.0


def test_token_usage_cost_gpt54_family_pricing():
    # The project default OpenAI model is gpt-5.4-mini.
    # Pin every advertised tier so a future PRICING refactor can't silently
    # mis-attribute the audit's main OpenAI cost cells.
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    # gpt-5.4-mini: $0.75 in + $4.50 out → $5.25
    assert u.cost_usd("gpt-5.4-mini") == 5.25
    assert u.cost_usd("openai/gpt-5.4-mini") == 5.25
    # gpt-5.4: $2.50 in + $15.00 out → $17.50
    assert u.cost_usd("gpt-5.4") == 17.50
    # gpt-5.4-nano: $0.20 in + $1.25 out → $1.45
    assert u.cost_usd("gpt-5.4-nano") == 1.45
    # gpt-5.4-pro: $30 in + $180 out → $210.00 (no caching tier)
    assert u.cost_usd("gpt-5.4-pro") == 210.00


def test_token_usage_cost_subversion_disambiguation():
    # Critical regression invariant: gpt-5.4 must price as gpt-5.4, NOT fall
    # through to gpt-5 ($1.25/$10.00) via substring leakage. The matcher
    # rejects any match where the next character is a digit or '.', which is
    # how this is enforced. Same shape for gpt-5.5 vs gpt-5, etc.
    u = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert u.cost_usd("gpt-5") == 1.25  # baseline
    assert u.cost_usd("gpt-5.4") == 2.50  # NOT 1.25
    assert u.cost_usd("gpt-5.5") == 5.00  # NOT 1.25
    assert u.cost_usd("gpt-5.2") == 1.75  # NOT 1.25
    # And mini variants must beat their parent family by longest-match.
    assert u.cost_usd("gpt-5-mini") == 0.25  # NOT gpt-5 ($1.25)
    assert u.cost_usd("gpt-5.4-mini") == 0.75  # NOT gpt-5.4 ($2.50), NOT gpt-5 ($1.25)
    assert u.cost_usd("gpt-5.5-pro") == 30.00  # NOT gpt-5.5 ($5.00)


def test_token_usage_cost_openai_models():
    # 1M input + 1M output, no cache. Pricing sourced from litellm
    # model_prices_and_context_window.json (verified 2026-04).
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    # gpt-5: $1.25 in, $10.00 out → $11.25
    assert u.cost_usd("gpt-5") == 11.25
    # gpt-5-mini: $0.25 in, $2.00 out → $2.25
    assert u.cost_usd("gpt-5-mini") == 2.25
    # gpt-5-nano: $0.05 in, $0.40 out → $0.45
    assert u.cost_usd("gpt-5-nano") == 0.45
    # gpt-4o: $2.50 in, $10.00 out → $12.50
    assert u.cost_usd("gpt-4o") == 12.50
    # gpt-4o-mini: $0.15 in, $0.60 out → $0.75
    assert u.cost_usd("gpt-4o-mini") == 0.75
    # o3-mini: $1.10 in, $4.40 out → $5.50
    assert u.cost_usd("o3-mini") == 5.50


def test_token_usage_cost_openai_strips_litellm_prefixes():
    # HA emits litellm-style names like 'openai/gpt-5-mini'; DGM uses bare
    # names; bedrock-routed Anthropic comes through as 'bedrock/...'. The
    # matcher must strip provider prefixes before matching.
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert u.cost_usd("openai/gpt-5-mini") == 2.25
    assert u.cost_usd("azure/global/gpt-5.1") == 11.25
    assert u.cost_usd("bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0") == 18.0


def test_token_usage_cost_openai_dated_suffix():
    # Dated model snapshots (e.g. -2024-08-06, -2025-01-31) must price the
    # same as their base family. Dropping the date by relying on substring
    # match against the family name handles this without hardcoding dates.
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert u.cost_usd("gpt-5-2025-08-07") == 11.25
    assert u.cost_usd("gpt-4o-2024-08-06") == 12.50
    assert u.cost_usd("o3-mini-2025-01-31") == 5.50
    assert u.cost_usd("gpt-4o-mini-2024-07-18") == 0.75


def test_token_usage_cost_openai_longest_match_wins():
    # 'gpt-5-mini' must NOT be priced as 'gpt-5' (the shorter prefix). The
    # longest-key-first match guarantees correct disambiguation.
    u = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    # gpt-5 input: $1.25; gpt-5-mini input: $0.25. They must differ.
    assert u.cost_usd("gpt-5") == 1.25
    assert u.cost_usd("gpt-5-mini") == 0.25
    assert u.cost_usd("gpt-5-nano") == 0.05
    # gpt-5.1-codex-mini ($0.25 in) must beat gpt-5.1-codex ($1.25 in).
    assert u.cost_usd("gpt-5.1-codex-mini") == 0.25
    assert u.cost_usd("gpt-5.1-codex") == 1.25


def test_token_usage_cost_openai_cache_read():
    # OpenAI cache_read is 50% of input for gpt-4o, 10% for gpt-5 family.
    # cache_creation is structurally 0 — OpenAI auto-caches without write fee.
    u = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,  # should contribute $0 for OpenAI
    )
    # gpt-5: 1M*$1.25 + 1M*$0.125 + 1M*$0 + 0*$10 = $1.375
    assert u.cost_usd("gpt-5") == 1.375
    # gpt-4o: 1M*$2.50 + 1M*$1.25 + 1M*$0 + 0*$10 = $3.75
    assert u.cost_usd("gpt-4o") == 3.75


def test_token_usage_add():
    a = TokenUsage(100, 40)
    b = TokenUsage(200, 60)
    c = a + b
    assert c.input_tokens == 300
    assert c.output_tokens == 100
    assert c.total == 400


def test_token_usage_add_with_cache():
    a = TokenUsage(100, 40, cache_creation_input_tokens=50, cache_read_input_tokens=500)
    b = TokenUsage(200, 60, cache_creation_input_tokens=25, cache_read_input_tokens=1000)
    c = a + b
    assert c.cache_creation_input_tokens == 75
    assert c.cache_read_input_tokens == 1500
    # total == 300 + 100 + 75 + 1500
    assert c.total == 1975
    assert c.input_tokens + c.output_tokens == 400


def test_token_usage_default():
    u = TokenUsage()
    assert u.total == 0


def test_accumulator_record_task_and_total():
    acc = TokenAccumulator()
    acc.record_task(1, "peer-0", "task-a", TokenUsage(100, 40))
    acc.record_task(1, "peer-0", "task-b", TokenUsage(200, 60))
    acc.record_task(1, "peer-1", "task-a", TokenUsage(150, 50))
    t = acc.total()
    assert t.input_tokens == 450
    assert t.output_tokens == 150


def test_accumulator_by_agent():
    acc = TokenAccumulator()
    acc.record_task(1, "peer-0", "task-a", TokenUsage(100, 40))
    acc.record_task(1, "peer-1", "task-a", TokenUsage(200, 60))
    by_agent = acc.by_agent()
    assert by_agent["peer-0"].total == 140
    assert by_agent["peer-1"].total == 260


def test_accumulator_by_generation():
    acc = TokenAccumulator()
    acc.record_task(1, "peer-0", "task-a", TokenUsage(100, 40))
    acc.record_task(2, "peer-0", "task-a", TokenUsage(200, 60))
    by_gen = acc.by_generation()
    assert by_gen[1].total == 140
    assert by_gen[2].total == 260


def test_accumulator_reads_are_threadsafe_against_concurrent_writes():
    # Regression for issue #705: total()/by_agent()/by_generation() iterated
    # _entries without the writer lock. With many writers inserting *distinct*
    # keys (resizing the dict) while a reader iterates, the pre-fix code raised
    # RuntimeError("dictionary changed size during iteration"). The fix snapshots
    # under the lock, so this loop must complete without raising.
    acc = TokenAccumulator()
    n_writers = 20
    per_writer = 200
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer(wid: int) -> None:
        try:
            for i in range(per_writer):
                # Distinct (generation, agent, task) key per write so the dict
                # grows monotonically and reallocates mid-iteration.
                acc.record_task(wid, f"agent-{wid}", f"task-{i}", TokenUsage(1, 1))
        except BaseException as exc:  # noqa: BLE001 - surface to the assert
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                acc.total()
                acc.by_agent()
                acc.by_generation()
        except BaseException as exc:  # noqa: BLE001 - surface to the assert
            errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    writers = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
    for r in readers:
        r.start()
    for w in writers:
        w.start()
    for w in writers:
        w.join()
    # Let readers spin a touch longer against the now-full dict, then stop.
    deadline = time.monotonic() + 0.1
    while time.monotonic() < deadline:
        acc.total()
    stop.set()
    for r in readers:
        r.join()

    assert errors == [], f"concurrent read/write raised: {errors!r}"

    # Final aggregate must be exact: n_writers * per_writer writes of (1 in, 1 out).
    expected = n_writers * per_writer
    t = acc.total()
    assert t.input_tokens == expected
    assert t.output_tokens == expected
    by_agent = acc.by_agent()
    assert len(by_agent) == n_writers
    for wid in range(n_writers):
        assert by_agent[f"agent-{wid}"].input_tokens == per_writer


def test_accumulator_record_lifecycle():
    acc = TokenAccumulator()
    acc.record_task(1, "peer-0", "task-a", TokenUsage(100, 40))
    acc.record_lifecycle(1, "peer-0", "generate_insights", TokenUsage(50, 20))
    t = acc.total()
    assert t.input_tokens == 150
    assert t.output_tokens == 60


def test_accumulator_flush_retries_failed_store_insert():
    class FailingStore:
        def __init__(self):
            self.calls = 0

        def insert_token_phase(self, **kwargs):
            self.calls += 1
            raise RuntimeError("temporary store failure")

    class RecordingStore:
        def __init__(self):
            self.calls = []

        def insert_token_phase(self, **kwargs):
            self.calls.append(kwargs)

    acc = TokenAccumulator()
    acc.record_task(1, "peer-0", "task-a", TokenUsage(100, 40))

    failing = FailingStore()
    acc.flush_to_store(failing, run_id=7, generation=1, model="claude-3-5-haiku")
    assert failing.calls == 1

    recording = RecordingStore()
    acc.flush_to_store(recording, run_id=7, generation=1, model="claude-3-5-haiku")

    assert len(recording.calls) == 1
    assert recording.calls[0]["phase"] == "task_execution"
    assert recording.calls[0]["agent_ref"] == "peer-0"
    assert recording.calls[0]["token_usage"].total == 140


def test_accumulator_to_dict_keys():
    acc = TokenAccumulator()
    acc.record_task(1, "peer-0", "task-a", TokenUsage(100, 40))
    d = acc.to_dict()
    assert "total" in d
    assert "by_agent" in d
    assert "by_generation" in d
    assert d["total"]["input_tokens"] == 100


def test_token_usage_add_assignment():
    # __add__ + rebind; TokenUsage has no __iadd__ (immutable-style accumulation)
    u = TokenUsage(input_tokens=100, output_tokens=40)
    u = u + TokenUsage(input_tokens=50, output_tokens=10)
    assert u.input_tokens == 150
    assert u.output_tokens == 50
    assert u.total == 200


def test_token_usage_to_dict():
    u = TokenUsage(input_tokens=100, output_tokens=40)
    d = u.to_dict()
    assert d["input_tokens"] == 100
    assert d["output_tokens"] == 40
    assert d["total_tokens"] == 140
    assert d["uncached_input_tokens"] == 100
    # Cache fields are always emitted (0 when unused) so downstream consumers
    # can rely on the key being present.
    assert d["cache_creation_input_tokens"] == 0
    assert d["cache_read_input_tokens"] == 0


def test_token_usage_to_dict_with_cache():
    # Fixture mirroring a real Anthropic `usage` block with cache hits.
    u = TokenUsage(
        input_tokens=1_000,
        output_tokens=500,
        cache_creation_input_tokens=10_000,
        cache_read_input_tokens=50_000,
    )
    d = u.to_dict()
    assert d["input_tokens"] == 1_000
    assert d["output_tokens"] == 500
    assert d["cache_creation_input_tokens"] == 10_000
    assert d["cache_read_input_tokens"] == 50_000
    assert d["uncached_input_tokens"] == 1_000
    # total_tokens must reflect the full billed volume (1000 + 500 + 10k + 50k).
    assert d["total_tokens"] == 61_500


def test_runtime_result_has_token_usage():
    from ksi.runtime.types import RuntimeResult
    from ksi.tokens import TokenUsage

    r = RuntimeResult(output="hello", token_usage=TokenUsage(100, 40))
    assert r.token_usage.total == 140


def test_task_trace_has_token_usage():
    from ksi.models import TaskTrace
    from ksi.tokens import TokenUsage

    t = TaskTrace(
        generation=1,
        agent_id="peer-0",
        task_id="task-a",
        model_output="out",
        eval_result={},
        native_score=1.0,
        token_usage=TokenUsage(100, 40),
    )
    assert t.token_usage.total == 140


def test_agent_state_has_token_usage():
    from ksi.models import AgentState

    a = AgentState(id="peer-0")
    assert hasattr(a, "token_usage")
    assert a.token_usage == 0
    a.token_usage = 140
    assert a.token_usage == 140


def test_extract_token_usage_per_direction():
    from ksi.runtime.container_host import _extract_token_usage

    meta = {"input_tokens": 300, "output_tokens": 100}
    u = _extract_token_usage(meta)
    assert u.input_tokens == 300
    assert u.output_tokens == 100


def test_extract_token_usage_nested_usage():
    from ksi.runtime.container_host import _extract_token_usage

    meta = {"usage": {"input_tokens": 200, "output_tokens": 80}}
    u = _extract_token_usage(meta)
    assert u.input_tokens == 200
    assert u.output_tokens == 80


def test_extract_token_usage_tokens_used_fallback():
    from ksi.runtime.container_host import _extract_token_usage

    meta = {"tokens_used": 500}
    u = _extract_token_usage(meta)
    assert u.total == 500


def test_extract_token_usage_empty():
    from ksi.runtime.container_host import _extract_token_usage

    u = _extract_token_usage({})
    assert u.total == 0


def test_parse_runner_stdout_populates_token_usage():
    import json

    from ksi.runtime.container_host import _parse_runner_stdout

    stdout = json.dumps(
        {
            "result": "patch output",
            "tool_trace": [],
            "meta": {"input_tokens": 400, "output_tokens": 120},
        }
    )
    parsed = _parse_runner_stdout(stdout, key="result")
    assert parsed["token_usage"].input_tokens == 400
    assert parsed["token_usage"].output_tokens == 120


def test_extract_token_usage_total_tokens_fallback():
    from ksi.runtime.container_host import _extract_token_usage

    meta = {"total_tokens": 750}
    u = _extract_token_usage(meta)
    assert u.total == 750


def test_extract_token_usage_propagates_cache_tokens():
    """Regression: cache tokens from an Anthropic-style `usage` block must
    survive extraction and accumulation into the serialized breakdown.

    Prior behavior: TokenUsage.total excluded cache tokens, so
    `TokenAccumulator.to_dict()["total"]["total_tokens"]` dramatically
    undercounted billed volume on cache-heavy runs. Quantified on the
    arc2_haiku_swarm_g10 knowledge DB: reported total_tokens would have been
    8.4M (input 2.1M + output 6.3M), actual billed volume = 405.3M
    (including 370.1M cache_read + 26.8M cache_creation). Undercount:
    396.9M tokens (~98% of volume invisible in summaries).
    """
    from ksi.runtime.normalize import extract_token_usage

    # Fixture shaped exactly like the TS agent-runner's `meta.usage` emit.
    meta = {
        "usage": {
            "input_tokens": 200,
            "output_tokens": 50,
            "cache_creation_input_tokens": 1_000,
            "cache_read_input_tokens": 20_000,
        }
    }
    u = extract_token_usage(meta)
    assert u.input_tokens == 200
    assert u.output_tokens == 50
    assert u.cache_creation_input_tokens == 1_000
    assert u.cache_read_input_tokens == 20_000
    # total_tokens must include the cache buckets.
    assert u.total == 21_250
    assert u.input_tokens + u.output_tokens == 250

    # Round-trip via accumulator → to_dict keeps the breakdown visible.
    acc = TokenAccumulator()
    acc.record_task(1, "peer-0", "task-a", u)
    d = acc.to_dict()
    assert d["total"]["total_tokens"] == 21_250
    assert d["total"]["cache_read_input_tokens"] == 20_000
    assert d["total"]["cache_creation_input_tokens"] == 1_000


def test_extract_token_usage_openai_cached_tokens_details():
    from ksi.runtime.normalize import extract_token_usage

    meta = {
        "usage": {
            "input_tokens": 30_000,
            "output_tokens": 900,
            "input_tokens_details": {"cached_tokens": 24_000},
        }
    }
    u = extract_token_usage(meta)
    assert u.input_tokens == 6_000
    assert u.cache_read_input_tokens == 24_000
    assert u.uncached_input_tokens == 6_000
    assert u.output_tokens == 900


def test_openai_usage_helper_reads_object_details():
    from types import SimpleNamespace

    from ksi.runtime.llm import _usage_child, _usage_value

    usage = SimpleNamespace(
        input_tokens=30_000,
        output_tokens=900,
        input_tokens_details=SimpleNamespace(cached_tokens=24_000),
    )
    details = _usage_child(usage, "input_tokens_details")
    assert _usage_value(usage, "input_tokens") == 30_000
    assert _usage_value(details, "cached_tokens") == 24_000


def test_engine_accumulates_task_tokens():
    """Engine TokenAccumulator sums task tokens from MockRuntime (10 in + 5 out each)."""
    from ksi.models import GenerationConfig, TaskSpec
    from ksi.orchestrator.engine import GenerationalOrchestrator

    class AlwaysPassEvaluator:
        def evaluate(self, *, task, model_output, **kwargs):
            return {"native_score": 1.0}

    # 4 tasks + 2 agents → round-robin gives each agent 2 tasks
    tasks = [
        TaskSpec(id="t1", prompt="solve"),
        TaskSpec(id="t2", prompt="solve"),
        TaskSpec(id="t3", prompt="solve"),
        TaskSpec(id="t4", prompt="solve"),
    ]
    config = GenerationConfig(num_generations=1, num_agents=2)

    # MockRuntimeExecutor returns TokenUsage(input=10, output=5) per call
    from ksi.runtime.types import RuntimeResult as _RuntimeResult
    from ksi.tokens import LLMResponse as _LLMResponse
    from ksi.tokens import TokenUsage as _TokenUsage

    class MockRuntimeExecutor:
        def run_task(self, *, generation, agent_id, task, **kwargs):
            return _RuntimeResult(
                output="mock output",
                token_usage=_TokenUsage(input_tokens=10, output_tokens=5),
            )

    # Mock LLM that returns claim responses with all task IDs,
    # and no-op responses for forum/seed phases
    class MockLLM:
        def call(self, *, system, user, **kwargs):
            # Return all task IDs as claims so broker assigns them
            return _LLMResponse(text='{"claimed_tasks": ["t1", "t2", "t3", "t4"]}', usage=_TokenUsage())

    orch = GenerationalOrchestrator(
        config=config,
        runtime=MockRuntimeExecutor(),
        evaluator=AlwaysPassEvaluator(),
        llm=MockLLM(),
    )
    orch.run(tasks=tasks)
    # Each task executed once, each with (10 in + 5 out): 4 tasks → 40 in / 20 out.
    total = orch.accumulator.total()
    assert total.input_tokens == 40
    assert total.output_tokens == 20
