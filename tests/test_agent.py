"""Tests for sherlock.agent's pure helpers (the streaming run() loop itself
needs a live Anthropic client and is exercised via live/manual testing)."""

from __future__ import annotations

import pytest

from sherlock.agent import (
    _cached_system,
    _cached_tools,
    _dedupe_known_entities,
    _estimate_cost,
    _with_cache_breakpoint,
)


def test_first_known_entities_miss_passes_through_unchanged():
    result = {"results": [], "known_entities": [{"name": "action_rules", "entity_type": "concept"}], "note": "..."}
    out, shown = _dedupe_known_entities(result, already_shown=False)
    assert out == result
    assert shown is True


def test_repeat_known_entities_miss_is_truncated():
    """The scenario that motivated this: a real investigation missed twice,
    resending the full ~285-entity list both times -- ~14KB baked into
    history and re-billed on every turn after. The second miss should not
    repeat the list."""
    result = {"results": [], "known_entities": [{"name": "action_rules", "entity_type": "concept"}] * 285, "note": "..."}
    out, shown = _dedupe_known_entities(result, already_shown=True)
    assert "known_entities" not in out
    assert "already given earlier" in out["note"]
    assert shown is True


def test_result_without_known_entities_is_untouched():
    result = {"results": [{"title": "x"}], "entity": {"name": "action_rules"}}
    out, shown = _dedupe_known_entities(result, already_shown=False)
    assert out == result
    assert shown is False


# ── prompt caching ────────────────────────────────────────────────────────────

def test_estimate_cost_without_cache_matches_plain_input_output_pricing():
    cost = _estimate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == 3.00 + 15.00


def test_estimate_cost_cache_write_costs_more_than_plain_input():
    """A cache write is a 1.25x premium on the base input rate -- more
    expensive than a normal input token, since it's a one-time cost that
    only pays off if the prefix actually gets reused later."""
    plain = _estimate_cost("claude-sonnet-4-6", 1_000_000, 0)
    with_write = _estimate_cost("claude-sonnet-4-6", 0, 0, cache_creation_tokens=1_000_000)
    assert with_write == pytest.approx(plain * 1.25)


def test_estimate_cost_cache_read_is_a_tenth_of_plain_input():
    plain = _estimate_cost("claude-sonnet-4-6", 1_000_000, 0)
    with_read = _estimate_cost("claude-sonnet-4-6", 0, 0, cache_read_tokens=1_000_000)
    assert with_read == pytest.approx(plain * 0.1)


def test_cached_system_wraps_text_with_ephemeral_breakpoint():
    result = _cached_system("You are Sherlock.")
    assert result == [{
        "type": "text",
        "text": "You are Sherlock.",
        "cache_control": {"type": "ephemeral"},
    }]


def test_cached_tools_marks_only_the_last_definition():
    """A cache breakpoint caches everything up to and including the marked
    block -- one breakpoint on the last tool is enough to cover the whole
    tools array, and marking every tool would waste the 4-breakpoint budget
    shared with the system prompt and history breakpoints."""
    tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    result = _cached_tools(tools)
    assert "cache_control" not in result[0]
    assert "cache_control" not in result[1]
    assert result[2] == {"name": "c", "cache_control": {"type": "ephemeral"}}
    # Original list must be untouched -- DEFINITIONS is a shared module-level
    # constant reused across every session.
    assert "cache_control" not in tools[2]


def test_cached_tools_handles_empty_list():
    assert _cached_tools([]) == []


def test_with_cache_breakpoint_tags_last_block_of_last_message():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t1"}]},
    ]
    result = _with_cache_breakpoint(messages)
    assert result[0] == messages[0]  # untouched
    assert result[1]["content"][0] == {"type": "text", "text": "hi"}  # untouched
    assert result[1]["content"][1] == {"type": "tool_use", "id": "t1", "cache_control": {"type": "ephemeral"}}
    # Original messages list/dicts must be untouched -- these get persisted
    # to session.history as plain dicts; cache_control is an API-call-only
    # hint, not part of what was actually said.
    assert "cache_control" not in messages[1]["content"][1]


def test_with_cache_breakpoint_wraps_bare_string_content():
    """The very first seed message uses plain string content
    ({"role": "user", "content": "Investigate entity ..."}), not a list of
    blocks -- cache_control can only attach to a block, so this has to be
    promoted to a one-block list first."""
    messages = [{"role": "user", "content": "Investigate entity `x` and determine root cause."}]
    result = _with_cache_breakpoint(messages)
    assert result[0]["content"] == [{
        "type": "text",
        "text": "Investigate entity `x` and determine root cause.",
        "cache_control": {"type": "ephemeral"},
    }]
    assert messages[0]["content"] == "Investigate entity `x` and determine root cause."  # untouched


def test_with_cache_breakpoint_handles_empty_messages():
    assert _with_cache_breakpoint([]) == []
