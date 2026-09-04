"""Sherlock agentic investigation loop.

Streams DiagnoseChunk objects to the caller as the agent works through
the investigation ladder. Uses the Anthropic API with tool use.

The loop terminates when Claude calls submit_hypothesis (classification
committed) or ask_operator (waiting for operator input), or when the
turn cap is reached.
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
from collections.abc import AsyncGenerator

import anthropic

from sherlock import audit
from sherlock import guardrail
from sherlock import memory as mem
from sherlock import metrics as mtx
from sherlock import prompt
from sherlock import sessions
from sherlock.models import DiagnoseChunk, HypothesisData, InstrumentContext
from sherlock.sessions import Session  # noqa: F401 — used by _done_chunk type hint
from sherlock.tools import DEFINITIONS, dispatch

log = logging.getLogger(__name__)

MODEL      = os.environ.get("SHERLOCK_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 4096
# Caps internal back-and-forth (model turn -> tool calls -> model turn...) for
# a single question, not a whole conversation — main.py resets
# session.turn_count to 0 at the start of every fresh question (chat or
# reply), so a long-running thread never inherits an earlier question's
# spent budget. This is a runaway-loop safety valve, not a cost control: a
# well-investigated single answer can legitimately spend several tool calls
# (a search_kb miss + re-query + a couple of follow-ups is normal), so this
# is set generous on purpose.
MAX_TURNS  = 25

# Falls back into the audit log's instrument_id when a session carries none
# (e.g. general chat with no entity in play) — this deployment only ever
# serves one instrument, so untargeted exchanges are still that
# instrument's audit trail. Deliberately NOT used for session.instrument_id
# itself, which stays empty for general chat so context.load()/mem.load()
# don't pull in instrument-specific context for questions that never asked
# for it.
INSTRUMENT = os.environ.get("INSTRUMENT", "")

# Pricing per million tokens (USD). Update if model changes.
_PRICING: dict[str, tuple[float, float]] = {
    # model-id: (input $/1M, output $/1M)
    "claude-sonnet-4-6":          (3.00, 15.00),
    "claude-opus-4-7":            (15.00, 75.00),
    "claude-haiku-4-5-20251001":  (0.80,  4.00),
}
_DEFAULT_PRICING = (3.00, 15.00)


# Cache write/read multipliers on the base input rate (Anthropic's standard
# 5-minute ephemeral cache): a cache write costs 1.25x normal input price
# (one-time, the first time a prefix is cached), a cache read costs 0.1x
# (every subsequent turn that reuses it). Net win as soon as a cached
# prefix is reused even once within the TTL.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.1


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    input_rate, output_rate = _PRICING.get(model, _DEFAULT_PRICING)
    cost = input_tokens * input_rate + output_tokens * output_rate
    cost += cache_creation_tokens * input_rate * _CACHE_WRITE_MULTIPLIER
    cost += cache_read_tokens * input_rate * _CACHE_READ_MULTIPLIER
    return cost / 1_000_000


def _dedupe_known_entities(result: dict, already_shown: bool) -> tuple[dict, bool]:
    """search_kb's known_entities fallback is a full ~285-entity dump (see
    its own docstring) -- fine once, but every repeat miss within the same
    investigation was resending the identical ~14KB list, permanently baked
    into history and re-billed on every turn after. Truncates it to a short
    pointer on the second and later miss within one run() call; the caller
    resets already_shown per question, so a genuinely new question later
    still gets the full list fresh.

    Returns (possibly-truncated result, updated already_shown).
    """
    if "known_entities" not in result:
        return result, already_shown
    if not already_shown:
        return result, True
    truncated = {
        **{k: v for k, v in result.items() if k not in ("known_entities", "note")},
        "note": (
            "No exact entity name/alias match for this query, same as a prior "
            "search this investigation — the full known_entities list was already "
            "given earlier in this conversation; re-read it there instead of "
            "requesting it again."
        ),
    }
    return truncated, already_shown


def _client() -> anthropic.AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------
# Without this, every turn -- including the final one that just writes prose
# with no tool call -- resends the full system prompt, tool definitions, and
# entire session.history as fresh, full-price input tokens. Measured on a
# real investigation: the no-op final-answer turn alone cost nearly as much
# as all four preceding search_kb turns combined, purely from re-sending
# context that hadn't changed. cache_control breakpoints mark a prefix as
# reusable: a cache write costs 1.25x normal input price once, a cache read
# costs 0.1x on every later turn that reuses it (see _estimate_cost).
#
# Three breakpoints (Anthropic allows up to 4 per request):
#   1. End of the system prompt -- identical every turn of one investigation,
#      and often across many investigations for the same instrument.
#   2. End of the tool definitions -- static, essentially never changes.
#   3. End of session.history as of the previous turn -- moves forward each
#      turn, so only the newest tool_result/reply pays full price; the
#      Anthropic SDK/cache doesn't care that the breakpoint moves, since
#      each turn's request is still byte-identical up to its own breakpoint.
# A prefix under ~1024 tokens (Sonnet's minimum) is silently not cached --
# no error, just no benefit, so this is always safe to include even on a
# short first turn.

def _cached_system(system_text: str) -> list[dict]:
    return [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]


def _cached_tools(definitions: list[dict]) -> list[dict]:
    if not definitions:
        return definitions
    return [
        {**d, "cache_control": {"type": "ephemeral"}} if i == len(definitions) - 1 else d
        for i, d in enumerate(definitions)
    ]


def _with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Return messages with a cache breakpoint on the last block of the last
    message -- a copy, so the cache_control marker never leaks into
    session.history itself (which gets persisted and resent as plain dicts;
    it should stay exactly what was actually said, not carry API-only
    hints from whichever turn happened to be last when it was added)."""
    if not messages:
        return messages
    last = messages[-1]
    content = last["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    if not blocks:
        return messages
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return [*messages[:-1], {**last, "content": blocks}]


def _serialize_content_block(block) -> dict:
    """Minimal, explicit reconstruction of an assistant content block for
    session.history — deliberately not block.model_dump(), which includes
    response-only fields the API rejects if they're sent back in a later
    request. Only the two block types Sherlock's loop actually produces
    are whitelisted; anything else falls back to model_dump() as a
    best-effort rather than dropping the block entirely."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump()


async def run(
    session: Session,
    instrument_ctx: InstrumentContext | None,
    agent_docs: list[tuple[str, str]],
    question: str = "",
    operator_id: str = "",
    operator_name: str = "",
    channel: str = "",
) -> AsyncGenerator[DiagnoseChunk, None]:
    """Run or continue the investigation for session.entity_id.

    question/operator_id/operator_name describe *this specific call*, not
    the session as a whole — a thread can have several operators, so
    identity is logged per exchange to sherlock_audit, not stored on the
    session. question is whatever the caller actually sent this turn (the
    synthesized entity-investigation prompt, a chat message, a reply, or a
    full re-fetched Slack transcript) — used only for the audit record, not
    reasoning.

    Yields DiagnoseChunk objects:
      type="step"         — streaming text token or tool call announcement
      type="evidence"     — raw tool result (shown in UI evidence panel)
      type="hypothesis"   — final classification (data from submit_hypothesis)
      type="question"     — operator question (data from ask_operator)
      type="error"        — unrecoverable error
      type="done"         — investigation complete
    """
    prior_memory = await mem.load(session.instrument_id) if session.instrument_id else []
    system = prompt.build(session.entity_id, instrument_ctx, agent_docs, prior_memory)
    # Built once per run() call, not per turn -- both are identical every
    # turn of this investigation, so precomputing avoids rebuilding the same
    # cache_control-tagged structure on every iteration of the loop below.
    cached_system = _cached_system(system)
    cached_tools = _cached_tools(DEFINITIONS)

    # Seed the conversation if this is a fresh entity investigation. General
    # chat sessions (no entity_id) are seeded by the /chat handler with the
    # operator's actual message instead — nothing to synthesize here.
    if not session.history and session.entity_id:
        session.history.append({
            "role": "user",
            "content": f"Investigate entity `{session.entity_id}` and determine the root cause of its error.",
        })

    try:
        client = _client()
    except RuntimeError as exc:
        yield DiagnoseChunk(type="error", text=str(exc))
        yield _done_chunk(session, successful=False)
        await audit.write(
            session_id=session.id, interface=session.interface,
            operator_id=operator_id, operator_name=operator_name, channel=channel,
            instrument_id=session.instrument_id, entity_id=session.entity_id,
            question=question, response=f"error: {exc}", tools_used=[],
            model=MODEL, cost_usd=0.0, latency_ms=0,
        )
        return

    investigation_start = _time.monotonic()
    tool_call_count = 0
    tools_used: list[str] = []
    # search_kb's known_entities fallback is a full ~285-entity dump (see its
    # own docstring) -- fine once, but every repeat miss within the same
    # investigation was resending the identical ~14KB list, permanently
    # baked into history and re-billed on every turn after. Scoped to this
    # run() call (one question's worth of retries), not the Session itself,
    # so a genuinely new question later still gets the full list fresh.
    kb_known_entities_shown = False

    while session.turn_count < MAX_TURNS:
        session.turn_count += 1
        full_text = ""
        tool_uses = []

        # ── Stream one turn ───────────────────────────────────────────────────
        # Text is buffered, not yielded live token-by-token: whichever turn
        # ends the loop (no further tool calls) IS the final answer, and it
        # has to pass through guardrail.sanitize() before anything reaches
        # the operator or the audit log — sanitization needs the complete
        # text, so nothing can be released mid-stream. We only find out
        # after a turn completes whether it was the final one, so this
        # applies uniformly rather than only to a turn we could've
        # predicted in advance. See helixobs/SHERLOCK_PLATFORM_DESIGN.md's
        # guardrail section.
        turn_start = _time.monotonic()
        first_token_at = None
        async with client.messages.stream(
            model=MODEL,
            system=cached_system,
            messages=_with_cache_breakpoint(session.history),
            tools=cached_tools,
            max_tokens=MAX_TOKENS,
        ) as stream:
            async for event in stream:
                # content_block_start fires for the first block of any kind
                # — text or tool_use — so a turn that goes straight to a
                # tool call with no narration still gets a TTFB sample.
                # content_block_delta alone would miss that case, since it
                # only checked for a .text attribute.
                if first_token_at is None and event.type == "content_block_start":
                    first_token_at = _time.monotonic()
                if (
                    event.type == "content_block_delta"
                    and hasattr(event.delta, "text")
                ):
                    full_text += event.delta.text

            final = await stream.get_final_message()

        if first_token_at is not None:
            mtx.ttfb_seconds.labels(model=MODEL).observe(first_token_at - turn_start)

        # Accumulate token usage for cost estimate. Also kept per-turn (not
        # just accumulated) so each tool call this turn triggered can be
        # attributed to the API call that requested it -- see the evidence
        # chunk below. Note this is the whole turn's usage, not a marginal
        # cost for an individual tool: input_tokens already reflects the
        # full resent history at this point, so with N tool-calling turns
        # in one investigation, each one resends everything before it.
        turn_input_tokens = final.usage.input_tokens if final.usage else 0
        turn_output_tokens = final.usage.output_tokens if final.usage else 0
        if final.usage:
            session.input_tokens  += final.usage.input_tokens
            session.output_tokens += final.usage.output_tokens
            # Reported separately from input_tokens by the API -- a cache
            # write/read is never double-counted in input_tokens itself.
            session.cache_creation_tokens += final.usage.cache_creation_input_tokens or 0
            session.cache_read_tokens     += final.usage.cache_read_input_tokens or 0

        # Collect tool_use blocks from the final message.
        for block in final.content:
            if block.type == "tool_use":
                tool_uses.append(block)

        # Record assistant turn (full content, including tool_use blocks).
        # final.content holds typed Anthropic SDK objects (TextBlock,
        # ToolUseBlock, ...) — every other entry in session.history is
        # already a plain dict (tool results, seed messages), and dicts
        # are what sessions.save() needs to persist history at all.
        #
        # block.model_dump() looked like the obvious conversion, but it
        # dumps every field the SDK's *response* model defines — including
        # response-only fields (e.g. parsed_output) that the API's request
        # schema rejects outright when they're echoed back in a later
        # message's history ("Extra inputs are not permitted"). Rebuilding
        # each block explicitly, with only the fields a request actually
        # accepts, avoids depending on exactly which extra fields today's
        # SDK version happens to attach to a response block.
        session.history.append({
            "role": "assistant",
            "content": [_serialize_content_block(block) for block in final.content],
        })

        # ── No tool calls → model finished speaking without submit_hypothesis ──
        if not tool_uses:
            break

        # This turn continues (more tool calls coming) — full_text here is
        # scene-setting narration ("Let me check X first…"), not a final
        # answer, so it's released as-is rather than sanitized: sanitizing
        # every turn would add a guardrail LLM call per tool-calling round,
        # for text that isn't the thing this guardrail exists to catch.
        if full_text:
            yield DiagnoseChunk(type="step", text=full_text)

        # ── Dispatch tool calls ───────────────────────────────────────────────
        tool_results = []
        for tu in tool_uses:
            # Loop-control tools — interpret directly, don't dispatch.
            if tu.name == "submit_hypothesis":
                # Append tool_result so history stays valid for follow-up replies.
                session.history.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": "Hypothesis submitted.",
                }]})

                # summary/recommendation are the operator-facing text — sanitize
                # both before they reach the chunk's data, not only before the
                # audit record. evidence/gaps are citations of what was found,
                # not directives, and are left as-is.
                sanitized_input = dict(tu.input)
                summary, summary_hit = await guardrail.sanitize(
                    tu.input.get("summary", ""), instrument_ctx,
                )
                recommendation, rec_hit = await guardrail.sanitize(
                    tu.input.get("recommendation", ""), instrument_ctx,
                )
                sanitized_input["summary"] = summary
                sanitized_input["recommendation"] = recommendation
                filter_hit = summary_hit or rec_hit

                yield DiagnoseChunk(
                    type="hypothesis",
                    text=summary,
                    data=sanitized_input,
                )
                # Persist outcome to Tier 2 memory.
                if session.instrument_id:
                    try:
                        hypothesis = HypothesisData(**sanitized_input)
                        await mem.save(session.instrument_id, session.entity_id, hypothesis)
                    except Exception:
                        log.exception("failed to save memory")
                done = _done_chunk(session, successful=True)
                await _persist_usage(session, tool_call_count, investigation_start, reached_hypothesis=True)
                await sessions.save(session)
                response_text = f"{summary} {recommendation}".strip()
                await _persist_audit(
                    session, question, response_text, tools_used,
                    operator_id, operator_name, investigation_start, filter_hit, channel,
                )
                yield done
                return

            if tu.name == "ask_operator":
                yield DiagnoseChunk(
                    type="question",
                    text=tu.input.get("question", ""),
                    data=tu.input,
                )
                # Append a placeholder result so the message list stays valid,
                # then pause — the caller will resume via /reply.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": "[waiting for operator reply]",
                })
                session.history.append({"role": "user", "content": tool_results})
                # Persist partial usage so sherlock_usage always has a record.
                await _persist_usage(session, tool_call_count, investigation_start, reached_hypothesis=False)
                # Durable now — a reply hours or days later resumes this exact state.
                await sessions.save(session)
                # Not sanitized: this is Sherlock asking the operator something,
                # not delivering a directive — lower risk than the other two
                # checkpoints, and out of scope for this guardrail deliberately.
                await _persist_audit(
                    session, question, tu.input.get("question", ""), tools_used,
                    operator_id, operator_name, investigation_start, False, channel,
                )
                return

            # Real investigation tool — announce and dispatch.
            args_summary = ", ".join(f"{k}={v!r}" for k, v in tu.input.items())
            yield DiagnoseChunk(
                type="step",
                text=f"\n→ **{tu.name}**({args_summary})\n",
            )

            args = dict(tu.input)
            if tu.name in ("fetch_github_file", "fetch_github_blame",
                           "fetch_github_file_history", "search_github_callers") and session.github_token:
                args["_token"] = session.github_token

            tool_start = _time.monotonic()
            try:
                result = await dispatch(tu.name, args)
                mtx.tool_calls_total.labels(tool_name=tu.name, status="success").inc()
            except Exception as exc:
                result = {"error": str(exc)}
                mtx.tool_calls_total.labels(tool_name=tu.name, status="failed").inc()
            finally:
                mtx.tool_call_duration_seconds.labels(tool_name=tu.name).observe(
                    _time.monotonic() - tool_start
                )

            if tu.name == "search_kb" and isinstance(result, dict):
                result, kb_known_entities_shown = _dedupe_known_entities(result, kb_known_entities_shown)

            tool_call_count += 1
            tools_used.append(tu.name)
            log.info(
                "tool %s(%s) → %d chars (turn used %d in / %d out tokens)",
                tu.name, args_summary, len(json.dumps(result)), turn_input_tokens, turn_output_tokens,
            )

            yield DiagnoseChunk(
                type="evidence",
                data={
                    "tool": tu.name,
                    "args": tu.input,
                    "result": result,
                    # The turn that decided to call this tool, not a marginal
                    # cost for the tool alone -- see the comment above
                    # turn_input_tokens. When a turn requests several tools
                    # at once, they share this same figure; the number to
                    # watch is how it grows turn over turn as history builds.
                    "turn_tokens_input": turn_input_tokens,
                    "turn_tokens_output": turn_output_tokens,
                },
            )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })

        session.history.append({"role": "user", "content": tool_results})

    # Reached turn cap, or the model finished speaking naturally with no
    # further tool calls — either way, full_text from the last turn is the
    # actual answer. It was buffered, not streamed live (see the comment
    # above the streaming block), so this is the first point it can be
    # released — after sanitization, not before.
    sanitized_text, filter_hit = await guardrail.sanitize(full_text.strip(), instrument_ctx)

    if session.turn_count >= MAX_TURNS:
        yield DiagnoseChunk(
            type="step",
            text="\n\n*Turn limit reached. Summarising based on evidence gathered so far.*\n",
        )
    if sanitized_text:
        yield DiagnoseChunk(type="step", text=sanitized_text)

    await _persist_usage(session, tool_call_count, investigation_start, reached_hypothesis=False)
    await sessions.save(session)
    await _persist_audit(
        session, question, sanitized_text, tools_used,
        operator_id, operator_name, investigation_start, filter_hit, channel,
    )
    yield _done_chunk(session, successful=False)


async def _persist_audit(
    session: Session,
    question: str,
    response_text: str,
    tools_used: list[str],
    operator_id: str,
    operator_name: str,
    start: float,
    filter_hit: bool = False,
    channel: str = "",
) -> None:
    cost = _estimate_cost(
        MODEL, session.input_tokens, session.output_tokens,
        session.cache_creation_tokens, session.cache_read_tokens,
    )
    duration_ms = int((_time.monotonic() - start) * 1000)
    await audit.write(
        session_id=session.id,
        interface=session.interface,
        operator_id=operator_id,
        operator_name=operator_name,
        channel=channel,
        instrument_id=session.instrument_id or INSTRUMENT,
        entity_id=session.entity_id,
        question=question,
        response=response_text,
        tools_used=tools_used,
        model=MODEL,
        cost_usd=cost,
        latency_ms=duration_ms,
        filter_hit=filter_hit,
    )


async def _persist_usage(
    session: Session,
    tool_call_count: int,
    start: float,
    reached_hypothesis: bool,
) -> None:
    cost = _estimate_cost(
        MODEL, session.input_tokens, session.output_tokens,
        session.cache_creation_tokens, session.cache_read_tokens,
    )
    duration_ms = int((_time.monotonic() - start) * 1000)
    await mtx.record_usage(
        session_id=session.id,
        instrument_id=session.instrument_id or "",
        entity_id=session.entity_id,
        model=MODEL,
        tokens_input=session.input_tokens,
        tokens_output=session.output_tokens,
        cost_usd=cost,
        duration_ms=duration_ms,
        tool_call_count=tool_call_count,
        reached_hypothesis=reached_hypothesis,
    )


def _done_chunk(session: Session, successful: bool = True) -> DiagnoseChunk:
    cost = _estimate_cost(
        MODEL, session.input_tokens, session.output_tokens,
        session.cache_creation_tokens, session.cache_read_tokens,
    )
    return DiagnoseChunk(
        type="done",
        text="",
        data={
            "input_tokens":         session.input_tokens,
            "output_tokens":        session.output_tokens,
            "cache_creation_tokens": session.cache_creation_tokens,
            "cache_read_tokens":     session.cache_read_tokens,
            "cost_usd":             round(cost, 6),
            "model":                MODEL,
        },
    )
