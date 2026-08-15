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
MAX_TURNS  = 10

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


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = _PRICING.get(model, _DEFAULT_PRICING)
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _client() -> anthropic.AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(api_key=api_key)


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
        async with client.messages.stream(
            model=MODEL,
            system=system,
            messages=session.history,
            tools=DEFINITIONS,
            max_tokens=MAX_TOKENS,
        ) as stream:
            async for event in stream:
                if (
                    event.type == "content_block_delta"
                    and hasattr(event.delta, "text")
                ):
                    full_text += event.delta.text

            final = await stream.get_final_message()

        # Accumulate token usage for cost estimate.
        if final.usage:
            session.input_tokens  += final.usage.input_tokens
            session.output_tokens += final.usage.output_tokens

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
                await _persist_usage(session, tool_call_count, investigation_start, successful=True)
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
                await _persist_usage(session, tool_call_count, investigation_start, successful=False)
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

            tool_call_count += 1
            tools_used.append(tu.name)
            log.info("tool %s → %d chars", tu.name, len(json.dumps(result)))

            yield DiagnoseChunk(
                type="evidence",
                data={"tool": tu.name, "result": result},
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

    await _persist_usage(session, tool_call_count, investigation_start, successful=False)
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
    cost = _estimate_cost(MODEL, session.input_tokens, session.output_tokens)
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
    successful: bool,
) -> None:
    cost = _estimate_cost(MODEL, session.input_tokens, session.output_tokens)
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
        successful=successful,
    )


def _done_chunk(session: Session, successful: bool = True) -> DiagnoseChunk:
    cost = _estimate_cost(MODEL, session.input_tokens, session.output_tokens)
    return DiagnoseChunk(
        type="done",
        text="",
        data={
            "input_tokens":  session.input_tokens,
            "output_tokens": session.output_tokens,
            "cost_usd":      round(cost, 6),
            "model":         MODEL,
        },
    )
