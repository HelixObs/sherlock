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
from collections.abc import AsyncGenerator

import anthropic

from sherlock import prompt
from sherlock.models import DiagnoseChunk, InstrumentContext
from sherlock.sessions import Session
from sherlock.tools import DEFINITIONS, dispatch

log = logging.getLogger(__name__)

MODEL          = os.environ.get("SHERLOCK_MODEL", "claude-sonnet-4-6")
MAX_TOKENS     = 4096
MAX_TURNS      = 10


def _client() -> anthropic.AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(api_key=api_key)


async def run(
    session: Session,
    instrument_ctx: InstrumentContext | None,
    agent_docs: list[tuple[str, str]],
) -> AsyncGenerator[DiagnoseChunk, None]:
    """Run or continue the investigation for session.entity_id.

    Yields DiagnoseChunk objects:
      type="step"         — streaming text token or tool call announcement
      type="hypothesis"   — final classification (data from submit_hypothesis)
      type="question"     — operator question (data from ask_operator)
      type="error"        — unrecoverable error
      type="done"         — investigation complete
    """
    system = prompt.build(session.entity_id, instrument_ctx, agent_docs)

    # Seed the conversation if this is a fresh session.
    if not session.history:
        session.history.append({
            "role": "user",
            "content": f"Investigate entity `{session.entity_id}` and determine the root cause of its error.",
        })

    try:
        client = _client()
    except RuntimeError as exc:
        yield DiagnoseChunk(type="error", text=str(exc))
        yield DiagnoseChunk(type="done", text="")
        return

    while session.turn_count < MAX_TURNS:
        session.turn_count += 1
        full_text = ""
        tool_uses = []

        # ── Stream one turn ───────────────────────────────────────────────────
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
                    yield DiagnoseChunk(type="step", text=event.delta.text)

            final = await stream.get_final_message()

        # Collect tool_use blocks from the final message.
        for block in final.content:
            if block.type == "tool_use":
                tool_uses.append(block)

        # Record assistant turn (full content, including tool_use blocks).
        session.history.append({
            "role": "assistant",
            "content": final.content,
        })

        # ── No tool calls → model finished speaking, no structured output ────
        if not tool_uses:
            break

        # ── Dispatch tool calls ───────────────────────────────────────────────
        tool_results = []
        for tu in tool_uses:
            # Loop-control tools — interpret directly, don't dispatch.
            if tu.name == "submit_hypothesis":
                yield DiagnoseChunk(
                    type="hypothesis",
                    text=tu.input.get("summary", ""),
                    data=tu.input,
                )
                yield DiagnoseChunk(type="done", text="")
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
                return

            # Real investigation tool — announce and dispatch.
            args_summary = ", ".join(f"{k}={v!r}" for k, v in tu.input.items())
            yield DiagnoseChunk(
                type="step",
                text=f"\n→ **{tu.name}**({args_summary})\n",
            )

            result = await dispatch(tu.name, tu.input)
            log.info("tool %s → %d chars", tu.name, len(json.dumps(result)))

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })

        session.history.append({"role": "user", "content": tool_results})

    # Reached turn cap without a submit_hypothesis call.
    if session.turn_count >= MAX_TURNS:
        yield DiagnoseChunk(
            type="step",
            text="\n\n*Turn limit reached. Summarising based on evidence gathered so far.*\n",
        )

    yield DiagnoseChunk(type="done", text="")
