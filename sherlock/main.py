"""Sherlock — FastAPI application entry point."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from sherlock import agent, context, memory as mem, sessions
from sherlock.models import DiagnoseChunk, DiagnoseRequest, MemoryEntry, ReplyRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Sherlock", description="AI troubleshooting agent for HelixObs")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── Diagnose ──────────────────────────────────────────────────────────────────

@app.post("/diagnose/{entity_id}")
async def diagnose(entity_id: str, body: DiagnoseRequest) -> StreamingResponse:
    """Start a new investigation for entity_id.

    Returns a session_id in the first chunk so the caller can send follow-up
    replies via POST /diagnose/{session_id}/reply.
    """
    instrument_ctx = context.load(body.instrument_id) if body.instrument_id else None
    agent_docs = await context.fetch_agent_docs(instrument_ctx) if instrument_ctx else []
    session = sessions.create(entity_id, body.instrument_id, body.github_token)

    # If we have a prior investigation for this entity, replay it from memory
    # instead of running the full agent loop again.
    prior = await mem.load_for_entity(entity_id)
    if prior:
        return StreamingResponse(
            _stream_from_memory(session, prior),
            media_type="application/x-ndjson",
        )

    return StreamingResponse(
        _stream(session, instrument_ctx, agent_docs, session_id_in_first_chunk=session.id),
        media_type="application/x-ndjson",
    )


@app.post("/diagnose/{session_id}/reply")
async def reply(session_id: str, body: ReplyRequest) -> StreamingResponse:
    """Continue an existing investigation with an operator reply."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found or expired")

    # Replace the [waiting for operator reply] placeholder with the real answer.
    _patch_waiting_placeholder(session, body.content)
    sessions.touch(session_id)

    instrument_ctx = context.load(session.instrument_id) if session.instrument_id else None
    agent_docs = await context.fetch_agent_docs(instrument_ctx) if instrument_ctx else []

    return StreamingResponse(
        _stream(session, instrument_ctx, agent_docs),
        media_type="application/x-ndjson",
    )


# ── Memory ────────────────────────────────────────────────────────────────────

@app.get("/memory/{instrument_id}")
async def get_memory(instrument_id: str) -> list[MemoryEntry]:
    """Return all Tier 2 memory entries for an instrument."""
    return await mem.get_all(instrument_id)


@app.delete("/memory/{memory_id}", status_code=204)
async def delete_memory(memory_id: str) -> None:
    """Delete a Tier 2 memory entry by id."""
    await mem.delete(memory_id)


# ── Streaming helpers ─────────────────────────────────────────────────────────

async def _stream_from_memory(
    session: sessions.Session,
    prior: mem.MemoryEntry,
) -> AsyncGenerator[bytes, None]:
    """Replay a prior investigation from memory without calling the API."""
    from sherlock.models import HypothesisData

    age_days = ""
    if prior.created_at:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(prior.created_at)
            delta = datetime.now(timezone.utc) - dt
            age_days = f"{delta.days}d ago" if delta.days > 0 else "today"
        except Exception:
            age_days = prior.created_at[:10]

    yield _encode(DiagnoseChunk(
        type="step",
        text=(
            f"I've investigated `{session.entity_id}` before ({age_days}). "
            f"Here's what I found — let me know if anything has changed or "
            f"if you'd like to dig deeper into a specific area."
        ),
        data={"session_id": session.id},
    ))

    # Seed the session history so follow-up replies have context.
    session.history.append({
        "role": "user",
        "content": f"Investigate entity `{session.entity_id}` and determine the root cause of its error.",
    })
    session.history.append({
        "role": "assistant",
        "content": f"I found a prior investigation for this entity from {age_days}. "
                   f"Classification: {prior.classification} ({prior.confidence} confidence). "
                   f"Summary: {prior.summary}",
    })

    yield _encode(DiagnoseChunk(
        type="hypothesis",
        text=prior.summary,
        data={
            "classification": prior.classification,
            "confidence":     prior.confidence,
            "summary":        prior.summary,
            "evidence":       [f"Prior investigation ({age_days}): {prior.error_type or 'see summary'}", f"Stage: {prior.stage}"] if prior.stage else [f"Prior investigation ({age_days})"],
            "recommendation": prior.recommendation,
            "gaps":           "This is a cached result. Reply to re-investigate or ask a follow-up question.",
        },
    ))

    yield _encode(DiagnoseChunk(type="done", text="", data={
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        "model": "memory",
    }))


def _encode(c: DiagnoseChunk) -> bytes:
    return (json.dumps(c.model_dump()) + "\n").encode()


async def _stream(
    session: sessions.Session,
    instrument_ctx,
    agent_docs: list,
    session_id_in_first_chunk: str = "",
) -> AsyncGenerator[bytes, None]:
    if session_id_in_first_chunk:
        yield _encode(DiagnoseChunk(
            type="step",
            text=f"Starting investigation for `{session.entity_id}`.",
            data={"session_id": session_id_in_first_chunk},
        ))

    try:
        async for chunk in agent.run(session, instrument_ctx, agent_docs):
            yield _encode(chunk)
    except Exception as exc:
        log.exception("agent error")
        yield _encode(DiagnoseChunk(type="error", text=str(exc)))
        yield _encode(DiagnoseChunk(type="done", text=""))


def _patch_waiting_placeholder(session: sessions.Session, operator_reply: str) -> None:
    """Replace the [waiting for operator reply] tool result with the real answer."""
    for msg in reversed(session.history):
        if msg["role"] != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "tool_result"
                and item.get("content") == "[waiting for operator reply]"
            ):
                item["content"] = operator_reply
                return
    # If no placeholder found, append as a plain user message.
    session.history.append({"role": "user", "content": operator_reply})
