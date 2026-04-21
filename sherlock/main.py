"""Sherlock — FastAPI application entry point."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from sherlock import agent, context, sessions
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
    session = sessions.create(entity_id, body.instrument_id)

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
    """Return all Tier 2 memory entries for an instrument. (Stub — Task 8)"""
    return []


@app.post("/memory/{instrument_id}", status_code=201)
async def write_memory(instrument_id: str, entry: MemoryEntry) -> MemoryEntry:
    """Save a new Tier 2 memory entry. (Stub — Task 8)"""
    entry.instrument_id = instrument_id
    return entry


@app.delete("/memory/{memory_id}", status_code=204)
async def delete_memory(memory_id: str) -> None:
    """Delete a Tier 2 memory entry by id. (Stub — Task 8)"""


# ── Streaming helpers ─────────────────────────────────────────────────────────

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
