"""Sherlock — FastAPI application entry point."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from sherlock import context, sessions
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
    session = sessions.create(entity_id, body.instrument_id)

    return StreamingResponse(
        _run_investigation(session.id, entity_id, instrument_ctx),
        media_type="application/x-ndjson",
    )


@app.post("/diagnose/{session_id}/reply")
async def reply(session_id: str, body: ReplyRequest) -> StreamingResponse:
    """Continue an existing investigation with an operator reply."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found or expired")

    session.history.append({"role": "user", "content": body.content})
    sessions.touch(session_id)

    return StreamingResponse(
        _continue_investigation(session),
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


# ── Investigation generator (stub — Task 5 wires real tools) ─────────────────

async def _run_investigation(
    session_id: str,
    entity_id: str,
    instrument_ctx,
) -> AsyncGenerator[bytes, None]:
    def chunk(c: DiagnoseChunk) -> bytes:
        return (json.dumps(c.model_dump()) + "\n").encode()

    yield chunk(DiagnoseChunk(
        type="step",
        text=f"Starting investigation for entity `{entity_id}`.",
        data={"session_id": session_id},
    ))

    if instrument_ctx:
        yield chunk(DiagnoseChunk(
            type="step",
            text=f"Loaded instrument context for **{instrument_ctx.instrument_id}**: "
                 f"{instrument_ctx.description}",
        ))
    else:
        yield chunk(DiagnoseChunk(
            type="step",
            text="No instrument context found — proceeding without Tier 1 config.",
        ))

    # Placeholder until Task 5 (investigation tools) is implemented.
    yield chunk(DiagnoseChunk(
        type="hypothesis",
        text="Investigation tools not yet wired. This is a skeleton response.",
        data={"classification": "unknown", "confidence": "none"},
    ))

    yield chunk(DiagnoseChunk(type="done", text=""))


async def _continue_investigation(session: sessions.Session) -> AsyncGenerator[bytes, None]:
    def chunk(c: DiagnoseChunk) -> bytes:
        return (json.dumps(c.model_dump()) + "\n").encode()

    yield chunk(DiagnoseChunk(
        type="step",
        text="Follow-up received. Investigation continuation not yet wired (Task 5/6).",
    ))
    yield chunk(DiagnoseChunk(type="done", text=""))
