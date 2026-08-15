"""Sherlock — FastAPI application entry point."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from sherlock import agent, audit, context, memory as mem, metrics as mtx, sessions
from sherlock.models import (
    AuditEntry,
    ChatRequest,
    DiagnoseChunk,
    DiagnoseRequest,
    MemoryEntry,
    ReplyRequest,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Sherlock", description="AI troubleshooting agent for HelixObs")
mtx.start_metrics_server()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── Diagnose ──────────────────────────────────────────────────────────────────

@app.post("/diagnose/{entity_id}")
async def diagnose(entity_id: str, body: DiagnoseRequest) -> StreamingResponse:
    """Start or resume an investigation for entity_id.

    Pass body.session_id for a deterministic, resumable key (e.g. a Slack
    thread's slack:{channel}:{thread_ts}); omit it for a server-minted UUID.
    Either way the session_id is returned in the first chunk so the caller
    can send follow-up replies via POST /diagnose/{session_id}/reply.
    """
    instrument_ctx = context.load(body.instrument_id) if body.instrument_id else None
    # Operator-supplied token takes precedence; fall back to instrument config token.
    github_token = body.github_token or (instrument_ctx.github_token if instrument_ctx else "")
    agent_docs = await context.fetch_agent_docs(instrument_ctx, github_token) if instrument_ctx else []
    session = await sessions.get_or_create(
        session_id=body.session_id, entity_id=entity_id,
        instrument_id=body.instrument_id, interface=body.interface,
        github_token=github_token,
    )

    # Held from here through the end of the streamed turn (released inside
    # _stream/_stream_from_memory's finally) — serializes concurrent
    # requests against the same session_id, e.g. two operators messaging
    # the same Slack thread within a few seconds of each other. See
    # sessions.lock_for.
    lock = sessions.lock_for(session.id)
    await lock.acquire()

    # From here until a StreamingResponse is actually returned, an
    # exception would otherwise leak the lock — nothing left to release it,
    # permanently deadlocking this session_id. Release-and-reraise instead.
    try:
        question = f"Investigate entity `{entity_id}` and determine the root cause of its error."

        # A resumed session already has real history — run it like any
        # other continuation. Only a genuinely fresh session checks Tier 2
        # memory for a prior investigation to replay instead of running
        # the full agent loop.
        if not session.history:
            prior = await mem.load_for_entity(entity_id)
            if prior:
                return StreamingResponse(
                    _stream_from_memory(session, prior, question, body.operator_id, body.operator_name, body.channel, lock=lock),
                    media_type="application/x-ndjson",
                )

        return StreamingResponse(
            _stream(
                session, instrument_ctx, agent_docs, question,
                body.operator_id, body.operator_name, body.channel,
                session_id_in_first_chunk=session.id, lock=lock,
            ),
            media_type="application/x-ndjson",
        )
    except Exception:
        lock.release()
        raise


@app.post("/diagnose/{session_id}/reply")
async def reply(session_id: str, body: ReplyRequest) -> StreamingResponse:
    """Continue an existing investigation with an operator reply."""
    return await _reply(session_id, body.content, body.operator_id, body.operator_name, body.channel)


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    """Start or resume a general conversation — no entity required.

    Pass body.session_id for a deterministic, resumable key (e.g. a Slack
    thread's slack:{channel}:{thread_ts}); omit it for a server-minted UUID.

    body.replace_history=True (Slack): body.message is treated as the
    complete, already-cumulative thread transcript fetched fresh from Slack,
    and *replaces* session.history rather than being appended to it — the
    fetched transcript already contains every prior turn as plain text, so
    appending on top of the existing structured history would resend the
    same conversation repeatedly. See helixobs/SHERLOCK_PLATFORM_DESIGN.md §6.

    body.replace_history=False (web, default): safe to call against an
    existing session_id too — a resumed session's message is appended via
    the same placeholder-aware logic /reply uses, so this can double as
    "start or continue" for a caller that doesn't track which it is.
    """
    instrument_ctx = context.load(body.instrument_id) if body.instrument_id else None
    github_token = body.github_token or (instrument_ctx.github_token if instrument_ctx else "")
    agent_docs = await context.fetch_agent_docs(instrument_ctx, github_token) if instrument_ctx else []
    session = await sessions.get_or_create(
        session_id=body.session_id, entity_id="",
        instrument_id=body.instrument_id, interface=body.interface,
        github_token=github_token,
    )

    # See the matching comment in diagnose() — held through the end of the
    # streamed turn, released inside _stream's finally.
    lock = sessions.lock_for(session.id)
    await lock.acquire()

    # See the matching comment in diagnose() — release-and-reraise so a
    # failure here can't leak the lock and deadlock this session_id.
    try:
        if body.replace_history:
            session.history = [{"role": "user", "content": body.message}]
            # Each invocation is a fresh, bounded reasoning task over freshly
            # replaced context — turn_count must reset per invocation here, not
            # accumulate over the session's lifetime like it does for a single
            # bounded entity investigation, or a long-lived thread would
            # silently hit MAX_TURNS after a handful of unrelated exchanges.
            # input_tokens/output_tokens stay cumulative — that's cost tracking,
            # not loop control, and resetting it would undercount real spend.
            session.turn_count = 0
            await sessions.touch(session.id)
        elif not session.history:
            session.history.append({"role": "user", "content": body.message})
        else:
            _patch_waiting_placeholder(session, body.message)
            await sessions.touch(session.id)

        return StreamingResponse(
            _stream(
                session, instrument_ctx, agent_docs, body.message,
                body.operator_id, body.operator_name, body.channel,
                session_id_in_first_chunk=session.id, lock=lock,
            ),
            media_type="application/x-ndjson",
        )
    except Exception:
        lock.release()
        raise


@app.post("/chat/{session_id}/reply")
async def chat_reply(session_id: str, body: ReplyRequest) -> StreamingResponse:
    """Continue an existing chat with an operator reply.

    Identical mechanics to /diagnose/{session_id}/reply — entity vs. chat
    sessions differ only in how they were seeded, not in how a reply is
    handled, so both routes share _reply().
    """
    return await _reply(session_id, body.content, body.operator_id, body.operator_name, body.channel)


async def _reply(
    session_id: str, content: str,
    operator_id: str = "", operator_name: str = "", channel: str = "",
) -> StreamingResponse:
    session = await sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found or expired")

    # See the matching comment in diagnose() — held through the end of the
    # streamed turn, released inside _stream's finally.
    lock = sessions.lock_for(session.id)
    await lock.acquire()

    # See the matching comment in diagnose() — release-and-reraise so a
    # failure here can't leak the lock and deadlock this session_id.
    try:
        # Replace the [waiting for operator reply] placeholder with the real answer.
        _patch_waiting_placeholder(session, content)
        await sessions.touch(session_id)

        instrument_ctx = context.load(session.instrument_id) if session.instrument_id else None
        agent_docs = await context.fetch_agent_docs(instrument_ctx) if instrument_ctx else []

        return StreamingResponse(
            _stream(session, instrument_ctx, agent_docs, content, operator_id, operator_name, channel, lock=lock),
            media_type="application/x-ndjson",
        )
    except Exception:
        lock.release()
        raise


# ── Memory ────────────────────────────────────────────────────────────────────

@app.get("/memory/{instrument_id}")
async def get_memory(instrument_id: str) -> list[MemoryEntry]:
    """Return all Tier 2 memory entries for an instrument."""
    return await mem.get_all(instrument_id)


@app.delete("/memory/{memory_id}", status_code=204)
async def delete_memory(memory_id: str) -> None:
    """Delete a Tier 2 memory entry by id."""
    await mem.delete(memory_id)


# ── Audit ─────────────────────────────────────────────────────────────────────

@app.get("/audit")
async def get_audit(
    instrument_id: str = "",
    operator_id: str = "",
    operator_name: str = "",
    channel: str = "",
    since: str = "",
    until: str = "",
    limit: int = 100,
    offset: int = 0,
) -> list[AuditEntry]:
    """Paginated, filterable audit log. Write-only from Sherlock's own
    perspective — this endpoint is for human review, never called by the
    agent loop itself.

    operator_name is a partial, case-insensitive match — filtering by the
    raw operator_id isn't something anyone can actually do without looking
    it up first. since/until are ISO8601 timestamps."""
    return await audit.get_all(
        instrument_id, operator_id, operator_name, channel, since, until, limit, offset,
    )


# ── Streaming helpers ─────────────────────────────────────────────────────────

async def _stream_from_memory(
    session: sessions.Session,
    prior: mem.MemoryEntry,
    question: str = "",
    operator_id: str = "",
    operator_name: str = "",
    channel: str = "",
    lock: asyncio.Lock | None = None,
) -> AsyncGenerator[bytes, None]:
    """Replay a prior investigation from memory without calling the API."""
    from sherlock.models import HypothesisData

    try:
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

        await audit.write(
            session_id=session.id, interface=session.interface,
            operator_id=operator_id, operator_name=operator_name, channel=channel,
            instrument_id=session.instrument_id, entity_id=session.entity_id,
            question=question, response=prior.summary, tools_used=[],
            model="memory", cost_usd=0.0, latency_ms=0,
        )
    finally:
        if lock is not None and lock.locked():
            lock.release()


def _encode(c: DiagnoseChunk) -> bytes:
    return (json.dumps(c.model_dump()) + "\n").encode()


async def _stream(
    session: sessions.Session,
    instrument_ctx,
    agent_docs: list,
    question: str = "",
    operator_id: str = "",
    operator_name: str = "",
    channel: str = "",
    session_id_in_first_chunk: str = "",
    lock: asyncio.Lock | None = None,
) -> AsyncGenerator[bytes, None]:
    try:
        if session_id_in_first_chunk:
            text = (
                f"Starting investigation for `{session.entity_id}`."
                if session.entity_id else "..."
            )
            yield _encode(DiagnoseChunk(
                type="step",
                text=text,
                data={"session_id": session_id_in_first_chunk},
            ))

        try:
            async for chunk in agent.run(session, instrument_ctx, agent_docs, question, operator_id, operator_name, channel):
                yield _encode(chunk)
        except Exception as exc:
            log.exception("agent error")
            yield _encode(DiagnoseChunk(type="error", text=str(exc)))
            yield _encode(DiagnoseChunk(type="done", text=""))
            await audit.write(
                session_id=session.id, interface=session.interface,
                operator_id=operator_id, operator_name=operator_name, channel=channel,
                instrument_id=session.instrument_id, entity_id=session.entity_id,
                question=question, response=f"error: {exc}", tools_used=[],
                model="", cost_usd=0.0, latency_ms=0,
            )
    finally:
        if lock is not None and lock.locked():
            lock.release()


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
