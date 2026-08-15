"""Session store with a durable TimescaleDB backing store.

The in-memory dict is a hot cache, not the source of truth: sessions are
write-through persisted to `sherlock_sessions` (see herald migration
011_sherlock_sessions.sql), and a cache miss — whether from idle eviction or
a process restart — falls back to a DB read before giving up. This means the
idle TTL only tunes memory usage; it no longer determines whether a
conversation is recoverable.

Session ids are either a server-minted UUID (web) or a client-supplied,
deterministic key derived from a Slack thread (slack:{channel_id}:{thread_ts}),
so a reply landing in an old thread resumes the exact prior conversation
regardless of how much time has passed.

`github_token` is deliberately excluded from persistence — it lives only in
the in-memory Session and is lost on cache eviction; the operator resupplies
it if needed, since it isn't part of the conversational state.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DB_URL = os.environ.get("HERALD_DB_URL", "")
_HOT_TTL = 30 * 60  # seconds — RAM-tuning only; DB is the durable copy


@dataclass
class Session:
    id: str
    entity_id: str
    instrument_id: str
    interface: str = "web"   # "web" | "slack"
    history: list[dict] = field(default_factory=list)
    turn_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    github_token: str = ""   # operator-supplied PAT; lives only in memory, never persisted
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


_store: dict[str, Session] = {}


async def get_or_create(
    session_id: str = "",
    entity_id: str = "",
    instrument_id: str = "",
    interface: str = "web",
    github_token: str = "",
) -> Session:
    """Resume session_id if it exists (memory, then DB); otherwise create it.

    Pass an explicit session_id for a deterministic key (Slack threads);
    leave it empty for a server-minted UUID (web default). Safe to call
    against a session_id that already has durable state — e.g. after a bot
    restart re-derives the same slack:{channel}:{thread_ts} key — since the
    DB is checked before anything is created fresh.
    """
    if session_id:
        existing = await get(session_id)
        if existing is not None:
            return existing
    else:
        session_id = str(uuid.uuid4())

    s = Session(
        id=session_id, entity_id=entity_id, instrument_id=instrument_id,
        interface=interface, github_token=github_token,
    )
    _store[s.id] = s
    await save(s)
    return s


async def get(session_id: str) -> Session | None:
    """Return the Session for session_id, checking the hot cache then the DB."""
    s = _store.get(session_id)
    if s is not None:
        if time.time() - s.updated_at > _HOT_TTL:
            del _store[session_id]  # evict; still recoverable from DB below
        else:
            return s

    s = await _load(session_id)
    if s is not None:
        _store[s.id] = s
    return s


async def touch(session_id: str) -> None:
    """Cheap in-memory access-time bump; call save() to persist real changes."""
    if session_id in _store:
        _store[session_id].updated_at = time.time()


async def save(s: Session) -> None:
    """Write-through persist the full session state to sherlock_sessions.

    Never raises — a persistence hiccup degrades to "this turn isn't
    durable yet," not a failed investigation. The in-memory copy (the one
    the agent loop is actually using) is unaffected either way.
    """
    s.updated_at = time.time()
    if not DB_URL:
        return
    try:
        import asyncpg
    except ImportError:
        return
    try:
        conn = await asyncpg.connect(DB_URL)
        try:
            await conn.execute(
                """
                INSERT INTO sherlock_sessions
                    (id, interface, entity_id, instrument_id, history,
                     turn_count, input_tokens, output_tokens, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, now())
                ON CONFLICT (id) DO UPDATE SET
                    history       = EXCLUDED.history,
                    turn_count    = EXCLUDED.turn_count,
                    input_tokens  = EXCLUDED.input_tokens,
                    output_tokens = EXCLUDED.output_tokens,
                    updated_at    = now()
                """,
                s.id, s.interface, s.entity_id, s.instrument_id,
                # default=str is a backstop, not the fix — history should
                # always be plain dicts by the time it gets here (agent.py
                # converts SDK response objects before appending). This just
                # means a future oversight degrades to a garbled-but-present
                # string instead of silently failing the whole write.
                json.dumps(s.history, default=str), s.turn_count, s.input_tokens, s.output_tokens,
            )
        finally:
            await conn.close()
    except Exception:
        log.exception("sessions.save failed for %s", s.id)


async def _load(session_id: str) -> Session | None:
    if not DB_URL:
        return None
    try:
        import asyncpg
    except ImportError:
        return None
    try:
        conn = await asyncpg.connect(DB_URL)
        try:
            row = await conn.fetchrow(
                """
                SELECT id, interface, entity_id, instrument_id, history,
                       turn_count, input_tokens, output_tokens
                FROM sherlock_sessions
                WHERE id = $1
                """,
                session_id,
            )
        finally:
            await conn.close()
    except Exception:
        log.exception("sessions.load failed for %s", session_id)
        return None

    if row is None:
        return None

    history = row["history"]
    if isinstance(history, str):
        history = json.loads(history)

    return Session(
        id=row["id"],
        entity_id=row["entity_id"],
        instrument_id=row["instrument_id"],
        interface=row["interface"],
        history=history,
        turn_count=row["turn_count"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
    )


def expire_old() -> None:
    """Prune the in-memory cache. Harmless — evicted sessions stay in the DB."""
    now = time.time()
    dead = [sid for sid, s in _store.items() if now - s.updated_at > _HOT_TTL]
    for sid in dead:
        del _store[sid]
