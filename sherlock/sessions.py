"""In-memory session store with 30-minute TTL.

Sessions hold the conversation history for an ongoing Sherlock investigation.
Next.js owns auth and passes a session_id it generated — Sherlock trusts it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

_TTL = 30 * 60  # seconds


@dataclass
class Session:
    id: str
    entity_id: str
    instrument_id: str
    history: list[dict] = field(default_factory=list)
    turn_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    github_token: str = ""   # operator-supplied PAT; lives only in memory, never persisted
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


_store: dict[str, Session] = {}


def create(entity_id: str, instrument_id: str, github_token: str = "") -> Session:
    s = Session(id=str(uuid.uuid4()), entity_id=entity_id,
                instrument_id=instrument_id, github_token=github_token)
    _store[s.id] = s
    return s


def get(session_id: str) -> Session | None:
    s = _store.get(session_id)
    if s is None:
        return None
    if time.time() - s.updated_at > _TTL:
        del _store[session_id]
        return None
    return s


def touch(session_id: str) -> None:
    if session_id in _store:
        _store[session_id].updated_at = time.time()


def expire_old() -> None:
    now = time.time()
    dead = [sid for sid, s in _store.items() if now - s.updated_at > _TTL]
    for sid in dead:
        del _store[sid]
