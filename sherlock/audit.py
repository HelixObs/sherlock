"""Sherlock audit log — write-only record of every exchange, in TimescaleDB.

Every question/response pair is logged against the operator's identity,
whatever the interface or outcome. This table is never a retrieval source
for Sherlock itself — the agent loop must never query it. Operators paste
raw, unsanitized material into conversations, so a retrievable audit log
would be a live path for that material to leak back into future answers.
Same reasoning the original Herodotus design applied to its own audit log.

Distinct from sessions.py's sherlock_sessions: that table is read+write,
meant to be read back to resume context. This one is write-only by
contract, not just by convention.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from sherlock.models import AuditEntry

log = logging.getLogger(__name__)

DB_URL = os.environ.get("HERALD_DB_URL", "")
_PAGE_MAX = 500


async def write(
    *,
    session_id: str,
    interface: str,
    operator_id: str,
    operator_name: str,
    instrument_id: str,
    entity_id: str,
    question: str,
    response: str,
    tools_used: list[str],
    model: str,
    cost_usd: float,
    latency_ms: int,
    filter_hit: bool = False,
    channel: str = "",
) -> None:
    """Append one exchange. Never raises — a logging failure must not fail
    the investigation that produced it."""
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
                INSERT INTO sherlock_audit
                    (session_id, interface, operator_id, operator_name, channel,
                     instrument_id, entity_id, profile, kb_version,
                     question, response, tools_used, model, cost_usd,
                     latency_ms, filter_hit)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                session_id, interface, operator_id or "unknown", operator_name,
                channel or None,
                instrument_id or None, entity_id or None,
                os.environ.get("SHERLOCK_PROFILE", "full"), None,
                question, response, tools_used, model, cost_usd, latency_ms,
                filter_hit,
            )
        finally:
            await conn.close()
    except Exception:
        log.exception("audit.write failed")


async def get_all(
    instrument_id: str = "",
    operator_id: str = "",
    operator_name: str = "",
    channel: str = "",
    since: str = "",
    until: str = "",
    limit: int = 100,
    offset: int = 0,
) -> list[AuditEntry]:
    """Paginated, filterable read for GET /audit.

    operator_name is a case-insensitive partial match (ILIKE) — nobody
    filters by the raw Slack operator_id in practice, per the actual
    feedback this was built from. since/until arrive as ISO8601 strings
    and must be parsed to datetime before binding: asyncpg encodes
    timestamptz parameters client-side via its own codec, which requires
    an actual datetime object — a raw str fails there even though the
    query casts the parameter with ::timestamptz (that cast only tells
    Postgres how to type-check the expression, it doesn't make asyncpg
    parse the string first). Passed through as NULL rather than '' when
    unset, since casting an empty string would error regardless of OR
    short-circuiting.
    """
    if not DB_URL:
        return []
    try:
        import asyncpg
    except ImportError:
        return []

    def _parse_ts(s: str) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None

    since_dt = _parse_ts(since)
    until_dt = _parse_ts(until)

    limit = max(1, min(limit, _PAGE_MAX))
    try:
        conn = await asyncpg.connect(DB_URL)
        try:
            rows = await conn.fetch(
                """
                SELECT id, ts, session_id, interface, operator_id, operator_name,
                       channel, instrument_id, entity_id, profile, kb_version,
                       question, response, tools_used, model, cost_usd,
                       latency_ms, filter_hit
                FROM sherlock_audit
                WHERE ($1 = '' OR instrument_id = $1)
                  AND ($2 = '' OR operator_id = $2)
                  AND ($3 = '' OR operator_name ILIKE '%' || $3 || '%')
                  AND ($4 = '' OR channel = $4)
                  AND ($5::timestamptz IS NULL OR ts >= $5::timestamptz)
                  AND ($6::timestamptz IS NULL OR ts <= $6::timestamptz)
                ORDER BY ts DESC
                LIMIT $7 OFFSET $8
                """,
                instrument_id, operator_id, operator_name, channel,
                since_dt, until_dt,
                limit, offset,
            )
        finally:
            await conn.close()
    except Exception:
        log.exception("audit.get_all failed")
        return []

    return [
        AuditEntry(
            id=row["id"],
            ts=row["ts"].isoformat(),
            session_id=row["session_id"],
            interface=row["interface"],
            operator_id=row["operator_id"],
            operator_name=row["operator_name"] or "",
            channel=row["channel"] or "",
            instrument_id=row["instrument_id"] or "",
            entity_id=row["entity_id"] or "",
            profile=row["profile"],
            kb_version=row["kb_version"] or "",
            question=row["question"],
            response=row["response"],
            tools_used=list(row["tools_used"] or []),
            model=row["model"] or "",
            cost_usd=float(row["cost_usd"]),
            latency_ms=row["latency_ms"],
            filter_hit=row["filter_hit"],
        )
        for row in rows
    ]
