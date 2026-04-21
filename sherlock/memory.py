"""Tier 2 instrument memory — persisted investigation outcomes in TimescaleDB.

Sherlock reads recent memory entries at investigation start so it can
recognise recurring error patterns and skip redundant tool calls.
Entries are written after submit_hypothesis is called.
"""

from __future__ import annotations

import json
import logging
import os

from sherlock.models import HypothesisData, MemoryEntry

log = logging.getLogger(__name__)

DB_URL = os.environ.get("GATEWAY_DB_URL", "")
_MAX_ENTRIES = 20   # entries fed into the system prompt per investigation


async def load(instrument_id: str) -> list[MemoryEntry]:
    """Return the most recent memory entries for an instrument."""
    if not DB_URL:
        return []
    try:
        import asyncpg
    except ImportError:
        return []
    try:
        conn = await asyncpg.connect(DB_URL)
        try:
            rows = await conn.fetch(
                """
                SELECT id, instrument_id, entity_id, error_type, stage,
                       classification, confidence, summary, recommendation,
                       created_at
                FROM instrument_memory
                WHERE instrument_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                instrument_id, _MAX_ENTRIES,
            )
        finally:
            await conn.close()
    except Exception:
        log.exception("memory.load failed")
        return []

    return [
        MemoryEntry(
            id=row["id"],
            instrument_id=row["instrument_id"],
            entity_id=row["entity_id"],
            error_type=row["error_type"],
            stage=row["stage"],
            classification=row["classification"],
            confidence=row["confidence"],
            summary=row["summary"],
            recommendation=row["recommendation"],
            created_at=row["created_at"].isoformat(),
        )
        for row in rows
    ]


async def save(instrument_id: str, entity_id: str, hypothesis: HypothesisData) -> str | None:
    """Persist a hypothesis outcome. Returns the new entry id, or None on failure."""
    if not DB_URL:
        return None
    try:
        import asyncpg
    except ImportError:
        return None

    evidence_str = "; ".join(hypothesis.evidence) if hypothesis.evidence else ""
    error_type = _infer_error_type(hypothesis)
    stage = _infer_stage(hypothesis)

    try:
        conn = await asyncpg.connect(DB_URL)
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO instrument_memory
                    (instrument_id, entity_id, error_type, stage,
                     classification, confidence, summary, recommendation)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                instrument_id, entity_id, error_type, stage,
                hypothesis.classification, hypothesis.confidence,
                hypothesis.summary, hypothesis.recommendation or "",
            )
        finally:
            await conn.close()
    except Exception:
        log.exception("memory.save failed")
        return None

    return row["id"] if row else None


async def get_all(instrument_id: str) -> list[MemoryEntry]:
    """Return all memory entries for an instrument (for the /memory API)."""
    return await load(instrument_id)


async def delete(memory_id: str) -> None:
    """Delete a memory entry by id."""
    if not DB_URL:
        return
    try:
        import asyncpg
        conn = await asyncpg.connect(DB_URL)
        try:
            await conn.execute("DELETE FROM instrument_memory WHERE id = $1", memory_id)
        finally:
            await conn.close()
    except Exception:
        log.exception("memory.delete failed")


def _infer_error_type(h: HypothesisData) -> str:
    """Extract a short error type token from the summary/evidence."""
    text = (h.summary + " " + " ".join(h.evidence)).lower()
    for token in ["replication_timeout", "classifier_timeout", "disk_full",
                  "write_timeout", "model_load_failure", "oom"]:
        if token.replace("_", " ") in text or token in text:
            return token
    return ""


def _infer_stage(h: HypothesisData) -> str:
    text = (h.summary + " " + " ".join(h.evidence)).lower()
    for stage in ["replication", "hdf5_archiver", "frb_classifier",
                  "beam_former", "candidate_filter", "registration"]:
        if stage.replace("_", " ") in text or stage in text:
            return stage
    return ""
