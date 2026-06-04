"""Herald / TimescaleDB investigation tools.

query_entity          — fetch a single entity's metadata and events from the graph API
query_entity_ancestors — fetch the full provenance DAG for an entity
query_similar_errors  — find other entities with helix.error events in a time window
                        (queries TimescaleDB directly — HERALD_DB_URL required)
"""

from __future__ import annotations

import os

import httpx

from .grafana import tempo_url

HERALD_URL   = os.environ.get("HERALD_URL",    "http://herald:8080")
HERALD_DB_URL = os.environ.get("HERALD_DB_URL", "")  # direct DB access for similar-error scan

# ── Tool implementations ──────────────────────────────────────────────────────

async def query_entity_events(entity_id: str) -> dict:
    """Fetch all span events recorded for an entity from TimescaleDB.

    Returns helix.error messages, helix.event.* signals, and their metadata.
    Always call this first when an entity has has_error=true — the error message
    is here, not in the provenance graph.
    """
    if not HERALD_DB_URL:
        return {
            "error": "HERALD_DB_URL not configured",
            "hint": "Set HERALD_DB_URL in the Sherlock environment.",
        }
    try:
        import asyncpg
    except ImportError:
        return {"error": "asyncpg not installed"}

    sql = """
        SELECT event_name, timestamp_ns, metadata
        FROM entity_events
        WHERE entity_id = $1
        ORDER BY timestamp_ns ASC
    """
    try:
        conn = await asyncpg.connect(HERALD_DB_URL)
        try:
            rows = await conn.fetch(sql, entity_id)
        finally:
            await conn.close()
    except Exception as exc:
        return {"error": f"DB query failed: {exc}"}

    import json
    return {
        "entity_id": entity_id,
        "events": [
            {
                "event_name":   row["event_name"],
                "timestamp_ns": row["timestamp_ns"],
                "metadata":     json.loads(row["metadata"]) if row["metadata"] else {},
            }
            for row in rows
        ],
    }


async def query_entity_operations(entity_id: str) -> dict:
    """Fetch all operations recorded for an entity from TimescaleDB.

    Returns operation name, timestamp, metadata, and any associated error events.
    Use this when entity has_error=true to identify which operation failed and why.
    """
    if not HERALD_DB_URL:
        return {
            "error": "HERALD_DB_URL not configured",
            "hint": "Set HERALD_DB_URL in the Sherlock environment.",
        }
    try:
        import asyncpg
    except ImportError:
        return {"error": "asyncpg not installed"}

    ops_sql = """
        SELECT operation, trace_id, created_at, metadata
        FROM entity_operations
        WHERE entity_id = $1
        ORDER BY created_at ASC
    """
    events_sql = """
        SELECT event_name, timestamp_ns, metadata
        FROM entity_events
        WHERE entity_id = $1
        ORDER BY timestamp_ns ASC
    """
    try:
        conn = await asyncpg.connect(HERALD_DB_URL)
        try:
            ops  = await conn.fetch(ops_sql,    entity_id)
            evts = await conn.fetch(events_sql, entity_id)
        finally:
            await conn.close()
    except Exception as exc:
        return {"error": f"DB query failed: {exc}"}

    import json
    return {
        "entity_id": entity_id,
        "operations": [
            {
                "operation":  row["operation"],
                "trace_id":   row["trace_id"] or "",
                "tempo_url":  tempo_url(row["trace_id"]) if row["trace_id"] else "",
                "created_at": row["created_at"].isoformat(),
                "metadata":   json.loads(row["metadata"]) if row["metadata"] else {},
            }
            for row in ops
        ],
        "events": [
            {
                "event_name":   row["event_name"],
                "timestamp_ns": row["timestamp_ns"],
                "metadata":     json.loads(row["metadata"]) if row["metadata"] else {},
            }
            for row in evts
        ],
    }


async def query_entity(entity_id: str) -> dict:
    """Fetch a single entity's provenance graph (depth=1) from the herald API.

    Returns the entity node's metadata, parent_ids, trace_id, and whether it
    has a helix.error event.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{HERALD_URL}/api/v1/entity/{entity_id}/graph")
        if r.status_code == 404:
            return {"error": f"entity {entity_id!r} not found"}
        if r.status_code != 200:
            return {"error": f"herald returned HTTP {r.status_code}"}

    graph = r.json()
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    entity = nodes.get(entity_id)
    if entity is None:
        return {"error": f"entity {entity_id!r} missing from graph response"}

    tid = entity.get("trace_id", "")
    return {
        "id":           entity["id"],
        "instrument_id": entity.get("instrument_id", ""),
        "trace_id":     tid,
        "tempo_url":    tempo_url(tid) if tid else "",
        "timestamp_ns": entity.get("timestamp_ns", 0),
        "parent_ids":   entity.get("parent_ids", []),
        "metadata":     entity.get("metadata", {}),
        "has_error":    entity.get("has_error", False),
    }


async def query_entity_ancestors(entity_id: str, max_depth: int = 10) -> dict:
    """Fetch the full provenance DAG for an entity from the herald API.

    Returns all ancestor and immediate descendant nodes with edges.
    Use this at Level 3 to check whether parent entities were already degraded.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{HERALD_URL}/api/v1/entity/{entity_id}/graph")
        if r.status_code == 404:
            return {"error": f"entity {entity_id!r} not found"}
        if r.status_code != 200:
            return {"error": f"herald returned HTTP {r.status_code}"}

    graph = r.json()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    return {
        "entity_id":   entity_id,
        "node_count":  len(nodes),
        "edge_count":  len(edges),
        "nodes": [
            {
                "id":        n["id"],
                "has_error": n.get("has_error", False),
                "metadata":  n.get("metadata", {}),
                "parent_ids": n.get("parent_ids", []),
            }
            for n in nodes
        ],
        "edges": edges,
    }


async def query_similar_errors(
    instrument_id: str,
    error_type: str = "",
    operation: str = "",
    window_minutes: int = 60,
    anchor_ts: int = 0,
) -> dict:
    """Find other entities/operations with helix.error events for the same instrument
    in a window centred on anchor_ts (the affected entity's error timestamp).

    anchor_ts is Unix nanoseconds — pass the entity's error timestamp so the
    search covers concurrent failures rather than recent ones. If omitted the
    window falls back to now() which is only correct for live errors.

    For operation errors pass `operation` (e.g. "hdf5-conversion") to restrict the
    search to failures of that same operation — errors in unrelated operations or
    distant DAG ancestors are rarely the root cause.

    Requires HERALD_DB_URL to be set (direct TimescaleDB access).
    Returns entity IDs, timestamps, and error metadata to identify whether
    this error is isolated (one entity) or a pattern (many entities).
    """
    if not HERALD_DB_URL:
        return {
            "error": "HERALD_DB_URL not configured — cannot query similar errors directly",
            "hint":  "Set HERALD_DB_URL in the Sherlock environment to enable this tool.",
        }

    try:
        import asyncpg
    except ImportError:
        return {"error": "asyncpg not installed — add it to pyproject.toml dependencies"}

    import json

    # Anchor the window on the entity's error time when known; fall back to now()
    # for live errors where anchor_ts is not yet available.
    anchor_expr = "to_timestamp($4 / 1e9)" if anchor_ts else "now()"

    try:
        conn = await asyncpg.connect(HERALD_DB_URL)
        try:
            if operation:
                sql = f"""
                    SELECT DISTINCT ee.entity_id, ee.timestamp_ns, ee.metadata
                    FROM entity_events ee
                    JOIN entity_operations eo
                      ON eo.entity_id = ee.entity_id
                     AND eo.operation = $3
                    WHERE ee.instrument_id = $1
                      AND ee.event_name    = 'helix.error'
                      AND ee.created_at   >= {anchor_expr} - ($2 * interval '1 minute')
                      AND ee.created_at   <= {anchor_expr} + interval '10 minutes'
                    ORDER BY ee.timestamp_ns DESC
                    LIMIT 50
                """
                args = [instrument_id, window_minutes, operation]
                if anchor_ts:
                    args.append(anchor_ts)
                rows = await conn.fetch(sql, *args)
            else:
                sql = f"""
                    SELECT ee.entity_id, ee.timestamp_ns, ee.metadata
                    FROM entity_events ee
                    WHERE ee.instrument_id = $1
                      AND ee.event_name    = 'helix.error'
                      AND ee.created_at   >= {anchor_expr} - ($2 * interval '1 minute')
                      AND ee.created_at   <= {anchor_expr} + interval '10 minutes'
                    ORDER BY ee.created_at DESC
                    LIMIT 50
                """
                args = [instrument_id, window_minutes]
                if anchor_ts:
                    args.append(anchor_ts)
                rows = await conn.fetch(sql, *args)
        finally:
            await conn.close()
    except Exception as exc:
        return {"error": f"DB query failed: {exc}"}

    results = [
        {
            "entity_id":    row["entity_id"],
            "timestamp_ns": row["timestamp_ns"],
            "metadata":     json.loads(row["metadata"]) if row["metadata"] else {},
        }
        for row in rows
    ]

    # Filter by error_type if specified (matches metadata.type or metadata.message)
    if error_type:
        results = [
            r for r in results
            if error_type.lower() in str(r["metadata"]).lower()
        ]

    return {
        "instrument_id":  instrument_id,
        "operation":      operation,
        "error_type":     error_type,
        "window_minutes": window_minutes,
        "count":          len(results),
        "pattern":        len(results) > 3,   # simple heuristic — many = pattern
        "entities":       results,
    }


# ── Claude tool definitions ───────────────────────────────────────────────────

DEFINITIONS = [
    {
        "name": "query_entity_events",
        "description": (
            "Fetch all span events (helix.error, helix.event.*) recorded for an entity. "
            "ALWAYS call this first when an entity has has_error=true — the error message "
            "and type live here, not in the provenance graph. Narrate the error event "
            "you find (name, message, timestamp) before proceeding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "The HelixObs entity ID"},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "query_entity_operations",
        "description": (
            "Fetch all operations recorded for an entity (e.g. replication, hdf5-conversion, "
            "registration) along with their metadata. Use this when the error event suggests "
            "a post-creation failure — operations are work done on an existing entity and "
            "have their own error events. Narrate which operation failed and its metadata "
            "(destination, size, etc.) before proceeding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "The HelixObs entity ID"},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "query_entity",
        "description": (
            "Fetch a single entity's metadata, parent_ids, trace_id, and error status "
            "from the HelixObs herald. Use this at the start of an investigation to "
            "understand what the entity is and whether it already has errors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "The HelixObs entity ID"},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "query_entity_ancestors",
        "description": (
            "Fetch the full provenance DAG for an entity — all ancestors and direct "
            "descendants. Use this at Level 3 to check whether parent entities were already "
            "degraded before this error, identifying data quality cascade failures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "The HelixObs entity ID"},
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum ancestor depth to traverse (default 10)",
                },
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "query_similar_errors",
        "description": (
            "Find other entities from the same instrument that had helix.error events "
            "in a window centred on anchor_ts (the affected entity's error timestamp). "
            "Use this to distinguish an isolated failure from a widespread pattern "
            "(code bug vs infrastructure). Always pass anchor_ts from the entity's "
            "error event so you search concurrent failures, not recent ones. "
            "For operation errors always pass `operation` to scope the search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument_id": {
                    "type": "string",
                    "description": "Instrument identifier (e.g. 'CHIMEFRB')",
                },
                "anchor_ts": {
                    "type": "integer",
                    "description": (
                        "Unix nanoseconds of the affected entity's error event — anchors the "
                        "search window to the time when this error occurred. Always pass this "
                        "from query_entity_events so concurrent failures are found, not recent ones."
                    ),
                },
                "window_minutes": {
                    "type": "integer",
                    "description": "Width of the search window in minutes before anchor_ts (default 60)",
                },
                "operation": {
                    "type": "string",
                    "description": (
                        "Operation name to restrict the search to (e.g. 'hdf5-conversion'). "
                        "Required when the error is on an operation — omit for entity-level errors."
                    ),
                },
                "error_type": {
                    "type": "string",
                    "description": "Error type string to filter by (e.g. 'CLASSIFIER_TIMEOUT'). Leave empty to return all errors.",
                },
            },
            "required": ["instrument_id"],
        },
    },
]

HANDLERS = {
    "query_entity_events":    query_entity_events,
    "query_entity_operations": query_entity_operations,
    "query_entity":           query_entity,
    "query_entity_ancestors": query_entity_ancestors,
    "query_similar_errors":   query_similar_errors,
}
