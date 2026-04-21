"""Loki log query tool.

Queries Loki for logs correlated with a specific entity_id in a time window.
helix_entity_id is NOT a Loki stream label (cardinality constraint) — it is
filtered at query time using the JSON pipeline operator.
"""

from __future__ import annotations

import os

import httpx

LOKI_URL = os.environ.get("LOKI_URL", "http://loki:3100")

# Cap log lines returned to the agent to avoid token bloat.
_MAX_LINES = 50

# ── Tool implementation ───────────────────────────────────────────────────────

async def query_loki(entity_id: str, start_ts: int, end_ts: int) -> dict:
    """Query Loki for log lines associated with entity_id in [start_ts, end_ts].

    Timestamps are Unix nanoseconds (matching the OTel/HelixObs convention).
    Returns up to _MAX_LINES log lines ordered oldest-first.
    """
    # Loki query: stream selector + JSON filter on helix_entity_id.
    # {job=~".+"} selects all streams; the json filter narrows to the entity.
    logql = f'{{job=~".+"}} | json | helix_entity_id=`{entity_id}`'

    params = {
        "query": logql,
        "start": str(start_ts),
        "end":   str(end_ts),
        "limit": str(_MAX_LINES),
        "direction": "forward",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{LOKI_URL}/loki/api/v1/query_range", params=params)
        if r.status_code != 200:
            return {"error": f"Loki returned HTTP {r.status_code}: {r.text[:200]}"}

    data = r.json()
    results = data.get("data", {}).get("result", [])

    lines = []
    for stream in results:
        for ts, msg in stream.get("values", []):
            lines.append({"ts": ts, "msg": msg})

    lines.sort(key=lambda x: x["ts"])
    lines = lines[:_MAX_LINES]

    return {
        "entity_id":   entity_id,
        "start_ts":    start_ts,
        "end_ts":      end_ts,
        "line_count":  len(lines),
        "lines":       lines,
    }


# ── Claude tool definition ────────────────────────────────────────────────────

DEFINITIONS = [
    {
        "name": "query_loki",
        "description": (
            "Fetch log lines for a specific entity_id from Loki in a given time window. "
            "Use this at Level 2 of the investigation ladder to find warnings or exceptions "
            "preceding an error. Timestamps are Unix nanoseconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The HelixObs entity ID to filter logs for",
                },
                "start_ts": {
                    "type": "integer",
                    "description": "Window start as Unix nanoseconds (e.g. error_ts - 300_000_000_000 for -5 min)",
                },
                "end_ts": {
                    "type": "integer",
                    "description": "Window end as Unix nanoseconds (e.g. error_ts + 60_000_000_000 for +1 min)",
                },
            },
            "required": ["entity_id", "start_ts", "end_ts"],
        },
    },
]

HANDLERS = {
    "query_loki": query_loki,
}
