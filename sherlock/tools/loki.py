"""Loki log query tool.

Queries Loki for logs correlated with a specific entity_id in a time window.
helix_entity_id is NOT a Loki stream label (cardinality constraint) — it is
filtered at query time as a structured metadata filter (| helix_entity_id = ...).
Pass instrument_id to narrow the stream selector and avoid a full cross-instrument scan.
"""

from __future__ import annotations

import json
import os

import httpx

from .grafana import loki_url

LOKI_URL = os.environ.get("LOKI_URL", "http://loki:3100")

# Cap log lines returned to the agent to avoid token bloat.
_MAX_LINES = 50

# ── Tool implementation ───────────────────────────────────────────────────────

_MIN_WINDOW_NS = 5 * 60 * 1_000_000_000   # 5 minutes in nanoseconds


async def query_loki(entity_id: str, start_ts: int, end_ts: int, instrument_id: str | None = None) -> dict:
    """Query Loki for log lines associated with entity_id in [start_ts, end_ts].

    Timestamps are Unix nanoseconds (matching the OTel/HelixObs convention).
    Returns up to _MAX_LINES log lines ordered oldest-first.
    """
    # Enforce a minimum window so the Grafana link is always useful even if
    # the caller passes nearly-identical timestamps.
    if end_ts - start_ts < _MIN_WINDOW_NS:
        mid = (start_ts + end_ts) // 2
        start_ts = mid - _MIN_WINDOW_NS
        end_ts   = mid + _MIN_WINDOW_NS

    # Use an exact-match stream selector when instrument_id is known — avoids
    # a full cross-instrument scan.  Fall back to regex only when unknown.
    if instrument_id:
        stream_selector = f'{{helix_instrument_id="{instrument_id}"}}'
    else:
        stream_selector = '{helix_instrument_id=~".+"}'

    # helix_entity_id is structured metadata (set by both Alloy and OTLP paths),
    # not a JSON body field — use the structured metadata filter directly.
    logql = f'{stream_selector} | helix_entity_id = `{entity_id}`'
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
            entry: dict = {"ts": ts, "msg": msg}
            # Parse JSON log lines to surface structured fields directly.
            try:
                parsed = json.loads(msg)
                if parsed.get("level"):
                    entry["level"] = parsed["level"]
                if parsed.get("msg"):
                    entry["msg"] = parsed["msg"]
                # Extract src (GitHub permalink) — present on every log line
                # emitted by the mock-telescope and real pipeline code.
                if parsed.get("src"):
                    entry["src"] = parsed["src"]
            except (json.JSONDecodeError, AttributeError):
                pass
            lines.append(entry)

    lines.sort(key=lambda x: x["ts"])
    lines = lines[:_MAX_LINES]

    # Collect unique src links seen across all lines for easy reference.
    src_links = list(dict.fromkeys(
        ln["src"] for ln in lines if ln.get("src")
    ))

    return {
        "entity_id":   entity_id,
        "line_count":  len(lines),
        "src_links":   src_links,
        "grafana_url": loki_url(logql, start_ts, end_ts),
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
                "instrument_id": {
                    "type": "string",
                    "description": "Instrument ID (e.g. CHIMEFRB) — narrows the Loki stream selector to avoid a full cross-instrument scan. Always pass this when known from query_entity.",
                },
            },
            "required": ["entity_id", "start_ts", "end_ts"],
        },
    },
]

HANDLERS = {
    "query_loki": query_loki,
}
