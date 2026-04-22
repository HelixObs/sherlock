"""Grafana deep-link builder for Explore pages.

Generates URLs using Grafana 11's panes format (?panes=...&schemaVersion=1).
"""

from __future__ import annotations

import json
from urllib.parse import quote

GRAFANA = "http://localhost:3001"

_SCHEMA = 1


def _panes_url(pane: dict) -> str:
    return f"{GRAFANA}/explore?schemaVersion={_SCHEMA}&panes={quote(json.dumps({'abc': pane}))}&orgId=1"


def loki_url(logql: str, start_ns: int, end_ns: int) -> str:
    pane = {
        "datasource": "loki",
        "queries": [{
            "refId": "A",
            "expr": logql,
            "queryType": "range",
            "datasource": {"type": "loki", "uid": "loki"},
            "editorMode": "code",
            "direction": "backward",
        }],
        "range": {"from": str(start_ns // 1_000_000), "to": str(end_ns // 1_000_000)},
    }
    return _panes_url(pane)


def prometheus_url(expr: str, timestamp_s: int) -> str:
    pane = {
        "datasource": "prometheus",
        "queries": [{
            "refId": "A",
            "expr": expr,
            "instant": True,
            "datasource": {"type": "prometheus", "uid": "prometheus"},
        }],
        "range": {
            "from": str((timestamp_s - 300) * 1000),
            "to":   str((timestamp_s + 300) * 1000),
        },
    }
    return _panes_url(pane)


def tempo_url(trace_id: str) -> str:
    pane = {
        "datasource": "tempo",
        "queries": [{
            "refId": "A",
            "query": trace_id,
            "queryType": "traceql",
            "datasource": {"type": "tempo", "uid": "tempo"},
        }],
        "range": {"from": "now-1h", "to": "now"},
    }
    return _panes_url(pane)
