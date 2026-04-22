"""Prometheus metric query tool.

Queries Prometheus for a metric expression at a specific timestamp.
Used at Level 4 of the investigation ladder to correlate node resource
usage (memory, CPU, disk) with the exact time of an error.
"""

from __future__ import annotations

import os

import httpx

from .grafana import prometheus_url

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")

# ── Tool implementation ───────────────────────────────────────────────────────

async def query_prometheus(metric_expr: str, timestamp: int) -> dict:
    """Run a Prometheus instant query at a specific Unix timestamp (seconds).

    Returns all matching time series with their labels and values.
    timestamp should be in Unix seconds (divide nanosecond OTel timestamps by 1e9).
    """
    params = {
        "query": metric_expr,
        "time":  str(timestamp),
    }

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{PROMETHEUS_URL}/api/v1/query", params=params)
        if r.status_code != 200:
            return {"error": f"Prometheus returned HTTP {r.status_code}: {r.text[:200]}"}

    data   = r.json()
    status = data.get("status")
    if status != "success":
        return {"error": f"Prometheus query failed: {data.get('error', 'unknown')}"}

    result_type = data["data"]["resultType"]
    results     = data["data"]["result"]

    if not results:
        return {
            "metric_expr": metric_expr,
            "timestamp":   timestamp,
            "result_type": result_type,
            "series":      [],
            "grafana_url": prometheus_url(metric_expr, timestamp),
            "note":        "No data returned — metric may not exist or timestamp is out of retention.",
        }

    series = []
    for item in results:
        labels = item.get("metric", {})
        if result_type == "vector":
            ts, value = item["value"]
            series.append({"labels": labels, "value": float(value), "ts": ts})
        elif result_type == "matrix":
            values = [{"ts": ts, "value": float(v)} for ts, v in item.get("values", [])]
            series.append({"labels": labels, "values": values})

    return {
        "metric_expr": metric_expr,
        "timestamp":   timestamp,
        "result_type": result_type,
        "series":      series,
        "grafana_url": prometheus_url(metric_expr, timestamp),
    }


# ── Claude tool definition ────────────────────────────────────────────────────

DEFINITIONS = [
    {
        "name": "query_prometheus",
        "description": (
            "Run a Prometheus instant query at a specific Unix timestamp (seconds). "
            "Use this at Level 4 to check node metrics (memory, CPU, disk) at the exact "
            "moment of an error. Metric names come from the instrument's Tier 1 YAML config "
            "(prometheus.node_memory, prometheus.node_cpu, etc.). "
            "timestamp is Unix seconds — divide a nanosecond OTel timestamp by 1_000_000_000."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_expr": {
                    "type": "string",
                    "description": (
                        "PromQL expression, e.g. "
                        "'node_memory_MemAvailable_bytes{rack_id=\"gpu-rack-3\"}' "
                        "or 'rate(node_cpu_seconds_total{mode!=\"idle\"}[5m])'"
                    ),
                },
                "timestamp": {
                    "type": "integer",
                    "description": "Unix timestamp in seconds at which to evaluate the query",
                },
            },
            "required": ["metric_expr", "timestamp"],
        },
    },
]

HANDLERS = {
    "query_prometheus": query_prometheus,
}
