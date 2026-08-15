"""Sherlock Prometheus metrics and usage persistence.

Metrics are exposed on :9102/metrics via a background thread server.
Usage records are also written to the sherlock_usage TimescaleDB table
for durable cost tracking (Prometheus data is ephemeral).
"""

from __future__ import annotations

import logging
import os
import time as _time

from prometheus_client import (
    Counter,
    Histogram,
    start_http_server,
)

log = logging.getLogger(__name__)

DB_URL = os.environ.get("HERALD_DB_URL", "")
METRICS_PORT = int(os.environ.get("SHERLOCK_METRICS_PORT", "9102"))

# ── Prometheus metrics ────────────────────────────────────────────────────────

queries_total = Counter(
    "sherlock_queries_total",
    "Total Sherlock investigations by outcome.",
    ["query_type", "status"],  # status: success | failed
)

query_duration_seconds = Histogram(
    "sherlock_query_duration_seconds",
    "End-to-end investigation duration in seconds.",
    ["query_type"],
    buckets=[1, 2, 5, 10, 20, 30, 60, 120, 300],
)

tokens_input_total = Counter(
    "sherlock_tokens_input_total",
    "Input tokens consumed.",
    ["model"],
)

tokens_output_total = Counter(
    "sherlock_tokens_output_total",
    "Output tokens generated.",
    ["model"],
)

tokens_total = Counter(
    "sherlock_tokens_total",
    "Combined input + output tokens consumed.",
    ["model"],
)

cost_usd_total = Counter(
    "sherlock_cost_usd_total",
    "Cumulative investigation cost in USD.",
    ["model"],
)

cost_per_query_usd = Histogram(
    "sherlock_cost_per_query_usd",
    "Cost distribution per investigation in USD.",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

tool_calls_total = Counter(
    "sherlock_tool_calls_total",
    "Tool call outcomes.",
    ["tool_name", "status"],  # status: success | failed
)

tool_call_duration_seconds = Histogram(
    "sherlock_tool_call_duration_seconds",
    "Individual tool call latency.",
    ["tool_name"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
)


_QUERY_TYPES = ["diagnosis", "reply"]
_STATUSES = ["success", "failed"]
_MODELS = ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001", "memory"]


def _init_label_combinations() -> None:
    """Pre-initialize all label combinations so Prometheus emits samples on startup.

    prometheus_client only emits sample lines after the first inc()/observe().
    Without this, stat panels show no data until the first investigation runs.
    """
    for qt in _QUERY_TYPES:
        for st in _STATUSES:
            queries_total.labels(query_type=qt, status=st)
        query_duration_seconds.labels(query_type=qt)
    for model in _MODELS:
        tokens_input_total.labels(model=model)
        tokens_output_total.labels(model=model)
        tokens_total.labels(model=model)
        cost_usd_total.labels(model=model)
        cost_per_query_usd.labels(model=model)


def start_metrics_server() -> None:
    """Start the Prometheus metrics HTTP server in a background thread."""
    _init_label_combinations()
    try:
        start_http_server(METRICS_PORT)
        log.info("sherlock metrics listening on :%d", METRICS_PORT)
    except Exception:
        log.exception("failed to start metrics server")


# ── Usage persistence ─────────────────────────────────────────────────────────

async def record_usage(
    session_id: str,
    instrument_id: str,
    entity_id: str,
    model: str,
    tokens_input: int,
    tokens_output: int,
    cost_usd: float,
    duration_ms: int,
    tool_call_count: int,
    reached_hypothesis: bool,
    query_type: str = "diagnosis",
) -> None:
    """Write one row to sherlock_usage and update Prometheus counters."""
    status = "success" if reached_hypothesis else "failed"
    queries_total.labels(query_type=query_type, status=status).inc()
    query_duration_seconds.labels(query_type=query_type).observe(duration_ms / 1000)
    tokens_input_total.labels(model=model).inc(tokens_input)
    tokens_output_total.labels(model=model).inc(tokens_output)
    tokens_total.labels(model=model).inc(tokens_input + tokens_output)
    cost_usd_total.labels(model=model).inc(cost_usd)
    cost_per_query_usd.labels(model=model).observe(cost_usd)

    if not DB_URL:
        return
    try:
        import asyncpg
        conn = await asyncpg.connect(DB_URL)
        try:
            await conn.execute(
                """
                INSERT INTO sherlock_usage
                    (session_id, instrument_id, entity_id, model,
                     tokens_input, tokens_output, cost_usd,
                     duration_ms, tool_calls, reached_hypothesis)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                """,
                session_id, instrument_id, entity_id, model,
                tokens_input, tokens_output, cost_usd,
                duration_ms, tool_call_count, reached_hypothesis,
            )
        finally:
            await conn.close()
    except Exception:
        log.exception("sherlock_usage write failed")
