"""Tests for Sherlock investigation tools (HTTP calls mocked with respx)."""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from sherlock.tools import dispatch
from sherlock.tools.github import _to_raw_url, fetch_github_file, search_github_callers
from sherlock.tools.loki import query_loki
from sherlock.tools.gateway import query_entity, query_entity_ancestors
from sherlock.tools.prometheus import query_prometheus


# ── github.py ─────────────────────────────────────────────────────────────────

def test_to_raw_url_blob():
    url = _to_raw_url("https://github.com/HelixObs/mock-telescope/blob/main/simulate.py")
    assert url == "https://raw.githubusercontent.com/HelixObs/mock-telescope/main/simulate.py"


def test_to_raw_url_already_raw():
    raw = "https://raw.githubusercontent.com/HelixObs/mock-telescope/main/simulate.py"
    assert _to_raw_url(raw) == raw


def test_to_raw_url_invalid():
    assert _to_raw_url("https://example.com/not-github") is None


@respx.mock
async def test_fetch_github_file_returns_excerpt():
    content = "\n".join(f"line {i}" for i in range(1, 101))
    respx.get("https://raw.githubusercontent.com/HelixObs/mock-telescope/main/simulate.py").mock(
        return_value=Response(200, text=content)
    )
    result = await fetch_github_file(
        "https://github.com/HelixObs/mock-telescope/blob/main/simulate.py",
        center_line=50,
        context_lines=5,
    )
    assert "error" not in result
    assert result["center_line"] == 50
    assert "→" in result["content"]   # center line marked


@respx.mock
async def test_fetch_github_file_http_error():
    respx.get("https://raw.githubusercontent.com/HelixObs/mock-telescope/main/simulate.py").mock(
        return_value=Response(404)
    )
    result = await fetch_github_file(
        "https://github.com/HelixObs/mock-telescope/blob/main/simulate.py",
        center_line=10,
    )
    assert "error" in result


@respx.mock
async def test_search_github_callers():
    respx.get("https://api.github.com/search/code").mock(
        return_value=Response(200, json={
            "items": [
                {"path": "pipeline/frb.py", "html_url": "https://github.com/HelixObs/mock-telescope/blob/main/pipeline/frb.py"},
            ]
        })
    )
    result = await search_github_callers("HelixObs/mock-telescope", "classify_candidate")
    assert result["function"] == "classify_candidate"
    assert len(result["matches"]) == 1


# ── loki.py ───────────────────────────────────────────────────────────────────

@respx.mock
async def test_query_loki_returns_lines():
    respx.get("http://loki:3100/loki/api/v1/query_range").mock(
        return_value=Response(200, json={
            "data": {
                "result": [
                    {
                        "stream": {"job": "myservice"},
                        "values": [
                            ["1000000000", '{"msg": "classified", "helix_entity_id": "cand-abc"}'],
                            ["2000000000", '{"msg": "error occurred", "helix_entity_id": "cand-abc"}'],
                        ],
                    }
                ]
            }
        })
    )
    result = await query_loki("cand-abc", 0, 5_000_000_000)
    assert result["line_count"] == 2
    assert result["lines"][0]["msg"] != ""


@respx.mock
async def test_query_loki_http_error():
    respx.get("http://loki:3100/loki/api/v1/query_range").mock(
        return_value=Response(500, text="internal error")
    )
    result = await query_loki("cand-abc", 0, 1)
    assert "error" in result


# ── gateway.py ────────────────────────────────────────────────────────────────

@respx.mock
async def test_query_entity_returns_node():
    graph = {
        "nodes": [{"id": "frb-123", "instrument_id": "CHIME", "trace_id": "aabb", "timestamp_ns": 1000, "parent_ids": ["block-1"], "metadata": {"snr": "22"}, "has_error": True}],
        "edges": [],
    }
    respx.get("http://gateway:8080/api/v1/entity/frb-123/graph").mock(
        return_value=Response(200, json=graph)
    )
    result = await query_entity("frb-123")
    assert result["id"] == "frb-123"
    assert result["has_error"] is True
    assert result["metadata"]["snr"] == "22"


@respx.mock
async def test_query_entity_not_found():
    respx.get("http://gateway:8080/api/v1/entity/missing/graph").mock(
        return_value=Response(404)
    )
    result = await query_entity("missing")
    assert "error" in result


@respx.mock
async def test_query_entity_ancestors_returns_graph():
    graph = {
        "nodes": [
            {"id": "frb-123", "has_error": True,  "metadata": {}, "parent_ids": ["cand-1"]},
            {"id": "cand-1",  "has_error": False, "metadata": {}, "parent_ids": []},
        ],
        "edges": [{"source": "cand-1", "target": "frb-123"}],
    }
    respx.get("http://gateway:8080/api/v1/entity/frb-123/graph").mock(
        return_value=Response(200, json=graph)
    )
    result = await query_entity_ancestors("frb-123")
    assert result["node_count"] == 2
    assert result["edge_count"] == 1


# ── prometheus.py ─────────────────────────────────────────────────────────────

@respx.mock
async def test_query_prometheus_returns_series():
    respx.get("http://prometheus:9090/api/v1/query").mock(
        return_value=Response(200, json={
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"rack_id": "gpu-rack-3"},
                        "value": [1714000000, "1234567890"],
                    }
                ],
            },
        })
    )
    result = await query_prometheus("node_memory_MemAvailable_bytes", 1714000000)
    assert result["result_type"] == "vector"
    assert result["series"][0]["labels"]["rack_id"] == "gpu-rack-3"
    assert result["series"][0]["value"] == pytest.approx(1234567890.0)


@respx.mock
async def test_query_prometheus_no_data():
    respx.get("http://prometheus:9090/api/v1/query").mock(
        return_value=Response(200, json={
            "status": "success",
            "data": {"resultType": "vector", "result": []},
        })
    )
    result = await query_prometheus("nonexistent_metric", 1714000000)
    assert result["series"] == []
    assert "note" in result


# ── dispatch ──────────────────────────────────────────────────────────────────

async def test_dispatch_unknown_tool():
    result = await dispatch("nonexistent_tool", {})
    assert "error" in result


@respx.mock
async def test_dispatch_routes_correctly():
    respx.get("http://gateway:8080/api/v1/entity/frb-abc/graph").mock(
        return_value=Response(200, json={
            "nodes": [{"id": "frb-abc", "instrument_id": "CHIME", "trace_id": "", "timestamp_ns": 1, "parent_ids": [], "metadata": {}, "has_error": False}],
            "edges": [],
        })
    )
    result = await dispatch("query_entity", {"entity_id": "frb-abc"})
    assert result["id"] == "frb-abc"
