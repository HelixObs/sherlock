"""Smoke tests for the Sherlock FastAPI endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

from sherlock.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_diagnose_streams_ndjson():
    r = client.post("/diagnose/test-entity-123", json={"instrument_id": "CHIME"})
    assert r.status_code == 200
    assert "ndjson" in r.headers["content-type"]

    chunks = [json.loads(line) for line in r.text.strip().splitlines()]
    assert len(chunks) >= 2
    assert chunks[0]["type"] == "step"
    assert chunks[0]["data"]["session_id"]          # session_id in first chunk
    assert chunks[-1]["type"] == "done"


def test_diagnose_unknown_instrument_still_streams():
    # Unknown instrument → no Tier 1 context, agent still starts.
    # Without ANTHROPIC_API_KEY the agent yields an error chunk, not a 500.
    r = client.post("/diagnose/test-entity-456", json={"instrument_id": "UNKNOWN_XYZ"})
    assert r.status_code == 200
    chunks = [json.loads(line) for line in r.text.strip().splitlines()]
    assert chunks[-1]["type"] == "done"


def test_reply_unknown_session_returns_404():
    r = client.post("/diagnose/nonexistent-session-id/reply", json={"content": "hello"})
    assert r.status_code == 404


def test_memory_get_returns_list():
    r = client.get("/memory/CHIME")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_memory_post_returns_entry():
    r = client.post("/memory/CHIME", json={
        "instrument_id": "CHIME",
        "content": "gpu-rack-3 OOMs during RFI storms",
        "confidence": "high",
        "error_type": "CLASSIFIER_TIMEOUT",
    })
    assert r.status_code == 201
    assert r.json()["instrument_id"] == "CHIME"
