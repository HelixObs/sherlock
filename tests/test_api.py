"""Smoke tests for the Sherlock FastAPI endpoints."""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from sherlock import sessions
from sherlock.main import _patch_waiting_placeholder, app

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


# ── turn_count reset (a long chat must not inherit an earlier question's
#    spent turn budget — see MAX_TURNS comment in agent.py) ────────────────────

def test_patch_waiting_placeholder_reports_resume_vs_fresh():
    resumed = sessions.Session(id="x", entity_id="", instrument_id="")
    resumed.history = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "[waiting for operator reply]"},
    ]}]
    assert _patch_waiting_placeholder(resumed, "the answer") is True
    assert resumed.history[0]["content"][0]["content"] == "the answer"

    fresh = sessions.Session(id="y", entity_id="", instrument_id="")
    fresh.history = [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}]
    assert _patch_waiting_placeholder(fresh, "a new question") is False


async def test_chat_resets_turn_count_for_a_new_question_in_an_old_thread():
    """A long-lived web chat session must not silently die once the sum of
    every past question's turns crosses MAX_TURNS -- each new question (one
    with no pending ask_operator placeholder to resume) gets its own budget."""
    session_id = f"test-{uuid.uuid4()}"
    session = await sessions.get_or_create(session_id=session_id, interface="web")
    session.history = [{"role": "assistant", "content": [{"type": "text", "text": "earlier answer"}]}]
    session.turn_count = 25  # simulates a thread that already spent a full budget on a prior question
    await sessions.save(session)

    r = client.post("/chat", json={"session_id": session_id, "message": "a completely new question"})
    assert r.status_code == 200

    updated = await sessions.get(session_id)
    assert updated.turn_count == 0


async def test_chat_reply_does_not_reset_turn_count_mid_investigation():
    """Resuming a paused ask_operator question is a continuation of the same
    bounded task, not a fresh one -- its spent turns must carry over."""
    session_id = f"test-{uuid.uuid4()}"
    session = await sessions.get_or_create(session_id=session_id, interface="web")
    session.history = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "[waiting for operator reply]"},
    ]}]
    session.turn_count = 7
    await sessions.save(session)

    r = client.post(f"/chat/{session_id}/reply", json={"content": "here's the clarification"})
    assert r.status_code == 200

    updated = await sessions.get(session_id)
    assert updated.turn_count == 7


def test_memory_post_returns_entry():
    r = client.post("/memory/CHIME", json={
        "instrument_id": "CHIME",
        "content": "gpu-rack-3 OOMs during RFI storms",
        "confidence": "high",
        "error_type": "CLASSIFIER_TIMEOUT",
    })
    assert r.status_code == 201
    assert r.json()["instrument_id"] == "CHIME"
