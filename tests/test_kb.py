"""Tests for sherlock.kb (HTTP calls mocked with respx, sqlite built from
a small synthetic schema matching frb-ai's build_kb.py output)."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest
import respx
from httpx import Response

from sherlock import kb


def _build_kb_file(path: str) -> None:
    """Build a tiny kb.sqlite3 matching frb-ai/build_kb.py's real schema."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE VIRTUAL TABLE pages USING fts5(title, content, source, reference UNINDEXED);
        CREATE TABLE entities (name TEXT PRIMARY KEY, entity_type TEXT, aliases TEXT);
        CREATE TABLE entity_docs (entity_name TEXT, doc_title TEXT, relevance TEXT);
    """)
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?)",
        ("baseband-converter", "The baseband-converter service converts ring buffer dumps to HDF5.",
         "git", "https://github.com/CHIMEFRB/baseband-converter/blob/main/README.md"),
    )
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?)",
        ("FRB Operations Manual", "Node 10.7.201.12 aka cf1n2 handles beams 16-19.",
         "mediawiki", "https://bao.chimenet.ca/wiki/index.php/FRB_Operations_Manual"),
    )
    conn.execute(
        "INSERT INTO entities VALUES (?, ?, ?)",
        ("baseband-converter", "service", json.dumps(["baseband converter"])),
    )
    conn.execute(
        "INSERT INTO entity_docs VALUES (?, ?, ?)",
        ("baseband-converter", "baseband-converter", "primary"),
    )
    conn.execute(
        "INSERT INTO entity_docs VALUES (?, ?, ?)",
        ("baseband-converter", "FRB Operations Manual", "secondary"),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def kb_path(tmp_path, monkeypatch):
    p = str(tmp_path / "kb.sqlite3")
    monkeypatch.setattr(kb, "KB_CACHE_PATH", p)
    monkeypatch.setattr(kb, "_state", {"tag": None, "checked_at": 0.0})
    return p


# ── _quote_fts_query ────────────────────────────────────────────────────────

def test_quote_fts_query_hyphenated():
    assert kb._quote_fts_query("frb-analysis backup") == '"frb-analysis" backup'


def test_quote_fts_query_no_hyphens():
    assert kb._quote_fts_query("baseband converter") == "baseband converter"


# ── ensure_fresh ──────────────────────────────────────────────────────────────

@respx.mock
async def test_ensure_fresh_downloads_when_no_cache(kb_path):
    respx.get(f"https://api.github.com/repos/{kb.KB_REPO}/releases/latest").mock(
        return_value=Response(200, json={
            "tag_name": "kb-2026-09-01",
            "assets": [{"name": "kb.sqlite3", "url": "https://api.github.com/asset/1"}],
        })
    )
    respx.get("https://api.github.com/asset/1").mock(
        return_value=Response(200, content=b"fake sqlite bytes")
    )
    ok = await kb.ensure_fresh()
    assert ok is True
    with open(kb_path, "rb") as f:
        assert f.read() == b"fake sqlite bytes"


@respx.mock
async def test_ensure_fresh_skips_recheck_within_interval(kb_path):
    with open(kb_path, "w") as f:
        f.write("existing cache")
    kb._state["checked_at"] = time.time()  # just checked -- within the interval

    route = respx.get(f"https://api.github.com/repos/{kb.KB_REPO}/releases/latest")
    ok = await kb.ensure_fresh()
    assert ok is True
    assert route.call_count == 0  # no HTTP call at all -- cache is fresh enough


@respx.mock
async def test_ensure_fresh_same_tag_skips_redownload(kb_path):
    with open(kb_path, "w") as f:
        f.write("existing cache")
    kb._state["tag"] = "kb-2026-09-01"
    kb._state["checked_at"] = 0.0  # force a release check, but tag hasn't changed

    respx.get(f"https://api.github.com/repos/{kb.KB_REPO}/releases/latest").mock(
        return_value=Response(200, json={
            "tag_name": "kb-2026-09-01",
            "assets": [{"name": "kb.sqlite3", "url": "https://api.github.com/asset/1"}],
        })
    )
    asset_route = respx.get("https://api.github.com/asset/1")
    ok = await kb.ensure_fresh()
    assert ok is True
    assert asset_route.call_count == 0  # same release already cached -- no re-download


@respx.mock
async def test_ensure_fresh_falls_back_to_stale_cache_on_github_error(kb_path):
    with open(kb_path, "w") as f:
        f.write("stale but usable")
    respx.get(f"https://api.github.com/repos/{kb.KB_REPO}/releases/latest").mock(
        return_value=Response(500)
    )
    ok = await kb.ensure_fresh()
    assert ok is True  # stale cache still counts as usable
    with open(kb_path) as f:
        assert f.read() == "stale but usable"


@respx.mock
async def test_ensure_fresh_no_cache_and_github_unreachable(kb_path):
    respx.get(f"https://api.github.com/repos/{kb.KB_REPO}/releases/latest").mock(
        return_value=Response(500)
    )
    ok = await kb.ensure_fresh()
    assert ok is False


# ── search ────────────────────────────────────────────────────────────────────

async def test_search_returns_results_and_reference(kb_path):
    _build_kb_file(kb_path)
    kb._state["checked_at"] = time.time()  # skip the network check entirely

    result = await kb.search("baseband-converter")
    assert len(result["results"]) == 1
    assert result["results"][0]["reference"] == (
        "https://github.com/CHIMEFRB/baseband-converter/blob/main/README.md"
    )
    assert "**baseband-converter**" in result["results"][0]["excerpt"]


async def test_search_finds_entity_by_exact_name(kb_path):
    _build_kb_file(kb_path)
    kb._state["checked_at"] = time.time()

    result = await kb.search("baseband-converter")
    assert result["entity"]["name"] == "baseband-converter"
    assert result["entity"]["entity_type"] == "service"
    relevances = {d["doc_title"]: d["relevance"] for d in result["entity"]["docs"]}
    assert relevances["baseband-converter"] == "primary"
    assert relevances["FRB Operations Manual"] == "secondary"


async def test_search_finds_entity_by_alias_case_insensitive(kb_path):
    _build_kb_file(kb_path)
    kb._state["checked_at"] = time.time()

    result = await kb.search("Baseband Converter")  # matches the alias, different case
    assert result["entity"]["name"] == "baseband-converter"


async def test_search_content_without_matching_entity(kb_path):
    _build_kb_file(kb_path)
    kb._state["checked_at"] = time.time()

    result = await kb.search("cf1n2")
    assert len(result["results"]) == 1
    assert "entity" not in result  # cf1n2 isn't its own entity row, just page content
    # No exact match -> the model gets the full list to reason over itself,
    # rather than this function guessing via string heuristics.
    assert result["known_entities"] == [{"name": "baseband-converter", "entity_type": "service"}]
    assert "note" in result


async def test_search_no_exact_alias_but_related_entity_in_known_entities(kb_path):
    """The scenario that motivated known_entities over fuzzy string matching:
    "actions" shares no substring with "action_rules" (no hand-written
    heuristic would connect them), but a model reading the full entity list
    can trivially recognize the match itself."""
    _build_kb_file(kb_path)
    conn = sqlite3.connect(kb_path)
    conn.execute("INSERT INTO entities VALUES (?, ?, ?)", ("action_rules", "concept", "[]"))
    conn.commit()
    conn.close()
    kb._state["checked_at"] = time.time()

    result = await kb.search("actions")
    assert "entity" not in result
    names = [e["name"] for e in result["known_entities"]]
    assert "action_rules" in names


async def test_search_no_matches_returns_note(kb_path):
    _build_kb_file(kb_path)
    kb._state["checked_at"] = time.time()

    result = await kb.search("completely-unrelated-nonexistent-term")
    assert result["results"] == []
    assert "note" in result
    assert "known_entities" in result  # still offered even with zero text hits


@respx.mock
async def test_search_unavailable_when_no_cache_and_github_unreachable(kb_path):
    respx.get(f"https://api.github.com/repos/{kb.KB_REPO}/releases/latest").mock(
        return_value=Response(500)
    )
    result = await kb.search("anything")
    assert result["results"] == []
    assert "unavailable" in result["note"].lower()
