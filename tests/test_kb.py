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


def test_quote_fts_query_or_joins_terms_with_or():
    assert kb._quote_fts_query_or("baseband converter status") == "baseband OR converter OR status"


def test_quote_fts_query_or_quotes_hyphenated_terms():
    assert kb._quote_fts_query_or("frb-analysis backup") == '"frb-analysis" OR backup'


# ── _fts_search (AND-first, OR-fallback) ────────────────────────────────────────

def _fts_conn():
    """A minimal in-memory pages table -- _fts_search only needs the
    connection, not the full kb.sqlite3 schema/file path machinery."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE VIRTUAL TABLE pages USING fts5(title, content, source, reference UNINDEXED)")
    return conn


def test_fts_search_uses_precise_and_match_when_it_finds_something():
    conn = _fts_conn()
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?)",
        ("Datatrail", "datatrail postgres mongodb integration", "mediawiki", "https://x/Datatrail"),
    )
    conn.commit()
    results = kb._fts_search(conn, "datatrail postgres mongodb", top_k=5)
    assert [r["title"] for r in results] == ["Datatrail"]


def test_fts_search_falls_back_to_or_when_and_finds_nothing():
    """The exact scenario that motivated this: a verbose multi-word query
    where no single page contains every word, but a page containing most
    of them is clearly the right answer. AND alone returns nothing; OR
    should still surface it."""
    conn = _fts_conn()
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?)",
        ("Datatrail", "Datatrail's backing store is postgres; it also talks to MINOC and CADC.",
         "mediawiki", "https://x/Datatrail"),
    )
    conn.commit()
    # None of these six words all co-occur in the page -- "mongodb" and
    # "alpenhorn" aren't there at all -- so a plain AND match finds nothing.
    assert kb._fts_search(conn, "datatrail postgres mongodb MINOC CADC alpenhorn", top_k=5) == \
        kb._fts_search(conn, "datatrail postgres mongodb MINOC CADC alpenhorn", top_k=5)  # stable
    results = kb._fts_search(conn, "datatrail postgres mongodb MINOC CADC alpenhorn", top_k=5)
    assert [r["title"] for r in results] == ["Datatrail"]


def test_fts_search_single_word_miss_does_not_retry_with_or():
    """AND and OR are identical for a single term -- retrying would just
    re-run the same query for no benefit."""
    conn = _fts_conn()
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?)",
        ("Unrelated", "nothing relevant here", "mediawiki", "https://x/Unrelated"),
    )
    conn.commit()
    assert kb._fts_search(conn, "nonexistentterm", top_k=5) == []


def test_fts_search_returns_nothing_when_neither_and_nor_or_match():
    conn = _fts_conn()
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?)",
        ("Grafana", "dashboards and panels", "mediawiki", "https://x/Grafana"),
    )
    conn.commit()
    assert kb._fts_search(conn, "datatrail postgres mongodb", top_k=5) == []


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


async def test_search_content_without_matching_entity_skips_known_entities(kb_path):
    """known_entities is only worth its ~14KB cost when the query otherwise
    came up empty. cf1n2 already has a real page result -- the model has
    something to work with, so it shouldn't also pay for the full entity
    list on top of it."""
    _build_kb_file(kb_path)
    kb._state["checked_at"] = time.time()

    result = await kb.search("cf1n2")
    assert len(result["results"]) == 1
    assert "entity" not in result  # cf1n2 isn't its own entity row, just page content
    assert "known_entities" not in result


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


async def test_search_deprioritizes_run_notes_when_other_results_exist(kb_path):
    """Run Notes are per-day operator logs, not documentation -- there are
    hundreds of them, so by volume alone they'd otherwise crowd out the
    actual runbook. If something else also matches, Run Notes shouldn't
    show up at all."""
    _build_kb_file(kb_path)
    conn = sqlite3.connect(kb_path)
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?)",
        ("Run Notes - January 2023", "restarted baseband-converter at 03:40",
         "mediawiki", "https://bao.chimenet.ca/wiki/index.php/Run_Notes_-_January_2023"),
    )
    conn.commit()
    conn.close()
    kb._state["checked_at"] = time.time()

    result = await kb.search("baseband-converter")
    titles = [r["title"] for r in result["results"]]
    assert "Run Notes - January 2023" not in titles
    assert "baseband-converter" in titles


async def test_search_falls_back_to_run_notes_as_last_resort(kb_path):
    """If Run Notes are the *only* match, last resort still means a result,
    not an empty one."""
    _build_kb_file(kb_path)
    conn = sqlite3.connect(kb_path)
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?)",
        ("Run Notes - January 2023", "restarted the frobnicator service at 03:40",
         "mediawiki", "https://bao.chimenet.ca/wiki/index.php/Run_Notes_-_January_2023"),
    )
    conn.commit()
    conn.close()
    kb._state["checked_at"] = time.time()

    result = await kb.search("frobnicator")
    titles = [r["title"] for r in result["results"]]
    assert titles == ["Run Notes - January 2023"]


async def test_entity_docs_exclude_run_notes_when_other_docs_exist(kb_path):
    _build_kb_file(kb_path)
    conn = sqlite3.connect(kb_path)
    conn.execute(
        "INSERT INTO entity_docs VALUES (?, ?, ?)",
        ("baseband-converter", "Run Notes - January 2023", "secondary"),
    )
    conn.commit()
    conn.close()
    kb._state["checked_at"] = time.time()

    result = await kb.search("baseband-converter")
    doc_titles = [d["doc_title"] for d in result["entity"]["docs"]]
    assert "Run Notes - January 2023" not in doc_titles


async def test_entity_docs_keep_run_notes_if_thats_all_there_is(kb_path):
    _build_kb_file(kb_path)
    conn = sqlite3.connect(kb_path)
    conn.execute("DELETE FROM entity_docs WHERE entity_name = 'baseband-converter'")
    conn.execute(
        "INSERT INTO entity_docs VALUES (?, ?, ?)",
        ("baseband-converter", "Run Notes - January 2023", "secondary"),
    )
    conn.commit()
    conn.close()
    kb._state["checked_at"] = time.time()

    result = await kb.search("baseband-converter")
    doc_titles = [d["doc_title"] for d in result["entity"]["docs"]]
    assert doc_titles == ["Run Notes - January 2023"]


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
