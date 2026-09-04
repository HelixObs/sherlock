"""Sherlock's local knowledge-base cache, synced from frb-ai's GitHub releases.

frb-ai (see KB_REBUILD.md there) publishes kb.sqlite3 — an FTS5 `pages` table
plus `entities`/`entity_docs` lookup tables — as a GitHub Release asset on
each rebuild. This module downloads the latest release on first use and
re-checks for a newer one at most once per KB_REFRESH_INTERVAL_SECONDS,
which is cheap enough to check on every search_kb call rather than running a
background refresh task.

frb-ai is private, so downloads go through the GitHub API (not the public
browser_download_url) using the same GITHUB_TOKEN already used by
sherlock.tools.github.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time

import httpx

log = logging.getLogger(__name__)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
KB_REPO = os.environ.get("KB_REPO", "CHIMEFRB/frb-ai")
KB_CACHE_PATH = os.environ.get("KB_CACHE_PATH", "/tmp/kb.sqlite3")
KB_REFRESH_INTERVAL_SECONDS = int(os.environ.get("KB_REFRESH_INTERVAL_SECONDS", 3600))

_JSON_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Process-global — deliberately module state, not a class, matching
# GITHUB_TOKEN's own module-level pattern in tools/github.py. Sherlock runs
# as a single-process FastAPI app; there's no multi-instance cache to keep
# in sync.
_state = {"tag": None, "checked_at": 0.0}


async def _latest_release() -> dict | None:
    headers = {**_JSON_HEADERS, **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {})}
    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        r = await client.get(f"https://api.github.com/repos/{KB_REPO}/releases/latest")
        if r.status_code != 200:
            log.warning("kb: failed to check latest release: HTTP %d", r.status_code)
            return None
        return r.json()


async def ensure_fresh() -> bool:
    """Download kb.sqlite3 if missing, or re-check for a newer release if
    the last check was more than KB_REFRESH_INTERVAL_SECONDS ago. No-op
    otherwise. Returns True if a usable cached file exists afterward.

    A GitHub hiccup never blocks queries against an already-cached copy —
    only a cold start with no cache at all and no reachable GitHub fails.
    """
    now = time.time()
    have_cache = os.path.exists(KB_CACHE_PATH)
    if have_cache and now - _state["checked_at"] < KB_REFRESH_INTERVAL_SECONDS:
        return True

    release = await _latest_release()
    if release is None:
        return have_cache  # keep serving the stale copy rather than failing outright

    _state["checked_at"] = now
    tag = release.get("tag_name", "")
    if tag == _state["tag"] and have_cache:
        return True  # already have this exact release cached

    asset = next((a for a in release.get("assets", []) if a["name"] == "kb.sqlite3"), None)
    if asset is None:
        log.warning("kb: release %s has no kb.sqlite3 asset", tag)
        return have_cache

    headers = {
        "Accept": "application/octet-stream",
        **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
    }
    # GitHub's asset download redirects to a signed, time-limited blob URL —
    # follow_redirects is required or this returns the 302 itself, not data.
    async with httpx.AsyncClient(timeout=60, headers=headers, follow_redirects=True) as client:
        r = await client.get(asset["url"])
        if r.status_code != 200:
            log.warning("kb: failed to download asset for %s: HTTP %d", tag, r.status_code)
            return have_cache
        # Write to a temp path and atomically swap in — a reader connecting
        # mid-download must never see a truncated/corrupt sqlite file.
        tmp_path = KB_CACHE_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(r.content)
        os.replace(tmp_path, KB_CACHE_PATH)

    _state["tag"] = tag
    log.info("kb: refreshed to release %s (%d bytes)", tag, len(r.content))
    return True


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{KB_CACHE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _quote_fts_query(query: str) -> str:
    """Quote hyphenated terms for FTS5, which otherwise mis-parses them —
    e.g. "frb-analysis" must become '"frb-analysis"'. See KB_REBUILD.md."""
    return " ".join(f'"{w}"' if "-" in w else w for w in query.split())


def _quote_fts_query_or(query: str) -> str:
    """Same quoting as _quote_fts_query, but OR-joined instead of the bare
    space FTS5 reads as an implicit AND. A verbose, natural-language query
    like "datatrail postgres mongodb MINOC CADC alpenhorn" requires all six
    words to co-occur in one page under AND -- almost nothing will, even
    when a page is clearly relevant and matches four or five of them. See
    search()'s AND-first-then-OR-fallback for why this is only a fallback,
    not the default: AND's precision is worth keeping whenever it actually
    finds something."""
    return " OR ".join(f'"{w}"' if "-" in w else w for w in query.split())


# Run Notes pages are per-day/month operator logs ("restarted X at HH:MM") --
# record-keeping, not documentation. There are hundreds of them (one per
# station per month), so by sheer volume they dominate both FTS ranking and
# entity_docs lists, crowding out the actual runbooks/READMEs a "how do I"
# question needs. Not excluded entirely -- still useful as a last resort,
# e.g. finding real precedent for an unusual failure -- just sorted after
# everything else rather than competing with it on equal footing.
_RUN_NOTES_ORDER_SQL = "(title LIKE '%Run Notes%')"


def _fts_search(conn: sqlite3.Connection, query: str, top_k: int) -> list[dict]:
    """Full-text search with an AND-first, OR-fallback strategy.

    An exact AND match (every word must co-occur in the same page) is tried
    first -- it's the most precise result when it finds anything, since
    FTS5's rank naturally favors pages matching more/rarer terms. But a
    verbose multi-word query (the natural style a model tends to write --
    e.g. "datatrail postgres mongodb MINOC CADC alpenhorn") requires ALL of
    those words to appear in one page under AND, which almost nothing will
    satisfy even when a page is clearly relevant and matches most of them.
    Falling back to OR only when AND finds literally nothing preserves
    AND's precision whenever it works, while still surfacing something for
    exactly the queries that were previously returning zero results and
    forcing the model into repeated, costly retries.
    """
    def _run(fts_query: str) -> list[dict]:
        # Fetch a superset ordered by rank (non-Run-Notes first), then only
        # fall back to Run Notes rows if nothing else matched at all -- a
        # last resort, not just a lower-ranked competitor. top_k*5 is a
        # generous enough superset to almost never actually truncate the
        # non-Run-Notes group before it gets partitioned below.
        cur = conn.execute(
            "SELECT title, source, reference, "
            "snippet(pages, 1, '**', '**', '...', 24) AS excerpt "
            "FROM pages WHERE pages MATCH ? "
            f"ORDER BY {_RUN_NOTES_ORDER_SQL}, rank LIMIT ?",
            (fts_query, top_k * 5),
        )
        return [dict(row) for row in cur.fetchall()]

    all_matches = _run(_quote_fts_query(query))
    if not all_matches and len(query.split()) > 1:
        all_matches = _run(_quote_fts_query_or(query))

    non_run_notes = [r for r in all_matches if "Run Notes" not in r["title"]]
    return (non_run_notes or all_matches)[:top_k]


def _entity_with_docs(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    docs = conn.execute(
        "SELECT doc_title, relevance FROM entity_docs WHERE entity_name = ? "
        "ORDER BY relevance",  # "primary" sorts before "secondary"
        (row["name"],),
    ).fetchall()
    # Same last-resort treatment as search results (see _RUN_NOTES_ORDER_SQL):
    # an entity like L4_pipeline can have 19 Run Notes entries out of 29 total
    # docs, drowning out the handful of real runbooks/READMEs. Drop them
    # entirely unless they're literally the only thing documenting this
    # entity at all.
    non_run_notes = [d for d in docs if "Run Notes" not in d["doc_title"]]
    kept = non_run_notes or docs
    return {**dict(row), "docs": [dict(d) for d in kept]}


async def search(query: str, top_k: int = 5) -> dict:
    """Full-text search over pages, plus an entity lookup for the raw query
    string — a query like "cf1n2" or "baseband-converter" is often itself
    an entity name, not just search terms.

    Entity matching is deliberately NOT fuzzy-matched here. An earlier
    version of this function tried normalizing separators and substring
    matching against every alias -- but a query like "actions" (asked
    about the `action_rules` entity, no "rules" at all) shares no
    substring with "action_rules" and no hand-written string heuristic
    generalizes to every real phrasing a person will actually use. That's
    a losing, endless game to keep patching.

    Instead: on an exact name/alias miss, this returns the full list of
    every entity's name and type as `known_entities`. Aliases can never be
    exhaustive by hand, but the model reading the question already knows
    what it means well enough to recognize the right entity in a list --
    that's a judgment call, which is what the model is for, not something
    to keep re-deriving in SQL. See search_kb's tool description for what
    it's told to do with this list (re-query with the right name, or ask
    the operator if it's genuinely ambiguous).
    """
    if not await ensure_fresh():
        return {
            "results": [],
            "note": "Knowledge base unavailable — no cached copy and could not reach GitHub to fetch one.",
        }

    conn = _connect()
    try:
        results = _fts_search(conn, query, top_k)

        entity_row = conn.execute(
            "SELECT name, entity_type, aliases FROM entities "
            "WHERE name = ? COLLATE NOCASE "
            "OR EXISTS (SELECT 1 FROM json_each(aliases) WHERE value = ? COLLATE NOCASE)",
            (query, query),
        ).fetchone()

        entity_info = None
        known_entities = None
        if entity_row is not None:
            entity_info = _entity_with_docs(conn, entity_row)
        elif not results:
            # Only worth the full ~285-entity, ~14KB dump when the query
            # otherwise came up empty -- if page search already found real
            # content, the model has something to work with and doesn't
            # need to go entity-hunting on top of it. This was previously
            # unconditional on any entity-name miss, so even a query that
            # already had 5 good results (e.g. "L4 pipeline restart") paid
            # for the full list anyway -- baked into history and re-billed
            # on every turn after.
            known_entities = [
                {"name": row["name"], "entity_type": row["entity_type"]}
                for row in conn.execute("SELECT name, entity_type FROM entities ORDER BY name")
            ]
    finally:
        conn.close()

    out: dict = {"results": results}
    if entity_info is not None:
        out["entity"] = entity_info
    elif known_entities:
        out["known_entities"] = known_entities
        out["note"] = (
            "No exact entity name/alias match for this query. known_entities above is the "
            "full list known to the knowledge base — if one clearly relates to the question "
            "(the alias list can't anticipate every phrasing), call search_kb again with that "
            "exact name. If more than one plausibly fits and it actually changes the answer, "
            "ask the operator which one they mean rather than guessing."
        )
    if not results and entity_info is None:
        out["note"] = "No matches — do not answer from general knowledge, say this isn't documented."

    # Titles/entity name, not full page content -- enough to see what a call
    # actually matched (and to spot a repeated/wasted query) from `kubectl
    # logs` alone, without the log line itself becoming as big as the
    # content it's describing.
    if entity_info is not None:
        matched = f"entity={entity_info['name']!r}"
    elif known_entities:
        matched = f"no entity match, {len(known_entities)} known_entities offered"
    else:
        matched = "no entity match"
    titles = [r["title"] for r in results]
    log.info("search(%r) -> %s; %d page result(s): %s", query, matched, len(results), titles)
    return out
