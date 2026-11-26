import sqlite3

from sherlock import kb, prefetch

GRAPH = {"nodes": [
    {"id": "frb-archiver", "name": "frb-archiver", "kind": "host"},
    {"id": "l4-pipeline", "name": "L4 pipeline", "kind": "service"},
    {"id": "frb-master", "name": "frb-master", "kind": "service"},
]}


def test_latest_message_from_transcript():
    t = "Alice: hi\nSherlock: hello\nthere\nAlice: is the archiver\nfull?"
    assert prefetch.latest_message(t, "Alice") == "is the archiver\nfull?"
    assert prefetch.latest_message("Alice: just one", "Alice") == "just one"
    assert prefetch.latest_message("plain question", "") == "plain question"


def test_keywords_drop_filler():
    assert prefetch.keywords("How do I restart the L4 pipeline?") == "restart L4 pipeline"
    assert prefetch.keywords("is it ok?") == ""


def test_matching_nodes_by_id_name_and_short_name():
    assert prefetch.matching_nodes(GRAPH, "Is the archiver full?") == ["frb-archiver"]
    assert prefetch.matching_nodes(GRAPH, "restart the L4 pipeline?") == ["l4-pipeline"]
    assert prefetch.matching_nodes(GRAPH, "is frb-master down?") == ["frb-master"]
    assert prefetch.matching_nodes(GRAPH, "what is a tsar") == []


def _kb(tiered: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cols = "title, content, source, reference UNINDEXED" + (", tier UNINDEXED, edited UNINDEXED" if tiered else "")
    conn.execute(f"CREATE VIRTUAL TABLE pages USING fts5({cols})")
    rows = [
        ("Old log", "archiver disk filling up again", "mediawiki", "u1", "reference", "2016-01-01"),
        ("Ops page", "archiver disk filling up", "mediawiki", "u2", "ops", "2026-01-01"),
        ("Runbook", "archiver disk filling up", "git", "u3", "runbook", ""),
        ("Only old", "pathfinder correlator", "mediawiki", "u4", "reference", "2012-01-01"),
    ]
    for r in rows:
        conn.execute(f"INSERT INTO pages VALUES ({','.join('?' * (6 if tiered else 4))})", r if tiered else r[:4])
    return conn


def test_search_prefers_current_tiers():
    titles = [r["title"] for r in kb._fts_search(_kb(True), "archiver disk", 5)]
    assert titles == ["Runbook", "Ops page"]


def test_search_falls_back_to_reference():
    results = kb._fts_search(_kb(True), "correlator", 5)
    assert [(r["title"], r["tier"]) for r in results] == [("Only old", "reference")]


def test_search_works_on_untiered_kb():
    titles = {r["title"] for r in kb._fts_search(_kb(False), "archiver disk", 5)}
    assert titles == {"Old log", "Ops page", "Runbook"}
