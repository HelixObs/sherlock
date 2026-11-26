"""Sherlock's copy of the CHIME/FRB System Explorer graph.

CHIMEFRB/systems-overview publishes graph.json with every node (services,
operations, databases, storage, hosts, sites) and how they connect:
depends_on, flows_to and runs_on. This module fetches it on first use and
re-checks at most once per EXPLORER_REFRESH_INTERVAL_SECONDS, the same way
sherlock.kb handles kb.sqlite3. A failed refresh keeps serving the last copy.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

EXPLORER_GRAPH_URL = os.environ.get(
    "EXPLORER_GRAPH_URL", "https://chimefrb.github.io/systems-overview/graph.json"
)
EXPLORER_PAGE_URL = os.environ.get(
    "EXPLORER_PAGE_URL", "https://chimefrb.github.io/systems-overview/explorer"
)
EXPLORER_REFRESH_INTERVAL_SECONDS = int(os.environ.get("EXPLORER_REFRESH_INTERVAL_SECONDS", 3600))

_state: dict = {"graph": None, "checked_at": 0.0}


async def graph() -> dict | None:
    now = time.time()
    if _state["graph"] is not None and now - _state["checked_at"] < EXPLORER_REFRESH_INTERVAL_SECONDS:
        return _state["graph"]

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(EXPLORER_GRAPH_URL)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data.get("nodes"), list):
                _state["graph"] = data
                log.info("explorer: loaded %d nodes (generated %s)", len(data["nodes"]), data.get("generated"))
            else:
                log.warning("explorer: %s has no nodes list", EXPLORER_GRAPH_URL)
        else:
            log.warning("explorer: failed to fetch graph: HTTP %d", r.status_code)
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("explorer: failed to fetch graph: %s", exc)

    _state["checked_at"] = now
    return _state["graph"]


def _brief(node: dict) -> dict:
    return {"id": node["id"], "name": node["name"], "kind": node["kind"], "summary": node.get("summary", "")}


def _find(nodes: list[dict], query: str) -> list[dict]:
    q = query.strip().lower()
    exact = [n for n in nodes if q in (n["id"].lower(), n["name"].lower())]
    if exact:
        return exact
    named = [n for n in nodes if q in n["id"].lower() or q in n["name"].lower()]
    if named:
        return named
    return [n for n in nodes if q in n.get("summary", "").lower()]


def describe(g: dict, query: str) -> dict:
    """Look a node up by id or name and return it with its neighbours."""
    nodes = g["nodes"]
    by_id = {n["id"]: n for n in nodes}
    matches = _find(nodes, query) if query.strip() else []

    if len(matches) != 1:
        result = {
            "generated": g.get("generated"),
            "matches": [_brief(n) for n in matches],
        }
        if not matches:
            result["note"] = f"No node matches {query!r}. known_nodes lists every node in the explorer."
            result["known_nodes"] = [f"{n['id']} ({n['kind']})" for n in nodes]
        else:
            result["note"] = "Several nodes match. Call again with one exact id."
        return result

    node = matches[0]
    nid = node["id"]

    def label(other_id: str) -> str:
        other = by_id.get(other_id)
        return f"{other_id} ({other['kind']})" if other else other_id

    upstream = [
        {"node": label(e["target"]), "why": e.get("why", "")} for e in node.get("depends_on", [])
    ] + [
        {"node": label(n["id"]), "sends": e.get("what", "")}
        for n in nodes for e in n.get("flows_to", []) if e["target"] == nid
    ]
    downstream = [
        {"node": label(e["target"]), "receives": e.get("what", "")} for e in node.get("flows_to", [])
    ] + [
        {"node": label(n["id"]), "needs_it_for": e.get("why", "")}
        for n in nodes for e in n.get("depends_on", []) if e["target"] == nid
    ]

    placement = []
    parent = node.get("runs_on")
    while parent and parent not in placement:
        placement.append(parent)
        parent = by_id.get(parent, {}).get("runs_on")

    detail = {k: v for k, v in node.items() if k not in ("path", "depends_on", "flows_to")}
    detail.update({
        "explorer_url": f"{EXPLORER_PAGE_URL}?node={nid}",
        "runs_on_chain": [label(p) for p in placement],
        "runs_here": [label(n["id"]) for n in nodes if n.get("runs_on") == nid],
        "upstream": upstream,
        "downstream": downstream,
        "tour_notes": [
            {"tour": t.get("title", t["id"]), "text": s["text"]}
            for t in g.get("tours", []) for s in t.get("steps", []) if s.get("node") == nid
        ],
        "generated": g.get("generated"),
    })
    return {"node": detail}
