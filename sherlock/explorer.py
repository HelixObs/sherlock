"""Sherlock's copy of the CHIME/FRB System Explorer graph.

CHIMEFRB/systems-overview publishes graph.json with every node (services,
operations, databases, storage, hosts, sites) and how they connect:
depends_on, flows_to and runs_on. This module fetches it on first use and
re-checks at most once per EXPLORER_REFRESH_INTERVAL_SECONDS, the same way
sherlock.kb handles kb.sqlite3. A failed refresh keeps serving the last copy.
"""

from __future__ import annotations

import heapq
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


def _upstream_edges(g: dict) -> dict[str, list[tuple[str, str, bool]]]:
    """node id → [(upstream id, how it is connected, is a data-flow edge)]."""
    edges: dict[str, list[tuple[str, str, bool]]] = {n["id"]: [] for n in g["nodes"]}
    for n in g["nodes"]:
        for e in n.get("depends_on", []):
            edges[n["id"]].append((e["target"], f"depends on ({e.get('why', '')})", False))
        if n.get("runs_on"):
            edges[n["id"]].append((n["runs_on"], "runs on", False))
        for e in n.get("flows_to", []):
            if not e.get("loop") and e["target"] in edges:
                edges[e["target"]].append((n["id"], f"is fed by ({e.get('what', '')})", True))
    return edges


_FLOW_COST = 2


def _walk_up(edges: dict, start: str, max_depth: int) -> dict[str, tuple[int, list[str]]]:
    """Cheapest walk towards causes: reached node → (cost, path from start as text steps).

    A dependency or host step costs 1. A data-flow step costs _FLOW_COST,
    because missing input is weaker evidence than a broken dependency, and
    flows are followed only along start's own data path: once the walk steps
    onto something start depends on or runs on, what feeds that dependency
    can no longer break start.
    """
    best: dict[str, tuple[int, list[str]]] = {}
    seen: set[tuple[str, bool]] = set()
    heap = [(0, 0, start, True, [start])]
    while heap:
        cost, hops, cur, on_data_path, path = heapq.heappop(heap)
        if (cur, on_data_path) in seen:
            continue
        seen.add((cur, on_data_path))
        best.setdefault(cur, (cost, path))
        if hops == max_depth:
            continue
        for up, how, is_flow in edges.get(cur, []):
            if is_flow and not on_data_path:
                continue
            step_cost = _FLOW_COST if is_flow else 1
            heapq.heappush(heap, (cost + step_cost, hops + 1, up, on_data_path and is_flow,
                                  path + [f"{how} → {up}"]))
    return best


def shared_causes(g: dict, symptoms: list[str | list[str]], healthy: list[str] | None = None,
                  max_depth: int = 6, limit: int = 8) -> dict:
    """Rank the upstream nodes that could explain the reported symptoms.

    Each entry in symptoms is one thing the operator reported: a node, or a
    list of nodes the report could mean (an operation and the service it
    runs in, say). A candidate explains a symptom when any of its nodes
    depends on it, runs on it or is fed by it, directly or through other
    nodes, so mapping one report to several nodes never counts it twice.
    Reports whose nodes run on one another (an operation and its service)
    are merged into one.
    Candidates that
    explain more symptoms, over shorter paths (a data-flow step counts
    double), and affect fewer other nodes rank first. Nodes reported healthy
    are dropped; a candidate that a healthy node also relies on is kept but
    flagged, and ranks lower among equally close candidates.
    """
    nodes = g["nodes"]
    by_id = {n["id"]: n for n in nodes}

    def resolve(names: list[str]) -> tuple[list[str], dict]:
        ids, problems = [], {}
        for name in names:
            found = _find(nodes, name)
            if len(found) == 1:
                ids.append(found[0]["id"])
            else:
                problems[name] = [_brief(n) for n in found] or "no node matches"
        return ids, problems

    reports = [[s] if isinstance(s, str) else list(s) for s in symptoms]
    report_ids, unresolved = [], {}
    for names in reports:
        ids, problems = resolve(names)
        report_ids.append(ids)
        unresolved.update(problems)
    healthy_ids, unresolved_healthy = resolve(healthy or [])
    unresolved.update(unresolved_healthy)
    if unresolved:
        return {
            "unresolved": unresolved,
            "note": "Map each of these to one exact node id (or ask the operator), then call again.",
            "known_nodes": [f"{n['id']} ({n['kind']})" for n in nodes],
        }

    def placement(nid: str) -> set[str]:
        chain, cur = set(), nid
        while cur and cur not in chain:
            chain.add(cur)
            cur = by_id.get(cur, {}).get("runs_on")
        return chain

    merged_reports: list[list[str]] = []
    for ids in report_ids:
        home = next((r for r in merged_reports
                     if any(a in placement(b) or b in placement(a) for a in ids for b in r)), None)
        if home is None:
            merged_reports.append(list(ids))
        else:
            home.extend(i for i in ids if i not in home)
    report_ids = merged_reports

    edges = _upstream_edges(g)
    reached: dict[str, dict[str, tuple[int, list[str]]]] = {}
    for ids in report_ids:
        merged: dict[str, tuple[int, list[str]]] = {}
        for sid in ids:
            for nid, (cost, path) in _walk_up(edges, sid, max_depth).items():
                if nid not in merged or cost < merged[nid][0]:
                    merged[nid] = (cost, path)
        reached[" / ".join(ids)] = merged
    healthy_reach = {h: _walk_up(edges, h, max_depth) for h in healthy_ids}
    downstream_size: dict[str, int] = {}
    for n in nodes:
        for up in _walk_up(edges, n["id"], max_depth):
            if up != n["id"]:
                downstream_size[up] = downstream_size.get(up, 0) + 1

    candidates = []
    for cid in set().union(*reached.values()) - set(healthy_ids):
        explains = {s: p[cid] for s, p in reached.items() if cid in p}
        cand = by_id[cid]
        entry = {
            **_brief(cand),
            "explains": len(explains),
            "paths": {s: " ".join([p[0]] + [f"—{step}" for step in p[1:]]) for s, (_, p) in explains.items()},
            "explorer_url": f"{EXPLORER_PAGE_URL}?node={cid}",
            "affects_other_nodes": downstream_size.get(cid, 0),
        }
        if cand.get("runbook"):
            entry["runbook"] = cand["runbook"]
        if cand.get("review"):
            entry["review"] = cand["review"]
        relied_on_by_healthy = [h for h, p in healthy_reach.items() if cid in p]
        if relied_on_by_healthy:
            entry["also_relied_on_by_healthy"] = relied_on_by_healthy
        entry["_rank"] = (
            -len(explains),
            max(c for c, _ in explains.values()),
            len(relied_on_by_healthy),
            sum(c for c, _ in explains.values()),
            downstream_size.get(cid, 0),
        )
        candidates.append(entry)

    candidates.sort(key=lambda c: (c["_rank"], c["id"]))
    for c in candidates:
        del c["_rank"]
    return {
        "symptoms": list(reached),
        "healthy": healthy_ids,
        "candidates": candidates[:limit],
        "candidates_total": len(candidates),
        "generated": g.get("generated"),
        "note": (
            "Candidates come from the explorer's edges, not from live data. "
            "Treat them as suspects to check, not a diagnosis."
        ),
    }
