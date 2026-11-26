"""Tests for sherlock.explorer and the explore_systems tool (HTTP mocked with respx)."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from sherlock import explorer
from sherlock.tools import dispatch

GRAPH = {
    "generated": "2026-10-30T13:08:33.424Z",
    "nodes": [
        {"id": "chime", "name": "CHIME site", "kind": "site", "summary": "The telescope site."},
        {"id": "frb-archiver", "name": "frb-archiver", "kind": "host", "runs_on": "chime",
         "summary": "Storage host."},
        {"id": "intensity-nfs", "name": "Intensity NFS", "kind": "storage", "runs_on": "frb-archiver",
         "summary": "Intensity event directories."},
        {"id": "datatrail-clients", "name": "Datatrail Clients", "kind": "service", "runs_on": "chime",
         "summary": "Registration, replication and deletion."},
        {"id": "datatrail-client-side-registration", "name": "client-side-registration",
         "kind": "operation", "runs_on": "datatrail-clients",
         "summary": "Submits new event data for registration.",
         "runbook": "https://github.com/CHIMEFRB/troubleshooting/tree/main/runbooks/client-side-registration",
         "depends_on": [{"target": "intensity-nfs", "why": "intensity event directories"}],
         "flows_to": [{"target": "datatrail-server-side-registration", "what": "registration payload"}],
         "path": "systems/nodes/operation/datatrail-client-side-registration.yaml"},
        {"id": "datatrail-server-side-registration", "name": "server-side-registration",
         "kind": "operation", "runs_on": "datatrail-clients", "summary": "Registers datasets."},
    ],
    "tours": [
        {"id": "datatrail-dataset-lifecycle", "title": "Datatrail dataset lifecycle",
         "steps": [{"node": "datatrail-client-side-registration", "text": "New data is found on disk."}]},
    ],
}


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(explorer, "_state", {"graph": None, "checked_at": 0.0})


@pytest.mark.asyncio
@respx.mock
async def test_operation_lookup_by_name_includes_neighbours():
    respx.get(explorer.EXPLORER_GRAPH_URL).mock(return_value=Response(200, json=GRAPH))

    result = await dispatch("explore_systems", {"node": "client-side-registration"})

    node = result["node"]
    assert node["id"] == "datatrail-client-side-registration"
    assert node["upstream"] == [{"node": "intensity-nfs (storage)", "why": "intensity event directories"}]
    assert node["downstream"] == [
        {"node": "datatrail-server-side-registration (operation)", "receives": "registration payload"}
    ]
    assert node["runs_on_chain"] == ["datatrail-clients (service)", "chime (site)"]
    assert node["tour_notes"] == [{"tour": "Datatrail dataset lifecycle", "text": "New data is found on disk."}]
    assert node["explorer_url"].endswith("?node=datatrail-client-side-registration")
    assert "path" not in node


@pytest.mark.asyncio
@respx.mock
async def test_storage_lookup_shows_who_needs_it():
    respx.get(explorer.EXPLORER_GRAPH_URL).mock(return_value=Response(200, json=GRAPH))

    node = (await dispatch("explore_systems", {"node": "intensity-nfs"}))["node"]

    assert node["downstream"] == [
        {"node": "datatrail-client-side-registration (operation)", "needs_it_for": "intensity event directories"}
    ]
    assert node["runs_on_chain"] == ["frb-archiver (host)", "chime (site)"]


@pytest.mark.asyncio
@respx.mock
async def test_partial_name_lists_matches():
    respx.get(explorer.EXPLORER_GRAPH_URL).mock(return_value=Response(200, json=GRAPH))

    result = await dispatch("explore_systems", {"node": "registration"})

    assert {m["id"] for m in result["matches"]} == {
        "datatrail-client-side-registration", "datatrail-server-side-registration",
    }
    assert "known_nodes" not in result


@pytest.mark.asyncio
@respx.mock
async def test_unknown_node_lists_every_node():
    respx.get(explorer.EXPLORER_GRAPH_URL).mock(return_value=Response(200, json=GRAPH))

    result = await dispatch("explore_systems", {"node": "kubernetes"})

    assert result["matches"] == []
    assert len(result["known_nodes"]) == len(GRAPH["nodes"])


@pytest.mark.asyncio
@respx.mock
async def test_failed_refresh_keeps_last_graph(monkeypatch):
    route = respx.get(explorer.EXPLORER_GRAPH_URL)
    route.mock(return_value=Response(200, json=GRAPH))
    assert (await explorer.graph())["generated"] == GRAPH["generated"]

    monkeypatch.setitem(explorer._state, "checked_at", 0.0)
    route.mock(return_value=Response(503))
    assert (await explorer.graph())["generated"] == GRAPH["generated"]


@pytest.mark.asyncio
@respx.mock
async def test_no_graph_returns_error():
    respx.get(explorer.EXPLORER_GRAPH_URL).mock(return_value=Response(503))

    result = await dispatch("explore_systems", {"node": "frb-archiver"})

    assert "error" in result


CAUSES_GRAPH = {
    "generated": "2026-11-25T00:00:00Z",
    "nodes": [
        {"id": "site", "name": "site", "kind": "site", "summary": ""},
        {"id": "host", "name": "host", "kind": "host", "runs_on": "site", "summary": ""},
        {"id": "db", "name": "db", "kind": "database", "runs_on": "host", "summary": ""},
        {"id": "queue", "name": "queue", "kind": "queue", "runs_on": "host", "summary": ""},
        {"id": "writer", "name": "writer", "kind": "operation", "summary": "",
         "flows_to": [{"target": "db", "what": "rows"}]},
        {"id": "feeder", "name": "feeder", "kind": "operation", "summary": "",
         "flows_to": [{"target": "svc-a", "what": "work"}]},
        {"id": "svc-a", "name": "svc-a", "kind": "service", "summary": "",
         "depends_on": [{"target": "db", "why": "state"}, {"target": "queue", "why": "jobs"}]},
        {"id": "svc-b", "name": "svc-b", "kind": "service", "summary": "",
         "depends_on": [{"target": "db", "why": "state"}]},
        {"id": "svc-c", "name": "svc-c", "kind": "service", "summary": "",
         "depends_on": [{"target": "queue", "why": "jobs"}]},
    ],
    "tours": [],
}


def _ids(result: dict) -> list[str]:
    return [c["id"] for c in result["candidates"]]


def test_shared_dependency_ranks_first():
    result = explorer.shared_causes(CAUSES_GRAPH, ["svc-a", "svc-b"])

    top = result["candidates"][0]
    assert top["id"] == "db"
    assert top["explains"] == 2
    assert top["paths"] == {
        "svc-a": "svc-a —depends on (state) → db",
        "svc-b": "svc-b —depends on (state) → db",
    }
    assert "queue" not in _ids(result)[:2]


def test_what_feeds_a_dependency_is_not_a_cause():
    result = explorer.shared_causes(CAUSES_GRAPH, ["svc-b"])

    assert "writer" not in _ids(result)


def test_own_inputs_rank_after_dependencies():
    ids = _ids(explorer.shared_causes(CAUSES_GRAPH, ["svc-a"]))

    assert ids.index("feeder") > ids.index("db")
    assert ids.index("feeder") > ids.index("queue")


def test_healthy_nodes_drop_out_and_flag_what_they_rely_on():
    result = explorer.shared_causes(CAUSES_GRAPH, ["svc-a"], healthy=["db", "svc-c"])

    assert "db" not in _ids(result)
    queue = next(c for c in result["candidates"] if c["id"] == "queue")
    assert queue["also_relied_on_by_healthy"] == ["svc-c"]


def test_unresolved_symptom_asks_for_exact_ids():
    result = explorer.shared_causes(CAUSES_GRAPH, ["svc", "kubernetes"])

    assert set(result["unresolved"]) == {"svc", "kubernetes"}
    assert "candidates" not in result
