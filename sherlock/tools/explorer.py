"""explore_systems and find_shared_causes — the CHIME/FRB System Explorer graph.

Backed by sherlock.explorer, which fetches and caches the explorer's
published graph.json.
"""

from __future__ import annotations

from sherlock import explorer

DEFINITIONS = [
    {
        "name": "explore_systems",
        "description": (
            "Look up a CHIME/FRB system in the System Explorer: what it is, "
            "where it runs, what it depends on, what feeds it and what it "
            "feeds, plus its repo, config, logs, runbook and how it is "
            "deployed. Nodes are services, operations (units of work inside "
            "a service, named like the HelixObs operation), databases, "
            "queues, storage, hosts, clusters and sites. "
            "\n\n"
            "Use it to place an alert or error in the system: which upstream "
            "dependency to check before blaming the service itself, and "
            "which downstream systems a failure will affect. Use it for "
            "\"what is X\", \"where does X run\", \"what depends on X\" and "
            "\"how is X deployed\" questions, alongside search_kb for the "
            "wiki. "
            "\n\n"
            "Every answer has an explorer_url — link it so the operator can "
            "see the node in the explorer, and cite `sources` when they "
            "back a claim. A node with a `review` note has a placeholder "
            "description; say so rather than relying on it. If no node "
            "matches, the result lists every node id: pick the one that "
            "clearly fits and call again, or ask the operator."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": (
                        "A node id or name, e.g. `datatrail-clients`, "
                        "`client-side-registration`, `frb-archiver`. A partial "
                        "name returns the matching nodes."
                    ),
                },
            },
            "required": ["node"],
        },
    },
    {
        "name": "find_shared_causes",
        "description": (
            "Given what an operator reports as broken (and, optionally, what "
            "they report as working), walk the System Explorer graph "
            "upstream and rank the systems that could explain the symptoms: "
            "what the broken things depend on, run on or are fed by, "
            "directly or through other systems. A candidate that explains "
            "more symptoms over shorter paths ranks first. Each candidate "
            "comes with the path from every symptom it explains. "
            "\n\n"
            "Use it when the operator describes symptoms in words and there "
            "is no entity to investigate, especially when there is more "
            "than one symptom or you cannot see the logs or metrics. Map "
            "each symptom to an explorer node first (explore_systems helps; "
            "ask the operator if it is unclear). "
            "\n\n"
            "The ranking comes from how the systems are connected, not from "
            "live data. Present the top candidates as suspects, then ask "
            "the operator the one check that best tells the top two apart "
            "— for example whether another system that depends on only one "
            "of them is working. Pass what they confirm is working as "
            "`healthy` on the next call: those nodes drop out, and a "
            "candidate that a working system also relies on is flagged in "
            "also_relied_on_by_healthy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explorer node ids (or exact names) the operator reports as failing, slow or missing.",
                },
                "healthy": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explorer node ids the operator has confirmed are working.",
                },
            },
            "required": ["symptoms"],
        },
    },
]


async def explore_systems(node: str) -> dict:
    g = await explorer.graph()
    if g is None:
        return {"error": "The System Explorer graph could not be loaded. Use search_kb instead."}
    return explorer.describe(g, node)


async def find_shared_causes(symptoms: list[str], healthy: list[str] | None = None) -> dict:
    g = await explorer.graph()
    if g is None:
        return {"error": "The System Explorer graph could not be loaded. Ask the operator what else is affected instead."}
    return explorer.shared_causes(g, symptoms, healthy)


HANDLERS = {"explore_systems": explore_systems, "find_shared_causes": find_shared_causes}
