"""explore_systems — looks up a node in the CHIME/FRB System Explorer.

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
]


async def explore_systems(node: str) -> dict:
    g = await explorer.graph()
    if g is None:
        return {"error": "The System Explorer graph could not be loaded. Use search_kb instead."}
    return explorer.describe(g, node)


HANDLERS = {"explore_systems": explore_systems}
