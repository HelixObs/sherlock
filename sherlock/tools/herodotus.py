"""search_kb — queries Herodotus's operations knowledge base.

Stub implementation: Herodotus's git repo isn't populated yet (see
PIPELINE_STATE.md), so this always returns an empty result with a fixed
note. The real version replaces this with a local SQLite FTS5 query against
a periodically-synced snapshot of the Herodotus repo — see
helixobs/SHERLOCK_PLATFORM_DESIGN.md §5. Callers (agent.py, prompt.py)
don't need to change when that lands; the DEFINITIONS/HANDLERS shape stays
the same.
"""

from __future__ import annotations

DEFINITIONS = [
    {
        "name": "search_kb",
        "description": (
            "Search the operations knowledge base for documented concepts, "
            "known failure modes, and past case studies. Always try this "
            "before answering a general question — do not answer from "
            "general knowledge if this returns nothing relevant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 5, "description": "Max results to return"},
            },
            "required": ["query"],
        },
    },
]


async def search_kb(query: str, top_k: int = 5) -> dict:
    return {
        "results": [],
        "note": "I'm not yet knowledgeable about that — the knowledge base hasn't been populated yet.",
    }


HANDLERS = {"search_kb": search_kb}
