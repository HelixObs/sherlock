"""search_kb — queries the operations knowledge base built by frb-ai.

Backed by sherlock.kb, which downloads and caches kb.sqlite3 from frb-ai's
latest GitHub release (see frb-ai/KB_REBUILD.md). Callers (agent.py,
prompt.py) don't need to change — the DEFINITIONS/HANDLERS shape is
unchanged from the earlier stub.
"""

from __future__ import annotations

from sherlock import kb

DEFINITIONS = [
    {
        "name": "search_kb",
        "description": (
            "Search the operations knowledge base for documented concepts, "
            "known failure modes, and past case studies. Always try this "
            "before answering a general question — do not answer from "
            "general knowledge if this returns nothing relevant. Results "
            "include a reference URL for every match — cite it, so an "
            "operator can verify or correct the source if it's wrong. "
            "\n\n"
            "If the response includes known_entities instead of a single "
            "entity, that means no exact name/alias match was found — "
            "known_entities is the complete list of entities in the "
            "knowledge base. Read through it yourself: if one clearly "
            "relates to the question (the alias list can't anticipate "
            "every phrasing — e.g. a question about 'actions' should "
            "still recognize 'action_rules' in the list), call search_kb "
            "again with that exact name. If more than one plausibly fits "
            "and it actually changes the answer, use ask_operator to ask "
            "which one they mean rather than guessing — clarification is "
            "a normal part of investigating, not a fallback of last resort."
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
    return await kb.search(query, top_k)


HANDLERS = {"search_kb": search_kb}
