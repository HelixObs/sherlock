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
            "knowledge base. Read through it yourself: if one CLEARLY and "
            "CONFIDENTLY relates to the question (the alias list can't "
            "anticipate every phrasing — e.g. a question about 'actions' "
            "should still recognize 'action_rules' in the list), call "
            "search_kb again with that exact name. If more than one "
            "plausibly fits and it actually changes the answer, use "
            "ask_operator to ask which one they mean rather than guessing. "
            "The same applies if only one entity comes to mind but the "
            "connection is a stretch, not an obvious match — a weak guess "
            "presented confidently is worse than no answer, especially "
            "once you start filling in specifics (numbers, dates, quotes) "
            "the retrieved content doesn't actually contain to make it "
            "sound complete. In either case, ask — e.g. 'I don't see this "
            "documented directly. Can you point me to a related wiki page "
            "or subsystem?' Clarification is a normal part of "
            "investigating, not a fallback of last resort."
            "\n\n"
            "Once a call returns a matched entity with docs (or page results "
            "with a reference URL), that IS the answer to 'where is this "
            "documented' — cite it and stop. The knowledge base indexes whole "
            "wiki pages, not individual procedures within them, so retrying "
            "search_kb with more specific or differently-phrased queries "
            "hoping to surface exact step-by-step commands from inside an "
            "already-found page won't turn up anything new — it'll just cost "
            "more tokens for the same answer. Point the operator at the page "
            "(and the relevant section heading, if the excerpt names one) "
            "rather than trying to fetch and reproduce its exact contents."
            "\n\n"
            "Word choice in `query` matters more than it looks. A query "
            "first tries an exact match (every word must appear together on "
            "one page); if that finds nothing, it automatically retries "
            "matching ANY of the words instead — so a generic word that "
            "could appear on almost any page ('pipeline', 'status', "
            "'service', 'system', 'data') adds little on the first try and "
            "actively dilutes the fallback, since pages will match on that "
            "one common word alone. Favor a few distinctive, specific terms "
            "(entity names, exact identifiers, uncommon nouns) over a "
            "descriptive natural-language phrase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A few distinctive keywords, not a descriptive sentence. "
                        "Prefer specific entity names/identifiers over generic "
                        "terms that could match almost any page."
                    ),
                },
                "top_k": {"type": "integer", "default": 5, "description": "Max results to return"},
            },
            "required": ["query"],
        },
    },
]


async def search_kb(query: str, top_k: int = 5) -> dict:
    return await kb.search(query, top_k)


HANDLERS = {"search_kb": search_kb}
