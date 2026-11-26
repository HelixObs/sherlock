"""search_kb — queries the operations knowledge base built by frb-ai.

Backed by sherlock.kb, which downloads and caches kb.sqlite3 from frb-ai's
latest GitHub release (see frb-ai/KB_REBUILD.md). Callers (agent.py,
prompt.py) don't need to change — the DEFINITIONS/HANDLERS shape is
unchanged from the earlier stub.
"""

from __future__ import annotations

import re

import httpx

from sherlock import kb
from sherlock.tools.github import GITHUB_TOKEN, _to_raw_url

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
            "Results are short excerpts. Once a call returns the page that "
            "answers the question, don't retry search_kb with other wording "
            "to get more of it: call read_doc with that result's reference "
            "to read the whole section when the answer needs what it says "
            "(the steps of a procedure, an order, a threshold). For 'where "
            "is this documented', the reference alone is the answer."
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


DEFINITIONS.append({
    "name": "read_doc",
    "description": (
        "Read the full text of a document Sherlock already has a link to: a "
        "knowledge-base entry (pass a reference exactly as search_kb returned "
        "it; with a #section you get that section, without one the whole "
        "page) or a runbook or doc on GitHub (a runbook link from "
        "explore_systems, or a github.com file URL). Use it when the answer "
        "needs what the document says, such as the steps of a procedure or "
        "the order to do them in, and quote or summarise it with the link. "
        "Don't use it to look for things: search_kb finds, read_doc reads."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reference": {
                "type": "string",
                "description": "A reference URL from search_kb, or a GitHub runbook/doc URL.",
            },
        },
        "required": ["reference"],
    },
})


async def search_kb(query: str, top_k: int = 5) -> dict:
    return await kb.search(query, top_k)


def _github_doc_url(url: str) -> str | None:
    """A GitHub folder link (a runbook) reads as its README.md."""
    m = re.match(r"https?://github\.com/([^/]+/[^/]+)/tree/([^/]+)/(.+?)/?$", url)
    if m:
        url = f"https://github.com/{m.group(1)}/blob/{m.group(2)}/{m.group(3)}/README.md"
    return _to_raw_url(url)


async def read_doc(reference: str) -> dict:
    result = await kb.read(reference)
    if "error" not in result:
        return result
    raw = _github_doc_url(reference)
    if raw is None:
        return result
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    async with httpx.AsyncClient(timeout=10, headers=headers, follow_redirects=True) as client:
        r = await client.get(raw)
    if r.status_code != 200:
        return {"error": f"Could not read {reference} (HTTP {r.status_code}). Give the operator the link instead."}
    text = kb.redact(r.text)
    out = {"reference": reference, "content": text[:kb.READ_MAX_CHARS]}
    if len(text) > kb.READ_MAX_CHARS:
        out["truncated"] = True
    return out


HANDLERS = {"search_kb": search_kb, "read_doc": read_doc}
