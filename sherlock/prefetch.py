"""Look a chat question up before the model's first turn.

Most tsar questions are lookups: the answer is on one knowledge-base page or
one System Explorer node. Running search_kb and explore_systems up front, in
parallel, and handing the results over with the question saves the tool
rounds the model would otherwise spend finding them. The tools stay
available for whatever this misses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from sherlock import explorer, kb

log = logging.getLogger(__name__)

MAX_NODES = 3
TOP_K = 5

_STOPWORDS = set("""
a about after all also am an and any are as at be been before being but by can
could did do does doing done for from get gets getting got had has have having
he her here hers him his how i if in into is it its just know let like look
looks me might more most my need needs no not now of off on once only or our
out over please should so some still such than that the their them then there
these they this those to too up us very was we were what when where which while
who whom why will with would you your yours anyone someone something anything
hey hi thanks thank ok okay sherlock happening going wrong right currently today
check checking tell show find seems seem see help working work broken issue
problem error errors fine running run
""".split())

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*[A-Za-z0-9]|[A-Za-z0-9]")


def latest_message(text: str, speaker: str) -> str:
    """The operator's newest message from a Slack transcript (`name: text`
    lines, see bot._fetch_thread_transcript), or the text itself."""
    if speaker:
        idx = max(text.rfind(f"\n{speaker}: "), -1)
        if idx >= 0:
            return text[idx + len(speaker) + 3:]
        if text.startswith(f"{speaker}: "):
            return text[len(speaker) + 2:]
    return text


def keywords(question: str, limit: int = 8) -> str:
    words, seen = [], set()
    for w in _WORD_RE.findall(question):
        lw = w.lower().strip(".")
        if len(lw) < 2 or lw in _STOPWORDS or lw in seen:
            continue
        seen.add(lw)
        words.append(w.strip("."))
    return " ".join(words[:limit])


def _norm(s: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


def matching_nodes(g: dict, question: str) -> list[str]:
    q = _norm(question)
    scored = []
    for node in g.get("nodes", []):
        names = {_norm(node["id"]), _norm(node["name"])}
        names |= {_norm(re.sub(r"^(frb|chime)[-_]", "", node["id"]))}
        best = max((len(n) for n in names if len(n.strip()) >= 3 and n in q), default=0)
        if best:
            scored.append((best, node["id"]))
    scored.sort(reverse=True)
    return [nid for _, nid in scored[:MAX_NODES]]


async def _explorer_entries(question: str) -> list[dict]:
    g = await explorer.graph()
    if g is None:
        return []
    return [explorer.describe(g, nid).get("node") for nid in matching_nodes(g, question)]


async def context_for(question: str) -> str:
    query = keywords(question)
    if not query:
        return ""
    search, nodes = await asyncio.gather(
        kb.search(query, top_k=TOP_K), _explorer_entries(question), return_exceptions=True,
    )
    parts = []
    if isinstance(nodes, list) and nodes:
        parts.append("System Explorer entries for systems the question names:\n"
                     + json.dumps([n for n in nodes if n], ensure_ascii=False))
    if isinstance(search, dict) and search.get("results"):
        parts.append(f"search_kb({query!r}):\n"
                     + json.dumps(search["results"], ensure_ascii=False))
    for name, result in (("search_kb", search), ("explorer", nodes)):
        if isinstance(result, Exception):
            log.warning("prefetch: %s failed: %s", name, result)
    if not parts:
        return ""
    log.info("prefetch(%r): %d explorer node(s), %d search result(s)",
             query, len(nodes) if isinstance(nodes, list) else 0,
             len(search.get("results", [])) if isinstance(search, dict) else 0)
    return ("Looked up before you started, for the operator's latest message. "
            "If this answers it, answer from it and cite it; call tools only "
            "for what is missing (read_doc for the full steps of a procedure).\n\n"
            + "\n\n".join(parts))
