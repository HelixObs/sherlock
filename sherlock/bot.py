"""Sherlock Slack bot — slack-bolt over Socket Mode.

v1 requires an explicit @sherlock mention on every message, including
follow-ups — no auto-pickup of unmentioned thread replies. A thread with
several operators routinely has human-to-human discussion in between
questions actually directed at Sherlock; any heuristic based on "did this
reply follow one of Sherlock's own messages" silently drops that
intervening context, since those messages were never sent to Sherlock at
all. Requiring an explicit mention every time avoids the ambiguity rather
than trying to resolve it. See helixobs/SHERLOCK_PLATFORM_DESIGN.md §6.

DMs don't require re-mentioning per message (a DM is inherently 1:1, there's
no "operators talking to each other" case to disambiguate) but still use
Slack's own reply-in-thread to continue a conversation, same as channels —
a flat, unthreaded second DM message starts a fresh session rather than
continuing the first, which is a real v1 limitation worth knowing about.

Every invocation re-fetches the complete thread from Slack (Slack is the
durable, authoritative record) and sends it to Sherlock as the full context
for this turn, *replacing* session.history rather than appending to it —
the fetched transcript already contains every prior turn as plain text.

Ack timing: Slack expects an ack within ~3 seconds of delivering an event.
Sherlock's investigation loop routinely runs well past that, so every
listener acks first and does the real work in a background task. Retry
de-duplication is handled via Slack's own event_id, not the ack alone —
a redelivered event resolves to the same session either way (thread-derived
session keys), but only event_id tracking stops a retry that arrives while
the original is still mid-investigation from running a second, racing call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

import httpx
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "")
SHERLOCK_API_URL = os.environ.get("SHERLOCK_API_URL", "http://localhost:8082")
DEFAULT_INSTRUMENT_ID = os.environ.get("SHERLOCK_DEFAULT_INSTRUMENT", "")

_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_MENTION_RE = re.compile(r"<@[A-Z0-9]+>\s*")
_ENTITY_ID_RE = re.compile(r"\b[a-z][a-z0-9]*-[0-9a-f]{6,}\b", re.IGNORECASE)

# Long enough to comfortably cover Slack's retry window (up to 3 retries,
# each backed off) without the cache growing unbounded.
_DEDUP_TTL = 600
_seen_events: dict[str, float] = {}
_name_cache: dict[str, str] = {}
_channel_name_cache: dict[str, str] = {}

app = AsyncApp(token=SLACK_BOT_TOKEN)


# ── Event listeners — ack immediately, do the real work in the background ──────

@app.event("app_mention")
async def on_mention(event: dict, ack, client, body: dict) -> None:
    await ack()
    if _already_seen(body.get("event_id")):
        return
    asyncio.create_task(_handle(event, client, is_dm=False))


@app.event("message")
async def on_message(event: dict, ack, client, body: dict) -> None:
    await ack()
    # app_mention already covers channel/group mentions — Slack delivers both
    # app_mention and message events for the same mention, so only handling
    # DMs here avoids double-processing the same message.
    if event.get("channel_type") != "im":
        return
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return  # ignore the bot's own messages
    if _already_seen(body.get("event_id")):
        return
    asyncio.create_task(_handle(event, client, is_dm=True))


def _already_seen(event_id: str | None) -> bool:
    """True if this Slack event_id has already been dispatched.

    Fails open (returns False) when event_id is missing rather than risk
    dropping a real message — better an occasional duplicate than a silent
    no-response.
    """
    if not event_id:
        return False
    now = time.time()
    for eid, seen_at in list(_seen_events.items()):
        if now - seen_at > _DEDUP_TTL:
            del _seen_events[eid]
    if event_id in _seen_events:
        return True
    _seen_events[event_id] = now
    return False


# ── Core handling ────────────────────────────────────────────────────────────

async def _handle(event: dict, client, is_dm: bool) -> None:
    channel = event["channel"]
    user = event.get("user", "")
    ts = event["ts"]
    thread_ts = event.get("thread_ts") or ts
    text = _MENTION_RE.sub("", event.get("text", "")).strip()

    session_id = f"slack:{channel}:{thread_ts}"
    entity_id = _detect_entity_id(text)

    try:
        posted = await client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text="Thinking…",
        )
    except SlackApiError:
        log.exception("failed to post placeholder message")
        return
    reply_ts = posted["ts"]

    # A reply within an existing thread has its own thread_ts distinct from
    # this event's ts — fetch the full thread so far. A brand-new mention or
    # DM has nothing to fetch yet; the transcript is just this one message.
    if event.get("thread_ts"):
        transcript = await _fetch_thread_transcript(client, channel, thread_ts)
    else:
        speaker = await _resolve_display_name(client, user)
        transcript = f"{speaker}: {text}"

    operator_name = await _resolve_display_name(client, user)
    channel_name = await _resolve_channel_name(client, channel)

    try:
        if entity_id:
            chunks = _call_diagnose(entity_id, session_id, user, operator_name, channel_name)
        else:
            chunks = _call_chat(transcript, session_id, user, operator_name, channel_name)
        await _stream_to_slack(client, channel, reply_ts, chunks)
    except Exception:
        log.exception("sherlock call failed")
        try:
            await client.chat_update(
                channel=channel, ts=reply_ts,
                text="Something went wrong talking to Sherlock — sorry about that.",
            )
        except SlackApiError:
            log.exception("failed to post error update")


async def _fetch_thread_transcript(client, channel: str, thread_ts: str) -> str:
    try:
        resp = await client.conversations_replies(channel=channel, ts=thread_ts, limit=200)
        messages = resp.get("messages", [])
    except SlackApiError:
        log.exception("failed to fetch thread history")
        return ""

    # v1 compression, matching Herodotus's original thread-length rule: full
    # verbatim under ~15 messages. Beyond that, root + last 14 — a real
    # summariser for the dropped middle is a follow-up, not built here.
    if len(messages) > 15:
        messages = [messages[0], *messages[-14:]]

    lines = []
    for m in messages:
        text = _MENTION_RE.sub("", m.get("text", "")).strip()
        if not text:
            continue
        if m.get("bot_id"):
            lines.append(f"Sherlock: {text}")
        else:
            speaker = await _resolve_display_name(client, m.get("user", ""))
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


async def _resolve_display_name(client, user_id: str) -> str:
    if not user_id:
        return "Unknown"
    if user_id in _name_cache:
        return _name_cache[user_id]
    try:
        resp = await client.users_info(user=user_id)
        name = (
            resp.get("user", {}).get("profile", {}).get("display_name")
            or resp.get("user", {}).get("real_name")
            or user_id
        )
    except SlackApiError:
        log.warning("users_info failed for %s — falling back to raw ID", user_id, exc_info=True)
        name = user_id  # best-effort — a raw ID is still usable context
    _name_cache[user_id] = name
    return name


async def _resolve_channel_name(client, channel_id: str) -> str:
    """Human-readable channel name (e.g. "#chime-ops"), not the raw Slack
    channel ID — cached the same way _resolve_display_name is."""
    if not channel_id:
        return ""
    if channel_id in _channel_name_cache:
        return _channel_name_cache[channel_id]
    try:
        resp = await client.conversations_info(channel=channel_id)
        ch = resp.get("channel", {})
        if ch.get("name"):
            name = f"#{ch['name']}"
        elif ch.get("is_im"):
            name = "DM"
        else:
            name = channel_id
    except SlackApiError:
        log.warning("conversations_info failed for %s — falling back to raw ID", channel_id, exc_info=True)
        name = channel_id  # best-effort — a raw ID is still usable context
    _channel_name_cache[channel_id] = name
    return name


def _detect_entity_id(text: str) -> str:
    m = _ENTITY_ID_RE.search(text)
    return m.group(0) if m else ""


# ── Sherlock HTTP calls ──────────────────────────────────────────────────────

async def _call_chat(transcript: str, session_id: str, operator_id: str, operator_name: str, channel: str):
    payload = {
        "message": transcript,
        "session_id": session_id,
        "interface": "slack",
        "replace_history": True,
        "instrument_id": DEFAULT_INSTRUMENT_ID,
        "operator_id": operator_id,
        "operator_name": operator_name,
        "channel": channel,
    }
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        async with client.stream("POST", f"{SHERLOCK_API_URL}/chat", json=payload) as r:
            async for line in r.aiter_lines():
                if line.strip():
                    yield json.loads(line)


async def _call_diagnose(entity_id: str, session_id: str, operator_id: str, operator_name: str, channel: str):
    payload = {
        "session_id": session_id,
        "interface": "slack",
        "instrument_id": DEFAULT_INSTRUMENT_ID,
        "operator_id": operator_id,
        "operator_name": operator_name,
        "channel": channel,
    }
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        async with client.stream(
            "POST", f"{SHERLOCK_API_URL}/diagnose/{entity_id}", json=payload,
        ) as r:
            async for line in r.aiter_lines():
                if line.strip():
                    yield json.loads(line)


# ── NDJSON → Slack translation ───────────────────────────────────────────────

# Sherlock's prompt (prompt.py) writes GFM — **bold**, [text](url) — since
# that's correct for the web UI. Slack's mrkdwn dialect uses different
# syntax for both, so it doesn't render GFM at all; it shows as literal
# asterisks and brackets. This is a Slack-side presentation concern, not a
# reason to make the model channel-aware — converted here, once, rather
# than by asking the model to write two dialects.
_GFM_LINK_RE = re.compile(r"\[([^\]]+)\]\((\S+?)\)")
_GFM_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

# Tool names → a short, human label for the "checked" summary line. Falls
# back to the raw name for anything not listed — not meant to be exhaustive.
_TOOL_LABELS = {
    "search_kb": "knowledge base",
    "query_entity_events": "entity events",
    "query_entity_operations": "entity operations",
    "query_entity_ancestors": "provenance",
    "query_similar_errors": "similar errors",
    "get_logs": "logs",
    "search_logs": "logs",
    "get_metrics": "metrics",
    "query_prometheus": "metrics",
    "fetch_github_file": "GitHub",
    "fetch_github_blame": "GitHub",
    "fetch_github_file_history": "GitHub",
    "search_github_callers": "GitHub",
}


def _to_slack_mrkdwn(text: str) -> str:
    text = _GFM_LINK_RE.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", text)
    text = _GFM_BOLD_RE.sub(lambda m: f"*{m.group(1)}*", text)
    return text


async def _stream_to_slack(client, channel: str, ts: str, chunks) -> None:
    """Consume the NDJSON stream and issue exactly one chat_update — start
    (the "Thinking…" placeholder already posted) and final, per the "short
    and sweet" cadence, not one edit per tool call.

    Renders as Block Kit, not a flat string: a muted context line naming
    what was checked (from evidence chunks' structured tool field — not by
    parsing the raw "→ **tool**(...)" announcement text, which is dropped
    here rather than shown verbatim) and a proper section block for the
    actual answer, so the two are visually distinct instead of one wall of
    text with plumbing, tool syntax, and prose all glued together.
    """
    narration_parts: list[str] = []
    tools_called: list[str] = []
    final_text = ""
    hypothesis_chunk: dict | None = None

    async for chunk in chunks:
        ctype = chunk.get("type")
        data = chunk.get("data") or {}
        if ctype == "step" and chunk.get("text"):
            text = chunk["text"]
            if "session_id" in data:
                continue  # the first chunk — plumbing, not real content
            if text.lstrip().startswith("→"):
                continue  # raw tool-announcement line — tools_called covers this
            narration_parts.append(text)
        elif ctype == "evidence":
            tool = data.get("tool")
            if tool:
                tools_called.append(tool)
        elif ctype == "question":
            final_text = chunk.get("text", "")
        elif ctype == "hypothesis":
            hypothesis_chunk = chunk
            final_text = _format_hypothesis(chunk)  # plain-text fallback only
        elif ctype == "error":
            final_text = f"Something went wrong: {chunk.get('text', 'unknown error')}"

    if not final_text:
        final_text = "".join(narration_parts).strip() or "(no response)"
    final_text = _to_slack_mrkdwn(final_text)

    blocks = []
    if tools_called:
        checked = ", ".join(dict.fromkeys(_TOOL_LABELS.get(t, t) for t in tools_called))
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"🔍 Checked: {checked}"}],
        })

    if hypothesis_chunk is not None:
        blocks.extend(_hypothesis_blocks(hypothesis_chunk))
    else:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": final_text}})

    # text is the required fallback (notifications, screen readers, clients
    # that don't render blocks) — blocks is what actually displays.
    await client.chat_update(channel=channel, ts=ts, text=final_text, blocks=blocks)


def _format_hypothesis(chunk: dict) -> str:
    """Plain-text fallback for the required text= parameter — not what
    actually displays when blocks render; see _hypothesis_blocks."""
    d = chunk.get("data", {})
    parts = [
        f"*{d.get('classification', 'unknown')}* ({d.get('confidence', 'low')} confidence)",
        d.get("summary", ""),
    ]
    if d.get("recommendation"):
        parts.append(f"→ {d['recommendation']}")
    return "\n".join(p for p in parts if p)


def _hypothesis_blocks(chunk: dict) -> list[dict]:
    """Classification/confidence as Block Kit fields (a real, underused
    Block Kit feature — short key/value pairs shown side by side) rather
    than folding everything into one plain-text paragraph. Summary and
    recommendation stay as prose, since they're not short key/value data."""
    d = chunk.get("data", {})
    blocks = [{
        "type": "section",
        "text": {"type": "mrkdwn", "text": _to_slack_mrkdwn(d.get("summary", ""))},
        "fields": [
            {"type": "mrkdwn", "text": f"*Classification*\n{d.get('classification', 'unknown')}"},
            {"type": "mrkdwn", "text": f"*Confidence*\n{d.get('confidence', 'low')}"},
        ],
    }]
    if d.get("recommendation"):
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Recommendation*\n{_to_slack_mrkdwn(d['recommendation'])}"},
        })
    return blocks


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        raise RuntimeError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set")
    asyncio.run(_run())


async def _run() -> None:
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    main()
