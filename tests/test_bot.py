"""Tests for the Slack bot's NDJSON → Slack translation (no Slack connection)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test")

from sherlock import bot  # noqa: E402


class FakeClient:
    def __init__(self):
        self.updates = []

    async def chat_update(self, **kwargs):
        self.updates.append(kwargs)


async def _chunks(items):
    for item in items:
        yield item


def test_headings_and_rules_become_slack_mrkdwn():
    text = "## Top suspects\n\n**#1 — intensity-nfs**\n\n---\n\nSee [explorer](https://x.test/e)"

    assert bot._to_slack_mrkdwn(text) == "*Top suspects*\n\n*#1 — intensity-nfs*\n\n\nSee <https://x.test/e|explorer>"


@pytest.mark.asyncio
async def test_only_text_after_the_last_tool_call_is_posted():
    client = FakeClient()
    chunks = [
        {"type": "step", "text": "Let me map both symptoms first."},
        {"type": "evidence", "data": {"tool": "explore_systems"}},
        {"type": "step", "text": "Good — now the shared causes."},
        {"type": "evidence", "data": {"tool": "find_shared_causes"}},
        {"type": "step", "text": "Top suspect: "},
        {"type": "step", "text": "intensity-nfs."},
    ]

    await bot._stream_to_slack(client, "C1", "1.0", _chunks(chunks))

    update = client.updates[0]
    assert update["text"] == "Top suspect: intensity-nfs."
    assert update["blocks"][0]["elements"][0]["text"] == "🔍 Checked: System Explorer"
