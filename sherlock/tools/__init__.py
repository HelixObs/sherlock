"""Sherlock investigation tools — Claude API tool definitions and dispatch.

Usage:
    from sherlock.tools import DEFINITIONS, dispatch

    # Pass DEFINITIONS to the Anthropic client:
    response = client.messages.create(..., tools=DEFINITIONS)

    # Dispatch a tool call from Claude's response:
    result = await dispatch(tool_use.name, tool_use.input)
"""

from __future__ import annotations

import os

from sherlock.tools import explorer, herald, github, herodotus, loki, output, prometheus

# Loki and Prometheus are off unless explicitly enabled: with them off,
# Sherlock has no path to logs or metrics at all, not just a prompt rule.
TELEMETRY_TOOLS = os.environ.get("SHERLOCK_TELEMETRY_TOOLS", "").lower() in ("1", "true", "yes")
_telemetry = [loki, prometheus] if TELEMETRY_TOOLS else []

# All Claude tool schema definitions — pass directly to anthropic.messages.create()
DEFINITIONS: list[dict] = [
    *github.DEFINITIONS,
    *herald.DEFINITIONS,
    *(d for m in _telemetry for d in m.DEFINITIONS),
    *herodotus.DEFINITIONS,
    *explorer.DEFINITIONS,
    *output.DEFINITIONS,
]

# name → async callable (output tools have no handlers — loop interprets them)
_HANDLERS: dict = {
    **github.HANDLERS,
    **herald.HANDLERS,
    **{k: v for m in _telemetry for k, v in m.HANDLERS.items()},
    **herodotus.HANDLERS,
    **explorer.HANDLERS,
}


async def dispatch(name: str, args: dict) -> dict:
    """Call the named tool with the given arguments.

    Returns the tool result as a dict. Unknown tool names return an error dict
    rather than raising, so the agent can handle gracefully.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name!r}"}
    return await handler(**args)
