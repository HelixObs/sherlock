"""Sherlock investigation tools — Claude API tool definitions and dispatch.

Usage:
    from sherlock.tools import DEFINITIONS, dispatch

    # Pass DEFINITIONS to the Anthropic client:
    response = client.messages.create(..., tools=DEFINITIONS)

    # Dispatch a tool call from Claude's response:
    result = await dispatch(tool_use.name, tool_use.input)
"""

from __future__ import annotations

from sherlock.tools import gateway, github, loki, prometheus

# All Claude tool schema definitions — pass directly to anthropic.messages.create()
DEFINITIONS: list[dict] = [
    *github.DEFINITIONS,
    *loki.DEFINITIONS,
    *gateway.DEFINITIONS,
    *prometheus.DEFINITIONS,
]

# name → async callable
_HANDLERS: dict = {
    **github.HANDLERS,
    **loki.HANDLERS,
    **gateway.HANDLERS,
    **prometheus.HANDLERS,
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
