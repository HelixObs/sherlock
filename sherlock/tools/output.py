"""Loop-control output tools.

These tools are not investigation tools — they signal to the agent loop
what to do next. Claude calls them instead of just producing free text,
giving the UI structured data for the hypothesis card and operator prompt.

submit_hypothesis  — terminates the investigation with a classification
ask_operator       — pauses for a targeted operator question (Tier 3)
"""

from __future__ import annotations

DEFINITIONS = [
    {
        "name": "submit_hypothesis",
        "description": (
            "Submit your final hypothesis and end the investigation. "
            "Call this when you have gathered sufficient evidence to commit to a classification. "
            "Do NOT call this speculatively — work through the investigation ladder first. "
            "State your confidence honestly: if evidence is thin, say so."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": ["code_bug", "data_quality", "configuration", "infrastructure", "unknown"],
                    "description": "Root cause classification",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence in this classification",
                },
                "summary": {
                    "type": "string",
                    "description": "2-3 sentence plain-English explanation of what happened and why",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific evidence items that support this classification (logs, metrics, code lines, provenance)",
                },
                "recommendation": {
                    "type": "string",
                    "description": "Concrete next action for the operator",
                },
                "gaps": {
                    "type": "string",
                    "description": "What remains unexplained or what you would check next if you had more access. Leave empty if confident.",
                },
            },
            "required": ["classification", "confidence", "summary", "evidence", "recommendation"],
        },
    },
    {
        "name": "ask_operator",
        "description": (
            "Ask the operator a specific, targeted question when you cannot proceed without "
            "information that is not available from tools. "
            "Use this sparingly — exhaust tool-based investigation first. "
            "Ask one focused question, not multiple at once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question for the operator",
                },
                "context": {
                    "type": "string",
                    "description": "Brief explanation of why you need this information and what you've already ruled out",
                },
            },
            "required": ["question", "context"],
        },
    },
]

# These tools have no Python handlers — the loop interprets them directly.
HANDLERS: dict = {}
