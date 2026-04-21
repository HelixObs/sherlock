"""System prompt builder for the Sherlock investigation agent."""

from __future__ import annotations

from sherlock.models import InstrumentContext

_LADDER = """
## Investigation ladder

Work through these levels in order. Stop and call submit_hypothesis when you
have sufficient evidence. Do not skip levels.

Level 1 — Code context
  If the entity has a helixSource field in its metadata or in log lines,
  fetch the source file and read ±30 lines around the error line.
  Follow one level up the call chain with search_github_callers.

Level 2 — Logs
  Query Loki for this entity_id in a ±5 minute window around the error timestamp.
  Look for warnings, exceptions, or resource alerts preceding the error.

Level 3 — Entity provenance
  Fetch the full provenance DAG. Were parent entities already degraded?
  Query for similar errors across the instrument in the last 60 minutes.
  Isolated failure vs widespread pattern changes the classification.

Level 4 — Node metrics
  Use metric names from the instrument config to query Prometheus at the
  exact error timestamp (convert nanoseconds to seconds).
  Correlate memory / CPU / disk with the error time.
  If the instrument config has no relevant metric, call ask_operator once.

Level 5 — Open question
  If you still cannot classify after levels 1-4, call submit_hypothesis with
  classification="unknown", state what you found and what remains unexplained.
""".strip()

_CLASSIFICATIONS = """
## Hypothesis classifications

Commit to one before calling submit_hypothesis:

  code_bug       — logic error at a specific line, likely a regression
  data_quality   — upstream entity degraded; this is a cascade failure
  configuration  — parameter outside expected range for current state
  infrastructure — pattern across many entities; node metric correlation
  unknown        — insufficient evidence after exhausting all levels
""".strip()

_FORMAT = """
## Output format

- Write in plain English, not bullet soup.
- Be specific: name files, line numbers, metric values, entity IDs.
- State confidence honestly. "This looks like X (high confidence — metric Y
  confirms Z)" or "I'm not certain — the code path looks correct but I cannot
  rule out a data quality issue without seeing the parent entity's metadata."
- Do not produce a confident-sounding answer when evidence is thin.
- Cap yourself at 10 conversation turns total.
""".strip()


def build(entity_id: str, instrument_ctx: InstrumentContext | None,
          agent_docs: list[tuple[str, str]]) -> str:
    parts = [
        "You are Sherlock, the HelixObs AI troubleshooting agent.",
        f"You are investigating entity `{entity_id}` which has a recorded error.",
        "",
        _LADDER,
        "",
        _CLASSIFICATIONS,
        "",
        _FORMAT,
    ]

    if instrument_ctx:
        parts += ["", "## Instrument configuration", _format_ctx(instrument_ctx)]

    if agent_docs:
        parts += ["", "## Repository context (AGENT.md files)"]
        for url, content in agent_docs:
            parts += [f"\n### {url}\n", content]

    return "\n".join(parts)


def _format_ctx(ctx: InstrumentContext) -> str:
    lines = [
        f"Instrument: {ctx.instrument_id}",
        f"Description: {ctx.description}",
    ]

    if ctx.prometheus.node_memory or ctx.prometheus.node_cpu:
        lines.append("\nPrometheus metrics:")
        if ctx.prometheus.node_memory:
            lines.append(f"  node_memory: {ctx.prometheus.node_memory}")
        if ctx.prometheus.node_cpu:
            lines.append(f"  node_cpu:    {ctx.prometheus.node_cpu}")
        if ctx.prometheus.host_label:
            lines.append(f"  host_label:  {ctx.prometheus.host_label}")
        for k, v in ctx.prometheus.extra.items():
            lines.append(f"  {k}: {v}")

    if ctx.pipeline:
        lines.append("\nPipeline stages and known failures:")
        for stage in ctx.pipeline:
            lines.append(f"  {stage.name}: {stage.description}")
            for f in stage.known_failures:
                lines.append(f"    - {f.type}: {f.hint}")

    if ctx.contacts.oncall:
        lines.append(f"\nOn-call: {ctx.contacts.oncall}")
    if ctx.contacts.escalation:
        lines.append(f"Escalation: {ctx.contacts.escalation}")

    return "\n".join(lines)
