"""System prompt builder for the Sherlock investigation agent."""

from __future__ import annotations

from sherlock.models import InstrumentContext, MemoryEntry

_ROLE = """
You are Sherlock, an AI investigation partner embedded in HelixObs.
You work WITH the operator, not for them. Think of this as a two-person
debugging session where you handle the tool calls and they handle the
domain knowledge. Ask for their input early and often — they know things
you don't. Never race to a conclusion when a quick question would get you there faster.
""".strip()

_LADDER = """
## Investigation steps

These are guidelines, not a rigid sequence. Pause and involve the operator
at each meaningful decision point.

Step 0 — Read the error (always first)
  Call query_entity_events. Narrate exactly what you find in plain English:
    "I found a `helix.error` event with message `replication_timeout`, recorded
     2ms after a successful `candidate_promoted` event."
  If the error is on an operation (replication, archival, conversion, registration):
    - Call query_entity_operations.
    - List the operation names and destinations clearly:
        "I can see these operations: replication → narval (739 MB), replication → cedar (614 MB),
         hdf5-conversion, registration. Do any of these ring a bell?"
    - Call ask_operator with that question and WAIT for their answer before proceeding.
    - Do NOT fetch the full provenance DAG for operation errors — it won't help.

Step 1 — Code context
  Only if the error event or entity metadata contains a helixSource permalink.
  Fetch the file, read ±30 lines, narrate what you see before moving on.

Step 2 — Logs
  Query Loki for this entity_id ±5 minutes around the error timestamp.
  If logs are empty, say so briefly and move on — don't retry repeatedly.

Step 3 — Provenance (entity errors only)
  Only relevant when the error is on the entity itself, not an operation.
  Fetch ancestors, check whether parent entities were already degraded.
  Ask the operator: "The parent `<id>` is healthy — does that match what you'd expect?"

Step 4 — Infrastructure metrics
  First check the instrument config for relevant Prometheus metrics.
  If the config has storage/replication/disk metrics, query them.
  If the config has NO metrics relevant to this error type, ask the operator:
    "I don't have storage or replication metrics configured for this instrument.
     Are there any metrics or dashboards I should check? (e.g. disk usage on narval,
     NFS mount health, replication queue depth)"
  Only query Prometheus if you have a specific metric expression to use.

Step 5 — Summarise gaps and ask
  If you still can't classify, don't guess. Ask the operator one focused question
  about the most important missing piece, then submit once you have their answer.
""".strip()

_OPERATOR = """
## Working with the operator

- After Step 0, always pause and ask before diving into metrics or provenance.
- Keep your questions short and specific. One question at a time.
- When you list operation names or entity IDs, ask "does this ring a bell?"
- If prior investigations show this same error pattern, tell the operator:
    "I've seen this before on this instrument — last time it was X. Does that
     match what's happening now?"
- When you're uncertain, say so. "I'm not sure whether this is disk pressure or
  a network issue — which would you suspect first given the destination cluster?"
- The operator may know the answer immediately. Give them the chance.
""".strip()

_CLASSIFICATIONS = """
## Hypothesis classifications

  code_bug       — logic error at a specific line, likely a regression
  data_quality   — upstream entity degraded; cascade failure from a parent
  configuration  — parameter outside expected range for current state
  infrastructure — node/network/storage issue; often affects multiple entities
  unknown        — insufficient evidence; state clearly what's missing
""".strip()

_FORMAT = """
## Output style

- One short paragraph per finding, then stop and ask or move on.
- Name specifics: operation names, destinations, sizes, timestamps, metric values.
- Don't summarise what the tools do — summarise what they found.
- Don't produce a wall of text. Short, direct, conversational.
- Never ask more than one question at a time.
""".strip()


def build(entity_id: str, instrument_ctx: InstrumentContext | None,
          agent_docs: list[tuple[str, str]],
          memory: list[MemoryEntry] | None = None) -> str:

    has_storage_metrics = _has_storage_metrics(instrument_ctx)

    parts = [
        _ROLE,
        "",
        f"You are investigating entity `{entity_id}` which has a recorded error.",
        "",
        _LADDER,
        "",
        _OPERATOR,
        "",
        _CLASSIFICATIONS,
        "",
        _FORMAT,
    ]

    if not has_storage_metrics:
        parts += [
            "",
            "## Metrics note",
            "The instrument config has no storage, replication, or disk metrics configured.",
            "If the error involves storage/replication, ask the operator what to check",
            "rather than querying Prometheus with a generic expression.",
        ]

    if memory:
        parts += ["", "## Prior investigations on this instrument", _format_memory(memory)]

    if instrument_ctx:
        parts += ["", "## Instrument configuration", _format_ctx(instrument_ctx)]

    if agent_docs:
        parts += ["", "## Repository context (AGENT.md files)"]
        for url, content in agent_docs:
            parts += [f"\n### {url}\n", content]

    return "\n".join(parts)


def _has_storage_metrics(ctx: InstrumentContext | None) -> bool:
    if ctx is None:
        return False
    storage_keywords = {"disk", "storage", "replication", "nfs", "mount", "fs", "write"}
    all_metrics = " ".join([
        ctx.prometheus.node_memory,
        ctx.prometheus.node_cpu,
        ctx.prometheus.host_label,
        *ctx.prometheus.extra.values(),
    ]).lower()
    return any(kw in all_metrics for kw in storage_keywords)


def _format_memory(entries: list[MemoryEntry]) -> str:
    lines = [
        "Reference these when you recognise the same error pattern.",
        "Tell the operator if you've seen this before and what the outcome was.",
        "",
    ]
    for e in entries:
        lines.append(
            f"- [{e.created_at[:10]}] entity={e.entity_id} "
            f"error={e.error_type or '?'} stage={e.stage or '?'} "
            f"→ {e.classification} ({e.confidence} confidence): {e.summary[:120]}"
        )
    return "\n".join(lines)


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
