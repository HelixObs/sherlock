# sherlock

Sherlock is the HelixObs AI troubleshooting agent. It performs structured, on-demand root cause analysis for entity errors using a three-tier context system and the Claude API.

## What this is

A FastAPI service that runs an investigation ladder (code context → logs → provenance → metrics → open question) when an operator clicks "Diagnose with AI" in the HelixObs UI. Investigation results stream to the browser as NDJSON via a Next.js SSE proxy.

## Package layout

```
sherlock/
  __init__.py
  main.py         FastAPI app — /diagnose, /diagnose/{session_id}/reply, /memory, /health
  agent.py        Agentic investigation loop — streams DiagnoseChunk objects via Claude API
  context.py      Tier 1 instrument context loader (YAML files from instruments/)
  memory.py       Tier 2 memory — load/save/delete instrument_memory rows in TimescaleDB
  metrics.py      Prometheus metrics + record_usage() → sherlock_usage TimescaleDB ledger
  models.py       Pydantic models — InstrumentContext, DiagnoseChunk, MemoryEntry, etc.
  prompt.py       System prompt builder — assembles Tier 1/2/3 context into Claude prompt
  sessions.py     In-memory session store with 30-min TTL
  tools/
    __init__.py   DEFINITIONS list + dispatch() — routes Claude tool calls to handlers
    gateway.py    query_entity, query_entity_ancestors, query_entity_events,
                  query_entity_operations, query_similar_errors
    loki.py       get_logs, search_logs
    prometheus.py get_metrics, query_prometheus
    github.py     fetch_github_file, fetch_github_blame, fetch_github_file_history,
                  search_github_callers
    output.py     submit_hypothesis, ask_operator (loop-control tools — no handlers)
    grafana.py    Grafana/Tempo URL helpers (not a tool, imported by gateway.py)
instruments/
  chime-context.yml   Reference CHIME instrument config
tests/
  test_api.py
  test_context.py
```

## Three-tier context system

| Tier | Source | Status |
|------|--------|--------|
| 1 | YAML files in `instruments/` keyed by `instrument_id` | Implemented |
| 2 | `instrument_memory` TimescaleDB table (past diagnoses) | Implemented |
| 3 | Live operator replies via `/diagnose/{session_id}/reply` | Implemented |

## Investigation ladder

1. **Code context** — `helixSource` GitHub permalink from error metadata (fetch_github_file, fetch_github_blame)
2. **Logs** — Loki query for entity_id ± 5 min (get_logs, search_logs)
3. **Entity provenance** — gateway recursive CTE (query_entity_ancestors)
4. **Node metrics** — Prometheus, using Tier 1/2 metric names (get_metrics, query_prometheus)
5. **Open question** — state gaps, ask operator (ask_operator tool → pauses session)

## Streaming protocol

Every endpoint returns `application/x-ndjson`. Each line is a JSON-encoded `DiagnoseChunk`:

| `type` | Meaning |
|---|---|
| `step` | Streaming text token or tool call announcement |
| `evidence` | Raw tool result (shown in UI evidence panel) |
| `hypothesis` | Final classification from `submit_hypothesis` |
| `question` | Operator question from `ask_operator` |
| `error` | Unrecoverable error |
| `done` | Investigation complete — includes `input_tokens`, `output_tokens`, `cost_usd`, `model` |

The first `step` chunk from a new `/diagnose` call always carries `data.session_id` so the
UI can wire follow-up replies.

## Memory (Tier 2)

After a successful `submit_hypothesis`, Sherlock writes to `instrument_memory` in TimescaleDB
via `memory.save()`. On a subsequent `/diagnose` for the same `entity_id`, `memory.load_for_entity()`
is checked first — if a prior record exists, the investigation is replayed from memory without
a Claude API call (`cost_usd=0`, `model="memory"` in the done chunk).

## Prometheus metrics

| Metric | Labels | Description |
|---|---|---|
| `sherlock_queries_total` | `query_type`, `status` | Investigations by type and outcome |
| `sherlock_query_duration_seconds` | `query_type` | End-to-end latency histogram |
| `sherlock_tokens_input_total` | `model` | Input tokens consumed |
| `sherlock_tokens_output_total` | `model` | Output tokens generated |
| `sherlock_tokens_total` | `model` | Combined input + output tokens |
| `sherlock_cost_usd_total` | `model` | Cumulative USD cost |
| `sherlock_cost_per_query_usd` | `model` | Per-investigation cost histogram |
| `sherlock_tool_calls_total` | `tool_name`, `status` | Tool call outcomes |
| `sherlock_tool_call_duration_seconds` | `tool_name` | Tool call latency |

All Prometheus counters are also persisted to the `sherlock_usage` TimescaleDB hypertable
via `record_usage()` for durable cost tracking — Prometheus data resets on restart.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SHERLOCK_INSTRUMENTS_DIR` | `../instruments` (relative to package) | Path to YAML instrument configs |
| `ANTHROPIC_API_KEY` | — | Required for the agent loop |
| `SHERLOCK_MODEL` | `claude-sonnet-4-6` | Claude model ID |
| `GATEWAY_URL` | `http://gateway:8080` | Gateway HTTP API base URL |
| `GATEWAY_DB_URL` | — | Direct TimescaleDB URL for entity_events / similar_errors queries |
| `SHERLOCK_METRICS_PORT` | `9102` | Prometheus metrics port |

## Running locally

```bash
pip install -e ".[dev]"
uvicorn sherlock.main:app --reload --port 8082
```

## Adding a new instrument config

Create `instruments/<instrument_id_lowercase>-context.yml` following the CHIME example.
The context loader tries `{slug}-context.yml` then `{slug}.yml` — no code changes needed.

## Adding a new investigation tool

1. Implement the async handler in the appropriate `tools/*.py` file.
2. Add the Claude tool schema to `DEFINITIONS` in that file.
3. Register the handler in `HANDLERS`.
4. The `tools/__init__.py` auto-imports from all tool modules — no changes needed there.
5. Tool calls are timed and counted automatically in `agent.py` via `sherlock_tool_calls_total`
   and `sherlock_tool_call_duration_seconds`.
