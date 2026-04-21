# sherlock

Sherlock is the HelixObs AI troubleshooting agent. It performs structured, on-demand root cause analysis for entity errors using a three-tier context system and the Claude API.

## What this is

A FastAPI service that runs an investigation ladder (code context → logs → provenance → metrics → open question) when an operator clicks "Diagnose with AI" in the HelixObs UI. Investigation results stream to the browser as NDJSON via a Next.js SSE proxy.

## Package layout

```
sherlock/
  __init__.py
  main.py        FastAPI app — routes for /diagnose, /memory, /health
  context.py     Tier 1 instrument context loader (YAML files from instruments/)
  models.py      Pydantic models — InstrumentContext, DiagnoseChunk, MemoryEntry, etc.
  sessions.py    In-memory session store with 30-min TTL
instruments/
  chime-context.yml   Reference CHIME instrument config
tests/
  test_context.py
```

## Three-tier context system

| Tier | Source | Status |
|------|--------|--------|
| 1 | YAML files in `instruments/` keyed by `instrument_id` | Implemented |
| 2 | `instrument_memory` TimescaleDB table | Task 8 |
| 3 | Live operator replies via `/diagnose/{session_id}/reply` | Task 6 |

## Investigation ladder (Task 5/6)

Level 1 → code context (helixSource GitHub permalink)
Level 2 → logs (Loki query for entity_id ± 5 min)
Level 3 → entity provenance (gateway recursive CTE)
Level 4 → node metrics (Prometheus, using Tier 1/2 metric names)
Level 5 → open question (state gaps, ask operator)

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SHERLOCK_INSTRUMENTS_DIR` | `../instruments` relative to package | Path to YAML instrument configs |
| `ANTHROPIC_API_KEY` | — | Required for Task 6 (agent loop) |

## Running locally

```bash
pip install -e ".[dev]"
uvicorn sherlock.main:app --reload --port 8082
```

## Adding a new instrument config

Create `instruments/<instrument_id_lowercase>-context.yml` following the CHIME example. The context loader tries `{slug}-context.yml` then `{slug}.yml` — no code changes needed.
