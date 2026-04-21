"""Pydantic models for Sherlock — instrument config, API request/response shapes."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Tier 1: Instrument config (loaded from YAML) ──────────────────────────────

class KnownFailure(BaseModel):
    type: str
    hint: str = ""


class PipelineStage(BaseModel):
    name: str
    description: str = ""
    known_failures: list[KnownFailure] = Field(default_factory=list)


class PrometheusConfig(BaseModel):
    node_memory: str = ""
    node_cpu: str = ""
    host_label: str = ""
    extra: dict[str, str] = Field(default_factory=dict)


class Contacts(BaseModel):
    oncall: str = ""
    escalation: str = ""


class InstrumentContext(BaseModel):
    instrument_id: str
    description: str = ""
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    pipeline: list[PipelineStage] = Field(default_factory=list)
    contacts: Contacts = Field(default_factory=Contacts)


# ── API shapes ────────────────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    instrument_id: str = ""


class ReplyRequest(BaseModel):
    content: str


class MemoryEntry(BaseModel):
    id: str = ""
    instrument_id: str
    error_type: str = ""
    stage: str = ""
    content: str
    confidence: str = "low"   # low | medium | high
    learned_by: str = ""
    learned_from: str = ""    # entity_id of triggering incident


class DiagnoseChunk(BaseModel):
    """One NDJSON chunk streamed during an investigation."""
    type: str           # "step" | "hypothesis" | "question" | "memory_prompt" | "done" | "error"
    text: str = ""
    data: dict = Field(default_factory=dict)
