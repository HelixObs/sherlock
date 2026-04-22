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


class AgentDoc(BaseModel):
    """A GitHub repo from which Sherlock fetches AGENT.md files for context."""
    repo: str                        # e.g. https://github.com/HelixObs/mock-telescope
    paths: list[str] = Field(default_factory=lambda: ["AGENT.md"])  # paths to fetch


class InstrumentContext(BaseModel):
    instrument_id: str
    description: str = ""
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    pipeline: list[PipelineStage] = Field(default_factory=list)
    contacts: Contacts = Field(default_factory=Contacts)
    agent_docs: list[AgentDoc] = Field(default_factory=list)


# ── API shapes ────────────────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    instrument_id: str = ""
    github_token: str = ""   # operator's PAT for private repo source fetching; never logged


class ReplyRequest(BaseModel):
    content: str


class HypothesisData(BaseModel):
    classification: str = "unknown"
    confidence: str = "low"
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)
    recommendation: str = ""
    gaps: str = ""


class MemoryEntry(BaseModel):
    id: str = ""
    instrument_id: str
    entity_id: str = ""
    error_type: str = ""
    stage: str = ""
    classification: str = "unknown"
    confidence: str = "low"
    summary: str = ""
    recommendation: str = ""
    created_at: str = ""


class DiagnoseChunk(BaseModel):
    """One NDJSON chunk streamed during an investigation."""
    type: str           # "step" | "evidence" | "hypothesis" | "question" | "memory_prompt" | "done" | "error"
    text: str = ""
    data: dict = Field(default_factory=dict)
