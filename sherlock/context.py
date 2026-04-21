"""Tier 1 instrument context loader.

Reads YAML files from the instruments/ directory (relative to the package root,
or the path set by SHERLOCK_INSTRUMENTS_DIR). Files are named by instrument_id
in lowercase: CHIME → chime-context.yml.

Results are cached in-process after first load. Call reload() to invalidate.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from sherlock.models import (
    Contacts,
    InstrumentContext,
    KnownFailure,
    PipelineStage,
    PrometheusConfig,
)

log = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).parent.parent / "instruments"
_cache: dict[str, InstrumentContext] = {}


def _instruments_dir() -> Path:
    custom = os.environ.get("SHERLOCK_INSTRUMENTS_DIR")
    return Path(custom) if custom else _DEFAULT_DIR


def _candidates(instrument_id: str) -> list[Path]:
    d = _instruments_dir()
    slug = instrument_id.lower()
    return [
        d / f"{slug}-context.yml",
        d / f"{slug}.yml",
        d / f"{instrument_id}-context.yml",
        d / f"{instrument_id}.yml",
    ]


def load(instrument_id: str) -> InstrumentContext | None:
    """Return the InstrumentContext for instrument_id, or None if not found."""
    if instrument_id in _cache:
        return _cache[instrument_id]

    for path in _candidates(instrument_id):
        if path.exists():
            try:
                raw = yaml.safe_load(path.read_text())
                ctx = _parse(raw)
                _cache[instrument_id] = ctx
                log.info("loaded instrument context %s from %s", instrument_id, path)
                return ctx
            except Exception as exc:
                log.warning("failed to parse %s: %s", path, exc)

    log.warning("no context file found for instrument %r", instrument_id)
    return None


def reload() -> None:
    """Invalidate the in-process cache (useful in tests and on config reload)."""
    _cache.clear()


def _parse(raw: dict) -> InstrumentContext:
    prom_raw = raw.get("prometheus") or {}
    known_keys = {"node_memory", "node_cpu", "host_label"}
    prometheus = PrometheusConfig(
        node_memory=prom_raw.get("node_memory", ""),
        node_cpu=prom_raw.get("node_cpu", ""),
        host_label=prom_raw.get("host_label", ""),
        extra={k: v for k, v in prom_raw.items() if k not in known_keys},
    )

    stages = []
    for s in (raw.get("pipeline") or {}).get("stages", []):
        failures = [KnownFailure(**f) for f in s.get("known_failures", [])]
        stages.append(PipelineStage(
            name=s["name"],
            description=s.get("description", ""),
            known_failures=failures,
        ))

    contacts_raw = raw.get("contacts") or {}
    contacts = Contacts(
        oncall=contacts_raw.get("oncall", ""),
        escalation=contacts_raw.get("escalation", ""),
    )

    return InstrumentContext(
        instrument_id=raw["instrument_id"],
        description=raw.get("description", ""),
        prometheus=prometheus,
        pipeline=stages,
        contacts=contacts,
    )
