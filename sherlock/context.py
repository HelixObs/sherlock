"""Tier 1 instrument context loader.

Reads YAML files from the instruments/ directory (relative to the package root,
or the path set by SHERLOCK_INSTRUMENTS_DIR). Files are named by instrument_id
in lowercase: CHIME → chime-context.yml.

Results are cached in-process after first load. Call reload() to invalidate.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import httpx
import yaml

from sherlock.models import (
    AgentDoc,
    Contacts,
    InstrumentContext,
    KnownFailure,
    PipelineStage,
    PrometheusConfig,
)

log = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).parent.parent / "instruments"
_cache: dict[str, InstrumentContext] = {}

# Fetched AGENT.md content is cached for 1 hour to avoid hammering GitHub.
_doc_cache: dict[str, tuple[str, float]] = {}  # url → (content, fetched_at)
_DOC_TTL = 3600


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
    _doc_cache.clear()


def _raw_url(repo: str, path: str) -> str:
    """Convert a GitHub repo URL + file path to a raw.githubusercontent.com URL."""
    # Normalise: strip trailing slash and .git
    repo = repo.rstrip("/").removesuffix(".git")
    m = re.match(r"https?://github\.com/([^/]+/[^/]+)", repo)
    if not m:
        raise ValueError(f"unsupported repo URL: {repo!r}")
    return f"https://raw.githubusercontent.com/{m.group(1)}/main/{path}"


async def fetch_agent_docs(ctx: InstrumentContext, github_token: str = "") -> list[tuple[str, str]]:
    """Fetch all AGENT.md files listed in ctx.agent_docs from GitHub.

    Returns a list of (url, content) pairs for documents that were successfully
    fetched. Failures are logged and skipped — a missing doc is not fatal.
    Results are cached for _DOC_TTL seconds.
    """
    token = github_token or (ctx.github_token if ctx else "") or os.environ.get("GITHUB_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    results: list[tuple[str, str]] = []
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        for doc in ctx.agent_docs:
            for path in doc.paths:
                try:
                    url = _raw_url(doc.repo, path)
                except ValueError as exc:
                    log.warning("skip agent doc: %s", exc)
                    continue

                now = time.monotonic()
                if url in _doc_cache:
                    content, fetched_at = _doc_cache[url]
                    if now - fetched_at < _DOC_TTL:
                        results.append((url, content))
                        continue

                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        content = r.text
                        _doc_cache[url] = (content, now)
                        results.append((url, content))
                        log.info("fetched agent doc %s (%d chars)", url, len(content))
                    else:
                        log.warning("agent doc %s returned %d — skipping", url, r.status_code)
                except Exception as exc:
                    log.warning("failed to fetch agent doc %s: %s", url, exc)

    return results


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

    agent_docs = []
    for entry in raw.get("agent_docs", []):
        paths = entry.get("paths", ["AGENT.md"])
        agent_docs.append(AgentDoc(repo=entry["repo"], paths=paths))

    # Resolve GitHub token from notifications.github_token_env (same pattern as herald notifier).
    github_token = ""
    token_env = (raw.get("notifications") or {}).get("github_token_env", "")
    if token_env:
        github_token = os.environ.get(token_env, "")

    return InstrumentContext(
        instrument_id=raw["instrument_id"],
        description=raw.get("description", ""),
        prometheus=prometheus,
        pipeline=stages,
        contacts=contacts,
        agent_docs=agent_docs,
        github_token=github_token,
    )
