"""Tests for the Tier 1 instrument context loader."""

import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from sherlock import context


@pytest.fixture(autouse=True)
def clear_cache():
    context.reload()
    yield
    context.reload()


def _write_config(dir: Path, name: str, content: str) -> None:
    (dir / name).write_text(textwrap.dedent(content))


def test_load_chime_bundled_config():
    """The bundled chime-context.yml should load without error."""
    ctx = context.load("CHIME")
    assert ctx is not None
    assert ctx.instrument_id == "CHIME"
    assert ctx.prometheus.node_memory != ""
    assert len(ctx.pipeline) > 0
    assert ctx.contacts.oncall != ""


def test_load_returns_none_for_unknown_instrument():
    ctx = context.load("NONEXISTENT_XYZ")
    assert ctx is None


def test_load_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERLOCK_INSTRUMENTS_DIR", str(tmp_path))
    _write_config(tmp_path, "test-context.yml", """
        instrument_id: TEST
        description: test instrument
        prometheus: {}
        pipeline:
          stages: []
        contacts: {}
    """)

    ctx1 = context.load("TEST")
    ctx2 = context.load("TEST")
    assert ctx1 is ctx2  # same object — cached


def test_reload_clears_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERLOCK_INSTRUMENTS_DIR", str(tmp_path))
    _write_config(tmp_path, "test-context.yml", """
        instrument_id: TEST
        description: first
        prometheus: {}
        pipeline:
          stages: []
        contacts: {}
    """)
    ctx1 = context.load("TEST")
    assert ctx1.description == "first"

    context.reload()
    _write_config(tmp_path, "test-context.yml", """
        instrument_id: TEST
        description: second
        prometheus: {}
        pipeline:
          stages: []
        contacts: {}
    """)
    ctx2 = context.load("TEST")
    assert ctx2.description == "second"


def test_known_failures_parsed(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERLOCK_INSTRUMENTS_DIR", str(tmp_path))
    _write_config(tmp_path, "mytel-context.yml", """
        instrument_id: MYTEL
        pipeline:
          stages:
            - name: classifier
              description: classifies things
              known_failures:
                - type: TIMEOUT
                  hint: check GPU memory
                - type: OOM
                  hint: reduce batch size
    """)
    ctx = context.load("MYTEL")
    assert ctx is not None
    stage = ctx.pipeline[0]
    assert stage.name == "classifier"
    assert len(stage.known_failures) == 2
    assert stage.known_failures[0].type == "TIMEOUT"
    assert "GPU" in stage.known_failures[0].hint


def test_agent_docs_parsed(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERLOCK_INSTRUMENTS_DIR", str(tmp_path))
    _write_config(tmp_path, "mytel-context.yml", """
        instrument_id: MYTEL
        agent_docs:
          - repo: https://github.com/HelixObs/mock-telescope
          - repo: https://github.com/HelixObs/gateway
            paths: [AGENT.md, internal/db/AGENT.md]
    """)
    ctx = context.load("MYTEL")
    assert ctx is not None
    assert len(ctx.agent_docs) == 2
    assert ctx.agent_docs[0].repo == "https://github.com/HelixObs/mock-telescope"
    assert ctx.agent_docs[0].paths == ["AGENT.md"]   # default
    assert ctx.agent_docs[1].paths == ["AGENT.md", "internal/db/AGENT.md"]


def test_raw_url_conversion():
    from sherlock.context import _raw_url
    url = _raw_url("https://github.com/HelixObs/gateway", "AGENT.md")
    assert url == "https://raw.githubusercontent.com/HelixObs/gateway/main/AGENT.md"


def test_raw_url_strips_git_suffix():
    from sherlock.context import _raw_url
    url = _raw_url("https://github.com/HelixObs/gateway.git", "internal/db/AGENT.md")
    assert "raw.githubusercontent.com" in url
    assert "gateway.git" not in url   # .git suffix stripped from repo slug


def test_prometheus_extra_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERLOCK_INSTRUMENTS_DIR", str(tmp_path))
    _write_config(tmp_path, "mytel-context.yml", """
        instrument_id: MYTEL
        prometheus:
          node_memory: mem_metric
          custom_metric: my_custom_gauge
    """)
    ctx = context.load("MYTEL")
    assert ctx.prometheus.node_memory == "mem_metric"
    assert ctx.prometheus.extra["custom_metric"] == "my_custom_gauge"
