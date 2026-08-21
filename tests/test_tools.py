"""MCP tool tests — FastMCP app invoked in-process with fakes (offline)."""
import asyncio
import os

import pytest

from mcp_rag.chunker import Chunker
from mcp_rag.config import AppConfig
from mcp_rag.core import RagService
from mcp_rag.indexer import Indexer
from mcp_rag.qdrant_store import QdrantStore, DIMENSIONS
from mcp_rag.server import create_app

# Reuse the fakes from test_core (same deterministic behavior).
from tests.test_core import FakeEmbedder, FakeLLM, _write  # noqa: E402


@pytest.fixture
def service(tmp_path):
    store = QdrantStore(cfg=AppConfig(), client=__import__("qdrant_client").QdrantClient(":memory:"))
    store.ensure_collection()
    cfg = AppConfig(vault_root="", score_threshold=0.0)
    embedder = FakeEmbedder()
    llm = FakeLLM()
    chunker = Chunker()
    indexer = Indexer(store, embedder, chunker, cfg)
    return RagService(store, embedder, llm, chunker, indexer, cfg)


@pytest.fixture
def app(service):
    return create_app(service)


def call(app, name, args=None):
    """Run a tool in-process, bypassing JSON/ContentBlock conversion."""
    async def _run():
        tool = app._tool_manager.get_tool(name)
        # tool.run injects the `ctx` kwarg (context_kwarg) and returns the
        # plain Python result with convert_result=False.
        result = await tool.run(args or {}, convert_result=False)
        return result
    return asyncio.run(_run())


def tool_names(app):
    async def _run():
        return {t.name for t in await app.list_tools()}
    return asyncio.run(_run())


def test_all_tools_registered(app):
    names = tool_names(app)
    assert names == {"index", "query", "search", "delete", "list", "stats", "config"}


def test_index_returns_report_shape(app, service, tmp_path):
    _write(tmp_path, "a.md", "# Setup\nconfigurar el servidor linux\n", mtime=1)
    result = call(app, "index", {"path": str(tmp_path)})
    assert result["files_scanned"] == 1
    assert result["files_indexed"] == 1
    assert result["chunks_upserted"] > 0
    assert result["errors"] == []


def test_index_reports_progress_dict(app, service, tmp_path):
    # Progress via ctx.report_progress requires a live MCP request context;
    # in-process it is a safe no-op (see _safe_report). We assert the index
    # report is returned with the expected fields.
    _write(tmp_path, "a.md", "# A\nhola\n", mtime=1)
    result = call(app, "index", {"path": str(tmp_path)})
    assert "files_scanned" in result
    assert "duration_s" in result
    assert "stale_removed" in result


def test_query_returns_answer(app, service, tmp_path):
    _write(tmp_path, "a.md", "# Setup\nconfigurar el servidor linux\n", mtime=1)
    call(app, "index", {"path": str(tmp_path)})
    result = call(app, "query", {"question": "configurar el servidor linux"})
    assert result["answer"].startswith("answer for:")
    assert result["model"] == "fake"
    assert result["sources"][0]["path"].endswith("a.md")


def test_query_requires_question(app):
    with pytest.raises(Exception):
        call(app, "query", {})


def test_query_top_k_clamped(app, service, tmp_path):
    for i in range(10):
        _write(tmp_path, f"f{i}.md", f"# F{i}\ncontenido del archivo numero {i}\n", mtime=i)
    call(app, "index", {"path": str(tmp_path), "max_files": 10})
    result = call(app, "query", {"question": "contenido del archivo", "top_k": 20})
    assert len(result["sources"]) <= 8


def test_search_raw_no_llm(app, service, tmp_path):
    _write(tmp_path, "a.md", "# Doc\ntermino unico busqueda\n", mtime=1)
    call(app, "index", {"path": str(tmp_path)})
    hits = call(app, "search", {"text": "termino unico busqueda"})
    assert isinstance(hits, list)
    assert hits[0]["path"].endswith("a.md")
    assert "score" in hits[0]


def test_delete_requires_path_or_source(app):
    with pytest.raises(Exception):
        call(app, "delete", {})


def test_delete_by_path(app, service, tmp_path):
    p = _write(tmp_path, "a.md", "# A\nhola mundo\n", mtime=1)
    call(app, "index", {"path": str(tmp_path)})
    result = call(app, "delete", {"path": p})
    assert result["deleted_points"] >= 1
    remaining = call(app, "list", {})
    assert all(d["path"] != p for d in remaining)


def test_list_paginated(app, service, tmp_path):
    for i in range(6):
        _write(tmp_path, f"f{i}.md", f"# F{i}\ncontenido {i}\n", mtime=i)
    call(app, "index", {"path": str(tmp_path)})
    page = call(app, "list", {"limit": 2, "offset": 0})
    assert len(page) >= 1
    assert all("path" in d and "chunks" in d and "last_indexed" in d for d in page)


def test_stats_shape(app, service, tmp_path):
    _write(tmp_path, "a.md", "# A\ncontenido\n", mtime=5)
    call(app, "index", {"path": str(tmp_path)})
    stats = call(app, "stats", {})
    assert stats["collection"] == "supervisor"
    assert stats["vectors_count"] >= 1
    assert stats["llm_configured"] is True


def test_config_get_and_set(app, service, tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("MCP_RAG_CONFIG", str(cfg_path))
    full = call(app, "config", {"action": "get"})
    assert full["top_k"] == 5
    updated = call(app, "config", {"action": "set", "key": "top_k", "value": 7})
    assert updated["top_k"] == 7


def test_config_set_invalid_field(app, service, tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("MCP_RAG_CONFIG", str(cfg_path))
    with pytest.raises(Exception):
        call(app, "config", {"action": "set", "key": "bogus", "value": 1})
