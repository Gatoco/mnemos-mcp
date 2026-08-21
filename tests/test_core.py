"""Core layer tests — in-memory Qdrant + FakeEmbedder + FakeLLM (offline)."""
import math
import os

import pytest
from qdrant_client import QdrantClient

from mcp_rag.chunker import Chunker
from mcp_rag.config import AppConfig, JSON_FIELDS
from mcp_rag.core import RagService
from mcp_rag.errors import RagError
from mcp_rag.indexer import Indexer
from mcp_rag.llm import Answer
from mcp_rag.qdrant_store import QdrantStore, DIMENSIONS


class FakeEmbedder:
    @property
    def dimensions(self):
        return DIMENSIONS

    def embed(self, texts):
        return [
            [math.sin(i + hash(t) % 1000) for i in range(DIMENSIONS)] for t in texts
        ]

    def health(self):
        return (True, "ok")


class FakeLLM:
    """Records the context passed to answer(); raises if called with no hits."""

    def __init__(self, fail_if_called=False):
        self.calls = []
        self.fail_if_called = fail_if_called

    def answer(self, question, context):
        if self.fail_if_called:
            raise AssertionError("LLM called when it should not be")
        self.calls.append({"question": question, "context": context})
        return Answer(text=f"answer for: {question}", sources=context, model="fake")

    def health(self):
        return True


class FailLLM(FakeLLM):
    def health(self):
        return False


@pytest.fixture
def cfg():
    return AppConfig(vault_root="", score_threshold=0.0)


@pytest.fixture
def store():
    s = QdrantStore(cfg=AppConfig(), client=QdrantClient(":memory:"))
    s.ensure_collection()
    return s


@pytest.fixture
def service(store, cfg):
    embedder = FakeEmbedder()
    llm = FakeLLM()
    chunker = Chunker()
    indexer = Indexer(store, embedder, chunker, cfg)
    svc = RagService(store, embedder, llm, chunker, indexer, cfg)
    svc._fake_llm = llm  # for asserting LLM calls
    return svc


def _write(root, name, content, mtime=1):
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.utime(path, (mtime, mtime))
    return path


def test_index_then_search_vec(tmp_path, service):
    _write(tmp_path, "a.md", "# Setup\nconfigurar el servidor\n", mtime=1)
    rep = service.index_files(path=str(tmp_path))
    assert rep.files_indexed == 1
    assert rep.chunks_upserted > 0
    hits = service.search_vec("configurar el servidor", top_k=5)
    assert hits
    assert hits[0].path.endswith("a.md")


def test_index_single_file(tmp_path, service):
    path = _write(tmp_path, "single.md", "# Solo\nun archivo\n", mtime=1)
    rep = service.index_files(path=path)
    assert rep.files_scanned == 1
    assert rep.files_indexed == 1
    hits = service.search_vec("un archivo")
    assert hits and hits[0].path == path


def test_query_rag_returns_answer(tmp_path, service):
    _write(tmp_path, "a.md", "# Setup\nconfigurar el servidor linux\n", mtime=1)
    service.index_files(path=str(tmp_path))
    # Query with the indexed text so the deterministic FakeEmbedder vector matches
    # (identical text -> identical hash-derived vector -> score 1.0, above threshold).
    result = service.query_rag("configurar el servidor linux")
    assert result["answer"].startswith("answer for:")
    assert result["model"] == "fake"
    assert len(result["sources"]) >= 1
    assert result["sources"][0]["path"].endswith("a.md")
    # FakeLLM received the hits as context.
    assert service._fake_llm.calls and service._fake_llm.calls[0]["context"]


def test_no_hits_skips_llm(service):
    # Empty collection -> no hits -> no LLM call, generic message.
    service.llm = FailLLM()  # would raise if answer() were called
    result = service.query_rag("pregunta sin datos en el indice")
    assert result["answer"] == "No relevant documents found for this query."
    assert result["sources"] == []
    assert result["model"] is None
    assert not service._fake_llm.calls


def test_top_k_clamped(tmp_path, service):
    for i in range(10):
        _write(tmp_path, f"f{i}.md", f"# F{i}\ncontenido del archivo numero {i}\n", mtime=i)
    service.index_files(path=str(tmp_path), max_files=10)
    hits = service.search_vec("contenido del archivo", top_k=20)
    assert len(hits) <= 8


def test_delete_by_path_and_source(tmp_path, service):
    _write(tmp_path, "a.md", "# A\nhola mundo\n", mtime=1)
    _write(tmp_path, "b.md", "# B\nadios mundo\n", mtime=2)
    service.index_files(path=str(tmp_path))
    assert service.delete_docs(path=os.path.join(str(tmp_path), "a.md")) >= 1
    assert service.delete_docs(source="vault") >= 1
    with pytest.raises(RagError):
        service.delete_docs()


def test_list_docs_paginated_and_clamped(tmp_path, service):
    for i in range(15):
        _write(tmp_path, f"f{i}.md", f"# F{i}\ncontenido {i}\n", mtime=i)
    service.index_files(path=str(tmp_path))
    # Pagination is by chunk-scroll then grouped by path (QdrantStore behavior);
    # a doc's chunks may straddle a boundary, so pages are not strictly disjoint.
    page1 = service.list_docs(limit=5, offset=0)
    page2 = service.list_docs(limit=5, offset=5)
    assert len(page1) >= 1
    assert len(page2) >= 1
    all_docs = service.list_docs(limit=1000)
    assert len(all_docs) == 15
    assert len({d["path"] for d in all_docs}) == 15
    big = service.list_docs(limit=9999)
    assert len(big) <= 1000


def test_get_stats_shape(tmp_path, service):
    _write(tmp_path, "a.md", "# A\ncontenido\n", mtime=5)
    service.index_files(path=str(tmp_path))
    stats = service.get_stats()
    assert stats["collection"] == "supervisor"
    assert stats["vectors_count"] >= 1
    assert any(s["source"] == "vault" and s["files"] >= 1 and s["chunks"] >= 1 for s in stats["sources"])
    assert stats["last_index_at"] >= 5
    assert stats["qdrant_health"] == (True, "ok")
    assert stats["ollama_health"] == (True, "ok")
    assert stats["llm_configured"] is True  # FakeLLM.health() True


def test_llm_not_configured_reported(store, cfg):
    embedder = FakeEmbedder()
    llm = FailLLM()  # health() False
    chunker = Chunker()
    indexer = Indexer(store, embedder, chunker, cfg)
    svg = RagService(store, embedder, llm, chunker, indexer, cfg)
    stats = svg.get_stats()
    assert stats["llm_configured"] is False


def test_config_get_full_and_set_persists(tmp_path, monkeypatch, store):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("MCP_RAG_CONFIG", str(cfg_path))
    svc_cfg = AppConfig()  # defaults: score_threshold 0.5, top_k 5
    embedder = FakeEmbedder()
    llm = FakeLLM()
    chunker = Chunker()
    indexer = Indexer(store, embedder, chunker, svc_cfg)
    svg = RagService(store, embedder, llm, chunker, indexer, svc_cfg)

    full = svg.config_get()
    assert full["score_threshold"] == 0.5
    assert full["top_k"] == 5
    assert full["model"] == "deepseek-v4-flash"

    updated = svg.config_set(top_k=7)
    assert updated["top_k"] == 7
    # Persisted to the temp config file.
    assert cfg_path.exists()
    with open(cfg_path, encoding="utf-8") as fh:
        import json
        assert json.load(fh)["top_k"] == 7


def test_config_set_invalid_field(service):
    with pytest.raises(RagError):
        service.config_set(bogus=1)


def test_config_set_type_check(service):
    with pytest.raises(RagError):
        service.config_set(top_k="seven")


def test_from_config_builds_without_network():
    svc = RagService.from_config(AppConfig())
    assert svc.store.collection == "supervisor"
    assert svc.embedder.dimensions == DIMENSIONS
    assert svc.llm is not None
