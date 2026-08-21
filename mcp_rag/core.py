"""Shared core layer — used by BOTH MCP tools and admin HTTP (ADMIN-002).

No transport logic here: pure orchestration over the module layer
(chunker → embed → store for indexing; store → llm for query). Errors
surface as `RagError` subclasses; health paths never raise.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict

from mcp_rag.chunker import Chunker
from mcp_rag.config import JSON_FIELDS, AppConfig, load_config, save_config
from mcp_rag.embed import OllamaEmbedder
from mcp_rag.errors import RagError
from mcp_rag.indexer import IndexReport, Indexer, MAX_FILE_BYTES
from mcp_rag.llm import LLMProvider
from mcp_rag.qdrant_store import QdrantStore

MAX_TOP_K = 8
MAX_LIST_LIMIT = 1000

INT_FIELDS = ("top_k", "chunk_size", "chunk_overlap")
FLOAT_FIELDS = ("score_threshold",)


class RagService:
    """Dependency-injected orchestrator. Construct with explicit deps for
    tests, or via `RagService.from_config()` for real modules."""

    def __init__(self, store, embedder, llm, chunker, indexer, cfg):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.chunker = chunker
        self.indexer = indexer
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: AppConfig | None = None) -> "RagService":
        """Build a fully-wired service from config. Constructing does NOT
        touch the network (Qdrant/Ollama connect only on operations)."""
        cfg = cfg or load_config()
        store = QdrantStore(cfg)
        embedder = OllamaEmbedder(cfg)
        llm = LLMProvider(cfg)
        chunker = Chunker(cfg.chunk_size, cfg.chunk_overlap)
        indexer = Indexer(store, embedder, chunker, cfg)
        return cls(store, embedder, llm, chunker, indexer, cfg)

    # ------------------------------------------------------------------ index
    def index_files(
        self,
        source=None,
        path=None,
        force_rescan=False,
        max_files=None,
        progress_cb=None,
    ) -> IndexReport:
        """Index the vault root, or a sub-path when `path` is given. If `path`
        is a file, only that file is indexed; if a dir, that tree is walked."""
        source = source or self.cfg.default_source
        root = path or self.cfg.vault_root
        if not root:
            raise RagError("no vault root: set VAULT_ROOT or pass 'path'")
        if os.path.isfile(root):
            return self._index_single(root, source, force_rescan)
        return self.indexer.scan(
            root, source, force=force_rescan, max_files=max_files, progress_cb=progress_cb
        )

    def _index_single(self, path: str, source: str, force: bool) -> IndexReport:
        report = IndexReport()
        start = time.monotonic()
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            report.errors.append(f"{path}: {exc}")
            return report
        if size > MAX_FILE_BYTES:
            report.errors.append(f"{path}: file too large (>16MB)")
            return report

        mtime = int(os.path.getmtime(path))
        indexed = {
            d["path"]: d["last_indexed"]
            for d in self.store.list_documents(source=source)
        }
        report.files_scanned = 1
        if not force and path in indexed and indexed[path] == mtime:
            report.files_skipped = 1
            return report

        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            report.errors.append(f"{path}: {exc}")
            return report

        md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        try:
            chunks = self.chunker.chunk(path, text, mtime, md5)
        except RagError as exc:
            report.errors.append(str(exc))
            return report
        report.files_indexed = 1

        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            points = [
                {
                    "vector": vec,
                    "source": source,
                    "path": path,
                    "mtime": mtime,
                    "md5": md5,
                    "heading_path": c.heading_path,
                    "text": c.text,
                    "index": i,
                }
                for i, (c, vec) in enumerate(zip(chunks, vectors))
            ]
            self.store.upsert(points)
            report.chunks_upserted = len(points)

        report.duration_s = time.monotonic() - start
        return report

    # ------------------------------------------------------------------ query
    def query_rag(
        self,
        question: str,
        top_k: int = 5,
        source=None,
        path_prefix=None,
        score_threshold=None,
    ) -> dict:
        """Embed the question, retrieve, then (only if hits) call the LLM."""
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        threshold = score_threshold if score_threshold is not None else self.cfg.score_threshold
        vector = self.embedder.embed([question])[0]
        hits = self.store.search(
            vector, source=source, path_prefix=path_prefix, limit=top_k, threshold=threshold
        )
        if not hits:
            return {"answer": "No relevant documents found for this query.", "sources": [], "model": None}

        answer = self.llm.answer(question, hits)
        sources = [
            {
                "path": h.path,
                "heading_path": h.heading_path,
                "score": h.score,
                "snippet": h.snippet,
                "mtime": h.mtime,
            }
            for h in answer.sources
        ]
        return {"answer": answer.text, "sources": sources, "model": answer.model}

    def search_vec(
        self,
        query: str,
        top_k: int = 5,
        source=None,
        path_prefix=None,
        score_threshold=None,
    ) -> list:
        """Raw vector retrieval — no LLM (MCP-004)."""
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        threshold = score_threshold if score_threshold is not None else self.cfg.score_threshold
        vector = self.embedder.embed([query])[0]
        return self.store.search(
            vector, source=source, path_prefix=path_prefix, limit=top_k, threshold=threshold
        )

    # ---------------------------------------------------------------- delete
    def delete_docs(self, path=None, source=None) -> int:
        if path is not None:
            return self.store.delete_by_path(path)
        if source is not None:
            return self.store.delete_by_source(source)
        raise RagError("delete requires 'path' (exact) or 'source'")

    # ------------------------------------------------------------------- list
    def list_docs(self, source=None, path_prefix=None, limit=100, offset=0) -> list[dict]:
        limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        return self.store.list_documents(
            source=source, path_prefix=path_prefix, limit=limit, offset=offset
        )

    # ------------------------------------------------------------------ stats
    def get_stats(self) -> dict:
        stats = self.store.get_collection_stats()
        sources = []
        for src, count in stats.get("by_source", {}).items():
            files = len(self.store.list_documents(source=src, limit=MAX_LIST_LIMIT))
            sources.append({"source": src, "files": files, "chunks": count})

        docs = self.store.list_documents(limit=MAX_LIST_LIMIT)
        last_index_at = max((d["last_indexed"] for d in docs), default=0)

        return {
            "collection": self.cfg.collection,
            "vectors_count": stats.get("vectors_count", 0),
            "sources": sources,
            "last_index_at": last_index_at,
            "qdrant_health": self.store.health(),
            "ollama_health": self.embedder.health(),
            "llm_configured": bool(self.llm.health()),
        }

    # ----------------------------------------------------------------- config
    def config_get(self) -> dict:
        data = asdict(self.cfg)
        if isinstance(data.get("ext"), tuple):
            data["ext"] = list(data["ext"])
        return data

    def config_set(self, **kwargs) -> dict:
        for key, value in kwargs.items():
            if key not in JSON_FIELDS:
                raise RagError(
                    f"invalid config key '{key}'; valid keys: {', '.join(JSON_FIELDS)}"
                )
            if key in INT_FIELDS and not isinstance(value, int):
                raise RagError(f"config '{key}' must be an int")
            if key in FLOAT_FIELDS and not isinstance(value, (int, float)):
                raise RagError(f"config '{key}' must be a number")
            setattr(self.cfg, key, value)
        save_config(self.cfg)
        return self.config_get()
