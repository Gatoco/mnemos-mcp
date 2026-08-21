#!/usr/bin/env python3
"""Smoke test end-to-end (offline).

Builds a RagService with fake embedder/LLM + a real in-memory Qdrant store +
the real chunker and indexer, indexes 3 sample markdown files, then exercises
index -> search -> query. Prints a PASS line per stage and exits 0 on success,
nonzero on failure.

Runs entirely offline (no network, no real ollama/qdrant servers). Self-contained
(no imports from tests/) so it runs from anywhere.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

# Allow running as `python scripts/smoke.py` from the repo root or any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient  # noqa: E402

from mcp_rag.chunker import Chunker  # noqa: E402
from mcp_rag.config import AppConfig  # noqa: E402
from mcp_rag.core import RagService  # noqa: E402
from mcp_rag.indexer import Indexer  # noqa: E402
from mcp_rag.llm import Answer  # noqa: E402
from mcp_rag.qdrant_store import DIMENSIONS, QdrantStore  # noqa: E402


class FakeEmbedder:
    """Deterministic 1024-dim embedder (same text -> same vector)."""

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    def embed(self, texts):
        return [[math.sin(i + hash(t) % 1000) for i in range(DIMENSIONS)] for t in texts]

    def health(self):
        return (True, "ok")


class FakeLLM:
    def answer(self, question, context):
        return Answer(text=f"answer for: {question}", sources=context, model="fake")

    def health(self):
        return True


def _check(ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {detail}")
    return ok


def _write(root, name, content):
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def main() -> int:
    passed = True

    # 1. Service: real in-memory Qdrant store + real chunker/indexer + fakes.
    cfg = AppConfig()
    store = QdrantStore(cfg, client=QdrantClient(":memory:"))
    store.ensure_collection()
    embedder = FakeEmbedder()
    llm = FakeLLM()
    chunker = Chunker()
    indexer = Indexer(store, embedder, chunker, cfg)
    svc = RagService(store, embedder, llm, chunker, indexer, cfg)

    # 2. Sample files (one with frontmatter + nested headings to exercise the chunker).
    with tempfile.TemporaryDirectory() as tmp:
        _write(
            tmp,
            "setup.md",
            "---\ntags: [servidor]\n---\n"
            "# Setup\nconfigurar el servidor linux\n"
            "## Hardware\ndetalle del hardware instalado\n",
        )
        _write(tmp, "red.md", "# Red\nconfigurar la red del laboratorio\n")
        _write(tmp, "notas.txt", "contenido de respaldo sin headings\n")

        # 3. Index -> assert scanned/indexed counts and no errors.
        rep = svc.index_files(path=tmp)
        passed &= _check(
            rep.files_scanned == 3 and rep.files_indexed == 3,
            f"index: scanned={rep.files_scanned} indexed={rep.files_indexed}",
        )
        passed &= _check(rep.chunks_upserted > 0, f"chunks_upserted={rep.chunks_upserted}")
        passed &= _check(rep.errors == [], f"errors={rep.errors}")

        # 4. Raw retrieval (no LLM).
        hits = svc.search_vec("configurar el servidor linux")
        passed &= _check(bool(hits), f"search_vec -> {len(hits)} hits")
        passed &= _check(hits[0].path.endswith("setup.md"), f"top hit = {hits[0].path}")

        # 5. Grounded query (FakeLLM).
        result = svc.query_rag("configurar el servidor linux")
        passed &= _check(
            result["answer"].startswith("answer for:"), f"answer={result['answer'][:40]!r}"
        )
        passed &= _check(result["model"] == "fake", f"model={result['model']}")
        passed &= _check(len(result["sources"]) >= 1, "sources no vacias")

    print("SMOKE-ALL-PASS" if passed else "SMOKE-FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
