import math
import os

import pytest
from qdrant_client import QdrantClient

from mcp_rag.chunker import Chunker
from mcp_rag.config import AppConfig
from mcp_rag.indexer import Indexer, MAX_FILE_BYTES
from mcp_rag.qdrant_store import QdrantStore, DIMENSIONS


class FakeEmbedder:
    """Deterministic 1024-dim vectors derived from text (offline)."""

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    def embed(self, texts):
        return [
            [math.sin(i + hash(t) % 1000) for i in range(DIMENSIONS)] for t in texts
        ]


@pytest.fixture
def store():
    cfg = AppConfig()
    s = QdrantStore(cfg, client=QdrantClient(":memory:"))
    s.ensure_collection()
    return s


@pytest.fixture
def indexer(store):
    return Indexer(store, FakeEmbedder(), Chunker(), AppConfig())


def _write(root, name, content, mtime=1):
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.utime(path, (mtime, mtime))
    return path


def test_fresh_index_counts(tmp_path, indexer, store):
    _write(tmp_path, "a.md", "# A\nhola mundo\n", mtime=1)
    _write(tmp_path, "b.md", "# B\nsegundo archivo\n", mtime=2)
    rep = indexer.scan(str(tmp_path), "vault")
    assert rep.files_scanned == 2
    assert rep.files_indexed == 2
    assert rep.files_skipped == 0
    assert rep.chunks_upserted > 0
    assert rep.errors == []
    assert store.existing_paths("vault") == {
        os.path.join(str(tmp_path), "a.md"),
        os.path.join(str(tmp_path), "b.md"),
    }


def test_unchanged_skipped_on_second_run(tmp_path, indexer, store):
    _write(tmp_path, "a.md", "# A\nhola mundo\n", mtime=1)
    indexer.scan(str(tmp_path), "vault")
    rep = indexer.scan(str(tmp_path), "vault")
    assert rep.files_scanned == 1
    assert rep.files_indexed == 0
    assert rep.files_skipped == 1
    assert rep.chunks_upserted == 0


def test_changed_file_reindexed(tmp_path, indexer, store):
    path = _write(tmp_path, "a.md", "# A\ncontenido viejo\n", mtime=1)
    indexer.scan(str(tmp_path), "vault")
    # Change content + bump mtime.
    _write(tmp_path, "a.md", "# A\ncontenido NUEVO y distinto\n", mtime=2)
    rep = indexer.scan(str(tmp_path), "vault")
    assert rep.files_indexed == 1
    assert rep.files_skipped == 0
    # Old chunk gone, new chunk present (search with the new text's vector).
    vec = FakeEmbedder().embed(["contenido NUEVO y distinto"])[0]
    hits = store.search(vec, limit=10)
    assert all("viejo" not in h.snippet for h in hits)
    assert any("NUEVO" in h.snippet for h in hits)


def test_deleted_file_stale_cleanup(tmp_path, indexer, store):
    _write(tmp_path, "a.md", "# A\nhola\n", mtime=1)
    _write(tmp_path, "b.md", "# B\nmundo\n", mtime=1)
    indexer.scan(str(tmp_path), "vault")
    os.remove(os.path.join(str(tmp_path), "a.md"))
    rep = indexer.scan(str(tmp_path), "vault")
    assert rep.stale_removed > 0
    assert os.path.join(str(tmp_path), "a.md") not in store.existing_paths("vault")
    assert os.path.join(str(tmp_path), "b.md") in store.existing_paths("vault")


def test_oversized_file_recorded_in_errors(tmp_path, indexer):
    path = os.path.join(str(tmp_path), "big.md")
    with open(path, "wb") as fh:
        fh.write(b"x" * (MAX_FILE_BYTES + 1))
    rep = indexer.scan(str(tmp_path), "vault")
    assert rep.files_scanned == 1
    assert rep.files_indexed == 0
    assert any("file too large" in e for e in rep.errors)


def test_max_files_respected(tmp_path, indexer):
    for i in range(5):
        _write(tmp_path, f"f{i}.md", f"# F{i}\ncontenido {i}\n", mtime=i)
    rep = indexer.scan(str(tmp_path), "vault", max_files=2)
    assert rep.files_scanned == 2
    assert rep.files_indexed == 2


def test_progress_cb_increasing(tmp_path, indexer):
    for i in range(30):
        _write(tmp_path, f"f{i}.md", f"# F{i}\ncontenido {i}\n", mtime=i)
    seen = []
    indexer.scan(str(tmp_path), "vault", progress_cb=seen.append)
    assert seen
    assert seen == sorted(seen)
    assert seen[-1] == 1.0
