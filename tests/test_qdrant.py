import pytest

from qdrant_client.models import Distance

from mcp_rag.config import AppConfig
from mcp_rag.qdrant_store import QdrantStore, DIMENSIONS


@pytest.fixture
def store():
    cfg = AppConfig()
    s = QdrantStore(cfg, client=__import__("qdrant_client").QdrantClient(":memory:"))
    s.ensure_collection()
    return s


def _point(path, source="vault", mtime=1, md5="m", heading="H", text="hola mundo", vector=None, index=0):
    return {
        "vector": vector or [1.0] + [0.0] * (DIMENSIONS - 1),
        "source": source,
        "path": path,
        "mtime": mtime,
        "md5": md5,
        "heading_path": heading,
        "text": text,
        "index": index,
    }


def test_collection_created_cosine_1024(store):
    info = store.client.get_collection(store.collection)
    assert info.config.params.vectors.size == 1024
    assert info.config.params.vectors.distance == Distance.COSINE


def test_upsert_and_search_returns_dochit(store):
    store.upsert([_point("a.md", text="hola mundo")])
    hits = store.search([1.0] + [0.0] * (DIMENSIONS - 1), limit=5)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.path == "a.md"
    assert hit.heading_path == "H"
    assert hit.score > 0
    assert hit.snippet == "hola mundo"
    assert hit.mtime == 1


def test_filter_by_source(store):
    store.upsert([_point("a.md", source="vault"), _point("b.md", source="other")])
    hits = store.search([1.0] + [0.0] * (DIMENSIONS - 1), source="other", limit=5)
    assert [h.path for h in hits] == ["b.md"]


def test_filter_by_path_prefix(store):
    store.upsert([_point("Proyecto/a.md"), _point("Otro/b.md")])
    hits = store.search(
        [1.0] + [0.0] * (DIMENSIONS - 1), path_prefix="Proyecto", limit=5
    )
    assert [h.path for h in hits] == ["Proyecto/a.md"]


def test_delete_by_path_and_source(store):
    store.upsert([_point("a.md", source="vault"), _point("b.md", source="vault")])
    assert store.delete_by_path("a.md") == 1
    assert store.delete_by_source("vault") == 1


def test_list_documents_groups_by_path(store):
    store.upsert(
        [
            _point("a.md", mtime=1, index=0),
            _point("a.md", mtime=2, index=1),
            _point("b.md", mtime=3, index=0),
        ]
    )
    docs = store.list_documents()
    by_path = {d["path"]: d for d in docs}
    assert by_path["a.md"]["chunks"] == 2
    assert by_path["a.md"]["last_indexed"] == 2
    assert by_path["b.md"]["chunks"] == 1


def test_existing_paths(store):
    store.upsert([_point("a.md", source="vault"), _point("b.md", source="other")])
    assert store.existing_paths("vault") == {"a.md"}


def test_stats(store):
    store.upsert([_point("a.md", source="vault"), _point("b.md", source="other")])
    stats = store.get_collection_stats()
    assert stats["vectors_count"] == 2
    assert stats["by_source"] == {"vault": 1, "other": 1}


def test_health(store):
    ok, msg = store.health()
    assert ok is True
    assert msg == "ok"
