"""Admin HTTP tests — in-process ThreadingHTTPServer on an ephemeral port."""
import json
import os
import threading
import time
import urllib.request
import urllib.error

import pytest

from mcp_rag.admin import create_server
from mcp_rag.chunker import Chunker
from mcp_rag.config import AppConfig
from mcp_rag.core import RagService
from mcp_rag.indexer import Indexer
from mcp_rag.jobs import JobRegistry
from mcp_rag.qdrant_store import QdrantStore

from tests.test_core import FakeEmbedder, FakeLLM, FailLLM, _write  # noqa: E402


@pytest.fixture
def server(tmp_path):
    store = QdrantStore(cfg=AppConfig(), client=__import__("qdrant_client").QdrantClient(":memory:"))
    store.ensure_collection()
    cfg = AppConfig(vault_root="", score_threshold=0.0)
    embedder = FakeEmbedder()
    llm = FailLLM()  # llm_configured False
    chunker = Chunker()
    indexer = Indexer(store, embedder, chunker, cfg)
    service = RagService(store, embedder, llm, chunker, indexer, cfg)
    jobs = JobRegistry()
    httpd = create_server(service, jobs=jobs, port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield {"httpd": httpd, "port": port, "service": service, "tmp": tmp_path}
    httpd.shutdown()


def _req(port, method, path, body=None):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def test_get_root_serves_page(server):
    status, _ = _req_raw(server["port"])
    assert status == 200


def _req_raw(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as resp:
        body = resp.read().decode()
        return resp.status, body


def test_stats_endpoint(server):
    status, data = _req(server["port"], "GET", "/api/stats")
    assert status == 200
    assert "vectors_count" in data
    assert "sources" in data
    assert data["llm_configured"] is False


def test_index_job_then_poll_done(server, tmp_path):
    _write(tmp_path, "a.md", "# A\ncontenido del vault\n", mtime=1)
    status, data = _req(
        server["port"], "POST", "/api/index", {"path": str(tmp_path)}
    )
    assert status == 202
    job_id = data["job_id"]
    result = None
    for _ in range(50):
        s, jd = _req(server["port"], "GET", f"/api/jobs/{job_id}")
        if jd["status"] in ("done", "error"):
            result = jd
            break
        time.sleep(0.05)
    assert result is not None
    assert result["status"] == "done"
    assert result["result"]["files_indexed"] >= 1


def test_search_endpoint(server, tmp_path):
    _write(tmp_path, "a.md", "# A\ntermino unico admin\n", mtime=1)
    _req(server["port"], "POST", "/api/index", {"path": str(tmp_path)})
    time.sleep(0.1)
    status, hits = _req(server["port"], "GET", "/api/search?text=termino%20unico%20admin")
    assert status == 200
    assert hits and hits[0]["path"].endswith("a.md")


def test_delete_documents(server, tmp_path):
    p = _write(tmp_path, "a.md", "# A\nhola\n", mtime=1)
    _req(server["port"], "POST", "/api/index", {"path": str(tmp_path)})
    time.sleep(0.1)
    status, data = _req(server["port"], "DELETE", "/api/documents", {"path": p})
    assert status == 200
    assert data["deleted_points"] >= 1


def test_health_endpoint(server):
    status, data = _req(server["port"], "GET", "/api/health")
    assert status == 200
    assert data["qdrant"] is True
    assert data["ollama"] is True
    assert data["llm_configured"] is False
    assert data["admin"] == "ok"


def test_bad_route_404_json(server):
    status, data = _req(server["port"], "GET", "/api/nonexistent")
    assert status == 404
    assert "error" in data


def test_search_missing_text_400(server):
    status, data = _req(server["port"], "GET", "/api/search")
    assert status == 400
    assert "error" in data
