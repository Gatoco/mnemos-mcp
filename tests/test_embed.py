import json

import httpx
import pytest

from mcp_rag.config import AppConfig
from mcp_rag.embed import OllamaEmbedder, MODEL_NOT_FOUND, _l2_normalize
from mcp_rag.errors import RagError


def _cfg():
    return AppConfig(ollama_url="http://ollama:11434", embed_model="bge-m3")


def _embedder(handler, retry_sleep=0.0):
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return OllamaEmbedder(_cfg(), client=client, retry_sleep=retry_sleep)


def test_batch_100_sends_4_requests_of_32():
    calls = []

    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        n = len(body["input"])
        return httpx.Response(200, json={"embeddings": [[0.0] * 1024] * n})

    emb = _embedder(handler)
    texts = [f"t{i}" for i in range(100)]
    vectors = emb.embed(texts)
    assert len(vectors) == 100
    assert len(calls) == 4
    assert [len(json.loads(c.content)["input"]) for c in calls] == [32, 32, 32, 4]


def test_retry_then_fail_raises_ragerror():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(500)

    emb = _embedder(handler, retry_sleep=0.0)
    with pytest.raises(RagError):
        emb.embed(["x"])
    assert attempts["n"] == 3


def test_404_raises_exact_message():
    def handler(request):
        return httpx.Response(404)

    emb = _embedder(handler)
    with pytest.raises(RagError) as exc:
        emb.embed(["x"])
    assert str(exc.value) == MODEL_NOT_FOUND


def test_health_true():
    def handler(request):
        return httpx.Response(200, json={"models": []})

    emb = _embedder(handler)
    assert emb.health() == (True, "ok")


def test_health_404_returns_model_not_found():
    def handler(request):
        return httpx.Response(404)

    emb = _embedder(handler)
    ok, msg = emb.health()
    assert ok is False
    assert msg == MODEL_NOT_FOUND


def test_embed_vectors_are_l2_normalized():
    def handler(request):
        n = len(json.loads(request.content)["input"])
        # Vectors deliberately not unit-length (bge-m3 raw output).
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] + [0.0] * 1022] * n})

    emb = _embedder(handler)
    vectors = emb.embed(["x"])
    v = vectors[0]
    assert v[0] == pytest.approx(0.6) and v[1] == pytest.approx(0.8)
    norm = sum(x * x for x in v) ** 0.5
    assert norm == pytest.approx(1.0)


def test_l2_normalize_zero_vector_unchanged():
    assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]
