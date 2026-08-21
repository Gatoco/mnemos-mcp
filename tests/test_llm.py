import json

import httpx
import pytest

from mcp_rag.config import AppConfig
from mcp_rag.llm import LLMProvider, Answer
from mcp_rag.qdrant_store import DocHit
from mcp_rag.errors import RagError


def _cfg(api_key="secret"):
    return AppConfig(
        api_key=api_key,
        model="deepseek-v4-flash",
        llm_base_url="https://ollama.com/v1",
    )


def _provider(handler, api_key="secret"):
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return LLMProvider(_cfg(api_key), client=client)


def _hit(path="a.md"):
    return DocHit(path=path, heading_path="H", score=0.9, snippet="contenido", mtime=1)


def test_request_shape():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "respuesta"}}]}
        )

    prov = _provider(handler)
    ans = prov.answer("pregunta", [_hit()])
    assert captured["url"] == "https://ollama.com/v1/chat/completions"
    assert captured["body"]["model"] == "deepseek-v4-flash"
    assert captured["body"]["temperature"] == 0.3
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert captured["body"]["messages"][0]["role"] == "system"
    assert isinstance(ans, Answer)
    assert ans.text == "respuesta"
    assert ans.model == "deepseek-v4-flash"


def test_missing_key_raises_ragerror():
    prov = _provider(lambda r: httpx.Response(200, json={}), api_key="")
    with pytest.raises(RagError) as exc:
        prov.answer("q", [_hit()])
    assert "OLLAMA_API_KEY" in str(exc.value)


def test_context_includes_citations():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    prov = _provider(handler)
    prov.answer("q", [_hit("a.md"), _hit("b.md")])
    user_msg = captured["body"]["messages"][1]["content"]
    assert "a.md" in user_msg
    assert "b.md" in user_msg


def test_health_key_presence_only():
    assert _provider(lambda r: None, api_key="k").health() is True
    assert _provider(lambda r: None, api_key="").health() is False
