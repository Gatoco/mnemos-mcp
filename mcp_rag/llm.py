"""OpenAI-compatible LLM provider (deepseek-v4-flash via ollama.com/v1)."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from mcp_rag.errors import RagError
from mcp_rag.qdrant_store import DocHit

MAX_CONTEXT_CHUNKS = 8
SNIPPET_CHARS = 400
TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "Answer ONLY from the provided context. Cite the source paths you used. "
    "If the context does not contain the answer, say you don't know."
)


@dataclass
class Answer:
    text: str
    sources: list[DocHit]
    model: str


class LLMProvider:
    def __init__(self, cfg, client: httpx.Client | None = None):
        self.cfg = cfg
        self.client = client or httpx.Client(timeout=120.0)

    def _require_key(self) -> str:
        key = (self.cfg.api_key or "").strip()
        if not key:
            raise RagError("missing OLLAMA_API_KEY: set it in .env or the environment")
        return key

    def answer(self, question: str, context: list[DocHit]) -> Answer:
        key = self._require_key()
        context_block = self._build_context(context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {question}",
            },
        ]
        url = f"{self.cfg.llm_base_url}/chat/completions"
        body = {
            "model": self.cfg.model,
            "temperature": TEMPERATURE,
            "messages": messages,
        }
        resp = self.client.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {key}"},
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return Answer(text=text, sources=context, model=self.cfg.model)

    def _build_context(self, context: list[DocHit]) -> str:
        blocks = []
        for hit in context[:MAX_CONTEXT_CHUNKS]:
            blocks.append(
                f"[{hit.path} | {hit.heading_path}]\n{hit.snippet[:SNIPPET_CHARS]}"
            )
        return "\n\n".join(blocks)

    def health(self) -> bool:
        """Key presence only — no network."""
        return bool((self.cfg.api_key or "").strip())
