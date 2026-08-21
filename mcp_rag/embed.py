"""Ollama embedding provider via `/api/embed` (bge-m3, 1024 dims)."""
from __future__ import annotations

import math
import time

import httpx

from mcp_rag.errors import RagError

DIMENSIONS = 1024
BATCH = 32
TIMEOUT = 120.0
RETRIES = 3
BACKOFF = (1, 2, 4)  # seconds

MODEL_NOT_FOUND = "model not found, run: ollama pull bge-m3"


def _l2_normalize(vector: list[float]) -> list[float]:
    """L2-normalize (bge-m3 returns unnormalized vectors).

    Cosine over unnormalized vectors saturates around 0.45-0.5; with
    normalized vectors cosine == dot product, restoring the ~0.6-0.9 scale
    and making score_threshold usable.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class OllamaEmbedder:
    def __init__(self, cfg, client: httpx.Client | None = None, retry_sleep: float | None = None):
        self.cfg = cfg
        self.client = client or httpx.Client(timeout=TIMEOUT)
        # Injectable sleep for tests (tiny backoff).
        self._sleep = (lambda s: time.sleep(s)) if retry_sleep is None else (lambda _s: retry_sleep)

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        url = f"{self.cfg.ollama_url}/api/embed"
        payload = {"model": self.cfg.embed_model, "input": batch}
        last = None
        for attempt in range(RETRIES):
            try:
                resp = self.client.post(url, json=payload)
                if resp.status_code == 404:
                    raise RagError(MODEL_NOT_FOUND)
                resp.raise_for_status()
                data = resp.json()
                return [_l2_normalize(list(v)) for v in data["embeddings"]]
            except RagError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry transient failures
                last = exc
                if attempt < RETRIES - 1:
                    self._sleep(BACKOFF[attempt])
        raise RagError(f"embedding failed after {RETRIES} retries: {last}")

    def health(self) -> tuple[bool, str]:
        try:
            resp = self.client.get(f"{self.cfg.ollama_url}/api/tags")
            if resp.status_code == 404:
                return False, MODEL_NOT_FOUND
            resp.raise_for_status()
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 - never raises
            return False, str(exc)
