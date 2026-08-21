"""Qdrant collection management, CRUD, search, stats and health.

Supports an injected `QdrantClient` so tests can pass `QdrantClient(":memory:")`.
Point id is a deterministic md5 of `path` so upserts are idempotent per file.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from mcp_rag.errors import RagError

DIMENSIONS = 1024
UPSERT_BATCH = 128
RETRIES = 3
BACKOFF = (1, 2, 4)  # seconds


@dataclass
class DocHit:
    path: str
    heading_path: str
    score: float
    snippet: str
    mtime: int


def _point_id(path: str, index: int) -> str:
    """Deterministic point id from path + chunk index (idempotent upsert).

    `index` disambiguates multiple chunks of the same file; re-indexing an
    unchanged file yields the same ids, so upserts overwrite in place.
    """
    return hashlib.md5(f"{path}#{index}".encode("utf-8")).hexdigest()


def _match_filter(source: str | None, path_prefix: str | None) -> Filter | None:
    must = []
    if source:
        must.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if path_prefix:
        # MatchText on the keyword `path` index supports prefix/substring.
        must.append(FieldCondition(key="path", match=MatchText(text=path_prefix)))
    return Filter(must=must) if must else None


class QdrantStore:
    def __init__(self, cfg, client: QdrantClient | None = None):
        self.cfg = cfg
        self.collection = cfg.collection
        self.client = client or QdrantClient(url=cfg.qdrant_url)

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE),
            )
        # Payload indexes (no-op in :memory:, required on server).
        for field, schema in (
            ("path", PayloadSchemaType.KEYWORD),
            ("source", PayloadSchemaType.KEYWORD),
            ("mtime", PayloadSchemaType.INTEGER),
        ):
            try:
                self.client.create_payload_index(
                    self.collection, field_name=field, field_schema=schema
                )
            except Exception:
                pass  # index already exists or unsupported locally

    def upsert(self, points: list[dict]) -> None:
        """Upsert points in batches of 128 with retry/backoff.

        Each point dict: {vector, source, path, mtime, md5, heading_path, text}.
        """
        for i in range(0, len(points), UPSERT_BATCH):
            batch = points[i : i + UPSERT_BATCH]
            structs = [
                PointStruct(
                    id=_point_id(p["path"], p.get("index", 0)),
                    vector=p["vector"],
                    payload={
                        "source": p["source"],
                        "path": p["path"],
                        "mtime": int(p["mtime"]),
                        "md5": p["md5"],
                        "heading_path": p["heading_path"],
                        "text": p["text"],
                    },
                )
                for p in batch
            ]
            self._retry(lambda: self.client.upsert(self.collection, points=structs))

    def _retry(self, fn):
        last = None
        for attempt in range(RETRIES):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last = exc
                if attempt < RETRIES - 1:
                    time.sleep(BACKOFF[attempt])
        raise RagError(f"qdrant operation failed after {RETRIES} retries: {last}")

    def search(
        self,
        vector: list[float],
        source: str | None = None,
        path_prefix: str | None = None,
        limit: int = 5,
        threshold: float = 0.5,
    ) -> list[DocHit]:
        qfilter = _match_filter(source, path_prefix)
        resp = self.client.query_points(
            self.collection,
            query=vector,
            query_filter=qfilter,
            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        )
        hits = []
        for p in resp.points:
            payload = p.payload or {}
            hits.append(
                DocHit(
                    path=payload.get("path", ""),
                    heading_path=payload.get("heading_path", ""),
                    score=float(p.score),
                    snippet=(payload.get("text") or "")[:400],
                    mtime=int(payload.get("mtime", 0)),
                )
            )
        return hits

    def delete_by_path(self, path: str) -> int:
        qfilter = Filter(
            must=[FieldCondition(key="path", match=MatchValue(value=path))]
        )
        return self._delete(qfilter)

    def delete_by_source(self, source: str) -> int:
        qfilter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        )
        return self._delete(qfilter)

    def _delete(self, qfilter: Filter) -> int:
        before = self.client.count(self.collection, count_filter=qfilter, exact=True).count
        self.client.delete(self.collection, points_selector=qfilter)
        return before

    def list_documents(
        self,
        source: str | None = None,
        path_prefix: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        qfilter = _match_filter(source, path_prefix)
        records, _ = self.client.scroll(
            self.collection,
            scroll_filter=qfilter,
            limit=limit,
            offset=offset,
            with_payload=True,
        )
        grouped: dict[str, dict] = {}
        for rec in records:
            payload = rec.payload or {}
            path = payload.get("path", "")
            entry = grouped.setdefault(path, {"chunks": 0, "last_indexed": 0})
            entry["chunks"] += 1
            entry["last_indexed"] = max(entry["last_indexed"], int(payload.get("mtime", 0)))
        return [
            {"path": path, "chunks": data["chunks"], "last_indexed": data["last_indexed"]}
            for path, data in grouped.items()
        ]

    def existing_paths(self, source: str) -> set[str]:
        qfilter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        )
        paths: set[str] = set()
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                self.collection,
                scroll_filter=qfilter,
                limit=1000,
                offset=offset,
                with_payload=["path"],
            )
            for rec in records:
                if rec.payload and rec.payload.get("path"):
                    paths.add(rec.payload["path"])
            if next_offset is None:
                break
            offset = next_offset
        return paths

    def get_collection_stats(self) -> dict:
        total = self.client.count(self.collection, exact=True).count
        by_source: dict[str, int] = {}
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                self.collection,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=["source"],
            )
            for rec in records:
                src = (rec.payload or {}).get("source", "unknown")
                by_source[src] = by_source.get(src, 0) + 1
            if next_offset is None:
                break
            offset = next_offset
        return {"vectors_count": total, "by_source": by_source}

    def health(self) -> tuple[bool, str]:
        try:
            self.client.get_collections()
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 - never raises
            return False, str(exc)
