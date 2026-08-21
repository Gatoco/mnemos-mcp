"""Incremental indexer: walk, mtime skip, batch embed/upsert, stale delete.

Walks a root for `cfg.ext` files, skips files whose mtime is unchanged since
the stored payload, embeds new/changed chunks in batches, upserts them, and
deletes stale points for files that disappeared from disk (IDX-001).
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field

from mcp_rag.errors import RagError

# 16 MiB guard (CHUNK-002) — mirrors chunker.MAX_FILE_BYTES.
MAX_FILE_BYTES = 16 * 1024 * 1024
# progress_cb fires every N files and at the end.
PROGRESS_EVERY = 25


@dataclass
class IndexReport:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_upserted: int = 0
    stale_removed: int = 0
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)


class Indexer:
    def __init__(self, store, embedder, chunker, cfg):
        self.store = store
        self.embedder = embedder
        self.chunker = chunker
        self.cfg = cfg

    def scan(
        self,
        root: str,
        source: str,
        force: bool = False,
        max_files: int | None = None,
        progress_cb=None,
    ) -> IndexReport:
        start = time.monotonic()
        report = IndexReport()
        root = os.fspath(root)

        # 1. Walk root, filter by cfg.ext (case-insensitive), sort for determinism.
        on_disk: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in self.cfg.ext:
                    on_disk.append(os.path.join(dirpath, name))
        on_disk.sort()

        # Already-indexed paths -> stored mtime (for skip) and stale set.
        indexed = {
            d["path"]: d["last_indexed"]
            for d in self.store.list_documents(source=source)
        }

        points: list[dict] = []
        total = len(on_disk)
        processed = 0
        for path in on_disk:
            if max_files is not None and report.files_scanned >= max_files:
                break
            report.files_scanned += 1
            processed += 1
            if progress_cb and (processed % PROGRESS_EVERY == 0 or processed == total):
                progress_cb(processed / total if total else 1.0)

            # 2. Skip >16MB files -> errors[].
            try:
                size = os.path.getsize(path)
            except OSError as exc:
                report.errors.append(f"{path}: {exc}")
                continue
            if size > MAX_FILE_BYTES:
                report.errors.append(f"{path}: file too large (>16MB)")
                continue

            # 3. Skip unchanged (mtime == stored mtime).
            mtime = int(os.path.getmtime(path))
            if not force and path in indexed and indexed[path] == mtime:
                report.files_skipped += 1
                continue

            # 4. Read + chunk.
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError as exc:
                report.errors.append(f"{path}: {exc}")
                continue

            md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
            try:
                chunks = self.chunker.chunk(path, text, mtime, md5)
            except RagError as exc:
                report.errors.append(str(exc))
                continue

            report.files_indexed += 1
            if not chunks:
                continue  # empty/heading-only file -> nothing to embed/upsert

            # 5. Embed (embedder batches internally at 32). Network failure aborts.
            try:
                vectors = self.embedder.embed([c.text for c in chunks])
            except RagError:
                raise  # embedder failure mid-way -> abort, keep state consistent

            # 6. Build points; `index` = chunk index within file (point id md5(path#index)).
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                points.append(
                    {
                        "vector": vec,
                        "source": source,
                        "path": path,
                        "mtime": mtime,
                        "md5": md5,
                        "heading_path": chunk.heading_path,
                        "text": chunk.text,
                        "index": i,
                    }
                )

        # 6. Upsert all points (store batches internally at 128).
        if points:
            self.store.upsert(points)
            report.chunks_upserted = len(points)

        # 7. Stale deletion: paths in store not on disk.
        on_disk_set = set(on_disk)
        for p in set(indexed) - on_disk_set:
            report.stale_removed += self.store.delete_by_path(p)

        report.duration_s = time.monotonic() - start
        return report
