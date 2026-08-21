"""stdlib ThreadingHTTPServer admin page (ADMIN-001..003).

Binds 127.0.0.1:{admin_port}, serves a vanilla-JS page at `/` and JSON
endpoints under `/api/*`. Every endpoint delegates to `core.py` (ADMIN-002) —
zero duplicate logic. All handlers return JSON `{error}` on failure and never
crash the server.
"""
from __future__ import annotations

import json
import logging
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from mcp_rag.config import AppConfig, load_config
from mcp_rag.core import RagService
from mcp_rag.errors import RagError
from mcp_rag.jobs import JobRegistry

log = logging.getLogger("mcp_rag.admin")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

JOB_ROUTE = re.compile(r"^/api/jobs/([0-9a-f]+)$")


def _make_handler(service: RagService, jobs: JobRegistry):
    """Build a handler class bound to the given service + job registry."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet stdlib default
            log.debug("%s - %s", self.address_string(), fmt % args)

        # ------------------------------------------------------------ helpers
        def _json(self, data, status=200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            self._json({"error": message}, status=status)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                raise RagError("invalid JSON body")

        # ---------------------------------------------------------------- GET
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._serve_index()
            if path == "/api/stats":
                return self._json(service.get_stats())
            if path == "/api/documents":
                return self._api_documents()
            if path == "/api/search":
                return self._api_search()
            if path == "/api/health":
                return self._api_health()
            m = JOB_ROUTE.match(path)
            if m:
                return self._api_job(m.group(1))
            self._error(404, f"not found: {path}")

        def _serve_index(self):
            if not os.path.exists(INDEX_FILE):
                return self._error(500, "index.html not found")
            with open(INDEX_FILE, "rb") as fh:
                body = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _query(self):
            return parse_qs(urlparse(self.path).query)

        def _int(self, q, key, default):
            vals = q.get(key, [])
            if not vals:
                return default
            try:
                return int(vals[0])
            except ValueError:
                raise RagError(f"'{key}' must be an int")

        def _api_documents(self):
            q = self._query()
            try:
                docs = service.list_docs(
                    source=q.get("source", [None])[0],
                    path_prefix=q.get("path_prefix", [None])[0],
                    limit=self._int(q, "limit", 100),
                    offset=self._int(q, "offset", 0),
                )
            except RagError as exc:
                return self._error(400, str(exc))
            self._json(docs)

        def _api_search(self):
            q = self._query()
            text = q.get("text", [None])[0]
            if not text:
                return self._error(400, "missing 'text' query param")
            try:
                hits = service.search_vec(
                    text,
                    top_k=self._int(q, "top_k", 5),
                    source=q.get("source", [None])[0],
                    path_prefix=q.get("path_prefix", [None])[0],
                )
            except RagError as exc:
                return self._error(400, str(exc))
            self._json(
                [
                    {"path": h.path, "heading_path": h.heading_path,
                     "score": h.score, "snippet": h.snippet, "mtime": h.mtime}
                    for h in hits
                ]
            )

        def _api_health(self):
            self._json(
                {
                    "qdrant": bool(service.store.health()[0]),
                    "ollama": bool(service.embedder.health()[0]),
                    "llm_configured": bool(service.llm.health()),
                    "admin": "ok",
                }
            )

        def _api_job(self, job_id):
            job = jobs.get(job_id)
            if job is None:
                return self._error(404, f"unknown job: {job_id}")
            self._json(job.as_dict())

        # -------------------------------------------------------------- POST
        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/index":
                return self._api_index()
            self._error(404, f"not found: {path}")

        def _api_index(self):
            try:
                body = self._read_body()
            except RagError as exc:
                return self._error(400, str(exc))

            def run():
                report = service.index_files(
                    source=body.get("source"),
                    path=body.get("path"),
                    force_rescan=bool(body.get("force_rescan")),
                )
                return {
                    "files_scanned": report.files_scanned,
                    "files_indexed": report.files_indexed,
                    "files_skipped": report.files_skipped,
                    "chunks_upserted": report.chunks_upserted,
                    "stale_removed": report.stale_removed,
                    "duration_s": report.duration_s,
                    "errors": report.errors,
                }

            job_id = jobs.submit(run)
            self._json({"job_id": job_id}, status=202)

        # ------------------------------------------------------------ DELETE
        def do_DELETE(self):
            path = urlparse(self.path).path
            if path != "/api/documents":
                return self._error(404, f"not found: {path}")
            try:
                body = self._read_body()
            except RagError as exc:
                return self._error(400, str(exc))
            try:
                deleted = service.delete_docs(path=body.get("path"), source=body.get("source"))
            except RagError as exc:
                return self._error(400, str(exc))
            self._json({"deleted_points": deleted})

    return Handler


def create_server(service: RagService, jobs: JobRegistry | None = None, port: int = 8310) -> ThreadingHTTPServer:
    jobs = jobs or JobRegistry()
    handler = _make_handler(service, jobs)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config()
    service = RagService.from_config(cfg)
    server = create_server(service, port=cfg.admin_port)
    log.info("admin page on http://127.0.0.1:%d", cfg.admin_port)
    server.serve_forever()
