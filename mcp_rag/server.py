"""FastMCP server exposing RAG tools over stdio (MCP-001..008).

Thin transport layer: each tool validates args (via JSON schema) and
delegates to the shared `core.py` (ADMIN-002). `RagError` is mapped to a
`ToolError` so clients receive a clean message. Progress reporting uses the
mcp 1.12.4 `ctx.report_progress(progress, total)` API; outside a live request
context it is a safe no-op (kept so in-process tests can invoke tools).
"""
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from mcp_rag.config import AppConfig, load_config
from mcp_rag.core import RagService
from mcp_rag.errors import RagError


def _safe_report(ctx: Context | None, progress: float, total: float) -> None:
    if ctx is None:
        return
    try:
        import anyio

        async def _do():
            await ctx.report_progress(progress, total)

        anyio.from_thread.run(_do)
    except Exception:
        pass  # no request context (in-process test) -> no-op


def create_app(service: RagService) -> FastMCP:
    """Build the FastMCP app with all RAG tools wired to `service`."""
    app = FastMCP("mcp-rag-opencode")

    @app.tool()
    async def index(
        ctx: Context,
        source: str | None = None,
        path: str | None = None,
        force_rescan: bool = False,
        max_files: int | None = None,
    ) -> dict:
        """Index the vault (source root or sub-path) into Qdrant."""
        try:

            def cb(fraction: float) -> None:
                _safe_report(ctx, fraction, 1.0)

            report = service.index_files(
                source=source,
                path=path,
                force_rescan=force_rescan,
                max_files=max_files,
                progress_cb=cb,
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
        except RagError as exc:
            raise ToolError(str(exc)) from exc

    @app.tool()
    async def query(
        ctx: Context,
        question: str,
        top_k: int = 5,
        source: str | None = None,
        path_prefix: str | None = None,
        score_threshold: float | None = None,
    ) -> dict:
        """Retrieve context and answer the question with the LLM."""
        try:
            return service.query_rag(
                question=question,
                top_k=top_k,
                source=source,
                path_prefix=path_prefix,
                score_threshold=score_threshold,
            )
        except RagError as exc:
            raise ToolError(str(exc)) from exc

    @app.tool()
    async def search(
        ctx: Context,
        text: str,
        top_k: int = 5,
        source: str | None = None,
        path_prefix: str | None = None,
        score_threshold: float | None = None,
    ) -> list:
        """Raw vector retrieval without the LLM."""
        try:
            hits = service.search_vec(
                text, top_k=top_k, source=source, path_prefix=path_prefix,
                score_threshold=score_threshold,
            )
            return [
                {
                    "path": h.path,
                    "heading_path": h.heading_path,
                    "score": h.score,
                    "snippet": h.snippet,
                    "mtime": h.mtime,
                }
                for h in hits
            ]
        except RagError as exc:
            raise ToolError(str(exc)) from exc

    @app.tool()
    async def delete(
        ctx: Context,
        path: str | None = None,
        source: str | None = None,
    ) -> dict:
        """Delete indexed docs by exact path or by source."""
        try:
            return {"deleted_points": service.delete_docs(path=path, source=source)}
        except RagError as exc:
            raise ToolError(str(exc)) from exc

    @app.tool(name="list")
    async def list_docs_tool(
        ctx: Context,
        source: str | None = None,
        path_prefix: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        """List indexed documents (paginated, filter by source/path prefix)."""
        try:
            return service.list_docs(
                source=source, path_prefix=path_prefix, limit=limit, offset=offset
            )
        except RagError as exc:
            raise ToolError(str(exc)) from exc

    @app.tool()
    async def stats(ctx: Context) -> dict:
        """Return collection stats and provider health."""
        try:
            return service.get_stats()
        except RagError as exc:
            raise ToolError(str(exc)) from exc

    @app.tool()
    async def config(
        ctx: Context,
        action: str = "get",
        key: str | None = None,
        value: float | int | str | None = None,
    ) -> dict:
        """Get the full config, or set one persisted field."""
        try:
            if action == "set":
                if key is None or value is None:
                    raise RagError("config set requires 'key' and 'value'")
                return service.config_set(**{key: value})
            return service.config_get()
        except RagError as exc:
            raise ToolError(str(exc)) from exc

    return app


def main() -> None:
    """Entry point (pyproject script `mcp-rag-opencode`)."""
    cfg = load_config()
    service = RagService.from_config(cfg)
    app = create_app(service)
    app.run()
