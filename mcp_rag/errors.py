"""Shared exception hierarchy for mcp-rag-opencode.

Core exceptions subclass `RagError`; the MCP layer maps them to
`ClearError` and the admin HTTP layer to JSON `{error}` responses.
"""


class RagError(Exception):
    """Base error for the RAG pipeline. Message is user-facing."""
